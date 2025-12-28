from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import HKBConfig, resolve_run_dir
from .download import download_file
from .embed import Embedder
from .storage import LocalBackend, StorageBackend
from . import schemas
from .utils import now_utc_iso, sha256_file, set_seeds, slug, write_json, write_jsonl


# -------------------------
# Core pipeline
# -------------------------


def _infer_csv_inside_zip(zip_path: Path) -> str:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV found inside zip: {zip_path}")
        # Prefer the largest CSV (often the main file)
        names.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        return names[0]


def _read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    import zipfile

    csv_name = _infer_csv_inside_zip(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace in column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, desired: str) -> str:
    # case-insensitive match
    desired_l = desired.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == desired_l:
            return c
    raise KeyError(f"Column not found: {desired} (available: {list(df.columns)[:20]}...)")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return None


def _series_stats(values: List[Optional[float]]) -> Dict[str, Any]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return dict(min=None, max=None, mean=None, median=None, variance=None, entropy=None)
    # entropy: histogram-based (simple, robust)
    try:
        hist, _ = np.histogram(arr, bins=min(20, max(2, int(np.sqrt(arr.size)))))
        p = hist / (hist.sum() + 1e-12)
        ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    except Exception:
        ent = None
    return dict(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        variance=float(np.var(arr)),
        entropy=ent,
    )


@dataclass
class BuildOutput:
    run_dir: str
    data_dir: str
    domain_jsonl: str
    series_jsonl: str
    records_jsonl: str
    tar_gz: Optional[str] = None
    counts: Dict[str, int] = None  # type: ignore


class FAO2HKBPipeline:
    """Drive-agnostic FAOSTAT Bulk Download → HKB generator."""

    def __init__(self, cfg: HKBConfig, storage: Optional[StorageBackend] = None):
        self.cfg = cfg
        self.storage: StorageBackend = storage or LocalBackend()

        set_seeds(cfg.run.seed)

        self.run_dir = resolve_run_dir(cfg)
        self.raw_dir = self.run_dir / cfg.io.raw_dir
        self.work_dir = self.run_dir / cfg.io.work_dir
        self.artifacts_dir = self.run_dir / cfg.io.artifacts_dir
        self.data_dir = self.artifacts_dir / "data"
        self.repro_dir = self.artifacts_dir / "_repro"

        for p in [self.raw_dir, self.work_dir, self.data_dir, self.repro_dir]:
            self.storage.mkdir(p)

        # Persist resolved config for auditability
        self.storage.write_bytes(
            self.run_dir / "config.resolved.yaml",
            _yaml_dump(cfg).encode("utf-8"),
        )

    # -------------------------
    # Download
    # -------------------------

    def download_data(self, overwrite: bool = False) -> List[str]:
        out_paths = []
        for s in self.cfg.sources:
            name = f"table_{s.table:02d}__{slug(str(s.meta.get('Domain_table', 'faostat')))}.zip"
            dst = self.raw_dir / name
            download_file(s.url, dst, overwrite=overwrite)
            out_paths.append(str(dst))
        return out_paths

    # -------------------------
    # Load
    # -------------------------

    def load_normalized_faostat(self) -> pd.DataFrame:
        """Load all selected normalized FAOSTAT tables into one DataFrame."""
        frames = []
        for s in self.cfg.sources:
            name = f"table_{s.table:02d}__{slug(str(s.meta.get('Domain_table', 'faostat')))}.zip"
            zip_path = self.raw_dir / name
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing source zip (run download_data first): {zip_path}")
            df = _read_csv_from_zip(zip_path)
            df = _normalize_columns(df)
            # attach L3 meta (as columns)
            for k, v in (s.meta or {}).items():
                df[k] = v
            df["__source_table__"] = s.table
            df["__source_url__"] = s.url
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        df_all = pd.concat(frames, ignore_index=True)
        return df_all

    # -------------------------
    # Build HKB
    # -------------------------

    def build_hkb_jsonl(self, df_all: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.cfg
        b = cfg.build

        # Column resolution (case-insensitive)
        area_col = _find_col(df_all, b.area_col)
        item_col = _find_col(df_all, b.item_col)
        element_col = _find_col(df_all, b.element_col)
        unit_col = _find_col(df_all, b.unit_col)
        year_col = _find_col(df_all, b.year_col)
        value_col = _find_col(df_all, b.value_col)

        # Drop noisy columns
        if b.drop_columns_regex:
            import re

            pat = re.compile(b.drop_columns_regex)
            keep = [c for c in df_all.columns if not pat.match(str(c))]
            df_all = df_all.loc[:, keep].copy()

        # Optional: detect *Code columns (if user didn't drop them)
        def _try_find(df: pd.DataFrame, name: str) -> Optional[str]:
            try:
                return _find_col(df, name)
            except KeyError:
                return None

        area_code_col = _try_find(df_all, "Area Code")
        item_code_col = _try_find(df_all, "Item Code")
        element_code_col = _try_find(df_all, "Element Code")
        unit_code_col = _try_find(df_all, "Unit Code")

        # Determine L3 columns (meta fields)
        # Default to the user's meta keys if present, else fall back to Domain/Sub_domain/Domain_table.
        l3_keys = ["Domain", "Sub_domain", "Domain_table"]
        for k in l3_keys:
            if k not in df_all.columns:
                # If user used lowercase keys, try best-effort match
                try:
                    df_all[k] = df_all[_find_col(df_all, k)]
                except Exception:
                    raise KeyError(
                        f"Missing required L3 column '{k}' in dataframe. Ensure source meta includes it."
                    )

        # Build L3 entries
        l3_rows: List[Dict[str, Any]] = []
        domain_cards = df_all[l3_keys].drop_duplicates()
        for _, row in domain_cards.iterrows():
            dims_l3 = {k: row[k] for k in l3_keys}
            domain_key = schemas.make_domain_key_sha256(dims_l3)
            desc = schemas.build_domain_description_string(dims_l3)

            entry = schemas.DomainL3Entry(
                identity=schemas.DomainIdentity(domain_key=domain_key),
                dims=dims_l3,
                description=schemas.DomainDescription(string=desc, embedding=None),
                embedding_meta=schemas.EmbeddingMeta(
                    embedding_model=cfg.embeddings.model_name if cfg.embeddings.enabled else None,
                    embedding_dim=0,
                ),
            )
            l3_rows.append(entry.model_dump())

        # Group to series (L2)
        series_dims = [
            "Domain",
            "Sub_domain",
            "Domain_table",
            area_col,
            item_col,
            element_col,
            unit_col,
        ]
        df_all = df_all.copy()
        df_all.loc[:, "_year_"] = df_all[year_col].apply(_safe_int)
        df_all.loc[:, "_value_"] = df_all[value_col].apply(_safe_float)

        g = df_all.groupby(series_dims, dropna=False, sort=False)

        l2_rows: List[Dict[str, Any]] = []
        l1_rows: List[Dict[str, Any]] = []

        for keys, df_g in g:
            kmap = dict(zip(series_dims, keys))

            dims_l3 = {k: kmap[k] for k in ["Domain", "Sub_domain", "Domain_table"]}
            dims_l2: Dict[str, Any] = {
                "Area": kmap[area_col],
                "Item": kmap[item_col],
                "Element": kmap[element_col],
                "Unit": kmap[unit_col],
            }

            # Optionally attach codes to dims_l2 (if present and requested)
            if b.include_codes:
                if area_code_col and area_code_col in df_g.columns:
                    dims_l2["Area_code"] = df_g[area_code_col].iloc[0]
                if item_code_col and item_code_col in df_g.columns:
                    dims_l2["Item_code"] = df_g[item_code_col].iloc[0]
                if element_code_col and element_code_col in df_g.columns:
                    dims_l2["Element_code"] = df_g[element_code_col].iloc[0]
                if unit_code_col and unit_code_col in df_g.columns:
                    dims_l2["Unit_code"] = df_g[unit_code_col].iloc[0]

            domain_key = schemas.make_domain_key_sha256(dims_l3)
            series_key = schemas.make_series_key_sha256(dims_l3, dims_l2)

            # -------------------------
            # Build raw (year,value) pairs
            # -------------------------
            pairs: List[Tuple[int, Optional[float]]] = []
            for _, r in df_g.iterrows():
                y = r.get("_year_", None)
                v = r.get("_value_", None)
                if y is None:
                    continue
                # y ya viene como int/None; forzamos int por seguridad
                try:
                    yi = int(y)
                except Exception:
                    continue
                pairs.append((yi, v))

            if not pairs:
                continue

            # -------------------------
            # Deduplicate by year (keep last)
            # -------------------------
            dedup: Dict[int, Optional[float]] = {}
            for yi, v in pairs:
                dedup[yi] = v  # keep last

            recs2 = [{"Year": yi, "Value": dedup[yi]} for yi in sorted(dedup)]
            n_members = len(recs2)

            if n_members < cfg.build.filters.min_points:
                continue

            # -------------------------
            # Compute robust properties from FINAL stream
            # -------------------------
            years = [d["Year"] for d in recs2]
            values = [d["Value"] for d in recs2]

            year_min = min(years) if years else None
            year_max = max(years) if years else None

            zeros = sum(1 for v in values if v == 0)
            nulls = sum(1 for v in values if v is None)

            # zeros percentage (sobre puntos válidos no-nulos)
            non_null = max(1, n_members - nulls)
            z_pct = zeros / non_null

            if cfg.build.filters.drop_if_zeros_pct_gte is not None:
                if z_pct >= float(cfg.build.filters.drop_if_zeros_pct_gte):
                    continue

            stats = _series_stats(values)

            if cfg.build.filters.drop_constant and stats.get("variance") is not None:
                # tolerancia numérica (más robusto que == 0.0)
                if float(stats["variance"]) <= 1e-12:
                    continue

            # -------------------------
            # L2 series card
            # -------------------------
            s_desc = schemas.build_series_description_string(
                dims_l3,
                dims_l2,
                include_codes=cfg.build.include_codes,
            )

            series_entry = schemas.SeriesL2Entry(
                identity=schemas.SeriesIdentity(domain_key=domain_key, series_key=series_key),
                dims=dims_l2,
                description=schemas.SeriesDescription(string=s_desc, embedding=None),
                properties=schemas.SeriesProperties(
                    year_min=year_min,
                    year_max=year_max,
                    n_members=n_members,
                    zeros=zeros,
                    nulls=nulls,
                    min=stats["min"],
                    max=stats["max"],
                    mean=stats["mean"],
                    median=stats["median"],
                    variance=stats["variance"],
                    entropy=stats["entropy"],
                ),
                embedding_meta=schemas.EmbeddingMeta(
                    embedding_model=cfg.embeddings.model_name if cfg.embeddings.enabled else None,
                    embedding_dim=0,
                ),
            )
            l2_rows.append(series_entry.model_dump())

            # -------------------------
            # L1 records entry
            # -------------------------
            records_entry = schemas.RecordsL1Entry(
                identity=schemas.RecordsIdentity(domain_key=domain_key, series_key=series_key),
                data=[schemas.RecordItem(Year=d["Year"], Value=d["Value"]) for d in recs2],
            )
            l1_rows.append(records_entry.model_dump())

        # Optional embeddings (L3/L2)
        if cfg.embeddings.enabled:
            emb = Embedder(
                cfg.embeddings.model_name,
                device=cfg.embeddings.device,
                batch_size=cfg.embeddings.batch_size,
            )
            norm = bool(getattr(cfg.embeddings, "l2_normalize", False))

            if "l3" in cfg.embeddings.levels:
                texts = [r["description"]["string"] for r in l3_rows]
                vecs = emb.encode(texts, normalize=norm) if texts else np.zeros((0, 0), dtype=np.float32)
                for r, v in zip(l3_rows, vecs):
                    r["description"]["embedding"] = v.tolist()
                    r["embedding_meta"]["embedding_dim"] = int(vecs.shape[1]) if vecs.size else 0

            if "l2" in cfg.embeddings.levels:
                texts = [r["description"]["string"] for r in l2_rows]
                vecs = emb.encode(texts, normalize=norm) if texts else np.zeros((0, 0), dtype=np.float32)
                for r, v in zip(l2_rows, vecs):
                    r["description"]["embedding"] = v.tolist()
                    r["embedding_meta"]["embedding_dim"] = int(vecs.shape[1]) if vecs.size else 0


        # Write JSONL artifacts
        domain_path = self.data_dir / "DOMAIN.jsonl"
        series_path = self.data_dir / "SERIES.jsonl"
        records_path = self.data_dir / "RECORDS.jsonl"

        write_jsonl(domain_path, l3_rows)
        write_jsonl(series_path, l2_rows)
        write_jsonl(records_path, l1_rows)

        # counts: domains, series, total points (Year-Value)
        points_total = sum(int(s["properties"].get("n_members", 0)) for s in l2_rows)

        # Manifest (integrity-first)
        manifest = {
            "run_id": cfg.run.run_id,
            "created_utc": now_utc_iso(),
            "counts": {
                "domains": len(l3_rows),
                "series": len(l2_rows),
                "records": int(points_total),  # total Year-Value points
            },
            "files": {
                "DOMAIN.jsonl": {
                    "path": str(domain_path),
                    "sha256": sha256_file(domain_path),
                    "bytes": domain_path.stat().st_size,
                },
                "SERIES.jsonl": {
                    "path": str(series_path),
                    "sha256": sha256_file(series_path),
                    "bytes": series_path.stat().st_size,
                },
                "RECORDS.jsonl": {
                    "path": str(records_path),
                    "sha256": sha256_file(records_path),
                    "bytes": records_path.stat().st_size,
                },
            },
        }
        write_json(self.run_dir / "manifest.json", manifest)

        tar_path = None
        if cfg.packaging.tar_gz:
            tar_path = str(self._tar_gz_one())

        out = BuildOutput(
            run_dir=str(self.run_dir),
            data_dir=str(self.data_dir),
            domain_jsonl=str(domain_path),
            series_jsonl=str(series_path),
            records_jsonl=str(records_path),
            tar_gz=tar_path,
            counts=manifest["counts"],
        )
        return out.__dict__

    def _tar_gz_one(self) -> Path:
        cfg = self.cfg
        tar_path = self.run_dir / f"{cfg.run.run_id}.tar.gz"
        # Always package artifacts + config + manifest
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(self.run_dir / "config.resolved.yaml", arcname="config.resolved.yaml")
            tar.add(self.run_dir / "manifest.json", arcname="manifest.json")
            tar.add(self.artifacts_dir, arcname="artifacts")
            if cfg.packaging.include_raw:
                tar.add(self.raw_dir, arcname="raw")
        return tar_path


def _yaml_dump(cfg: HKBConfig) -> str:
    # avoid importing yaml at runtime in hot loop
    try:
        import yaml

        return yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(cfg.model_dump(), ensure_ascii=False, indent=2)
