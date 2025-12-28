from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .schemas import (
    CANON_DIMS_L3_ORDER,
    CANON_DIMS_L2_ORDER,
    DomainL3Entry,
    SeriesL2Entry,
    RecordsL1Entry,
)

PLOT_BLUE = "#1f77b4"

# -----------------------------------------------------------------------------
# JSONL utils
# -----------------------------------------------------------------------------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _try_upload(pipeline: Any, local_path: str) -> None:
    fn = getattr(pipeline, "_upload_file", None)
    if callable(fn):
        try:
            fn(local_path)
        except Exception:
            pass

def _default_domain_jsonl(pipeline: Any) -> str:
    return os.path.join(str(pipeline.data_folder), "DOMAIN.jsonl")

def _default_series_jsonl(pipeline: Any) -> str:
    return os.path.join(str(pipeline.data_folder), "SERIES.jsonl")

def _default_records_jsonl(pipeline: Any) -> str:
    return os.path.join(str(pipeline.data_folder), "RECORDS.jsonl")

def _collapse_topk(values: pd.Series, topk: int = 12) -> pd.Series:
    s = values.fillna("").astype(str)
    s = s.where(s != "", other="(empty)")
    vc = s.value_counts()
    if vc.shape[0] <= topk:
        return s
    keep = set(vc.index[:topk].tolist())
    return s.where(s.isin(keep), other="Other")

def _coerce_year_value_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "Year" not in df.columns:
        raise KeyError("Records.data debe contener 'Year'")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values("Year").reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# Projection (PCA/TSNE) + robustness for degenerate projections
# -----------------------------------------------------------------------------
def compute_2d_projection(
    X: np.ndarray,
    *,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """
    X: (n, d) -> Z: (n, 2)
    """
    if X.ndim != 2:
        raise ValueError(f"X debe ser 2D. shape={X.shape}")
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    method = (method or "pca").lower()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == "tsne":
        perp = max(2, min(30, (n - 1) // 3))
        perp = min(perp, n - 1)
        try:
            Z = TSNE(
                n_components=2,
                random_state=int(random_state),
                perplexity=float(perp),
                init="pca",
                learning_rate="auto",
            ).fit_transform(X)
            return Z.astype(np.float32)
        except Exception:
            Z = PCA(n_components=2).fit_transform(X)
            return Z.astype(np.float32)

    Z = PCA(n_components=2).fit_transform(X)
    return Z.astype(np.float32)

def _apply_visual_jitter(Z: np.ndarray, *, seed: int = 42, eps: float = 1e-4) -> np.ndarray:
    """
    Si PCA colapsa (var ~0 o todos puntos iguales), mete jitter pequeñito solo para visualización.
    """
    if Z.size == 0:
        return Z
    Z = Z.astype(np.float32, copy=True)

    # colapso: var muy pequeña o casi todos iguales
    v = np.nanvar(Z, axis=0)
    collapsed = (np.all(v < 1e-18)) or (np.unique(Z.round(12), axis=0).shape[0] <= 2)

    if not collapsed:
        return Z

    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=float(eps), size=Z.shape).astype(np.float32)
    return Z + noise

def compute_2d_projection_joint(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    method: str = "pca",
    random_state: int = 42,
    embedding_col: str = "embedding",
    jitter_if_degenerate: bool = True,
    jitter_eps: float = 1e-4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proyecta A y B en el MISMO espacio 2D (fit sobre A∪B).
    """
    if df_a is None or df_b is None:
        raise ValueError("df_a/df_b no pueden ser None.")
    if embedding_col not in df_a.columns or embedding_col not in df_b.columns:
        raise KeyError(f"Ambos df deben contener columna '{embedding_col}'.")

    def _to_mat(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        arrs = []
        keep = []
        for i, v in enumerate(df[embedding_col].values):
            if v is None:
                continue
            try:
                a = np.asarray(v, dtype=np.float32).reshape(-1)
                if a.size == 0:
                    continue
                arrs.append(a)
                keep.append(i)
            except Exception:
                continue
        if not arrs:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.vstack(arrs), keep

    Xa, ka = _to_mat(df_a)
    Xb, kb = _to_mat(df_b)

    if Xa.size == 0 and Xb.size == 0:
        out_a = df_a.iloc[0:0].copy()
        out_b = df_b.iloc[0:0].copy()
        out_a["Dim 1"] = np.nan; out_a["Dim 2"] = np.nan
        out_b["Dim 1"] = np.nan; out_b["Dim 2"] = np.nan
        return out_a, out_b

    if Xa.size == 0:
        Zb = compute_2d_projection(Xb, method=method, random_state=int(random_state))
        if jitter_if_degenerate:
            Zb = _apply_visual_jitter(Zb, seed=int(random_state), eps=float(jitter_eps))
        out_b = df_b.iloc[kb].copy()
        out_b["Dim 1"] = Zb[:, 0]; out_b["Dim 2"] = Zb[:, 1]
        out_a = df_a.iloc[0:0].copy()
        out_a["Dim 1"] = np.nan; out_a["Dim 2"] = np.nan
        return out_a, out_b

    if Xb.size == 0:
        Za = compute_2d_projection(Xa, method=method, random_state=int(random_state))
        if jitter_if_degenerate:
            Za = _apply_visual_jitter(Za, seed=int(random_state), eps=float(jitter_eps))
        out_a = df_a.iloc[ka].copy()
        out_a["Dim 1"] = Za[:, 0]; out_a["Dim 2"] = Za[:, 1]
        out_b = df_b.iloc[0:0].copy()
        out_b["Dim 1"] = np.nan; out_b["Dim 2"] = np.nan
        return out_a, out_b

    X = np.vstack([Xa, Xb])
    Z = compute_2d_projection(X, method=method, random_state=int(random_state))
    if jitter_if_degenerate:
        Z = _apply_visual_jitter(Z, seed=int(random_state), eps=float(jitter_eps))

    Za = Z[: Xa.shape[0]]
    Zb = Z[Xa.shape[0] :]

    out_a = df_a.iloc[ka].copy()
    out_a["Dim 1"] = Za[:, 0]; out_a["Dim 2"] = Za[:, 1]

    out_b = df_b.iloc[kb].copy()
    out_b["Dim 1"] = Zb[:, 0]; out_b["Dim 2"] = Zb[:, 1]

    return out_a, out_b

# -----------------------------------------------------------------------------
# HKB Dataviz
# -----------------------------------------------------------------------------
@dataclass
class HKBDataViz:
    pipeline: Any
    domain_jsonl_path: Optional[str] = None
    series_jsonl_path: Optional[str] = None
    records_jsonl_path: Optional[str] = None

    _domain_dims_by_key: Optional[Dict[str, Dict[str, Any]]] = None

    def _domain_path(self) -> str:
        return self.domain_jsonl_path or _default_domain_jsonl(self.pipeline)

    def _series_path(self) -> str:
        return self.series_jsonl_path or _default_series_jsonl(self.pipeline)

    def _records_path(self) -> str:
        return self.records_jsonl_path or _default_records_jsonl(self.pipeline)

    def _plots_dir(self) -> str:
        return str(getattr(self.pipeline, "plots_folder", "plots"))

    # -------------------------------------------------------------------------
    # Domain dims (for joining L3 labels onto L2 points)
    # -------------------------------------------------------------------------
    def _load_domain_dims(self) -> Dict[str, Dict[str, Any]]:
        if self._domain_dims_by_key is not None:
            return self._domain_dims_by_key
        dims_by_key: Dict[str, Dict[str, Any]] = {}
        for obj in iter_jsonl(self._domain_path()):
            e = DomainL3Entry.model_validate(obj)
            dims_by_key[e.identity.domain_key] = dict(e.dims)
        self._domain_dims_by_key = dims_by_key
        return dims_by_key

    # -------------------------------------------------------------------------
    # Records (single series stream scan)
    # -------------------------------------------------------------------------
    def get_records_by_series_key(self, series_key: str) -> RecordsL1Entry:
        path = self._records_path()
        for obj in iter_jsonl(path):
            e = RecordsL1Entry.model_validate(obj)
            if e.identity.series_key == series_key:
                return e
        raise KeyError(f"series_key no encontrado en RECORDS.jsonl: {series_key}")

    # -------------------------------------------------------------------------
    # Series preview plotting (unchanged, useful for Step 9)
    # -------------------------------------------------------------------------
    def _pretty_title_series(self, dims_l3: Dict[str, Any], dims_l2: Dict[str, Any]) -> str:
        parts = ["FAOSTAT series"]
        for k in CANON_DIMS_L3_ORDER:
            v = str(dims_l3.get(k, "") or "").strip()
            if v:
                label = "Sub-domain" if k == "Sub_domain" else ("Domain-table" if k == "Domain_table" else k)
                parts.append(f"{label}: {v}")
        for k in CANON_DIMS_L2_ORDER:
            v = str(dims_l2.get(k, "") or "").strip()
            if v:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def plot_series(
        self,
        series_key: str,
        *,
        save: bool = True,
        upload: bool = True,
        filename: Optional[str] = None,
    ) -> str:
        dom_dims = self._load_domain_dims()

        # localizar la serie y dims
        series_obj = None
        for obj in iter_jsonl(self._series_path()):
            e = SeriesL2Entry.model_validate(obj)
            if e.identity.series_key == series_key:
                series_obj = e
                break
        if series_obj is None:
            raise KeyError(f"series_key no encontrado en SERIES.jsonl: {series_key}")

        dims_l3 = dom_dims.get(series_obj.identity.domain_key, {}) or {}
        dims_l2 = dict(series_obj.dims)

        rec = self.get_records_by_series_key(series_key)
        df = _coerce_year_value_df([it.model_dump() for it in rec.data])
        if df.empty:
            raise RuntimeError("Records.data vacío para esa serie (no hay años válidos).")

        title = self._pretty_title_series(dims_l3, dims_l2)

        fig, (ax_ts, ax_hist) = plt.subplots(
            ncols=2,
            figsize=(14, 4.8),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        ax_ts.plot(df["Year"], df["Value"], marker="o", linestyle="-", color=PLOT_BLUE)
        ax_ts.set_title("Time-series")
        ax_ts.set_xlabel("Year")
        ax_ts.set_ylabel("Value")
        ax_ts.grid(alpha=0.3)

        vals = pd.to_numeric(df["Value"], errors="coerce").dropna().values
        ax_hist.hist(vals, bins=20, orientation="horizontal", color=PLOT_BLUE, alpha=0.85, edgecolor="none")
        ax_hist.set_title("Value distribution")
        ax_hist.set_xlabel("Frequency")
        ax_hist.set_ylabel("Value")
        ax_hist.grid(axis="x", alpha=0.3)

        fig.suptitle(title[:240] + ("…" if len(title) > 240 else ""), fontsize=10, y=1.02)
        plt.tight_layout()

        if not save:
            plt.show()
            plt.close(fig)
            return ""

        _ensure_dir(self._plots_dir())
        if filename is None:
            filename = f"series_{series_key.replace(':','_')[:32]}_2panel.png"
        out_path = os.path.join(self._plots_dir(), filename)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if upload:
            _try_upload(self.pipeline, out_path)

        return out_path

    def plot_series_by_rank_n_members(
        self,
        rank: int,
        *,
        descending: bool = True,
        save: bool = True,
        upload: bool = True,
    ) -> str:
        # rank por n_members: tomamos de SERIES.jsonl
        metas = []
        for obj in iter_jsonl(self._series_path()):
            e = SeriesL2Entry.model_validate(obj)
            n = int((e.properties.model_dump() or {}).get("n_members") or 0)
            metas.append((e.identity.series_key, n))
        if not metas:
            raise RuntimeError("SERIES.jsonl no tiene entradas.")

        metas_sorted = sorted(metas, key=lambda t: t[1], reverse=bool(descending))
        if not (1 <= int(rank) <= len(metas_sorted)):
            raise ValueError(f"rank fuera de rango: 1..{len(metas_sorted)}")
        sk = metas_sorted[int(rank) - 1][0]
        return self.plot_series(sk, save=save, upload=upload)

    # -------------------------------------------------------------------------
    # Sampling helpers (L2 series + join L3 dims; L3 domains)
    # -------------------------------------------------------------------------
    def _reservoir_sample_series_with_embeddings(
        self,
        *,
        sample_n: int,
        seed: int = 42,
        target_points: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rng = random.Random(int(seed))
        dom_dims = self._load_domain_dims()

        k = int(sample_n)
        out: List[Dict[str, Any]] = []
        seen = 0

        for obj in iter_jsonl(self._series_path()):
            e = SeriesL2Entry.model_validate(obj)
            emb = e.description.embedding
            if emb is None:
                continue
            try:
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                if v.size == 0:
                    continue
            except Exception:
                continue

            props = e.properties.model_dump() if hasattr(e, "properties") else {}
            row: Dict[str, Any] = {
                "domain_key": e.identity.domain_key,
                "series_key": e.identity.series_key,
                "n_members": int((props or {}).get("n_members") or 0),
                "embedding": v,
            }

            # L2 dims (y extras si existen) como strings
            for kk, vv in e.dims.items():
                row[str(kk)] = "" if vv is None else str(vv)

            # L3 dims join como strings
            l3 = dom_dims.get(e.identity.domain_key, {}) or {}
            for kk in CANON_DIMS_L3_ORDER:
                row[kk] = "" if l3.get(kk, None) is None else str(l3.get(kk))

            # asegura canon L2 existan
            for kk in CANON_DIMS_L2_ORDER:
                row.setdefault(kk, "")

            seen += 1
            if len(out) < k:
                out.append(row)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    out[j - 1] = row

        if target_points is not None and len(out) > int(target_points):
            out = rng.sample(out, k=int(target_points))

        return out

    def _load_l3_domains_with_embeddings(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for obj in iter_jsonl(self._domain_path()):
            e = DomainL3Entry.model_validate(obj)
            emb = e.description.embedding
            if emb is None:
                continue
            try:
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                if v.size == 0:
                    continue
            except Exception:
                continue

            row: Dict[str, Any] = {"domain_key": e.identity.domain_key, "embedding": v}
            for kk in CANON_DIMS_L3_ORDER:
                row[kk] = "" if e.dims.get(kk, None) is None else str(e.dims.get(kk))
            rows.append(row)
        return rows

    # -------------------------------------------------------------------------
    # Main: L2 scatter (colors) + L3 symbols (Domain_table)
    # -------------------------------------------------------------------------
    def plot_l2_with_l3_symbols(
        self,
        *,
        color_dim: str,
        symbol_dim: str = "Domain_table",
        method: str = "pca",
        sample_n_l2: int = 5000,
        seed: int = 42,
        target_points: Optional[int] = None,
        topk_colors: int = 12,
        palette: str = "tab20",
        layered: bool = True,
        save: bool = True,
        upload: bool = True,
        filename: Optional[str] = None,
        # styling
        l2_alpha_min: float = 0.22,
        l2_alpha_max: float = 0.80,
        l2_s_min: float = 8.0,
        l2_s_max: float = 26.0,
        l3_size: float = 220.0,
        l3_alpha: float = 0.95,
        l3_color: str = "black",
        l3_edgecolor: str = "white",
        l3_linewidth: float = 0.8,
        jitter_eps: float = 1e-4,
    ) -> str:
        rows_l2 = self._reservoir_sample_series_with_embeddings(
            sample_n=int(sample_n_l2),
            seed=int(seed),
            target_points=target_points,
        )
        rows_l3 = self._load_l3_domains_with_embeddings()

        if not rows_l2:
            raise RuntimeError("No encontré embeddings en SERIES.jsonl (description.embedding).")
        if not rows_l3:
            raise RuntimeError("No encontré embeddings en DOMAIN.jsonl (description.embedding).")

        df_l2 = pd.DataFrame(rows_l2)
        df_l3 = pd.DataFrame(rows_l3)

        if color_dim not in df_l2.columns:
            raise ValueError(f"color_dim='{color_dim}' no existe en L2. Ejemplos: {sorted(df_l2.columns)[:40]} ...")
        if symbol_dim not in df_l3.columns:
            raise ValueError(f"symbol_dim='{symbol_dim}' no existe en L3. Columnas: {sorted(df_l3.columns)[:40]} ...")

        # joint projection (same 2D space)
        df_l2p, df_l3p = compute_2d_projection_joint(
            df_l2, df_l3,
            method=str(method),
            random_state=int(seed),
            embedding_col="embedding",
            jitter_if_degenerate=True,
            jitter_eps=float(jitter_eps),
        )
        if df_l2p.empty:
            raise RuntimeError("df_l2p vacío tras la proyección (L2).")
        if df_l3p.empty:
            raise RuntimeError("df_l3p vacío tras la proyección (L3).")

        # --- colors for L2
        cats_raw = df_l2p[color_dim]
        cats = _collapse_topk(cats_raw, topk=int(topk_colors))
        counts = cats.value_counts(dropna=False)
        order = counts.index.tolist()
        if not layered:
            order = sorted([str(x) for x in order])

        cmap = plt.get_cmap(str(palette))
        colors = [cmap(i % cmap.N) for i in range(max(1, len(order)))]
        color_map = {order[i]: colors[i] for i in range(len(order))}

        # --- symbols for L3 (Domain_table)
        sym_series = df_l3p[symbol_dim].fillna("").astype(str).replace({"": "(empty)"})
        sym_cats = sym_series.unique().tolist()
        sym_cats = sorted(sym_cats)

        # good marker set (no star)
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p", "H", "d", "1", "2", "3", "4", "|", "_"]
        marker_map = {sym_cats[i]: markers[i % len(markers)] for i in range(len(sym_cats))}

        # --- plot
        fig, ax = plt.subplots(figsize=(11.5, 7.2))

        x = df_l2p["Dim 1"].to_numpy()
        y = df_l2p["Dim 2"].to_numpy()
        maxc = float(counts.iloc[0]) if len(counts) else 1.0

        # L2 points by color category
        l2_handles = []
        l2_labels = []

        for i, c in enumerate(order):
            mask = (cats.astype(object).values == c)
            n_c = float(counts.loc[c]) if c in counts.index else float(mask.sum())
            frac = (n_c / maxc) if maxc > 0 else 1.0

            # IMPORTANT: visibility-first (no “ghost points”)
            alpha = float(l2_alpha_max - (l2_alpha_max - l2_alpha_min) * frac)
            size = float(l2_s_max - (l2_s_max - l2_s_min) * frac)

            sc = ax.scatter(
                x[mask], y[mask],
                s=size,
                alpha=alpha,
                color=color_map[c],
                edgecolors="none",
                linewidths=0.0,
                label=str(c),
                zorder=1,
                rasterized=True,
            )
            l2_handles.append(sc)
            l2_labels.append(str(c))

        # L3 points by symbol category (Domain_table)
        l3_handles = []
        l3_labels = []

        for cat in sym_cats:
            m = (sym_series.values == cat)
            if not np.any(m):
                continue
            sc = ax.scatter(
                df_l3p.loc[m, "Dim 1"].to_numpy(),
                df_l3p.loc[m, "Dim 2"].to_numpy(),
                marker=marker_map[cat],
                s=float(l3_size),
                alpha=float(l3_alpha),
                color=str(l3_color),
                edgecolors=str(l3_edgecolor),
                linewidths=float(l3_linewidth),
                label=str(cat),
                zorder=10_000,
            )
            l3_handles.append(sc)
            l3_labels.append(str(cat))

        ax.set_title(f"L2 SERIES ({str(method).upper()}) · colors={color_dim} + symbols(L3)={symbol_dim}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(alpha=0.25)

        # Two legends: colors (L2) and symbols (L3)
        leg1 = ax.legend(
            handles=l2_handles[:35],
            labels=l2_labels[:35] + (["…"] if len(l2_labels) > 35 else []),
            title=f"L2 colors: {color_dim}",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=l3_handles[:35],
            labels=l3_labels[:35] + (["…"] if len(l3_labels) > 35 else []),
            title=f"L3 symbols: {symbol_dim}",
            bbox_to_anchor=(1.02, 0.55),
            loc="upper left",
            frameon=True,
        )

        fig.tight_layout()

        if not save:
            plt.show()
            plt.close(fig)
            return ""

        _ensure_dir(self._plots_dir())
        if filename is None:
            filename = f"l2_colors_{color_dim}__l3_symbols_{symbol_dim}__{str(method).lower()}.png"
        out_path = os.path.join(self._plots_dir(), filename)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        if upload:
            _try_upload(self.pipeline, out_path)

        return out_path


__all__ = [
    "HKBDataViz",
    "iter_jsonl",
    "compute_2d_projection",
    "compute_2d_projection_joint",
]