from __future__ import annotations

import os
import sys
import json
import hashlib
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .schemas import SCHEMA_VERSION, DomainL3Entry, SeriesL2Entry, RecordsL1Entry


# -----------------------------------------------------------------------------
# Hashing / I/O
# -----------------------------------------------------------------------------
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_stat(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    }


def iter_jsonl(path: str):
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


def _run_cmd(cmd: List[str], timeout: int = 15) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return 0, out.decode("utf-8", errors="replace").strip()
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1), (e.output or b"").decode("utf-8", errors="replace").strip()
    except Exception as e:
        return 1, repr(e)


def pip_freeze_text(timeout: int = 30) -> Optional[str]:
    code, out = _run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=timeout)
    if code == 0 and out:
        return out
    return None


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def collect_environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uname": {
            "system": platform.uname().system,
            "node": platform.uname().node,
            "release": platform.uname().release,
            "version": platform.uname().version,
            "machine": platform.uname().machine,
        },
    }

    versions: Dict[str, Optional[str]] = {}
    try:
        from importlib import metadata
        for pkg in [
            "numpy",
            "pandas",
            "torch",
            "sentence-transformers",
            "pydantic",
            "scikit-learn",
            "matplotlib",
            "tqdm",
            "gdown",
        ]:
            try:
                versions[pkg] = metadata.version(pkg)
            except Exception:
                versions[pkg] = None
    except Exception:
        pass
    info["packages"] = versions

    code, root = _run_cmd(["git", "rev-parse", "--show-toplevel"], timeout=5)
    if code == 0 and root:
        _, commit = _run_cmd(["git", "rev-parse", "HEAD"], timeout=5)
        _, branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5)
        _, describe = _run_cmd(["git", "describe", "--always", "--dirty", "--tags"], timeout=5)
        info["git"] = {
            "root": root,
            "commit": commit or None,
            "branch": branch or None,
            "describe": describe or None,
        }

    return info


# -----------------------------------------------------------------------------
# Validation (STRICT to schemas.py)
# -----------------------------------------------------------------------------
@dataclass
class ValidationSummary:
    ok: bool
    n_domains_l3: int = 0
    n_series_l2: int = 0
    n_series_l1: int = 0

    # key coverage
    series_keys_match: bool = False
    domain_keys_covered: bool = False

    # embedding dims
    embedding_dim_l3: int = 0
    embedding_dim_l2: int = 0

    # sanity checks
    n_members_mismatch: int = 0
    missing_series_in_l1: int = 0
    missing_series_in_l2: int = 0
    missing_domain_in_l3: int = 0

    # errors (limited)
    domain_errors: List[Dict[str, Any]] = field(default_factory=list)
    series_errors: List[Dict[str, Any]] = field(default_factory=list)
    records_errors: List[Dict[str, Any]] = field(default_factory=list)


def validate_hkb_jsonl(
    domain_jsonl: str,
    series_jsonl: str,
    records_jsonl: str,
    *,
    max_errors: int = 20,
) -> ValidationSummary:
    """
    Valida:
      1) DOMAIN.jsonl cumple DomainL3Entry (strict).
      2) SERIES.jsonl cumple SeriesL2Entry (strict).
      3) RECORDS.jsonl cumple RecordsL1Entry (strict).
      4) Keys: SERIES.series_key == RECORDS.series_key
      5) Domain coverage: SERIES.domain_key ⊆ DOMAIN.domain_key
      6) n_members == len(data)
      7) embedding_dim consistente por capa (L3 y L2)
    """
    vs = ValidationSummary(ok=True)

    # --- DOMAIN (L3)
    domain_keys: set = set()
    emb_dims_l3: List[int] = []

    for i, obj in enumerate(iter_jsonl(domain_jsonl), 1):
        try:
            e = DomainL3Entry.model_validate(obj)
            dk = e.identity.domain_key
            domain_keys.add(dk)

            emb = e.description.embedding
            if emb is not None:
                v = np.asarray(emb, dtype=np.float32)
                if v.ndim == 1 and v.shape[0] > 0:
                    emb_dims_l3.append(int(v.shape[0]))
        except Exception as ex:
            vs.ok = False
            if len(vs.domain_errors) < int(max_errors):
                vs.domain_errors.append({"line": i, "error": repr(ex)})

    vs.n_domains_l3 = int(len(domain_keys))

    # --- SERIES (L2)
    series_map: Dict[str, Dict[str, Any]] = {}
    series_domain_keys: set = set()
    emb_dims_l2: List[int] = []

    for i, obj in enumerate(iter_jsonl(series_jsonl), 1):
        try:
            e = SeriesL2Entry.model_validate(obj)
            sk = e.identity.series_key
            dk = e.identity.domain_key
            series_domain_keys.add(dk)

            props = e.properties.model_dump()
            nm = int(props.get("n_members") or 0)

            emb = e.description.embedding
            if emb is not None:
                v = np.asarray(emb, dtype=np.float32)
                if v.ndim == 1 and v.shape[0] > 0:
                    emb_dims_l2.append(int(v.shape[0]))

            series_map[sk] = {"domain_key": dk, "n_members": nm}
        except Exception as ex:
            vs.ok = False
            if len(vs.series_errors) < int(max_errors):
                vs.series_errors.append({"line": i, "error": repr(ex)})

    vs.n_series_l2 = int(len(series_map))

    # --- RECORDS (L1)
    records_map: Dict[str, int] = {}
    for i, obj in enumerate(iter_jsonl(records_jsonl), 1):
        try:
            e = RecordsL1Entry.model_validate(obj)
            sk = e.identity.series_key
            n = int(len(e.data))
            records_map[sk] = n
        except Exception as ex:
            vs.ok = False
            if len(vs.records_errors) < int(max_errors):
                vs.records_errors.append({"line": i, "error": repr(ex)})

    vs.n_series_l1 = int(len(records_map))

    # --- SERIES keys match
    keys_l2 = set(series_map.keys())
    keys_l1 = set(records_map.keys())
    vs.series_keys_match = (keys_l2 == keys_l1)

    missing_in_l1 = sorted(list(keys_l2 - keys_l1))
    missing_in_l2 = sorted(list(keys_l1 - keys_l2))
    vs.missing_series_in_l1 = int(len(missing_in_l1))
    vs.missing_series_in_l2 = int(len(missing_in_l2))
    if missing_in_l1 or missing_in_l2:
        vs.ok = False

    # --- DOMAIN coverage
    missing_domains = sorted(list(series_domain_keys - domain_keys))
    vs.missing_domain_in_l3 = int(len(missing_domains))
    vs.domain_keys_covered = (len(missing_domains) == 0)
    if missing_domains:
        vs.ok = False

    # --- n_members sanity
    mism = 0
    common = keys_l2.intersection(keys_l1)
    for sk in common:
        if int(series_map[sk]["n_members"]) != int(records_map[sk]):
            mism += 1
    vs.n_members_mismatch = int(mism)
    if mism > 0:
        vs.ok = False

    # --- embedding dims consistency
    if emb_dims_l3:
        d0 = emb_dims_l3[0]
        if any(d != d0 for d in emb_dims_l3):
            vs.ok = False
        vs.embedding_dim_l3 = int(d0)

    if emb_dims_l2:
        d0 = emb_dims_l2[0]
        if any(d != d0 for d in emb_dims_l2):
            vs.ok = False
        vs.embedding_dim_l2 = int(d0)

    return vs


# -----------------------------------------------------------------------------
# Repro manager for FAO_NORM2HKB (3 artifacts)
# -----------------------------------------------------------------------------
@dataclass
class ReproducibilityManager:
    pipeline: Any
    domain_jsonl_name: str = "DOMAIN.jsonl"
    series_jsonl_name: str = "SERIES.jsonl"
    records_jsonl_name: str = "RECORDS.jsonl"

    def _data_dir(self) -> str:
        return str(getattr(self.pipeline, "data_folder", "data"))

    def _plots_dir(self) -> str:
        return str(getattr(self.pipeline, "plots_folder", "plots"))

    def _domain_path(self) -> str:
        return os.path.join(self._data_dir(), self.domain_jsonl_name)

    def _series_path(self) -> str:
        return os.path.join(self._data_dir(), self.series_jsonl_name)

    def _records_path(self) -> str:
        return os.path.join(self._data_dir(), self.records_jsonl_name)

    def _run_manifest_path(self) -> str:
        return os.path.join(self._data_dir(), "run_manifest.json")

    def discover_artifacts(self) -> Dict[str, str]:
        d: Dict[str, str] = {
            "run_manifest": self._run_manifest_path(),
            "DOMAIN_jsonl": self._domain_path(),
            "SERIES_jsonl": self._series_path(),
            "RECORDS_jsonl": self._records_path(),
        }
        # opcionales
        for name in ["DOMAIN.tar.gz", "SERIES.tar.gz", "RECORDS.tar.gz"]:
            p = os.path.join(self._data_dir(), name)
            if os.path.exists(p):
                d[name.replace(".", "_")] = p
        return d

    def export_pip_freeze(self, *, filename: str = "requirements_freeze.txt", upload: bool = True) -> Optional[str]:
        txt = pip_freeze_text()
        if not txt:
            return None
        out = os.path.join(self._data_dir(), filename)
        _ensure_dir(self._data_dir())
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt.strip() + "\n")
        if upload:
            _try_upload(self.pipeline, out)
        return out

    def write_repro_manifest(
        self,
        *,
        filename: str = "repro_manifest.json",
        validate: bool = True,
        max_errors: int = 20,
        include_pip_freeze_file: bool = True,
        upload: bool = True,
    ) -> str:
        artifacts = self.discover_artifacts()

        pip_freeze_path = None
        if include_pip_freeze_file:
            pip_freeze_path = self.export_pip_freeze(upload=upload)

        val = None
        if validate:
            val = validate_hkb_jsonl(
                artifacts["DOMAIN_jsonl"],
                artifacts["SERIES_jsonl"],
                artifacts["RECORDS_jsonl"],
                max_errors=int(max_errors),
            )

        cfg = {
            "schema_version": SCHEMA_VERSION,
            "run_id": getattr(self.pipeline, "run_id", None),
            "run_started_utc": getattr(self.pipeline, "run_started_utc", None),
            "seed": getattr(self.pipeline, "seed", None),
            "dims_canon": getattr(self.pipeline, "dims_canon", None),
            "include_codes": getattr(self.pipeline, "include_codes", None),
            "embedding_model": getattr(self.pipeline, "embedding_model", None),
            "max_seq_length": getattr(self.pipeline, "max_seq_length", None),
            "device": getattr(self.pipeline, "device", None),
            "batch_size": getattr(self.pipeline, "batch_size", None),
            "encode_chunk_size": getattr(self.pipeline, "encode_chunk_size", None),
            "l2_normalize": getattr(self.pipeline, "l2_normalize", None),
            "persist_backend": getattr(self.pipeline, "persist_backend", None),
            "strict_persistence": getattr(self.pipeline, "strict_persistence", None),
            "drivefs_run_root": getattr(self.pipeline, "drivefs_run_root", None),
            "download_folder": getattr(self.pipeline, "download_folder", None),
            "local_root": getattr(self.pipeline, "local_root", None),
            "data_folder": getattr(self.pipeline, "data_folder", None),
            "plots_folder": getattr(self.pipeline, "plots_folder", None),
        }

        art_info: Dict[str, Any] = {}
        for k, p in artifacts.items():
            art_info[k] = {
                "stat": file_stat(p),
                "sha256": sha256_file(p) if os.path.exists(p) else None,
            }

        payload: Dict[str, Any] = {
            "written_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "contract": {
                "schema_version": SCHEMA_VERSION,
                "artifacts": ["DOMAIN.jsonl (L3)", "SERIES.jsonl (L2)", "RECORDS.jsonl (L1)"],
                "hrp_expansion": "L3.domain_key -> L2.domain_key -> L1.series_key",
                "embeddings": "L3 + L2 (DOMAIN.description.embedding, SERIES.description.embedding)",
            },
            "pipeline_config": cfg,
            "environment": collect_environment_info(),
            "artifacts": art_info,
            "pip_freeze_file": pip_freeze_path,
            "validation": None if val is None else {
                "ok": val.ok,
                "n_domains_l3": val.n_domains_l3,
                "n_series_l2": val.n_series_l2,
                "n_series_l1": val.n_series_l1,
                "series_keys_match": val.series_keys_match,
                "domain_keys_covered": val.domain_keys_covered,
                "embedding_dim_l3": val.embedding_dim_l3,
                "embedding_dim_l2": val.embedding_dim_l2,
                "n_members_mismatch": val.n_members_mismatch,
                "missing_series_in_l1": val.missing_series_in_l1,
                "missing_series_in_l2": val.missing_series_in_l2,
                "missing_domain_in_l3": val.missing_domain_in_l3,
                "domain_errors": val.domain_errors,
                "series_errors": val.series_errors,
                "records_errors": val.records_errors,
            },
        }

        out = os.path.join(self._data_dir(), filename)
        _ensure_dir(self._data_dir())
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if upload:
            _try_upload(self.pipeline, out)

        return out


__all__ = [
    "ReproducibilityManager",
    "validate_hkb_jsonl",
    "collect_environment_info",
    "pip_freeze_text",
    "sha256_file",
]