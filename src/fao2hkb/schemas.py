
from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# =============================================================================
# FAO2HKB schema (L3 → L2 → L1)
#
# L3 (Domain card):     Domain / Sub_domain / Domain_table
# L2 (Series card):     Area / Item / Element / Unit (+ extras opcionales)
# L1 (Records stream):  list of {Year, Value} for a given series_key
#
# Keys:
#   domain_key = sha256(canon(dims_L3))
#   series_key = sha256(canon(dims_L3) + canon(dims_L2))
# =============================================================================

# ─────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────
SCHEMA_VERSION: str = "v_0_1_0"

# ─────────────────────────────────────────────────────────────
# Canon dims (3-level HKB)
# ─────────────────────────────────────────────────────────────
CANON_DIMS_L3_ORDER: List[str] = ["Domain", "Sub_domain", "Domain_table"]

# Nota: canon actual (sin Source). Si metes Source aquí, CAMBIA series_key -> bump version.
CANON_DIMS_L2_ORDER: List[str] = ["Area", "Item", "Element", "Unit"]

# L1: sólo eje temporal; Value es medida
CANON_DIMS_L1_ORDER: List[str] = ["Year"]
CANON_MEAS_L1: List[str] = ["Value"]
CANON_FIELDS_L1_ITEM_ORDER: List[str] = CANON_DIMS_L1_ORDER + CANON_MEAS_L1

# Conveniencia: L3 + L2
CANON_DIMS_SERIES_KEY_ORDER: List[str] = CANON_DIMS_L3_ORDER + CANON_DIMS_L2_ORDER


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _is_nan(x: Any) -> bool:
    try:
        import pandas as pd  # type: ignore
        return bool(pd.isna(x))
    except Exception:
        return x is None

def _to_clean_str(x: Any) -> str:
    if _is_nan(x):
        return ""
    return str(x).strip()

def _json_compact(obj: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))

def canonize_dims(
    dims: Dict[str, Any],
    *,
    order: List[str],
    keep_extras: bool = True,
) -> Dict[str, str]:
    """
    Canoniza un dict de dims:
      - asegura todas las keys en `order` (si faltan → "")
      - normaliza strings (strip) y NaN/None → ""
      - opcionalmente conserva extras (también limpiándolos)
    """
    d = dims if isinstance(dims, dict) else {}
    out: Dict[str, str] = {}

    for k in order:
        out[k] = _to_clean_str(d.get(k, ""))

    if keep_extras:
        for k, v in d.items():
            kk = _to_clean_str(k)
            if not kk:
                continue
            if kk in out:
                continue
            out[kk] = _to_clean_str(v)

    return out

def _sha256_prefixed(payload_obj: Any) -> str:
    b = _json_compact(payload_obj, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    return f"sha256:{h}"

def make_domain_key_sha256(dims_l3: Dict[str, Any]) -> str:
    payload = {"dims_l3": canonize_dims(dims_l3, order=CANON_DIMS_L3_ORDER, keep_extras=False)}
    return _sha256_prefixed(payload)

def make_series_key_sha256(dims_l3: Dict[str, Any], dims_l2: Dict[str, Any]) -> str:
    payload = {
        "dims_l3": canonize_dims(dims_l3, order=CANON_DIMS_L3_ORDER, keep_extras=False),
        "dims_l2": canonize_dims(dims_l2, order=CANON_DIMS_L2_ORDER, keep_extras=False),
    }
    return _sha256_prefixed(payload)

def build_domain_description_string(dims_l3: Dict[str, Any]) -> str:
    parts: List[str] = ["FAOSTAT domain"]
    for k in CANON_DIMS_L3_ORDER:
        v = _to_clean_str(dims_l3.get(k, ""))
        if v:
            label = "Sub-domain" if k == "Sub_domain" else ("Domain-table" if k == "Domain_table" else k)
            parts.append(f"{label}: {v}")
    return " | ".join(parts)

def build_series_description_string(
    dims_l3: Dict[str, Any],
    dims_l2: Dict[str, Any],
    *,
    include_codes: bool,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = ["FAOSTAT series"]

    # 1) L3 canon
    for k in CANON_DIMS_L3_ORDER:
        v = _to_clean_str(dims_l3.get(k, ""))
        if v:
            label = "Sub-domain" if k == "Sub_domain" else ("Domain-table" if k == "Domain_table" else k)
            parts.append(f"{label}: {v}")

    # 2) L2 canon
    for k in CANON_DIMS_L2_ORDER:
        v = _to_clean_str(dims_l2.get(k, ""))
        if v:
            parts.append(f"{k}: {v}")

    # 3) Extras opcionales
    if include_codes and extras:
        for k in sorted(extras.keys()):
            if k in set(CANON_DIMS_SERIES_KEY_ORDER):
                continue
            v = _to_clean_str(extras.get(k))
            if v:
                parts.append(f"{k}: {v}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
# Shared validators
# ─────────────────────────────────────────────────────────────
def _clean_sha256_strict(v: Any, *, field_name: str) -> str:
    s = _to_clean_str(v)
    if not s:
        raise ValueError(f"{field_name} cannot be empty")
    # Estricto (recomendado): exige prefijo sha256:
    if not s.startswith("sha256:"):
        raise ValueError(f"{field_name} must start with 'sha256:'")
    return s


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────
class EmbeddingMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding_model: Optional[str] = None
    embedding_dim: int = 0
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check(self):
        # Si hay embedding real (dim>0), exige modelo
        if self.embedding_dim > 0 and not self.embedding_model:
            raise ValueError("embedding_model is required when embedding_dim > 0")
        return self



# ---------- L3 DOMAIN ----------
class DomainIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_key: str = Field(..., description="sha256:<hex>")

    @field_validator("domain_key", mode="before")
    @classmethod
    def _vk(cls, v: Any) -> Any:
        return _clean_sha256_strict(v, field_name="domain_key")

class DomainDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    string: str
    embedding: Optional[List[float]] = None

class DomainL3Entry(BaseModel):
    """
    DOMAIN.jsonl (L3) — 1 entry por domain card
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["domain"] = "domain"
    schema_version: str = SCHEMA_VERSION

    identity: DomainIdentity
    dims: Dict[str, Any]  # mínimo: Domain/Sub_domain/Domain_table (+ extras)
    description: DomainDescription
    embedding_meta: EmbeddingMeta

    @field_validator("dims", mode="before")
    @classmethod
    def _canon_dims_l3(cls, v: Any) -> Any:
        d = v if isinstance(v, dict) else {}
        return canonize_dims(d, order=CANON_DIMS_L3_ORDER, keep_extras=True)


# ---------- L2 SERIES ----------
class SeriesIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_key: str = Field(..., description="sha256:<hex> (L3)")
    series_key: str = Field(..., description="sha256:<hex> (L3+L2)")

    @field_validator("domain_key", mode="before")
    @classmethod
    def _vk_domain(cls, v: Any) -> Any:
        return _clean_sha256_strict(v, field_name="domain_key")

    @field_validator("series_key", mode="before")
    @classmethod
    def _vk_series(cls, v: Any) -> Any:
        return _clean_sha256_strict(v, field_name="series_key")

class SeriesProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year_min: Optional[int] = None
    year_max: Optional[int] = None
    n_members: int = 0

    zeros: int = 0
    nulls: int = 0

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    variance: Optional[float] = None
    entropy: Optional[float] = None

class SeriesDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    string: str
    embedding: Optional[List[float]] = None

class SeriesL2Entry(BaseModel):
    """
    SERIES.jsonl (L2) — 1 entry por serie
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["series"] = "series"
    schema_version: str = SCHEMA_VERSION

    identity: SeriesIdentity
    dims: Dict[str, Any]  # mínimo: Area/Item/Element/Unit (+ extras)
    description: SeriesDescription
    properties: SeriesProperties
    embedding_meta: EmbeddingMeta

    @field_validator("dims", mode="before")
    @classmethod
    def _canon_dims_l2(cls, v: Any) -> Any:
        d = v if isinstance(v, dict) else {}
        return canonize_dims(d, order=CANON_DIMS_L2_ORDER, keep_extras=True)


# ---------- L1 RECORDS ----------
class RecordItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Year: int
    Value: Optional[float] = None

    @field_validator("Year", mode="before")
    @classmethod
    def _year_int(cls, v: Any) -> Any:
        if _is_nan(v):
            raise ValueError("Year cannot be null/NaN")
        return int(float(v))

    @field_validator("Value", mode="before")
    @classmethod
    def _value_float(cls, v: Any) -> Any:
        if _is_nan(v):
            return None
        try:
            return float(v)
        except Exception:
            s = _to_clean_str(v)
            return float(s) if s else None

class RecordsIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ✅ ESTRICTO: obligatorio
    domain_key: str = Field(..., description="sha256:<hex> (L3)")
    series_key: str = Field(..., description="sha256:<hex> (heredado de L2)")

    @field_validator("domain_key", mode="before")
    @classmethod
    def _vk_domain(cls, v: Any) -> Any:
        return _clean_sha256_strict(v, field_name="domain_key")

    @field_validator("series_key", mode="before")
    @classmethod
    def _vk_series(cls, v: Any) -> Any:
        return _clean_sha256_strict(v, field_name="series_key")

class RecordsL1Entry(BaseModel):
    """
    RECORDS.jsonl (L1) — 1 entry por serie (bundle)
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["records"] = "records"
    schema_version: str = SCHEMA_VERSION

    identity: RecordsIdentity
    data: List[RecordItem] = Field(default_factory=list)

# ---------- JSONL helpers ----------
def jsonl_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(jsonl_dumps(r) + "\n")
    return path