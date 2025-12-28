from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator

# -------------------------
# YAML config models
# -------------------------


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default="FAO2HKB_RUN", description="Unique run identifier.")
    output_root: str = Field(default="./runs", description="Root folder for all runs.")
    overwrite_downloads: bool = Field(default=False, description="Re-download sources even if present.")
    seed: int = Field(default=42, description="Seed for reproducibility.")


class SourceMeta(BaseModel):
    # allow Domain, Sub_domain, Domain_table (+ future fields)
    model_config = ConfigDict(extra="allow")


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: int
    url: str = Field(description="Official FAOSTAT Bulk Download URL (zip).")
    meta: Dict[str, Any] = Field(default_factory=dict)


class IOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: str = "raw"
    work_dir: str = "work"
    artifacts_dir: str = "artifacts"


class FiltersConfig(BaseModel):
    """
    Series-level filters.

    Defaults are chosen to NOT drop any series, so that:
    - drop_constant = False          → no filtro por var≈0
    - drop_if_zeros_pct_gte = None   → sin filtro por % de ceros
    - min_points = 1                 → acepta series con ≥1 punto
    """

    model_config = ConfigDict(extra="forbid")

    drop_constant: bool = False
    drop_if_zeros_pct_gte: Optional[float] = None  # None disables zeros filter
    min_points: int = 1


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_codes: bool = False

    # Canonical column names (will be matched case-insensitively)
    area_col: str = "Area"
    item_col: str = "Item"
    element_col: str = "Element"
    unit_col: str = "Unit"
    year_col: str = "Year"
    value_col: str = "Value"

    # Columns to drop (regex). Default drops common FAOSTAT Source/Code/Flag/Note/Unnamed columns.
    drop_columns_regex: str = r".*(Source|Code|Flag|Note|Unnamed).*"

    # L2 dims used to compute series_key
    dims_l2: List[str] = Field(default_factory=lambda: ["Area", "Item", "Element", "Unit"])

    filters: FiltersConfig = Field(default_factory=FiltersConfig)


class EmbeddingsConfig(BaseModel):
    """
    Embedding configuration.

    Permite dos formas de YAML válidas:

    embeddings:
      enabled: false

    embeddings:
      enabled: true
      levels: ["l3", "l2"]
      model_name: "Qwen/Qwen3-Embedding-0.6B"
      device: "auto"          # o "cuda"/"cpu"
      batch_size: "auto"      # o un entero
      max_seq_length: 1024
      encode_chunk_size: 25000
      l2_normalize: true
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    levels: List[Literal["l3", "l2"]] = Field(default_factory=lambda: ["l3", "l2"])
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    # Device selection: "auto" → cuda if available, else cpu.
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Batch size: "auto" o entero explícito.
    batch_size: Union[int, Literal["auto"]] = "auto"

    # Extra hyperparams (con defaults razonables del notebook)
    max_seq_length: int = 1024
    encode_chunk_size: int = 25000
    l2_normalize: bool = True


class PackagingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tar_gz: bool = True
    include_raw: bool = False


class ReproConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_env: bool = True
    record_pip_freeze: bool = True


class HKBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=RunConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    sources: List[SourceConfig]
    build: BuildConfig = Field(default_factory=BuildConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    packaging: PackagingConfig = Field(default_factory=PackagingConfig)
    reproducibility: ReproConfig = Field(default_factory=ReproConfig)

    @field_validator("sources")
    @classmethod
    def _non_empty_sources(cls, v: List[SourceConfig]) -> List[SourceConfig]:
        if not v:
            raise ValueError("sources must be non-empty")
        return v


def load_config(path: str | Path) -> HKBConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = HKBConfig.model_validate(data)

    # normalize output_root to absolute-ish for stability
    cfg.run.output_root = str(Path(cfg.run.output_root).expanduser())
    return cfg


def resolve_run_dir(cfg: HKBConfig) -> Path:
    return Path(cfg.run.output_root) / cfg.run.run_id
