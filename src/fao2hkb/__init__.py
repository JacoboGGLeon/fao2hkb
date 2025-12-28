from __future__ import annotations

from .config import HKBConfig, load_config
from .pipeline import FAO2HKBPipeline

def run(config_path: str) -> dict:
    """Convenience entrypoint: load YAML config and run pipeline."""
    cfg = load_config(config_path)
    pipe = FAO2HKBPipeline(cfg)
    pipe.download_data(overwrite=cfg.run.overwrite_downloads)
    df_all = pipe.load_normalized_faostat()
    out = pipe.build_hkb_jsonl(df_all=df_all)
    return out

__all__ = ["HKBConfig", "load_config", "FAO2HKBPipeline", "run"]
