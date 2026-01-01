from __future__ import annotations

from .config import HKBConfig, load_config
from .pipeline import FAO2HKBPipeline

def run(config_path: str, *, execution: bool = False) -> dict:
    """Convenience entrypoint: load YAML config and run pipeline.

    Parameters
    ----------
    execution:
        If True, writes an execution timeline under <run_dir>/work/execution/
        and returns pointers to those files in the output dict.
    """
    cfg = load_config(config_path)
    pipe = FAO2HKBPipeline(cfg, execution=bool(execution))
    pipe.download_data(overwrite=cfg.run.overwrite_downloads)
    df_all = pipe.load_normalized_faostat()
    out = pipe.build_hkb_jsonl(df_all=df_all)

    # Attach execution pointers (only when enabled)
    if getattr(pipe, "exec", None) is not None and getattr(pipe.exec, "enabled", False):
        try:
            pipe.exec.finalize()
        except Exception:
            # never fail the run because of reporting
            pass

        out["execution_dir"] = str(pipe.exec.out_dir)
        out["execution_milestones_jsonl"] = str(pipe.exec.jsonl_path)
        out["execution_milestones_json"] = str(pipe.exec.json_path)
        out["execution_summary"] = str(pipe.exec.summary_path)

    return out

__all__ = ["HKBConfig", "load_config", "FAO2HKBPipeline", "run"]
