from __future__ import annotations

import contextlib
import contextvars
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union, cast

from .utils import now_utc_iso

_T = TypeVar("_T")

# Context variable so *any* decorated function (not just methods) can record events
_CURRENT_TRACKER: contextvars.ContextVar["ExecutionTracker | None"] = contextvars.ContextVar(
    "fao2hkb_execution_tracker", default=None
)

def get_tracker() -> "ExecutionTracker | None":
    return _CURRENT_TRACKER.get()

@contextlib.contextmanager
def execution_context(tracker: "ExecutionTracker | None"):
    token = None
    if tracker is not None:
        token = _CURRENT_TRACKER.set(tracker)
    try:
        yield
    finally:
        if token is not None:
            _CURRENT_TRACKER.reset(token)

def _jsonable(x: Any) -> Any:
    """Best-effort JSON-safe conversion (keeps logs robust)."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    # numpy scalars
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    # fallback
    return str(x)

@dataclass
class ExecutionTracker:
    """Write an execution timeline under <run_dir>/work/execution/.

    - milestones.jsonl is appended *as events happen* (survives crashes)
    - milestones.json + summary.json are written on finalize()
    """

    work_dir: Path
    enabled: bool = True
    run_id: Optional[str] = None

    t0: float = field(default_factory=time.perf_counter, init=False)
    _events: List[Dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def out_dir(self) -> Path:
        return self.work_dir / "execution"

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "milestones.jsonl"

    @property
    def json_path(self) -> Path:
        return self.out_dir / "milestones.json"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"

    def mark(self, name: str, **meta: Any) -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)

        event: Dict[str, Any] = {
            "name": str(name),
            "utc": now_utc_iso(),
            "t_seconds": float(time.perf_counter() - self.t0),
        }
        if self.run_id:
            event["run_id"] = self.run_id
        if meta:
            event["meta"] = _jsonable(meta)

        self._events.append(event)

        # append JSONL immediately (crash-safe)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finalize(self) -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # full JSON
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, ensure_ascii=False, indent=2)

        # simple summary
        total = float(self._events[-1]["t_seconds"]) if self._events else 0.0
        summary = {
            "run_id": self.run_id,
            "n_events": int(len(self._events)),
            "total_wall_seconds": total,
            "generated_utc": now_utc_iso(),
            "paths": {
                "milestones_jsonl": str(self.jsonl_path),
                "milestones_json": str(self.json_path),
            },
        }

        # stage durations (best-effort: stage.start -> stage.done)
        starts: Dict[str, float] = {}
        durs: Dict[str, float] = {}
        for e in self._events:
            nm = str(e.get("name", ""))
            t = float(e.get("t_seconds", 0.0))
            if nm.endswith(".start"):
                stage = nm[:-6]
                starts.setdefault(stage, t)
            elif nm.endswith(".done"):
                stage = nm[:-5]
                if stage in starts:
                    durs[stage] = float(t - starts[stage])
        if durs:
            summary["stage_seconds"] = dict(sorted(durs.items(), key=lambda kv: kv[0]))

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

MetaFn = Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]

def tracked(
    stage: str | None = None,
    *,
    meta: MetaFn | None = None,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator that emits <stage>.start/.done/.error events if a tracker exists.

    Tracker resolution priority:
      1) if first arg looks like a pipeline instance with .exec enabled, use that
      2) else use contextvar tracker (execution_context)
    """
    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        stg = stage or fn.__qualname__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            tracker = None
            if args:
                self0 = args[0]
                tr = getattr(self0, "exec", None)
                if tr is not None and getattr(tr, "enabled", False):
                    tracker = cast(ExecutionTracker, tr)
            if tracker is None:
                tracker = get_tracker()

            token = None
            if tracker is not None:
                token = _CURRENT_TRACKER.set(tracker)

            t_start = time.perf_counter()
            m: Dict[str, Any] = {}
            if meta is not None:
                try:
                    m = meta(args, kwargs) or {}
                except Exception:
                    m = {}

            if tracker is not None:
                tracker.mark(f"{stg}.start", **m)

            try:
                out = fn(*args, **kwargs)
                if tracker is not None:
                    tracker.mark(f"{stg}.done", elapsed_seconds=float(time.perf_counter() - t_start), **m)
                return out
            except Exception as e:
                if tracker is not None:
                    tracker.mark(
                        f"{stg}.error",
                        error_type=type(e).__name__,
                        error=str(e)[:500],
                        elapsed_seconds=float(time.perf_counter() - t_start),
                        **m,
                    )
                raise
            finally:
                if token is not None:
                    _CURRENT_TRACKER.reset(token)

        return wrapper
    return deco
