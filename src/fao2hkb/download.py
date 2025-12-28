from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

DEFAULT_TIMEOUT = (10, 120)  # (connect, read)

def download_file(
    url: str,
    out_path: str | Path,
    overwrite: bool = False,
    timeout=DEFAULT_TIMEOUT,
    chunk_size: int = 1 << 20,
) -> Path:
    """Robust HTTP(S) downloader (streaming + atomic rename)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    return out_path
