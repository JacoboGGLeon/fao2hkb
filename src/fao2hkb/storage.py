from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional

class StorageBackend(Protocol):
    """Agnostic persistence backend."""
    def mkdir(self, path: Path) -> None: ...
    def write_bytes(self, path: Path, data: bytes) -> None: ...
    def copy_in(self, src: Path, dst: Path) -> None: ...
    def exists(self, path: Path) -> bool: ...

@dataclass
class LocalBackend:
    """Default backend: write to local filesystem."""
    def mkdir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.mkdir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def copy_in(self, src: Path, dst: Path) -> None:
        self.mkdir(dst.parent)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def exists(self, path: Path) -> bool:
        return path.exists()

@dataclass
class NullBackend:
    """No-op backend (useful for testing)."""
    def mkdir(self, path: Path) -> None:
        return

    def write_bytes(self, path: Path, data: bytes) -> None:
        return

    def copy_in(self, src: Path, dst: Path) -> None:
        return

    def exists(self, path: Path) -> bool:
        return False
