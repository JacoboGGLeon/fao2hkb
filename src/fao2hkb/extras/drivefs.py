from __future__ import annotations

"""Optional DriveFS helpers (NOT used by the core library).

This exists only to preserve the notebook's DriveFS workflow as an *optional* add-on.
The core pipeline is intentionally Drive-agnostic.

To use DriveFS in Colab you must:
- mount Drive manually (google.colab.drive.mount)
- install optional deps yourself (e.g., gdown) if needed
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DriveFSBackend:
    mydrive_root: str = "MyDrive"
    run_root: str = "FAO2HKB_Runs"

    def base(self) -> Path:
        return Path("/content/drive") / self.mydrive_root / self.run_root

    # TODO: implement verified copy (size + sha256) if you really need it.
