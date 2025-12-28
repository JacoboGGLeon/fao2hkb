from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np

@dataclass
class Embedder:
    model_name: str
    device: str = "auto"
    batch_size: int | str = "auto"

    def _lazy_load(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Embeddings requested but dependencies are missing. "
                "Install with: pip install 'fao2hkb[embeddings]'"
            ) from e
        dev = None if self.device == "auto" else self.device
        return SentenceTransformer(self.model_name, device=dev)

    def encode(self, texts: Sequence[str], *, normalize: bool = False) -> np.ndarray:
        model = self._lazy_load()
        bs = None if self.batch_size == "auto" else int(self.batch_size)
        emb = model.encode(
            list(texts),
            batch_size=bs,
            show_progress_bar=True,
            normalize_embeddings=bool(normalize),
        )
        return np.asarray(emb, dtype=np.float32)


