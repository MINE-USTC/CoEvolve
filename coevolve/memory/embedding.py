from __future__ import annotations

import os
import threading
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

_EMBEDDER: SentenceTransformer | None = None
_LOCK = threading.Lock()


def _resolve_embedder_path() -> str:
    model_path = str(os.getenv("EMBEDDING_MODEL_PATH", "PATH_TO_EMBEDDING_MODEL")).strip()
    if not model_path or model_path == "PATH_TO_EMBEDDING_MODEL":
        raise ValueError(
            "EMBEDDING_MODEL_PATH is not configured. "
            "Please set EMBEDDING_MODEL_PATH in .env to a valid local sentence-transformers model directory."
        )
    path = Path(model_path)
    if (path.is_absolute() or model_path.startswith(".")) and not path.exists():
        raise FileNotFoundError(f"EMBEDDING_MODEL_PATH does not exist: {model_path}")
    return model_path


def init_embedder() -> SentenceTransformer:
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = _resolve_embedder_path()
            _EMBEDDER = SentenceTransformer(model_path, device=device)
    return _EMBEDDER


def get_embedder() -> SentenceTransformer:
    return init_embedder()
