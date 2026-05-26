from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..storage.io import dump_json, ensure_dir


class BaseMemoryStore(ABC):
    def __init__(self, storage_path: str) -> None:
        self.storage_dir = ensure_dir(storage_path) if storage_path else None

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def persist_snapshot(self, filename: str, payload: Dict[str, Any] | None = None) -> None:
        if not self.storage_dir:
            return
        data = payload if payload is not None else self.snapshot()
        dump_json(Path(self.storage_dir) / filename, data)

