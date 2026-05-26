from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_openai_client(dotenv_path: str | Path = ".env") -> OpenAI:
    _load_dotenv(dotenv_path)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in .env or the environment.")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


class RoundRobinClientPool:
    """Thread-safe pool with automatic failover when a key runs out of credits."""

    def __init__(self, api_keys: List[str], base_url: str | None = None):
        if not api_keys:
            raise ValueError("At least one API key is required.")
        self._clients: List[OpenAI] = []
        self._dead_keys: set[int] = set()
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        for key in api_keys:
            self._clients.append(OpenAI(api_key=key.strip(), **kwargs))
        self._index = 0
        self._lock = threading.Lock()
        self._thread_map: dict[int, int] = {}  # thread_id -> client_index

    def get_client_for_thread(self) -> OpenAI:
        """Assign a dedicated client to the current thread (sticky)."""
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._thread_map:
                idx = self._next_alive_index()
                self._thread_map[tid] = idx
                self._index += 1
            return self._clients[self._thread_map[tid]]

    def mark_dead(self, client: OpenAI) -> None:
        """Mark a client's key as exhausted and reassign affected threads."""
        with self._lock:
            for i, c in enumerate(self._clients):
                if c is client:
                    self._dead_keys.add(i)
                    # Reassign all threads using this dead key
                    new_idx = self._next_alive_index()
                    for tid, idx in self._thread_map.items():
                        if idx == i:
                            self._thread_map[tid] = new_idx
                    break

    def _next_alive_index(self) -> int:
        for offset in range(len(self._clients)):
            idx = (self._index + offset) % len(self._clients)
            if idx not in self._dead_keys:
                return idx
        # All dead, fall back to first (will error but nothing we can do)
        return 0

    @property
    def primary(self) -> OpenAI:
        return self._clients[0]

    @property
    def all_dead(self) -> bool:
        return len(self._dead_keys) >= len(self._clients)

    def __len__(self) -> int:
        return len(self._clients)


class ThreadAwareClient:
    """Drop-in OpenAI client proxy with automatic key failover.

    Wraps API calls: if a call fails with quota/auth error, marks that key dead
    and retries with the next available key.
    """

    RETRY_ERRORS = (
        "insufficient_quota",
        "rate_limit_exceeded",
        "billing_hard_limit_reached",
        "access_terminated",
        "account_deactivated",
        "credits_exhausted",
    )
    RETRY_MESSAGE_MARKERS = (
        "quota",
        "billing",
        "insufficient",
        "balance",
        "credit",
        "payment",
        "余额",
        "额度",
        "欠费",
    )

    def __init__(self, pool: RoundRobinClientPool):
        self._pool = pool
        self.usage_callback = None
        self._chat = _ChatProxy(self)

    @property
    def chat(self):
        return self._chat

    def __getattr__(self, name):
        return getattr(self._pool.get_client_for_thread(), name)

    def _notify_usage(self, response) -> None:
        callback = getattr(self, "usage_callback", None)
        if callable(callback):
            try:
                callback(response)
            except Exception:
                pass


class _ChatProxy:
    def __init__(self, owner: ThreadAwareClient):
        self._owner = owner

    @property
    def completions(self):
        return _CompletionsProxy(self._owner)


class _CompletionsProxy:
    def __init__(self, owner: ThreadAwareClient):
        self._owner = owner

    def create(self, **kwargs):
        last_error = None
        for _attempt in range(len(self._owner._pool)):
            client = self._owner._pool.get_client_for_thread()
            try:
                response = client.chat.completions.create(**kwargs)
                self._owner._notify_usage(response)
                return response
            except Exception as e:
                error_code = getattr(e, "code", "") or ""
                error_msg = str(e).lower()
                is_quota_error = (
                    any(code in error_code for code in ThreadAwareClient.RETRY_ERRORS)
                    or any(marker in error_msg for marker in ThreadAwareClient.RETRY_MESSAGE_MARKERS)
                )
                if is_quota_error:
                    self._owner._pool.mark_dead(client)
                    last_error = e
                    continue
                raise
        raise last_error or RuntimeError("All API keys exhausted.")


def build_client_pool(api_keys_csv: str = "", dotenv_path: str | Path = ".env") -> RoundRobinClientPool:
    """Build a client pool. Priority: --api-keys arg > OPENAI_API_KEYS env > OPENAI_API_KEY env."""
    _load_dotenv(dotenv_path)
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_keys_csv:
        keys = [k.strip() for k in api_keys_csv.split(",") if k.strip()]
    else:
        multi = os.getenv("OPENAI_API_KEYS", "")
        if multi:
            keys = [k.strip() for k in multi.split(",") if k.strip()]
        else:
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("No API keys. Set OPENAI_API_KEYS or OPENAI_API_KEY in .env, or use --api-keys.")
            keys = [key]

    return RoundRobinClientPool(api_keys=keys, base_url=base_url)
