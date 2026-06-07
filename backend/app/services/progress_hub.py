"""Progress hub — bridges the sync inference worker thread to async WebSocket clients.

The job worker runs on a background thread (the single-accelerator queue, DESIGN §9.3) and
publishes progress messages synchronously. WebSocket handlers are async. This hub fans a
published message out to each subscribed client's ``asyncio.Queue`` via
``loop.call_soon_threadsafe`` and remembers the last message per job so a client that
connects mid-run gets an immediate snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


class ProgressHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._last: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the main event loop (called from the app lifespan)."""
        self._loop = loop

    def publish(self, job_id: str, message: dict[str, Any]) -> None:
        """Publish a progress/status message (safe to call from the worker thread)."""
        with self._lock:
            self._last[job_id] = message
            queues = list(self._subs.get(job_id, ()))
        loop = self._loop
        if loop is None:
            return
        import contextlib

        for q in queues:
            with contextlib.suppress(RuntimeError):  # loop closed during shutdown
                loop.call_soon_threadsafe(q.put_nowait, message)

    def last(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._last.get(job_id)

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._subs.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subs.get(job_id)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subs.pop(job_id, None)

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._last.pop(job_id, None)


_hub: ProgressHub | None = None


def get_progress_hub() -> ProgressHub:
    global _hub
    if _hub is None:
        _hub = ProgressHub()
    return _hub
