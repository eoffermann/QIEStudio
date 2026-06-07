"""Logging helpers with elapsed-time context and flushed output.

Per the user's hard convention: any slow phase MUST emit progress *before* it begins, with
elapsed-time context, and output must be flushed (Windows/redirected stdout is block-
buffered otherwise). Use :func:`phase` around slow startup/inference phases.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

_PROCESS_START = time.monotonic()


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a flushed stream handler and elapsed-time prefix."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [+%(rel)7.1fs] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    class _RelFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rel = time.monotonic() - _PROCESS_START
            return True

    handler.addFilter(_RelFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Make sure logs flush immediately (block-buffered pipes otherwise hide stalls).
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def elapsed() -> float:
    """Seconds since process start."""
    return time.monotonic() - _PROCESS_START


@contextmanager
def phase(logger: logging.Logger, message: str, *, level: int = logging.INFO) -> Iterator[None]:
    """Log *before* a slow phase and again on completion with how long it took.

    Usage::

        with phase(log, "Loading Qwen-Image pipeline (bf16) — this can take a minute"):
            pipe = load_pipeline(...)
    """
    logger.log(level, "%s ...", message)
    for h in logging.getLogger().handlers:
        h.flush()
    start = time.monotonic()
    try:
        yield
    finally:
        logger.log(level, "%s — done in %.1fs", message, time.monotonic() - start)
        for h in logging.getLogger().handlers:
            h.flush()
