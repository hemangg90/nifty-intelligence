"""Per-context override of the backtest execution timing.

Research evidence must be built on fills that were achievable (`next_open`), whatever EXECUTION_TIMING the live process
runs with. A global flip would race with a scan running on another thread, so the override lives in a ContextVar: it is
seen only by code running inside the `with` block on the calling thread (and tasks it spawns), and nothing else.
"""
from __future__ import annotations

import contextlib
import contextvars

_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("backtest_execution_timing", default=None)


def timing_override() -> str | None:
    return _override.get()


@contextlib.contextmanager
def execution_timing(value: str):
    if value not in ("signal_close", "next_open"):
        raise ValueError(f"unknown execution timing {value!r}")
    token = _override.set(value)
    try:
        yield
    finally:
        _override.reset(token)
