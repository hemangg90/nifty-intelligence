"""
Long-history candle backfill.

The research pipeline needs far more history than the live cache keeps (a couple of months). This walks
BACKWARD from the oldest cached bar in windows Dhan will serve (<= 90 days per intraday request) and merges
each window into the same parquet cache the DataManager uses, so every later backtest reads the longer
history. It also fills FORWARD from the newest cached bar (the cache goes stale whenever the token lapses).

It stops going back when Dhan returns nothing for several consecutive windows, and records the earliest
bar it actually obtained: the real lookback is whatever the source serves, never an assumed figure.
Resumable by construction (it always continues from the cache edges) and idempotent (merging is by
timestamp). A failed window is reported, not skipped silently.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from quant_intelligence.data.data_manager import CACHE_DIR, _flight_lock, _read_cache, _write_cache
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.timeframe import parse_timeframe_minutes

WINDOW_DAYS = 85  # under the adapter's 90-day intraday request limit
EMPTY_WINDOWS_TO_STOP = 2

FetchFn = Callable[[str, str, dt.datetime, dt.datetime], pd.DataFrame]


@dataclass
class BackfillResult:
    instrument: str
    timeframe: str
    rows_before: int = 0
    rows_after: int = 0
    earliest: dt.datetime | None = None
    latest: dt.datetime | None = None
    windows_fetched: int = 0
    stopped_reason: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def rows_added(self) -> int:
        return self.rows_after - self.rows_before

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["rows_added"] = self.rows_added
        d["earliest"] = str(self.earliest) if self.earliest else None
        d["latest"] = str(self.latest) if self.latest else None
        return d


def _default_fetch(instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    from quant_intelligence.data_adapters.dhan_adapter import DhanAdapter

    adapter = DhanAdapter()
    if not adapter.is_available():
        raise RuntimeError("Dhan credentials are not set (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)")
    return adapter.get_ohlcv(instrument, timeframe, start, end)


def _merge(cached: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if new is None or len(new) == 0:
        return cached if cached is not None else pd.DataFrame()
    new = new.assign(timestamp=pd.to_datetime(new["timestamp"]))
    if cached is None or len(cached) == 0:
        merged = new
    else:
        merged = pd.concat([cached, new], ignore_index=True)
    merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    # Prices are quoted to the paisa. Some responses arrive as float32 (22532.050781 for 22532.05), which can put a low a
    # hair above min(open, close) and trip the OHLC check on a bar that is really fine.
    for col in ("open", "high", "low", "close"):
        if col in merged:
            merged[col] = merged[col].astype("float64").round(2)
    return merged


def backfill_instrument(
    instrument: str,
    timeframe: str = "5min",
    target_start: dt.datetime | None = None,
    now: dt.datetime | None = None,
    fetch: FetchFn | None = None,
    cache_dir: Path | None = None,
    fill_forward: bool = True,
    max_windows: int = 60,
) -> BackfillResult:
    """Extend the cached history of `instrument` back to `target_start` (default: 5 years) and forward to `now`."""
    from quant_intelligence.utils.timeutil import now_ist

    if parse_timeframe_minutes(timeframe) >= 1440:
        raise ValueError("backfill_instrument handles intraday timeframes; daily history uses data/daily_history.py")
    fetch = fetch or _default_fetch
    now = (now or now_ist()).replace(tzinfo=None)
    target_start = target_start or (now - dt.timedelta(days=5 * 365))
    path = (cache_dir or CACHE_DIR) / f"{instrument}_{timeframe}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    result = BackfillResult(instrument=instrument, timeframe=timeframe)
    with _flight_lock(instrument, timeframe):
        cached = _read_cache(path) if path.exists() else None
        result.rows_before = 0 if cached is None else len(cached)

        # Forward: from the newest cached bar to now.
        if fill_forward and cached is not None and len(cached):
            last = cached["timestamp"].max().to_pydatetime()
            cursor = last - dt.timedelta(minutes=2 * parse_timeframe_minutes(timeframe))
            while cursor < now and result.windows_fetched < max_windows:
                end = min(cursor + dt.timedelta(days=WINDOW_DAYS), now)
                window = _fetch_window(fetch, instrument, timeframe, cursor, end, result)
                if window is None:
                    break
                cached = _merge(cached, window)
                cursor = end

        # Backward: from the oldest cached bar (or from now, if nothing is cached) to target_start.
        edge = now if cached is None or len(cached) == 0 else cached["timestamp"].min().to_pydatetime()
        empty = 0
        while edge > target_start and result.windows_fetched < max_windows:
            start = max(edge - dt.timedelta(days=WINDOW_DAYS), target_start)
            window = _fetch_window(fetch, instrument, timeframe, start, edge, result)
            if window is None:
                break
            if len(window) == 0:
                empty += 1
                if empty >= EMPTY_WINDOWS_TO_STOP:
                    result.stopped_reason = f"source returned no data for {EMPTY_WINDOWS_TO_STOP} consecutive windows before {start:%Y-%m-%d}"
                    break
            else:
                empty = 0
                cached = _merge(cached, window)
            edge = start
        else:
            if not result.stopped_reason:
                result.stopped_reason = "reached target start" if edge <= target_start else "window limit reached; run again to continue"

        if cached is not None and len(cached) and len(cached) != result.rows_before:
            _write_cache(cached, path)
        result.rows_after = 0 if cached is None else len(cached)
        if cached is not None and len(cached):
            result.earliest = cached["timestamp"].min().to_pydatetime()
            result.latest = cached["timestamp"].max().to_pydatetime()

    if not result.stopped_reason:
        result.stopped_reason = "stopped after an error" if result.errors else "done"
    log_event("backfill", f"{instrument} {timeframe}: {result.rows_added:+d} rows, earliest {result.earliest}",
              level="WARNING" if result.errors else "INFO", **result.as_dict())
    return result


def _fetch_window(fetch: FetchFn, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime,
                  result: BackfillResult) -> pd.DataFrame | None:
    """One window of candles (possibly empty), or None if the request failed (the error is recorded)."""
    try:
        df = fetch(instrument, timeframe, start, end)
    except Exception as e:  # network, auth, rate limit: stop and report, the next run resumes from the cache edge
        result.errors.append(f"{start:%Y-%m-%d}..{end:%Y-%m-%d}: {e}")
        return None
    result.windows_fetched += 1
    return df if df is not None else pd.DataFrame()
