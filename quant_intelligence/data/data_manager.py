"""
DataManager: unified entry point for fetching OHLCV data.

Order of preference: CSV/Parquet (if the user has dropped real market data
into data_cache/csv) -> Dhan (if credentials configured). There is NO
synthetic fallback: if neither source yields data, DataUnavailableError is
raised. Results are cached to Parquet in data_cache/parquet_cache.

Cached candles are refreshed only when a new bar has CLOSED (about one small request per
instrument per bar), fetching just the missing tail and merging it into the cache. That keeps signals
on the latest closed bar while staying far below Dhan's data-API limits.

Every fetch runs through data-quality validation and records a
MarketDataMetadata row so the system has an auditable record of what data
backed every decision.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from pathlib import Path

import pandas as pd

from quant_intelligence.config.settings import DATA_CACHE_DIR
from quant_intelligence.data_adapters.csv_adapter import CSVAdapter
from quant_intelligence.data_adapters.dhan_adapter import DhanAdapter
from quant_intelligence.data.quality import (
    MAX_OUT_OF_SESSION_FRACTION,
    QUALITY_FAIL,
    clean_ohlcv,
    out_of_session_mask,
    validate_ohlcv,
)
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_calendar import expected_last_closed_bar_start, most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeframe import parse_timeframe_minutes as _parse_timeframe_minutes

CACHE_DIR = Path(DATA_CACHE_DIR) / "parquet_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Kept for reference: validate_ohlcv's default staleness threshold. Cache freshness is now decided by
# whether a newer bar has closed (expected_last_closed_bar_start), not by this age.
CACHE_STALENESS_THRESHOLD_MINUTES = 30

# After asking Dhan for new bars, don't ask again for the same instrument for this long: a just-closed
# bar may not be published yet, and every runner/tab reaching here would otherwise re-request it.
MIN_REFETCH_INTERVAL_SEC = 20.0
# A cache older than this is rebuilt in full (a long tail request would fall back to daily candles).
MAX_TAIL_GAP_DAYS = 30
_last_attempt: dict[tuple[str, str], float] = {}
_last_refresh_error: dict[tuple[str, str], str] = {}  # why the latest refresh of an instrument failed
_attempt_lock = threading.Lock()

# One request-or-read-modify-write at a time per instrument: the background keeper and the pages share the same
# parquet files, so without this two callers fetch the same instrument twice (the second would find a fresh
# cache and make no request) and a reader can meet a half-written file.
_flights: dict[tuple[str, str], threading.RLock] = {}
_flights_guard = threading.Lock()


def _flight_lock(instrument: str, timeframe: str) -> threading.RLock:
    with _flights_guard:
        return _flights.setdefault((instrument, timeframe), threading.RLock())


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Atomic: write beside the target, then rename over it, so a reader never sees a partial file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_cache(path: Path) -> pd.DataFrame | None:
    """The cached candles, or None if the file is unreadable (then it is simply fetched again)."""
    try:
        cached = pd.read_parquet(path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"])
        return cached
    except Exception as e:
        log_event("data_manager", f"Cache file {path.name} is unreadable, fetching it again: {e}", level="WARNING")
        return None


class DataUnavailableError(RuntimeError):
    """No real market data source (CSV or Dhan) could supply the requested data."""


class DataManager:
    def __init__(self):
        self.csv_adapter = CSVAdapter()
        self.dhan_adapter = DhanAdapter()
        self._last_errors: list[str] = []

    def _cache_path(self, instrument: str, timeframe: str) -> Path:
        return CACHE_DIR / f"{instrument}_{timeframe}.parquet"

    def get_ohlcv(
        self,
        instrument: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
        force_refresh: bool = False,
        prefer_source: str | None = None,
        quiet: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        with _flight_lock(instrument, timeframe):  # a second caller waits, then finds the fresh cache
            return self._get_ohlcv(instrument, timeframe, start, end, force_refresh, prefer_source, quiet)

    def _get_ohlcv(
        self,
        instrument: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
        force_refresh: bool = False,
        prefer_source: str | None = None,
        quiet: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Return (dataframe, metadata_dict). metadata includes source + quality report.

        `quiet=True` (used by the background data keeper) skips the per-call metadata row and INFO log when the
        data is OK, so a refresh every minute does not flood the database and the log."""
        self._quiet = quiet
        cache_path = self._cache_path(instrument, timeframe)

        cached = None if force_refresh or not cache_path.exists() else _read_cache(cache_path)
        if cached is not None:
            in_range = cached[(cached["timestamp"] >= start) & (cached["timestamp"] <= end)]
            if len(in_range) > 0:
                tf_minutes = _parse_timeframe_minutes(timeframe)
                last_cached = in_range["timestamp"].max().to_pydatetime().replace(tzinfo=None)
                expected = expected_last_closed_bar_start(end, profile_for(instrument), tf_minutes)
                if last_cached >= expected:
                    return self._finalize(in_range, instrument, timeframe, "cache")  # up to date: no request

                # A newer bar should exist. Fetch only the missing tail (at most once per
                # MIN_REFETCH_INTERVAL_SEC per instrument) and merge it; fall back to a full fetch only
                # if the cache is too old to patch.
                if (end - last_cached).days <= MAX_TAIL_GAP_DAYS:
                    if not self._may_request(instrument, timeframe):
                        return self._finalize(in_range, instrument, timeframe, "cache",
                                              refresh_error=_last_refresh_error.get((instrument, timeframe)))
                    merged = self._refresh_tail(instrument, timeframe, cached, last_cached, tf_minutes, end, prefer_source)
                    if merged is not None:
                        _last_refresh_error.pop((instrument, timeframe), None)
                        _write_cache(merged, cache_path)
                        in_range = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)]
                        return self._finalize(in_range, instrument, timeframe, "cache+tail")
                    # Refresh failed (rate limit, network): serve what we have. The quality gate flags it
                    # stale/DEGRADED if it is old enough, which forces NO TRADE - never a silent guess.
                    reason = "; ".join(self._last_errors) or "no response from the data source"
                    _last_refresh_error[(instrument, timeframe)] = reason
                    log_event("data_manager", f"Tail refresh failed for {instrument} {timeframe}; using cached bars ({reason})",
                              level="WARNING")
                    return self._finalize(in_range, instrument, timeframe, "cache", refresh_error=reason)

        with _attempt_lock:
            _last_attempt[(instrument, timeframe)] = time.monotonic()

        source, df = self._fetch(instrument, timeframe, start, end, prefer_source)
        if len(df) == 0:
            reasons = "; ".join(self._last_errors) or "no CSV data found and Dhan is not configured"
            log_event("data_manager", f"No real data for {instrument} {timeframe}: {reasons}", level="ERROR")
            raise DataUnavailableError(
                f"No real market data for {instrument} {timeframe}: {reasons}. "
                "Set your Dhan credentials in the sidebar or add CSV files to data_cache/csv."
            )
        # Keep older history: a rebuild asks for only the requested window, and replacing the file with it would
        # silently discard a long backfill (e.g. after the token lapsed for over MAX_TAIL_GAP_DAYS).
        existing = _read_cache(cache_path) if cache_path.exists() else None
        if existing is not None and len(existing):
            to_write = pd.concat([existing, df.assign(timestamp=pd.to_datetime(df["timestamp"]))], ignore_index=True)
            to_write = to_write.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        else:
            to_write = df
        _write_cache(to_write, cache_path)
        return self._finalize(df, instrument, timeframe, source)

    def _may_request(self, instrument: str, timeframe: str) -> bool:
        """Claim the right to ask the data source for this instrument now (spacing across all callers)."""
        key = (instrument, timeframe)
        with _attempt_lock:
            now = time.monotonic()
            if now - _last_attempt.get(key, float("-inf")) < MIN_REFETCH_INTERVAL_SEC:
                return False
            _last_attempt[key] = now
            return True

    def _refresh_tail(self, instrument, timeframe, cached, last_cached, tf_minutes, end, prefer_source):
        """Fetch bars from just before the last cached one and merge. None if the fetch failed."""
        tail_start = last_cached - dt.timedelta(minutes=2 * tf_minutes)
        source, tail = self._fetch(instrument, timeframe, tail_start, end, prefer_source)
        if source == "none":
            return None  # error (already recorded in _last_errors)
        if len(tail) > 0:
            # Only bars from the requested window may be merged: anything older (e.g. a misaligned
            # clock) would otherwise overwrite good cached bars via drop_duplicates(keep="last").
            tail = tail.assign(timestamp=pd.to_datetime(tail["timestamp"]))
            tail = tail[tail["timestamp"] >= tail_start]
        if len(tail) == 0:
            return cached  # source answered but the new bar is not published yet
        merged = pd.concat([cached, tail], ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        return merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

    def _fetch(self, instrument, timeframe, start, end, prefer_source) -> tuple[str, pd.DataFrame]:
        order = [prefer_source] if prefer_source else []
        order += ["csv", "dhan"]
        self._last_errors = []

        for source in order:
            if source == "csv" and self.csv_adapter.is_available():
                df = self.csv_adapter.get_ohlcv(instrument, timeframe, start, end)
                if len(df) > 0:
                    return "csv", df
            elif source == "dhan":
                if not self.dhan_adapter.is_available():
                    self._last_errors.append("Dhan credentials not set")
                    continue
                try:
                    df = self.dhan_adapter.get_ohlcv(instrument, timeframe, start, end)
                    if len(df) > 0:
                        return "dhan", df
                    self._last_errors.append("Dhan returned no rows")
                except Exception as e:
                    log_event("data_manager", f"Dhan fetch failed: {e}", level="WARNING")
                    self._last_errors.append(str(e))
        return "none", pd.DataFrame()

    def _finalize(self, df: pd.DataFrame, instrument: str, timeframe: str, source: str,
                  refresh_error: str | None = None) -> tuple[pd.DataFrame, dict]:
        tf_minutes = _parse_timeframe_minutes(timeframe)
        # Measure misalignment BEFORE cleaning: clean_ohlcv drops out-of-session bars, which
        # would otherwise hide a wholesale timezone error.
        profile = profile_for(instrument)
        misaligned = 0
        if tf_minutes < 1440 and len(df):
            misaligned = int(out_of_session_mask(df["timestamp"], profile).sum())
        df = clean_ohlcv(df, tf_minutes, profile)
        report = validate_ohlcv(df, tf_minutes, profile=profile)
        if misaligned / max(len(df) + misaligned, 1) > MAX_OUT_OF_SESSION_FRACTION:
            report.status = QUALITY_FAIL
            report.issues.append(f"{misaligned} bars outside the {profile.label} session before cleaning (timezone/data error)")

        if refresh_error:
            # Say WHY the data is old: the quality gate only knows it is stale, not that the refresh failed.
            report.issues.append(f"latest refresh failed: {refresh_error}")

        metadata = {
            "instrument": instrument,
            "timeframe": timeframe,
            "source": source,
            "start_ts": df["timestamp"].min() if len(df) else None,
            "end_ts": df["timestamp"].max() if len(df) else None,
            "n_rows": len(df),
            "quality_status": report.status,
            "quality_report": report.as_dict(),
        }

        if getattr(self, "_quiet", False) and report.status == "OK":
            return df, metadata

        try:
            from quant_intelligence.database.db import get_session
            from quant_intelligence.database.models import MarketDataMetadata

            with get_session() as session:
                session.add(MarketDataMetadata(**metadata))
        except Exception:
            pass

        log_event(
            "data_manager",
            f"Fetched {len(df)} rows for {instrument} {timeframe} from {source}: {report.status}",
            level="WARNING" if report.status != "OK" else "INFO",
            **({"issues": report.issues} if report.status != "OK" else {}),
        )

        if report.status == QUALITY_FAIL:
            log_event("data_manager", "Data quality FAIL - downstream should force NO TRADE", level="ERROR", issues=report.issues)

        return df, metadata
