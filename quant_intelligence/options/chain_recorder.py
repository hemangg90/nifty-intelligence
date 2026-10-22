"""
Records the live option chain, every strike, as it is fetched (behind CHAIN_RECORDER, default off).

Dhan's historical endpoint only returns a window of strikes around spot, so a true whole-chain history (the PCR the chain
strategies' thresholds were written for) can only be built by saving the live chain as we go. This does that: each snapshot
the app already fetches is appended to <CHAIN_RECORD_DIR>/<SYMBOL>/<YYYY-MM-DD>.parquet, at most one per underlying per
CHAIN_RECORD_INTERVAL_SEC. No extra Dhan requests are made, so it cannot add to the option-chain rate limit.

Recording never raises into trading: a failed write is logged and dropped.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.utils.logging_utils import log_event

_lock = threading.Lock()
_last_written: dict[str, float] = {}

_ROW_FIELDS = ["ce_ltp", "ce_oi", "ce_oi_change", "ce_volume", "ce_iv", "ce_delta",
               "pe_ltp", "pe_oi", "pe_oi_change", "pe_volume", "pe_iv", "pe_delta"]


def snapshot_frame(snapshot: ChainSnapshot, taken_at: dt.datetime) -> pd.DataFrame:
    rows = [{"timestamp": taken_at, "underlying": snapshot.underlying, "expiry": snapshot.expiry,
             "spot": snapshot.spot_price, "strike": s.strike, **{f: getattr(s, f) for f in _ROW_FIELDS}} for s in snapshot.strikes]
    return pd.DataFrame(rows)


def record_snapshot(snapshot: ChainSnapshot, now: dt.datetime | None = None, root: str | Path | None = None,
                    interval_sec: float | None = None, force: bool = False) -> bool:
    """Append `snapshot` to today's file for its underlying. Returns True when something was written."""
    if not (SETTINGS.chain_recorder or force) or snapshot is None or not snapshot.strikes or snapshot.partial:
        return False
    interval = SETTINGS.chain_record_interval_sec if interval_sec is None else interval_sec
    key = snapshot.underlying.upper()
    try:
        with _lock:
            mono = time.monotonic()
            if not force and key in _last_written and mono - _last_written[key] < interval:
                return False
            from quant_intelligence.data.data_manager import _read_cache, _write_cache
            from quant_intelligence.utils.timeutil import now_ist

            taken_at = (now or now_ist()).replace(tzinfo=None)
            folder = Path(root or SETTINGS.chain_record_dir) / key
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{taken_at:%Y-%m-%d}.parquet"
            fresh = snapshot_frame(snapshot, taken_at)
            old = _read_cache(path) if path.exists() else None
            _write_cache(fresh if old is None else pd.concat([old, fresh], ignore_index=True), path)
            _last_written[key] = mono
        return True
    except Exception as e:
        log_event("chain_recorder", f"Could not record the {snapshot.underlying} chain: {e}", level="WARNING")
        return False


def load_recorded(underlying: str, date: dt.date, root: str | Path | None = None) -> list[ChainSnapshot]:
    """Every snapshot recorded for `underlying` on `date`, oldest first, as complete (not partial) ChainSnapshots."""
    path = Path(root or SETTINGS.chain_record_dir) / underlying.upper() / f"{date:%Y-%m-%d}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    out = []
    for ts, g in df.groupby("timestamp", sort=True):
        strikes = [StrikeRow(strike=float(r.strike), ce_security_id=None, pe_security_id=None,
                             **{f: (None if getattr(r, f) != getattr(r, f) else getattr(r, f)) for f in _ROW_FIELDS})
                   for r in g.itertuples(index=False)]
        first = g.iloc[0]
        out.append(ChainSnapshot(str(first["underlying"]), str(first["expiry"]), float(first["spot"]), strikes, source="recorded"))
    return out


def reset_throttle() -> None:
    with _lock:
        _last_written.clear()
