"""
Historical option prices from Dhan's "expired options data" endpoint.

    POST /v2/charts/rollingoption   (https://dhanhq.co/docs/v2/expired-options-data/)

The endpoint is ROLLING: you ask for "the ATM call of the nearest weekly expiry" and get, for every bar, that
contract's open/high/low/close/volume/iv/oi plus its strike and the underlying's spot. The strike changes as spot moves.
Offsets run ATM, ATM+1 ... ATM-1 ... (up to +/-10 for index options near expiry, +/-3 otherwise). Requests are limited
to 30 days and Dhan states about 5 years of history. Checked live on 2026-10-21: toDate was effectively INCLUSIVE
(the docs say it is not), the epoch is UTC (+5:30 gives IST, first bar 09:15), strikes roll with spot, and the server is
slow: about 20 s for a 2-day window, with 504s on repeat. So windows are small, the timeout long and gateway errors retried.
Windows overlap by one day on purpose, which is correct under either reading of toDate; duplicates merge by timestamp.

What this gives, and what it does not:
* real option prices at bar resolution, for the offsets we ask for. No bid/ask, so spread stays an assumption.
* a window of strikes around spot, not the whole chain. Anything that needs whole-chain totals (a true PCR) cannot be
  rebuilt from it; options/historical_chain.py labels such snapshots partial.
* the response schema below follows Dhan's documentation. It has not been checked against a live response from this
  account yet (the token expired); `parse_rolling_response` therefore validates shape strictly and refuses to guess.

Files: <OPTION_HISTORY_DIR>/<SYMBOL>/<WEEK|MONTH><code>_<CALL|PUT>_<offset>.parquet, merged by timestamp, atomic writes.
Resumable and idempotent: only the missing ends of the requested range are fetched.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event

ENDPOINT = "/charts/rollingoption"
MAX_DAYS_PER_REQUEST = 30      # Dhan's documented limit per request
DEFAULT_WINDOW_DAYS = 5        # what we ask for: this endpoint is slow (about 20 s for 2 days on 2026-10-21) and returns 504 on big windows
REQUEST_TIMEOUT_SEC = 120
GATEWAY_RETRIES = 3
RETRY_BACKOFF_SEC = (5.0, 20.0, 45.0)
REQUIRED_DATA = ["open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot"]
COLUMNS = ["timestamp", *REQUIRED_DATA]
SIDES = {"CALL": "ce", "PUT": "pe"}
EMPTY_WINDOWS_TO_STOP = 2
# Dhan sends a couple of prints after the close (15:30, 15:35) every day, about 2.6% of 5-minute bars, so the clock-basis check
# cannot demand 98% as the candle adapter does. The wrong basis moves most bars out of the session, never just a few.
MIN_IN_SESSION_FRACTION = 0.90


@dataclass(frozen=True)
class SeriesKey:
    symbol: str
    expiry_flag: str     # WEEK | MONTH
    expiry_code: int     # 1 = nearest expiry of that flag
    option_type: str     # CALL | PUT
    offset: int          # 0 = ATM, +1 = one strike above spot, -1 = one below

    @property
    def strike_label(self) -> str:
        return "ATM" if self.offset == 0 else f"ATM{self.offset:+d}"

    @property
    def filename(self) -> str:
        return f"{self.expiry_flag}{self.expiry_code}_{self.option_type}_{self.offset:+d}.parquet"


def history_dir(symbol: str, root: str | Path | None = None) -> Path:
    return Path(root or SETTINGS.option_history_dir) / symbol.strip().upper()


def series_path(key: SeriesKey, root: str | Path | None = None) -> Path:
    return history_dir(key.symbol, root) / key.filename


# ---- parsing ------------------------------------------------------------------------------------------------------

class RollingResponseError(RuntimeError):
    pass


def _to_ist(epoch_seconds: list, profile=None) -> pd.Series:
    """Naive IST timestamps. Dhan's epoch basis is verified from the data (bars must sit in the session), as the candle
    adapter does; a shifted clock would silently corrupt everything keyed on time."""
    from quant_intelligence.data_adapters.dhan_adapter import _in_session_fraction
    from quant_intelligence.utils.market_profile import NSE

    profile = profile or NSE
    raw = pd.to_datetime(pd.Series(epoch_seconds, dtype="float64"), unit="s")
    for shift in (pd.Timedelta(hours=5, minutes=30), pd.Timedelta(0)):
        shifted = raw + shift
        if _in_session_fraction(shifted, profile) >= MIN_IN_SESSION_FRACTION:
            return shifted
    raise RollingResponseError("option timestamps do not fall inside the market session under any known timezone basis")


def NSE_PROFILE():
    from quant_intelligence.utils.market_profile import NSE

    return NSE


def parse_rolling_response(body: dict, option_type: str, profile=None) -> pd.DataFrame:
    """One side (CALL -> 'ce', PUT -> 'pe') of a rolling-options response as a tidy frame (COLUMNS). Empty if Dhan
    returned no bars for the window. Raises RollingResponseError on a shape it does not recognise."""
    data = body.get("data", body) if isinstance(body, dict) else None
    if not isinstance(data, dict):
        raise RollingResponseError(f"unexpected rolling-options response: {str(body)[:200]}")
    side = data.get(SIDES[option_type])
    if not side:
        return pd.DataFrame(columns=COLUMNS)
    if "timestamp" not in side:
        raise RollingResponseError(f"rolling-options response has no timestamp array (keys: {sorted(side)})")
    n = len(side["timestamp"])
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)
    cols: dict[str, object] = {"timestamp": _to_ist(side["timestamp"], profile)}
    for name in REQUIRED_DATA:
        arr = side.get(name)
        if arr is None:
            cols[name] = np.full(n, np.nan)
            continue
        if len(arr) != n:
            raise RollingResponseError(f"'{name}' has {len(arr)} values for {n} timestamps")
        cols[name] = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    df = pd.DataFrame(cols).dropna(subset=["timestamp"])
    profile = profile or NSE_PROFILE()
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(minutes >= profile.open_min) & (minutes < profile.close_min)]     # Dhan also sends 15:30 and 15:35 prints
    return df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)[COLUMNS]


# ---- fetching -----------------------------------------------------------------------------------------------------

def request_body(security_id, exchange_segment: str, instrument: str, key: SeriesKey, interval: str,
                 start: dt.date, end_exclusive: dt.date) -> dict:
    return {"exchangeSegment": exchange_segment, "interval": str(interval), "securityId": int(security_id),
            "instrument": instrument, "expiryCode": key.expiry_code, "expiryFlag": key.expiry_flag,
            "strike": key.strike_label, "drvOptionType": key.option_type, "requiredData": REQUIRED_DATA,
            "fromDate": start.isoformat(), "toDate": end_exclusive.isoformat()}


def windows(start: dt.date, end_exclusive: dt.date, days: int = DEFAULT_WINDOW_DAYS) -> list[tuple[dt.date, dt.date]]:
    """Contiguous [from, to) windows of at most `days` days covering [start, end_exclusive)."""
    out, cur = [], start
    while cur < end_exclusive:
        nxt = min(cur + dt.timedelta(days=days), end_exclusive)
        out.append((cur, nxt))
        cur = nxt
    return out


def _retryable(e: Exception) -> bool:
    import requests

    status = getattr(e, "status_code", None)
    return isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) or status in (500, 502, 503, 504)


def post_with_retry(client, body: dict, sleep: Callable[[float], None] | None = None) -> dict:
    """One rolling-options request with a long timeout, retried with backoff on timeouts and gateway errors (the endpoint is
    slow and sometimes answers 504). Auth, rate-limit and validation errors are not retried: they will repeat."""
    import time

    sleep = sleep or time.sleep
    for attempt in range(GATEWAY_RETRIES + 1):
        try:
            return client._post(ENDPOINT, body, category="data", timeout=REQUEST_TIMEOUT_SEC)
        except Exception as e:
            if attempt >= GATEWAY_RETRIES or not _retryable(e):
                raise
            sleep(RETRY_BACKOFF_SEC[min(attempt, len(RETRY_BACKOFF_SEC) - 1)])
    raise AssertionError("unreachable")


@dataclass
class SeriesResult:
    key: SeriesKey
    windows_fetched: int = 0
    rows_added: int = 0
    rows_total: int = 0
    earliest: dt.datetime | None = None
    latest: dt.datetime | None = None
    errors: list[str] = field(default_factory=list)


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        log_event("options_history", f"{path.name} is unreadable and will be fetched again: {e}", level="WARNING")
        return None


def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or len(old) == 0:
        return new
    return pd.concat([old, new], ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def fetch_series(
    client,
    key: SeriesKey,
    security_id,
    start: dt.date,
    end_exclusive: dt.date,
    exchange_segment: str = "NSE_FNO",
    instrument: str = "OPTIDX",
    interval: str = "5",
    root: str | Path | None = None,
    profile=None,
    post: Callable | None = None,
) -> SeriesResult:
    """Bring `key`'s stored history up to cover [start, end_exclusive), newest window first, requesting only windows that
    contain a day not yet done. A failed window is recorded and the run stops for this series (the next run resumes)."""
    from quant_intelligence.data.data_manager import _write_cache

    path = series_path(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = _read(path)
    result = SeriesResult(key)
    today = dt.date.today()
    ledger_file = path.with_name(path.stem + ".checked.json")
    ledger = _load_ledger(ledger_file)
    # A day counts as done when it has bars or a completed request covered it. Today never does (its data is partial).
    # Looking at the actual days, not just the first and last stored date, is what lets a stopped run leave holes and a
    # later run find and fill them.
    done = ledger | ({t.date() for t in stored["timestamp"]} if stored is not None and len(stored) else set())
    done = {d for d in done if d < today}
    todo = [w for w in windows(start, end_exclusive) if any(d not in done for d in _weekdays(*w))]
    todo.sort(reverse=True)                      # newest first: a stopped run leaves the oldest part undone, not a hole
    post = post or (lambda body: post_with_retry(client, body))
    merged, empties = stored, []
    for lo, hi in todo:
        try:
            body = post(request_body(security_id, exchange_segment, instrument, key, interval, lo, hi))
            df = parse_rolling_response(body, key.option_type, profile)
        except Exception as e:
            result.errors.append(f"{lo}..{hi}: {e}")
            break
        result.windows_fetched += 1
        if len(df) == 0:
            empties.append((lo, hi))
            if len(empties) >= EMPTY_WINDOWS_TO_STOP:      # the source has nothing this far back: everything older is settled too
                ledger |= set(_weekdays(start, empties[0][1]))
                _save_ledger(ledger_file, ledger, today)
                break
            continue
        empties = []
        merged = _merge(merged, df)
        ledger |= set(_weekdays(lo, hi))
        _write_cache(merged, path)               # checkpoint after every window: a stopped run keeps what it fetched
        _save_ledger(ledger_file, ledger, today)
    before = 0 if stored is None else len(stored)
    if merged is not None and len(merged):
        result.rows_total = len(merged)
        result.rows_added = len(merged) - before
        result.earliest, result.latest = merged["timestamp"].min().to_pydatetime(), merged["timestamp"].max().to_pydatetime()
    return result


def _weekdays(lo: dt.date, hi: dt.date) -> list[dt.date]:
    """Weekdays in [lo, hi). Holidays are not removed: a window request settles them (it returns no bars for them)."""
    return [lo + dt.timedelta(days=i) for i in range((hi - lo).days) if (lo + dt.timedelta(days=i)).weekday() < 5]


def _load_ledger(path: Path) -> set[dt.date]:
    import json

    try:
        return {dt.date.fromisoformat(x) for x in json.loads(path.read_text())} if path.exists() else set()
    except Exception:
        return set()


def _save_ledger(path: Path, days: set[dt.date], today: dt.date) -> None:
    import json

    path.write_text(json.dumps(sorted(d.isoformat() for d in days if d < today)))


@dataclass
class OptionsBackfillResult:
    symbol: str
    series: list[SeriesResult] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [f"{r.key.filename}: {e}" for r in self.series for e in r.errors]

    @property
    def rows_added(self) -> int:
        return sum(r.rows_added for r in self.series)

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "rows_added": self.rows_added, "errors": self.errors,
                "series": [{"file": r.key.filename, "windows": r.windows_fetched, "rows": r.rows_total,
                            "earliest": str(r.earliest) if r.earliest else None, "latest": str(r.latest) if r.latest else None}
                           for r in self.series]}


def backfill_options(
    client,
    symbol: str,
    start: dt.date,
    end_exclusive: dt.date,
    max_offset: int = 3,
    expiry_flag: str = "WEEK",
    expiry_code: int = 1,
    interval: str = "5",
    root: str | Path | None = None,
    info: dict | None = None,
    post: Callable | None = None,
) -> OptionsBackfillResult:
    """Fetch CALL and PUT series for offsets -max_offset..+max_offset. The underlying's id and segment come from the
    option selector's registry (`info` overrides it, for tests)."""
    from quant_intelligence.options.option_selector import get_underlying_info, option_segment_for

    symbol = symbol.strip().upper()
    info = info or get_underlying_info(symbol)
    instrument = "OPTIDX" if symbol in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"} else "OPTSTK"
    out = OptionsBackfillResult(symbol)
    for option_type in SIDES:
        for offset in range(-max_offset, max_offset + 1):
            key = SeriesKey(symbol, expiry_flag, expiry_code, option_type, offset)
            res = fetch_series(client, key, info["security_id"], start, end_exclusive, option_segment_for(symbol), instrument,
                               interval, root, post=post)
            out.series.append(res)
            if res.errors:                       # an auth or rate-limit error will repeat for every series: stop and report
                log_event("options_history", f"{symbol}: stopped at {key.filename}: {res.errors[0]}", level="WARNING")
                return out
    log_event("options_history", f"{symbol}: {out.rows_added:+d} option rows", level="INFO")
    return out


# ---- reading ------------------------------------------------------------------------------------------------------

class OptionsHistory:
    """All stored series of one symbol (one expiry flag/code), indexed by (option type, offset) and timestamp."""

    def __init__(self, symbol: str, frames: dict[tuple[str, int], pd.DataFrame], expiry_flag: str = "WEEK", expiry_code: int = 1):
        self.symbol, self.expiry_flag, self.expiry_code = symbol, expiry_flag, expiry_code
        self.frames = {k: v.set_index("timestamp") for k, v in frames.items() if len(v)}

    @classmethod
    def load(cls, symbol: str, expiry_flag: str = "WEEK", expiry_code: int = 1, root: str | Path | None = None) -> "OptionsHistory | None":
        frames: dict[tuple[str, int], pd.DataFrame] = {}
        for option_type in SIDES:
            for path in history_dir(symbol, root).glob(f"{expiry_flag}{expiry_code}_{option_type}_*.parquet"):
                df = _read(path)
                if df is not None and len(df):
                    frames[(option_type, int(path.stem.rsplit("_", 1)[1]))] = df
        return cls(symbol.strip().upper(), frames, expiry_flag, expiry_code) if frames else None

    @property
    def offsets(self) -> list[int]:
        return sorted({o for _, o in self.frames})

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        idx = [f.index for f in self.frames.values()]
        return min(i.min() for i in idx), max(i.max() for i in idx)

    def row(self, option_type: str, offset: int, ts) -> pd.Series | None:
        f = self.frames.get((option_type, offset))
        if f is None:
            return None
        try:
            r = f.loc[pd.Timestamp(ts)]
        except KeyError:
            return None
        return r.iloc[-1] if isinstance(r, pd.DataFrame) else r

    def fingerprint(self) -> str:
        """sha256 over the stored series (same canonicalisation as candle data), so a run can say which option data it used."""
        import hashlib

        h = hashlib.sha256()
        for (t, o) in sorted(self.frames):
            f = self.frames[(t, o)].reset_index()
            h.update(f"{t}{o}".encode())
            f = f.assign(**{c: f[c].round(6) for c in f.columns if c != "timestamp"})
            h.update(pd.util.hash_pandas_object(f, index=False).to_numpy().tobytes())
        return h.hexdigest()
