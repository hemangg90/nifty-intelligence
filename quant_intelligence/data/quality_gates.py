"""
Research-grade data-quality gate and dataset fingerprint (pipeline stage 1).

`validate_ohlcv` (data/quality.py) answers "is this safe to trade on right now?". This module answers
the research question: "is this history complete, correctly aligned and fit to backtest on?" It adds
what that check does not cover: session-calendar completeness per trading day, non-monotonic or
duplicate timestamps, outlier bar returns and overnight gaps, runs of zero-volume bars, and a
deterministic fingerprint so every backtest can say exactly which data it ran on.

Failure condition: status FAIL means the instrument is INSUFFICIENT DATA for research; nothing downstream
may rank a strategy on it. A DEGRADED history is usable but the issues are carried into the run manifest.
Staleness is deliberately not judged here: a backtest on last month's data is not wrong, only old.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.data.quality import (
    QUALITY_DEGRADED,
    QUALITY_FAIL,
    QUALITY_OK,
    _ohlc_violation_mask,
    out_of_session_mask,
)
from quant_intelligence.utils.market_profile import NSE, MarketProfile

# A session with fewer than this fraction of its expected bars is incomplete (a few missing bars degrade
# the history; a mostly-empty day is dropped from research use by flagging it).
MIN_SESSION_COMPLETENESS = 0.90
# Fraction of sessions allowed to be incomplete before the whole history is DEGRADED / FAIL.
DEGRADED_BAD_SESSION_FRACTION = 0.02
FAIL_BAD_SESSION_FRACTION = 0.10
# Bar-to-bar close move above this (in robust sigmas) is flagged as an outlier print.
OUTLIER_SIGMA = 12.0
# ...and larger than this in absolute terms. On an illiquid contract most bars are flat, the robust sigma collapses
# towards zero, and every ordinary tick would otherwise look like a 12-sigma event.
OUTLIER_MIN_ABS_LOG_RETURN = 0.03
MAX_OUTLIER_FRACTION = 0.001
# This many consecutive zero-volume bars inside a session means a dead feed, not a quiet market.
ZERO_VOLUME_RUN = 12
MIN_ROWS = 50


@dataclass
class DataQualityReport:
    status: str
    issues: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_sessions: int = 0
    first_session: dt.date | None = None
    last_session: dt.date | None = None
    n_duplicates: int = 0
    n_non_monotonic: int = 0
    n_ohlc_violations: int = 0
    n_zero_or_negative: int = 0
    n_out_of_session: int = 0
    n_incomplete_sessions: int = 0
    n_missing_sessions: int = 0       # trading days inside the range with no bars at all
    n_unexpected_sessions: int = 0    # bars on a weekend / exchange holiday (special sessions: informational only)
    n_outlier_bars: int = 0
    n_zero_volume_runs: int = 0
    incomplete_sessions: list[str] = field(default_factory=list)
    missing_sessions: list[str] = field(default_factory=list)
    special_sessions: list[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def usable_for_research(self) -> bool:
        return self.status != QUALITY_FAIL

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["first_session"] = str(self.first_session) if self.first_session else None
        d["last_session"] = str(self.last_session) if self.last_session else None
        return d


def dataset_fingerprint(df: pd.DataFrame, source: str = "", timeframe: str = "", instrument: str = "") -> str:
    """sha256 over the canonicalised rows (sorted by timestamp, fixed columns and float formatting) plus
    where it came from. Identical data always hashes identically; any changed bar or date range changes it."""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if len(df) == 0:
        canon = pd.DataFrame(columns=cols)
    else:
        canon = df[cols].copy()
        canon["timestamp"] = pd.to_datetime(canon["timestamp"])
        canon = canon.sort_values("timestamp").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"{instrument}|{timeframe}|{source}|{len(canon)}".encode())
    if len(canon):
        h.update(canon["timestamp"].astype("int64").to_numpy().tobytes())
        for c in cols[1:]:
            h.update(np.round(canon[c].to_numpy(dtype="float64"), 6).tobytes())
    return h.hexdigest()


def _expected_bars_per_session(profile: MarketProfile, tf_minutes: int) -> int:
    return profile.session_minutes // tf_minutes


def check_history(
    df: pd.DataFrame,
    timeframe_minutes: int,
    profile: MarketProfile = NSE,
    source: str = "",
    instrument: str = "",
    timeframe: str = "",
) -> DataQualityReport:
    """Research-grade audit of a candle history. Never mutates `df`."""
    issues: list[str] = []
    n_rows = len(df)
    if n_rows < MIN_ROWS:
        return DataQualityReport(
            status=QUALITY_FAIL, issues=[f"only {n_rows} rows (need at least {MIN_ROWS})"], n_rows=n_rows,
            fingerprint=dataset_fingerprint(df, source, timeframe, instrument),
        )

    fingerprint = dataset_fingerprint(df, source, timeframe, instrument)
    ts = pd.to_datetime(df["timestamp"])

    n_duplicates = int(ts.duplicated().sum())
    n_non_monotonic = int((ts.diff().dt.total_seconds() < 0).sum())
    if n_duplicates:
        issues.append(f"{n_duplicates} duplicate timestamps")
    if n_non_monotonic:
        issues.append(f"{n_non_monotonic} timestamps go backwards (history is not in chronological order)")

    # Work on a sorted, de-duplicated copy for everything below.
    d = df.assign(timestamp=ts).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    ts = d["timestamp"]

    n_zero_or_negative = int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    n_ohlc_violations = int(_ohlc_violation_mask(d).sum())
    if n_zero_or_negative:
        issues.append(f"{n_zero_or_negative} rows with zero/negative price")
    if n_ohlc_violations:
        issues.append(f"{n_ohlc_violations} rows with abnormal OHLC relationships")

    n_out_of_session = 0
    if timeframe_minutes < 1440:
        n_out_of_session = int(out_of_session_mask(ts, profile).sum())
        if n_out_of_session:
            issues.append(f"{n_out_of_session} bars outside the {profile.label} session (timezone misalignment?)")

    # ---- session-calendar completeness -------------------------------------------------------------
    n_incomplete = n_missing = n_unexpected = 0
    incomplete_days: list[str] = []
    missing_days: list[str] = []
    days = ts.dt.date
    per_day = d.groupby(days).size()
    first_day, last_day = per_day.index.min(), per_day.index.max()

    special_days: list[str] = []
    if timeframe_minutes < 1440:
        # Completeness against a fixed bar count is only meaningful for NSE: MCX sessions end at 17:00, 23:30 or 23:55
        # depending on the contract and US daylight saving, and illiquid bars may simply not be published.
        check_completeness = profile.name == "NSE"
        expected = _expected_bars_per_session(profile, timeframe_minutes)
        for day, n in per_day.items():
            if not is_trading_day(day, profile.name):
                n_unexpected += 1  # Budget-day Saturdays, Muhurat sessions: real, so kept, but not a defect
                special_days.append(f"{day} ({n} bars)")
            elif check_completeness and n < expected * MIN_SESSION_COMPLETENESS:
                n_incomplete += 1
                incomplete_days.append(f"{day} ({n}/{expected})")
        cursor = first_day
        while cursor <= last_day:
            if is_trading_day(cursor, profile.name) and cursor not in per_day.index:
                n_missing += 1
                missing_days.append(str(cursor))
            cursor += dt.timedelta(days=1)

    n_sessions = int(len(per_day))
    expected_sessions = n_sessions - n_unexpected + n_missing
    bad_fraction = (n_incomplete + n_missing) / max(expected_sessions, 1)
    if n_unexpected:
        issues.append(f"note: {n_unexpected} special session(s) on a weekend/exchange holiday: {', '.join(special_days[:5])}")
    if n_incomplete:
        issues.append(f"{n_incomplete} incomplete session(s) (<{MIN_SESSION_COMPLETENESS:.0%} of expected bars)")
    if n_missing:
        issues.append(f"{n_missing} trading day(s) in range with no bars")

    # ---- outlier prints and dead-feed volume runs --------------------------------------------------
    n_outliers = 0
    close = d["close"].astype(float)
    ret = np.log(close).diff()
    same_session = days == days.shift(1)
    intraday_ret = ret.where(same_session)
    if intraday_ret.notna().sum() > 30:
        med = intraday_ret.median()
        mad = (intraday_ret - med).abs().median()
        sigma = 1.4826 * mad if mad > 0 else intraday_ret.std()
        if sigma and sigma > 0:
            threshold = max(OUTLIER_SIGMA * sigma, OUTLIER_MIN_ABS_LOG_RETURN)
            n_outliers = int(((intraday_ret - med).abs() > threshold).sum())
    if n_outliers / n_rows > MAX_OUTLIER_FRACTION:
        issues.append(f"{n_outliers} bar-to-bar moves beyond {OUTLIER_SIGMA:.0f} robust sigmas and {OUTLIER_MIN_ABS_LOG_RETURN:.0%} (bad prints, or real shocks to check)")

    n_zero_vol_runs = 0
    vol = d["volume"].fillna(0)
    if vol.sum() > 0:  # an index feed with no volume at all is not a defect; it is handled by the volume-skip logic
        zero = (vol <= 0).astype(int)
        group = (zero != zero.shift()).cumsum()
        run_len = zero.groupby([days, group]).transform("sum")
        starts = (zero == 1) & (zero.shift(fill_value=0) == 0)
        n_zero_vol_runs = int((starts & (run_len >= ZERO_VOLUME_RUN)).sum())
        if n_zero_vol_runs:
            issues.append(f"{n_zero_vol_runs} run(s) of >= {ZERO_VOLUME_RUN} consecutive zero-volume bars")

    # ---- verdict ----------------------------------------------------------------------------------
    hard_fail = (
        n_zero_or_negative > 0
        or n_ohlc_violations > 0
        or n_out_of_session / n_rows > 0.02
        or bad_fraction > FAIL_BAD_SESSION_FRACTION
        or n_non_monotonic > 0
    )
    degraded = (
        n_duplicates > 0
        or n_out_of_session > 0
        or bad_fraction > DEGRADED_BAD_SESSION_FRACTION
        or n_outliers / n_rows > MAX_OUTLIER_FRACTION
        or n_zero_vol_runs > 0
    )
    status = QUALITY_FAIL if hard_fail else QUALITY_DEGRADED if degraded else QUALITY_OK

    return DataQualityReport(
        status=status,
        issues=issues,
        n_rows=n_rows,
        n_sessions=n_sessions,
        first_session=first_day,
        last_session=last_day,
        n_duplicates=n_duplicates,
        n_non_monotonic=n_non_monotonic,
        n_ohlc_violations=n_ohlc_violations,
        n_zero_or_negative=n_zero_or_negative,
        n_out_of_session=n_out_of_session,
        n_incomplete_sessions=n_incomplete,
        n_missing_sessions=n_missing,
        n_unexpected_sessions=n_unexpected,
        n_outlier_bars=n_outliers,
        n_zero_volume_runs=n_zero_vol_runs,
        incomplete_sessions=incomplete_days[:50],
        missing_sessions=missing_days[:50],
        special_sessions=special_days[:50],
        fingerprint=fingerprint,
    )
