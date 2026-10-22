"""Research-grade data-quality gate, dataset fingerprint and history backfill."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.data.backfill import backfill_instrument
from quant_intelligence.data.quality_gates import check_history, dataset_fingerprint


def _sessions(n_days: int, end: dt.date = dt.date(2024, 6, 28)) -> list[dt.date]:
    days, d = [], end
    while len(days) < n_days:
        if is_trading_day(d, "NSE"):
            days.append(d)
        d -= dt.timedelta(days=1)
    return sorted(days)


def _frame(days: list[dt.date], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 20000.0
    for day in days:
        t = dt.datetime.combine(day, dt.time(9, 15))
        for _ in range(75):
            o = price
            c = o * (1 + rng.normal(0, 0.0004))
            rows.append((t, o, max(o, c) * 1.0002, min(o, c) * 0.9998, c, 1000.0 + rng.integers(0, 500)))
            price, t = c, t + dt.timedelta(minutes=5)
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_clean_history_is_ok():
    rep = check_history(_frame(_sessions(10)), 5, instrument="NIFTY", timeframe="5min")
    assert rep.status == "OK", rep.issues
    assert rep.n_sessions == 10 and rep.usable_for_research
    assert len(rep.fingerprint) == 64


def test_too_few_rows_fails():
    rep = check_history(_frame(_sessions(10)).head(10), 5)
    assert rep.status == "FAIL" and not rep.usable_for_research


def test_duplicates_degrade_and_do_not_crash():
    df = _frame(_sessions(10))
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True).sort_values("timestamp")
    rep = check_history(df, 5)
    assert rep.n_duplicates == 2 and rep.status == "DEGRADED"


def test_ohlc_violation_fails():
    df = _frame(_sessions(10))
    df.loc[100, "high"] = df.loc[100, "low"] - 1
    assert check_history(df, 5).status == "FAIL"


def test_non_monotonic_fails():
    df = _frame(_sessions(10))
    df = pd.concat([df.iloc[100:], df.iloc[:100]], ignore_index=True)
    rep = check_history(df, 5)
    assert rep.n_non_monotonic > 0 and rep.status == "FAIL"


def test_missing_trading_day_is_reported():
    days = _sessions(30)
    gone = days[10]
    df = _frame(days)
    df = df[pd.to_datetime(df["timestamp"]).dt.date != gone]
    rep = check_history(df, 5)
    assert rep.n_missing_sessions == 1 and str(gone) in rep.missing_sessions
    assert rep.status == "DEGRADED"  # 1/30 sessions


def test_half_empty_sessions_fail():
    days = _sessions(10)
    df = _frame(days)
    keep = ~(pd.to_datetime(df["timestamp"]).dt.date.isin(days[:3]) & (df.index % 75 > 20))
    rep = check_history(df[keep], 5)
    assert rep.n_incomplete_sessions == 3 and rep.status == "FAIL"


def test_outlier_print_is_flagged():
    df = _frame(_sessions(20))
    df.loc[500, ["open", "high", "low", "close"]] = df.loc[500, "close"] * 1.15
    rep = check_history(df, 5)
    assert rep.n_outlier_bars >= 1


def test_special_session_on_a_non_trading_day_is_noted_not_penalised():
    days = _sessions(10)
    saturday = dt.date(2024, 6, 22)
    assert not is_trading_day(saturday, "NSE")
    rep = check_history(_frame(sorted(days + [saturday])), 5)
    assert rep.status == "OK" and rep.n_unexpected_sessions == 1 and str(saturday) in rep.special_sessions[0]


def test_flat_illiquid_bars_do_not_make_every_tick_an_outlier():
    from quant_intelligence.utils.market_profile import MCX

    df = _frame(_sessions(15))
    c = df["close"].to_numpy().copy()
    c[:] = np.round(c / 5) * 5  # coarse ticks: most bars flat, a few step by ~0.025%
    df["open"], df["close"] = np.roll(c, 1), c
    df["high"], df["low"] = np.maximum(df["open"], df["close"]) * 1.0001, np.minimum(df["open"], df["close"]) * 0.9999
    assert check_history(df, 5, profile=MCX).n_outlier_bars == 0


def test_mcx_is_not_judged_on_a_fixed_bar_count():
    from quant_intelligence.utils.market_profile import MCX

    days = _sessions(10)
    df = _frame(days)  # 75 bars a day, far fewer than the 179 an MCX 09:00-23:55 session would imply
    df["timestamp"] = df["timestamp"] - pd.Timedelta(minutes=15)  # 09:00 start, inside the MCX session
    rep = check_history(df, 5, profile=MCX)
    assert rep.n_incomplete_sessions == 0


def test_dead_volume_run_is_flagged():
    df = _frame(_sessions(10))
    df.loc[200:220, "volume"] = 0
    assert check_history(df, 5).n_zero_volume_runs == 1


def test_all_zero_volume_index_is_not_a_defect():
    df = _frame(_sessions(10))
    df["volume"] = 0.0
    assert check_history(df, 5).status == "OK"


def test_fingerprint_is_deterministic_and_sensitive():
    df = _frame(_sessions(5))
    assert dataset_fingerprint(df, "cache", "5min", "NIFTY") == dataset_fingerprint(df.sample(frac=1, random_state=3), "cache", "5min", "NIFTY")
    changed = df.copy()
    changed.loc[7, "close"] += 0.5
    assert dataset_fingerprint(changed, "cache", "5min", "NIFTY") != dataset_fingerprint(df, "cache", "5min", "NIFTY")
    assert dataset_fingerprint(df, "csv", "5min", "NIFTY") != dataset_fingerprint(df, "cache", "5min", "NIFTY")


# ---- backfill ------------------------------------------------------------------------------------------

def _fake_source(all_days: list[dt.date]):
    full = _frame(all_days)
    calls: list[tuple] = []

    def fetch(instrument, timeframe, start, end):
        calls.append((start, end))
        ts = pd.to_datetime(full["timestamp"])
        return full[(ts >= start) & (ts <= end)].reset_index(drop=True)

    return fetch, calls, full


def test_backfill_extends_backwards_and_stops_when_source_runs_out(tmp_path):
    days = _sessions(120)
    fetch, calls, full = _fake_source(days)
    now = dt.datetime.combine(days[-1], dt.time(16, 0))
    # Seed the cache with the newest 20 sessions only.
    seed = full[pd.to_datetime(full["timestamp"]).dt.date >= days[-20]]
    seed.to_parquet(tmp_path / "NIFTY_5min.parquet", index=False)

    res = backfill_instrument("NIFTY", "5min", target_start=now - dt.timedelta(days=5 * 365), now=now,
                              fetch=fetch, cache_dir=tmp_path, fill_forward=False)
    out = pd.read_parquet(tmp_path / "NIFTY_5min.parquet")
    assert len(out) == len(full)
    assert res.earliest == dt.datetime.combine(days[0], dt.time(9, 15))
    assert "no data" in res.stopped_reason
    assert not out["timestamp"].duplicated().any()
    assert check_history(out, 5).status == "OK"


def test_backfill_is_idempotent(tmp_path):
    days = _sessions(40)
    fetch, _, full = _fake_source(days)
    now = dt.datetime.combine(days[-1], dt.time(16, 0))
    kw = dict(target_start=now - dt.timedelta(days=100), now=now, fetch=fetch, cache_dir=tmp_path)
    backfill_instrument("NIFTY", "5min", **kw)
    first = pd.read_parquet(tmp_path / "NIFTY_5min.parquet")
    res2 = backfill_instrument("NIFTY", "5min", **kw)
    second = pd.read_parquet(tmp_path / "NIFTY_5min.parquet")
    assert len(first) == len(second) == len(full) and res2.rows_added == 0


def test_backfill_fills_forward_from_a_stale_cache(tmp_path):
    days = _sessions(30)
    fetch, _, full = _fake_source(days)
    now = dt.datetime.combine(days[-1], dt.time(16, 0))
    stale = full[pd.to_datetime(full["timestamp"]).dt.date <= days[-6]]
    stale.to_parquet(tmp_path / "NIFTY_5min.parquet", index=False)
    backfill_instrument("NIFTY", "5min", target_start=stale["timestamp"].min(), now=now, fetch=fetch, cache_dir=tmp_path)
    out = pd.read_parquet(tmp_path / "NIFTY_5min.parquet")
    assert out["timestamp"].max() == full["timestamp"].max()


def test_backfill_reports_errors_and_keeps_what_it_has(tmp_path):
    days = _sessions(30)
    fetch, _, full = _fake_source(days)
    now = dt.datetime.combine(days[-1], dt.time(16, 0))

    def flaky(instrument, timeframe, start, end):
        if start < now - dt.timedelta(days=100):
            raise RuntimeError("401 token expired")
        return fetch(instrument, timeframe, start, end)

    res = backfill_instrument("NIFTY", "5min", target_start=now - dt.timedelta(days=300), now=now,
                              fetch=flaky, cache_dir=tmp_path)
    assert res.errors and "token expired" in res.errors[0]
    assert res.rows_after > 0
