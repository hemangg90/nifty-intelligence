"""Real option prices in the backtest: how a trade is priced, when it is not, and how the engine counts the sources."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.backtesting import engine
from quant_intelligence.options import real_premium
from quant_intelligence.options.real_premium import DELTA_BOUNDS, RealPremiumModel, history_covers
from quant_intelligence.strategies.base_strategy import Setup, TradeResult
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.tests.options_helpers import make_history, session_bars

TF = 5


@pytest.fixture(scope="module")
def bars(synthetic_ohlcv):
    return session_bars(synthetic_ohlcv)


@pytest.fixture(scope="module")
def history(bars):
    return make_history(bars)


def trade(bars, i, j, direction="LONG", stop_dist=30.0):
    """A trade signalled on bar i that exits on bar j (same session)."""
    entry = float(bars["close"].iloc[i])
    stop = entry - stop_dist if direction == "LONG" else entry + stop_dist
    setup = Setup(bars["timestamp"].iloc[i], direction, entry, stop, entry + (2 * stop_dist if direction == "LONG" else -2 * stop_dist))
    return TradeResult(setup=setup, exit_timestamp=bars["timestamp"].iloc[j], exit_price=float(bars["close"].iloc[j]), r_multiple=0.0,
                       holding_period_bars=j - i, mfe=0.0, mae=0.0, gross_pnl=0.0, outcome="SCRATCH")


def same_day_pair(bars, gap=6):
    day = pd.to_datetime(bars["timestamp"]).dt.date
    i = int(np.flatnonzero(day.to_numpy() == day.iloc[20])[10])
    return i, i + gap


def test_entry_is_the_next_bar_open_and_exit_the_close_of_the_same_strike(bars, history):
    i, j = same_day_pair(bars)
    m = RealPremiumModel(history, TF, "next_open")
    p = m.price(trade(bars, i, j))
    entry_row = history.row("CALL", 0, bars["timestamp"].iloc[i + 1])
    assert p["entry"] == pytest.approx(entry_row["open"]) and p["strike"] == entry_row["strike"] and p["kind"] == "CALL"
    exit_close = next(history.row("CALL", o, bars["timestamp"].iloc[j])["close"] for o in history.offsets
                      if history.row("CALL", o, bars["timestamp"].iloc[j])["strike"] == p["strike"])
    assert p["exit"] == pytest.approx(exit_close) and p["source"] == "dhan_history"


def test_legacy_timing_enters_at_the_signal_bar_close(bars, history):
    i, j = same_day_pair(bars)
    p = RealPremiumModel(history, TF, "signal_close").price(trade(bars, i, j))
    assert p["entry"] == pytest.approx(history.row("CALL", 0, bars["timestamp"].iloc[i])["close"])


def test_a_short_signal_buys_the_put(bars, history):
    i, j = same_day_pair(bars)
    p = RealPremiumModel(history, TF, "next_open").price(trade(bars, i, j, "SHORT"))
    assert p["kind"] == "PUT" and p["entry"] == pytest.approx(history.row("PUT", 0, bars["timestamp"].iloc[i + 1])["open"])


def test_the_held_strike_is_followed_after_at_the_money_moves(bars, history):
    day = pd.to_datetime(bars["timestamp"]).dt.date.to_numpy()
    close = bars["close"].to_numpy()
    for i in range(len(bars) - 40):
        for j in range(i + 3, i + 30):
            if day[i] == day[j] and abs(close[j] - close[i]) > 120:
                p = RealPremiumModel(history, TF, "next_open").price(trade(bars, i, j))
                row_exit = history.row("CALL", 0, bars["timestamp"].iloc[j])
                assert p is not None and p["strike"] != row_exit["strike"]       # ATM has moved on, the trade has not
                return
    pytest.skip("no large same-session move in the fixture")


def test_risk_is_a_delta_from_the_neighbouring_strikes_times_the_stop_distance(bars, history):
    i, j = same_day_pair(bars)
    ts = bars["timestamp"].iloc[i + 1]
    up, down = history.row("CALL", 1, ts), history.row("CALL", -1, ts)
    delta = min(max((down["close"] - up["close"]) / (up["strike"] - down["strike"]), DELTA_BOUNDS[0]), DELTA_BOUNDS[1])
    p = RealPremiumModel(history, TF, "next_open").price(trade(bars, i, j, stop_dist=40.0))
    assert p["delta"] == pytest.approx(delta) and p["risk"] == pytest.approx(delta * 40.0)
    assert DELTA_BOUNDS[0] <= p["delta"] <= DELTA_BOUNDS[1]


def test_overnight_trades_are_not_priced(bars, history):
    day = pd.to_datetime(bars["timestamp"]).dt.date
    i = int(np.flatnonzero(day.to_numpy() == day.iloc[20])[-3])
    j = i + 6                                                  # lands on the next session
    m = RealPremiumModel(history, TF, "next_open")
    assert m.price(trade(bars, i, j)) is None and m.skipped_reasons["held overnight"] == 1


def test_missing_data_is_a_skip_with_a_reason_never_a_guess(bars):
    i, j = same_day_pair(bars)
    t = trade(bars, i, j)
    only_atm = make_history(bars, offsets=[0])
    m = RealPremiumModel(only_atm, TF, "next_open")
    assert m.price(t) is None and "no neighbouring strikes for delta" in m.skipped_reasons

    empty_after = make_history(bars[bars["timestamp"] < bars["timestamp"].iloc[i]], offsets=range(-3, 4))
    m2 = RealPremiumModel(empty_after, TF, "next_open")
    assert m2.price(t) is None and "no option bar at entry" in m2.skipped_reasons

    no_exit = dataclasses.replace(t, exit_price=None)
    assert RealPremiumModel(make_history(bars), TF).price(no_exit) is None


def test_a_strike_that_left_the_window_cannot_be_priced(bars):
    day = pd.to_datetime(bars["timestamp"]).dt.date.to_numpy()
    close = bars["close"].to_numpy()
    narrow = make_history(bars, offsets=range(-1, 2))
    for i in range(len(bars) - 40):
        for j in range(i + 3, i + 30):
            if day[i] == day[j] and abs(close[j] - close[i]) > 160:
                m = RealPremiumModel(narrow, TF, "next_open")
                assert m.price(trade(bars, i, j)) is None and "held strike not in stored offsets at exit" in m.skipped_reasons
                return
    pytest.skip("no large same-session move in the fixture")


def test_coverage_is_counted(bars, history):
    i, j = same_day_pair(bars)
    m = RealPremiumModel(history, TF, "next_open")
    m.price(trade(bars, i, j))
    assert m.coverage() == {"priced": 1, "skipped": 0, "skipped_reasons": {}}


def test_history_covers_only_overlapping_data(bars, history):
    assert history_covers(history, bars) and not history_covers(None, bars)
    later = bars.assign(timestamp=bars["timestamp"] + pd.Timedelta(days=900))
    assert not history_covers(history, later)


# ---- the engine ---------------------------------------------------------------------------------------------------

def run(monkeypatch, synthetic_ohlcv, synthetic_features, history, flag=True, timing="signal_close"):
    s = dataclasses.replace(engine.SETTINGS, option_history_in_backtest=flag, execution_timing=timing)
    monkeypatch.setattr(engine, "SETTINGS", s)
    monkeypatch.setattr(real_premium, "load_history_for", lambda inst: history)
    return engine.run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=75)


def test_flag_off_changes_nothing_and_everything_is_labelled_assumed(monkeypatch, synthetic_ohlcv, synthetic_features, history):
    r = run(monkeypatch, synthetic_ohlcv, synthetic_features, history, flag=False)
    m = r.metrics
    assert m["n_real_premium"] == 0 and m["n_assumed_premium"] == m["n_trades"] > 0 and m["real_premium_share"] == 0.0
    assert all(t.setup.meta["premium_source"] == "assumed" for t in r.trades)
    assert "option_premium" not in r.manifest


@pytest.mark.parametrize("timing", ["signal_close", "next_open"])
def test_flag_on_prices_trades_from_the_history_and_counts_each_source(monkeypatch, synthetic_ohlcv, synthetic_features, history, timing):
    r = run(monkeypatch, synthetic_ohlcv, synthetic_features, history, timing=timing)
    m = r.metrics
    assert m["n_real_premium"] > 0
    assert m["n_real_premium"] + m["n_modelled_premium"] + m["n_assumed_premium"] == m["n_trades"]
    assert m["real_premium_share"] == pytest.approx(m["n_real_premium"] / m["n_trades"])
    real = [t for t in r.trades if t.setup.meta["premium_source"] == "dhan_history"]
    assert len(real) == m["n_real_premium"] and all(t.setup.meta["bs_premium"]["source"] == "dhan_history" for t in real)
    op = r.manifest["option_premium"]
    assert op["source"] == "dhan_history" and op["history_fingerprint"] == history.fingerprint() and op["priced"] == m["n_real_premium"]


def test_real_prices_change_the_costed_result(monkeypatch, synthetic_ohlcv, synthetic_features, history):
    real = run(monkeypatch, synthetic_ohlcv, synthetic_features, history)
    assumed = run(monkeypatch, synthetic_ohlcv, synthetic_features, history, flag=False)
    assert real.metrics["n_trades"] == assumed.metrics["n_trades"]
    assert real.metrics["net_pnl_total"] != pytest.approx(assumed.metrics["net_pnl_total"])


def test_a_flag_with_no_overlapping_history_falls_back_and_says_so(monkeypatch, synthetic_ohlcv, synthetic_features, bars):
    far = make_history(bars.assign(timestamp=bars["timestamp"] + pd.Timedelta(days=900)))
    r = run(monkeypatch, synthetic_ohlcv, synthetic_features, far)
    assert r.metrics["n_real_premium"] == 0 and r.metrics["n_assumed_premium"] == r.metrics["n_trades"]


def test_no_history_on_disk_is_handled(monkeypatch, synthetic_ohlcv, synthetic_features):
    r = run(monkeypatch, synthetic_ohlcv, synthetic_features, None)
    assert r.metrics["n_real_premium"] == 0 and r.metrics["n_trades"] > 0
