"""Signal-to-execution timing: a signal known at a bar's close must fill at the NEXT bar's open."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from quant_intelligence.backtesting import engine
from quant_intelligence.strategies.base_strategy import Setup, next_open_fill
from quant_intelligence.strategies.orb_strategy import ORBStrategy


def _bars(rows, day="2024-06-03", start="09:15"):
    ts = pd.date_range(f"{day} {start}", periods=len(rows), freq="5min")
    return pd.DataFrame(
        [(t, o, h, l, c, 1000.0) for t, (o, h, l, c) in zip(ts, rows)],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _long(entry=100.0, stop=98.0, target=104.0):
    return Setup(pd.Timestamp("2024-06-03 09:15"), "LONG", entry, stop, target)


def test_fill_is_next_open_not_signal_close():
    bars = _bars([(99, 101, 99, 100), (100.5, 101, 100, 100.8), (100.8, 104.5, 100.5, 104)])
    result = ORBStrategy().simulate_trade(_long(), bars, 0, execution_timing="next_open")
    assert result.setup.entry_price == 100.5            # next open, not the signal close of 100
    assert result.setup.meta["signal_price"] == 100.0
    assert result.outcome == "WIN" and result.exit_price == 104.0
    assert result.r_multiple == pytest.approx((104 - 100.5) / (100.5 - 98))


def test_fill_bar_range_counts_after_the_fill():
    # The fill bar itself dips to the stop: that loss is real and must be taken (the legacy model skipped this bar).
    bars = _bars([(99, 101, 99, 100), (100.5, 101, 97.5, 99), (99, 105, 99, 105)])
    result = ORBStrategy().simulate_trade(_long(), bars, 0, execution_timing="next_open")
    assert result.outcome == "LOSS" and result.exit_price == 98.0 and result.holding_period_bars == 1


def test_stop_checked_before_target_in_same_bar():
    bars = _bars([(99, 101, 99, 100), (100.5, 105, 97, 101)])
    assert ORBStrategy().simulate_trade(_long(), bars, 0, execution_timing="next_open").outcome == "LOSS"


def test_short_side_mirrors_long():
    bars = _bars([(101, 101, 99, 100), (99.5, 100, 95.5, 96)])
    short = Setup(pd.Timestamp("2024-06-03 09:15"), "SHORT", 100.0, 102.0, 96.0)
    result = ORBStrategy().simulate_trade(short, bars, 0, execution_timing="next_open")
    assert result.setup.entry_price == 99.5 and result.outcome == "WIN" and result.exit_price == 96.0


def test_open_position_is_flattened_at_session_end_not_carried_overnight():
    day1 = _bars([(99, 101, 99, 100), (100.2, 101, 99.5, 100.4), (100.4, 101, 99.5, 100.6)])
    day2 = _bars([(110, 120, 109, 115)], day="2024-06-04")
    bars = pd.concat([day1, day2], ignore_index=True)
    result = ORBStrategy().simulate_trade(_long(), bars, 0, max_holding_bars=75, execution_timing="next_open")
    assert result.exit_timestamp == day1["timestamp"].iloc[-1]
    assert result.exit_price == 100.6                    # day 1's last close, not day 2's 115 gap


@pytest.mark.parametrize("rows, why", [
    ([(99, 101, 99, 100)], "no next bar"),
    ([(99, 101, 99, 100), (97.5, 100, 97, 99)], "open already through the stop"),
    ([(99, 101, 99, 100), (104.5, 106, 104, 105)], "open already through the target"),
    ([(99, 101, 99, 100), (98.5, 101, 98.4, 100)], "stop distance collapsed to under half the plan"),
])
def test_unexecutable_signals_have_no_fill(rows, why):
    assert next_open_fill(_long(), _bars(rows), 0) is None, why


def test_signal_on_last_bar_of_session_has_no_fill():
    bars = pd.concat([_bars([(99, 101, 99, 100)]), _bars([(100, 101, 99, 100)], day="2024-06-04")], ignore_index=True)
    assert next_open_fill(_long(), bars, 0) is None


def test_default_timing_is_unchanged_legacy():
    bars = _bars([(99, 101, 99, 100), (100.5, 101, 100, 100.8), (100.8, 104.5, 100.5, 104)])
    result = ORBStrategy().simulate_trade(_long(), bars, 0)  # no override: SETTINGS default
    assert result.setup.entry_price == 100.0 and "execution" not in result.setup.meta


def test_engine_skips_unexecutable_signals_and_reports_them(synthetic_ohlcv, synthetic_features, monkeypatch):
    from quant_intelligence.config import settings as settings_mod

    legacy = engine.run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)
    patched = dataclasses.replace(settings_mod.SETTINGS, execution_timing="next_open")
    monkeypatch.setattr(engine, "SETTINGS", patched)
    fixed = engine.run_backtest(ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False)

    assert fixed.metrics["execution_timing"] == "next_open" and legacy.metrics["execution_timing"] == "signal_close"
    assert fixed.metrics["n_signals"] == fixed.metrics["n_trades"] + fixed.metrics["n_unexecutable"] if fixed.metrics.get("n_trades") else True
    for t in fixed.trades:
        assert t.setup.meta["execution"] == "next_open"
        assert t.setup.meta["fill_timestamp"] > t.setup.timestamp   # filled strictly after the signal bar
