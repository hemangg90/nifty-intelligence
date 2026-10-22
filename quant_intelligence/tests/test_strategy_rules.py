"""Every strategy has explicit rules, sane parameter ranges, a version and the standard output record."""
from __future__ import annotations

import pandas as pd
import pytest

from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, STRATEGY_CLASSES
from quant_intelligence.strategies.rules import PARAMETER_RANGES, RULES, STRATEGY_VERSIONS

ALL = list(STRATEGY_CLASSES.values())


def test_rules_cover_exactly_the_registered_strategies():
    assert set(RULES) == set(STRATEGY_CLASSES)
    assert set(PARAMETER_RANGES) == set(STRATEGY_CLASSES)
    assert set(STRATEGY_VERSIONS) == set(STRATEGY_CLASSES)


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_rules_are_complete(cls):
    r = cls().rules
    assert r is not None
    for field in ("entry", "exit", "stop", "target", "sizing", "session", "time_exit"):
        assert getattr(r, field).strip(), f"{cls.name}: empty rule '{field}'"
    assert r.instruments and r.timeframes and r.max_holding_bars > 0
    assert cls().version


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_parameter_ranges_are_valid_and_contain_the_defaults(cls):
    ranges = cls().parameter_ranges
    assert ranges, f"{cls.name} has no parameter ranges"
    for param, (low, high, step) in ranges.items():
        assert param in cls.default_parameters, f"{cls.name}: range for unknown parameter {param}"
        assert low < high and step > 0, f"{cls.name}.{param}"
        assert low <= cls.default_parameters[param] <= high, f"{cls.name}.{param} default outside its range"


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_chain_strategies_are_flagged_not_backtestable(cls):
    assert cls().rules.backtestable == (cls.name not in CHAIN_AWARE_STRATEGY_NAMES)
    if cls.name in CHAIN_AWARE_STRATEGY_NAMES:
        assert cls().rules.limitations


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_every_historical_setup_has_a_valid_geometry(cls, synthetic_ohlcv, synthetic_features):
    for s in cls().generate_historical_setups(synthetic_ohlcv, synthetic_features):
        if s.direction == "LONG":
            assert s.stop_price < s.entry_price, f"{cls.name} LONG stop not below entry"
        else:
            assert s.stop_price > s.entry_price, f"{cls.name} SHORT stop not above entry"


def test_setup_record_has_the_standard_schema():
    strategy = STRATEGY_CLASSES["Opening Range Breakout"]()
    rec = Setup(pd.Timestamp("2024-06-03 10:00"), "LONG", 100.0, 98.0, 104.0).to_record(strategy, "NIFTY")
    assert set(rec) == {"strategy", "instrument", "timeframe", "timestamp", "signal", "entry_condition_met", "entry_price",
                        "stop_loss", "target", "invalidation_condition", "strategy_version"}
    assert rec["strategy"] == "Opening Range Breakout" and rec["signal"] == "LONG" and rec["strategy_version"] == "1.0"
    assert "stop" in rec["invalidation_condition"]
