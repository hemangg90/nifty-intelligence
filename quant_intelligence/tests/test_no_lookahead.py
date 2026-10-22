"""
Look-ahead audit: a decision at time t may only depend on data up to t.

For every price-action strategy, features and setups are recomputed on data truncated at several cut points.
Every setup the truncated run produces must equal the setup the full-data run produced for that timestamp, and
the truncated run must not miss one the full run has up to the cut. Any difference means a feature or a signal
used information from after the decision time.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, STRATEGY_CLASSES

PRICE_ACTION = [c for c in STRATEGY_CLASSES.values() if c.name not in CHAIN_AWARE_STRATEGY_NAMES]
CUT_FRACTIONS = (0.35, 0.6)
SIGNAL_CUTS = 3  # also truncate exactly AT sampled signal bars: a signal that needed the future cannot survive that


def _key(s):
    return (s.timestamp, s.direction, round(float(s.entry_price), 6), round(float(s.stop_price), 6), round(float(s.target_price), 6))


def _divergence(strategy, ohlcv, cut, full, features=True):
    """(only_in_truncated, missing_from_truncated) for a run on ohlcv[:cut] versus the full-data setups."""
    part = ohlcv.iloc[:cut].reset_index(drop=True)
    truncated = {_key(s) for s in strategy.generate_historical_setups(part, compute_features(part) if features else None)}
    expected = {k for k in full if k[0] <= part["timestamp"].iloc[-1]}
    return sorted(truncated - expected), sorted(expected - truncated)


def _cut_points(ohlcv, full):
    index = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}
    signal_bars = sorted(index[k[0]] for k in full if k[0] in index)
    picks = signal_bars[:: max(len(signal_bars) // SIGNAL_CUTS, 1)][:SIGNAL_CUTS]
    return sorted({int(len(ohlcv) * f) for f in CUT_FRACTIONS} | {i + 1 for i in picks})


@pytest.mark.parametrize("cls", PRICE_ACTION, ids=lambda c: c.name)
def test_setups_up_to_t_do_not_depend_on_data_after_t(cls, synthetic_ohlcv, synthetic_features):
    strategy = cls()
    full = {_key(s) for s in strategy.generate_historical_setups(synthetic_ohlcv, synthetic_features)}
    for cut in _cut_points(synthetic_ohlcv, full):
        extra, missing = _divergence(strategy, synthetic_ohlcv, cut, full)
        assert not extra and not missing, (
            f"{cls.name}: setups before {synthetic_ohlcv['timestamp'].iloc[cut - 1]} change with later data "
            f"(only in truncated: {extra[:2]}, missing from truncated: {missing[:2]})"
        )


def test_features_up_to_t_do_not_depend_on_data_after_t(synthetic_ohlcv, synthetic_features):
    cut = int(len(synthetic_ohlcv) * 0.6)
    truncated = compute_features(synthetic_ohlcv.iloc[:cut].reset_index(drop=True))
    full = synthetic_features.iloc[:cut].reset_index(drop=True)
    cols = [c for c in truncated.columns if c != "timestamp"]
    bad = []
    for c in cols:
        a, b = truncated[c], full[c]
        if a.dtype.kind in "fi":
            same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            bad.append(c)
    assert not bad, f"features that change when later bars are added: {bad}"


class _LeakyStrategy(STRATEGY_CLASSES["Donchian Channel Breakout"]):
    """Canary: peeks at the NEXT bar's close to decide. The audit must flag it."""

    name = "Leaky Canary"

    def generate_historical_setups(self, ohlcv, features):
        from quant_intelligence.strategies.base_strategy import Setup

        out = []
        for i in range(30, len(ohlcv) - 1):
            if ohlcv["close"].iloc[i + 1] > ohlcv["close"].iloc[i] * 1.0015:  # uses the future
                px = float(ohlcv["close"].iloc[i])
                out.append(Setup(ohlcv["timestamp"].iloc[i], "LONG", px, px - 1, px + 2))
        return out


def test_audit_catches_a_strategy_that_peeks_at_the_future(synthetic_ohlcv):
    strategy = _LeakyStrategy()
    full = {_key(s) for s in strategy.generate_historical_setups(synthetic_ohlcv, None)}
    assert len(full) >= SIGNAL_CUTS
    flagged = False
    for cut in _cut_points(synthetic_ohlcv, full):
        extra, missing = _divergence(strategy, synthetic_ohlcv, cut, full, features=False)
        flagged = flagged or bool(extra or missing)
    assert flagged, "the audit failed to detect a strategy that reads the next bar"
