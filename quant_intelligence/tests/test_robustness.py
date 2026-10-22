"""The five robustness tests: each passes a robust edge, fails a fragile one, and refuses to judge thin data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.backtesting.engine import run_backtest
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.strategies.supertrend_strategy import SupertrendStrategy
from quant_intelligence.tests.validation_helpers import edge_series, make_trades
import json

from quant_intelligence.validation.evidence import REQUIRED_ROBUSTNESS_TESTS, EvidenceState, OutOfSampleResult, classify_evidence
from quant_intelligence.validation.metrics import compute_metrics, trade_table
from quant_intelligence.validation.robustness import (
    UNKNOWN,
    cost_stress,
    label_trades,
    neighbour_values,
    parameter_sensitivity,
    period_consistency,
    persist_regime_stats,
    regime_sensitivity,
    run_robustness,
    sequence_analysis,
    stationary_bootstrap_paths,
)
from quant_intelligence.validation.uncertainty import estimate_uncertainty

QTY = 75


def table_of(rs, **kw):
    return trade_table(make_trades(rs, **kw), QTY)


# ---- 1. parameter sensitivity -------------------------------------------------------------------------------------

def test_neighbours_step_either_side_and_stay_inside_the_range():
    assert neighbour_values(1.0, 0.5, 2.5, 0.25) == [0.75, 1.25]
    assert neighbour_values(1.0, 0.5, 2.5, 0.25, steps=2) == [0.75, 1.25, 0.5, 1.5]
    assert neighbour_values(0.5, 0.5, 2.5, 0.25) == [0.75]            # clipped at the floor, the default itself is excluded
    assert neighbour_values(2.5, 0.5, 2.5, 0.25) == [2.25]


def test_integer_parameters_stay_integers():
    got = neighbour_values(10, 3, 20, 1)
    assert got == [9, 11] and all(isinstance(v, int) for v in got)


def test_parameter_sensitivity_reports_each_neighbour(synthetic_ohlcv, synthetic_features):
    t = parameter_sensitivity(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", QTY, baseline_r=0.1)
    params = {r["parameter"] for r in t.rows}
    assert params <= set(MomentumStrategy().parameter_ranges) and len(params) >= 3
    judged = [r for r in t.rows if r.get("net_expectancy_r") is not None]
    assert t.passed is None or isinstance(t.passed, bool)
    if t.passed is not None:
        assert t.value == pytest.approx(sum(r["net_expectancy_r"] > 0 for r in judged) / len(judged))


def test_a_strategy_with_too_few_neighbours_is_not_judged(synthetic_ohlcv, synthetic_features):
    t = parameter_sensitivity(SupertrendStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", QTY, baseline_r=0.3)
    assert t.passed is None and "neighbours" in t.detail


def test_an_unprofitable_default_cannot_pass_parameter_sensitivity(synthetic_ohlcv, synthetic_features):
    t = parameter_sensitivity(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", QTY, baseline_r=-0.05)
    assert t.passed is False and "not itself profitable" in t.detail


# ---- 2. regime sensitivity ----------------------------------------------------------------------------------------

def labelled(spec):
    """spec: {regime: [net_r...]} -> a labelled trade frame."""
    rows = []
    day = 0
    for regime, values in spec.items():
        for v in values:
            rows.append({"net_r": v, "regime": regime, "trend": "TREND" if regime.startswith("TREND") else "NON_TREND",
                         "volatility": "HIGH_VOL", "gap": "NO_GAP", "signal_date": day})
            day += 1
    return pd.DataFrame(rows)


def test_an_edge_spread_over_several_regimes_passes():
    rng = np.random.default_rng(1)
    df = labelled({"TREND_UP": rng.normal(0.2, 1, 80), "RANGE": rng.normal(0.2, 1, 80), "LOW_VOL": rng.normal(0.2, 1, 80)})
    t, table = regime_sensitivity(df)
    assert t.passed is True and {"regime", "trend", "volatility", "gap"} == set(table["dimension"])


def test_an_edge_that_lives_in_one_regime_fails():
    rng = np.random.default_rng(2)
    df = labelled({"TREND_UP": rng.normal(1.5, 1, 100), "RANGE": rng.normal(-0.1, 1, 100), "LOW_VOL": rng.normal(-0.05, 1, 100)})
    t, _ = regime_sensitivity(df)
    assert t.passed is False and "without the best regime" in t.detail


def test_a_regime_that_loses_badly_fails_even_if_the_total_is_positive():
    rng = np.random.default_rng(3)
    df = labelled({"TREND_UP": rng.normal(0.5, 1, 100), "RANGE": rng.normal(0.5, 1, 100), "LOW_VOL": rng.normal(-0.6, 1, 100)})
    assert regime_sensitivity(df)[0].passed is False


def test_thin_regimes_are_listed_but_cannot_fail_or_pass():
    rng = np.random.default_rng(4)
    df = labelled({"TREND_UP": rng.normal(0.3, 1, 100), "RANGE": rng.normal(-3.0, 1, 5)})
    t, table = regime_sensitivity(df)
    assert t.passed is None and "RANGE" in set(table["label"])


def test_labels_at_a_signal_bar_never_depend_on_later_bars(synthetic_ohlcv, synthetic_features):
    res = run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=QTY)
    table = trade_table(res.trades, QTY)
    full = label_trades(table, synthetic_features)
    cut = table["signal_ts"].iloc[len(table) // 2]
    early = table[table["signal_ts"] <= cut].reset_index(drop=True)
    truncated = synthetic_features[synthetic_features["timestamp"] <= cut].reset_index(drop=True)
    again = label_trades(early, truncated)
    cols = ["regime", "trend", "volatility", "gap"]
    pd.testing.assert_frame_equal(full[full["signal_ts"] <= cut].reset_index(drop=True)[cols], again[cols])
    assert set(full["volatility"]) <= {"HIGH_VOL", "LOW_VOL", UNKNOWN}


# ---- 3. cost stress -----------------------------------------------------------------------------------------------

def _constant_edge(extra_slopes: float):
    """Trades whose net expectancy at 1x slippage is `extra_slopes` times the per-1x slippage cost."""
    probe = cost_stress(make_trades([0.0] * 60), QTY, "NSE")
    e1 = next(r["net_expectancy_r"] for r in probe.rows if r["slippage_multiple"] == 1.0)
    e2 = next(r["net_expectancy_r"] for r in probe.rows if r["slippage_multiple"] == 2.0)
    slope = e1 - e2
    return -e1 + extra_slopes * slope, slope


def test_cost_stress_passes_an_edge_that_survives_double_slippage():
    c, _ = _constant_edge(2.5)
    t = cost_stress(make_trades([c] * 60), QTY, "NSE")
    assert t.passed is True and t.value > 0


def test_cost_stress_fails_an_edge_that_double_slippage_erases():
    c, _ = _constant_edge(0.5)               # net = slope * (m + 1 - k): positive at 1x, negative at 2x
    t = cost_stress(make_trades([c] * 60), QTY, "NSE")
    by_k = {r["slippage_multiple"]: r["net_expectancy_r"] for r in t.rows}
    assert by_k[1.0] > 0 > by_k[2.0] and t.passed is False


def test_break_even_multiple_is_exact_because_net_r_is_linear_in_slippage():
    c, _ = _constant_edge(4.0)
    t = cost_stress(make_trades([c] * 60), QTY, "NSE")
    by_k = {r["slippage_multiple"]: r["net_expectancy_r"] for r in t.rows}
    step = by_k[1.0] - by_k[2.0]
    assert by_k[2.0] - by_k[3.0] == pytest.approx(step, rel=1e-9)             # equal losses per extra 1x: linear
    expected = 1.0 + by_k[1.0] / step
    assert f"{expected:.2f}x" in t.detail
    assert by_k[1.0] - step * (expected - 1.0) == pytest.approx(0.0, abs=1e-9)  # zero exactly at the stated multiple


def test_cost_stress_with_no_trades_is_not_run():
    assert cost_stress([], QTY, "NSE").passed is None


def test_cost_override_does_not_leak_into_the_configured_costs():
    before = SETTINGS.costs.slippage_ticks
    cost_stress(make_trades(edge_series(60)), QTY, "NSE")
    assert SETTINGS.costs.slippage_ticks == before


# ---- 4. period consistency ----------------------------------------------------------------------------------------

def test_a_steady_edge_is_consistent_across_months():
    t = period_consistency(table_of(edge_series(400, mean=0.5, sd=1.0)))
    assert t.passed is True and t.value > 0.6


def test_profit_from_one_month_is_not_consistency():
    rs = np.r_[np.full(40, -0.3), np.full(20, 6.0), np.full(340, -0.2)]      # one big month in a losing year
    t = period_consistency(table_of(rs))
    assert t.passed is False


def test_a_strategy_with_a_negative_total_fails_period_consistency():
    assert period_consistency(table_of(edge_series(400, mean=-0.3))).passed is False


def test_too_few_months_is_not_judged():
    t = period_consistency(table_of(edge_series(60, mean=0.5)))
    assert t.passed is None and "usable calendar months" in t.detail


# ---- 5. sequence analysis -----------------------------------------------------------------------------------------

def test_a_robust_edge_survives_reordering():
    t = sequence_analysis(table_of(edge_series(400, mean=0.5, sd=1.0)))
    assert t.passed is True and t.value < SETTINGS.evidence.max_drawdown_r


def test_a_losing_series_fails_sequence_analysis():
    assert sequence_analysis(table_of(edge_series(400, mean=-0.3))).passed is False


def test_sequence_analysis_needs_enough_trades():
    assert sequence_analysis(table_of(edge_series(20))).passed is None


def test_sequence_analysis_is_deterministic():
    tb = table_of(edge_series(300, mean=0.2))
    assert sequence_analysis(tb).value == sequence_analysis(tb).value


def test_bootstrap_paths_resample_the_observed_values_only():
    x = np.arange(50, dtype=float)
    paths = stationary_bootstrap_paths(x, 200, 5.0, seed=1)
    assert paths.shape == (200, 50) and set(np.unique(paths)) <= set(x)
    assert abs(paths.mean() - x.mean()) < 3


def test_long_blocks_preserve_runs_that_iid_shuffling_destroys():
    x = np.repeat([1.0, -1.0], 25)           # one winning run then one losing run
    long_blocks = stationary_bootstrap_paths(x, 400, 25.0, seed=2)
    iid = stationary_bootstrap_paths(x, 400, 1.0, seed=2)
    run_len = lambda p: (np.diff(p, axis=1) == 0).mean()  # noqa: E731
    assert run_len(long_blocks) > run_len(iid)


# ---- orchestration ------------------------------------------------------------------------------------------------

def test_run_robustness_returns_every_required_test(synthetic_ohlcv, synthetic_features):
    res = run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=QTY)
    rep = run_robustness(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", res.trades, QTY)
    assert set(rep.tests) == set(REQUIRED_ROBUSTNESS_TESTS)
    outcomes = rep.outcomes()
    assert all(v is None or isinstance(v, bool) for v in outcomes.values())
    # the classifier accepts these outcomes as they are
    trades = make_trades(edge_series(400, mean=0.5))
    tbl = trade_table(trades, QTY)
    a = classify_evidence(compute_metrics(trades, QTY, table=tbl), estimate_uncertainty(tbl), n_sessions=500,
                          out_of_sample=OutOfSampleResult(60, 0.2), robustness=outcomes)
    assert a.state in (EvidenceState.ROBUSTNESS_TESTING, EvidenceState.VALIDATED_FOR_RESEARCH, EvidenceState.VALIDATION_FAILED)
    assert "NaN" not in json.dumps(rep.as_dict(), default=str)


def test_regime_table_is_persisted_append_only(synthetic_ohlcv, synthetic_features):
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import StrategyRegimeStat

    res = run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=QTY)
    rep = run_robustness(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", res.trades, QTY)
    n = persist_regime_stats(rep, "Momentum", "TESTPERSIST", "5min", "run-1", "cfg")
    assert n == len(rep.regime_table) > 0
    persist_regime_stats(rep, "Momentum", "TESTPERSIST", "5min", "run-2", "cfg")
    with get_session() as s:
        assert s.query(StrategyRegimeStat).filter_by(instrument="TESTPERSIST").count() == 2 * n
