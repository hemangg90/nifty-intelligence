"""Evidence states: each one is reachable, each is reached for the right reason, and none is awarded on weak evidence."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.tests.validation_helpers import edge_series, make_trades
from quant_intelligence.validation.evidence import (
    REQUIRED_ROBUSTNESS_TESTS,
    STATE_DEFINITIONS,
    EvidenceState,
    OutOfSampleResult,
    classify_evidence,
)
from quant_intelligence.validation.metrics import compute_metrics, trade_table
from quant_intelligence.validation.uncertainty import estimate_uncertainty

QTY = 75
S = EvidenceState
ALL_PASS = {name: True for name in REQUIRED_ROBUSTNESS_TESTS}
GOOD_OOS = OutOfSampleResult(n_trades=60, net_expectancy_r=0.2)


def assess(rs, per_day=1, **kw):
    trades = make_trades(rs, per_day=per_day)
    table = trade_table(trades, QTY)
    kw.setdefault("n_sessions", 500)
    return classify_evidence(compute_metrics(trades, QTY, table=table), estimate_uncertainty(table), **kw)


def strong(n=400):
    return edge_series(n, mean=0.5, sd=1.0)


# ---- the six states ----------------------------------------------------------------------------------

def test_insufficient_data_when_the_strategy_cannot_be_backtested():
    a = assess(strong(), backtestable=False)
    assert a.state is S.INSUFFICIENT_DATA and "option-chain" in a.reasons[0]


def test_insufficient_data_when_the_data_failed_its_quality_gate():
    assert assess(strong(), data_quality="FAIL").state is S.INSUFFICIENT_DATA


def test_insufficient_data_when_the_history_is_too_short():
    a = assess(strong(), n_sessions=SETTINGS.evidence.min_sessions - 1)
    assert a.state is S.INSUFFICIENT_DATA and "sessions" in a.reasons[0]


def test_degraded_data_is_not_a_reason_to_refuse():
    assert assess(strong(), data_quality="DEGRADED").state is not S.INSUFFICIENT_DATA


def test_insufficient_sample_for_a_handful_of_trades():
    a = assess(strong(30))
    assert a.state is S.INSUFFICIENT_SAMPLE and any("trades" in r for r in a.reasons)


def test_a_lucky_small_sample_is_not_qualified_however_good_it_looks():
    a = assess(np.full(25, 3.0))     # 25 straight 3R winners
    assert a.state is S.INSUFFICIENT_SAMPLE


def test_many_trades_on_few_days_have_too_little_independent_information():
    a = assess(strong(300), per_day=15)   # 300 trades but only 20 trade-days
    assert a.state is S.INSUFFICIENT_SAMPLE and any("trade_days" in r for r in a.reasons)


def test_strongly_clustered_trades_fail_the_effective_sample_test():
    rng = np.random.default_rng(9)
    clustered = np.repeat(rng.normal(0.5, 1.5, 50), 6) + rng.normal(0, 0.1, 300)   # 6 trades a day share one outcome
    a = assess(clustered, per_day=6)
    assert a.state is S.INSUFFICIENT_SAMPLE
    assert any(c.name == "effective_trades" and not c.passed for c in a.checks)


def test_validation_failed_for_a_negative_edge_with_a_large_sample():
    a = assess(edge_series(500, mean=-0.2))
    assert a.state is S.VALIDATION_FAILED and any("net expectancy" in r for r in a.reasons)


def test_validation_failed_for_an_edge_that_is_not_distinguishable_from_zero():
    a = assess(edge_series(150, mean=0.15, sd=1.5))     # mean net ~0.05 R on a noisy series: interval straddles zero
    assert a.state is S.VALIDATION_FAILED


def test_a_70_percent_win_rate_does_not_rescue_a_losing_strategy():
    a = assess(([0.3] * 7 + [-1.5] * 3) * 30)
    assert a.state is S.VALIDATION_FAILED
    assert a.checks[next(i for i, c in enumerate(a.checks) if c.name == "profit_factor")].value < 1.0


def test_drawdown_over_the_limit_fails_even_with_positive_expectancy():
    rs = edge_series(500, mean=0.7, sd=2.0, seed=3)         # a real edge, but a swingy equity curve (max drawdown ~10.5 R)
    assert assess(rs).state is S.RESEARCH_QUALIFIED          # fine under the default policy
    strict = dataclasses.replace(SETTINGS.evidence, max_drawdown_r=6.0)
    a = assess(rs, policy=strict)
    assert a.state is S.VALIDATION_FAILED and any("drawdown" in r for r in a.reasons)


def test_a_perfectly_regime_structured_series_has_almost_no_independent_information():
    rs = np.r_[np.full(100, -0.5), np.full(300, 1.2)]       # one losing block then one winning block
    a = assess(rs)
    assert a.state is S.INSUFFICIENT_SAMPLE
    assert next(c for c in a.checks if c.name == "effective_trades").value < 10


def test_research_qualified_when_in_sample_gates_pass_without_oos():
    a = assess(strong())
    assert a.state is S.RESEARCH_QUALIFIED and a.state.rankable
    assert "no out-of-sample" in a.reasons[0]


def test_out_of_sample_with_too_few_trades_cannot_confirm_or_reject():
    a = assess(strong(), out_of_sample=OutOfSampleResult(n_trades=5, net_expectancy_r=-1.0))
    assert a.state is S.RESEARCH_QUALIFIED and "not enough" in a.reasons[0]


def test_negative_out_of_sample_fails_validation():
    a = assess(strong(), out_of_sample=OutOfSampleResult(n_trades=80, net_expectancy_r=-0.1))
    assert a.state is S.VALIDATION_FAILED and "out-of-sample" in a.reasons[0]


def test_robustness_testing_until_every_required_test_has_run():
    assert assess(strong(), out_of_sample=GOOD_OOS).state is S.ROBUSTNESS_TESTING
    partial = {**ALL_PASS, "cost_stress": None}
    a = assess(strong(), out_of_sample=GOOD_OOS, robustness=partial)
    assert a.state is S.ROBUSTNESS_TESTING and "cost_stress" in a.reasons[0]


def test_a_failed_robustness_test_fails_validation_even_if_others_pass():
    a = assess(strong(), out_of_sample=GOOD_OOS, robustness={**ALL_PASS, "parameter_sensitivity": False})
    assert a.state is S.VALIDATION_FAILED and "parameter_sensitivity" in a.reasons[0]


def test_validated_for_research_needs_everything():
    a = assess(strong(), out_of_sample=GOOD_OOS, robustness=ALL_PASS)
    assert a.state is S.VALIDATED_FOR_RESEARCH and a.state.rankable
    assert not a.failed_checks()


def test_unknown_robustness_test_names_are_rejected():
    with pytest.raises(ValueError):
        assess(strong(), out_of_sample=GOOD_OOS, robustness={"astrology": True})


# ---- properties ---------------------------------------------------------------------------------------

def test_only_passing_states_are_rankable():
    assert {s for s in S if s.rankable} == {S.RESEARCH_QUALIFIED, S.ROBUSTNESS_TESTING, S.VALIDATED_FOR_RESEARCH}


def test_every_state_has_a_definition_and_none_promises_profit():
    assert set(STATE_DEFINITIONS) == set(S)
    for text in STATE_DEFINITIONS.values():
        assert "guarantee" not in text.lower() or "not" in text.lower()
    assert "not a prediction" in STATE_DEFINITIONS[S.VALIDATED_FOR_RESEARCH].lower()


def test_assessment_is_deterministic_and_records_its_policy_and_checks():
    a, b = assess(strong()), assess(strong())
    assert a.as_dict() == b.as_dict()
    d = a.as_dict()
    assert d["policy"]["min_trades"] == SETTINGS.evidence.min_trades and "bootstrap_seed" not in d["policy"]
    assert {c["name"] for c in d["checks"]} >= {"data_quality", "trades", "effective_trades", "net_expectancy_ci_lower", "profit_factor"}
    assert d["definition"] == STATE_DEFINITIONS[S.RESEARCH_QUALIFIED]


def test_policy_thresholds_are_what_decide_the_outcome():
    rs = strong(80)                      # 80 trades: below the default minimum of 100
    assert assess(rs).state is S.INSUFFICIENT_SAMPLE
    relaxed = dataclasses.replace(SETTINGS.evidence, min_trades=50, min_effective_trades=30, min_trade_days=30)
    assert assess(rs, policy=relaxed).state is not S.INSUFFICIENT_SAMPLE


def test_a_better_backtest_never_ranks_below_a_worse_one_at_the_gates():
    order = [S.INSUFFICIENT_DATA, S.INSUFFICIENT_SAMPLE, S.VALIDATION_FAILED, S.RESEARCH_QUALIFIED,
             S.ROBUSTNESS_TESTING, S.VALIDATED_FOR_RESEARCH]
    ladder = [
        assess(strong(), backtestable=False).state,
        assess(strong(30)).state,
        assess(edge_series(400, mean=-0.2)).state,
        assess(strong()).state,
        assess(strong(), out_of_sample=GOOD_OOS).state,
        assess(strong(), out_of_sample=GOOD_OOS, robustness=ALL_PASS).state,
    ]
    assert ladder == order
