"""The suite: development -> validation -> robustness -> (optional) holdout, with the holdout protected."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quant_intelligence.strategies.iv_crush_writer_strategy import IVCrushWriterStrategy
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.validation import suite
from quant_intelligence.validation.evidence import REQUIRED_ROBUSTNESS_TESTS, EvidenceState as S
from quant_intelligence.validation.robustness import RobustnessReport, RobustnessTest
from quant_intelligence.validation.splits import holdout_fingerprint, holdout_status


def test_a_short_history_stops_at_insufficient_data_and_skips_robustness(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75)
    assert r.state is S.INSUFFICIENT_DATA
    assert r.robustness is None and any("robustness tests not run" in n for n in r.notes)
    assert r.out_of_sample is not None and r.walk_forward is not None and len(r.walk_forward.folds) >= 2


def test_segments_are_run_on_their_own_sessions(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75)
    assert r.in_sample.metrics.n_trades > 0
    assert r.split.development[-1] < r.split.validation[0] < r.split.holdout[0]
    assert r.in_sample.evidence.policy and r.strategy == "Momentum" and r.strategy_version == "1.0"


def test_chain_aware_strategies_run_nothing(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(IVCrushWriterStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75)
    assert r.state is S.INSUFFICIENT_DATA and r.out_of_sample is None and r.walk_forward is None and r.robustness is None
    assert any("option-chain" in n for n in r.notes)


def test_robustness_can_be_forced_on(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, robustness="always")
    assert set(r.robustness.tests) == set(REQUIRED_ROBUSTNESS_TESTS)


def test_the_result_serialises_without_nan(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75)
    text = json.dumps(r.as_dict(), default=str)
    assert "NaN" not in text and "Infinity" not in text


def test_the_holdout_is_not_touched_for_a_strategy_that_is_not_validated(synthetic_ohlcv, synthetic_features):
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, include_holdout=True)
    assert r.holdout is None and any("holdout not touched" in n for n in r.notes)
    fp = holdout_fingerprint(synthetic_ohlcv, synthetic_features, r.split, "", "5min", "NIFTY")
    assert holdout_status("NIFTY", "5min", fp).evaluations == 0


# ---- the holdout path, with the verdicts faked so it can be reached on synthetic data -----------------------------

@pytest.fixture()
def validated(monkeypatch):
    """Every gate before the holdout passes: ROBUSTNESS_TESTING until robustness has run, then VALIDATED."""
    seen: list[dict] = []

    def fake_validate(result, ohlcv=None, quantity=50, **kw):
        seen.append({"robustness": kw.get("robustness"), "oos": kw.get("out_of_sample")})
        state = S.ROBUSTNESS_TESTING if kw.get("robustness") is None else S.VALIDATED_FOR_RESEARCH
        return SimpleNamespace(evidence=SimpleNamespace(state=state, policy={}), metrics=None,
                               as_dict=lambda: {"state": state.value})

    outcomes = {n: RobustnessTest(n, True) for n in REQUIRED_ROBUSTNESS_TESTS}
    monkeypatch.setattr(suite, "validate_backtest", fake_validate)
    monkeypatch.setattr(suite, "run_robustness", lambda *a, **k: RobustnessReport(outcomes))
    return seen


def test_a_validated_strategy_is_evaluated_on_the_holdout_and_the_look_is_recorded(synthetic_ohlcv, synthetic_features, validated):
    inst = "HOLDOUTTEST1"
    r = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, inst, quantity=75, include_holdout=True)
    assert r.holdout is not None and r.holdout["status"]["evaluations"] == 1 and not r.holdout["status"]["consumed"]
    assert validated[-1]["oos"].n_trades == r.holdout["n_trades"]          # the final verdict used the holdout as its OOS
    assert any("final out-of-sample result is the holdout" in n for n in r.notes)


def test_looking_at_the_holdout_again_marks_it_consumed(synthetic_ohlcv, synthetic_features, validated):
    inst = "HOLDOUTTEST2"
    first = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, inst, quantity=75, include_holdout=True)
    again = suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, inst, quantity=75, include_holdout=True)
    assert not first.holdout["status"]["consumed"]
    assert again.holdout["status"]["consumed"] and again.holdout["status"]["repeat_looks"] == 1
    assert any("no longer untouched" in n for n in again.notes)


def test_robustness_only_runs_after_the_earlier_gates_pass(synthetic_ohlcv, synthetic_features, validated, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "run_robustness", lambda *a, **k: calls.append(1) or RobustnessReport({}))
    suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, robustness="auto")
    assert len(calls) == 1
    calls.clear()
    suite.run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, robustness="never")
    assert calls == []
