"""Gated ranker: evidence gates cannot be bought back by a score, ranks are tiered, fields stay separate, risk keeps the last word."""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ranking import gated_ranker as gr
from quant_intelligence.ranking.ranking_engine import score_strategy
from quant_intelligence.validation.evidence import EvidenceState as S
from quant_intelligence.validation.evidence_store import EvidenceSnapshot

NOW = dt.datetime(2026, 10, 20, 12, 0)
POLICY = SETTINGS.gated
OOS_MIN = SETTINGS.evidence.min_oos_trades


def good_summary(**over):
    s = {"state": S.ROBUSTNESS_TESTING.value, "ci_low": 0.05, "net_expectancy_r": 0.12, "max_drawdown_r": -9.0,
         "oos": {"n_trades": OOS_MIN + 20, "net_expectancy_r": 0.1, "ci_low": 0.02},
         "walk_forward": {"share_positive": 0.8, "n_judged": 6},
         "robustness": {"parameter_sensitivity": True, "regime_sensitivity": True, "cost_stress": True,
                        "period_consistency": None, "sequence_analysis": True},
         "param_positive_share": 0.8, "slippage_breakeven": 3.5}
    s.update(over)
    return s


def snap(name, state=S.ROBUSTNESS_TESTING, age_days=1, version="1.0", **over):
    return EvidenceSnapshot(name, version, state, NOW - dt.timedelta(days=age_days), "cfg", good_summary(**over))


def strat(name, version="1.0"):
    return SimpleNamespace(name=name, version=version)


def legacy(name, mean=0.3, se=0.05):
    return score_strategy({"strategy_name": name, "confidence_label": "HIGH", "sample_size": 30, "n_days": 20, "expected_r": mean,
                           "prob_positive_return": 0.55, "expected_r_stderr": se, "global_expected_r": 0.0,
                           "stability_folds": 0, "stability_positive_frac": None}, -2.0)


def waiting(*a, **k):
    return SimpleNamespace(status="WAITING_FOR_SETUP")


def run(strategies, scores, evidence, regime_stats=None, dq="OK", regime="RANGE", setup_fn=waiting, policy=None):
    return gr.gated_rank(strategies, scores, "NIFTY", "5min", dq, regime, evidence=evidence, regime_stats=regime_stats or {},
                         policy=policy, now=NOW, setup_fn=setup_fn)


# ---- stage 1: gates -----------------------------------------------------------------------------------------------

def test_a_high_score_with_no_evidence_is_not_selected():
    r = run([strat("A")], [legacy("A", mean=0.9)], evidence={})
    assert r.legacy.selected_strategy == "A"                     # the ungated ranker would have traded it
    assert r.ranking.is_no_trade and r.ranking.selected_strategy is None
    d = r.decision_for("A")
    assert not d.eligible and d.research_status == gr.NO_EVIDENCE and d.final_status == gr.NOT_ELIGIBLE
    assert "evidence" in r.ranking.reason.lower()


@pytest.mark.parametrize("label,kwargs,gate", [
    ("lower bound at zero", dict(ci_low=0.0), "net_edge"),
    ("negative lower bound", dict(ci_low=-0.04), "net_edge"),
    ("negative OOS", dict(oos={"n_trades": OOS_MIN + 5, "net_expectancy_r": -0.02, "ci_low": -0.2}), "out_of_sample"),
    ("thin OOS", dict(oos={"n_trades": OOS_MIN - 1, "net_expectancy_r": 0.3, "ci_low": 0.1}), "out_of_sample"),
    ("no OOS", dict(oos=None), "out_of_sample"),
    ("drawdown over limit", dict(max_drawdown_r=-SETTINGS.evidence.max_drawdown_r - 1), "drawdown"),
    ("a robustness test failed", dict(robustness={"cost_stress": False, "sequence_analysis": True}), "robustness"),
])
def test_each_failed_gate_blocks_however_good_the_score(label, kwargs, gate):
    r = run([strat("A")], [legacy("A", mean=0.9)], {"A": snap("A", **kwargs)})
    d = r.decision_for("A")
    assert not d.eligible and gate in [g.name for g in d.failed_gates], label
    assert r.ranking.is_no_trade


def test_an_unvalidated_state_blocks():
    for state in (S.VALIDATION_FAILED, S.INSUFFICIENT_DATA, S.INSUFFICIENT_SAMPLE):
        r = run([strat("A")], [legacy("A")], {"A": snap("A", state=state)})
        assert "evidence_state" in [g.name for g in r.decision_for("A").failed_gates]


def test_stale_evidence_and_a_changed_strategy_version_do_not_count():
    stale = run([strat("A")], [legacy("A")], {"A": snap("A", age_days=POLICY.max_evidence_age_days + 1)})
    assert "evidence_current" in [g.name for g in stale.decision_for("A").failed_gates]
    changed = run([strat("A", version="2.0")], [legacy("A")], {"A": snap("A", version="1.0")})
    assert "evidence_current" in [g.name for g in changed.decision_for("A").failed_gates]


def test_degraded_live_data_blocks_everything():
    r = run([strat("A")], [legacy("A")], {"A": snap("A")}, dq="DEGRADED")
    assert r.ranking.is_no_trade and "Data quality is DEGRADED" in r.ranking.reason


def test_min_state_can_demand_full_validation():
    pol = type(POLICY)(min_state=S.VALIDATED_FOR_RESEARCH.value)
    assert run([strat("A")], [legacy("A")], {"A": snap("A", state=S.ROBUSTNESS_TESTING)}, policy=pol).ranking.is_no_trade
    ok = run([strat("A")], [legacy("A")], {"A": snap("A", state=S.VALIDATED_FOR_RESEARCH, robustness={k: True for k in (
        "parameter_sensitivity", "regime_sensitivity", "cost_stress", "period_consistency", "sequence_analysis")})}, policy=pol)
    assert ok.ranking.selected_strategy == "A"


# ---- stage 2: research rank ---------------------------------------------------------------------------------------

def test_research_rank_beats_the_legacy_score():
    ev = {"A": snap("A", state=S.ROBUSTNESS_TESTING), "B": snap("B", state=S.VALIDATED_FOR_RESEARCH)}
    r = run([strat("A"), strat("B")], [legacy("A", mean=0.9), legacy("B", mean=0.2)], ev)
    assert r.legacy.selected_strategy == "A" and r.ranking.selected_strategy == "B"
    assert r.decision_for("B").research_rank == 1 and r.decision_for("A").research_rank == 2
    assert r.decision_for("A").final_status == gr.NOT_SELECTED


def test_one_flattering_metric_cannot_buy_back_a_lower_tier():
    # A has a huge out-of-sample bound but a lower evidence state; the state tier is compared first
    ev = {"A": snap("A", state=S.ROBUSTNESS_TESTING, oos={"n_trades": 90, "net_expectancy_r": 0.9, "ci_low": 0.8}),
          "B": snap("B", state=S.VALIDATED_FOR_RESEARCH, oos={"n_trades": 90, "net_expectancy_r": 0.05, "ci_low": 0.001})}
    r = run([strat("A"), strat("B")], [legacy("A"), legacy("B")], ev)
    assert r.decision_for("B").research_rank < r.decision_for("A").research_rank


def test_within_a_tier_the_out_of_sample_lower_bound_orders():
    ev = {"A": snap("A", oos={"n_trades": 90, "net_expectancy_r": 0.1, "ci_low": 0.06}),
          "B": snap("B", oos={"n_trades": 90, "net_expectancy_r": 0.1, "ci_low": 0.09})}
    r = run([strat("A"), strat("B")], [legacy("A"), legacy("B")], ev)
    assert r.decision_for("A").tier == r.decision_for("B").tier        # same bands
    assert r.decision_for("B").research_rank == 1


def test_a_tie_inside_the_top_tier_goes_to_the_legacy_ranker_and_keeps_its_tie_break_flag():
    ev = {"A": snap("A"), "B": snap("B")}
    tied = [legacy("A", mean=0.30, se=0.2), legacy("B", mean=0.25, se=0.2)]
    r = run([strat("A"), strat("B")], tied, ev)
    assert r.ranking.selected_strategy == "A" and r.ranking.tie_break and r.ranking.tie_runner_up == "B"


# ---- stage 3: compatibility ---------------------------------------------------------------------------------------

def stats(strategy_rows):
    return {k: {lab: {"n_trades": n, "net_expectancy_r": e} for lab, (n, e) in rows.items()} for k, rows in strategy_rows.items()}


def test_a_regime_the_strategy_loses_in_makes_it_incompatible_and_the_next_is_chosen():
    ev = {"A": snap("A", state=S.VALIDATED_FOR_RESEARCH), "B": snap("B")}
    rs = stats({"A": {"RANGE": (80, -0.2)}, "B": {"RANGE": (80, 0.15)}})
    r = run([strat("A"), strat("B")], [legacy("A"), legacy("B", mean=0.6)], ev, rs, regime="RANGE")
    assert r.decision_for("A").compatibility == gr.INCOMPATIBLE and r.decision_for("A").final_status == gr.INCOMPATIBLE_NOW
    assert r.ranking.selected_strategy == "B" and r.decision_for("B").compatibility == gr.COMPATIBLE


def test_unknown_regime_history_is_flagged_not_blocking_by_default():
    r = run([strat("A")], [legacy("A")], {"A": snap("A")}, {"A": {"RANGE": {"n_trades": 3, "net_expectancy_r": -1.0}}})
    d = r.decision_for("A")
    assert d.compatibility == gr.UNKNOWN and r.ranking.selected_strategy == "A" and "no evidence either way" in d.compatibility_detail


def test_unknown_regime_history_blocks_when_required():
    pol = type(POLICY)(require_regime_evidence=True)
    r = run([strat("A")], [legacy("A")], {"A": snap("A")}, {}, policy=pol)
    assert r.ranking.is_no_trade and r.decision_for("A").compatibility == gr.INCOMPATIBLE


def test_current_conditions_still_apply_through_the_legacy_eligibility():
    weak_now = legacy("A", mean=-0.2)                      # legacy says the conditional edge is below the minimum
    r = run([strat("A")], [weak_now], {"A": snap("A")}, stats({"A": {"RANGE": (80, 0.2)}}))
    assert not weak_now.eligible and r.decision_for("A").compatibility == gr.INCOMPATIBLE
    assert r.ranking.is_no_trade and "current conditions" in r.decision_for("A").compatibility_detail


# ---- stages 4-5: setup and final status -----------------------------------------------------------------------------

def test_the_best_final_status_is_pending_risk_never_approved():
    trig = lambda *a, **k: SimpleNamespace(status="SETUP_TRIGGERED")  # noqa: E731
    r = run([strat("A")], [legacy("A")], {"A": snap("A")}, setup_fn=trig)
    assert r.decision_for("A").final_status == gr.SETUP_VALID_PENDING_RISK
    w = run([strat("A")], [legacy("A")], {"A": snap("A")}, setup_fn=waiting)
    assert w.decision_for("A").final_status == gr.WAITING_FOR_SETUP


def test_research_compatibility_setup_and_final_are_separate_fields():
    r = run([strat("A")], [legacy("A")], {"A": snap("A", state=S.VALIDATED_FOR_RESEARCH)}, stats({"A": {"RANGE": (80, 0.2)}}),
            setup_fn=lambda *a, **k: SimpleNamespace(status="SETUP_TRIGGERED"))
    d = r.decision_for("A")
    assert (d.research_status, d.compatibility, d.setup_status, d.final_status) == (
        "VALIDATED FOR RESEARCH", "COMPATIBLE", "SETUP_TRIGGERED", "SETUP_VALID_PENDING_RISK")
    assert d.research_rank == 1
    text = json.dumps(d.as_dict(), default=str)
    assert "research_status" in text and "final_status" in text


def test_a_failing_setup_check_does_not_take_the_ranking_down():
    def boom(*a, **k):
        raise RuntimeError("bad bar")
    r = run([strat("A")], [legacy("A")], {"A": snap("A")}, setup_fn=boom)
    assert r.ranking.selected_strategy == "A" and r.decision_for("A").setup_status == gr.NOT_ASSESSED


def test_no_trade_reason_names_what_blocked_the_strategies():
    r = run([strat("A"), strat("B")], [legacy("A"), legacy("B")], {"A": snap("A", ci_low=-0.1)})
    assert r.ranking.is_no_trade and "evidence_exists" in r.ranking.reason and "net_edge" in r.ranking.reason


def test_compare_rankings_explains_a_disagreement():
    r = run([strat("A"), strat("B")], [legacy("A", mean=0.9), legacy("B", mean=0.2)], {"B": snap("B")})
    c = gr.compare_rankings(r)
    assert c["legacy"] == "A" and c["gated"] == "B" and not c["agree"] and "evidence" in c["why"]


# ---- the risk engine keeps the last word ----------------------------------------------------------------------------

def test_the_ranker_does_not_import_the_risk_engine():
    import inspect
    imports = [ln for ln in inspect.getsource(gr).splitlines() if ln.lstrip().startswith(("import ", "from "))]
    assert not any("risk" in ln for ln in imports)


def test_a_gated_selection_still_goes_through_the_risk_engine(monkeypatch):
    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.execution import auto_trader
    from quant_intelligence.risk.risk_engine import AccountState

    ranking = run([strat("Momentum")], [legacy("Momentum")], {"Momentum": snap("Momentum")}).ranking
    assert ranking.selected_strategy == "Momentum"
    setup = SimpleNamespace(direction="LONG", entry_price=100.0, stop_price=99.0, target_price=102.0, meta={})
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    output = SimpleNamespace(ranking=ranking, instrument="TESTSTOCK", ohlcv=SimpleNamespace(tail=lambda n: None),
                             market_state=SimpleNamespace(get=lambda k, d=None: None), data_quality_status="OK")
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=True)
    result = auto_trader.run_auto_equity_cycle(output, PaperBroker(starting_capital=1_000_000), account)
    assert "RISK ENGINE VETO" in result.reason and result.order is None


# ---- evidence store and the pipeline ----------------------------------------------------------------------------------

def test_evidence_round_trips_and_the_latest_snapshot_wins(synthetic_ohlcv, synthetic_features):
    from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
    from quant_intelligence.validation.evidence_store import load_latest_evidence
    from quant_intelligence.validation.suite import run_validation_suite

    rv = run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "STORETEST", quantity=75, persist=True)
    run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, "STORETEST", quantity=75, persist=True)
    got = load_latest_evidence("STORETEST", "5min")
    assert list(got) == ["Momentum"] and got["Momentum"].state is rv.state
    assert got["Momentum"].strategy_version == "1.0" and got["Momentum"].summary["n_trades"] == rv.in_sample.metrics.n_trades
    assert "NaN" not in json.dumps(got["Momentum"].summary)
    assert load_latest_evidence("NOSUCHSYMBOL", "5min") == {}


def test_regime_stats_load_from_the_table(synthetic_ohlcv, synthetic_features):
    from quant_intelligence.backtesting.engine import run_backtest
    from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
    from quant_intelligence.validation.robustness import persist_regime_stats, run_robustness

    res = run_backtest(MomentumStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=75)
    rep = run_robustness(MomentumStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", res.trades, 75)
    persist_regime_stats(rep, "Momentum", "REGLOAD", "5min", "run-x", "cfg")
    loaded = gr.load_regime_stats("REGLOAD", "5min")
    assert "Momentum" in loaded and all("n_trades" in v for v in loaded["Momentum"].values())


def test_the_pipeline_is_unchanged_with_the_flag_off_and_gated_with_it_on(monkeypatch, tmp_path):
    import dataclasses

    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter
    from quant_intelligence.research import pipeline

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(DataManager, "_fetch", lambda self, i, tf, s, e, p: ("test", SyntheticAdapter().get_ohlcv(i, tf, s, e)))
    end = dt.datetime(2024, 6, 28, 15, 30)
    start = end - dt.timedelta(days=60)

    off = pipeline.run_pipeline("NIFTY", "5min", start, end)
    assert off.decisions == [] and off.legacy_ranking is None

    monkeypatch.setattr(pipeline, "SETTINGS", dataclasses.replace(pipeline.SETTINGS, gated_ranker=True))
    on = pipeline.run_pipeline("NIFTY", "5min", start, end)
    assert on.legacy_ranking is not None and len(on.decisions) == len(on.strategy_intel)
    assert on.ranking.is_no_trade                          # nothing in the store for this instrument: no evidence, no trade
    assert all(d.final_status == gr.NOT_ELIGIBLE for d in on.decisions)
    assert on.legacy_ranking.selected_strategy == off.ranking.selected_strategy    # the ungated pick is still reported
