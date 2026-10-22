"""Strategy Evidence dashboard: the stored detail block, the data behind each section, and the page itself."""
from __future__ import annotations

import datetime as dt
import json
import threading
from types import SimpleNamespace

import pytest

from quant_intelligence.research import dashboard_data as dd
from quant_intelligence.research import records
from quant_intelligence.research.scheduler import DataUnavailable, ResearchScheduler
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.tests.test_gated_ranker import NOW, legacy, run as gated_run, snap, strat
from quant_intelligence.validation.detail import MAX_POINTS, MAX_TRADES
from quant_intelligence.validation.evidence import REQUIRED_ROBUSTNESS_TESTS, EvidenceState as S
from quant_intelligence.validation.evidence_store import load_latest_evidence
from quant_intelligence.validation.suite import run_validation_suite

INST, TF = "DASHTEST", "5min"


@pytest.fixture(scope="module")
def stored(synthetic_ohlcv, synthetic_features):
    """A real verdict (robustness forced on) stored for DASHTEST."""
    return run_validation_suite(MomentumStrategy, synthetic_ohlcv, synthetic_features, INST, quantity=75, robustness="always", persist=True)


# ---- the detail block ---------------------------------------------------------------------------------------------

def test_the_detail_block_is_complete_compact_and_json_safe(stored):
    d = stored.detail
    assert d["n_trades"] == stored.in_sample.metrics.n_trades + stored.out_of_sample.n_trades
    assert 0 < len(d["equity"]) <= MAX_POINTS and len(d["trades"]) <= MAX_TRADES
    assert {row[3] for row in d["equity"]} <= {"development", "validation"}
    assert sum(d["histogram"]["counts"]) == d["n_trades"]
    assert {r["dimension"] for r in d["regimes"]} >= {"regime", "volatility"}
    assert set(d["robustness"]) == set(REQUIRED_ROBUSTNESS_TESTS)
    assert "NaN" not in json.dumps(d) and "Infinity" not in json.dumps(d)


def test_the_stored_equity_curve_ends_at_the_sum_of_net_r(stored):
    last = stored.detail["equity"][-1][1]
    total = sum(t["net_r"] for t in stored.detail["trades"])
    if stored.detail["n_trades"] <= MAX_TRADES:
        assert last == pytest.approx(total)
    drawdowns = [row[2] for row in stored.detail["equity"]]
    assert max(drawdowns) <= 1e-9                                         # never above the running peak


def test_the_stored_snapshot_carries_detail_folds_and_the_policy(stored):
    snap_ = load_latest_evidence(INST, TF)["Momentum"]
    s = snap_.summary
    assert s["detail"]["n_trades"] == stored.detail["n_trades"] and s["walk_forward"]["folds"] and s["policy_hash"] and s["data_end"]
    assert "NaN" not in json.dumps(s)


# ---- research detail -----------------------------------------------------------------------------------------------

def test_strategy_detail_builds_every_frame(stored):
    d = dd.strategy_detail(load_latest_evidence(INST, TF)["Momentum"])
    assert d["has_detail"] and d["equity"] is not None and list(d["equity"].columns) == ["exit_ts", "equity_r", "drawdown_r", "segment"]
    assert d["histogram"] is not None and d["periods"] is not None and d["regimes"] is not None and d["trades"] is not None
    assert d["walk_forward"] is not None and [t["test"] for t in d["robustness_tests"]] == list(REQUIRED_ROBUSTNESS_TESTS)
    assert d["limitations"] and "entry" in d["rules"]
    assert d["parameters"] is None or {"parameter", "value", "net_expectancy_r"} <= set(d["parameters"].columns)
    assert d["cost_stress"] is not None and {"slippage_multiple", "net_expectancy_r"} <= set(d["cost_stress"].columns)


def test_missing_detail_is_reported_not_invented():
    d = dd.strategy_detail(snap("Momentum"))                              # a snapshot stored without a detail block
    assert not d["has_detail"] and d["equity"] is None and d["histogram"] is None and d["trades"] is None and d["parameters"] is None
    assert all(t["result"] == "passed" or t["result"] in ("not run", "FAILED") for t in d["robustness_tests"])


# ---- leaderboard ---------------------------------------------------------------------------------------------------

UNREGISTERED = "0.0"      # what version_for() reports for a name that is not a registered strategy


def board_evidence():
    kw = dict(version=UNREGISTERED)
    return {"A": snap("A", state=S.ROBUSTNESS_TESTING, **kw), "B": snap("B", state=S.VALIDATED_FOR_RESEARCH, **kw),
            "C": snap("C", ci_low=-0.1, **kw), "D": snap("D", state=S.VALIDATION_FAILED, **kw)}


def test_eligible_strategies_come_first_in_research_rank_order():
    df = dd.leaderboard_frame("NIFTY", TF, board_evidence(), now=NOW)
    assert list(df.columns) == dd.LEADERBOARD_COLUMNS and len(df) == 4
    assert list(df["Strategy"][:2]) == ["B", "A"] and list(df["Rank"][:2]) == [1, 2] and df["Rank"][2:].isna().all()
    assert list(df["Eligible"]) == [True, True, False, False]
    assert (df["Regime fit"] == "-").all() and (df["Final status"] == "-").all()      # needs the live market


def test_the_live_columns_fill_in_from_a_ranking():
    ev = board_evidence()
    r = gated_run([strat(n, UNREGISTERED) for n in "ABCD"], [legacy(n) for n in "ABCD"], ev)
    df = dd.leaderboard_frame("NIFTY", TF, ev, ranking=r, now=NOW).set_index("Strategy")
    assert df.loc["B", "Final status"] == "WAITING_FOR_SETUP" and df.loc["C", "Final status"] == "NOT_ELIGIBLE"
    assert df.loc["B", "Regime fit"] in ("COMPATIBLE", "UNKNOWN")


def test_no_evidence_gives_an_empty_frame():
    df = dd.leaderboard_frame("NOEVIDENCE", TF, {})
    assert len(df) == 0 and list(df.columns) == dd.LEADERBOARD_COLUMNS


def test_the_sorting_banner_disclaims_performance():
    assert "not a performance guarantee" in dd.SORT_WARNING


def test_robustness_summary_wording():
    assert dd.robustness_summary(None) == "not run" and dd.robustness_summary({}) == "not run"
    done = {n: True for n in REQUIRED_ROBUSTNESS_TESTS}
    assert dd.robustness_summary(done) == "5 passed"
    done.update(cost_stress=False, sequence_analysis=None)
    assert dd.robustness_summary(done) == "3 passed, 1 FAILED, 1 not run"


def test_stored_targets_lists_what_has_evidence(stored):
    assert (INST, TF) in dd.stored_targets()


# ---- ranking explanation -------------------------------------------------------------------------------------------

def test_the_explanation_names_the_neighbours_gates_and_uncertainty():
    ev = {"A": snap("A", state=S.VALIDATED_FOR_RESEARCH), "B": snap("B", oos={"n_trades": 40, "net_expectancy_r": 0.1, "ci_low": 0.05},
                                                                    robustness={"cost_stress": True, "sequence_analysis": None}, proxy_data=True),
          "C": snap("C", state=S.ROBUSTNESS_TESTING, oos={"n_trades": 90, "net_expectancy_r": 0.1, "ci_low": 0.01}),
          "X": snap("X", ci_low=-0.2)}
    r = gated_run([strat(n) for n in "ABCX"], [legacy(n) for n in "ABCX"], ev)
    ex = dd.explain_rank(r, ev, "B", now=NOW)
    assert ex["found"] and ex["n_eligible"] == 3 and ex["eligible_position"] == 1
    assert ex["above"]["strategy"] == "A" and ex["below"]["strategy"] == "C" and not ex["failed_gates"]
    text = " ".join(ex["uncertainty"])
    assert "only 40 out-of-sample trades" in text and "proxy data" in text and "sequence_analysis" in text

    bad = dd.explain_rank(r, ev, "X", now=NOW)
    assert bad["eligible_position"] is None and any(g["name"] == "net_edge" for g in bad["failed_gates"]) and bad["final_status"] == "NOT_ELIGIBLE"
    assert dd.explain_rank(r, ev, "nope")["found"] is False


def test_the_first_ranked_strategy_has_nothing_above():
    ev = {"A": snap("A", state=S.VALIDATED_FOR_RESEARCH), "B": snap("B")}
    r = gated_run([strat("A"), strat("B")], [legacy("A"), legacy("B")], ev)
    ex = dd.explain_rank(r, ev, "A", now=NOW)
    assert ex["above"] is None and ex["below"]["strategy"] == "B" and ex["selected"]


def test_current_ranking_works_from_a_pipeline_output(synthetic_ohlcv, synthetic_features):
    from quant_intelligence.market_state.market_state_engine import build_current_state
    from quant_intelligence.strategies.registry import STRATEGY_CLASSES

    names = list(STRATEGY_CLASSES)[:4]
    out = SimpleNamespace(strategy_intel=[SimpleNamespace(strategy_name=n, score=legacy(n)) for n in names], instrument="NIFTY",
                          data_quality_status="OK", regime_label="RANGE", ohlcv=synthetic_ohlcv,
                          market_state=build_current_state("NIFTY", synthetic_features, "OK"))
    r = dd.current_ranking(out, TF, evidence={})
    assert {d.strategy for d in r.decisions} == set(names) and r.ranking.is_no_trade       # no evidence, no trade


# ---- control center ------------------------------------------------------------------------------------------------

def test_the_control_center_summarises_runs_failures_and_held_promotions():
    inst = "CCTEST"
    done = dt.datetime(2026, 10, 20, 9, 0)
    records.log_run(batch_id="B1", instrument=inst, timeframe=TF, strategy="Momentum", strategy_version="1.0", outcome="COMPLETED",
                    state="VALIDATION FAILED", decision="NEW", accepted=True, runtime_sec=12.5, finished_at=done)
    records.log_run(batch_id="B1", instrument=inst, timeframe=TF, strategy="Donchian Channel Breakout", outcome="FAILED", detail="RuntimeError: x", runtime_sec=3.0)
    records.log_run(batch_id="B2", instrument=inst, timeframe=TF, strategy="*", outcome="AWAITING_DATA", detail="no cached history")
    records.log_run(batch_id="B3", instrument=inst, timeframe=TF, strategy="Momentum", outcome="COMPLETED", state="RESEARCH QUALIFIED",
                    decision="PROMOTION_HELD", accepted=False, runtime_sec=7.0, finished_at=done)
    cc = dd.control_center(inst, TF, coverage={"available": False, "reason": "test"})
    assert cc["counts"] == {"COMPLETED": 2, "FAILED": 1, "AWAITING_DATA": 1} and cc["total_runs"] == 4
    assert cc["last_success"]["strategy"] == "Momentum" and cc["latest_batch"]["batch_id"] == "B3" and cc["latest_batch"]["runtime_sec"] == 7.0
    assert [r["detail"] for r in cc["failed"]] == ["RuntimeError: x"] and cc["awaiting_data"][0]["detail"] == "no cached history"
    assert [r["decision"] for r in cc["held"]] == ["PROMOTION_HELD"] and cc["version"] is None
    assert len(dd.runs_frame(cc["runs"])) == 4


def test_the_control_center_with_no_history_is_empty_not_broken():
    cc = dd.control_center("NEVERRAN", TF, coverage={"available": False, "reason": "none"})
    assert cc["counts"] == {} and cc["last_success"] is None and cc["latest_batch"] is None and cc["version"] is None


def test_coverage_for_an_instrument_with_no_cache_says_so():
    assert dd.data_coverage("NOCACHE", TF) == {"available": False, "reason": "no cached history"}


# ---- a manual batch on its own thread -------------------------------------------------------------------------------

def test_a_manual_batch_runs_in_the_background_once_at_a_time():
    gate = threading.Event()

    def slow_loader(instrument, timeframe):
        gate.wait(5)
        raise DataUnavailable("none")

    s = ResearchScheduler(targets=[("BGTEST", TF)], loader=slow_loader, strategies={"Momentum": MomentumStrategy},
                          now_fn=lambda: dt.datetime(2026, 10, 17, 12, 0))
    assert s.run_in_background([("BGTEST", TF)]) is True
    assert s.batch_running and s.status()["batch_running"] and s.status()["current"]["targets"] == [("BGTEST", TF)]
    assert s.run_in_background([("BGTEST", TF)]) is False                  # one at a time
    gate.set()
    s._batch_thread.join(timeout=10)
    assert not s.batch_running and s.current is None
    assert [r.status for r in s.last_results] == ["AWAITING_DATA"]


# ---- the page -------------------------------------------------------------------------------------------------------

def page():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parent.parent / "pages" / "17_Strategy_Evidence.py"
    return AppTest.from_file(str(path), default_timeout=120)


def test_the_page_renders_with_no_evidence_for_the_default_instrument():
    at = page()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [t.label for t in at.tabs] == ["Leaderboard", "Research detail", "Backtest control center", "Ranking explanation"]


def test_the_page_renders_a_stored_verdict_in_every_section(stored):
    at = page()
    at.run()
    at.selectbox[0].select((INST, TF)).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.dataframe) >= 2
    text = " ".join(m.value for m in at.markdown)
    assert "not a performance guarantee" in " ".join(w.value for w in at.warning)
    assert "Momentum" in text or any("Momentum" in str(df.value) for df in at.dataframe)
    assert any("Equity and drawdown" in m.value for m in at.markdown)
