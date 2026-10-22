"""Research run report: the evidence is in the report, and the wording stays on the safe side."""
from __future__ import annotations

import json

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import ResearchReport
from quant_intelligence.reports import research_report as rr

DATA = {"fingerprint": "ab" * 32, "start": "2024-10-21", "end": "2026-10-19", "n_sessions": 494, "quality": "DEGRADED",
        "issues": ["one zero-volume run"]}


def outcome(**over):
    o = {"strategy": "Momentum", "outcome": "COMPLETED", "detail": "", "state": "VALIDATION FAILED",
         "decision": {"accepted": True, "kind": "NEW", "reason": "first verdict"}, "reasons": ["net expectancy is not distinguishable from 0"],
         "metrics": {"n_trades": 407, "n_trade_days": 210, "net_expectancy_r": 0.055, "profit_factor": 1.04, "max_drawdown_r": -25.0},
         "effective_n": 407.0, "ci_low": -0.082, "ci_high": 0.195,
         "oos": {"n_trades": 80, "net_expectancy_r": 0.02, "ci_low": -0.1},
         "walk_forward": {"share_positive": 0.5, "n_judged": 6, "worst_fold_r": -0.2},
         "robustness": {"cost_stress": True, "sequence_analysis": False, "period_consistency": None}, "proxy_data": False}
    o.update(over)
    return o


def report(outcomes=None, **kw):
    return rr.build_run_report("NIFTY", "5min", "RS-1", DATA, outcomes or [outcome()], kw.get("version"), kw.get("changes", []))


def test_the_report_carries_the_numbers_behind_the_verdict():
    md = report()["content_markdown"]
    for needle in ("Momentum", "VALIDATION FAILED", "407 trades", "+0.055 R", "[-0.082, +0.195]", "profit factor 1.04", "max drawdown -25.0 R",
                   "Out of sample: 80 trades", "Walk-forward: 50% of 6 folds", "cost_stress passed", "sequence_analysis FAILED",
                   "period_consistency not run", "first verdict", "DEGRADED", "one zero-volume run"):
        assert needle in md, needle


def test_it_states_what_was_decided_and_what_was_held():
    held = outcome(decision={"accepted": False, "kind": "PROMOTION_HELD", "reason": "needs 20 sessions of new data"})
    md = report([held])["content_markdown"]
    assert "NOT accepted" in md and "needs 20 sessions" in md


def test_limitations_come_from_the_strategy_rules():
    md = report([outcome(strategy="PCR Extreme Contrarian", state="INSUFFICIENT DATA", metrics=None, oos=None, walk_forward=None, robustness=None)])["content_markdown"]
    assert "Limitation:" in md and "option-chain" in md


def test_status_changes_and_the_version_are_listed():
    md = report(changes=[{"strategy": "Momentum", "kind": "DEMOTION", "from_state": "ROBUSTNESS TESTING", "to_state": "VALIDATION FAILED",
                          "reason": "fell"}], version={"version_no": 4, "leaderboard": [{}] * 17})["content_markdown"]
    assert "DEMOTION" in md and "ROBUSTNESS TESTING -> VALIDATION FAILED" in md and "Accepted version 4" in md and "17 strategies" in md


def test_failed_and_deferred_runs_are_reported_plainly():
    md = report([outcome(outcome="FAILED", detail="RuntimeError: boom", state=None, decision=None, metrics=None, oos=None,
                         walk_forward=None, robustness=None, reasons=[])])["content_markdown"]
    assert "Run: FAILED - RuntimeError: boom" in md


def test_the_wording_never_promises_profit():
    md = report()["content_markdown"].lower()
    assert "not a prediction" in md and "not investment advice" in md
    assert "guarantee" not in md and "will be profitable" not in md


def test_json_is_clean_and_complete():
    content = report()["content_json"]
    text = json.dumps(content, default=str)
    assert "NaN" not in text and content["strategies"][0]["definition"] and content["disclaimer"]
    assert content["data"]["fingerprint"] == DATA["fingerprint"]


def test_it_persists_as_a_research_run_report():
    rid = rr.persist_run_report("RPTPERSIST", report())
    assert rid is not None
    with get_session() as s:
        row = s.get(ResearchReport, rid)
        assert row.report_type == "RESEARCH_RUN" and row.instrument == "RPTPERSIST" and "Momentum" in row.content_markdown
