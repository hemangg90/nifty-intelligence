"""Research scheduler: when it runs, what it accepts, and that every change leaves a trail."""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json

import pandas as pd
import pytest

from quant_intelligence.backtesting.timing import timing_override
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import ResearchReport
from quant_intelligence.research import records, scheduler as sch
from quant_intelligence.research.scheduler import DataUnavailable, ResearchScheduler, market_is_open
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.strategies.orb_strategy import ORBStrategy
from quant_intelligence.validation.evidence import EvidenceState as S
from quant_intelligence.validation.evidence_store import load_latest_evidence
from quant_intelligence.validation.suite import run_validation_suite

TF = "5min"
SATURDAY_NOON = dt.datetime(2026, 10, 17, 12, 0)     # the market is closed
TUESDAY_NOON = dt.datetime(2026, 10, 20, 12, 0)      # NSE is open


@pytest.fixture(autouse=True)
def _quantity(monkeypatch):
    from quant_intelligence.research import pipeline

    monkeypatch.setattr(pipeline, "default_quantity", lambda instrument, fallback=50: 75)


def sessions(df):
    return sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())


def loader_for(ohlcv, features, n_sessions=None):
    """A loader that serves the first n sessions (or all), as if the cache had grown to that point."""
    def load(instrument, timeframe):
        if n_sessions is None:
            return ohlcv, features
        cut = sessions(ohlcv)[n_sessions - 1]
        mask = (pd.to_datetime(ohlcv["timestamp"]).dt.date <= cut).to_numpy()
        return ohlcv[mask].reset_index(drop=True), features[mask].reset_index(drop=True)
    return load


def make(instrument_loader, strategies=None, suite=None, now=SATURDAY_NOON, policy=None, targets=None, **kw):
    s = ResearchScheduler(targets=targets or [("SENSEX", TF)], loader=instrument_loader, suite=suite,
                          strategies=strategies or {"Momentum": MomentumStrategy}, now_fn=lambda: now, **kw)
    if policy is not None:
        s.policy = policy
    return s


def one_batch(s, target, **kw):
    (res,) = s.run_once([target], **kw)
    return res


def state_forcing_suite(states):
    """The real suite, with the evidence state replaced by the next one in `states` (a test seam for governance rules)."""
    queue = list(states)

    def suite(cls, ohlcv, features, instrument, **kw):
        rv = run_validation_suite(cls, ohlcv, features, instrument, **{k: v for k, v in kw.items() if k != "robustness"}, robustness="never")
        rv = copy.deepcopy(rv)
        rv.in_sample.evidence.state = queue.pop(0)
        return rv
    return suite


# ---- data and timing ----------------------------------------------------------------------------------------------

def test_missing_history_is_awaiting_data_not_an_error():
    def load(i, t):
        raise DataUnavailable("no cached history")
    res = one_batch(make(load), ("AWAITING1", TF))
    assert res.status == "AWAITING_DATA" and "no cached history" in res.detail
    rows = [r for r in records.run_history("AWAITING1", TF)]
    assert [r["outcome"] for r in rows] == ["AWAITING_DATA"] and rows[0]["strategy"] == "*"
    assert load_latest_evidence("AWAITING1", TF) == {}


def test_history_that_fails_its_quality_gate_runs_nothing(synthetic_ohlcv, synthetic_features):
    tiny = loader_for(synthetic_ohlcv.iloc[:30], synthetic_features.iloc[:30])
    res = one_batch(make(tiny), ("AWAITING2", TF))
    assert res.status == "AWAITING_DATA" and "quality gate" in res.detail and not res.outcomes


def test_the_batch_waits_while_the_market_is_open_unless_forced(synthetic_ohlcv, synthetic_features):
    s = make(loader_for(synthetic_ohlcv, synthetic_features), now=TUESDAY_NOON)
    assert market_is_open("SENSEX", TUESDAY_NOON) and not market_is_open("SENSEX", SATURDAY_NOON)
    res = one_batch(s, ("DEFER1", TF))
    assert res.status == "DEFERRED" and records.run_history("DEFER1", TF) == []
    assert one_batch(s, ("DEFER1", TF), force=True).status == "RAN"


# ---- a full batch --------------------------------------------------------------------------------------------------

def test_a_first_batch_runs_every_strategy_and_leaves_a_full_trail(synthetic_ohlcv, synthetic_features):
    s = make(loader_for(synthetic_ohlcv, synthetic_features), strategies={"Momentum": MomentumStrategy, "Opening Range Breakout": ORBStrategy})
    res = one_batch(s, ("FIRSTRUN", TF))
    assert res.status == "RAN" and {o.strategy for o in res.outcomes} == {"Momentum", "Opening Range Breakout"}
    assert all(o.outcome == "COMPLETED" and o.decision.accepted and o.decision.kind == "NEW" for o in res.outcomes)

    stored = load_latest_evidence("FIRSTRUN", TF)
    assert set(stored) == {"Momentum", "Opening Range Breakout"}
    assert all(sn.summary["policy_hash"] and sn.summary["data_end"] for sn in stored.values())

    v = records.latest_version("FIRSTRUN", TF)
    assert v["version_no"] == 1 and {r["strategy"] for r in v["leaderboard"]} == set(stored) and res.version_no == 1
    changes = records.status_history("FIRSTRUN", TF)
    assert {c["kind"] for c in changes} == {"NEW"} and len(changes) == 2 and all(c["research_version_id"] == v["id"] for c in changes)

    runs = records.run_history("FIRSTRUN", TF)
    assert len(runs) == 2 and all(r["outcome"] == "COMPLETED" and r["accepted"] and r["dataset_fingerprint"] and r["runtime_sec"] >= 0 for r in runs)

    with get_session() as db:
        report = db.query(ResearchReport).filter_by(instrument="FIRSTRUN", report_type="RESEARCH_RUN").one()
        assert "Momentum" in report.content_markdown and "not a prediction" in report.content_markdown
        assert report.id == res.report_id and "NaN" not in json.dumps(report.content_json)


def test_research_always_uses_next_open_fills(synthetic_ohlcv, synthetic_features):
    seen = []

    def spy(cls, ohlcv, features, instrument, **kw):
        seen.append((timing_override(), kw.get("include_holdout"), kw.get("persist")))
        return run_validation_suite(cls, ohlcv, features, instrument, robustness="never")
    one_batch(make(loader_for(synthetic_ohlcv, synthetic_features), suite=spy), ("TIMING1", TF))
    assert seen == [("next_open", False, False)]       # the holdout is never touched and the suite does not persist by itself
    assert timing_override() is None                   # and the override does not leak out of the batch


def test_a_second_batch_on_unchanged_data_does_nothing(synthetic_ohlcv, synthetic_features):
    s = make(loader_for(synthetic_ohlcv, synthetic_features))
    one_batch(s, ("UNCHANGED", TF))
    again = one_batch(s, ("UNCHANGED", TF))
    assert again.status == "UP_TO_DATE" and not again.outcomes
    assert len(records.run_history("UNCHANGED", TF)) == 1 and records.latest_version("UNCHANGED", TF)["version_no"] == 1


def test_force_reruns_everything(synthetic_ohlcv, synthetic_features):
    s = make(loader_for(synthetic_ohlcv, synthetic_features))
    one_batch(s, ("FORCED", TF))
    assert one_batch(s, ("FORCED", TF), force=True).status == "RAN"


def test_enough_new_data_makes_a_strategy_stale(synthetic_ohlcv, synthetic_features):
    n = len(sessions(synthetic_ohlcv))
    first = make(loader_for(synthetic_ohlcv, synthetic_features, n - 9))
    one_batch(first, ("GROWING", TF))
    just_one_more = make(loader_for(synthetic_ohlcv, synthetic_features, n - 8))          # +1 session: below the threshold
    assert one_batch(just_one_more, ("GROWING", TF)).status == "UP_TO_DATE"
    grown = make(loader_for(synthetic_ohlcv, synthetic_features, n))                       # +9 sessions
    assert one_batch(grown, ("GROWING", TF)).status == "RAN"


def test_changed_thresholds_make_everything_stale(synthetic_ohlcv, synthetic_features, monkeypatch):
    s = make(loader_for(synthetic_ohlcv, synthetic_features))
    one_batch(s, ("POLICYCHG", TF))
    monkeypatch.setattr(sch, "policy_hash", lambda: "changed-policy")
    assert one_batch(s, ("POLICYCHG", TF)).status == "RAN"


# ---- governance ----------------------------------------------------------------------------------------------------

def test_demotions_are_accepted_and_promotions_are_earned(synthetic_ohlcv, synthetic_features):
    n = len(sessions(synthetic_ohlcv))
    pol = dataclasses.replace(sch.SETTINGS.research, rerun_after_new_sessions=1, min_new_sessions_for_promotion=5)
    suite = state_forcing_suite([S.VALIDATED_FOR_RESEARCH, S.INSUFFICIENT_SAMPLE, S.VALIDATED_FOR_RESEARCH, S.VALIDATED_FOR_RESEARCH])
    target = ("GOVERN", TF)

    def at(k):
        return make(loader_for(synthetic_ohlcv, synthetic_features, k), suite=suite, policy=pol)

    first = one_batch(at(n - 12), target)
    assert first.outcomes[0].decision.kind == "NEW"
    assert load_latest_evidence(*target)["Momentum"].state is S.VALIDATED_FOR_RESEARCH

    down = one_batch(at(n - 9), target)                                    # worse result, accepted at once
    assert down.outcomes[0].decision.kind == "DEMOTION" and down.outcomes[0].decision.accepted
    assert load_latest_evidence(*target)["Momentum"].state is S.INSUFFICIENT_SAMPLE

    held = one_batch(at(n - 7), target)                                    # better result, only 2 new sessions: held
    d = held.outcomes[0].decision
    assert not d.accepted and d.kind == "PROMOTION_HELD"
    assert load_latest_evidence(*target)["Momentum"].state is S.INSUFFICIENT_SAMPLE       # the accepted verdict stands
    run = records.run_history(*target)[0]
    assert run["accepted"] is False and run["state"] == "VALIDATED FOR RESEARCH" and run["decision"] == "PROMOTION_HELD"

    earned = one_batch(at(n), target)                                      # 7 more sessions since the accepted demotion
    assert earned.outcomes[0].decision.kind == "PROMOTION" and earned.outcomes[0].decision.accepted
    assert load_latest_evidence(*target)["Momentum"].state is S.VALIDATED_FOR_RESEARCH

    kinds = [c["kind"] for c in reversed(records.status_history(*target))]
    assert kinds == ["NEW", "DEMOTION", "PROMOTION"]                       # the hold left no status change
    assert [c["to_state"] for c in reversed(records.status_history(*target))] == [
        "VALIDATED FOR RESEARCH", "INSUFFICIENT SAMPLE", "VALIDATED FOR RESEARCH"]
    assert records.latest_version(*target)["version_no"] == 3             # held run wrote no version


def test_one_failing_strategy_does_not_stop_the_batch(synthetic_ohlcv, synthetic_features):
    def suite(cls, ohlcv, features, instrument, **kw):
        if cls is ORBStrategy:
            raise RuntimeError("boom")
        return run_validation_suite(cls, ohlcv, features, instrument, robustness="never")
    s = make(loader_for(synthetic_ohlcv, synthetic_features), suite=suite,
             strategies={"Opening Range Breakout": ORBStrategy, "Momentum": MomentumStrategy})
    res = one_batch(s, ("ISOLATE", TF))
    by = {o.strategy: o for o in res.outcomes}
    assert by["Opening Range Breakout"].outcome == "FAILED" and "boom" in by["Opening Range Breakout"].detail
    assert by["Momentum"].outcome == "COMPLETED"
    assert set(load_latest_evidence("ISOLATE", TF)) == {"Momentum"}
    assert {r["outcome"] for r in records.run_history("ISOLATE", TF)} == {"FAILED", "COMPLETED"}


def test_a_failed_run_is_retried_not_treated_as_done(synthetic_ohlcv, synthetic_features):
    calls = []

    def flaky(cls, ohlcv, features, instrument, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return run_validation_suite(cls, ohlcv, features, instrument, robustness="never")
    s = make(loader_for(synthetic_ohlcv, synthetic_features), suite=flaky)
    assert one_batch(s, ("RETRY", TF)).outcomes[0].outcome == "FAILED"
    assert one_batch(s, ("RETRY", TF)).outcomes[0].outcome == "COMPLETED"


def test_the_batch_time_budget_defers_the_rest(synthetic_ohlcv, synthetic_features):
    pol = dataclasses.replace(sch.SETTINGS.research, max_batch_runtime_sec=-1)
    res = one_batch(make(loader_for(synthetic_ohlcv, synthetic_features), policy=pol), ("BUDGET", TF))
    assert [o.outcome for o in res.outcomes] == ["DEFERRED"] and load_latest_evidence("BUDGET", TF) == {}
    assert records.last_completed_run("BUDGET", TF, "Momentum", "1.0") is None       # so it is picked up next round


def test_governance_tables_are_append_only(synthetic_ohlcv, synthetic_features):
    s = make(loader_for(synthetic_ohlcv, synthetic_features))
    one_batch(s, ("APPEND", TF))
    before = records.status_history("APPEND", TF)
    one_batch(s, ("APPEND", TF), force=True)
    after = records.status_history("APPEND", TF)
    assert [c for c in after if c["id"] in {b["id"] for b in before}] == before      # earlier rows are unchanged, only appended to
    assert records.latest_version("APPEND", TF)["version_no"] == 2


# ---- the thread ----------------------------------------------------------------------------------------------------

def test_the_thread_is_off_unless_enabled_and_stops_cleanly(monkeypatch):
    s = make(lambda i, t: (_ for _ in ()).throw(DataUnavailable("x")))
    assert s.start() is False and not s.running                 # RESEARCH_SCHEDULER is off by default
    rounds = []
    monkeypatch.setattr(s, "run_once", lambda *a, **k: rounds.append(1) or [])
    assert s.start(force=True) is True
    assert s.start(force=True) is False                         # idempotent
    for _ in range(200):
        if rounds:
            break
        import time as _t
        _t.sleep(0.01)
    s.stop()
    s._thread.join(timeout=5)
    assert rounds and not s.running


def test_a_failing_round_is_recorded_and_the_thread_survives(monkeypatch):
    s = make(lambda i, t: (_ for _ in ()).throw(DataUnavailable("x")))

    def boom(*a, **k):
        raise RuntimeError("round failed")
    monkeypatch.setattr(s, "run_once", boom)
    s.start(force=True)
    import time as _t
    for _ in range(200):
        if s.last_error:
            break
        _t.sleep(0.01)
    assert "round failed" in s.last_error and s.running
    s.stop()
    s._thread.join(timeout=5)
    assert s.status()["error"] and s.status()["targets"]
