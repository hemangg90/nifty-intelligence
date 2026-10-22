"""Acceptance rules: bad news at once, good news earned, improvement within a state never a promotion."""
from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.research import acceptance as acc
from quant_intelligence.validation.evidence import EvidenceState as S
from quant_intelligence.validation.evidence_store import EvidenceSnapshot

POLICY = SETTINGS.research
END = dt.date(2026, 6, 1)
MONDAY = dt.date(2026, 6, 1)


def prev(state, version="1.0", data_end=END, oos_low=0.05, phash="abc"):
    return EvidenceSnapshot("Momentum", version, state, dt.datetime(2026, 6, 2), "cfg",
                            {"data_end": str(data_end), "policy_hash": phash, "oos": {"n_trades": 60, "net_expectancy_r": 0.1, "ci_low": oos_low}})


def decide(previous, state, *, data_end=END, version="1.0", oos_low=0.05, phash="abc", policy=POLICY):
    return acc.decide(previous, state, {"oos": {"ci_low": oos_low}}, version, data_end, phash, policy)


def later(days):
    return END + dt.timedelta(days=days)


def test_a_first_verdict_is_accepted_whatever_it_says():
    d = decide(None, S.VALIDATION_FAILED)
    assert d.accepted and d.kind == acc.NEW and d.writes_status_change


def test_a_new_strategy_version_starts_afresh():
    d = decide(prev(S.VALIDATED_FOR_RESEARCH, version="1.0"), S.RESEARCH_QUALIFIED, version="2.0")
    assert d.accepted and d.kind == acc.NEW and "version changed" in d.reason


@pytest.mark.parametrize("old,new", [(S.VALIDATED_FOR_RESEARCH, S.ROBUSTNESS_TESTING), (S.ROBUSTNESS_TESTING, S.VALIDATION_FAILED),
                                     (S.RESEARCH_QUALIFIED, S.INSUFFICIENT_SAMPLE), (S.VALIDATED_FOR_RESEARCH, S.INSUFFICIENT_DATA)])
def test_any_fall_is_accepted_immediately_even_without_new_data(old, new):
    d = decide(prev(old), new, data_end=END)
    assert d.accepted and d.kind == acc.DEMOTION and d.writes_status_change and old.value in d.reason and new.value in d.reason


def test_a_demotion_is_accepted_even_if_the_thresholds_changed():
    assert decide(prev(S.ROBUSTNESS_TESTING, phash="old"), S.VALIDATION_FAILED, phash="new").accepted


def test_the_same_state_is_a_refresh_and_never_a_promotion():
    d = decide(prev(S.ROBUSTNESS_TESTING, oos_low=0.05), S.ROBUSTNESS_TESTING, oos_low=0.30, data_end=later(60))
    assert d.accepted and d.kind == acc.REFRESH and not d.writes_status_change


def test_a_fall_in_the_out_of_sample_bound_is_flagged_but_the_state_stays():
    d = decide(prev(S.ROBUSTNESS_TESTING, oos_low=0.10), S.ROBUSTNESS_TESTING, oos_low=0.10 - POLICY.deterioration_threshold_r - 0.01)
    assert d.accepted and d.kind == acc.DETERIORATION and d.writes_status_change and "fell" in d.reason


def test_a_small_wobble_is_not_deterioration():
    d = decide(prev(S.ROBUSTNESS_TESTING, oos_low=0.10), S.ROBUSTNESS_TESTING, oos_low=0.10 - POLICY.deterioration_threshold_r / 2)
    assert d.kind == acc.REFRESH


def test_deterioration_is_not_flagged_for_a_state_that_never_ranked():
    d = decide(prev(S.VALIDATION_FAILED, oos_low=0.10), S.VALIDATION_FAILED, oos_low=-0.5)
    assert d.kind == acc.REFRESH


def test_a_promotion_on_the_same_data_is_held():
    d = decide(prev(S.RESEARCH_QUALIFIED), S.VALIDATED_FOR_RESEARCH, data_end=END)
    assert not d.accepted and d.kind == acc.PROMOTION_HELD and not d.writes_status_change and "new data" in d.reason


def test_a_promotion_needs_enough_new_sessions():
    need = POLICY.min_new_sessions_for_promotion
    too_few = decide(prev(S.RESEARCH_QUALIFIED), S.ROBUSTNESS_TESTING, data_end=END + dt.timedelta(days=need // 5 * 7 - 7))
    assert not too_few.accepted and too_few.kind == acc.PROMOTION_HELD
    enough = decide(prev(S.RESEARCH_QUALIFIED), S.ROBUSTNESS_TESTING, data_end=END + dt.timedelta(days=(need // 5 + 1) * 7))
    assert enough.accepted and enough.kind == acc.PROMOTION and enough.writes_status_change and "new sessions" in enough.reason


def test_a_promotion_is_held_when_the_thresholds_changed():
    d = decide(prev(S.RESEARCH_QUALIFIED, phash="old"), S.VALIDATED_FOR_RESEARCH, data_end=later(120), phash="new")
    assert not d.accepted and "thresholds changed" in d.reason


def test_an_unknown_accepted_data_end_cannot_support_a_promotion():
    p = prev(S.RESEARCH_QUALIFIED)
    p.summary.pop("data_end")
    d = decide(p, S.VALIDATED_FOR_RESEARCH, data_end=later(120))
    assert not d.accepted and "unknown" in d.reason


def test_a_missing_policy_hash_on_an_old_snapshot_does_not_block():
    p = prev(S.RESEARCH_QUALIFIED)
    p.summary.pop("policy_hash")
    assert decide(p, S.ROBUSTNESS_TESTING, data_end=later(120)).accepted


def test_session_counting():
    assert acc.new_sessions_between(MONDAY, MONDAY + dt.timedelta(days=7)) == 5
    assert acc.new_sessions_between(MONDAY, MONDAY) == 0
    assert acc.new_sessions_between(MONDAY + dt.timedelta(days=7), MONDAY) == 0
    assert acc.new_sessions_between(None, MONDAY) is None


def test_the_policy_thresholds_decide():
    strict = dataclasses.replace(POLICY, min_new_sessions_for_promotion=500)
    assert not decide(prev(S.RESEARCH_QUALIFIED), S.ROBUSTNESS_TESTING, data_end=later(120), policy=strict).accepted
    loose = dataclasses.replace(POLICY, min_new_sessions_for_promotion=1)
    assert decide(prev(S.RESEARCH_QUALIFIED), S.ROBUSTNESS_TESTING, data_end=later(7), policy=loose).accepted
