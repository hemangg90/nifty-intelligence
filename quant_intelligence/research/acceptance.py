"""
Which new verdicts become the accepted evidence.

The scheduler re-runs validation as data arrives. A new result is a candidate, not a replacement: the ranker reads only the
ACCEPTED verdict, and these rules decide when a candidate becomes it. They are explicit, deterministic and conservative in
one direction:

* bad news is accepted at once. A fall to a lower state is a DEMOTION; a fall in the out-of-sample lower bound of more than
  `deterioration_threshold_r` within the same state is a DETERIORATION flag (the state is kept, the warning is on record).
* good news has to be earned. A higher state is accepted only if (a) at least `min_new_sessions_for_promotion` sessions of
  data have arrived since the accepted verdict, so a re-run on the same history cannot promote, and (b) the thresholds the
  verdict was made under are unchanged, so loosening a policy cannot promote anything. Otherwise it is PROMOTION_HELD: the
  candidate is logged with the reason and the accepted verdict stands.
* improvement within a state (better numbers, same state) is a REFRESH and never a promotion.
* a changed strategy version or a first verdict is NEW: there is nothing to compare against.

Every accepted change in state is written to `status_changes` with its reason; nothing is rewritten in place.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from quant_intelligence.config.settings import ResearchPolicy
from quant_intelligence.validation.evidence import EvidenceState
from quant_intelligence.validation.evidence_store import EvidenceSnapshot

STATE_ORDER = [EvidenceState.INSUFFICIENT_DATA, EvidenceState.INSUFFICIENT_SAMPLE, EvidenceState.VALIDATION_FAILED,
               EvidenceState.RESEARCH_QUALIFIED, EvidenceState.ROBUSTNESS_TESTING, EvidenceState.VALIDATED_FOR_RESEARCH]

NEW, REFRESH, PROMOTION, DEMOTION, DETERIORATION, PROMOTION_HELD = (
    "NEW", "REFRESH", "PROMOTION", "DEMOTION", "DETERIORATION", "PROMOTION_HELD")
STATUS_CHANGE_KINDS = {NEW, PROMOTION, DEMOTION, DETERIORATION}


@dataclass
class AcceptanceDecision:
    accepted: bool
    kind: str
    reason: str

    @property
    def writes_status_change(self) -> bool:
        return self.accepted and self.kind in STATUS_CHANGE_KINDS

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "kind": self.kind, "reason": self.reason}


def new_sessions_between(earlier: dt.date | None, later: dt.date | None) -> int | None:
    """Weekdays after `earlier` up to and including `later` (holidays are not removed, so this slightly over-counts).
    None when either date is unknown."""
    if earlier is None or later is None:
        return None
    return max(int(np.busday_count(earlier + dt.timedelta(days=1), later + dt.timedelta(days=1))), 0)


def _date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def decide(
    previous: EvidenceSnapshot | None,
    new_state: EvidenceState,
    new_summary: dict,
    strategy_version: str,
    data_end: dt.date | None,
    policy_hash: str,
    policy: ResearchPolicy,
) -> AcceptanceDecision:
    if previous is None:
        return AcceptanceDecision(True, NEW, f"first verdict for this strategy: {new_state.value}")
    if previous.strategy_version != strategy_version:
        return AcceptanceDecision(True, NEW, f"strategy version changed ({previous.strategy_version} -> {strategy_version}); "
                                             f"the earlier verdict ({previous.state.value}) does not apply")

    old, new = STATE_ORDER.index(previous.state), STATE_ORDER.index(new_state)
    if new < old:
        return AcceptanceDecision(True, DEMOTION, f"{previous.state.value} -> {new_state.value}")

    if new == old:
        was_oos = (previous.summary.get("oos") or {}).get("ci_low")
        now_oos = (new_summary.get("oos") or {}).get("ci_low")
        if previous.state.rankable and was_oos is not None and now_oos is not None \
                and was_oos - now_oos > policy.deterioration_threshold_r:
            return AcceptanceDecision(True, DETERIORATION,
                                      f"out-of-sample lower bound fell from {was_oos:+.3f} R to {now_oos:+.3f} R "
                                      f"(more than {policy.deterioration_threshold_r:g} R); state unchanged at {new_state.value}")
        return AcceptanceDecision(True, REFRESH, f"still {new_state.value}")

    # a higher state: earn it
    prior_hash = previous.summary.get("policy_hash")
    if prior_hash and prior_hash != policy_hash:
        return AcceptanceDecision(False, PROMOTION_HELD,
                                  f"{previous.state.value} -> {new_state.value} held: the evidence thresholds changed since the "
                                  "accepted verdict, so the two are not comparable; accept manually after review")
    fresh = new_sessions_between(_date(previous.summary.get("data_end")), data_end)
    if fresh is None or fresh < policy.min_new_sessions_for_promotion:
        have = "unknown" if fresh is None else str(fresh)
        return AcceptanceDecision(False, PROMOTION_HELD,
                                  f"{previous.state.value} -> {new_state.value} held: needs {policy.min_new_sessions_for_promotion} "
                                  f"sessions of new data since the accepted verdict (has {have})")
    return AcceptanceDecision(True, PROMOTION, f"{previous.state.value} -> {new_state.value} on {fresh} new sessions of data")
