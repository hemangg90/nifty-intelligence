"""
Gated ranker: evidence first, score second.

The legacy ranker (ranking_engine.rank_and_select) scores strategies on how the CURRENT market resembles their past
winners. It never asks whether the strategy has any demonstrated edge at all, so the best-looking of seventeen can win on
luck. This wrapper puts the validation evidence in front of it. It does not replace it: the legacy scores, eligibility
rules and tie logic are reused as the "does this fit right now" stage.

Stages, each a separate field on `StrategyDecision`:

1. Eligibility gates (`evaluate_gates`). Data quality OK; evidence exists, is current (same strategy version, not stale)
   and at least the configured state; the net-edge interval clears zero; out-of-sample exists and is positive; drawdown
   within policy; no failed robustness test. Every gate must pass. A good score cannot offset a failed gate.
2. Research rank. Eligible strategies are ordered in tiers (evidence state, then bands of out-of-sample lower bound,
   walk-forward consistency, parameter stability, cost sensitivity), then by out-of-sample lower bound. Tiers, not a
   weighted sum, so one flattering metric cannot buy back a weak one.
3. Compatibility with now: the strategy's own history in the current regime (strategy_regime_stats), plus the legacy
   conditional-edge eligibility for the current features. Unknown regime history is flagged, and blocks only if configured.
4. Setup validity: does the strategy's entry rule trigger on the latest bar (execution/setup_detector).
5. Final status. The ranker NEVER grants permission to trade: the best it says is SETUP_VALID_PENDING_RISK. The risk
   engine (risk/risk_engine.evaluate_trade, called by the auto traders) still decides, and nothing here imports it.

The selection is the top research-ranked candidate that is compatible. If several share the top tier, the legacy ranker
chooses among them, including its tie-break flag. With no stored evidence nothing is eligible and the answer is NO TRADE.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

from quant_intelligence.config.settings import SETTINGS, GatedRankerPolicy
from quant_intelligence.ranking.ranking_engine import RankingDecision, StrategyScore, rank_and_select
from quant_intelligence.validation.evidence import EvidenceState
from quant_intelligence.validation.evidence_store import EvidenceSnapshot, load_latest_evidence

NO_EVIDENCE = "NO EVIDENCE"
COMPATIBLE, INCOMPATIBLE, UNKNOWN, NOT_ASSESSED = "COMPATIBLE", "INCOMPATIBLE", "UNKNOWN", "NOT_ASSESSED"
NOT_ELIGIBLE, INCOMPATIBLE_NOW, NOT_SELECTED = "NOT_ELIGIBLE", "INCOMPATIBLE_NOW", "NOT_SELECTED"
WAITING_FOR_SETUP, SETUP_VALID_PENDING_RISK = "WAITING_FOR_SETUP", "SETUP_VALID_PENDING_RISK"

_STATE_ORDER = [EvidenceState.INSUFFICIENT_DATA, EvidenceState.INSUFFICIENT_SAMPLE, EvidenceState.VALIDATION_FAILED,
                EvidenceState.RESEARCH_QUALIFIED, EvidenceState.ROBUSTNESS_TESTING, EvidenceState.VALIDATED_FOR_RESEARCH]
_STATE_BAND = {EvidenceState.RESEARCH_QUALIFIED: 1, EvidenceState.ROBUSTNESS_TESTING: 2, EvidenceState.VALIDATED_FOR_RESEARCH: 3}


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    value: float | str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StrategyDecision:
    strategy: str
    research_status: str                      # the evidence state, or NO EVIDENCE
    eligible: bool                            # every gate passed
    gates: list[GateResult] = field(default_factory=list)
    research_rank: int | None = None          # 1 = best among eligible strategies
    tier: tuple = ()
    compatibility: str = NOT_ASSESSED
    compatibility_detail: str = ""
    setup_status: str = NOT_ASSESSED
    final_status: str = NOT_ELIGIBLE
    reasons: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def as_dict(self) -> dict:
        return {"strategy": self.strategy, "research_status": self.research_status, "eligible": self.eligible,
                "research_rank": self.research_rank, "tier": list(self.tier), "compatibility": self.compatibility,
                "compatibility_detail": self.compatibility_detail, "setup_status": self.setup_status,
                "final_status": self.final_status, "reasons": self.reasons, "gates": [g.as_dict() for g in self.gates]}


@dataclass
class GatedRanking:
    decisions: list[StrategyDecision]
    ranking: RankingDecision       # same shape the legacy consumers (auto traders, UI) already read
    legacy: RankingDecision        # what the ungated ranker would have chosen, for comparison

    def decision_for(self, strategy: str) -> StrategyDecision | None:
        return next((d for d in self.decisions if d.strategy == strategy), None)


def _state_at_least(state: EvidenceState, minimum: str) -> bool:
    try:
        floor = EvidenceState(minimum)
    except ValueError:
        floor = EvidenceState.RESEARCH_QUALIFIED
    return state.rankable and _STATE_ORDER.index(state) >= _STATE_ORDER.index(floor)


# ---- stage 1: gates -----------------------------------------------------------------------------------------------

def evaluate_gates(snapshot: EvidenceSnapshot | None, current_version: str | None, data_quality: str,
                   policy: GatedRankerPolicy, now: dt.datetime | None = None) -> list[GateResult]:
    ev = SETTINGS.evidence
    gates = [GateResult("data_quality", data_quality == "OK", f"data quality is {data_quality}", data_quality)]
    if snapshot is None:
        gates.append(GateResult("evidence_exists", False, "no stored validation evidence for this strategy, instrument and timeframe"))
        return gates
    gates.append(GateResult("evidence_exists", True, "evidence found", snapshot.state.value))

    age = snapshot.age_days(now)
    same_version = current_version is None or snapshot.strategy_version == current_version
    fresh = age is not None and age <= policy.max_evidence_age_days
    gates.append(GateResult("evidence_current", bool(same_version and fresh),
                            ("strategy version changed since the evidence was made" if not same_version else
                             f"evidence is {age:.0f} days old (limit {policy.max_evidence_age_days})" if not fresh else
                             f"evidence is {age:.0f} days old"), None if age is None else round(age, 1)))
    gates.append(GateResult("evidence_state", _state_at_least(snapshot.state, policy.min_state),
                            f"state {snapshot.state.value}, required at least {policy.min_state}", snapshot.state.value))

    s = snapshot.summary
    lo = s.get("ci_low")
    gates.append(GateResult("net_edge", lo is not None and lo > ev.min_net_expectancy_r,
                            "lower bound of the net expectancy interval " + ("is unknown" if lo is None else f"{lo:+.3f} R (must exceed {ev.min_net_expectancy_r:g})"), lo))

    if policy.require_oos:
        oos = s.get("oos") or {}
        n, exp = oos.get("n_trades", 0) or 0, oos.get("net_expectancy_r")
        ok = n >= ev.min_oos_trades and exp is not None and exp > 0
        gates.append(GateResult("out_of_sample", bool(ok), f"{n} out-of-sample trades, net expectancy " + ("n/a" if exp is None else f"{exp:+.3f} R")
                                + f" (need at least {ev.min_oos_trades} trades and a positive result)", exp))

    dd = s.get("max_drawdown_r")
    gates.append(GateResult("drawdown", dd is not None and abs(dd) <= ev.max_drawdown_r,
                            "max drawdown " + ("unknown" if dd is None else f"{abs(dd):.1f} R") + f" (limit {ev.max_drawdown_r:g})", dd))

    rob = s.get("robustness") or {}
    failed = sorted(k for k, v in rob.items() if v is False)
    unrun = sorted(k for k, v in rob.items() if v is None)
    need_all = policy.min_state == EvidenceState.VALIDATED_FOR_RESEARCH.value
    gates.append(GateResult("robustness", not failed and not (need_all and (unrun or not rob)),
                            ("failed: " + ", ".join(failed)) if failed else
                            ("not run: " + ", ".join(unrun or ["all"])) if need_all and (unrun or not rob) else
                            "no failed robustness test" + (f" ({len(unrun)} not run)" if unrun else "")))
    return gates


# ---- stage 2: research rank ---------------------------------------------------------------------------------------

def _band(x: float | None, hi: float, mid: float) -> int:
    return 0 if x is None else 2 if x >= hi else 1 if x >= mid else 0


def research_tier(snapshot: EvidenceSnapshot) -> tuple:
    """Higher is better, compared left to right. Coarse bands, so a strong later criterion cannot buy back a weak earlier one."""
    s = snapshot.summary
    oos_low = (s.get("oos") or {}).get("ci_low")
    wf = (s.get("walk_forward") or {}).get("share_positive")
    return (_STATE_BAND.get(snapshot.state, 0), _band(oos_low, 0.05, 0.0), _band(wf, 0.75, 0.5),
            _band(s.get("param_positive_share"), 0.9, 0.7), _band(s.get("slippage_breakeven"), 3.0, 2.0))


def _oos_low(snapshot: EvidenceSnapshot) -> float:
    v = (snapshot.summary.get("oos") or {}).get("ci_low")
    return float("-inf") if v is None else float(v)


def research_stage(names_versions: list[tuple[str, str | None]], evidence: dict[str, EvidenceSnapshot], data_quality: str,
                   policy: GatedRankerPolicy, now: dt.datetime | None = None
                   ) -> tuple[dict[str, StrategyDecision], dict[str, EvidenceSnapshot], list[str]]:
    """Stages 1 and 2 on their own: the gates and the research rank. Needs no market data, so the dashboard can show the
    leaderboard without running the pipeline. Returns (decisions, snapshots used, eligible names best first)."""
    decisions: dict[str, StrategyDecision] = {}
    snapshots: dict[str, EvidenceSnapshot] = {}
    for name, version in names_versions:
        snap = evidence.get(name)
        gates = evaluate_gates(snap, version, data_quality, policy, now)
        d = StrategyDecision(name, snap.state.value if snap else NO_EVIDENCE, all(g.passed for g in gates), gates)
        d.reasons = [f"{g.name}: {g.detail}" for g in gates if not g.passed]
        decisions[name] = d
        if snap is not None:
            snapshots[name] = snap
            d.tier = research_tier(snap)
    eligible_names = sorted((n for n, d in decisions.items() if d.eligible),
                            key=lambda n: (decisions[n].tier, _oos_low(snapshots[n])), reverse=True)
    for i, n in enumerate(eligible_names, start=1):
        decisions[n].research_rank = i
    return decisions, snapshots, eligible_names


# ---- stage 3: compatibility ---------------------------------------------------------------------------------------

def load_regime_stats(instrument: str, timeframe: str) -> dict[str, dict[str, dict]]:
    """{strategy: {regime_label: {n_trades, net_expectancy_r}}} from the latest strategy_regime_stats rows."""
    try:
        from sqlalchemy import func

        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import StrategyRegimeStat as T

        with get_session() as s:
            newest = (s.query(func.max(T.id).label("mid")).filter(T.instrument == instrument, T.timeframe == timeframe, T.dimension == "regime")
                      .group_by(T.strategy, T.label).subquery())
            rows = s.query(T).join(newest, T.id == newest.c.mid).all()
            out: dict[str, dict[str, dict]] = {}
            for r in rows:
                out.setdefault(r.strategy, {})[r.label] = {"n_trades": r.n_trades or 0, "net_expectancy_r": r.net_expectancy_r}
            return out
    except Exception:
        return {}


def regime_fit(strategy_stats: dict[str, dict] | None, regime_label: str, policy: GatedRankerPolicy) -> tuple[str, str]:
    row = (strategy_stats or {}).get(regime_label)
    if row is None or (row.get("n_trades") or 0) < policy.regime_min_trades or row.get("net_expectancy_r") is None:
        n = 0 if row is None else row.get("n_trades") or 0
        return UNKNOWN, f"{n} historical trades in the {regime_label} regime (need {policy.regime_min_trades}); no evidence either way"
    exp = row["net_expectancy_r"]
    detail = f"{row['n_trades']} historical trades in the {regime_label} regime, net expectancy {exp:+.3f} R"
    return (COMPATIBLE if exp > 0 else INCOMPATIBLE), detail


# ---- the ranker ---------------------------------------------------------------------------------------------------

def gated_rank(
    strategies: list,
    scores: list[StrategyScore],
    instrument: str,
    timeframe: str,
    data_quality_status: str,
    regime_label: str,
    market_state=None,
    recent_ohlcv=None,
    evidence: dict[str, EvidenceSnapshot] | None = None,
    regime_stats: dict[str, dict[str, dict]] | None = None,
    policy: GatedRankerPolicy | None = None,
    now: dt.datetime | None = None,
    setup_fn: Callable | None = None,
) -> GatedRanking:
    """Gate, rank, check fit and setup. `evidence`/`regime_stats` default to the stored ones; `setup_fn(strategy, state,
    ohlcv)` defaults to execution.setup_detector.detect_setup (a test seam)."""
    pol = policy or SETTINGS.gated
    evidence = load_latest_evidence(instrument, timeframe) if evidence is None else evidence
    regime_stats = load_regime_stats(instrument, timeframe) if regime_stats is None else regime_stats
    if setup_fn is None:
        from quant_intelligence.execution.setup_detector import detect_setup as setup_fn  # noqa: N813
    legacy_scores = {s.strategy_name: s for s in scores}
    legacy = rank_and_select(scores, data_quality_status, allow_tie_break=True)

    decisions, snapshots, eligible_names = research_stage([(s_.name, getattr(s_, "version", None)) for s_ in strategies],
                                                          evidence, data_quality_status, pol, now)

    # stage 3: compatibility with the current market
    candidates: list[str] = []
    for n in eligible_names:
        d = decisions[n]
        fit, detail = regime_fit(regime_stats.get(n), regime_label, pol)
        ls = legacy_scores.get(n)
        if ls is not None and not ls.eligible:
            d.compatibility, d.compatibility_detail = INCOMPATIBLE, f"current conditions: {ls.ineligibility_reason}"
        elif fit == INCOMPATIBLE:
            d.compatibility, d.compatibility_detail = INCOMPATIBLE, detail
        elif fit == UNKNOWN and pol.require_regime_evidence:
            d.compatibility, d.compatibility_detail = INCOMPATIBLE, detail + " (regime evidence required)"
        else:
            d.compatibility, d.compatibility_detail = fit, detail
        if d.compatibility != INCOMPATIBLE:
            candidates.append(n)

    # selection: best research tier among candidates; the legacy ranker breaks ties within that tier
    selected: str | None = None
    tie_break, runner_up, ranking_reason = False, None, ""
    if candidates:
        top_tier = decisions[candidates[0]].tier
        group = [n for n in candidates if decisions[n].tier == top_tier]
        if len(group) == 1:
            selected = group[0]
            ranking_reason = f"{selected} ranks first on validated evidence (research rank {decisions[selected].research_rank})"
        else:
            sub = rank_and_select([legacy_scores[n] for n in group if n in legacy_scores], data_quality_status, allow_tie_break=True)
            if not sub.is_no_trade:
                selected, tie_break, runner_up = sub.selected_strategy, sub.tie_break, sub.tie_runner_up
                ranking_reason = f"{sub.reason} (the leading evidence tier held {len(group)} strategies)"
            else:
                ranking_reason = f"{sub.reason} (the leading evidence tier held {len(group)} strategies)"

    # stage 4: setup validity for the candidates (cheap: the entry rule on the latest bar)
    by_name = {s.name: s for s in strategies}
    for n in candidates:
        try:
            status = setup_fn(by_name[n], market_state, recent_ohlcv)
            decisions[n].setup_status = status.status
        except Exception as e:  # a strategy's live check failing must not take the ranking down
            decisions[n].setup_status = NOT_ASSESSED
            decisions[n].reasons.append(f"setup check failed: {e}")

    # stage 5: final status (a research/fit/setup verdict; the risk engine still decides whether to trade)
    for n, d in decisions.items():
        if not d.eligible:
            d.final_status = NOT_ELIGIBLE
        elif d.compatibility == INCOMPATIBLE:
            d.final_status = INCOMPATIBLE_NOW
            d.reasons.append(f"compatibility: {d.compatibility_detail}")
        elif n != selected:
            d.final_status = NOT_SELECTED
        else:
            d.final_status = SETUP_VALID_PENDING_RISK if d.setup_status == "SETUP_TRIGGERED" else WAITING_FOR_SETUP

    ordered = (eligible_names + sorted((n for n in decisions if n not in eligible_names),
                                       key=lambda n: legacy_scores[n].score if n in legacy_scores else 0.0, reverse=True))
    ranked_scores = [legacy_scores[n] for n in ordered if n in legacy_scores]
    if selected is not None:
        ranking = RankingDecision(ranked_scores, selected, False, ranking_reason, tie_break=tie_break, tie_runner_up=runner_up)
    else:
        reason = ranking_reason if candidates and ranking_reason else _no_trade_reason(decisions, candidates, data_quality_status)
        ranking = RankingDecision(ranked_scores, None, True, reason)
    return GatedRanking(sorted(decisions.values(), key=lambda d: (d.research_rank is None, d.research_rank or 0, d.strategy)),
                        ranking, legacy)


def _no_trade_reason(decisions: dict[str, StrategyDecision], candidates: list[str], data_quality: str) -> str:
    if data_quality != "OK":
        return f"Data quality is {data_quality}: NO TRADE forced"
    n_elig = sum(d.eligible for d in decisions.values())
    if n_elig and not candidates:
        return f"{n_elig} strategies have validated evidence but none fits the current market; NO TRADE"
    blockers: dict[str, int] = {}
    for d in decisions.values():
        first = next((g.name for g in d.gates if not g.passed), None)
        if first:
            blockers[first] = blockers.get(first, 0) + 1
    why = ", ".join(f"{c} x {k}" for k, c in sorted(blockers.items(), key=lambda kv: -kv[1]))
    return f"No strategy passes the evidence gates ({why}); NO TRADE"


def compare_rankings(result: GatedRanking) -> dict:
    """Ungated versus gated choice for the same scan, and why they differ."""
    legacy_pick = result.legacy.selected_strategy
    gated_pick = result.ranking.selected_strategy
    out = {"legacy": legacy_pick, "gated": gated_pick, "agree": legacy_pick == gated_pick, "why": ""}
    if legacy_pick and legacy_pick != gated_pick:
        d = result.decision_for(legacy_pick)
        out["why"] = ("; ".join(d.reasons) if d and d.reasons else "not selected by the gated ranker")
    return out
