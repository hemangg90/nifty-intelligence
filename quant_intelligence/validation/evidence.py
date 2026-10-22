"""
Evidence states: what a backtest is allowed to claim.

A strategy is never given a numeric rank on the strength of a backtest alone. It is first placed in one of six states,
and only the last two can support a research ranking. The labels describe the evidence, never the future: none of them
means a strategy will be profitable.

  INSUFFICIENT_DATA     The history is missing, failed its quality gate, is too short, or the strategy cannot be
                        backtested with the data we have (e.g. it needs option-chain history that does not exist).
  INSUFFICIENT_SAMPLE   The data is sound but there are too few trades, trade-days or independent observations to
                        support any conclusion about the edge, in either direction.
  VALIDATION_FAILED     The sample is large enough and at least one required test failed: the net edge is not
                        distinguishable from zero or is negative, the profit factor or drawdown breaches the policy, or
                        a required out-of-sample or robustness test failed.
  RESEARCH_QUALIFIED    All in-sample statistical gates pass. Out-of-sample evidence has not been supplied, so this is
                        a candidate for further testing and nothing more.
  ROBUSTNESS_TESTING    In-sample gates and the out-of-sample test pass; one or more required robustness tests have
                        not yet been run or are still pending.
  VALIDATED_FOR_RESEARCH  In-sample gates, out-of-sample, and every required robustness test pass under the documented
                        policy. "Validated" refers to this research process only. It is not a prediction, not advice,
                        and a state that can be revoked by later data.

Thresholds come from `SETTINGS.evidence` (`EvidencePolicy`); every assessment records the policy it used and the value of
each check, so a state is always traceable to numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from quant_intelligence.config.settings import EvidencePolicy, SETTINGS
from quant_intelligence.validation.metrics import PerformanceMetrics
from quant_intelligence.validation.uncertainty import UncertaintyEstimate

REQUIRED_ROBUSTNESS_TESTS = (
    "parameter_sensitivity",
    "regime_sensitivity",
    "cost_stress",
    "period_consistency",
    "sequence_analysis",
)


class EvidenceState(str, Enum):
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT SAMPLE"
    VALIDATION_FAILED = "VALIDATION FAILED"
    RESEARCH_QUALIFIED = "RESEARCH QUALIFIED"
    ROBUSTNESS_TESTING = "ROBUSTNESS TESTING"
    VALIDATED_FOR_RESEARCH = "VALIDATED FOR RESEARCH"

    @property
    def rankable(self) -> bool:
        """Only states that passed the in-sample gates may enter the research ranking (Phase 4 gates on this)."""
        return self in (EvidenceState.RESEARCH_QUALIFIED, EvidenceState.ROBUSTNESS_TESTING, EvidenceState.VALIDATED_FOR_RESEARCH)


STATE_DEFINITIONS: dict[EvidenceState, str] = {
    EvidenceState.INSUFFICIENT_DATA: "History missing, failed quality checks, too short, or the strategy cannot be backtested with available data.",
    EvidenceState.INSUFFICIENT_SAMPLE: "Data is sound, but too few trades or independent observations to conclude anything about the edge.",
    EvidenceState.VALIDATION_FAILED: "Enough evidence to judge, and at least one required test failed.",
    EvidenceState.RESEARCH_QUALIFIED: "Passes the in-sample statistical gates; out-of-sample evidence not yet supplied.",
    EvidenceState.ROBUSTNESS_TESTING: "Passes in-sample and out-of-sample; required robustness tests incomplete.",
    EvidenceState.VALIDATED_FOR_RESEARCH: "Passes every gate of the documented research process. Not a prediction of future profit.",
}


@dataclass
class OutOfSampleResult:
    """Net-of-cost results on data the strategy was not developed on (see validation/splits.py in Phase 3)."""

    n_trades: int
    net_expectancy_r: float
    expectancy_ci_low: float | None = None  # lower bound of the OOS expectancy interval, when computed


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float | str | None
    threshold: float | str | None
    detail: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EvidenceAssessment:
    state: EvidenceState
    reasons: list[str]
    checks: list[CheckResult] = field(default_factory=list)
    policy: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.state.value

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {"state": self.state.value, "definition": STATE_DEFINITIONS[self.state], "reasons": list(self.reasons),
                "checks": [c.as_dict() for c in self.checks], "policy": dict(self.policy)}


def _isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def classify_evidence(
    metrics: PerformanceMetrics,
    uncertainty: UncertaintyEstimate,
    data_quality: str = "OK",
    backtestable: bool = True,
    n_sessions: int | None = None,
    out_of_sample: OutOfSampleResult | None = None,
    robustness: dict[str, bool | None] | None = None,
    policy: EvidencePolicy | None = None,
    data_limitations: list[str] | None = None,
) -> EvidenceAssessment:
    """Place a backtest in an evidence state. Pure function of its inputs; deterministic.

    `data_limitations`: non-empty when the backtest ran on a proxy for the data the strategy really uses (for example a
    near-the-money slice of the option chain instead of the whole chain). Such a strategy can pass every gate and still
    never reaches VALIDATED FOR RESEARCH.

    `robustness` maps each name in REQUIRED_ROBUSTNESS_TESTS to True (passed), False (failed) or None / absent (not run).
    A failed required test is VALIDATION_FAILED regardless of everything else.
    """
    pol = policy or SETTINGS.evidence
    policy_snapshot = {k: v for k, v in pol.__dict__.items() if k not in ("bootstrap_seed",)}
    checks: list[CheckResult] = []

    def result(state: EvidenceState, reasons: list[str]) -> EvidenceAssessment:
        return EvidenceAssessment(state, reasons, checks, policy_snapshot)

    # ---- 1. Data sufficiency -----------------------------------------------------------------------
    data_reasons: list[str] = []
    if not backtestable:
        data_reasons.append("strategy cannot be backtested with the available data (needs option-chain history)")
    if data_quality == "FAIL":
        data_reasons.append("the data failed its quality gate")
    if n_sessions is not None and n_sessions < pol.min_sessions:
        data_reasons.append(f"only {n_sessions} sessions of history (policy minimum {pol.min_sessions})")
    checks.append(CheckResult("data_quality", data_quality != "FAIL", data_quality, "not FAIL"))
    checks.append(CheckResult("backtestable", backtestable, backtestable, True))
    if n_sessions is not None:
        checks.append(CheckResult("history_sessions", n_sessions >= pol.min_sessions, n_sessions, pol.min_sessions))
    if data_reasons:
        return result(EvidenceState.INSUFFICIENT_DATA, data_reasons)

    # ---- 2. Sample sufficiency ---------------------------------------------------------------------
    sample = [
        CheckResult("trades", metrics.n_trades >= pol.min_trades, metrics.n_trades, pol.min_trades),
        CheckResult("effective_trades", uncertainty.effective_n >= pol.min_effective_trades,
                    round(uncertainty.effective_n, 1), pol.min_effective_trades,
                    f"{metrics.n_trades} trades discounted for same-session and serial dependence"),
        CheckResult("trade_days", metrics.n_trade_days >= pol.min_trade_days, metrics.n_trade_days, pol.min_trade_days),
    ]
    checks.extend(sample)
    thin = [c for c in sample if not c.passed]
    if thin:
        return result(EvidenceState.INSUFFICIENT_SAMPLE,
                      [f"{c.name} {c.value} is below the policy minimum {c.threshold}" for c in thin])

    # ---- 3. In-sample statistical gates ------------------------------------------------------------
    ci_low = uncertainty.expectancy_ci[0]
    gates = [
        CheckResult("net_expectancy_ci_lower", _isnum(ci_low) and ci_low > pol.min_net_expectancy_r,
                    None if not _isnum(ci_low) else round(ci_low, 4), pol.min_net_expectancy_r,
                    f"lower bound of the {uncertainty.confidence_level:.0%} interval for mean net R per trade "
                    f"(mean {metrics.net_expectancy_r:.3f})"),
        CheckResult("profit_factor", _isnum(metrics.profit_factor) and metrics.profit_factor > pol.min_profit_factor,
                    None if not _isnum(metrics.profit_factor) else round(metrics.profit_factor, 3), pol.min_profit_factor),
        CheckResult("max_drawdown_r", _isnum(metrics.max_drawdown_r) and -metrics.max_drawdown_r <= pol.max_drawdown_r,
                    None if not _isnum(metrics.max_drawdown_r) else round(-metrics.max_drawdown_r, 2), pol.max_drawdown_r,
                    "magnitude of the worst peak-to-trough fall in cumulative net R"),
    ]
    checks.extend(gates)
    failed = [c for c in gates if not c.passed]
    if failed:
        return result(EvidenceState.VALIDATION_FAILED, [_why_failed(c) for c in failed])

    # ---- 4. Out-of-sample ----------------------------------------------------------------------------
    if out_of_sample is None:
        return result(EvidenceState.RESEARCH_QUALIFIED,
                      ["in-sample gates passed; no out-of-sample result supplied yet"])
    oos_checks = [
        CheckResult("oos_trades", out_of_sample.n_trades >= pol.min_oos_trades, out_of_sample.n_trades, pol.min_oos_trades),
        CheckResult("oos_net_expectancy_r", _isnum(out_of_sample.net_expectancy_r) and out_of_sample.net_expectancy_r > pol.min_net_expectancy_r,
                    None if not _isnum(out_of_sample.net_expectancy_r) else round(out_of_sample.net_expectancy_r, 4),
                    pol.min_net_expectancy_r, "mean net R per trade on data the strategy was not developed on"),
    ]
    checks.extend(oos_checks)
    if not oos_checks[0].passed:
        return result(EvidenceState.RESEARCH_QUALIFIED,
                      [f"out-of-sample has only {out_of_sample.n_trades} trades (policy minimum {pol.min_oos_trades}); "
                       "not enough to confirm or reject"])
    if not oos_checks[1].passed:
        return result(EvidenceState.VALIDATION_FAILED, [_why_failed(oos_checks[1])])

    # ---- 5. Robustness ---------------------------------------------------------------------------------
    robustness = robustness or {}
    unknown = set(robustness) - set(REQUIRED_ROBUSTNESS_TESTS)
    if unknown:
        raise ValueError(f"unknown robustness test(s): {sorted(unknown)}; expected {REQUIRED_ROBUSTNESS_TESTS}")
    pending = []
    for name in REQUIRED_ROBUSTNESS_TESTS:
        outcome = robustness.get(name)
        checks.append(CheckResult(f"robustness:{name}", outcome is True, "passed" if outcome is True else
                                  "failed" if outcome is False else "not run", "passed"))
        if outcome is False:
            return result(EvidenceState.VALIDATION_FAILED, [f"required robustness test '{name}' failed"])
        if outcome is None:
            pending.append(name)
    if pending:
        return result(EvidenceState.ROBUSTNESS_TESTING, [f"robustness tests not yet run: {', '.join(pending)}"])
    if data_limitations:
        # Everything passed, but on a stand-in for the data the strategy really uses: never "validated".
        checks.append(CheckResult("proxy_data", False, "; ".join(data_limitations), "none"))
        return result(EvidenceState.ROBUSTNESS_TESTING,
                      ["all gates passed, but the backtest used proxy data, so the state is capped below VALIDATED FOR RESEARCH: "
                       + "; ".join(data_limitations)])
    return result(EvidenceState.VALIDATED_FOR_RESEARCH,
                  ["in-sample gates, out-of-sample and all required robustness tests passed under the documented policy"])


def _why_failed(c: CheckResult) -> str:
    if c.name == "net_expectancy_ci_lower":
        return f"net expectancy is not distinguishable from {c.threshold:g}: interval lower bound {c.value} R ({c.detail})"
    if c.name == "profit_factor":
        return f"net profit factor {c.value} is not above {c.threshold:g}"
    if c.name == "max_drawdown_r":
        return f"max drawdown {c.value} R exceeds the policy limit {c.threshold:g} R"
    if c.name == "oos_net_expectancy_r":
        return f"out-of-sample net expectancy {c.value} R is not above {c.threshold:g}"
    return f"{c.name} failed (value {c.value}, threshold {c.threshold})"
