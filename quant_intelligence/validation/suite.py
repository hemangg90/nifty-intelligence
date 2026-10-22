"""
The validation suite: one call from a strategy and its history to a defensible evidence state.

    split -> development backtest (in-sample gates) -> validation backtest (out-of-sample)
          -> robustness on the development window (only once everything before it has passed)
          -> holdout (only if asked for, and only for a strategy that would otherwise be VALIDATED FOR RESEARCH)

The holdout is the last thing touched and is touched rarely: every evaluation is appended to `holdout_ledger`, and the
result says whether the holdout has been looked at too often to count as untouched. Nothing here selects parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.backtesting.engine import BacktestResult, market_for, run_backtest
from quant_intelligence.config.settings import SETTINGS, SplitPolicy
from quant_intelligence.data.quality_gates import check_history
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeframe import parse_timeframe_minutes
from quant_intelligence.validation.evidence import EvidenceAssessment, EvidenceState, OutOfSampleResult
from quant_intelligence.validation.metrics import trade_table
from quant_intelligence.validation.report import ValidationReport, validate_backtest
from quant_intelligence.validation.robustness import RobustnessReport, persist_regime_stats, run_robustness
from quant_intelligence.validation.splits import (
    HoldoutStatus,
    SessionSplit,
    holdout_fingerprint,
    holdout_status,
    record_holdout_evaluation,
    register_holdout,
    slice_segment,
    split_sessions,
)
from quant_intelligence.validation.uncertainty import estimate_uncertainty
from quant_intelligence.validation.walk_forward import WalkForwardSummary, walk_forward_sessions


@dataclass
class ResearchValidation:
    strategy: str
    strategy_version: str
    instrument: str
    timeframe: str
    split: SessionSplit
    in_sample: ValidationReport                   # development segment, judged with the final OOS and robustness inputs
    out_of_sample: OutOfSampleResult | None       # validation segment
    walk_forward: WalkForwardSummary | None
    robustness: RobustnessReport | None
    holdout: dict | None = None                   # {n_trades, net_expectancy_r, ci_low, status: HoldoutStatus dict}
    notes: list[str] = field(default_factory=list)
    proxy_data: bool = False                      # ran on a stand-in for the data the strategy really uses
    detail: dict = field(default_factory=dict)    # validation/detail.py: curves, distribution, breakdowns, for the dashboard

    @property
    def evidence(self) -> EvidenceAssessment:
        return self.in_sample.evidence

    @property
    def state(self) -> EvidenceState:
        return self.in_sample.evidence.state

    def as_dict(self) -> dict:
        nn = lambda d: None if d is None else {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}  # noqa: E731
        return {"strategy": self.strategy, "strategy_version": self.strategy_version, "instrument": self.instrument,
                "timeframe": self.timeframe, "split": self.split.as_dict(), "in_sample": self.in_sample.as_dict(),
                "out_of_sample": nn(None if self.out_of_sample is None else dict(self.out_of_sample.__dict__)),
                "walk_forward": None if self.walk_forward is None else self.walk_forward.as_dict(),
                "robustness": None if self.robustness is None else self.robustness.as_dict(),
                "holdout": nn(self.holdout), "notes": self.notes, "proxy_data": self.proxy_data,
                "detail": self.detail}


def _oos(result: BacktestResult, quantity: int) -> OutOfSampleResult:
    table = trade_table(result.trades, quantity, market_for(result.instrument))
    if len(table) == 0:
        return OutOfSampleResult(0, float("nan"))
    u = estimate_uncertainty(table)
    return OutOfSampleResult(len(table), float(table["net_r"].mean()), u.expectancy_ci[0])


def run_validation_suite(
    strategy_cls,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    quantity: int = 50,
    timeframe: str = "5min",
    data_source: str = "",
    parameters: dict | None = None,
    include_holdout: bool = False,
    robustness: str = "auto",              # auto: only when the earlier gates pass | always | never
    split_policy: SplitPolicy | None = None,
    persist: bool = False,
    chain_history=None,                    # options.historical_chain.HistoricalChain: lets BUY-side chain strategies be replayed
) -> ResearchValidation:
    from quant_intelligence.strategies.rules import CHAIN_HISTORY_BACKTESTABLE, PROXY_DATA_LIMITATION

    def make(params: dict | None = None):
        strat = strategy_cls(params)
        return strat.attach_chain_history(chain_history) if chain_history is not None else strat

    probe = make(parameters)
    name, version = probe.name, probe.version
    factory = lambda: make(parameters)  # noqa: E731
    split = split_sessions(ohlcv, split_policy)
    notes: list[str] = []
    backtestable = True if probe.rules is None else probe.rules.backtestable
    proxy = False
    if not backtestable and chain_history is not None and name in CHAIN_HISTORY_BACKTESTABLE:
        backtestable, proxy = True, True
        notes.append("replayed on rebuilt near-the-money option-chain snapshots (a proxy for the whole chain)")
    limitations = [PROXY_DATA_LIMITATION] if proxy else None

    def run(window: tuple[str, ...], tag: str) -> tuple[BacktestResult, pd.DataFrame, pd.DataFrame]:
        o, f = slice_segment(ohlcv, features, split, *window)
        return run_backtest(factory(), o, f, instrument, split=tag, quantity=quantity, persist=False,
                            timeframe=timeframe, data_source=data_source), o, f

    dev, dev_o, _ = run(("development",), "DEVELOPMENT")
    quality = check_history(ohlcv, parse_timeframe_minutes(timeframe), profile_for(instrument)).status
    common = dict(quantity=quantity, data_quality=quality, n_sessions=len(split.development), backtestable=backtestable,
                  data_limitations=limitations)

    if not backtestable:
        report = validate_backtest(dev, dev_o, **common)
        return _finish(ResearchValidation(name, version, instrument, timeframe, split, report, None, None, None,
                                          notes=["strategy needs option-chain history that is not available; nothing was run"]),
                       ohlcv, data_source, persist, proxy)

    val, _, _ = run(("validation",), "VALIDATION")
    oos = _oos(val, quantity)
    prelim = validate_backtest(dev, dev_o, out_of_sample=oos, **common)

    dev_val_sessions = split.development + split.validation
    wf = walk_forward_sessions(factory, ohlcv, features, instrument, dev_val_sessions, quantity, policy=split_policy, timeframe=timeframe)

    rob: RobustnessReport | None = None
    run_it = robustness == "always" or (robustness == "auto" and prelim.evidence.state is EvidenceState.ROBUSTNESS_TESTING)
    if run_it:
        base, o, f = run(("development", "validation"), "DEV_VAL")
        rob = run_robustness(make, o, f, instrument, base.trades, quantity, timeframe)
        if persist:
            persist_regime_stats(rob, name, instrument, timeframe, base.run_id, (base.manifest or {}).get("config_hash", ""))
    elif robustness == "auto":
        notes.append(f"robustness tests not run: earlier gates ended at {prelim.evidence.state.value}")

    report = validate_backtest(dev, dev_o, out_of_sample=oos, robustness=None if rob is None else rob.outcomes(), **common)

    holdout = None
    if include_holdout and report.evidence.state is EvidenceState.VALIDATED_FOR_RESEARCH:
        fp = holdout_fingerprint(ohlcv, features, split, data_source, timeframe, instrument)
        register_holdout(instrument, timeframe, split, fp)
        hold, _, _ = run(("holdout",), "HOLDOUT")
        h_oos = _oos(hold, quantity)
        cfg = (hold.manifest or {}).get("config_hash", "")
        record_holdout_evaluation(instrument, timeframe, fp, name, version, cfg, h_oos.n_trades, h_oos.net_expectancy_r, split)
        status: HoldoutStatus = holdout_status(instrument, timeframe, fp, split_policy)
        holdout = {"n_trades": h_oos.n_trades, "net_expectancy_r": h_oos.net_expectancy_r, "ci_low": h_oos.expectancy_ci_low,
                   "status": status.as_dict()}
        report = validate_backtest(dev, dev_o, out_of_sample=h_oos, robustness=rob.outcomes() if rob else None, **common)
        notes.append("final out-of-sample result is the holdout" + ("; holdout is no longer untouched: " + status.reason if status.consumed else ""))
    elif include_holdout:
        notes.append(f"holdout not touched: the strategy is {report.evidence.state.value}, not VALIDATED FOR RESEARCH")

    try:
        from quant_intelligence.backtesting.engine import market_for
        from quant_intelligence.validation.detail import build_detail

        detail = build_detail(dev.trades, val.trades, quantity, market_for(instrument), features, rob)
    except Exception as e:  # the dashboard detail is a convenience: it must never cost a verdict
        detail, notes = {"error": f"{type(e).__name__}: {e}"}, notes + [f"detail not built: {e}"]
    rv = ResearchValidation(name, version, instrument, timeframe, split, report, oos, wf, rob, holdout, notes)
    rv.detail = detail
    return _finish(rv, ohlcv, data_source, persist, proxy)


def _finish(rv: ResearchValidation, ohlcv: pd.DataFrame, data_source: str, persist: bool, proxy: bool = False) -> ResearchValidation:
    """Store the verdict for the ranker when asked (every state is stored, including the failures: they are evidence too)."""
    rv.proxy_data = proxy
    if persist:
        from quant_intelligence.data.quality_gates import dataset_fingerprint
        from quant_intelligence.validation.evidence_store import save_evidence

        save_evidence(rv, dataset_fingerprint(ohlcv, source=data_source, timeframe=rv.timeframe, instrument=rv.instrument))
    return rv
