"""
One call from a completed backtest to its evidence state: metrics -> uncertainty -> classification.

`validate_backtest` is the seam the ranker (Phase 4) and the scheduler (Phase 6) use. It never ranks anything; it
answers "what is this backtest allowed to claim?" and keeps the numbers behind the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_intelligence.backtesting.engine import BacktestResult, market_for
from quant_intelligence.config.settings import EvidencePolicy
from quant_intelligence.strategies.rules import rules_for
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeframe import parse_timeframe_minutes
from quant_intelligence.validation.evidence import EvidenceAssessment, OutOfSampleResult, classify_evidence
from quant_intelligence.validation.metrics import PerformanceMetrics, compute_metrics, trade_table
from quant_intelligence.validation.uncertainty import UncertaintyEstimate, estimate_uncertainty


@dataclass
class ValidationReport:
    strategy: str
    instrument: str
    run_id: str
    config_hash: str
    metrics: PerformanceMetrics
    uncertainty: UncertaintyEstimate
    evidence: EvidenceAssessment

    def as_dict(self) -> dict:
        return {"strategy": self.strategy, "instrument": self.instrument, "run_id": self.run_id,
                "config_hash": self.config_hash, "metrics": self.metrics.as_dict(),
                "uncertainty": self.uncertainty.as_dict(), "evidence": self.evidence.as_dict()}


def validate_backtest(
    result: BacktestResult,
    ohlcv: pd.DataFrame | None = None,
    quantity: int = 50,
    data_quality: str | None = None,
    n_sessions: int | None = None,
    backtestable: bool | None = None,
    out_of_sample: OutOfSampleResult | None = None,
    robustness: dict[str, bool | None] | None = None,
    capital: float | None = None,
    policy: EvidencePolicy | None = None,
    data_limitations: list[str] | None = None,
) -> ValidationReport:
    """Evidence report for `result`. `ohlcv` (the data the backtest ran on) supplies the quality verdict, session count,
    bar count and date span when they are not passed explicitly; without it those inputs are simply unknown.
    `backtestable` defaults to the strategy's declared rules (chain-aware strategies are not)."""
    market = market_for(result.instrument)
    table = trade_table(result.trades, quantity, market)

    n_bars = len(ohlcv) if ohlcv is not None else None
    data_start = data_end = None
    if ohlcv is not None and len(ohlcv):
        ts = pd.to_datetime(ohlcv["timestamp"])
        data_start, data_end = ts.min(), ts.max()
        if n_sessions is None:
            n_sessions = int(ts.dt.date.nunique())
        if data_quality is None:
            from quant_intelligence.data.quality_gates import check_history

            tf = parse_timeframe_minutes((result.manifest or {}).get("timeframe", "5min"))
            data_quality = check_history(ohlcv, tf, profile_for(result.instrument)).status
    if backtestable is None:
        rules = rules_for(result.strategy_name)
        backtestable = True if rules is None else rules.backtestable

    metrics = compute_metrics(result.trades, quantity, market, n_bars=n_bars, n_sessions=n_sessions,
                              data_start=data_start, data_end=data_end, capital=capital, table=table)
    uncertainty = estimate_uncertainty(table)
    evidence = classify_evidence(metrics, uncertainty, data_quality=data_quality or "OK", backtestable=backtestable,
                                 n_sessions=n_sessions, out_of_sample=out_of_sample, robustness=robustness, policy=policy,
                                 data_limitations=data_limitations)
    return ValidationReport(result.strategy_name, result.instrument, result.run_id,
                            (result.manifest or {}).get("config_hash", ""), metrics, uncertainty, evidence)
