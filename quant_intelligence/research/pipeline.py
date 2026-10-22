"""
Research Pipeline Orchestrator.

Ties together: DataManager -> feature engine -> market state -> regime engine
-> (per strategy) historical backtest -> analogue engine -> ranking engine ->
NO-TRADE decision -> setup detector. This is the single source of truth the
Streamlit app calls into, so the UI never re-implements pipeline logic.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.backtesting.engine import run_backtest, BacktestResult
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.data_manager import DataManager
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.market_state.market_state_engine import MarketState, build_current_state
from quant_intelligence.ranking.ranking_engine import RankingDecision, StrategyScore, rank_and_select
from quant_intelligence.research.ranking_core import assess_strategy
from quant_intelligence.regimes.regime_engine import classify_regime, regime_confidence
from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.utils.logging_utils import log_event


@dataclass
class StrategyIntelligence:
    strategy_name: str
    global_metrics: dict
    conditional_metrics: dict
    analogues: pd.DataFrame
    score: StrategyScore
    backtest: BacktestResult


@dataclass
class PipelineOutput:
    instrument: str
    timestamp: dt.datetime
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    market_state: MarketState
    data_quality_status: str
    regime_label: str
    regime_probabilities: dict
    regime_confidence: float
    strategy_intel: list[StrategyIntelligence]
    ranking: RankingDecision
    data_quality_issues: list = field(default_factory=list)
    # Populated only when GATED_RANKER is on: per-strategy gate results and the ungated choice, for the dashboard.
    decisions: list = field(default_factory=list)
    legacy_ranking: RankingDecision | None = None


def run_pipeline(
    instrument: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    quantity: int | None = None,
) -> PipelineOutput:
    quantity = quantity or default_quantity(instrument)
    market = profile_for(instrument).name
    data_manager = DataManager()
    ohlcv, metadata = data_manager.get_ohlcv(instrument, timeframe, start, end)

    if len(ohlcv) < 120:
        raise ValueError(
            f"Not enough data to run the pipeline for {instrument} {timeframe} "
            f"({len(ohlcv)} rows). Widen the date range."
        )

    features = compute_features(ohlcv, profile_for(instrument))
    data_quality_status = metadata["quality_status"]

    market_state = build_current_state(instrument, features, data_quality_status)
    if market_state is None:
        raise ValueError("Not enough warmup bars to build a current market state.")

    regime_label, regime_probs = classify_regime(market_state.features)
    conf = regime_confidence(regime_probs)

    log_event(
        "pipeline",
        f"Market state built for {instrument} at {market_state.timestamp}: regime={regime_label} (conf={conf:.2f})",
        level="INFO",
    )

    strategies = get_all_strategies()
    if float(ohlcv["volume"].fillna(0).sum()) <= 0:
        # No traded volume (e.g. index candles): volume features are NaN and NaN comparisons
        # silently pass, so volume-dependent strategies would emit unreliable signals.
        skipped = [s.name for s in strategies if _uses_volume(s)]
        strategies = [s for s in strategies if not _uses_volume(s)]
        log_event("pipeline", f"No volume in data; skipped volume-dependent strategies: {skipped}", level="WARNING")
    strategy_intel: list[StrategyIntelligence] = []
    scores: list[StrategyScore] = []

    feat_by_ts = features.set_index("timestamp")
    for strategy in strategies:
        # persist=False: this runs on every scan/page refresh. Persisting wrote ~700k rows (1.6 GB) and nothing reads
        # them back; the Backtest Lab page still persists its explicit runs.
        backtest = run_backtest(strategy, ohlcv, features, instrument, split="RESEARCH", quantity=quantity, persist=False)
        assessment = assess_strategy(
            strategy.name, backtest.trades, feat_by_ts, market_state.features, quantity,
            metrics=backtest.metrics, market=market,
        )

        strategy_intel.append(
            StrategyIntelligence(
                strategy_name=strategy.name,
                global_metrics=backtest.metrics,
                conditional_metrics=assessment.conditional,
                analogues=assessment.analogues,
                score=assessment.score,
                backtest=backtest,
            )
        )
        scores.append(assessment.score)

    ranking = rank_and_select(scores, data_quality_status, allow_tie_break=True)
    decisions: list = []
    legacy_ranking = None
    if SETTINGS.gated_ranker:
        from quant_intelligence.ranking.gated_ranker import gated_rank

        gated = gated_rank(strategies, scores, instrument, timeframe, data_quality_status, regime_label,
                           market_state=market_state, recent_ohlcv=ohlcv.tail(5))
        legacy_ranking, ranking, decisions = gated.legacy, gated.ranking, gated.decisions
        log_event("pipeline", f"Gated ranker: {sum(d.eligible for d in decisions)} of {len(decisions)} strategies eligible "
                              f"(ungated pick: {legacy_ranking.selected_strategy})", level="INFO")
    quality_issues = list(metadata.get("quality_report", {}).get("issues", []))
    if ranking.is_no_trade and data_quality_status != "OK" and quality_issues:
        # "Data quality is DEGRADED" alone does not tell the user what to fix; add the actual findings.
        ranking.reason = f"{ranking.reason} - " + "; ".join(quality_issues)

    log_event(
        "pipeline",
        f"Ranking decision: selected={ranking.selected_strategy}, no_trade={ranking.is_no_trade}, reason={ranking.reason}",
        level="INFO",
    )

    return PipelineOutput(
        instrument=instrument,
        timestamp=market_state.timestamp,
        ohlcv=ohlcv,
        features=features,
        market_state=market_state,
        data_quality_status=data_quality_status,
        regime_label=regime_label,
        regime_probabilities=regime_probs,
        regime_confidence=conf,
        strategy_intel=strategy_intel,
        ranking=ranking,
        data_quality_issues=quality_issues,
        decisions=decisions,
        legacy_ranking=legacy_ranking,
    )


def default_quantity(instrument: str, fallback: int = 50) -> int:
    """Units traded per backtest trade = the instrument's option lot size (this sets the weight of
    the fixed per-order brokerage). Falls back to `fallback` if the lot size cannot be resolved."""
    try:
        from quant_intelligence.options.option_selector import get_underlying_info

        return int(get_underlying_info(instrument)["lot_size"]) or fallback
    except Exception:
        return fallback


_VOLUME_FEATURES = {"relative_volume", "volume_acceleration", "vwap", "vwap_distance_pct"}


def _uses_volume(strategy) -> bool:
    return bool(_VOLUME_FEATURES & set(strategy.required_features)) or "volume" in strategy.name.lower()
