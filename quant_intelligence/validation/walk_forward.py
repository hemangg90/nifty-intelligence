"""
Session-aligned walk-forward evaluation.

No parameter is fitted anywhere in this pipeline (strategy parameters are fixed in advance), so there is no training
window to roll: each fold is a consecutive block of whole sessions that the strategy is run on cold, as it would have
been live. What the folds add over a single backtest is *consistency over time*: how many blocks were profitable and how
widely the block results disperse. Purge drops the last sessions of a fold and embargo the first of the next so no trade
or indicator state spans a boundary. If parameter selection is ever added, it must happen on data before each fold.

`backtesting.engine.walk_forward` is the older bar-count version, kept for the Backtest Lab; this one is the research
standard.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quant_intelligence.backtesting.engine import market_for, run_backtest
from quant_intelligence.config.settings import SETTINGS, SplitPolicy
from quant_intelligence.strategies.base_strategy import BaseStrategy
from quant_intelligence.validation.metrics import trade_table
from quant_intelligence.validation.splits import slice_sessions

MIN_FOLD_TRADES = 10  # a fold with fewer trades is reported but does not vote on consistency


@dataclass
class FoldResult:
    fold: int
    start: dt.date
    end: dt.date
    n_sessions: int
    n_trades: int
    net_expectancy_r: float

    def as_dict(self) -> dict:
        return {"fold": self.fold, "start": str(self.start), "end": str(self.end), "n_sessions": self.n_sessions,
                "n_trades": self.n_trades, "net_expectancy_r": None if self.net_expectancy_r != self.net_expectancy_r else self.net_expectancy_r}


@dataclass
class WalkForwardSummary:
    folds: list[FoldResult] = field(default_factory=list)
    n_judged: int = 0                  # folds with at least MIN_FOLD_TRADES trades
    share_positive: float = float("nan")
    mean_expectancy_r: float = float("nan")
    expectancy_std: float = float("nan")  # dispersion across folds
    pooled_expectancy_r: float = float("nan")
    pooled_trades: int = 0
    worst_fold_r: float = float("nan")

    def as_dict(self) -> dict:
        nn = lambda v: None if isinstance(v, float) and v != v else v  # noqa: E731
        return {"folds": [f.as_dict() for f in self.folds], "n_judged": self.n_judged, "share_positive": nn(self.share_positive),
                "mean_expectancy_r": nn(self.mean_expectancy_r), "expectancy_std": nn(self.expectancy_std),
                "pooled_expectancy_r": nn(self.pooled_expectancy_r), "pooled_trades": self.pooled_trades,
                "worst_fold_r": nn(self.worst_fold_r)}


def fold_sessions(days: list[dt.date], n_folds: int, purge: int, embargo: int) -> list[list[dt.date]]:
    """Cut `days` into `n_folds` consecutive blocks, then purge the tail of each and embargo the head of each after
    the first. Blocks left with no sessions are dropped."""
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    size = len(days) // n_folds
    if size < 1:
        return []
    out = []
    for i in range(n_folds):
        block = days[i * size:(i + 1) * size if i < n_folds - 1 else len(days)]
        if i > 0 and embargo:
            block = block[embargo:]
        if i < n_folds - 1 and purge:
            block = block[:-purge] if len(block) > purge else []
        if block:
            out.append(block)
    return out


def walk_forward_sessions(
    strategy_factory: Callable[[], BaseStrategy],
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    sessions: list[dt.date],
    quantity: int = 50,
    n_folds: int | None = None,
    policy: SplitPolicy | None = None,
    timeframe: str = "5min",
) -> WalkForwardSummary:
    """Run the strategy on each consecutive block of `sessions` and summarise the blocks' net expectancy."""
    p = policy or SETTINGS.splits
    blocks = fold_sessions(sorted(sessions), n_folds or p.walk_forward_folds, p.purge_sessions, p.embargo_sessions)
    market = market_for(instrument)
    summary = WalkForwardSummary()
    pooled: list[np.ndarray] = []
    for i, block in enumerate(blocks, start=1):
        o, f = slice_sessions(ohlcv, features, block[0], block[-1])
        if len(o) < 20:
            continue
        result = run_backtest(strategy_factory(), o, f, instrument, split=f"WF_FOLD_{i}", quantity=quantity,
                              persist=False, timeframe=timeframe)
        table = trade_table(result.trades, quantity, market)
        exp = float(table["net_r"].mean()) if len(table) else float("nan")
        summary.folds.append(FoldResult(i, block[0], block[-1], len(block), len(table), exp))
        if len(table):
            pooled.append(table["net_r"].to_numpy(dtype=float))

    judged = [f.net_expectancy_r for f in summary.folds if f.n_trades >= MIN_FOLD_TRADES]
    summary.n_judged = len(judged)
    if judged:
        arr = np.array(judged)
        summary.share_positive = float((arr > 0).mean())
        summary.mean_expectancy_r = float(arr.mean())
        summary.expectancy_std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        summary.worst_fold_r = float(arr.min())
    if pooled:
        allr = np.concatenate(pooled)
        summary.pooled_expectancy_r, summary.pooled_trades = float(allr.mean()), int(len(allr))
    return summary
