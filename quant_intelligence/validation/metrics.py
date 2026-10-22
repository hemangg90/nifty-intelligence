"""
Performance metrics for a completed backtest, computed once, in one place.

Everything here is derived from the trade list the engine produced, using the engine's own cost and risk
functions (`_apply_costs`, `trade_risk`) so a metric can never disagree with the cost model. Point estimates only:
the uncertainty around them lives in `uncertainty.py`, and what they are allowed to conclude lives in `evidence.py`.

Units. "R" is net-of-cost P&L divided by the amount risked on the trade (premium lost if the stop is hit). Gross R is
the same before brokerage, charges and slippage, so `gross_r - cost_r = net_r` holds trade by trade. Rupee figures are
for the backtest quantity. Expectancy follows E = p*W - (1-p)*L - C with W and L the mean gross win and loss
magnitudes, p the share of gross winners and C the mean cost, all in R, and equals the mean net R exactly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.backtesting.engine import _apply_costs, max_drawdown_in_r, trade_risk
from quant_intelligence.strategies.base_strategy import TradeResult

TRADING_DAYS_PER_YEAR = 252.0
MIN_TRADES_PER_PERIOD = 5  # a calendar month with fewer trades is too thin to count as a period


def trade_table(trades: list[TradeResult], quantity: int, market: str = "NSE", cost_model=None) -> pd.DataFrame:
    """One row per trade, chronological by exit, with gross/net in both rupees and R. `cost_model` overrides the
    configured CostAssumptions (stress scenarios)."""
    rows = []
    for t in trades:
        commissions, fees, slippage, net_pnl = _apply_costs(t, t.setup.entry_price, quantity, market, cost_model)
        costs = commissions + fees + slippage
        risk = trade_risk(t, quantity)
        rows.append({
            "signal_ts": t.setup.timestamp,
            "exit_ts": t.exit_timestamp if t.exit_timestamp is not None else t.setup.timestamp,
            "direction": t.setup.direction,
            "holding_bars": t.holding_period_bars,
            "net_pnl": net_pnl,
            "costs": costs,
            "gross_pnl": net_pnl + costs,
            "risk": risk,
            "net_r": net_pnl / risk if risk > 0 else 0.0,
            "gross_r": (net_pnl + costs) / risk if risk > 0 else 0.0,
            "cost_r": costs / risk if risk > 0 else 0.0,
            "underlying_r": t.r_multiple,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["signal_ts"] = pd.to_datetime(df["signal_ts"])
        df["exit_ts"] = pd.to_datetime(df["exit_ts"])
        df = df.sort_values(["exit_ts", "signal_ts"]).reset_index(drop=True)
        df["signal_date"] = df["signal_ts"].dt.date
    return df


@dataclass
class PerformanceMetrics:
    n_trades: int = 0
    n_trade_days: int = 0                 # distinct sessions containing a trade
    # Sample
    win_rate: float = float("nan")        # share of trades with positive NET P&L
    gross_win_rate: float = float("nan")  # share with positive P&L before costs (the p in E = pW - (1-p)L - C)
    avg_win_r: float = float("nan")       # mean net R of winning trades
    avg_loss_r: float = float("nan")      # mean net R of losing trades (negative)
    avg_gross_win_r: float = float("nan")  # W
    avg_gross_loss_r: float = float("nan")  # L, as a positive magnitude
    avg_cost_r: float = float("nan")      # C
    # Edge
    net_expectancy_r: float = float("nan")
    gross_expectancy_r: float = float("nan")
    net_expectancy_rs: float = float("nan")  # per trade, rupees
    expectancy_identity_r: float = float("nan")  # p*W - (1-p)*L - C, recomputed from the parts (must equal net_expectancy_r)
    avg_r_multiple: float = float("nan")  # mean underlying R (price move / stop distance), before costs and option effects
    profit_factor: float = float("nan")   # gross profit / gross loss of NET P&L; inf when there are no losses
    # Risk
    max_drawdown_r: float = float("nan")  # negative, cumulative net R
    max_drawdown_rs: float = float("nan")  # negative, rupees
    net_r_std: float = float("nan")       # per-trade volatility of net R
    daily_pnl_std_rs: float = float("nan")
    # Activity
    exposure_time: float = float("nan")   # share of bars with a position open
    trades_per_session: float = float("nan")
    trades_per_year: float = float("nan")
    # Totals
    net_pnl_total: float = 0.0
    gross_pnl_total: float = 0.0
    total_costs: float = 0.0
    annualized_net_pnl_rs: float = float("nan")
    annualized_return_pct: float | None = None  # only when a capital base is given (simple, not compounded)
    # Variability across calendar months
    period_summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def period_table(table: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Net expectancy and P&L by calendar period of exit."""
    if len(table) == 0:
        return pd.DataFrame(columns=["period", "n", "net_expectancy_r", "net_pnl"])
    g = table.groupby(table["exit_ts"].dt.to_period(freq))
    out = g.agg(n=("net_r", "size"), net_expectancy_r=("net_r", "mean"), net_pnl=("net_pnl", "sum")).reset_index()
    return out.rename(columns={"exit_ts": "period"})


def _period_summary(table: pd.DataFrame) -> dict:
    periods = period_table(table)
    usable = periods[periods["n"] >= MIN_TRADES_PER_PERIOD]
    if len(usable) == 0:
        return {"n_periods": 0}
    exp = usable["net_expectancy_r"]
    total = float(table["net_pnl"].sum())
    best = float(usable["net_pnl"].max())
    return {
        "n_periods": int(len(usable)),
        "share_positive": float((exp > 0).mean()),
        "expectancy_std": float(exp.std(ddof=1)) if len(usable) > 1 else float("nan"),
        "worst_period_expectancy_r": float(exp.min()),
        "best_period_expectancy_r": float(exp.max()),
        "best_period_share_of_profit": (best / total) if total > 0 and best > 0 else None,
    }


def compute_metrics(
    trades: list[TradeResult],
    quantity: int,
    market: str = "NSE",
    n_bars: int | None = None,
    n_sessions: int | None = None,
    data_start: dt.datetime | None = None,
    data_end: dt.datetime | None = None,
    capital: float | None = None,
    table: pd.DataFrame | None = None,
) -> PerformanceMetrics:
    """All point metrics for `trades`. `n_bars`/`n_sessions`/`data_start`/`data_end` describe the data the backtest ran
    on, and are needed for exposure, frequency and annualisation; without them those fields stay NaN rather than guessed."""
    table = trade_table(trades, quantity, market) if table is None else table
    m = PerformanceMetrics(n_trades=len(table))
    if len(table) == 0:
        return m

    net_r, gross_r, cost_r = table["net_r"].to_numpy(), table["gross_r"].to_numpy(), table["cost_r"].to_numpy()
    net_pnl = table["net_pnl"].to_numpy()
    m.n_trade_days = int(table["signal_date"].nunique())

    wins, gross_wins = net_r > 0, gross_r > 0
    m.win_rate = float(wins.mean())
    m.gross_win_rate = float(gross_wins.mean())
    m.avg_win_r = float(net_r[wins].mean()) if wins.any() else float("nan")
    m.avg_loss_r = float(net_r[~wins].mean()) if (~wins).any() else float("nan")
    m.avg_gross_win_r = float(gross_r[gross_wins].mean()) if gross_wins.any() else 0.0
    m.avg_gross_loss_r = float(-gross_r[~gross_wins].mean()) if (~gross_wins).any() else 0.0
    m.avg_cost_r = float(cost_r.mean())

    m.net_expectancy_r = float(net_r.mean())
    m.gross_expectancy_r = float(gross_r.mean())
    m.net_expectancy_rs = float(net_pnl.mean())
    p = m.gross_win_rate
    m.expectancy_identity_r = p * m.avg_gross_win_r - (1 - p) * m.avg_gross_loss_r - m.avg_cost_r
    m.avg_r_multiple = float(table["underlying_r"].mean())

    gross_profit, gross_loss = float(net_pnl[net_pnl > 0].sum()), float(-net_pnl[net_pnl < 0].sum())
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))

    dd_r = max_drawdown_in_r(net_r)
    m.max_drawdown_r = float(dd_r) if dd_r is not None else float("nan")
    equity = np.cumsum(net_pnl)
    m.max_drawdown_rs = float((equity - np.maximum.accumulate(equity)).min())
    m.net_r_std = float(net_r.std(ddof=1)) if len(net_r) > 1 else float("nan")
    daily = table.groupby(table["exit_ts"].dt.date)["net_pnl"].sum()
    m.daily_pnl_std_rs = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")

    m.net_pnl_total = float(net_pnl.sum())
    m.total_costs = float(table["costs"].sum())
    m.gross_pnl_total = float(table["gross_pnl"].sum())

    if n_bars:
        m.exposure_time = float(min(table["holding_bars"].sum() / n_bars, 1.0))
    if n_sessions:
        m.trades_per_session = len(table) / n_sessions
    start = data_start if data_start is not None else table["signal_ts"].min()
    end = data_end if data_end is not None else table["exit_ts"].max()
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years > 0:
        m.trades_per_year = len(table) / years
        m.annualized_net_pnl_rs = m.net_pnl_total / years
        if capital and capital > 0:
            m.annualized_return_pct = m.net_pnl_total / capital / years * 100.0

    m.period_summary = _period_summary(table)
    return m
