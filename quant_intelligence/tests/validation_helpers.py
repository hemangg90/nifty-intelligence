"""Hand-built trade lists with a known return profile, for the validation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_intelligence.strategies.base_strategy import Setup, TradeResult

ENTRY, STOP_DIST = 20000.0, 20.0  # NIFTY-like: risk per unit is 0.5 delta x 20 points, so brokerage is a small share of R


def make_trades(underlying_rs, start="2024-01-01", per_day=1, hold_bars=4):
    """One trade per R value, `per_day` per session on consecutive business days. A trade's underlying R is the given
    value; its net R is that minus the (small, constant) cost in R."""
    days = pd.bdate_range(start, periods=int(np.ceil(len(underlying_rs) / per_day)) + 1)
    trades = []
    for i, r in enumerate(underlying_rs):
        ts = pd.Timestamp(days[i // per_day]) + pd.Timedelta(hours=9, minutes=30 + 10 * (i % per_day))
        move = STOP_DIST * float(r)
        trades.append(TradeResult(
            setup=Setup(ts, "LONG", ENTRY, ENTRY - STOP_DIST, ENTRY + 2 * STOP_DIST),
            exit_timestamp=ts + pd.Timedelta(minutes=5 * hold_bars), exit_price=ENTRY + move, r_multiple=float(r),
            holding_period_bars=hold_bars, mfe=max(move, 0.0), mae=max(-move, 0.0), gross_pnl=move,
            outcome="WIN" if r > 0 else "LOSS" if r < 0 else "SCRATCH",
        ))
    return trades


def edge_series(n=400, mean=0.4, sd=1.0, seed=3):
    return np.random.default_rng(seed).normal(mean, sd, n)
