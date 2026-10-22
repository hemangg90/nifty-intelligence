"""
The compact evidence detail stored beside each verdict, so the dashboard can draw it without re-running anything.

Curves are downsampled (at most `MAX_POINTS`), the distribution is pre-binned, and only the latest `MAX_TRADES` trade
records are kept, so a snapshot stays small. Everything is plain JSON types; NaN becomes None.

The equity curve is cumulative NET R over the development window followed by the validation window, in time order, with
each point labelled by segment, so the out-of-sample stretch is visible as such. It is a curve of past trades under the
documented assumptions, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_intelligence.validation.metrics import period_table, trade_table
from quant_intelligence.validation.robustness import label_trades, regime_table

MAX_POINTS = 400
MAX_TRADES = 300
HIST_RANGE = (-3.0, 5.0)     # net R values outside this are clipped into the end bins for display
HIST_BINS = 32


def _nn(v):
    return None if isinstance(v, float) and v != v else v


def _sample(n: int) -> np.ndarray:
    if n <= MAX_POINTS:
        return np.arange(n)
    idx = np.unique(np.linspace(0, n - 1, MAX_POINTS).round().astype(int))
    return idx


def build_detail(dev_trades, val_trades, quantity: int, market: str, features: pd.DataFrame, robustness_report=None) -> dict:
    dev = trade_table(dev_trades, quantity, market).assign(segment="development")
    val = trade_table(val_trades, quantity, market).assign(segment="validation")
    table = pd.concat([dev, val], ignore_index=True) if len(val) else dev
    if len(table) == 0:
        return {"n_trades": 0}
    table = table.sort_values("exit_ts", kind="stable").reset_index(drop=True)

    equity = table["net_r"].cumsum().to_numpy()
    drawdown = equity - np.maximum.accumulate(np.maximum(equity, 0.0))      # from the running peak, which starts at zero
    idx = _sample(len(table))
    curve = [[str(table["exit_ts"].iloc[i]), float(equity[i]), float(drawdown[i]), table["segment"].iloc[i]] for i in idx]

    counts, edges = np.histogram(np.clip(table["net_r"].to_numpy(), *HIST_RANGE), bins=HIST_BINS, range=HIST_RANGE)
    periods = period_table(table, "M")

    labelled = label_trades(table, features)
    regimes = regime_table(labelled)
    regime_rows = [{k: _nn(v) for k, v in r.items()} for r in regimes.to_dict("records")]

    recent = table.tail(MAX_TRADES)
    trades = [{"signal_ts": str(r.signal_ts), "exit_ts": str(r.exit_ts), "direction": r.direction, "holding_bars": int(r.holding_bars),
               "net_r": _nn(float(r.net_r)), "gross_r": _nn(float(r.gross_r)), "cost_r": _nn(float(r.cost_r)), "segment": r.segment}
              for r in recent.itertuples(index=False)]

    out = {
        "n_trades": int(len(table)),
        "equity": curve,
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts], "clipped_to": list(HIST_RANGE)},
        "periods": [{"period": str(r["period"]), "n": int(r["n"]), "net_expectancy_r": _nn(float(r["net_expectancy_r"])),
                     "net_pnl": _nn(float(r["net_pnl"]))} for _, r in periods.iterrows()],
        "regimes": regime_rows,
        "trades": trades,
        "trades_shown": len(trades),
    }
    if robustness_report is not None:
        out["robustness"] = robustness_report.as_dict()["tests"]
    return out
