"""
The five robustness tests. A strategy that is profitable on one fixed configuration, one cost assumption, one stretch
of calendar or one lucky ordering of trades has shown very little. Each test asks one way that could be true:

1. parameter sensitivity - do the neighbours of the default parameters (one parameter moved a step at a time, within the
   economically reasonable range) also work? The best in-sample setting is never reported or chosen.
2. regime sensitivity - is the edge a property of one market condition? Conditions are labelled from features known at
   the signal bar only. Each condition's expectancy is reported; low-count conditions are shown but do not vote.
3. cost stress - net expectancy as slippage is scaled up. Net R is linear in slippage, so the break-even multiple is exact.
4. period consistency - share of calendar months that are profitable, and how much of the profit the best months supply.
5. sequence analysis - stationary block-bootstrap of the trade order (blocks keep same-session clustering), giving the
   distribution of the worst drawdown and the chance a resampled run ends below zero. Order shuffling cannot be iid.

A test with too little data returns `None` (not run), which the evidence classifier treats as "robustness testing
incomplete", never as a pass. Pass marks are `RobustnessPolicy`; every result records the value it was judged on.
These tests describe sensitivity; passing them does not predict future profit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.backtesting.engine import market_for, run_backtest
from quant_intelligence.config.settings import SETTINGS, RobustnessPolicy
from quant_intelligence.regimes.regime_engine import classify_regime
from quant_intelligence.validation.evidence import REQUIRED_ROBUSTNESS_TESTS
from quant_intelligence.validation.metrics import MIN_TRADES_PER_PERIOD, period_table, trade_table
from quant_intelligence.validation.uncertainty import estimate_uncertainty

GAP_SESSION_FRACTION = 0.003   # |opening gap| of 0.3% or more marks a gap session
HIGH_VOL_PERCENTILE = 0.5      # trailing ATR percentile at or above this is "high volatility"
UNKNOWN = "UNKNOWN"


@dataclass
class RobustnessTest:
    name: str
    passed: bool | None            # None: could not be run on this data
    value: float | None = None     # the number judged
    threshold: float | None = None
    detail: str = ""
    rows: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        nn = lambda v: None if isinstance(v, float) and v != v else v  # noqa: E731
        return {"name": self.name, "passed": self.passed, "value": nn(self.value), "threshold": self.threshold,
                "detail": self.detail, "rows": [{k: nn(v) for k, v in r.items()} for r in self.rows]}


@dataclass
class RobustnessReport:
    tests: dict[str, RobustnessTest]
    regime_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def outcomes(self) -> dict[str, bool | None]:
        """The `robustness` argument of `classify_evidence`."""
        return {name: t.passed for name, t in self.tests.items()}

    def as_dict(self) -> dict:
        records = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
                   for r in self.regime_table.to_dict("records")]
        return {"tests": {n: t.as_dict() for n, t in self.tests.items()}, "regime_table": records}


# ---- 1. parameter sensitivity -------------------------------------------------------------------------------------

def neighbour_values(value: float, low: float, high: float, step: float, steps: int = 1) -> list[float]:
    """Values `1..steps` range-steps either side of `value`, clipped to [low, high], excluding `value` itself."""
    out: list[float] = []
    for k in range(1, steps + 1):
        for sign in (-1, 1):
            v = min(max(value + sign * k * step, low), high)
            if isinstance(value, int) and float(step).is_integer():
                v = int(round(v))
            else:
                v = round(float(v), 6)
            if v != value and v not in out:
                out.append(v)
    return out


def parameter_sensitivity(strategy_cls, ohlcv, features, instrument, quantity, baseline_r, timeframe="5min",
                          policy: RobustnessPolicy | None = None) -> RobustnessTest:
    """Re-run with each parameter moved to its neighbouring values (others at their current setting). The current
    setting includes any STOP_ATR_SCALE applied to the defaults, so the neighbourhood is centred on what actually runs."""
    pol = policy or SETTINGS.robustness
    base = strategy_cls()
    market = market_for(instrument)
    rows: list[dict] = []
    for key, (low, high, step) in base.parameter_ranges.items():
        if key not in base.parameters:
            continue
        cur = base.parameters[key]
        for nv in neighbour_values(cur, low, high, step, pol.param_steps):
            try:
                strat = strategy_cls({**base.parameters, key: nv})
            except ValueError as e:      # an invalid combination (for example a stop wider than the target rule allows)
                rows.append({"parameter": key, "value": nv, "skipped": str(e)})
                continue
            res = run_backtest(strat, ohlcv, features, instrument, split="PARAM_SENSITIVITY", quantity=quantity,
                               persist=False, timeframe=timeframe)
            t = trade_table(res.trades, quantity, market)
            exp = float(t["net_r"].mean()) if len(t) else float("nan")
            rows.append({"parameter": key, "from": cur, "value": nv, "n_trades": len(t),
                         "net_expectancy_r": None if exp != exp else exp})
    judged = [r for r in rows if r.get("net_expectancy_r") is not None]
    if len(judged) < 4:
        return RobustnessTest("parameter_sensitivity", None, None, pol.param_min_positive_share,
                              f"only {len(judged)} usable neighbours; need at least 4", rows)
    exps = np.array([r["net_expectancy_r"] for r in judged])
    share = float((exps > 0).mean())
    baseline_ok = baseline_r == baseline_r and baseline_r > 0
    passed = bool(baseline_ok and share >= pol.param_min_positive_share)
    detail = (f"{int((exps > 0).sum())} of {len(exps)} neighbouring settings have positive net expectancy "
              f"(median {np.median(exps):+.3f} R, worst {exps.min():+.3f} R; default {baseline_r:+.3f} R)")
    if not baseline_ok:
        detail += "; the default setting is not itself profitable, so there is no stable edge to test"
    return RobustnessTest("parameter_sensitivity", passed, share, pol.param_min_positive_share, detail, rows)


# ---- 2. regime sensitivity ----------------------------------------------------------------------------------------

def label_trades(table: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach the market condition at each trade's SIGNAL bar. Only features at that timestamp are read, and features
    are causal, so a label never uses information from after the signal."""
    if len(table) == 0:
        return table.assign(regime=[], trend=[], volatility=[], gap=[])
    feat = features.drop_duplicates("timestamp").set_index("timestamp")
    keys = pd.DatetimeIndex(table["signal_ts"])
    rows = feat.reindex(keys)
    regimes, trends, vols, gaps = [], [], [], []
    for _, r in rows.iterrows():
        d = r.to_dict()
        if d.get("trend_slope") is None or d.get("trend_slope") != d.get("trend_slope"):
            regimes.append(UNKNOWN)
            trends.append(UNKNOWN)
        else:
            label, _ = classify_regime(d)
            regimes.append(label)
            trends.append("TREND" if label.startswith("TREND_") else "NON_TREND")
        atrp = d.get("atr_percentile_100")
        vols.append(UNKNOWN if atrp is None or atrp != atrp else ("HIGH_VOL" if atrp >= HIGH_VOL_PERCENTILE else "LOW_VOL"))
        gap = d.get("gap_pct")
        gaps.append(UNKNOWN if gap is None or gap != gap else ("GAP_SESSION" if abs(gap) >= GAP_SESSION_FRACTION else "NO_GAP"))
    return table.assign(regime=regimes, trend=trends, volatility=vols, gap=gaps)


def regime_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Net expectancy and an interval for every condition, in every dimension."""
    rows = []
    for dim in ("regime", "trend", "volatility", "gap"):
        if dim not in labelled:
            continue
        for label, g in labelled.groupby(dim):
            u = estimate_uncertainty(g.reset_index(drop=True), n_resamples=500) if len(g) >= 2 else None
            rows.append({"dimension": dim, "label": label, "n_trades": len(g),
                         "net_expectancy_r": float(g["net_r"].mean()),
                         "ci_low": None if u is None else u.expectancy_ci[0], "ci_high": None if u is None else u.expectancy_ci[1]})
    return pd.DataFrame(rows, columns=["dimension", "label", "n_trades", "net_expectancy_r", "ci_low", "ci_high"])


def regime_sensitivity(labelled: pd.DataFrame, policy: RobustnessPolicy | None = None) -> tuple[RobustnessTest, pd.DataFrame]:
    """Fails when the edge lives in one regime (drop the best regime and it disappears) or when a well-populated
    condition loses more than the floor per trade. Conditions with few trades are listed but cannot fail the test."""
    pol = policy or SETTINGS.robustness
    table = regime_table(labelled)
    judged = table[(table["n_trades"] >= pol.regime_min_trades) & (table["label"] != UNKNOWN)]
    rows = table.to_dict("records")
    regimes = judged[judged["dimension"] == "regime"]
    if len(regimes) < 2:
        return RobustnessTest("regime_sensitivity", None, None, -pol.regime_worst_floor_r,
                              f"fewer than 2 regimes with at least {pol.regime_min_trades} trades", rows), table
    worst = judged.loc[judged["net_expectancy_r"].idxmin()]
    contrib = labelled[labelled["regime"].isin(regimes["label"])].groupby("regime")["net_r"].sum()
    best_regime = contrib.idxmax()
    rest = labelled[(labelled["regime"] != best_regime) & (labelled["regime"] != UNKNOWN)]["net_r"]
    ex_best = float(rest.mean()) if len(rest) else float("nan")
    survives = ex_best == ex_best and ex_best > 0
    floor_ok = float(worst["net_expectancy_r"]) >= -pol.regime_worst_floor_r
    detail = (f"worst condition {worst['dimension']}={worst['label']} {worst['net_expectancy_r']:+.3f} R over {int(worst['n_trades'])} trades "
              f"(floor -{pol.regime_worst_floor_r}); without the best regime ({best_regime}) expectancy is {ex_best:+.3f} R")
    return RobustnessTest("regime_sensitivity", bool(survives and floor_ok), float(worst["net_expectancy_r"]),
                          -pol.regime_worst_floor_r, detail, rows), table


# ---- 3. cost stress -----------------------------------------------------------------------------------------------

def cost_stress(trades, quantity: int, market: str, policy: RobustnessPolicy | None = None) -> RobustnessTest:
    """Net expectancy with slippage scaled by each scenario multiple. Statutory charges are not uncertain so they are
    not scaled. Slippage stands in for spread widening as well."""
    pol = policy or SETTINGS.robustness
    if not trades:
        return RobustnessTest("cost_stress", None, None, pol.cost_stress_multiple, "no trades")
    base_costs = SETTINGS.costs
    rows, by_mult = [], {}
    for k in sorted(set(pol.cost_scenarios) | {1.0, 2.0, pol.cost_stress_multiple}):
        costs = dataclasses.replace(base_costs, slippage_ticks=base_costs.slippage_ticks * k)
        t = trade_table(trades, quantity, market, cost_model=costs)
        by_mult[k] = float(t["net_r"].mean())
        rows.append({"slippage_multiple": k, "net_expectancy_r": by_mult[k], "avg_cost_r": float(t["cost_r"].mean())})
    e1, e2 = by_mult[1.0], by_mult[2.0]
    slope = e1 - e2                      # net R lost per +1x of slippage; exactly linear in the multiple
    breakeven = (e1 + slope) / slope if slope > 1e-12 else float("inf")
    at_stress = by_mult[pol.cost_stress_multiple]
    detail = (f"net expectancy {at_stress:+.3f} R at {pol.cost_stress_multiple:g}x slippage "
              f"({e1:+.3f} R at 1x); break-even slippage {'never' if breakeven == float('inf') else f'{max(breakeven, 0):.2f}x'}")
    return RobustnessTest("cost_stress", bool(at_stress > 0), at_stress, 0.0, detail, rows)


# ---- 4. period consistency ----------------------------------------------------------------------------------------

def period_consistency(table: pd.DataFrame, policy: RobustnessPolicy | None = None) -> RobustnessTest:
    pol = policy or SETTINGS.robustness
    periods = period_table(table, "M")
    usable = periods[periods["n"] >= MIN_TRADES_PER_PERIOD] if len(periods) else periods
    rows = [{"period": str(r["period"]), "n_trades": int(r["n"]), "net_expectancy_r": float(r["net_expectancy_r"]),
             "net_pnl": float(r["net_pnl"])} for _, r in periods.iterrows()]
    if len(usable) < pol.min_periods:
        return RobustnessTest("period_consistency", None, None, pol.min_positive_period_share,
                              f"{len(usable)} usable calendar months; need at least {pol.min_periods}", rows)
    share_pos = float((usable["net_expectancy_r"] > 0).mean())
    total = float(usable["net_pnl"].sum())
    top = float(usable["net_pnl"].nlargest(pol.top_periods).sum())
    top_share = top / total if total > 0 else float("inf")
    passed = share_pos >= pol.min_positive_period_share and total > 0 and top_share <= pol.max_top_period_profit_share
    detail = (f"{int((usable['net_expectancy_r'] > 0).sum())} of {len(usable)} months profitable; "
              + (f"the best {pol.top_periods} months supply {top_share:.0%} of net profit" if total > 0 else "total net profit is not positive"))
    return RobustnessTest("period_consistency", bool(passed), share_pos, pol.min_positive_period_share, detail, rows)


# ---- 5. sequence analysis -----------------------------------------------------------------------------------------

def stationary_bootstrap_paths(x: np.ndarray, n_resamples: int, mean_block: float, seed: int) -> np.ndarray:
    """(n_resamples, len(x)) resampled series: blocks of geometric length with mean `mean_block`, wrapping around."""
    n = len(x)
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_resamples)
    for j in range(1, n):
        restart = rng.random(n_resamples) < p_restart
        idx[:, j] = np.where(restart, rng.integers(0, n, size=n_resamples), (idx[:, j - 1] + 1) % n)
    return x[idx]


def sequence_analysis(table: pd.DataFrame, policy: RobustnessPolicy | None = None) -> RobustnessTest:
    """Drawdown and loss risk when the same trades arrive in a different (block-preserving) order."""
    pol = policy or SETTINGS.robustness
    ev = SETTINGS.evidence
    n = len(table)
    if n < 30:
        return RobustnessTest("sequence_analysis", None, None, ev.max_drawdown_r, f"only {n} trades; need at least 30")
    x = table["net_r"].to_numpy(dtype=float)
    per_session = n / max(table["signal_date"].nunique(), 1)
    block = max(1.0, round(n ** (1 / 3)), per_session)   # keeps same-session trades together
    paths = stationary_bootstrap_paths(x, pol.sequence_resamples, block, ev.bootstrap_seed)
    equity = np.cumsum(paths, axis=1)
    max_dd = (equity - np.maximum.accumulate(equity, axis=1)).min(axis=1)    # <= 0
    dd95 = float(-np.quantile(max_dd, 0.05))                                  # the drawdown 95% of orderings stay within
    p_loss = float((equity[:, -1] < 0).mean())
    passed = dd95 <= ev.max_drawdown_r and p_loss <= pol.sequence_max_p_loss
    detail = (f"95th-percentile worst drawdown {dd95:.1f} R (limit {ev.max_drawdown_r:g}); "
              f"{p_loss:.0%} of reordered runs end below zero (limit {pol.sequence_max_p_loss:.0%}); block length {block:.0f}")
    rows = [{"drawdown_p50": float(-np.quantile(max_dd, 0.5)), "drawdown_p95": dd95, "drawdown_p99": float(-np.quantile(max_dd, 0.01)),
             "p_end_below_zero": p_loss, "observed_drawdown": float(-(np.cumsum(x) - np.maximum.accumulate(np.cumsum(x))).min())}]
    return RobustnessTest("sequence_analysis", bool(passed), dd95, ev.max_drawdown_r, detail, rows)


# ---- orchestration ------------------------------------------------------------------------------------------------

def run_robustness(strategy_cls, ohlcv, features, instrument, trades, quantity: int, timeframe: str = "5min",
                   policy: RobustnessPolicy | None = None) -> RobustnessReport:
    """All five tests on the development window. `trades` is the baseline backtest on (ohlcv, features)."""
    market = market_for(instrument)
    table = trade_table(trades, quantity, market)
    baseline = float(table["net_r"].mean()) if len(table) else float("nan")
    labelled = label_trades(table, features)
    regime_result, regimes = regime_sensitivity(labelled, policy)
    tests = {
        "parameter_sensitivity": parameter_sensitivity(strategy_cls, ohlcv, features, instrument, quantity, baseline, timeframe, policy),
        "regime_sensitivity": regime_result,
        "cost_stress": cost_stress(trades, quantity, market, policy),
        "period_consistency": period_consistency(table, policy),
        "sequence_analysis": sequence_analysis(table, policy),
    }
    assert set(tests) == set(REQUIRED_ROBUSTNESS_TESTS)
    return RobustnessReport(tests, regimes)


def persist_regime_stats(report: RobustnessReport, strategy: str, instrument: str, timeframe: str, run_id: str,
                         config_hash: str) -> int:
    """Append the strategy x regime table to `strategy_regime_stats`. Returns rows written; never raises."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import StrategyRegimeStat

        clean = lambda v: None if v is None or v != v else float(v)  # noqa: E731
        with get_session() as s:
            for r in report.regime_table.to_dict("records"):
                s.add(StrategyRegimeStat(strategy=strategy, instrument=instrument, timeframe=timeframe, dimension=r["dimension"],
                                         label=str(r["label"]), n_trades=int(r["n_trades"]), net_expectancy_r=clean(r["net_expectancy_r"]),
                                         ci_low=clean(r["ci_low"]), ci_high=clean(r["ci_high"]), run_id=run_id, config_hash=config_hash))
        return len(report.regime_table)
    except Exception:
        return 0
