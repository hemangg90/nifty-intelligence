"""
Uncertainty around the point estimates in `metrics.py`.

A trade list is not an independent sample. Signals cluster inside a session, consecutive trades share a regime, and a
raw count of 800 trades can carry the information of far fewer. So:

* the **effective sample size** discounts the raw count for same-session clustering (intraclass correlation) and for
  serial correlation of consecutive trades, whichever is worse;
* the **expectancy interval** is the wider of two estimates: a stationary block bootstrap of the chronological net-R
  series (Politis-Romano; blocks of random geometric length preserve short-range dependence) and a normal interval
  using the effective sample size;
* the **win-rate interval** is a Wilson interval evaluated at the effective sample size.

Everything is seeded and deterministic. None of this removes the other ways a backtest can mislead (selection among
strategies, regime change, data errors); it only quantifies sampling noise given the trades that exist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quant_intelligence.config.settings import SETTINGS


def wilson_interval(successes: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Accepts a non-integer n (an effective sample size)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    p = min(max(successes / n, 0.0), 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def stationary_bootstrap_means(x: np.ndarray, n_resamples: int, mean_block: float, seed: int) -> np.ndarray:
    """Means of `n_resamples` stationary-bootstrap resamples of the ordered series `x`.

    Each resample walks the series: with probability 1/mean_block it jumps to a uniformly random start, otherwise it
    takes the next observation (wrapping around). Block lengths are therefore geometric with mean `mean_block`.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / max(mean_block, 1.0)
    idx = rng.integers(0, n, size=n_resamples)
    total = x[idx].astype(float).copy()
    for _ in range(1, n):
        restart = rng.random(n_resamples) < p_restart
        idx = np.where(restart, rng.integers(0, n, size=n_resamples), (idx + 1) % n)
        total += x[idx]
    return total / n


def intraclass_correlation(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA estimate of the correlation between two trades from the same group (session), clipped to [0, 1].
    0 when it cannot be estimated (fewer than two groups, or every group has a single trade)."""
    df = pd.DataFrame({"v": values, "g": groups})
    sizes = df.groupby("g")["v"].size()
    k, n = len(sizes), len(df)
    if k < 2 or n <= k:
        return 0.0
    grand = df["v"].mean()
    means = df.groupby("g")["v"].transform("mean")
    ssw = float(((df["v"] - means) ** 2).sum())
    gm = df.groupby("g")["v"].mean()
    ssb = float((sizes * (gm - grand) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - float((sizes ** 2).sum()) / n) / (k - 1)
    denom = msb + (m0 - 1.0) * msw
    if denom <= 0:
        return 0.0
    return float(min(max((msb - msw) / denom, 0.0), 1.0))


def lag1_autocorrelation(x: np.ndarray) -> float:
    if len(x) < 10 or float(np.std(x)) == 0.0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


@dataclass
class UncertaintyEstimate:
    n_trades: int
    effective_n: float
    icc: float
    lag1_autocorr: float
    design_effect: float
    confidence_level: float
    expectancy_r: float
    expectancy_ci: tuple[float, float]       # the conservative (wider) of bootstrap and effective-n normal
    expectancy_ci_bootstrap: tuple[float, float]
    expectancy_ci_normal: tuple[float, float]
    win_rate: float
    win_rate_ci: tuple[float, float]         # Wilson at the effective sample size
    win_rate_ci_raw: tuple[float, float]     # Wilson at the raw trade count (for comparison only)
    block_length: float
    n_resamples: int
    seed: int

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return {k: ([None if (isinstance(i, float) and i != i) else i for i in v] if isinstance(v, tuple) else
                    (None if isinstance(v, float) and v != v else v)) for k, v in d.items()}


def effective_sample_size(net_r: np.ndarray, sessions: np.ndarray) -> tuple[float, float, float, float]:
    """(effective_n, icc, lag1_autocorr, design_effect) for chronologically ordered trades and their session labels."""
    n = len(net_r)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0)
    icc = intraclass_correlation(net_r, sessions)
    mean_cluster = n / max(len(pd.unique(sessions)), 1)
    design = 1.0 + (mean_cluster - 1.0) * icc
    ess_cluster = n / design
    phi = lag1_autocorrelation(net_r)
    ess_serial = n * (1.0 - phi) / (1.0 + phi) if phi > 0 else float(n)
    ess = max(min(float(n), ess_cluster, ess_serial), 1.0)
    return (ess, icc, phi, float(n) / ess)


def estimate_uncertainty(
    table: pd.DataFrame,
    confidence: float | None = None,
    n_resamples: int | None = None,
    seed: int | None = None,
) -> UncertaintyEstimate:
    """Uncertainty for the trades in a `metrics.trade_table` (chronological by exit)."""
    pol = SETTINGS.evidence
    confidence = pol.confidence_level if confidence is None else confidence
    n_resamples = pol.bootstrap_resamples if n_resamples is None else n_resamples
    seed = pol.bootstrap_seed if seed is None else seed
    nan2 = (float("nan"), float("nan"))

    n = len(table)
    if n == 0:
        return UncertaintyEstimate(0, 0.0, 0.0, 0.0, 1.0, confidence, float("nan"), nan2, nan2, nan2,
                                   float("nan"), nan2, nan2, 0.0, n_resamples, seed)
    net_r = table["net_r"].to_numpy(dtype=float)
    ess, icc, phi, design = effective_sample_size(net_r, table["signal_date"].to_numpy())
    mean = float(net_r.mean())

    alpha = 1.0 - confidence
    block = max(1.0, round(n ** (1.0 / 3.0)))
    if n >= 2:
        boots = stationary_bootstrap_means(net_r, n_resamples, block, seed)
        ci_boot = (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))
        se = float(net_r.std(ddof=1)) / math.sqrt(ess)
        t = stats.t.ppf(1 - alpha / 2, max(ess - 1.0, 1.0))
        ci_norm = (mean - t * se, mean + t * se)
    else:
        ci_boot = ci_norm = nan2
    ci = (min(ci_boot[0], ci_norm[0]), max(ci_boot[1], ci_norm[1])) if n >= 2 else nan2

    wins = float((net_r > 0).sum())
    return UncertaintyEstimate(
        n_trades=n, effective_n=ess, icc=icc, lag1_autocorr=phi, design_effect=design, confidence_level=confidence,
        expectancy_r=mean, expectancy_ci=ci, expectancy_ci_bootstrap=ci_boot, expectancy_ci_normal=ci_norm,
        win_rate=wins / n, win_rate_ci=wilson_interval(wins / n * ess, ess, confidence),
        win_rate_ci_raw=wilson_interval(wins, n, confidence), block_length=block, n_resamples=n_resamples, seed=seed,
    )
