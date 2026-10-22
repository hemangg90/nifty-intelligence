"""Uncertainty estimates: intervals, effective sample size, determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.tests.validation_helpers import edge_series, make_trades
from quant_intelligence.validation.metrics import trade_table
from quant_intelligence.validation.uncertainty import (
    effective_sample_size,
    estimate_uncertainty,
    intraclass_correlation,
    stationary_bootstrap_means,
    wilson_interval,
)

QTY = 75


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5961), abs=1e-3)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.2           # stays inside [0, 1] and is not degenerate at the edge
    assert all(np.isnan(wilson_interval(1, 0)))


def test_wilson_interval_widens_for_a_smaller_effective_sample():
    narrow, wide = wilson_interval(60, 100), wilson_interval(30, 50)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_is_deterministic_for_a_seed_and_differs_across_seeds():
    x = edge_series(120)
    a = stationary_bootstrap_means(x, 300, 4, seed=1)
    assert np.array_equal(a, stationary_bootstrap_means(x, 300, 4, seed=1))
    assert not np.array_equal(a, stationary_bootstrap_means(x, 300, 4, seed=2))


def test_bootstrap_distribution_centres_on_the_sample_mean_with_the_right_spread():
    x = np.random.default_rng(0).normal(0.0, 1.0, 400)
    boots = stationary_bootstrap_means(x, 2000, 1, seed=5)   # block length 1 reduces to the ordinary bootstrap
    assert boots.mean() == pytest.approx(x.mean(), abs=0.02)
    assert boots.std() == pytest.approx(1.0 / np.sqrt(400), rel=0.15)


def test_blocks_widen_the_interval_for_a_serially_dependent_series():
    rng = np.random.default_rng(11)
    e, ar = rng.normal(size=600), np.zeros(600)
    for i in range(1, 600):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    iid = rng.permutation(ar)                                 # same values, dependence destroyed
    se_block = stationary_bootstrap_means(ar, 1500, 12, seed=1).std()
    se_iid = stationary_bootstrap_means(iid, 1500, 1, seed=1).std()
    assert se_block > 1.5 * se_iid


def test_intraclass_correlation_extremes():
    groups = np.repeat(np.arange(50), 4)
    rng = np.random.default_rng(2)
    assert intraclass_correlation(np.repeat(rng.normal(size=50), 4), groups) == pytest.approx(1.0, abs=1e-9)
    assert intraclass_correlation(rng.normal(size=200), groups) < 0.15
    assert intraclass_correlation(rng.normal(size=10), np.arange(10)) == 0.0   # one trade per group: nothing to estimate


def test_independent_one_per_day_trades_keep_almost_their_full_count():
    ess, icc, phi, design = effective_sample_size(edge_series(300), np.arange(300))
    assert ess > 0.8 * 300 and design < 1.25


def test_clustered_trades_shrink_the_effective_sample():
    rng = np.random.default_rng(4)
    day_effect = np.repeat(rng.normal(0, 1, 60), 5)            # 5 trades a day sharing the day's regime
    x = day_effect + rng.normal(0, 0.3, 300)
    ess, icc, phi, design = effective_sample_size(x, np.repeat(np.arange(60), 5))
    assert icc > 0.8 and ess < 100 and design > 3.0


def test_effective_n_never_exceeds_the_raw_count():
    for seed in range(5):
        x = edge_series(150, seed=seed)
        assert effective_sample_size(x, np.repeat(np.arange(30), 5))[0] <= 150


def test_estimate_has_a_conservative_interval_that_contains_the_mean():
    table = trade_table(make_trades(edge_series(300, mean=0.3)), QTY)
    u = estimate_uncertainty(table)
    lo, hi = u.expectancy_ci
    assert lo < u.expectancy_r < hi
    assert lo <= min(u.expectancy_ci_bootstrap[0], u.expectancy_ci_normal[0]) + 1e-12   # the wider of the two
    assert u.win_rate_ci[1] - u.win_rate_ci[0] >= u.win_rate_ci_raw[1] - u.win_rate_ci_raw[0] - 1e-12
    assert u.n_resamples > 0


def test_estimate_is_reproducible():
    table = trade_table(make_trades(edge_series(200)), QTY)
    assert estimate_uncertainty(table).as_dict() == estimate_uncertainty(table).as_dict()


def test_interval_narrows_as_trades_accumulate():
    small = estimate_uncertainty(trade_table(make_trades(edge_series(80, seed=1)), QTY))
    large = estimate_uncertainty(trade_table(make_trades(edge_series(800, seed=1)), QTY))
    assert (large.expectancy_ci[1] - large.expectancy_ci[0]) < (small.expectancy_ci[1] - small.expectancy_ci[0])


def test_a_strategy_with_no_net_edge_does_not_show_a_positive_interval():
    positive = 0
    for seed in range(20):
        # underlying mean 0.1R is eaten by ~0.1R costs, so the true net edge is about zero
        u = estimate_uncertainty(trade_table(make_trades(np.random.default_rng(seed).normal(0.1, 1.0, 250)), QTY))
        positive += u.expectancy_ci[0] > 0
    assert positive <= 3   # a 95% interval may exclude zero by chance, but rarely


def test_degenerate_inputs_do_not_crash():
    empty = estimate_uncertainty(pd.DataFrame(columns=["net_r", "signal_date"]))
    assert empty.n_trades == 0 and np.isnan(empty.expectancy_r)
    one = estimate_uncertainty(trade_table(make_trades([1.0]), QTY))
    assert one.n_trades == 1 and np.isnan(one.expectancy_ci[0])
