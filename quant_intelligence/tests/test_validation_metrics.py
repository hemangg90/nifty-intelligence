"""Performance metrics: definitions, identities and edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from quant_intelligence.backtesting.engine import max_drawdown_in_r
from quant_intelligence.tests.validation_helpers import edge_series, make_trades
from quant_intelligence.validation.metrics import compute_metrics, period_table, trade_table

QTY = 75


def test_net_is_gross_minus_cost_for_every_trade():
    t = trade_table(make_trades(edge_series(60)), QTY)
    assert np.allclose(t["gross_r"] - t["cost_r"], t["net_r"])
    assert np.allclose(t["gross_pnl"] - t["costs"], t["net_pnl"])
    assert (t["cost_r"] > 0).all()
    assert np.allclose(t["gross_r"], t["underlying_r"], atol=1e-9)  # fixed premium model: option R equals underlying R


def test_expectancy_identity_p_w_l_c_equals_mean_net_r():
    m = compute_metrics(make_trades(edge_series(300, mean=0.2)), QTY)
    assert m.expectancy_identity_r == pytest.approx(m.net_expectancy_r, abs=1e-9)
    assert m.net_expectancy_r == pytest.approx(m.gross_expectancy_r - m.avg_cost_r, abs=1e-9)


def test_a_high_win_rate_can_still_lose_money():
    # 70% winners of +0.3R, 30% losers of -1.5R: E = 0.7*0.3 - 0.3*1.5 = -0.24 R before costs
    rs = ([0.3] * 7 + [-1.5] * 3) * 30
    m = compute_metrics(make_trades(rs), QTY)
    assert m.gross_win_rate == pytest.approx(0.7)
    assert m.gross_expectancy_r == pytest.approx(-0.24, abs=1e-9)
    assert m.net_expectancy_r < -0.24 and m.profit_factor < 1.0


def test_profit_factor_and_win_loss_figures():
    trades = make_trades([2.0, 2.0, -1.0, -1.0, -1.0])
    m, t = compute_metrics(trades, QTY), trade_table(trades, QTY)
    expected = t.loc[t.net_pnl > 0, "net_pnl"].sum() / -t.loc[t.net_pnl < 0, "net_pnl"].sum()
    assert m.profit_factor == pytest.approx(expected)
    assert m.win_rate == pytest.approx(0.4)
    assert m.avg_win_r == pytest.approx(t.loc[t.net_r > 0, "net_r"].mean())
    assert m.avg_loss_r < 0


def test_profit_factor_without_losses_is_infinite_not_an_error():
    assert compute_metrics(make_trades([1.0, 1.5, 2.0]), QTY).profit_factor == float("inf")


def test_max_drawdown_matches_the_engine_definition():
    trades = make_trades([1, 1, -2, -2, 1, -3, 2])
    m, t = compute_metrics(trades, QTY), trade_table(trades, QTY)
    assert m.max_drawdown_r == pytest.approx(max_drawdown_in_r(t["net_r"].to_numpy()))
    assert m.max_drawdown_r < 0 and m.max_drawdown_rs < 0


def test_no_trades_gives_empty_metrics_not_fabricated_ones():
    m = compute_metrics([], QTY)
    assert m.n_trades == 0 and m.net_pnl_total == 0.0
    d = m.as_dict()
    assert d["net_expectancy_r"] is None and d["win_rate"] is None


def test_exposure_frequency_and_annualisation_need_the_data_span():
    trades = make_trades(edge_series(100), hold_bars=4, per_day=1)
    m = compute_metrics(trades, QTY, n_bars=7500, n_sessions=100)
    assert m.exposure_time == pytest.approx(100 * 4 / 7500)
    assert m.trades_per_session == pytest.approx(1.0)
    assert m.trades_per_year > 0 and m.annualized_return_pct is None  # no capital base given -> no percentage
    t = trade_table(trades, QTY)
    years = (t["exit_ts"].max() - t["signal_ts"].min()).days / 365.25
    with_capital = compute_metrics(trades, QTY, n_bars=7500, n_sessions=100, capital=500_000)
    assert with_capital.annualized_return_pct == pytest.approx(with_capital.net_pnl_total / 500_000 / years * 100)
    assert np.isnan(compute_metrics(trades, QTY).exposure_time)  # unknown bar count -> not guessed


def test_trades_are_counted_per_session_and_distinct_days():
    m = compute_metrics(make_trades(edge_series(60), per_day=3), QTY)
    assert m.n_trades == 60 and m.n_trade_days == 20


def test_period_variability_summary():
    rs = np.r_[np.full(40, 0.5), np.full(40, -0.5)]  # first ~two months profitable, then not
    trades = make_trades(rs, start="2024-01-01")
    ps = compute_metrics(trades, QTY).period_summary
    assert ps["n_periods"] >= 3 and 0.0 < ps["share_positive"] < 1.0
    assert ps["best_period_expectancy_r"] > 0 > ps["worst_period_expectancy_r"]
    assert period_table(trade_table(trades, QTY))["n"].sum() == 80


def test_thin_periods_are_ignored_in_the_variability_summary():
    assert compute_metrics(make_trades([0.5, -0.5, 0.5]), QTY).period_summary == {"n_periods": 0}
