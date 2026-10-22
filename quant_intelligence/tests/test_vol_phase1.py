"""Phase 1 of the volatility layer: estimators, Black-Scholes, expiry rules, and the model-based backtest
premium (flag-gated; flags off must leave every existing number untouched)."""
import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.backtesting import engine as engine_module
from quant_intelligence.backtesting.engine import _apply_costs, net_r_multiple
from quant_intelligence.config import settings as settings_module
from quant_intelligence.config.expiry_rules import next_expiry, rule_for, time_to_expiry_years, trading_days_between
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.backtest_premium import BacktestPremiumModel
from quant_intelligence.strategies.base_strategy import Setup, TradeResult
from quant_intelligence.utils.market_profile import MCX, NSE
from quant_intelligence.volatility import estimators as est


# ---------------------------------------------------------------- helpers
def gbm_ohlc(n_days=60, bars=75, sigma_annual=0.20, seed=1, start="2026-06-01"):
    """Intraday GBM bars with a known annual volatility (no overnight gap beyond the bar returns)."""
    rng = np.random.default_rng(seed)
    per_bar = sigma_annual / math.sqrt(252 * bars)
    days = pd.bdate_range(start, periods=n_days)
    rows, price = [], 20000.0
    for day in days:
        stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=bars, freq="5min")
        for ts in stamps:
            o = price
            path = o * np.exp(np.cumsum(rng.normal(0, per_bar / 3, 3)))  # 3 sub-steps give a real high/low
            c = o * math.exp(rng.normal(0, per_bar))
            hi, lo = max(o, c, *path), min(o, c, *path)
            rows.append((ts, o, hi, lo, c, 100.0))
            price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------- estimators
def test_close_close_matches_a_hand_calculation():
    close = pd.Series([100.0, 101.0, 100.0, 102.0, 101.0])
    r = np.log(close / close.shift(1)).dropna()
    expected = r.std(ddof=1) * math.sqrt(252)
    assert est.close_close_vol(close, 4, 252).iloc[-1] == pytest.approx(expected)
    assert est.close_close_vol(close, 4, 252).iloc[:3].isna().all()  # window not full yet


def test_parkinson_formula_on_one_bar():
    high, low = pd.Series([102.0]), pd.Series([100.0])
    expected = math.sqrt(math.log(1.02) ** 2 / (4 * math.log(2)) * 252)
    assert est.parkinson_vol(high, low, 1, 252).iloc[0] == pytest.approx(expected)


def test_range_estimators_recover_a_known_volatility():
    df = gbm_ohlc(n_days=40, sigma_annual=0.20)
    ppy = est.periods_per_year(NSE, 5)
    window = 75 * 20
    cc = est.close_close_vol(df["close"], window, ppy).iloc[-1]
    pk = est.parkinson_vol(df["high"], df["low"], window, ppy).iloc[-1]
    gk = est.garman_klass_vol(df["open"], df["high"], df["low"], df["close"], window, ppy).iloc[-1]
    yz = est.yang_zhang_vol(df["open"], df["high"], df["low"], df["close"], window, ppy).iloc[-1]
    assert cc == pytest.approx(0.20, rel=0.12)
    for value in (pk, gk, yz):
        assert 0.10 < value < 0.30  # range estimators are noisier/biased on this coarse bar construction


def test_yang_zhang_needs_a_window_of_two():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        est.yang_zhang_vol(s, s, s, s, 1, 252)


def test_ewma_follows_the_riskmetrics_recursion():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.03])
    out = est.ewma_variance(r, lam=0.94, min_periods=1)
    v = r.iloc[0] ** 2
    for x in r.iloc[1:]:
        v = 0.94 * v + 0.06 * x * x
    assert out.iloc[-1] == pytest.approx(v)
    with pytest.raises(ValueError):
        est.ewma_variance(r, lam=0.94, halflife=5)


def test_rolling_percentile_is_causal_and_bounded():
    s = pd.Series([1.0, 2.0, 3.0, 2.5, 10.0])
    p = est.rolling_percentile(s, 5, min_periods=2)
    assert p.iloc[-1] == 1.0 and (p.dropna() <= 1).all() and (p.dropna() > 0).all()


def test_ewma_model_ignores_data_after_asof():
    daily = pd.DataFrame({"timestamp": pd.bdate_range("2026-01-01", periods=60), "close": 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 60)))})
    asof = daily["timestamp"].iloc[40]
    base = est.EwmaModel().fit(daily, asof).forecast(5)
    spiked = daily.copy()
    spiked.loc[spiked["timestamp"] > asof, "close"] *= 3.0  # future shock must not leak
    again = est.EwmaModel().fit(spiked, asof).forecast(5)
    assert base.sigma_1d == pytest.approx(again.sigma_1d)
    assert base.sigma_horizon == pytest.approx(base.sigma_1d * math.sqrt(5)) and base.asof == pd.Timestamp(asof)


# ---------------------------------------------------------------- intraday vol: causal + sensible
def test_intraday_vol_is_close_to_the_truth_and_has_no_look_ahead():
    df = gbm_ohlc(n_days=60, sigma_annual=0.20)
    vol = est.intraday_ewma_annualised_vol(df, NSE, 5)
    assert vol.iloc[-1] == pytest.approx(0.20, rel=0.25)
    assert vol.iloc[:70].isna().all()  # needs a day of bars first

    cut = len(df) // 2
    head = est.intraday_ewma_annualised_vol(df.iloc[:cut].copy(), NSE, 5)
    mutated = df.copy()
    mutated.loc[cut:, ["open", "high", "low", "close"]] *= 1.5  # wreck the future
    after = est.intraday_ewma_annualised_vol(mutated, NSE, 5)
    np.testing.assert_allclose(head.dropna().values, after.iloc[:cut].dropna().values, rtol=1e-9)


def test_infer_tf_and_bars_per_year():
    assert est.infer_tf_minutes(gbm_ohlc(n_days=3)) == 5
    assert est.periods_per_year(NSE, 5) == 252 * 75
    assert est.periods_per_year(MCX, 5) == 252 * 895 / 5


# ---------------------------------------------------------------- Black-Scholes
def test_black_scholes_matches_hulls_textbook_example():
    # Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81, call delta 0.7791
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "CALL") == pytest.approx(4.7594, abs=1e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "PUT") == pytest.approx(0.8086, abs=1e-3)
    assert bs.greeks(42, 40, 0.5, 0.10, 0.0, 0.20, "CALL")["delta"] == pytest.approx(0.7791, abs=1e-3)


def test_put_call_parity_with_dividends():
    S, K, T, r, q, sig = 22500.0, 22600.0, 0.03, 0.065, 0.012, 0.15
    c, p = bs.price(S, K, T, r, q, sig, "CALL"), bs.price(S, K, T, r, q, sig, "PUT")
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-6)


def test_greeks_agree_with_finite_differences():
    args = (22500.0, 22500.0, 0.05, 0.065, 0.012, 0.16)
    g = bs.greeks(*args, "CALL")
    S, K, T, r, q, sig = args
    h = 1.0
    delta_fd = (bs.price(S + h, K, T, r, q, sig, "CALL") - bs.price(S - h, K, T, r, q, sig, "CALL")) / (2 * h)
    gamma_fd = (bs.price(S + h, K, T, r, q, sig, "CALL") - 2 * bs.price(S, K, T, r, q, sig, "CALL") + bs.price(S - h, K, T, r, q, sig, "CALL")) / h ** 2
    vega_fd = (bs.price(S, K, T, r, q, sig + 1e-4, "CALL") - bs.price(S, K, T, r, q, sig - 1e-4, "CALL")) / 2e-4
    theta_fd = -(bs.price(S, K, T + 1e-5, r, q, sig, "CALL") - bs.price(S, K, T - 1e-5, r, q, sig, "CALL")) / 2e-5
    assert g["delta"] == pytest.approx(delta_fd, rel=1e-4)
    assert g["gamma"] == pytest.approx(gamma_fd, rel=1e-3)
    assert g["vega"] == pytest.approx(vega_fd, rel=1e-4)
    assert g["theta"] == pytest.approx(theta_fd, rel=1e-3)
    assert g["theta"] < 0  # a long ATM option decays


def test_expiry_and_degenerate_inputs():
    assert bs.price(110, 100, 0.0, 0.05, 0.0, 0.2, "CALL") == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.05, 0.0, 0.2, "CALL") == 0.0
    assert bs.greeks(110, 100, 0.0, 0.05, 0.0, 0.2, "CALL")["delta"] == 1.0
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.05, 0, 0.2, "FUTURE")


@pytest.mark.parametrize("kind", ["CALL", "PUT"])
@pytest.mark.parametrize("sigma", [0.05, 0.18, 0.60, 1.5])
def test_implied_vol_round_trips(kind, sigma):
    args = (22500.0, 22700.0, 0.04, 0.065, 0.012)
    px = bs.price(*args, sigma, kind)
    assert bs.implied_vol(px, *args, kind) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_reports_why_it_cannot_solve():
    args = (22500.0, 22500.0, 0.04, 0.065, 0.012)
    assert bs.implied_vol_with_reason(0.01, 22500.0, 20000.0, 0.04, 0.065, 0.012, "CALL")[1].startswith("price below")
    assert "above" in bs.implied_vol_with_reason(22400.0, *args, "CALL")[1]
    assert math.isnan(bs.implied_vol(100.0, 22500, 22500, 0.0, 0.065, 0.012, "CALL"))
    assert math.isnan(bs.implied_vol(float("nan"), *args, "CALL"))
    assert math.isnan(bs.implied_vol(-5.0, *args, "CALL"))


# ---------------------------------------------------------------- expiry rules and trading time
def test_default_expiry_rules():
    assert next_expiry(dt.date(2026, 10, 1), "NIFTY") == dt.date(2026, 10, 6)  # Tuesday weekly
    assert next_expiry(dt.date(2026, 10, 6), "NIFTY") == dt.date(2026, 10, 6)  # expiry today still counts
    assert next_expiry(dt.date(2026, 10, 1), "BANKNIFTY") == dt.date(2026, 10, 27)  # last Tuesday
    assert next_expiry(dt.date(2026, 10, 28), "BANKNIFTY") == dt.date(2026, 11, 24)  # rolls to next month
    assert next_expiry(dt.date(2026, 10, 1), "SENSEX") == dt.date(2026, 10, 1)  # Thursday weekly
    assert next_expiry(dt.date(2026, 10, 1), "RELIANCE") == dt.date(2026, 10, 27)


def test_expiry_on_a_holiday_moves_to_the_previous_trading_day():
    # 25 Dec 2029 is a Tuesday and a fixed NSE holiday
    assert next_expiry(dt.date(2029, 12, 20), "NIFTY") == dt.date(2029, 12, 24)


def test_expiry_rule_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("EXPIRY_RULE_NIFTY", "weekly:3")
    assert rule_for("NIFTY").param == 3
    assert next_expiry(dt.date(2026, 10, 1), "NIFTY") == dt.date(2026, 10, 1)
    monkeypatch.setenv("EXPIRY_RULE_NIFTY", "garbage")
    assert rule_for("NIFTY").kind == "weekly" and rule_for("NIFTY").param == 1  # bad value ignored


def test_time_is_counted_in_trading_days_not_calendar_days():
    # Thu 1 Oct 2026 15:30 -> Tue 6 Oct: Fri 2 Oct is a holiday, weekend closed => only Mon 5 and Tue 6 count
    assert trading_days_between(dt.date(2026, 10, 1), dt.date(2026, 10, 6), "NSE") == 2
    T = time_to_expiry_years(dt.datetime(2026, 10, 1, 15, 30), dt.date(2026, 10, 6), NSE, 252)
    assert T == pytest.approx(2 / 252)
    half_day = time_to_expiry_years(dt.datetime(2026, 10, 6, 12, 22, 30), dt.date(2026, 10, 6), NSE, 252)
    assert half_day == pytest.approx(0.5 / 252, rel=1e-3)  # halfway through the expiry session
    assert time_to_expiry_years(dt.datetime(2026, 10, 7, 10, 0), dt.date(2026, 10, 6), NSE, 252) == 0.0


# ---------------------------------------------------------------- backtest premium path
def make_trade(entry_ts, exit_ts, entry=22600.0, exit_=22660.0, stop=22540.0, direction="LONG", meta=None):
    setup = Setup(pd.Timestamp(entry_ts), direction, entry, stop, entry + 60.0, dict(meta or {}))
    gross = (exit_ - entry) if direction == "LONG" else (entry - exit_)
    return TradeResult(setup, pd.Timestamp(exit_ts), exit_, gross / 60.0, 6, 70.0, 0.0, gross, "WIN" if gross > 0 else "LOSS")


def test_flags_off_means_the_fixed_assumption_exactly():
    trade = make_trade("2026-09-29 10:00", "2026-09-29 10:30")
    assert "bs_premium" not in trade.setup.meta
    # identical to the pre-existing arithmetic (see test_option_costs for the absolute values)
    delta = settings_module.SETTINGS.costs.assumed_option_delta
    assert _apply_costs(trade, 22600.0, 75, "NSE")[3] == pytest.approx(
        delta * trade.gross_pnl * 75 - _apply_costs(trade, 22600.0, 75, "NSE")[0] - _apply_costs(trade, 22600.0, 75, "NSE")[1]
        - _apply_costs(trade, 22600.0, 75, "NSE")[2])
    assert engine_module._premium_model(gbm_ohlc(n_days=3), "NIFTY") is None


def test_premium_model_prices_an_atm_option_with_time_value_and_leaves_no_look_ahead():
    df = gbm_ohlc(n_days=30, sigma_annual=0.20, start="2026-08-03")
    model = BacktestPremiumModel(df, "NIFTY")
    entry_ts = df["timestamp"].iloc[75 * 20 + 10]
    exit_ts = df["timestamp"].iloc[75 * 20 + 16]
    spot = float(df["close"].iloc[75 * 20 + 10])
    trade = make_trade(entry_ts, exit_ts, entry=spot, exit_=spot, stop=spot - 40.0)
    p = model.price(trade)
    assert p is not None and p["entry"] > 20.0  # an ATM option on ~22k spot carries real premium
    assert p["exit"] < p["entry"]  # flat underlying: theta only, so a long option loses value
    assert 0 < p["risk"] < p["entry"] and p["kind"] == "CALL"

    # the volatility used at entry cannot depend on bars after entry
    mutated = df.copy()
    cut = 75 * 20 + 10
    mutated.loc[cut + 1:, ["open", "high", "low", "close"]] *= 2.0
    p2 = BacktestPremiumModel(mutated, "NIFTY").price(trade)
    assert p2["sigma"] == pytest.approx(p["sigma"]) and p2["entry"] == pytest.approx(p["entry"])


def test_premium_model_returns_none_when_it_cannot_price():
    df = gbm_ohlc(n_days=3)
    model = BacktestPremiumModel(df, "NIFTY")
    early = make_trade(df["timestamp"].iloc[2], df["timestamp"].iloc[5])  # no vol history yet
    assert model.price(early) is None
    open_trade = make_trade(df["timestamp"].iloc[100], df["timestamp"].iloc[100])
    open_trade.exit_price = None
    assert model.price(open_trade) is None and model.skipped == 2


def test_model_premium_flows_into_costs_and_r():
    df = gbm_ohlc(n_days=30, sigma_annual=0.20, start="2026-08-03")
    model = BacktestPremiumModel(df, "NIFTY")
    i = 75 * 20 + 10
    spot = float(df["close"].iloc[i])
    fixed = make_trade(df["timestamp"].iloc[i], df["timestamp"].iloc[i + 6], entry=spot, exit_=spot + 60, stop=spot - 60)
    modelled = make_trade(df["timestamp"].iloc[i], df["timestamp"].iloc[i + 6], entry=spot, exit_=spot + 60, stop=spot - 60)
    modelled.setup.meta["bs_premium"] = model.price(modelled)

    c_fixed, c_model = _apply_costs(fixed, spot, 75, "NSE"), _apply_costs(modelled, spot, 75, "NSE")
    assert c_fixed[3] != pytest.approx(c_model[3])  # a different, model-based P&L
    bp = modelled.setup.meta["bs_premium"]
    expected_gross = (bp["exit"] - bp["entry"]) * 75
    assert c_model[3] == pytest.approx(expected_gross - c_model[0] - c_model[1] - c_model[2])
    assert net_r_multiple(modelled, 75, "NSE") == pytest.approx(c_model[3] / (bp["risk"] * 75))


def test_run_backtest_applies_the_model_only_when_the_flag_is_on(monkeypatch):
    from quant_intelligence.features.feature_engine import compute_features
    from quant_intelligence.strategies.base_strategy import BaseStrategy

    df = gbm_ohlc(n_days=30, sigma_annual=0.20, start="2026-08-03")
    feats = compute_features(df)

    class EveryNth(BaseStrategy):
        name = "test-nth"
        required_features = []
        default_parameters = {}

        def generate_historical_setups(self, ohlcv, features):
            return [Setup(ohlcv["timestamp"].iloc[i], "LONG", float(ohlcv["close"].iloc[i]),
                          float(ohlcv["close"].iloc[i]) - 30.0, float(ohlcv["close"].iloc[i]) + 30.0, {})
                    for i in range(75 * 15, len(ohlcv) - 80, 150)]

        def check_setup(self, market_state, recent_ohlcv):
            return None

    off = engine_module.run_backtest(EveryNth(), df, feats, "NIFTY", persist=False)
    assert off.trades and all("bs_premium" not in t.setup.meta for t in off.trades)

    on_settings = dataclasses.replace(settings_module.SETTINGS, vol_premium_model_in_backtest=True)
    monkeypatch.setattr(engine_module, "SETTINGS", on_settings)
    on = engine_module.run_backtest(EveryNth(), df, feats, "NIFTY", persist=False)
    assert len(on.trades) == len(off.trades)
    assert any("bs_premium" in t.setup.meta for t in on.trades)
    assert on.metrics["net_expected_r"] != pytest.approx(off.metrics["net_expected_r"])
