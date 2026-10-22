"""Rebuilt option-chain snapshots and the chain-aware strategies replayed on them."""
from __future__ import annotations

import pandas as pd
import pytest

from quant_intelligence.backtesting.engine import run_backtest
from quant_intelligence.data_adapters.dhan_options_history import OptionsHistory
from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.options.historical_chain import FeatureState, HistoricalChain, generate_chain_setups
from quant_intelligence.strategies.iv_crush_writer_strategy import IVCrushWriterStrategy
from quant_intelligence.strategies.oi_buildup_strategy import OIBuildupStrategy
from quant_intelligence.strategies.pcr_contrarian_strategy import PCRContrarianStrategy
from quant_intelligence.strategies.rules import CHAIN_HISTORY_BACKTESTABLE
from quant_intelligence.strategies.unusual_volume_fade_strategy import UnusualVolumeFadeStrategy
from quant_intelligence.tests.options_helpers import make_history, session_bars
from quant_intelligence.tests.validation_helpers import edge_series, make_trades
from quant_intelligence.validation.evidence import EvidenceState as S
from quant_intelligence.validation.evidence import OutOfSampleResult, REQUIRED_ROBUSTNESS_TESTS, classify_evidence
from quant_intelligence.validation.metrics import compute_metrics, trade_table
from quant_intelligence.validation.uncertainty import estimate_uncertainty


@pytest.fixture(scope="module")
def bars(synthetic_ohlcv):
    return session_bars(synthetic_ohlcv)


@pytest.fixture(scope="module")
def history(bars):
    return make_history(bars, put_oi_scale=2.0)           # puts carry twice the call open interest


@pytest.fixture(scope="module")
def chain(history):
    return HistoricalChain(history)


def test_a_snapshot_merges_calls_and_puts_by_strike_and_is_marked_partial(chain, bars):
    ts = bars["timestamp"].iloc[100]
    snap = chain.snapshot_at(ts)
    assert isinstance(snap, ChainSnapshot) and snap.partial and snap.source == "dhan_history"
    strikes = [s.strike for s in snap.strikes]
    assert strikes == sorted(strikes) and len(strikes) == 7
    atm = next(s for s in snap.strikes if s.strike == snap.atm_strike)
    assert atm.ce_ltp is not None and atm.pe_ltp is not None and atm.ce_iv == pytest.approx(15.0)
    assert snap.spot_price == pytest.approx(bars["close"].iloc[100])
    assert snap.total_pcr == pytest.approx(2.0)


def test_live_snapshots_are_not_partial():
    assert ChainSnapshot("NIFTY", "2026-10-06", 22000.0).partial is False


def test_no_snapshot_outside_the_stored_span(chain, bars):
    assert chain.snapshot_at(bars["timestamp"].iloc[0] - pd.Timedelta(days=30)) is None
    assert chain.snapshot_at(bars["timestamp"].iloc[-1] + pd.Timedelta(days=30)) is None


def test_volume_accumulates_within_a_session_and_resets_the_next_day(chain, bars):
    day = pd.to_datetime(bars["timestamp"]).dt.date
    idx = list(day[day == day.iloc[200]].index)
    # a strike that stays in the window for the first bars of the day: volume only grows
    a, b = chain.snapshot_at(bars["timestamp"][idx[1]]), chain.snapshot_at(bars["timestamp"][idx[5]])
    common = {s.strike for s in a.strikes} & {s.strike for s in b.strikes}
    assert common
    for k in common:
        va = next(s for s in a.strikes if s.strike == k).ce_volume
        vb = next(s for s in b.strikes if s.strike == k).ce_volume
        assert vb >= va
    first_of_day = chain.snapshot_at(bars["timestamp"][idx[0]])
    assert all(s.ce_volume is not None and s.ce_volume <= 110 * 10 for s in first_of_day.strikes)       # one bar's worth, not the day before's


def test_oi_change_is_against_the_previous_sessions_last_oi(chain, bars):
    day = pd.to_datetime(bars["timestamp"]).dt.date
    first_day_idx = day[day == day.iloc[0]].index
    assert all(s.ce_oi_change is None for s in chain.snapshot_at(bars["timestamp"][first_day_idx[3]]).strikes)   # no prior session
    snap = chain.snapshot_at(bars["timestamp"][day[day == day.iloc[200]].index[2]])
    changes = [s.ce_oi_change for s in snap.strikes if s.ce_oi_change is not None]
    assert changes and all(c > 0 for c in changes)                      # the fixture's open interest grows 100/day per series


def test_a_snapshot_never_uses_later_bars(history, bars):
    ts = bars["timestamp"].iloc[300]
    cut = OptionsHistory(history.symbol, {k: f[f.index <= ts].reset_index() for k, f in history.frames.items()})
    full, truncated = HistoricalChain(history).snapshot_at(ts), HistoricalChain(cut).snapshot_at(ts)
    assert [dataclass_tuple(s) for s in full.strikes] == [dataclass_tuple(s) for s in truncated.strikes]
    assert full.spot_price == truncated.spot_price


def dataclass_tuple(row):
    return tuple(row.__dict__.values())


def test_feature_state_mirrors_market_state_get():
    st = FeatureState({"a": 1.0, "b": float("nan"), "c": None})
    assert st.get("a") == 1.0 and st.get("b", 7) == 7 and st.get("c", 3) == 3 and st.get("zzz") is None


# ---- strategies replayed on the snapshots ---------------------------------------------------------------------------

def test_without_history_the_chain_strategies_still_produce_nothing(synthetic_ohlcv, synthetic_features):
    for cls in (PCRContrarianStrategy, OIBuildupStrategy, UnusualVolumeFadeStrategy, IVCrushWriterStrategy):
        assert cls().generate_historical_setups(synthetic_ohlcv, synthetic_features) == []


def test_pcr_strategy_replays_on_the_snapshots(chain, synthetic_ohlcv, synthetic_features):
    strat = PCRContrarianStrategy().attach_chain_history(chain)
    setups = strat.generate_historical_setups(synthetic_ohlcv, synthetic_features)
    assert setups, "PCR is 2.0 everywhere and RSI(2) dips below 20 in the fixture"
    assert all(s.direction == "LONG" and s.meta["transaction"] == "BUY" and s.meta["total_pcr"] == pytest.approx(2.0) for s in setups)
    ts = set(synthetic_ohlcv["timestamp"])
    assert all(s.timestamp in ts for s in setups)


def test_setups_are_causal_a_later_bar_cannot_change_an_earlier_signal(chain, synthetic_ohlcv, synthetic_features):
    strat = PCRContrarianStrategy().attach_chain_history(chain)
    full = strat.generate_historical_setups(synthetic_ohlcv, synthetic_features)
    cut = len(synthetic_ohlcv) // 2
    part = strat.generate_historical_setups(synthetic_ohlcv.iloc[:cut].reset_index(drop=True), synthetic_features.iloc[:cut].reset_index(drop=True))
    cutoff = synthetic_ohlcv["timestamp"].iloc[cut - 1]
    assert [(s.timestamp, s.direction) for s in full if s.timestamp <= cutoff] == [(s.timestamp, s.direction) for s in part]


def test_oi_and_volume_strategies_run_without_error_on_the_snapshots(chain, synthetic_ohlcv, synthetic_features):
    for cls in (OIBuildupStrategy, UnusualVolumeFadeStrategy):
        got = generate_chain_setups(cls().attach_chain_history(chain), synthetic_ohlcv, synthetic_features, chain)
        assert all(s.meta["transaction"] == "BUY" for s in got)


def test_sell_side_setups_are_never_replayed(chain, synthetic_ohlcv, synthetic_features):
    iv = IVCrushWriterStrategy(parameters={"min_iv_premium_pct_pts": -100.0, "max_trend_slope": 1.0}).attach_chain_history(chain)
    assert iv.check_chain_setup(FeatureState(synthetic_features.iloc[500].to_dict()), synthetic_ohlcv.iloc[496:501], chain.snapshot_at(synthetic_ohlcv["timestamp"].iloc[500])) is not None
    assert generate_chain_setups(iv, synthetic_ohlcv, synthetic_features, chain) == []        # it would SELL: no sell-side backtest model
    assert "IV Crush Premium Writer" not in CHAIN_HISTORY_BACKTESTABLE


def test_a_chain_strategy_can_be_backtested_through_the_engine(chain, synthetic_ohlcv, synthetic_features):
    strat = PCRContrarianStrategy().attach_chain_history(chain)
    r = run_backtest(strat, synthetic_ohlcv, synthetic_features, "NIFTY", persist=False, quantity=75)
    assert r.metrics["n_trades"] > 0


# ---- evidence: proxy data caps the state ---------------------------------------------------------------------------

def assess(**kw):
    trades = make_trades(edge_series(400, mean=0.5))
    tbl = trade_table(trades, 75)
    return classify_evidence(compute_metrics(trades, 75, table=tbl), estimate_uncertainty(tbl), n_sessions=500,
                             out_of_sample=OutOfSampleResult(60, 0.2), robustness={n: True for n in REQUIRED_ROBUSTNESS_TESTS}, **kw)


def test_proxy_data_caps_the_state_below_validated():
    assert assess().state is S.VALIDATED_FOR_RESEARCH
    capped = assess(data_limitations=["near-the-money strikes only"])
    assert capped.state is S.ROBUSTNESS_TESTING and "proxy data" in capped.reasons[0]
    assert any(c.name == "proxy_data" and not c.passed for c in capped.checks)


def test_the_proxy_cap_does_not_hide_a_failure():
    trades = make_trades(edge_series(400, mean=-0.3))
    tbl = trade_table(trades, 75)
    a = classify_evidence(compute_metrics(trades, 75, table=tbl), estimate_uncertainty(tbl), n_sessions=500, data_limitations=["x"])
    assert a.state is S.VALIDATION_FAILED


def test_the_suite_replays_a_chain_strategy_only_when_history_is_attached(chain, synthetic_ohlcv, synthetic_features):
    from quant_intelligence.validation.suite import run_validation_suite

    plain = run_validation_suite(PCRContrarianStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75)
    assert plain.state is S.INSUFFICIENT_DATA and plain.out_of_sample is None and not plain.proxy_data
    with_hist = run_validation_suite(PCRContrarianStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, chain_history=chain)
    assert with_hist.proxy_data and with_hist.out_of_sample is not None and with_hist.in_sample.metrics.n_trades > 0
    assert any("near-the-money" in n for n in with_hist.notes)
    iv = run_validation_suite(IVCrushWriterStrategy, synthetic_ohlcv, synthetic_features, "NIFTY", quantity=75, chain_history=chain)
    assert iv.out_of_sample is None and not iv.proxy_data                     # sell-side is not replayable
