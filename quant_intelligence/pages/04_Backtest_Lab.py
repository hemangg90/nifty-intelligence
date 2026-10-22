import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.backtesting.engine import run_backtest, walk_forward
from quant_intelligence.data.data_manager import DataManager, DataUnavailableError
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.strategies.registry import STRATEGY_CLASSES, get_strategy
from quant_intelligence.ui.theme import apply_theme

apply_theme()
st.title("Backtest Lab")
st.caption("Event-driven backtesting with in-sample / out-of-sample separation and walk-forward analysis.")

col1, col2, col3 = st.columns(3)
instrument = col1.text_input("Instrument", "NIFTY")
timeframe = col2.selectbox("Timeframe", ["5min", "15min", "1min", "3min"])
strategy_name = col3.selectbox("Strategy", list(STRATEGY_CLASSES.keys()))

col4, col5 = st.columns(2)
lookback_days = col4.slider("Total lookback (days)", 20, 730, 90)
oos_fraction = col5.slider("Out-of-sample holdout (fraction of most recent data)", 0.1, 0.5, 0.25)

strategy_cls = STRATEGY_CLASSES[strategy_name]
default_params = strategy_cls.default_parameters
st.markdown("**Parameters** (edit and re-run; parameters are NOT auto-optimized on out-of-sample data)")
param_cols = st.columns(len(default_params)) if default_params else []
edited_params = {}
for i, (k, v) in enumerate(default_params.items()):
    if isinstance(v, (int, float)):
        edited_params[k] = param_cols[i].number_input(k, value=float(v))
    else:
        edited_params[k] = param_cols[i].text_input(k, value=str(v))

run = st.button("Run backtest", type="primary")

if run:
    end = now_ist()
    start = end - dt.timedelta(days=lookback_days)
    dm = DataManager()
    with st.spinner("Fetching data and computing features..."):
        try:
            ohlcv, metadata = dm.get_ohlcv(instrument, timeframe, start, end)
        except DataUnavailableError as e:
            st.error(str(e))
            st.stop()
        if len(ohlcv) < 100:
            st.error("Not enough data for the selected window.")
            st.stop()
        features = compute_features(ohlcv)

    split_idx = int(len(ohlcv) * (1 - oos_fraction))
    is_ohlcv, is_features = ohlcv.iloc[:split_idx].reset_index(drop=True), features.iloc[:split_idx].reset_index(drop=True)
    oos_ohlcv, oos_features = ohlcv.iloc[split_idx:].reset_index(drop=True), features.iloc[split_idx:].reset_index(drop=True)

    strategy_is = get_strategy(strategy_name, edited_params)
    strategy_oos = get_strategy(strategy_name, edited_params)

    with st.spinner("Running in-sample backtest..."):
        is_result = run_backtest(strategy_is, is_ohlcv, is_features, instrument, split="IN_SAMPLE")
    with st.spinner("Running out-of-sample backtest (held-out most recent data)..."):
        oos_result = run_backtest(strategy_oos, oos_ohlcv, oos_features, instrument, split="OUT_OF_SAMPLE")

    st.session_state["last_backtest"] = {"is": is_result, "oos": oos_result, "ohlcv": ohlcv}

if "last_backtest" in st.session_state:
    is_result = st.session_state["last_backtest"]["is"]
    oos_result = st.session_state["last_backtest"]["oos"]

    st.caption("A single backtest is not evidence of an edge. The **Strategy Evidence** page shows what each strategy has demonstrated "
               "after statistical validation, out-of-sample and robustness testing.")
    with st.expander("Run manifest (what exactly produced these numbers)"):
        st.json({"in_sample": is_result.manifest, "out_of_sample": oos_result.manifest})

    tab1, tab2, tab3 = st.tabs(["In-Sample", "Out-of-Sample", "Walk-Forward"])

    def render_result(result, key_prefix):
        n_trades = result.metrics.get("n_trades", 0)
        if n_trades == 0:
            st.warning("No trades generated in this window.")
            return
        if n_trades < 30:
            st.warning(f"Sample size warning: only {n_trades} trades. Statistics may be unreliable.")

        cols = st.columns(5)
        cols[0].metric("Trades", n_trades)
        cols[1].metric("Win Rate", f"{result.metrics['win_rate']*100:.1f}%")
        cols[2].metric("Expected R", round(result.metrics["expected_r"], 3))
        cols[3].metric("Sharpe", round(result.metrics["sharpe"], 2) if not np.isnan(result.metrics["sharpe"]) else "N/A")
        cols[4].metric("Max Drawdown (INR)", round(result.metrics["max_drawdown"], 0))

        r_multiples = [t.r_multiple for t in result.trades]
        equity = np.cumsum(r_multiples)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity, mode="lines", name="Cumulative R"))
        fig.update_layout(template="plotly_dark", height=300, title="Equity Curve (R multiples)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_equity")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=r_multiples, nbinsx=30))
        fig2.update_layout(template="plotly_dark", height=300, title="R-Multiple Distribution", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, width="stretch", key=f"{key_prefix}_hist")

    with tab1:
        render_result(is_result, "is")
    with tab2:
        render_result(oos_result, "oos")
        if is_result.metrics.get("n_trades", 0) and oos_result.metrics.get("n_trades", 0):
            is_er = is_result.metrics["expected_r"]
            oos_er = oos_result.metrics["expected_r"]
            if (is_er > 0) and (oos_er < is_er * 0.3):
                st.error(
                    f"In-sample expected R ({is_er:.3f}) does not survive out-of-sample "
                    f"({oos_er:.3f}). This is evidence AGAINST the strategy having a robust edge "
                    "in the current window - do not assume profitability."
                )
    with tab3:
        if st.button("Run walk-forward (4 folds)"):
            ohlcv_all = st.session_state["last_backtest"]["ohlcv"]
            features_all = compute_features(ohlcv_all)
            wf_results = walk_forward(
                lambda: get_strategy(strategy_name, edited_params), ohlcv_all, features_all, instrument, n_folds=4
            )
            for r in wf_results:
                st.markdown(f"**{r.split}**")
                render_result(r, r.split)
