"""
Quant Strategy Intelligence & Execution System - Command Center.

Run with:  streamlit run quant_intelligence/app.py
(run from the project root so the `quant_intelligence` package resolves).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime as dt

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import init_db
from quant_intelligence.ui.credentials_panel import render_credentials_panel
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme, status_badge

init_db()
apply_theme()
init_session_state()

st.title("Quant Strategy Intelligence & Execution System")
st.caption("Research-first terminal for Indian index derivatives. Central question: "
           "which validated strategy has the strongest statistically validated conditional edge right now?")

with st.sidebar:
    st.header("Instrument & Data")
    st.session_state["instrument"] = st.text_input("Instrument", st.session_state["instrument"])
    st.session_state["timeframe"] = st.selectbox("Timeframe", ["5min", "15min", "1min", "3min"], index=0)
    st.session_state["lookback_days"] = st.slider("Lookback window (days)", 10, 730, st.session_state["lookback_days"])

    if st.button("Refresh pipeline", type="primary"):
        with st.spinner("Running data -> features -> regime -> strategy ranking pipeline..."):
            run_pipeline_cached(force=True)

    st.divider()
    st.header("API Keys")
    render_credentials_panel()

    st.divider()
    st.header("Trading Mode")
    if SETTINGS.is_live_mode and SETTINGS.live_mode_fully_authorized:
        st.markdown('<span class="qi-badge qi-badge-live">LIVE MODE ENABLED</span>', unsafe_allow_html=True)
        st.error("LIVE mode is active. Real orders can be sent once the Dhan adapter is implemented.")
    elif SETTINGS.is_live_mode:
        st.markdown('<span class="qi-badge qi-badge-notrade">LIVE requested, NOT authorized</span>', unsafe_allow_html=True)
        st.warning("TRADING_MODE=LIVE but TRADING_LIVE_CONFIRM is missing/incorrect. Falling back to PAPER behavior.")
    else:
        st.markdown('<span class="qi-badge qi-badge-paper">PAPER MODE</span>', unsafe_allow_html=True)
        st.success("All orders are simulated. No real orders can be sent.")

output = st.session_state.get("pipeline_output")
error = st.session_state.get("pipeline_error")

if output is None and error is None:
    with st.spinner("Running initial pipeline on live Dhan / CSV market data..."):
        output = run_pipeline_cached()
        error = st.session_state.get("pipeline_error")

if error:
    st.error(f"Pipeline could not run: {error}")
    st.stop()

st.subheader(f"{output.instrument} — as of {output.timestamp}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Data Quality", output.data_quality_status)
col2.metric("Regime", output.regime_label, f"{output.regime_confidence*100:.0f}% confidence")
col3.metric("Strategies Evaluated", len(output.strategy_intel))
col4.metric("Eligible Strategies", sum(1 for s in output.strategy_intel if s.score.eligible))
decision_label = "NO TRADE" if output.ranking.is_no_trade else output.ranking.selected_strategy
col5.metric("Decision", decision_label)

st.divider()

badge_class = "qi-badge-notrade" if output.ranking.is_no_trade else "qi-badge-trade"
badge_text = "NO TRADE" if output.ranking.is_no_trade else f"SELECTED: {output.ranking.selected_strategy}"
st.markdown(f'<span class="qi-badge {badge_class}" style="font-size:1rem;">{badge_text}</span>', unsafe_allow_html=True)
st.write(f"**Reason:** {output.ranking.reason}")

st.subheader("Strategy Ranking")
rows = []
for s in output.ranking.ranked:
    rows.append(
        {
            "Strategy": s.strategy_name,
            "Score": round(s.score, 4),
            "Expected R (conditional)": round(s.expected_r, 3) if s.expected_r is not None else None,
            "Win Probability": f"{s.prob_positive_return*100:.1f}%" if s.prob_positive_return is not None else "N/A",
            "Confidence": s.confidence_label,
            "Sample Size": s.sample_size,
            "Max Drawdown": round(s.max_drawdown, 1) if s.max_drawdown is not None else None,
            "Status": "ELIGIBLE" if s.eligible else "INELIGIBLE",
            "Reason (if ineligible)": s.ineligibility_reason or "",
        }
    )
st.dataframe(rows, width="stretch", hide_index=True)

st.subheader("Regime Probabilities")
regime_cols = st.columns(len(output.regime_probabilities))
for col, (regime, prob) in zip(regime_cols, sorted(output.regime_probabilities.items(), key=lambda x: -x[1])):
    col.metric(regime, f"{prob*100:.0f}%")

st.divider()
st.caption(
    "Use the pages in the sidebar for Market State detail, Strategy Intelligence drill-down, "
    "Backtest Lab, Historical Analogues, Paper Trading, Positions & Risk Control, Research Reports, the Daily Report, "
    "and the NSE Heatmap (pick the day's movers for the paper traders)."
)
