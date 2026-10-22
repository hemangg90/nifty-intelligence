"""Display helpers for the Strategy Evidence page: state chips, a styled leaderboard and the evidence charts.

No data logic lives here (see research/dashboard_data.py). Charts never smooth or extrapolate: they draw the stored
numbers, label out-of-sample stretches as such, and say "not available" when a piece was not computed.
"""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.ui.charts import _base_layout
from quant_intelligence.ui.components import chip
from quant_intelligence.ui.theme import AQUA, BLUE, CRITICAL, GOOD, MUTED, ORANGE, TEXT_2

STATE_TONE = {
    "VALIDATED FOR RESEARCH": "good",
    "ROBUSTNESS TESTING": "info",
    "RESEARCH QUALIFIED": "info",
    "VALIDATION FAILED": "critical",
    "INSUFFICIENT SAMPLE": "warning",
    "INSUFFICIENT DATA": "muted",
    "NO EVIDENCE": "muted",
}


def state_chip(state: str) -> str:
    return chip(state, STATE_TONE.get(state, "muted"))


def unavailable(what: str, why: str) -> None:
    st.markdown(f'<div class="qi-empty"><b>{html.escape(what)} not available.</b> {html.escape(why)}</div>', unsafe_allow_html=True)


def leaderboard_table(df: pd.DataFrame):
    """A pandas Styler for st.dataframe: fixed decimals, percentages, and colour on the evidence state and the R columns."""
    tone_colour = {"good": GOOD, "info": BLUE, "critical": CRITICAL, "warning": ORANGE, "muted": MUTED}

    def state_colour(v):
        return f"color: {tone_colour.get(STATE_TONE.get(v, 'muted'), MUTED)}; font-weight: 600"

    def r_colour(v):
        if v is None or v != v:
            return ""
        return f"color: {GOOD if v > 0 else CRITICAL if v < 0 else TEXT_2}"

    r3 = lambda v: "–" if v is None or v != v else f"{v:+.3f}"  # noqa: E731
    sty = df.style.format({"Net R/trade": r3, "95% CI low": r3, "95% CI high": r3, "OOS net R": r3, "OOS lower bound": r3,
                           "Profit factor": lambda v: "–" if v is None or v != v else f"{v:.2f}",
                           "Max DD (R)": lambda v: "–" if v is None or v != v else f"{v:.1f}",
                           "Walk-forward +": lambda v: "–" if v is None or v != v else f"{v:.0%}",
                           "Evidence age (d)": lambda v: "–" if v is None or v != v else f"{v:.0f}",
                           "OOS trades": lambda v: "–" if v is None or v != v else f"{int(v)}",
                           "Rank": lambda v: "–" if v is None or v != v else f"{int(v)}",
                           "Eligible": lambda v: "yes" if v else "no"}, na_rep="–")
    return sty.map(state_colour, subset=["Evidence state"]).map(r_colour, subset=["Net R/trade", "OOS net R"])


# ---- charts -------------------------------------------------------------------------------------------------------

def equity_drawdown_chart(eq: pd.DataFrame, height: int = 360) -> go.Figure:
    """Cumulative net R with the out-of-sample stretch drawn separately, and the drawdown from the running peak below."""
    fig = go.Figure()
    for seg, colour in (("development", BLUE), ("validation", AQUA)):
        part = eq[eq["segment"] == seg]
        if len(part):
            fig.add_trace(go.Scatter(x=part["exit_ts"], y=part["equity_r"], mode="lines", name=f"{seg} (net R)", line=dict(color=colour, width=2)))
    fig.add_trace(go.Scatter(x=eq["exit_ts"], y=eq["drawdown_r"], mode="lines", name="drawdown (R)", fill="tozeroy",
                             line=dict(color=CRITICAL, width=1), fillcolor="rgba(239,68,68,0.18)", yaxis="y2"))
    fig.update_layout(yaxis=dict(title="cumulative net R", domain=[0.32, 1.0]), yaxis2=dict(title="drawdown (R)", domain=[0.0, 0.26]))
    return _base_layout(fig, height)


def histogram_chart(h: pd.DataFrame, height: int = 260) -> go.Figure:
    colours = [GOOD if x > 0 else CRITICAL for x in h["net_r"]]
    fig = go.Figure(go.Bar(x=h["net_r"], y=h["trades"], marker_color=colours, name="trades"))
    fig.update_xaxes(title="net R per trade (end bins hold clipped values)")
    return _base_layout(fig, height, legend=False)


def period_chart(p: pd.DataFrame, height: int = 260) -> go.Figure:
    colours = [GOOD if x > 0 else CRITICAL for x in p["net_expectancy_r"]]
    fig = go.Figure(go.Bar(x=p["period"], y=p["net_expectancy_r"], marker_color=colours, customdata=p["n"],
                           hovertemplate="%{x}: %{y:+.3f} R over %{customdata} trades<extra></extra>"))
    fig.update_yaxes(title="net R per trade")
    return _base_layout(fig, height, legend=False)


def regime_chart(r: pd.DataFrame, height: int = 300) -> go.Figure:
    r = r.dropna(subset=["net_expectancy_r"])
    fig = go.Figure(go.Bar(
        x=[f"{a}: {b}" for a, b in zip(r["dimension"], r["label"])], y=r["net_expectancy_r"],
        marker_color=[GOOD if x > 0 else CRITICAL for x in r["net_expectancy_r"]],
        error_y=dict(type="data", symmetric=False, array=(r["ci_high"] - r["net_expectancy_r"]).fillna(0),
                     arrayminus=(r["net_expectancy_r"] - r["ci_low"]).fillna(0), color=MUTED),
        customdata=r["n_trades"], hovertemplate="%{x}: %{y:+.3f} R over %{customdata} trades<extra></extra>"))
    fig.update_yaxes(title="net R per trade (95% interval)")
    return _base_layout(fig, height, legend=False)


def parameter_chart(p: pd.DataFrame, height: int = 300) -> go.Figure:
    """Net expectancy at each neighbouring setting of each parameter: a stability surface, not an optimiser."""
    p = p.assign(label=p["parameter"].astype(str) + " = " + p["value"].astype(str))
    fig = go.Figure(go.Bar(x=p["label"], y=p["net_expectancy_r"], marker_color=[GOOD if x > 0 else CRITICAL for x in p["net_expectancy_r"]],
                           customdata=p["n_trades"], hovertemplate="%{x}: %{y:+.3f} R over %{customdata} trades<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_yaxes(title="net R per trade")
    return _base_layout(fig, height, legend=False)


def cost_chart(c: pd.DataFrame, height: int = 260) -> go.Figure:
    fig = go.Figure(go.Scatter(x=c["slippage_multiple"], y=c["net_expectancy_r"], mode="lines+markers", line=dict(color=BLUE)))
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_xaxes(title="slippage multiple of the base assumption")
    fig.update_yaxes(title="net R per trade")
    return _base_layout(fig, height, legend=False)


def walk_forward_chart(w: pd.DataFrame, height: int = 260) -> go.Figure:
    w = w.assign(label=[f"{a} to {b}" for a, b in zip(w["start"], w["end"])])
    fig = go.Figure(go.Bar(x=w["label"], y=w["net_expectancy_r"], marker_color=[GOOD if (x or 0) > 0 else CRITICAL for x in w["net_expectancy_r"]],
                           customdata=w["n_trades"], hovertemplate="%{x}: %{y:+.3f} R over %{customdata} trades<extra></extra>"))
    fig.update_yaxes(title="net R per trade")
    return _base_layout(fig, height, legend=False)
