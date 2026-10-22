import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.research import dashboard_data as dd
from quant_intelligence.research.scheduler import RESEARCH_SCHEDULER
from quant_intelligence.ui import evidence_views as ev
from quant_intelligence.ui.components import empty_state, kpi_row, page_header
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.validation.evidence import STATE_DEFINITIONS, EvidenceState
from quant_intelligence.validation.evidence_store import load_latest_evidence

apply_theme()
init_session_state()
page_header("Strategy Evidence", "What each strategy has actually demonstrated on past data, under a documented policy. "
                                  "Not a forecast and not advice.")

# ---- which instrument ---------------------------------------------------------------------------------------------
options = list(dict.fromkeys(dd.stored_targets() + [(st.session_state["instrument"], st.session_state["timeframe"])]
                             + RESEARCH_SCHEDULER.targets()))
target = st.selectbox("Instrument and timeframe", options, format_func=lambda t: f"{t[0]} · {t[1]}")
instrument, timeframe = target
evidence = load_latest_evidence(instrument, timeframe)

ranking, ranking_note = None, None
is_live_target = (instrument, timeframe) == (st.session_state["instrument"], st.session_state["timeframe"])
if evidence and is_live_target:
    output = run_pipeline_cached()
    if output is None:
        ranking_note = st.session_state.get("pipeline_error") or "The live pipeline is not available."
    else:
        try:
            ranking = dd.current_ranking(output, timeframe, evidence)
        except Exception as e:  # the explanation is a view; it must not break the rest of the page
            ranking_note = f"Could not rank the current market: {e}"
elif evidence:
    ranking_note = ("Regime fit, setup and final status depend on the live market, which is loaded for the instrument chosen "
                    f"in the sidebar ({st.session_state['instrument']} · {st.session_state['timeframe']}). Select that one here to see them.")

tab_board, tab_detail, tab_control, tab_explain = st.tabs(["Leaderboard", "Research detail", "Backtest control center", "Ranking explanation"])

# ---- 1. leaderboard -----------------------------------------------------------------------------------------------
with tab_board:
    if not evidence:
        empty_state("No stored evidence for this instrument yet. ",
                    f"Run `python scripts/run_research.py {instrument}` (needs cached candles: `python scripts/backfill_history.py {instrument}`), "
                    "or start the scheduler with RESEARCH_SCHEDULER=true.")
    else:
        st.warning(dd.SORT_WARNING)
        if not SETTINGS.gated_ranker:
            st.info("GATED_RANKER is off, so the auto traders still use the legacy ranker. This table shows what the gated ranker would do.")
        board = dd.leaderboard_frame(instrument, timeframe, evidence, ranking)
        kpi_row([
            {"label": "Strategies with evidence", "value": len(board)},
            {"label": "Pass every gate", "value": int(board["Eligible"].sum()), "tone": "good"},
            {"label": "Validation failed", "value": int((board["Evidence state"] == EvidenceState.VALIDATION_FAILED.value).sum())},
            {"label": "Too little data or sample", "value": int(board["Evidence state"].isin(
                [EvidenceState.INSUFFICIENT_DATA.value, EvidenceState.INSUFFICIENT_SAMPLE.value]).sum())},
        ])
        st.dataframe(ev.leaderboard_table(board), width="stretch", hide_index=True)
        if ranking_note:
            st.caption(ranking_note)
        elif ranking is not None:
            pick = ranking.ranking.selected_strategy
            st.caption(f"Gated choice for the current market: **{pick or 'NO TRADE'}**. {ranking.ranking.reason} "
                       "The risk engine still evaluates every order.")
        with st.expander("What the evidence states mean"):
            for s, text in STATE_DEFINITIONS.items():
                st.markdown(f"{ev.state_chip(s.value)} {text}", unsafe_allow_html=True)

# ---- 2. research detail -------------------------------------------------------------------------------------------
with tab_detail:
    if not evidence:
        empty_state("Nothing to show yet. ", "Detail appears once a strategy has a stored verdict.")
    else:
        names = list(dd.leaderboard_frame(instrument, timeframe, evidence)["Strategy"])
        name = st.selectbox("Strategy", names, key="detail_strategy")
        snap = evidence[name]
        d = dd.strategy_detail(snap)
        s = d["summary"]
        st.markdown(f"{ev.state_chip(d['state'])} &nbsp; strategy version {snap.strategy_version} · verdict made {snap.created_at:%Y-%m-%d %H:%M} UTC"
                    + (" · **proxy data**" if s.get("proxy_data") else ""), unsafe_allow_html=True)
        for reason in s.get("reasons", []):
            st.caption(f"• {reason}")
        oos = d["oos"] or {}
        f = lambda v, spec="+.3f": "–" if v is None else format(v, spec)  # noqa: E731
        kpi_row([
            {"label": "Net R / trade", "value": f(s.get("net_expectancy_r")), "delta": f"95% CI [{f(s.get('ci_low'))}, {f(s.get('ci_high'))}]"},
            {"label": "Trades (effective n)", "value": f"{s.get('n_trades', 0)} ({f(s.get('effective_n'), '.0f')})"},
            {"label": "Profit factor", "value": f(s.get("profit_factor"), ".2f")},
            {"label": "Max drawdown", "value": f(None if s.get("max_drawdown_r") is None else abs(s["max_drawdown_r"]), ".1f") + " R"},
            {"label": "Out of sample", "value": f(oos.get("net_expectancy_r")), "delta": f"{oos.get('n_trades', 0)} trades"},
        ])
        if not d["has_detail"]:
            ev.unavailable("Curves and breakdowns", d["error"] or "this verdict was stored without a detail block; re-run the research for this strategy")
        else:
            st.markdown("**Equity and drawdown** (cumulative net R of past trades; the validation stretch is the out-of-sample part)")
            st.plotly_chart(ev.equity_drawdown_chart(d["equity"]), width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Trade distribution**")
                st.plotly_chart(ev.histogram_chart(d["histogram"]), width="stretch")
            with c2:
                st.markdown("**By calendar month**")
                if d["periods"] is not None:
                    st.plotly_chart(ev.period_chart(d["periods"]), width="stretch")
                else:
                    ev.unavailable("Monthly breakdown", "no trades")
            st.markdown("**By market condition** (regime, trend, volatility, gap; intervals are wide where trades are few)")
            if d["regimes"] is not None:
                st.plotly_chart(ev.regime_chart(d["regimes"]), width="stretch")
            else:
                ev.unavailable("Regime breakdown", "no trades")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Parameter sensitivity** (neighbours of the defaults, never a best setting)")
            if d["parameters"] is not None:
                st.plotly_chart(ev.parameter_chart(d["parameters"]), width="stretch")
            else:
                ev.unavailable("Parameter sensitivity", "robustness tests run only once the earlier gates have passed")
        with c4:
            st.markdown("**Cost stress** (slippage scaled up)")
            if d["cost_stress"] is not None:
                st.plotly_chart(ev.cost_chart(d["cost_stress"]), width="stretch")
            else:
                ev.unavailable("Cost stress", "robustness tests run only once the earlier gates have passed")

        st.markdown("**Walk-forward** (consecutive cold runs; no parameter is fitted)")
        if d["walk_forward"] is not None:
            st.plotly_chart(ev.walk_forward_chart(d["walk_forward"]), width="stretch")
        else:
            ev.unavailable("Walk-forward folds", "not stored for this verdict")

        st.markdown("**Robustness tests**")
        st.dataframe(d["robustness_tests"], width="stretch", hide_index=True)
        if d["holdout"]:
            h = d["holdout"]
            st.caption(f"Final holdout: {h['n_trades']} trades, net {f(h.get('net_expectancy_r'))} R"
                       + (" (the holdout has been used too often to count as untouched)" if h.get("consumed") else ""))

        with st.expander("Trade records"):
            if d["trades"] is not None:
                st.caption(d["trades_note"])
                st.dataframe(d["trades"], width="stretch", hide_index=True, height=300)
            else:
                ev.unavailable("Trade records", "none stored")
        with st.expander("Rules and known limitations", expanded=True):
            for k, v in (d.get("rules") or {}).items():
                st.markdown(f"**{k.title()}:** {v}")
            for lim in d["limitations"]:
                st.markdown(f"- {lim}")

# ---- 3. control center --------------------------------------------------------------------------------------------
with tab_control:
    cc = dd.control_center(instrument, timeframe)
    status = RESEARCH_SCHEDULER.status()
    b1, b2, b3 = st.columns([1, 1, 2])
    if b1.button("Run stale research now", type="primary", disabled=status["batch_running"], help="Re-validates strategies with new data or a changed definition. Runs in the background."):
        RESEARCH_SCHEDULER.run_in_background([(instrument, timeframe)], force=False)
        st.rerun()
    if b2.button("Re-run everything", disabled=status["batch_running"], help="Ignores the up-to-date check and the market-hours deferral."):
        RESEARCH_SCHEDULER.run_in_background([(instrument, timeframe)], force=True)
        st.rerun()
    b3.caption(("A batch is running" + (f" since {status['current']['started_at']:%H:%M:%S}" if status.get("current") else "") + ". Refresh to see progress.")
               if status["batch_running"] else
               ("The scheduler thread is running." if status["running"] else "The scheduler thread is off (RESEARCH_SCHEDULER=false); runs happen only when you press a button or run scripts/run_research.py."))
    if status.get("error"):
        st.error(f"Last research error: {status['error']}")

    v = cc["version"]
    last = cc["last_success"]
    batch = cc["latest_batch"]
    kpi_row([
        {"label": "Accepted research version", "value": "none" if v is None else f"v{v['version_no']}", "delta": None if v is None else f"{v['created_at']:%Y-%m-%d %H:%M} UTC"},
        {"label": "Completed / failed runs", "value": f"{cc['counts'].get('COMPLETED', 0)} / {cc['counts'].get('FAILED', 0)}"},
        {"label": "Awaiting data", "value": cc["counts"].get("AWAITING_DATA", 0)},
        {"label": "Last success", "value": "none" if last is None else f"{last['at']:%d %b %H:%M}", "delta": None if last is None else last["strategy"]},
        {"label": "Last batch runtime", "value": "–" if batch is None else f"{batch['runtime_sec']:.0f}s", "delta": None if batch is None else f"{batch['runs']} runs"},
    ])
    cov = cc["coverage"]
    st.markdown("**Data coverage**")
    if cov.get("available"):
        st.markdown(f"{cov['start']:%Y-%m-%d} to {cov['end']:%Y-%m-%d} · {cov['sessions']} sessions · {cov['bars']:,} bars · quality **{cov['quality']}**")
        for issue in cov["issues"]:
            st.caption(f"• {issue}")
        if cov["sessions"] < SETTINGS.evidence.min_sessions:
            st.warning(f"{cov['sessions']} sessions is below the policy minimum of {SETTINGS.evidence.min_sessions}; strategies will be INSUFFICIENT DATA.")
    else:
        ev.unavailable("Data coverage", cov.get("reason", ""))
    if cc["awaiting_data"]:
        st.markdown("**Awaiting data / quality failures**")
        st.dataframe(dd.runs_frame(cc["awaiting_data"]), width="stretch", hide_index=True)
    if cc["failed"]:
        st.markdown("**Failed runs**")
        st.dataframe(dd.runs_frame(cc["failed"]), width="stretch", hide_index=True)
    if cc["held"]:
        st.markdown("**Promotions held** (the accepted verdict stands until enough new data arrives)")
        st.dataframe(dd.runs_frame(cc["held"]), width="stretch", hide_index=True)
    st.markdown("**Status changes** (append-only, with reasons)")
    if cc["status_changes"]:
        st.dataframe([{k: c[k] for k in ("created_at", "strategy", "kind", "from_state", "to_state", "reason")} for c in cc["status_changes"]],
                     width="stretch", hide_index=True)
    else:
        st.caption("None yet.")
    with st.expander("All recent runs"):
        st.dataframe(dd.runs_frame(cc["runs"]), width="stretch", hide_index=True, height=320)

# ---- 4. ranking explanation ---------------------------------------------------------------------------------------
with tab_explain:
    if not evidence:
        empty_state("No evidence stored for this instrument. ", "There is nothing to explain yet.")
    elif ranking is None:
        ev.unavailable("Ranking explanation", ranking_note or "the live market could not be loaded")
    else:
        names = [d_.strategy for d_ in ranking.decisions]
        pick = ranking.ranking.selected_strategy
        name = st.selectbox("Strategy", names, index=names.index(pick) if pick in names else 0, key="explain_strategy")
        ex = dd.explain_rank(ranking, evidence, name)
        dec = ex["decision"]
        st.markdown(f"{ev.state_chip(dec['research_status'])} &nbsp; **{name}** · final status **{ex['final_status']}**"
                    + (" · selected for the current market" if ex["selected"] else ""), unsafe_allow_html=True)
        if ex["eligible_position"] is not None:
            st.markdown(f"Research rank **{dec['research_rank']} of {ex['n_eligible']}** among strategies that pass every gate.")
            c1, c2 = st.columns(2)
            c1.markdown("**Ranked just above**<br>" + ("none (this is first)" if ex["above"] is None else f"{ex['above']['strategy']} ({ex['above']['state']})"), unsafe_allow_html=True)
            c2.markdown("**Ranked just below**<br>" + ("none (this is last)" if ex["below"] is None else f"{ex['below']['strategy']} ({ex['below']['state']})"), unsafe_allow_html=True)
            st.caption("Order is by evidence tier (state, then bands of out-of-sample lower bound, walk-forward consistency, parameter stability, cost "
                       "sensitivity), then by out-of-sample lower bound. It is not a weighted score.")
        else:
            st.markdown("This strategy does not pass every gate, so it has no research rank.")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Gates passed**")
            for g in ex["passed_gates"]:
                st.markdown(f"✔ {g['name']}: {g['detail']}")
        with g2:
            st.markdown("**Gates failed**")
            for g in ex["failed_gates"] or [{"name": "none", "detail": ""}]:
                st.markdown(f"✖ {g['name']}: {g['detail']}" if ex["failed_gates"] else "none")
        st.markdown(f"**Fit with the current market:** {ex['regime_fit']['status']}. {ex['regime_fit']['detail']}")
        st.markdown(f"**Entry setup on the latest bar:** {ex['setup'] or 'not assessed (the strategy is not a candidate right now)'}")
        st.markdown("**What remains uncertain**")
        for u in ex["uncertainty"] or ["Nothing specific is flagged, which does not mean the result is certain."]:
            st.markdown(f"- {u}")
        st.caption("The ranker never grants permission to trade. Every order still goes through the risk engine.")
