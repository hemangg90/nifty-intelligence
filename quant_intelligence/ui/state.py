"""Shared Streamlit session-state helpers.

Keeps pipeline results, the PaperBroker instance, and account state in
st.session_state so every page sees a consistent snapshot without
recomputing the whole research pipeline on every rerun.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
from quant_intelligence.utils.timeutil import now_ist

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.dhan_rate_limit import LIMITER
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.data_keeper import DATA_KEEPER
from quant_intelligence.execution.engine import COMMODITY_RUNNER, ENGINE
from quant_intelligence.risk.risk_engine import AccountState

# A not-OK pipeline result is recomputed after this long (the data feed may have recovered meanwhile).
_NOT_OK_RETRY_SEC = 60.0
# A new closed bar triggers a recompute, but never more often than this (a pipeline run is expensive).
_MIN_RERUN_SEC = 60.0


def credentials_fingerprint() -> str:
    """Changes whenever the Client ID or access token changes (never exposes either)."""
    raw = f"{SETTINGS.dhan_client_id}|{SETTINGS.dhan_access_token}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def pipeline_needs_refresh(output, ran_at: float | None, token_fp: str | None, instrument: str, timeframe: str,
                           now: float | None = None, now_dt: dt.datetime | None = None) -> str | None:
    """Why a cached pipeline result should be recomputed, or None if it is still good.

    A browser session used to keep its first pipeline answer forever, so a "data quality degraded" result
    stayed on screen after the token was fixed or the feed recovered. Recompute when the credentials changed,
    when the last result was not OK and a minute has passed, or when a newer bar should exist by now.
    """
    from quant_intelligence.data.data_manager import _parse_timeframe_minutes
    from quant_intelligence.utils.market_calendar import expected_last_closed_bar_start
    from quant_intelligence.utils.market_profile import profile_for

    now = time.monotonic() if now is None else now
    age = None if ran_at is None else now - ran_at
    if token_fp != credentials_fingerprint():
        return "credentials changed"
    if age is not None and age < _MIN_RERUN_SEC and output.data_quality_status == "OK":
        return None
    if output.data_quality_status != "OK" and (age is None or age >= _NOT_OK_RETRY_SEC):
        return f"last result was {output.data_quality_status}"
    if age is not None and age >= _MIN_RERUN_SEC:
        expected = expected_last_closed_bar_start(now_dt or now_ist(), profile_for(instrument), _parse_timeframe_minutes(timeframe))
        if expected > output.timestamp.replace(tzinfo=None):
            return "a newer bar has closed"
    return None


def invalidate_cached_analysis() -> None:
    """Forget every cached analysis so the next page run recomputes it from fresh data (call after the
    credentials change, so a DEGRADED answer computed with the old token does not linger)."""
    for key in ("pipeline_output", "pipeline_error", "option_chain", "pipeline_for", "pipeline_ran_at", "pipeline_token_fp"):
        st.session_state[key] = None
    ENGINE.last_rows = []
    COMMODITY_RUNNER.last_rows = []
    from quant_intelligence.data import data_manager as dm

    dm._last_attempt.clear()
    dm._last_refresh_error.clear()
    DATA_KEEPER.wake()


def init_session_state() -> None:
    if "instrument" not in st.session_state:
        st.session_state["instrument"] = SETTINGS.default_instrument
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = SETTINGS.default_timeframe
    if "lookback_days" not in st.session_state:
        st.session_state["lookback_days"] = 60
    if "pipeline_output" not in st.session_state:
        st.session_state["pipeline_output"] = None
    if "pipeline_error" not in st.session_state:
        st.session_state["pipeline_error"] = None
    # The paper account, risk counters and kill switch are process-wide (ENGINE), not per
    # browser session, so the background auto-trader and every page see the same account.
    ENGINE.restore_from_db()  # no-op after the first call in this process
    st.session_state["paper_broker"] = ENGINE.broker
    if "dhan_api_client" not in st.session_state:
        st.session_state["dhan_api_client"] = DhanApiClient()
    if "option_underlying" not in st.session_state:
        st.session_state["option_underlying"] = SETTINGS.option_underlyings[0] if SETTINGS.option_underlyings else "NIFTY"
    if "option_chain" not in st.session_state:
        st.session_state["option_chain"] = None
    DATA_KEEPER.start()  # idempotent: keeps every watchlist instrument's candles current in the background
    if SETTINGS.research_scheduler:
        from quant_intelligence.research.scheduler import RESEARCH_SCHEDULER

        RESEARCH_SCHEDULER.start()  # idempotent; off unless RESEARCH_SCHEDULER=true
    render_engine_sidebar()


def render_engine_sidebar() -> None:
    """Visible on every page: shows the background auto-trader and lets you stop it."""
    with st.sidebar:
        snap = LIMITER.snapshot()
        if snap["auth_block_remaining"] > 0:
            st.error(
                "Dhan rejected your access token (it expires about every 24 hours). Requests are paused so the "
                "account is not hammered - enter a fresh token under 'API Keys' and they resume immediately."
            )
        paused = snap["cooldown_remaining"]
        if paused > 0:
            st.warning(
                f"Dhan rate limit reached - data requests are paused for {paused:.0f}s to protect your account "
                "and resume automatically. Make sure only one copy of the app uses this Dhan account."
            )
        for runner, label in ((ENGINE, "NSE"), (COMMODITY_RUNNER, "Commodities")):
            status = runner.status()
            if not status["running"]:
                continue
            st.success(f"{label} auto paper trading is RUNNING (keeps running while you switch pages)")
            st.caption(
                f"Cycles: {status['cycles']} | Last: "
                f"{status['last_cycle_at']:%H:%M:%S} IST" if status["last_cycle_at"] else "No cycle completed yet"
            )
            st.caption(status["last_status"])
            if st.button(f"Stop {label} auto trading", key=f"sidebar_stop_{runner.name}"):
                runner.stop()
                st.rerun()


def get_account_state() -> AccountState:
    return ENGINE.account_state()


def get_live_account_state(broker) -> AccountState:
    """AccountState for the live Dhan broker, using real fund/position data instead of
    the PaperBroker's in-memory book. `broker` is a DhanBroker."""
    fund = broker.get_fund_limit()
    equity = fund.get("availabelBalance", 0.0)
    ENGINE.peak_equity = max(ENGINE.peak_equity, equity)

    open_positions = [p for p in broker.get_positions() if p.get("netQty", 0) != 0]
    exposure_by_strategy: dict[str, float] = {}
    total_exposure = 0.0
    open_summary = []
    for p in open_positions:
        risk_amt = abs(p.get("netQty", 0)) * abs(p.get("buyAvg", 0.0) - p.get("sellAvg", 0.0))
        exposure_by_strategy["dhan_live"] = exposure_by_strategy.get("dhan_live", 0.0) + risk_amt
        total_exposure += risk_amt
        open_summary.append(live_position_view(p))

    return AccountState(
        equity=equity,
        peak_equity=ENGINE.peak_equity,
        daily_pnl=ENGINE.daily_pnl,
        open_positions_count=len(open_positions),
        trades_today=ENGINE.trades_today,
        exposure_by_strategy=exposure_by_strategy,
        total_exposure=total_exposure,
        broker_connected=broker.is_connected(),
        kill_switch_engaged=ENGINE.kill_switch,
        open_positions=open_summary,
    )


def live_position_view(p: dict) -> dict:
    """{"instrument", "direction", "risk"} for a Dhan position, for the risk engine's open-count and
    correlated-group checks. Direction is the view on the UNDERLYING: long call / short put = LONG,
    long put / short call = SHORT; a non-option position follows its own sign. Per-position stop risk
    is not known for live positions, so `risk` is 0 (the group COUNT limit still applies)."""
    symbol = str(p.get("tradingSymbol") or "")
    underlying = symbol.split("-", 1)[0].strip().upper()
    long_qty = (p.get("netQty") or 0) > 0
    option = str(p.get("drvOptionType") or "").upper()
    if option in ("CALL", "CE"):
        direction = "LONG" if long_qty else "SHORT"
    elif option in ("PUT", "PE"):
        direction = "SHORT" if long_qty else "LONG"
    else:
        direction = "LONG" if long_qty else "SHORT"
    return {"instrument": underlying, "direction": direction, "risk": 0.0}


def run_pipeline_cached(force: bool = False):
    from quant_intelligence.research.pipeline import run_pipeline

    cached = st.session_state["pipeline_output"]
    if cached is not None and not force:
        why = pipeline_needs_refresh(
            cached, st.session_state.get("pipeline_ran_at"), st.session_state.get("pipeline_token_fp"),
            st.session_state["instrument"], st.session_state["timeframe"],
        )
        if why is None:
            return cached

    end = now_ist()
    start = end - dt.timedelta(days=st.session_state["lookback_days"])

    try:
        output = run_pipeline(st.session_state["instrument"], st.session_state["timeframe"], start, end)
        st.session_state["pipeline_output"] = output
        st.session_state["pipeline_error"] = None
        st.session_state["pipeline_ran_at"] = time.monotonic()
        st.session_state["pipeline_token_fp"] = credentials_fingerprint()
        return output
    except Exception as e:
        st.session_state["pipeline_error"] = str(e)
        st.session_state["pipeline_output"] = None
        return None
