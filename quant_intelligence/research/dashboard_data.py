"""
Data for the Strategy Evidence dashboard, with no Streamlit in it (so it is testable and the page stays thin).

Everything here reads accepted, stored evidence (strategy_evidence, research_versions, status_changes, research_run_log);
nothing re-runs a backtest. The one live input is the ranking decision for the current market, which the page passes in
when it has run the pipeline. Where something has not been computed the frames say so (None / empty) instead of filling in.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ranking.gated_ranker import (
    NOT_ASSESSED,
    GatedRanking,
    StrategyDecision,
    research_stage,
)
from quant_intelligence.research import records
from quant_intelligence.strategies.rules import rules_for, version_for
from quant_intelligence.validation.evidence import REQUIRED_ROBUSTNESS_TESTS
from quant_intelligence.validation.evidence_store import EvidenceSnapshot, load_latest_evidence

SORT_WARNING = ("Sorting is by evidence tier, not by profit. A high position is not a performance guarantee, and a "
                "VALIDATED FOR RESEARCH label refers to this research process only.")

LEADERBOARD_COLUMNS = ["Rank", "Strategy", "Evidence state", "Net R/trade", "95% CI low", "95% CI high", "Profit factor", "Max DD (R)",
                       "OOS net R", "OOS lower bound", "OOS trades", "Walk-forward +", "Robustness", "Regime fit", "Setup",
                       "Final status", "Evidence age (d)", "Data", "Eligible"]


def stored_targets() -> list[tuple[str, str]]:
    """(instrument, timeframe) pairs that have stored evidence, most recently updated first."""
    try:
        from sqlalchemy import func

        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import StrategyEvidence as T

        with get_session() as s:
            rows = s.query(T.instrument, T.timeframe, func.max(T.id)).group_by(T.instrument, T.timeframe).order_by(func.max(T.id).desc()).all()
            return [(str(i), str(t)) for i, t, _ in rows]
    except Exception:
        return []


def robustness_summary(rob: dict | None) -> str:
    """'3 passed, 1 failed, 1 not run' over the five required tests; 'not run' when robustness never ran."""
    if not rob:
        return "not run"
    outcomes = [rob.get(n) for n in REQUIRED_ROBUSTNESS_TESTS]
    parts = [f"{sum(o is True for o in outcomes)} passed"]
    if any(o is False for o in outcomes):
        parts.append(f"{sum(o is False for o in outcomes)} FAILED")
    if any(o is None for o in outcomes):
        parts.append(f"{sum(o is None for o in outcomes)} not run")
    return ", ".join(parts)


def _age(snap: EvidenceSnapshot, now: dt.datetime | None) -> float | None:
    a = snap.age_days(now)
    return None if a is None else round(a, 1)


def leaderboard_frame(instrument: str, timeframe: str, evidence: dict[str, EvidenceSnapshot] | None = None,
                      ranking: GatedRanking | None = None, regime_label: str | None = None,
                      regime_stats: dict | None = None, now: dt.datetime | None = None) -> pd.DataFrame:
    """One row per strategy with stored evidence, eligible strategies first in research-rank order.

    Without `ranking` (the current-market decision) the Regime fit, Setup and Final status columns are '-' because they
    depend on the market right now; the gates and rank do not."""
    evidence = load_latest_evidence(instrument, timeframe) if evidence is None else evidence
    if not evidence:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    names = [(n, version_for(n)) for n in evidence]
    decisions, _, _ = research_stage(names, evidence, "OK", SETTINGS.gated, now)    # data quality is judged live, not here
    live: dict[str, StrategyDecision] = {} if ranking is None else {d.strategy: d for d in ranking.decisions}
    rows = []
    for name, snap in evidence.items():
        s, d = snap.summary, decisions[name]
        oos, wf = s.get("oos") or {}, s.get("walk_forward") or {}
        l = live.get(name)
        rows.append({
            "Rank": d.research_rank, "Strategy": name, "Evidence state": snap.state.value, "Net R/trade": s.get("net_expectancy_r"),
            "95% CI low": s.get("ci_low"), "95% CI high": s.get("ci_high"), "Profit factor": s.get("profit_factor"),
            "Max DD (R)": None if s.get("max_drawdown_r") is None else abs(s["max_drawdown_r"]),
            "OOS net R": oos.get("net_expectancy_r"), "OOS lower bound": oos.get("ci_low"), "OOS trades": oos.get("n_trades"),
            "Walk-forward +": wf.get("share_positive"), "Robustness": robustness_summary(s.get("robustness")),
            "Regime fit": "-" if l is None else l.compatibility, "Setup": "-" if l is None else l.setup_status,
            "Final status": "-" if l is None else l.final_status, "Evidence age (d)": _age(snap, now),
            "Data": "proxy" if s.get("proxy_data") else "price history", "Eligible": d.eligible,
        })
    df = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
    df["_sort"] = df["Rank"].fillna(10_000)
    return df.sort_values(["_sort", "Net R/trade"], ascending=[True, False], na_position="last").drop(columns="_sort").reset_index(drop=True)


# ---- research detail ----------------------------------------------------------------------------------------------

def strategy_detail(snap: EvidenceSnapshot) -> dict:
    """Frames and tables for one strategy from its stored snapshot. Missing pieces are None, never invented."""
    s = snap.summary
    d = s.get("detail") or {}
    rob = d.get("robustness") or {}
    out: dict = {"state": snap.state.value, "summary": s, "limitations": [], "has_detail": bool(d) and "error" not in d, "error": d.get("error")}
    rules = rules_for(snap.strategy)
    if rules is not None:
        out["limitations"] = list(rules.limitations)
        out["rules"] = {"entry": rules.entry, "exit": rules.exit, "stop": rules.stop, "target": rules.target}

    eq = d.get("equity")
    out["equity"] = None if not eq else pd.DataFrame(eq, columns=["exit_ts", "equity_r", "drawdown_r", "segment"]).assign(
        exit_ts=lambda f: pd.to_datetime(f["exit_ts"]))
    h = d.get("histogram")
    out["histogram"] = None if not h else pd.DataFrame({"net_r": [(a + b) / 2 for a, b in zip(h["edges"][:-1], h["edges"][1:])], "trades": h["counts"]})
    out["periods"] = pd.DataFrame(d["periods"]) if d.get("periods") else None
    out["regimes"] = pd.DataFrame(d["regimes"]) if d.get("regimes") else None
    out["trades"] = pd.DataFrame(d["trades"]) if d.get("trades") else None
    out["trades_note"] = None if not d.get("trades") else f"latest {d.get('trades_shown')} of {d.get('n_trades')} trades"

    ps = rob.get("parameter_sensitivity") or {}
    out["parameters"] = pd.DataFrame([r for r in ps.get("rows", []) if r.get("net_expectancy_r") is not None]) if ps.get("rows") else None
    cs = rob.get("cost_stress") or {}
    out["cost_stress"] = pd.DataFrame(cs["rows"]) if cs.get("rows") else None
    sq = rob.get("sequence_analysis") or {}
    out["sequence"] = (sq.get("rows") or [None])[0]
    out["robustness_tests"] = [{"test": n, "result": ("passed" if (rob.get(n) or {}).get("passed") is True else
                                                      "FAILED" if (rob.get(n) or {}).get("passed") is False else "not run"),
                                "detail": (rob.get(n) or {}).get("detail", "")} for n in REQUIRED_ROBUSTNESS_TESTS]
    wf = (s.get("walk_forward") or {}).get("folds")
    out["walk_forward"] = pd.DataFrame(wf) if wf else None
    out["oos"] = s.get("oos")
    out["holdout"] = s.get("holdout")
    return out


# ---- ranking explanation ------------------------------------------------------------------------------------------

def explain_rank(ranking: GatedRanking, evidence: dict[str, EvidenceSnapshot], strategy: str,
                 now: dt.datetime | None = None) -> dict:
    """Why `strategy` sits where it does in the current gated ranking."""
    d = ranking.decision_for(strategy)
    if d is None:
        return {"found": False}
    eligible = [x for x in ranking.decisions if x.eligible]            # already in research-rank order
    pos = next((i for i, x in enumerate(eligible) if x.strategy == strategy), None)
    brief = lambda x: {"strategy": x.strategy, "research_rank": x.research_rank, "tier": list(x.tier), "state": x.research_status}  # noqa: E731
    snap = evidence.get(strategy)
    s = snap.summary if snap else {}
    uncertainty: list[str] = []
    if snap:
        lo, hi = s.get("ci_low"), s.get("ci_high")
        if lo is not None and hi is not None:
            uncertainty.append(f"net expectancy is only known to within [{lo:+.3f}, {hi:+.3f}] R per trade")
        n, eff = s.get("n_trades"), s.get("effective_n")
        if n and eff and eff < 0.8 * n:
            uncertainty.append(f"{n} trades carry the information of about {eff:.0f} independent ones (same-session and serial dependence)")
        oos = s.get("oos") or {}
        if oos.get("n_trades") is not None and oos["n_trades"] < 2 * SETTINGS.evidence.min_oos_trades:
            uncertainty.append(f"only {oos['n_trades']} out-of-sample trades")
        unrun = [k for k, v in (s.get("robustness") or {}).items() if v is None]
        if not s.get("robustness"):
            uncertainty.append("robustness tests have not been run")
        elif unrun:
            uncertainty.append("robustness tests not run: " + ", ".join(unrun))
        if s.get("proxy_data"):
            uncertainty.append("validated on proxy data (near-the-money option strikes), so it can never reach VALIDATED FOR RESEARCH")
        if (s.get("holdout") or {}).get("consumed"):
            uncertainty.append("the final holdout has been looked at too often to count as untouched")
        age = snap.age_days(now)
        if age is not None and age > SETTINGS.gated.max_evidence_age_days / 2:
            uncertainty.append(f"evidence is {age:.0f} days old")
    return {"found": True, "decision": d.as_dict(), "eligible_position": pos, "n_eligible": len(eligible),
            "above": None if pos in (None, 0) else brief(eligible[pos - 1]),
            "below": None if pos is None or pos + 1 >= len(eligible) else brief(eligible[pos + 1]),
            "passed_gates": [g.as_dict() for g in d.gates if g.passed], "failed_gates": [g.as_dict() for g in d.gates if not g.passed],
            "regime_fit": {"status": d.compatibility, "detail": d.compatibility_detail},
            "setup": d.setup_status if d.setup_status != NOT_ASSESSED else None, "final_status": d.final_status, "uncertainty": uncertainty,
            "selected": ranking.ranking.selected_strategy == strategy}


def current_ranking(output, timeframe: str, evidence: dict[str, EvidenceSnapshot] | None = None) -> GatedRanking:
    """What the gated ranker says for the market in `output` (a research.pipeline.PipelineOutput), whether or not
    GATED_RANKER is switched on. Nothing is traded or stored."""
    from quant_intelligence.ranking.gated_ranker import gated_rank
    from quant_intelligence.strategies.registry import get_strategy

    strategies = [get_strategy(si.strategy_name) for si in output.strategy_intel]
    return gated_rank(strategies, [si.score for si in output.strategy_intel], output.instrument, timeframe, output.data_quality_status,
                      output.regime_label, market_state=output.market_state, recent_ohlcv=output.ohlcv.tail(5), evidence=evidence)


# ---- backtest control center --------------------------------------------------------------------------------------

def data_coverage(instrument: str, timeframe: str) -> dict:
    """What the cached candles cover and how they audit. Reads the cache only (no network)."""
    try:
        from quant_intelligence.data.data_manager import CACHE_DIR, _read_cache
        from quant_intelligence.data.quality import clean_ohlcv
        from quant_intelligence.data.quality_gates import check_history
        from quant_intelligence.utils.market_profile import profile_for
        from quant_intelligence.utils.timeframe import parse_timeframe_minutes

        path = CACHE_DIR / f"{instrument}_{timeframe}.parquet"
        raw = _read_cache(path) if path.exists() else None
        if raw is None or len(raw) == 0:
            return {"available": False, "reason": "no cached history"}
        tf = parse_timeframe_minutes(timeframe)
        df = clean_ohlcv(raw, tf, profile_for(instrument))
        rep = check_history(df, tf, profile_for(instrument), instrument=instrument, timeframe=timeframe)
        ts = pd.to_datetime(df["timestamp"])
        return {"available": True, "start": ts.min(), "end": ts.max(), "bars": int(len(df)), "sessions": int(ts.dt.date.nunique()),
                "quality": rep.status, "issues": list(rep.issues)}
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def control_center(instrument: str, timeframe: str, now: dt.datetime | None = None, coverage: dict | None = None) -> dict:
    runs = records.run_history(instrument, timeframe, limit=500)
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    counts: dict[str, int] = {}
    for r in runs:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    last_ok = next((r for r in runs if r["outcome"] == "COMPLETED"), None)
    latest_batch = runs[0]["batch_id"] if runs else None
    batch = [r for r in runs if r["batch_id"] == latest_batch]
    return {
        "counts": counts,
        "total_runs": len(runs),
        "last_success": None if last_ok is None else {"strategy": last_ok["strategy"], "at": last_ok["finished_at"] or last_ok["started_at"],
                                                      "state": last_ok["state"]},
        "latest_batch": None if not batch else {"batch_id": latest_batch, "runs": len(batch),
                                                "runtime_sec": round(sum((r["runtime_sec"] or 0.0) for r in batch), 1),
                                                "started_at": min(r["started_at"] for r in batch if r["started_at"])},
        "awaiting_data": [r for r in runs if r["outcome"] == "AWAITING_DATA"][:5],
        "failed": [r for r in runs if r["outcome"] == "FAILED"][:10],
        "held": [r for r in runs if r["decision"] == "PROMOTION_HELD"][:10],
        "version": records.latest_version(instrument, timeframe),
        "status_changes": records.status_history(instrument, timeframe, limit=15),
        "coverage": coverage if coverage is not None else data_coverage(instrument, timeframe),
        "runs": runs[:100],
        "as_of": now,
    }


def runs_frame(runs: list[dict]) -> pd.DataFrame:
    cols = ["started_at", "strategy", "outcome", "state", "decision", "accepted", "runtime_sec", "detail"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in runs], columns=cols)
