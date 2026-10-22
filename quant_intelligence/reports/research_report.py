"""
Per-run research report: the evidence behind every verdict, in Markdown and JSON.

A verdict without its numbers is just a label, so each strategy section carries the sample, the net edge with its interval,
the out-of-sample and walk-forward results, the robustness outcomes, the reasons for the state, what the scheduler did with
the verdict (accepted or held, and why), and the known limitations of the strategy's backtest. Wording describes evidence
under a documented policy and never predicts profit. Stored in `research_reports` (report_type RESEARCH_RUN).
"""
from __future__ import annotations

import datetime as dt

from quant_intelligence.strategies.rules import rules_for
from quant_intelligence.validation.evidence import STATE_DEFINITIONS, EvidenceState

DISCLAIMER = ("These results describe how a strategy behaved on past data under the stated assumptions and policy. "
              "They are not a prediction, not investment advice, and a state can be revoked by later data.")


def _fmt(x, spec="+.3f", suffix=""):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:{spec}}{suffix}"


def strategy_section(outcome: dict) -> dict:
    """The JSON section for one strategy from a scheduler outcome dict (see research/scheduler.RunOutcome.as_report_dict)."""
    rules = rules_for(outcome["strategy"])
    state = outcome.get("state")
    return {**outcome, "definition": STATE_DEFINITIONS.get(EvidenceState(state)) if state else None,
            "limitations": list(rules.limitations) if rules is not None else []}


def build_run_report(instrument: str, timeframe: str, batch_id: str, data: dict, outcomes: list[dict], version: dict | None,
                     changes: list[dict], generated_at: dt.datetime | None = None) -> dict:
    """`data`: {fingerprint, start, end, n_sessions, quality, issues}. `version`: the research version written by this batch, if any."""
    sections = [strategy_section(o) for o in outcomes]
    content = {"instrument": instrument, "timeframe": timeframe, "batch_id": batch_id,
               "generated_at": str(generated_at or dt.datetime.now()), "data": data, "version": version,
               "status_changes": changes, "strategies": sections, "disclaimer": DISCLAIMER}
    return {"content_json": content, "content_markdown": render_markdown(content)}


def render_markdown(c: dict) -> str:
    d = c["data"]
    lines = [f"# Research run: {c['instrument']} {c['timeframe']}", "",
             f"Batch `{c['batch_id']}`, generated {c['generated_at']}.", "",
             "## Data",
             f"- {d.get('start')} to {d.get('end')}, {d.get('n_sessions')} sessions, quality **{d.get('quality')}**, "
             f"fingerprint `{(d.get('fingerprint') or '')[:16]}`"]
    lines += [f"- issue: {i}" for i in d.get("issues", [])]
    if c.get("version"):
        lines += ["", f"## Accepted version {c['version']['version_no']}", f"- {len(c['version']['leaderboard'])} strategies on the accepted leaderboard"]
    if c["status_changes"]:
        lines += ["", "## Status changes"]
        lines += [f"- **{s['strategy']}**: {s['kind']} ({s.get('from_state') or 'none'} -> {s['to_state']}). {s['reason']}" for s in c["status_changes"]]
    lines += ["", "## Strategies"]
    for s in c["strategies"]:
        lines += ["", f"### {s['strategy']}", f"- Run: {s['outcome']}" + (f" - {s['detail']}" if s.get("detail") else "")]
        if s.get("state"):
            lines.append(f"- Evidence state: **{s['state']}**. {s.get('definition') or ''}")
        if s.get("decision"):
            dec = s["decision"]
            lines.append(f"- Decision: {dec['kind']} ({'accepted' if dec['accepted'] else 'NOT accepted'}). {dec['reason']}")
        m = s.get("metrics")
        if m:
            lines.append(f"- Development sample: {m['n_trades']} trades over {m['n_trade_days']} days "
                         f"(effective n {_fmt(s.get('effective_n'), '.0f')}); net expectancy {_fmt(m['net_expectancy_r'])} R "
                         f"[{_fmt(s.get('ci_low'))}, {_fmt(s.get('ci_high'))}], profit factor {_fmt(m['profit_factor'], '.2f')}, "
                         f"max drawdown {_fmt(m['max_drawdown_r'], '.1f')} R")
        if s.get("oos"):
            o = s["oos"]
            lines.append(f"- Out of sample: {o['n_trades']} trades, net {_fmt(o['net_expectancy_r'])} R (lower bound {_fmt(o.get('ci_low'))})")
        if s.get("walk_forward"):
            w = s["walk_forward"]
            lines.append(f"- Walk-forward: {_fmt(w.get('share_positive'), '.0%')} of {w.get('n_judged')} folds positive, worst fold {_fmt(w.get('worst_fold_r'))} R")
        if s.get("robustness"):
            lines.append("- Robustness: " + ", ".join(f"{k} {'passed' if v else 'FAILED' if v is False else 'not run'}" for k, v in s["robustness"].items()))
        lines += [f"- Reason: {r}" for r in s.get("reasons", [])]
        lines += [f"- Limitation: {x}" for x in s.get("limitations", [])]
    lines += ["", "## Caveats", f"- {c['disclaimer']}"]
    return "\n".join(lines)


def persist_run_report(instrument: str, report: dict) -> int | None:
    """Store the report; returns its id, or None if storage failed (never raises into a research run)."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import ResearchReport

        with get_session() as session:
            row = ResearchReport(instrument=instrument, report_type="RESEARCH_RUN", content_json=report["content_json"],
                                 content_markdown=report["content_markdown"])
            session.add(row)
            session.flush()
            return int(row.id)
    except Exception:
        return None
