"""Run the research pipeline by hand: validate strategies, decide what to accept, write the report.

    python scripts/run_research.py NIFTY                       # stale strategies only, as the scheduler would
    python scripts/run_research.py NIFTY BANKNIFTY --force     # everything, even while the market is open
    python scripts/run_research.py NIFTY --only "Momentum" "Donchian Channel Breakout" --json
    python scripts/run_research.py NIFTY --status              # accepted versions and recent status changes, nothing run

Uses the cached candles (python scripts/backfill_history.py first). Always next-open fills. The final holdout is never
touched by this command; it is evaluated only through validation.suite with include_holdout=True, on purpose.
Exit code 1 if any strategy failed to run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_intelligence.config.settings import SETTINGS  # noqa: E402
from quant_intelligence.research import records  # noqa: E402
from quant_intelligence.research.scheduler import ResearchScheduler  # noqa: E402


def show_status(symbols: list[str], timeframe: str) -> dict:
    out = {}
    for sym in symbols:
        v = records.latest_version(sym, timeframe)
        out[sym] = {"version": None if v is None else {k: v[k] for k in ("version_no", "created_at", "data_end", "policy_hash")},
                    "leaderboard": [] if v is None else v["leaderboard"],
                    "recent_status_changes": records.status_history(sym, timeframe, limit=10)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--timeframe", default=SETTINGS.default_timeframe)
    ap.add_argument("--force", action="store_true", help="re-run every strategy and ignore the market-hours deferral")
    ap.add_argument("--only", nargs="+", metavar="STRATEGY", help="restrict to these strategies")
    ap.add_argument("--status", action="store_true", help="show the accepted versions and status changes; run nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    symbols = [s.upper() for s in args.symbols]

    if args.status:
        data = show_status(symbols, args.timeframe)
        print(json.dumps(data, indent=2, default=str) if args.json else "\n".join(
            f"{s}: " + ("no accepted version" if d["version"] is None else f"version {d['version']['version_no']}, "
                        + ", ".join(f"{r['strategy']}={r['state']}" for r in d["leaderboard"])) for s, d in data.items()))
        return 0

    sched = ResearchScheduler(targets=[(s, args.timeframe) for s in symbols])
    results = sched.run_once(force=args.force, only=args.only)
    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2, default=str))
    else:
        for r in results:
            print(f"{r.instrument} {r.timeframe}: {r.status}" + (f" - {r.detail}" if r.detail else "")
                  + (f" (accepted version {r.version_no}, report {r.report_id})" if r.version_no else ""))
            for o in r.outcomes:
                dec = "" if o.decision is None else f" [{o.decision.kind}{'' if o.decision.accepted else ', not accepted'}]"
                print(f"   {o.strategy:34s} {o.outcome:10s} {o.state or '':22s}{dec} {o.detail}")
    return 1 if any(o.outcome == "FAILED" for r in results for o in r.outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
