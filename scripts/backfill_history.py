"""Backfill long candle history into the parquet cache and audit it.

    python scripts/backfill_history.py NIFTY BANKNIFTY --years 3
    python scripts/backfill_history.py NIFTY --audit-only --json

Needs a valid Dhan token (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN). Re-running continues from the cache edges.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_intelligence.data.backfill import backfill_instrument  # noqa: E402
from quant_intelligence.data.data_manager import CACHE_DIR, _read_cache  # noqa: E402
from quant_intelligence.data.quality import clean_ohlcv  # noqa: E402
from quant_intelligence.data.quality_gates import check_history  # noqa: E402
from quant_intelligence.utils.market_profile import profile_for  # noqa: E402
from quant_intelligence.utils.timeframe import parse_timeframe_minutes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--timeframe", default="5min")
    ap.add_argument("--years", type=float, default=5.0, help="how far back to try (default 5)")
    ap.add_argument("--max-windows", type=int, default=60)
    ap.add_argument("--audit-only", action="store_true", help="skip fetching; just report data quality")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = []
    for sym in (s.upper() for s in args.symbols):
        entry: dict = {"symbol": sym}
        if not args.audit_only:
            target = dt.datetime.now() - dt.timedelta(days=int(args.years * 365))
            entry["backfill"] = backfill_instrument(
                sym, args.timeframe, target_start=target, max_windows=args.max_windows
            ).as_dict()
        df = _read_cache(CACHE_DIR / f"{sym}_{args.timeframe}.parquet")
        if df is None:
            entry["quality"] = {"status": "FAIL", "issues": ["no cached history"]}
        else:
            tf = parse_timeframe_minutes(args.timeframe)
            df = clean_ohlcv(df, tf, profile_for(sym))  # audit what the pipeline sees, after its deterministic cleanup
            rep = check_history(df, tf, profile_for(sym),
                                source="cache", instrument=sym, timeframe=args.timeframe)
            entry["quality"] = rep.as_dict()
        out.append(entry)

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        for e in out:
            q = e["quality"]
            print(f"{e['symbol']}: {q['status']}  rows={q.get('n_rows', 0)}  sessions={q.get('n_sessions', 0)}  "
                  f"{q.get('first_session')} -> {q.get('last_session')}")
            if "backfill" in e:
                b = e["backfill"]
                print(f"   backfill: +{b['rows_added']} rows, earliest {b['earliest']}, {b['stopped_reason']}")
                for err in b["errors"][:3]:
                    print(f"   error: {err}")
            for issue in q.get("issues", []):
                print(f"   - {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
