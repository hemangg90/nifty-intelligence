"""Fetch historical option prices (Dhan rolling-strike endpoint) and report what came back.

    python scripts/backfill_options.py NIFTY --days 365
    python scripts/backfill_options.py NIFTY --days 90 --strikes 3 --expiry-flag WEEK --json
    python scripts/backfill_options.py NIFTY --audit-only

Needs a valid Dhan token and a data plan that includes expired-options data. It fetches CALL and PUT series for offsets
-N..+N (default 3) in 30-day requests, resumes from what is stored, and stops at the first error (auth, rate limit, plan)
so a bad token does not burn through hundreds of failing calls. Check the first run's output against Dhan's docs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_intelligence.brokers.dhan_api_client import DhanApiClient  # noqa: E402
from quant_intelligence.data_adapters.dhan_options_history import OptionsHistory, backfill_options  # noqa: E402


def audit(symbol: str, expiry_flag: str, expiry_code: int) -> dict:
    h = OptionsHistory.load(symbol, expiry_flag, expiry_code)
    if h is None:
        return {"symbol": symbol, "stored": False}
    lo, hi = h.span()
    return {"symbol": symbol, "stored": True, "offsets": h.offsets, "series": len(h.frames), "from": str(lo), "to": str(hi),
            "rows": int(sum(len(f) for f in h.frames.values())), "fingerprint": h.fingerprint()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--days", type=int, default=365, help="how far back to fetch (Dhan states up to ~5 years)")
    ap.add_argument("--strikes", type=int, default=3, help="fetch offsets -N..+N around the money (stocks allow up to 3)")
    ap.add_argument("--expiry-flag", choices=["WEEK", "MONTH"], default="WEEK")
    ap.add_argument("--expiry-code", type=int, default=1, help="1 = nearest expiry")
    ap.add_argument("--interval", default="5", choices=["1", "5", "15", "25", "60"])
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    client = DhanApiClient()
    if not args.audit_only and not client.is_configured():
        print("Dhan credentials are not set (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN).", file=sys.stderr)
        return 2
    end = dt.date.today() + dt.timedelta(days=1)   # toDate is not inclusive
    out = []
    for sym in (s.upper() for s in args.symbols):
        entry: dict = {"symbol": sym}
        if not args.audit_only:
            res = backfill_options(client, sym, end - dt.timedelta(days=args.days), end, args.strikes,
                                   args.expiry_flag, args.expiry_code, args.interval)
            entry["backfill"] = res.as_dict()
        entry["audit"] = audit(sym, args.expiry_flag, args.expiry_code)
        out.append(entry)

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        for e in out:
            a = e["audit"]
            if not a["stored"]:
                print(f"{e['symbol']}: nothing stored")
            else:
                print(f"{e['symbol']}: {a['rows']:,} rows in {a['series']} series, {a['from']} .. {a['to']}, offsets {a['offsets']}")
            for err in e.get("backfill", {}).get("errors", []):
                print(f"   ERROR {err}")
    return 1 if any(e.get("backfill", {}).get("errors") for e in out) else 0


if __name__ == "__main__":
    raise SystemExit(main())
