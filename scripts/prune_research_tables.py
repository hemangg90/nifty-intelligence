"""Shrink the dashboard database by removing the automatic RESEARCH backtests.

Every scan/page refresh used to save a full backtest per strategy (split "RESEARCH"), which grew the database
to gigabytes (~700k trade rows, 11k runs) and nothing reads them back. This removes those rows - and only
those: the explicit Backtest Lab runs (IN_SAMPLE / OUT_OF_SAMPLE / WALK_FORWARD*) and every order, position
and fill are kept.

    python scripts/prune_research_tables.py                  # DRY RUN: shows what would go, changes nothing
    python scripts/prune_research_tables.py --apply          # backs the DB up first, deletes, then VACUUMs
    python scripts/prune_research_tables.py --keep-days 3    # keep RESEARCH runs from the last 3 days
    python scripts/prune_research_tables.py --apply --no-backup   # skip the backup copy (it is as big as the DB)
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data_cache" / "quant_intelligence.db"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--keep-days", type=float, default=1.0, help="keep RESEARCH runs newer than this (default 1)")
    parser.add_argument("--apply", action="store_true", help="actually delete (backs up first unless --no-backup)")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}")
        return 1
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=args.keep_days)).isoformat(sep=" ")
    conn = sqlite3.connect(args.db)
    old_runs = "SELECT run_id FROM backtest_runs WHERE split = 'RESEARCH' AND created_at < ?"
    plan = [
        ("backtest_trades", f"run_id IN ({old_runs})"),
        ("strategy_observations", f"run_id IN ({old_runs})"),
        ("backtest_runs", "split = 'RESEARCH' AND created_at < ?"),
    ]
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE {w}", (cutoff,)).fetchone()[0] for t, w in plan}
    kept = conn.execute("SELECT split, COUNT(*) FROM backtest_runs GROUP BY split").fetchall()
    size_mb = args.db.stat().st_size / 1e6
    print(f"Database: {args.db} ({size_mb:,.0f} MB); RESEARCH runs older than {args.keep_days:g} day(s) are removable")
    for table, n in counts.items():
        print(f"  {table:<22} {n:>9,} rows to remove")
    print("  backtest runs by split now: " + ", ".join(f"{s}={n}" for s, n in kept))
    print("  orders, positions and fills are never touched")
    # Research governance is an audit trail: append-only and small, so it is counted for information and never deleted.
    for table in ("research_versions", "status_changes", "research_run_log", "strategy_evidence", "strategy_regime_stats", "holdout_ledger"):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<22} {n:>9,} rows kept (audit trail, never pruned)")
        except sqlite3.OperationalError:
            pass  # table not created yet

    if not any(counts.values()):
        print("Nothing to remove.")
        return 0
    if not args.apply:
        print("\nDry run - nothing changed. Re-run with --apply to delete.")
        return 0

    conn.close()
    if not args.no_backup:
        backup = args.db.with_name(f"{args.db.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{args.db.suffix}")
        shutil.copy2(args.db, backup)
        print(f"\nBackup written: {backup}")
    conn = sqlite3.connect(args.db)
    with conn:  # one transaction: all or nothing
        for table, where in plan:
            conn.execute(f"DELETE FROM {table} WHERE {where}", (cutoff,))
    print("Deleted: " + ", ".join(f"{n:,} {t}" for t, n in counts.items()))
    print("Compacting the file (VACUUM) ...")
    conn.execute("VACUUM")
    conn.close()
    print(f"Done: {args.db.stat().st_size / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
