"""Database access for research governance: run log, accepted versions, status changes. All writes are appends; nothing
here updates or deletes a row. Readers return plain dicts so UI and report code never touch ORM objects."""
from __future__ import annotations

import datetime as dt

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import ResearchRunLog, ResearchVersion, StatusChange


def _run_dict(r: ResearchRunLog) -> dict:
    return {"id": r.id, "started_at": r.started_at, "finished_at": r.finished_at, "batch_id": r.batch_id, "instrument": r.instrument,
            "timeframe": r.timeframe, "strategy": r.strategy, "strategy_version": r.strategy_version, "outcome": r.outcome,
            "state": r.state, "decision": r.decision, "accepted": r.accepted, "detail": r.detail, "runtime_sec": r.runtime_sec,
            "dataset_fingerprint": r.dataset_fingerprint, "data_end": r.data_end, "policy_hash": r.policy_hash}


def log_run(**fields) -> int:
    with get_session() as s:
        row = ResearchRunLog(**fields)
        s.add(row)
        s.flush()
        return int(row.id)


def last_completed_run(instrument: str, timeframe: str, strategy: str, strategy_version: str) -> dict | None:
    with get_session() as s:
        r = (s.query(ResearchRunLog).filter_by(instrument=instrument, timeframe=timeframe, strategy=strategy,
                                               strategy_version=strategy_version, outcome="COMPLETED")
             .order_by(ResearchRunLog.id.desc()).first())
        return _run_dict(r) if r else None


def run_history(instrument: str | None = None, timeframe: str | None = None, limit: int = 200) -> list[dict]:
    with get_session() as s:
        q = s.query(ResearchRunLog)
        if instrument:
            q = q.filter_by(instrument=instrument)
        if timeframe:
            q = q.filter_by(timeframe=timeframe)
        return [_run_dict(r) for r in q.order_by(ResearchRunLog.id.desc()).limit(limit).all()]


def write_version(instrument: str, timeframe: str, batch_id: str, fingerprint: str, data_end: dt.date | None, policy_hash: str,
                  leaderboard: list[dict], notes: str = "") -> tuple[int, int]:
    """Append the next accepted version. Returns (row id, version_no)."""
    with get_session() as s:
        last = (s.query(ResearchVersion.version_no).filter_by(instrument=instrument, timeframe=timeframe)
                .order_by(ResearchVersion.version_no.desc()).first())
        no = (last[0] if last else 0) + 1
        row = ResearchVersion(instrument=instrument, timeframe=timeframe, version_no=no, batch_id=batch_id,
                              dataset_fingerprint=fingerprint, data_end=data_end, policy_hash=policy_hash,
                              leaderboard=leaderboard, notes=notes)
        s.add(row)
        s.flush()
        return int(row.id), no


def latest_version(instrument: str, timeframe: str) -> dict | None:
    with get_session() as s:
        r = (s.query(ResearchVersion).filter_by(instrument=instrument, timeframe=timeframe)
             .order_by(ResearchVersion.version_no.desc()).first())
        return None if r is None else {"id": r.id, "created_at": r.created_at, "version_no": r.version_no, "batch_id": r.batch_id,
                                       "dataset_fingerprint": r.dataset_fingerprint, "data_end": r.data_end,
                                       "policy_hash": r.policy_hash, "leaderboard": r.leaderboard, "notes": r.notes}


def write_status_changes(rows: list[dict]) -> int:
    with get_session() as s:
        for r in rows:
            s.add(StatusChange(**r))
    return len(rows)


def status_history(instrument: str | None = None, timeframe: str | None = None, strategy: str | None = None,
                   limit: int = 200) -> list[dict]:
    with get_session() as s:
        q = s.query(StatusChange)
        for col, val in (("instrument", instrument), ("timeframe", timeframe), ("strategy", strategy)):
            if val:
                q = q.filter(getattr(StatusChange, col) == val)
        return [{"id": r.id, "created_at": r.created_at, "instrument": r.instrument, "timeframe": r.timeframe,
                 "strategy": r.strategy, "kind": r.kind, "from_state": r.from_state, "to_state": r.to_state,
                 "reason": r.reason, "batch_id": r.batch_id, "research_version_id": r.research_version_id}
                for r in q.order_by(StatusChange.id.desc()).limit(limit).all()]
