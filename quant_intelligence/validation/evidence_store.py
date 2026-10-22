"""
Storing and reading validation verdicts.

The validation suite is slow (it re-runs backtests); the ranker runs on every scan. So the suite's conclusion is written
to `strategy_evidence` as an append-only snapshot and the ranker reads the latest one per strategy. A snapshot is only
as good as its age and the strategy version it was made for; the ranker checks both (ranking/gated_ranker.py).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from quant_intelligence.validation.evidence import EvidenceState


def policy_hash() -> str:
    """Short fingerprint of every threshold a verdict depends on (evidence, splits, robustness). A verdict made under
    different thresholds is not comparable with one made under these."""
    import hashlib

    from quant_intelligence.config.settings import SETTINGS

    parts = repr((SETTINGS.evidence, SETTINGS.splits, SETTINGS.robustness)).encode()
    return hashlib.sha256(parts).hexdigest()[:12]


@dataclass
class EvidenceSnapshot:
    strategy: str
    strategy_version: str | None
    state: EvidenceState
    created_at: dt.datetime | None
    config_hash: str = ""
    summary: dict = field(default_factory=dict)

    def age_days(self, now: dt.datetime | None = None) -> float | None:
        if self.created_at is None:
            return None
        return ((now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)) - self.created_at).total_seconds() / 86400.0


def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def _breakeven(rows: list[dict]) -> float | None:
    by_k = {r["slippage_multiple"]: r["net_expectancy_r"] for r in rows if "slippage_multiple" in r}
    if 1.0 not in by_k or 2.0 not in by_k:
        return None
    slope = by_k[1.0] - by_k[2.0]
    return None if slope <= 1e-12 else float(1.0 + by_k[1.0] / slope)


def snapshot_from_validation(rv) -> dict:
    """The numbers the ranker needs, from a `suite.ResearchValidation`. Plain JSON types only."""
    m, u, e = rv.in_sample.metrics, rv.in_sample.uncertainty, rv.in_sample.evidence
    oos, wf, rob = rv.out_of_sample, rv.walk_forward, rv.robustness
    out = {
        "state": e.state.value,
        "reasons": list(e.reasons[:6]),
        "n_trades": m.n_trades,
        "n_trade_days": m.n_trade_days,
        "net_expectancy_r": _num(m.net_expectancy_r),
        "ci_low": _num(u.expectancy_ci[0]),
        "ci_high": _num(u.expectancy_ci[1]),
        "effective_n": _num(u.effective_n),
        "profit_factor": _num(m.profit_factor) if m.profit_factor != float("inf") else None,
        "max_drawdown_r": _num(m.max_drawdown_r),
        "oos": None if oos is None else {"n_trades": oos.n_trades, "net_expectancy_r": _num(oos.net_expectancy_r),
                                          "ci_low": _num(None if oos.expectancy_ci_low is None else float(oos.expectancy_ci_low))},
        "walk_forward": None if wf is None else {"share_positive": _num(wf.share_positive), "n_judged": wf.n_judged,
                                                   "worst_fold_r": _num(wf.worst_fold_r), "folds": wf.as_dict()["folds"]},
        "robustness": None, "param_positive_share": None, "slippage_breakeven": None,
        "holdout": None if rv.holdout is None else {"n_trades": rv.holdout["n_trades"],
                                                     "net_expectancy_r": _num(rv.holdout["net_expectancy_r"]),
                                                     "consumed": rv.holdout["status"]["consumed"]},
        "split": rv.split.as_dict(),
        "data_end": str(rv.split.holdout[-1]) if rv.split.holdout else None,
        "policy_hash": policy_hash(),
        "proxy_data": bool(getattr(rv, "proxy_data", False)),
        "detail": getattr(rv, "detail", None) or None,
    }
    if rob is not None:
        out["robustness"] = rob.outcomes()
        ps = rob.tests.get("parameter_sensitivity")
        out["param_positive_share"] = None if ps is None else _num(ps.value)
        cs = rob.tests.get("cost_stress")
        out["slippage_breakeven"] = None if cs is None else _breakeven(cs.rows)
    return out


def save_evidence(rv, dataset_fingerprint: str = "") -> None:
    """Append the verdict for `rv`. Never raises into a research run."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import StrategyEvidence

        with get_session() as s:
            s.add(StrategyEvidence(strategy=rv.strategy, strategy_version=rv.strategy_version, instrument=rv.instrument,
                                   timeframe=rv.timeframe, state=rv.state.value,
                                   config_hash=rv.in_sample.config_hash, dataset_fingerprint=dataset_fingerprint,
                                   summary=snapshot_from_validation(rv)))
    except Exception as e:
        from quant_intelligence.utils.logging_utils import log_event

        log_event("evidence", f"Could not store evidence for {getattr(rv, 'strategy', '?')}: {e}", level="WARNING")


def load_latest_evidence(instrument: str, timeframe: str) -> dict[str, EvidenceSnapshot]:
    """Latest snapshot per strategy for this instrument and timeframe. Empty on any storage problem."""
    try:
        from sqlalchemy import func

        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import StrategyEvidence as T

        with get_session() as s:
            newest = (s.query(T.strategy, func.max(T.id).label("mid")).filter(T.instrument == instrument, T.timeframe == timeframe)
                      .group_by(T.strategy).subquery())
            rows = s.query(T).join(newest, T.id == newest.c.mid).all()
            return {r.strategy: EvidenceSnapshot(r.strategy, r.strategy_version, EvidenceState(r.state), r.created_at,
                                                  r.config_hash or "", dict(r.summary or {})) for r in rows}
    except Exception:
        return {}
