"""
Chronological data splits and the holdout ledger.

History is cut into whole sessions, in time order, never shuffled:

    |---- development ----|purge|embargo|-- validation --|purge|embargo|---- holdout ----|

* development: where a strategy may be studied, tuned and rejected;
* validation: out-of-sample for the evidence classifier, and still allowed to be looked at repeatedly;
* holdout: one final look. Every evaluation of it is appended to `holdout_ledger`, so a holdout that has been used to
  choose between candidates is visibly no longer untouched.

Purge removes the last sessions of a segment (a trade opened there could resolve in the next one); embargo removes the
first sessions of the next (indicator and regime state that carries across the boundary). Features are causal, so they
are computed once on the full history and sliced; a segment never sees a later bar than the one it ends on.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.config.settings import SETTINGS, SplitPolicy
from quant_intelligence.data.quality_gates import dataset_fingerprint

SEGMENTS = ("development", "validation", "holdout")


@dataclass
class SessionSplit:
    development: list[dt.date]
    validation: list[dt.date]
    holdout: list[dt.date]
    purged: list[dt.date] = field(default_factory=list)      # dropped from the end of a segment
    embargoed: list[dt.date] = field(default_factory=list)   # dropped from the start of a segment
    policy: dict = field(default_factory=dict)

    def sessions(self, name: str) -> list[dt.date]:
        return getattr(self, name)

    def window(self, *names: str) -> tuple[dt.date, dt.date]:
        """First and last session spanned by the named segments (a continuous window, gaps included)."""
        days = [d for n in names for d in self.sessions(n)]
        return min(days), max(days)

    def as_dict(self) -> dict:
        span = lambda days: [str(days[0]), str(days[-1]), len(days)] if days else None  # noqa: E731
        return {"development": span(self.development), "validation": span(self.validation), "holdout": span(self.holdout),
                "n_purged": len(self.purged), "n_embargoed": len(self.embargoed), "policy": self.policy}


def split_sessions(ohlcv: pd.DataFrame, policy: SplitPolicy | None = None) -> SessionSplit:
    """Cut the sessions in `ohlcv` into development / validation / holdout by `policy` fractions."""
    p = policy or SETTINGS.splits
    if p.dev_fraction <= 0 or p.validation_fraction <= 0 or p.dev_fraction + p.validation_fraction >= 1.0:
        raise ValueError("dev_fraction and validation_fraction must be positive and leave room for a holdout")
    days = sorted(pd.to_datetime(ohlcv["timestamp"]).dt.date.unique())
    n = len(days)
    n_dev, n_val = int(n * p.dev_fraction), int(n * p.validation_fraction)
    if n_dev < 1 or n_val < 1 or n - n_dev - n_val < 1:
        raise ValueError(f"{n} sessions are too few to split into three segments")
    dev, val, hold = days[:n_dev], days[n_dev:n_dev + n_val], days[n_dev + n_val:]

    purged: list[dt.date] = []
    embargoed: list[dt.date] = []
    k, e = p.purge_sessions, p.embargo_sessions
    if k and len(dev) > k:
        purged += dev[-k:]
        dev = dev[:-k]
    if e and len(val) > e:
        embargoed += val[:e]
        val = val[e:]
    if k and len(val) > k:
        purged += val[-k:]
        val = val[:-k]
    if e and len(hold) > e:
        embargoed += hold[:e]
        hold = hold[e:]
    return SessionSplit(dev, val, hold, purged, embargoed,
                        {"dev_fraction": p.dev_fraction, "validation_fraction": p.validation_fraction,
                         "purge_sessions": k, "embargo_sessions": e})


def slice_sessions(ohlcv: pd.DataFrame, features: pd.DataFrame, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The bars of `ohlcv` (and the aligned `features`) whose session lies in [start, end], index reset."""
    dates = pd.to_datetime(ohlcv["timestamp"]).dt.date
    mask = ((dates >= start) & (dates <= end)).to_numpy()
    return ohlcv[mask].reset_index(drop=True), features[mask].reset_index(drop=True)


def slice_segment(ohlcv, features, split: SessionSplit, *names: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = split.window(*names)
    return slice_sessions(ohlcv, features, start, end)


# ---- holdout ledger -----------------------------------------------------------------------------------------------

@dataclass
class HoldoutStatus:
    registered: bool
    evaluations: int
    distinct_strategies: int
    repeat_looks: int           # evaluations of a (strategy, version, config) that had already been evaluated
    first_evaluated_at: dt.datetime | None
    consumed: bool              # looked at too often to still count as an untouched test
    reason: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def holdout_fingerprint(ohlcv: pd.DataFrame, features: pd.DataFrame, split: SessionSplit, source: str = "",
                        timeframe: str = "", instrument: str = "") -> str:
    hold_ohlcv, _ = slice_segment(ohlcv, features, split, "holdout")
    return dataset_fingerprint(hold_ohlcv, source=source, timeframe=timeframe, instrument=instrument)


def _key(instrument: str, timeframe: str, fingerprint: str):
    from quant_intelligence.database.models import HoldoutLedger as L

    return (L.instrument == instrument, L.timeframe == timeframe, L.holdout_fingerprint == fingerprint)


def register_holdout(instrument: str, timeframe: str, split: SessionSplit, fingerprint: str) -> None:
    """Record that this holdout exists (once per fingerprint). New data means a new fingerprint and a fresh holdout."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import HoldoutLedger as L

    with get_session() as s:
        if s.query(L.id).filter(*_key(instrument, timeframe, fingerprint), L.event == "REGISTERED").first():
            return
        s.add(L(event="REGISTERED", instrument=instrument, timeframe=timeframe, holdout_start=split.holdout[0],
                holdout_end=split.holdout[-1], holdout_fingerprint=fingerprint))


def record_holdout_evaluation(instrument: str, timeframe: str, fingerprint: str, strategy: str, strategy_version: str,
                              config_hash: str, n_trades: int, net_expectancy_r: float | None,
                              split: SessionSplit | None = None) -> None:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import HoldoutLedger as L

    with get_session() as s:
        s.add(L(event="EVALUATED", instrument=instrument, timeframe=timeframe, holdout_fingerprint=fingerprint,
                holdout_start=split.holdout[0] if split else None, holdout_end=split.holdout[-1] if split else None,
                strategy=strategy, strategy_version=strategy_version, config_hash=config_hash, n_trades=n_trades,
                net_expectancy_r=None if net_expectancy_r is None or net_expectancy_r != net_expectancy_r else float(net_expectancy_r)))


def holdout_status(instrument: str, timeframe: str, fingerprint: str, policy: SplitPolicy | None = None) -> HoldoutStatus:
    """How used this holdout is. 'Consumed' means it has been looked at by more distinct strategies than the budget
    allows (choosing the best of many on a holdout turns it into a second validation set), or the same strategy
    configuration was evaluated on it more than once."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import HoldoutLedger as L

    p = policy or SETTINGS.splits
    with get_session() as s:
        registered = s.query(L.id).filter(*_key(instrument, timeframe, fingerprint), L.event == "REGISTERED").first() is not None
        rows = (s.query(L.strategy, L.strategy_version, L.config_hash, L.created_at)
                .filter(*_key(instrument, timeframe, fingerprint), L.event == "EVALUATED").order_by(L.created_at).all())
    seen: set[tuple] = set()
    repeats = 0
    for strategy, version, cfg, _ in rows:
        k = (strategy, version, cfg)
        repeats += k in seen
        seen.add(k)
    distinct = len({r[0] for r in rows})
    first = rows[0][3] if rows else None
    if repeats:
        consumed, reason = True, f"{repeats} configuration(s) were evaluated on this holdout more than once"
    elif distinct > p.holdout_look_budget:
        consumed, reason = True, f"{distinct} different strategies have been tried on it (budget {p.holdout_look_budget})"
    elif rows:
        consumed, reason = False, f"{len(rows)} evaluation(s) so far, within budget"
    else:
        consumed, reason = False, "never evaluated"
    return HoldoutStatus(registered, len(rows), distinct, repeats, first, consumed, reason)
