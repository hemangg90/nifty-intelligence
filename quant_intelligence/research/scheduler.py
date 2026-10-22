"""
Research scheduler: keeps the evidence behind the ranker current without anyone pressing a button.

One batch per (instrument, timeframe):

    load cached history -> quality gate -> which strategies are stale? -> run the validation suite on those
        -> accept or hold each verdict (research/acceptance.py) -> append status changes and a new research version
        -> write the run report

* Stale means: never run at this strategy version, run under different thresholds, or the data has grown by at least
  `rerun_after_new_sessions` sessions since the last completed run. A fingerprint change alone (one new bar) is not enough.
* Backtests always use next-open fills (a context override, so a live scan on another thread is unaffected).
* The holdout is never touched here; it is looked at only deliberately (scripts/run_research.py --holdout).
* Data problems are an outcome, not an error: missing or failed-quality history is logged as AWAITING_DATA and nothing is run.
* While the market is open the batch is deferred (RESEARCH_MARKET_CLOSED_ONLY) so scans keep the CPU.
* A failure in one strategy is recorded and the batch carries on with the next.

The thread follows the DataKeeper pattern (daemon, wake/stop events, status()). It starts only when RESEARCH_SCHEDULER is on.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from quant_intelligence.backtesting.timing import execution_timing
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.quality_gates import check_history, dataset_fingerprint
from quant_intelligence.reports.research_report import build_run_report, persist_run_report
from quant_intelligence.research import records
from quant_intelligence.research.acceptance import AcceptanceDecision, decide, new_sessions_between
from quant_intelligence.strategies.registry import STRATEGY_CLASSES
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeframe import parse_timeframe_minutes
from quant_intelligence.utils.timeutil import now_ist
from quant_intelligence.validation.evidence_store import load_latest_evidence, policy_hash, save_evidence, snapshot_from_validation


class DataUnavailable(RuntimeError):
    pass


@dataclass
class RunOutcome:
    strategy: str
    strategy_version: str | None
    outcome: str                       # COMPLETED | FAILED | DEFERRED
    state: str | None = None
    decision: AcceptanceDecision | None = None
    detail: str = ""
    runtime_sec: float = 0.0
    summary: dict | None = None
    reasons: list[str] = field(default_factory=list)

    def as_report_dict(self) -> dict:
        s = self.summary or {}
        return {"strategy": self.strategy, "outcome": self.outcome, "detail": self.detail, "state": self.state,
                "decision": None if self.decision is None else self.decision.as_dict(), "reasons": self.reasons,
                "metrics": None if not s else {"n_trades": s.get("n_trades"), "n_trade_days": s.get("n_trade_days"),
                                               "net_expectancy_r": s.get("net_expectancy_r"), "profit_factor": s.get("profit_factor"),
                                               "max_drawdown_r": s.get("max_drawdown_r")},
                "effective_n": s.get("effective_n"), "ci_low": s.get("ci_low"), "ci_high": s.get("ci_high"),
                "oos": s.get("oos"), "walk_forward": s.get("walk_forward"), "robustness": s.get("robustness"),
                "proxy_data": s.get("proxy_data")}


@dataclass
class BatchResult:
    batch_id: str
    instrument: str
    timeframe: str
    status: str                        # RAN | UP_TO_DATE | AWAITING_DATA | DEFERRED
    detail: str = ""
    outcomes: list[RunOutcome] = field(default_factory=list)
    version_no: int | None = None
    report_id: int | None = None

    def as_dict(self) -> dict:
        return {"batch_id": self.batch_id, "instrument": self.instrument, "timeframe": self.timeframe, "status": self.status,
                "detail": self.detail, "version_no": self.version_no, "report_id": self.report_id,
                "outcomes": [{"strategy": o.strategy, "outcome": o.outcome, "state": o.state,
                              "decision": None if o.decision is None else o.decision.as_dict(), "detail": o.detail} for o in self.outcomes]}


def market_is_open(instrument: str, now: dt.datetime) -> bool:
    p = profile_for(instrument)
    return now.weekday() < 5 and p.open_min <= now.hour * 60 + now.minute < p.close_min


def load_history(instrument: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached candles (kept current by the data keeper), cleaned and with features. Raises DataUnavailable."""
    from quant_intelligence.data.data_manager import CACHE_DIR, _read_cache
    from quant_intelligence.data.quality import clean_ohlcv
    from quant_intelligence.features.feature_engine import compute_features

    path = CACHE_DIR / f"{instrument}_{timeframe}.parquet"
    raw = _read_cache(path) if path.exists() else None
    if raw is None or len(raw) == 0:
        raise DataUnavailable(f"no cached history for {instrument} {timeframe}; run scripts/backfill_history.py")
    profile = profile_for(instrument)
    ohlcv = clean_ohlcv(raw, parse_timeframe_minutes(timeframe), profile)
    return ohlcv, compute_features(ohlcv, profile)


def _chain_history(instrument: str):
    try:
        from quant_intelligence.data_adapters.dhan_options_history import OptionsHistory
        from quant_intelligence.options.historical_chain import HistoricalChain

        h = OptionsHistory.load(instrument)
        return None if h is None else HistoricalChain(h)
    except Exception:
        return None


class ResearchScheduler:
    def __init__(self, targets: list[tuple[str, str]] | None = None, interval_sec: int | None = None,
                 loader: Callable | None = None, suite: Callable | None = None, strategies: dict | None = None,
                 now_fn: Callable[[], dt.datetime] | None = None) -> None:
        from quant_intelligence.validation.suite import run_validation_suite

        self.policy = SETTINGS.research
        self.interval = max(60, int(interval_sec or self.policy.interval_sec))
        self._targets = targets
        self._loader = loader or load_history
        self._suite = suite or run_validation_suite
        self._strategies = strategies if strategies is not None else STRATEGY_CLASSES
        self._now = now_fn or (lambda: now_ist().replace(tzinfo=None))
        self._thread: threading.Thread | None = None
        self._wake, self._stop, self._lock = threading.Event(), threading.Event(), threading.Lock()
        self.rounds = 0
        self.last_round_at: dt.datetime | None = None
        self.last_error: str | None = None
        self.last_results: list[BatchResult] = []
        self._batch_thread: threading.Thread | None = None
        self.current: dict | None = None  # the manual batch in flight, for the control center

    # ---- lifecycle (same shape as DataKeeper) ----------------------------------------------------------
    def targets(self) -> list[tuple[str, str]]:
        return self._targets or [(sym, SETTINGS.default_timeframe) for sym in SETTINGS.option_underlyings]

    def start(self, force: bool = False) -> bool:
        if not (SETTINGS.research_scheduler or force):
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="research-scheduler", daemon=True)
            self._thread.start()
        log_event("research", f"Research scheduler started (every {self.interval}s)", level="INFO")
        return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.last_results = self.run_once()
                self.last_error = None
            except Exception as e:  # a bad round must never kill the thread
                self.last_error = f"{type(e).__name__}: {e}"
                log_event("research", f"Research round failed: {e}", level="ERROR")
            self._wake.wait(self.interval)
            self._wake.clear()

    def run_in_background(self, targets: list[tuple[str, str]] | None = None, force: bool = False, only: list[str] | None = None) -> bool:
        """Run one batch now on its own thread (the dashboard's button). False if one is already running. The result lands
        in `last_results` and, as always, in the run log; a failure is recorded in `last_error`."""
        with self._lock:
            if self._batch_thread is not None and self._batch_thread.is_alive():
                return False
            self.current = {"targets": targets or self.targets(), "force": force, "started_at": self._now()}

            def work() -> None:
                try:
                    self.last_results = self.run_once(targets, force=force, only=only)
                    self.last_error = None
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"
                    log_event("research", f"Manual research batch failed: {e}", level="ERROR")
                finally:
                    self.current = None

            self._batch_thread = threading.Thread(target=work, name="research-batch", daemon=True)
            self._batch_thread.start()
        return True

    @property
    def batch_running(self) -> bool:
        return bool(self._batch_thread and self._batch_thread.is_alive())

    def status(self) -> dict:
        return {"running": self.running, "batch_running": self.batch_running, "current": self.current, "rounds": self.rounds, "last_round_at": self.last_round_at, "error": self.last_error,
                "targets": self.targets(), "last": [r.as_dict() for r in self.last_results]}

    # ---- one round --------------------------------------------------------------------------------------
    def run_once(self, targets: list[tuple[str, str]] | None = None, force: bool = False,
                 only: list[str] | None = None) -> list[BatchResult]:
        out = []
        for instrument, timeframe in targets or self.targets():
            if self._stop.is_set():
                break
            out.append(self.run_target(instrument, timeframe, force=force, only=only))
        self.rounds += 1
        self.last_round_at = self._now()
        return out

    def run_target(self, instrument: str, timeframe: str, force: bool = False, only: list[str] | None = None) -> BatchResult:
        batch_id = "RS-" + uuid.uuid4().hex[:10]
        res = BatchResult(batch_id, instrument, timeframe, "RAN")
        now = self._now()
        if self.policy.market_closed_only and not force and market_is_open(instrument, now):
            res.status, res.detail = "DEFERRED", "market is open; research waits so scans keep the CPU"
            return res

        try:
            ohlcv, features = self._loader(instrument, timeframe)
        except DataUnavailable as e:
            return self._awaiting(res, str(e))
        except Exception as e:
            return self._awaiting(res, f"could not load history: {e}")
        quality = check_history(ohlcv, parse_timeframe_minutes(timeframe), profile_for(instrument), instrument=instrument, timeframe=timeframe)
        if quality.status == "FAIL":
            return self._awaiting(res, "data failed its quality gate: " + "; ".join(quality.issues[:3]))

        ts = pd.to_datetime(ohlcv["timestamp"])
        data_end = ts.max().date()
        fingerprint = dataset_fingerprint(ohlcv, source="cache", timeframe=timeframe, instrument=instrument)
        phash = policy_hash()
        data = {"fingerprint": fingerprint, "start": str(ts.min().date()), "end": str(data_end), "n_sessions": int(ts.dt.date.nunique()),
                "quality": quality.status, "issues": list(quality.issues)}

        names = only or list(self._strategies)
        stale = [n for n in names if force or self._is_stale(instrument, timeframe, n, phash, data_end)]
        if not stale:
            res.status, res.detail = "UP_TO_DATE", "no strategy has enough new data or a changed definition to re-run"
            return res

        from quant_intelligence.research.pipeline import default_quantity

        quantity = default_quantity(instrument)
        chain = _chain_history(instrument)
        accepted_before = load_latest_evidence(instrument, timeframe)
        started = time.monotonic()
        changes: list[dict] = []
        any_accepted = False
        for name in stale:
            cls = self._strategies[name]
            if time.monotonic() - started > self.policy.max_batch_runtime_sec:
                res.outcomes.append(RunOutcome(name, None, "DEFERRED", detail="batch time budget reached; runs next round"))
                self._log(res, instrument, timeframe, res.outcomes[-1], fingerprint, data_end, phash)
                continue
            t0 = time.monotonic()
            try:
                with execution_timing("next_open"):
                    rv = self._suite(cls, ohlcv, features, instrument, quantity=quantity, timeframe=timeframe, data_source="cache",
                                     chain_history=chain, robustness="auto", include_holdout=False, persist=False)
                summary = snapshot_from_validation(rv)
                decision = decide(accepted_before.get(name), rv.state, summary, rv.strategy_version, data_end, phash, self.policy)
                outcome = RunOutcome(name, rv.strategy_version, "COMPLETED", rv.state.value, decision, "", time.monotonic() - t0,
                                     summary, list(rv.evidence.reasons[:6]))
                if decision.accepted:
                    save_evidence(rv, fingerprint)
                    any_accepted = True
                    if decision.writes_status_change:
                        prev = accepted_before.get(name)
                        changes.append({"instrument": instrument, "timeframe": timeframe, "strategy": name, "kind": decision.kind,
                                        "from_state": prev.state.value if prev else None, "to_state": rv.state.value,
                                        "reason": decision.reason, "batch_id": batch_id})
            except Exception as e:  # one strategy failing must not stop the batch
                log_event("research", f"{instrument} {name}: validation failed: {e}", level="ERROR")
                outcome = RunOutcome(name, None, "FAILED", detail=f"{type(e).__name__}: {e}", runtime_sec=time.monotonic() - t0)
            res.outcomes.append(outcome)
            self._log(res, instrument, timeframe, outcome, fingerprint, data_end, phash)

        version_row = None
        if any_accepted:
            board = self._leaderboard(instrument, timeframe)
            vid, no = records.write_version(instrument, timeframe, batch_id, fingerprint, data_end, phash, board,
                                            notes=f"{sum(1 for c in changes)} status change(s)")
            for c in changes:
                c["research_version_id"] = vid
            res.version_no, version_row = no, {"version_no": no, "leaderboard": board}
        if changes:
            records.write_status_changes(changes)

        report = build_run_report(instrument, timeframe, batch_id, data, [o.as_report_dict() for o in res.outcomes], version_row, changes)
        res.report_id = persist_run_report(instrument, report)
        log_event("research", f"{instrument} {timeframe}: {len(res.outcomes)} strategies run, {len(changes)} status change(s)", level="INFO")
        return res

    # ---- helpers ---------------------------------------------------------------------------------------
    def _awaiting(self, res: BatchResult, why: str) -> BatchResult:
        res.status, res.detail = "AWAITING_DATA", why
        records.log_run(batch_id=res.batch_id, instrument=res.instrument, timeframe=res.timeframe, strategy="*", outcome="AWAITING_DATA",
                        detail=why, finished_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
        return res

    def _is_stale(self, instrument: str, timeframe: str, name: str, phash: str, data_end: dt.date) -> bool:
        version = STRATEGY_CLASSES[name]().version if name in STRATEGY_CLASSES else "0"
        last = records.last_completed_run(instrument, timeframe, name, version)
        if last is None or last["policy_hash"] != phash:
            return True
        fresh = new_sessions_between(last["data_end"], data_end)
        return fresh is None or fresh >= self.policy.rerun_after_new_sessions

    def _log(self, res: BatchResult, instrument: str, timeframe: str, o: RunOutcome, fingerprint: str, data_end: dt.date, phash: str) -> None:
        records.log_run(batch_id=res.batch_id, instrument=instrument, timeframe=timeframe, strategy=o.strategy,
                        strategy_version=o.strategy_version, outcome=o.outcome, state=o.state,
                        decision=None if o.decision is None else o.decision.kind,
                        accepted=None if o.decision is None else o.decision.accepted,
                        detail=(o.decision.reason if o.decision else o.detail), runtime_sec=o.runtime_sec,
                        finished_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                        dataset_fingerprint=fingerprint, data_end=data_end, policy_hash=phash)

    @staticmethod
    def _leaderboard(instrument: str, timeframe: str) -> list[dict]:
        out = []
        for name, snap in sorted(load_latest_evidence(instrument, timeframe).items()):
            s = snap.summary
            out.append({"strategy": name, "strategy_version": snap.strategy_version, "state": snap.state.value,
                        "accepted_at": str(snap.created_at), "net_expectancy_r": s.get("net_expectancy_r"), "ci_low": s.get("ci_low"),
                        "n_trades": s.get("n_trades"), "oos": s.get("oos"), "walk_forward": s.get("walk_forward"),
                        "robustness": s.get("robustness"), "proxy_data": s.get("proxy_data")})
        return out


RESEARCH_SCHEDULER = ResearchScheduler()
