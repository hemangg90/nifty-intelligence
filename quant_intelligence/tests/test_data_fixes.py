"""Data-fetching fixes: keeper pauses on a dead token, atomic/concurrent-safe candle cache, single-flight
downloads, instrument master recovery + daily roll, live broker credentials, no persisted scan backtests."""
import dataclasses
import datetime as dt
import importlib.util
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.data import data_keeper as keeper_module
from quant_intelligence.data import data_manager as dm_module
from quant_intelligence.data.data_keeper import DataKeeper
from quant_intelligence.data_adapters import dhan_instrument_master as master

from quant_intelligence.tests.test_data_keeper import FakeManager
from quant_intelligence.tests.test_dhan_auth import make_token


# ---------------------------------------------------------------- 1. keeper pauses on a dead token
def _with_token(monkeypatch, token):
    monkeypatch.setattr(keeper_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, dhan_access_token=token,
                                                                       dhan_client_id="1000000001"))


def test_keeper_sends_nothing_and_logs_once_while_the_token_is_expired(monkeypatch):
    _with_token(monkeypatch, make_token(-5))
    events = []
    monkeypatch.setattr(keeper_module, "log_event", lambda component, message, **kw: events.append(message))
    mgr = FakeManager()
    k = DataKeeper(interval_seconds=15)
    k._manager = mgr
    for _ in range(3):
        out = k.run_once(["NIFTY", "TCS"])
    assert mgr.calls == [] and out["paused"] and "expired" in out["error"]
    assert k.status()["paused"] is True and len(events) == 1  # ONE log line, not 3 x instruments


def test_keeper_resumes_by_itself_when_a_valid_token_appears(monkeypatch):
    _with_token(monkeypatch, make_token(-5))
    events = []
    monkeypatch.setattr(keeper_module, "log_event", lambda component, message, **kw: events.append(message))
    mgr = FakeManager()
    k = DataKeeper(interval_seconds=15)
    k._manager = mgr
    k.run_once(["NIFTY"])
    assert k.paused
    _with_token(monkeypatch, make_token(20))
    out = k.run_once(["NIFTY", "TCS"])
    assert not k.paused and out.get("paused") is None and [c[0] for c in mgr.calls] == ["NIFTY", "TCS"]
    assert any("resuming" in e for e in events) and k.last_error is None


def test_a_missing_token_does_not_pause_the_keeper(monkeypatch):
    """CSV files can still supply candles, so only an expired/rejected token pauses it."""
    _with_token(monkeypatch, "")
    mgr = FakeManager()
    k = DataKeeper(interval_seconds=15)
    k._manager = mgr
    k.run_once(["NIFTY"])
    assert mgr.calls and not k.paused


def test_a_rejected_token_pauses_the_keeper(monkeypatch):
    from quant_intelligence.brokers.dhan_rate_limit import LIMITER

    token = make_token(20)
    _with_token(monkeypatch, token)
    LIMITER.record_auth_failure(token)
    k = DataKeeper(interval_seconds=15)
    k._manager = FakeManager()
    assert k.run_once(["NIFTY"])["paused"] and "rejected" in k.last_error


# ---------------------------------------------------------------- 2. atomic cache + corrupt read
def _frame(n=5):
    ts = pd.date_range("2026-10-01 09:15", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0})


def test_cache_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "X_5min.parquet"
    dm_module._write_cache(_frame(3), path)
    dm_module._write_cache(_frame(6), path)
    assert len(pd.read_parquet(path)) == 6 and [p.name for p in tmp_path.iterdir()] == ["X_5min.parquet"]


def test_a_reader_never_sees_a_torn_file_while_a_writer_replaces_it(tmp_path):
    path = tmp_path / "Y_5min.parquet"
    dm_module._write_cache(_frame(50), path)
    stop, errors = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            dm_module._write_cache(_frame(50 + i % 20), path)
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        deadline = time.time() + 1.5
        while time.time() < deadline:
            try:
                pd.read_parquet(path)
            except Exception as e:  # a torn read would raise an Arrow error here
                errors.append(e)
    finally:
        stop.set()
        t.join()
    assert errors == []


def test_an_unreadable_cache_file_is_refetched_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    (tmp_path / "BROKEN_5min.parquet").write_bytes(b"not a parquet file")
    assert dm_module._read_cache(tmp_path / "BROKEN_5min.parquet") is None
    manager = dm_module.DataManager()
    end = dt.datetime(2026, 10, 1, 15, 31)
    stamps = pd.date_range("2026-10-01 09:15", "2026-10-01 15:25", freq="5min")
    manager._fetch = lambda *a, **k: ("dhan", _frame(0).reindex(range(len(stamps))).assign(
        timestamp=stamps, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0))
    df, meta = manager.get_ohlcv("BROKEN", "5min", end - dt.timedelta(days=3), end)
    assert len(df) > 0 and meta["source"] == "dhan"
    assert len(pd.read_parquet(tmp_path / "BROKEN_5min.parquet")) > 0  # the bad file was replaced


def test_a_rebuild_of_a_stale_cache_keeps_the_older_history(tmp_path, monkeypatch):
    """Cache older than MAX_TAIL_GAP_DAYS is refetched for the requested window only; the long backfilled
    history before that window must survive, not be replaced by it."""
    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    old = pd.date_range("2024-11-01 09:15", "2024-11-01 15:25", freq="5min")
    pd.DataFrame({"timestamp": old, "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.5, "volume": 5.0}).to_parquet(
        tmp_path / "KEEP_5min.parquet", index=False)
    new = pd.date_range("2026-10-01 09:15", "2026-10-01 15:25", freq="5min")
    manager = dm_module.DataManager()
    manager._fetch = lambda *a, **k: ("dhan", pd.DataFrame({
        "timestamp": new, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}))
    end = dt.datetime(2026, 10, 1, 15, 31)
    df, _ = manager.get_ohlcv("KEEP", "5min", end - dt.timedelta(days=3), end)
    assert df["timestamp"].min() >= pd.Timestamp("2026-09-28")          # the caller still gets only its window
    on_disk = pd.read_parquet(tmp_path / "KEEP_5min.parquet")
    assert on_disk["timestamp"].min() == old[0] and on_disk["timestamp"].max() == new[-1]
    assert len(on_disk) == len(old) + len(new) and not on_disk["timestamp"].duplicated().any()


# ---------------------------------------------------------------- 3. single flight
def test_two_callers_for_one_uncached_instrument_download_it_once(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    stamps = pd.date_range("2026-10-01 09:15", "2026-10-01 15:25", freq="5min")
    frame = pd.DataFrame({"timestamp": stamps, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0})
    calls = []
    start_gate = threading.Barrier(2)

    def slow_fetch(self, *a, **k):
        calls.append(1)
        time.sleep(0.4)
        return "dhan", frame

    monkeypatch.setattr(dm_module.DataManager, "_fetch", slow_fetch)
    end = dt.datetime(2026, 10, 1, 15, 31)
    results = []

    def caller():
        start_gate.wait()
        results.append(dm_module.DataManager().get_ohlcv("ONCE", "5min", end - dt.timedelta(days=3), end)[1]["source"])

    threads = [threading.Thread(target=caller) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and sorted(results) == ["cache", "dhan"]  # the second caller reused the fresh cache


# ---------------------------------------------------------------- 4/5. instrument master
HEADER = "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_TRADING_SYMBOL,SEM_LOT_UNITS,SEM_CUSTOM_SYMBOL,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,SM_SYMBOL_NAME"


def _rows(*rows):
    pad = "NSE,E,9,EQUITY,PAD,1," + "X" * 1200 + ",,,PAD"  # a valid row that lifts the file past the size guard
    return HEADER + "\n" + "\n".join(rows) + "\n" + pad


@pytest.fixture
def clean_master(monkeypatch, tmp_path):
    monkeypatch.setattr(master, "_CACHE_FILE", tmp_path / "scrip.csv")
    for name in ("_cache", "_fno_cache", "_mcx_cache"):
        monkeypatch.setattr(master, name, None)
    monkeypatch.setattr(master, "_built_day", {})
    monkeypatch.setattr(master, "_retry_download_at", 0.0)
    return tmp_path / "scrip.csv"


def test_a_failed_scrip_master_download_is_retried_not_remembered_forever(clean_master, monkeypatch):
    attempts = []

    def failing():
        attempts.append(1)
        raise RuntimeError("network down")

    monkeypatch.setattr(master, "_download_scrip_master", failing)
    assert master.resolve_equity("TCS") is None
    assert master.resolve_equity("TCS") is None and len(attempts) == 1  # backoff: no download storm
    assert master._cache is None  # the failure is NOT cached as "nothing listed"

    monkeypatch.setattr(master, "_retry_download_at", 0.0)  # backoff elapsed

    def working():
        clean_master.write_text(_rows("NSE,E,11536,EQUITY,TCS,1,TCS,,,TCS"), encoding="utf-8")

    monkeypatch.setattr(master, "_download_scrip_master", working)
    assert master.resolve_equity("TCS") == {"security_id": "11536", "exchange_segment": "NSE_EQ"}


def test_download_is_written_atomically_and_tiny_responses_are_rejected(clean_master, monkeypatch):
    class Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    monkeypatch.setattr(master.requests, "get", lambda *a, **k: Resp(b"oops"))
    with pytest.raises(RuntimeError, match="too small"):
        master._download_scrip_master()
    assert not clean_master.exists()

    monkeypatch.setattr(master.requests, "get", lambda *a, **k: Resp(_rows("NSE,E,1,EQUITY,AAA,1,AAA,,,AAA").encode()))
    master._download_scrip_master()
    assert clean_master.exists() and not list(clean_master.parent.glob("*.tmp"))


def test_mcx_index_rolls_to_the_next_contract_when_the_day_changes(clean_master, monkeypatch):
    today = master.now_ist().date()
    soon = (today + dt.timedelta(days=1)).isoformat()
    later = (today + dt.timedelta(days=30)).isoformat()
    clean_master.write_text(_rows(
        f"MCX,M,111,FUTCOM,CRUDEOIL-SOON,100,CRUDEOIL,{soon} 23:30:00,,CRUDEOIL",
        f"MCX,M,222,FUTCOM,CRUDEOIL-LATER,100,CRUDEOIL,{later} 23:30:00,,CRUDEOIL",
        f"MCX,M,333,OPTFUT,CRUDEOIL-OPT,100,CRUDEOIL,{later} 23:30:00,8000,CRUDEOIL",
    ), encoding="utf-8")
    monkeypatch.setattr(master, "_CACHE_FILE", clean_master)
    first = master._load_mcx_index()["CRUDEOIL"]["security_ids"]
    assert first == ["111", "222"]  # front month is the one expiring soonest

    class Tomorrow(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return super().now(tz) + dt.timedelta(days=2)

    monkeypatch.setattr(master, "now_ist", lambda: dt.datetime.combine(today + dt.timedelta(days=2), dt.time(10, 0)))
    assert master._load_mcx_index()["CRUDEOIL"]["security_ids"] == ["222"]  # expired contract dropped without a restart


def test_mcx_uses_the_ist_date_not_the_servers_date(clean_master, monkeypatch):
    # A contract expiring "today" in IST must still be listed at 02:00 IST even though the UTC date is yesterday.
    ist_day = dt.date(2026, 10, 6)
    clean_master.write_text(_rows(
        f"MCX,M,111,FUTCOM,CRUDEOIL-X,100,CRUDEOIL,{ist_day.isoformat()} 23:30:00,,CRUDEOIL"), encoding="utf-8")
    monkeypatch.setattr(master, "now_ist", lambda: dt.datetime(2026, 10, 6, 2, 0))
    assert master._load_mcx_index()["CRUDEOIL"]["security_ids"] == ["111"]


# ---------------------------------------------------------------- 6. positions use the IST clock
def test_open_position_age_uses_the_ist_clock(monkeypatch):
    from quant_intelligence.ui import position_views

    monkeypatch.setattr(position_views, "now_ist", lambda: dt.datetime(2026, 10, 6, 11, 0))
    src = Path(position_views.__file__).read_text(encoding="utf-8")
    assert "dt.datetime.now()" not in src


# ---------------------------------------------------------------- 7. live broker follows credentials
def test_live_broker_picks_up_credentials_saved_after_it_was_built(monkeypatch):
    from quant_intelligence.brokers import dhan_broker

    empty = dataclasses.replace(settings_module.SETTINGS, dhan_client_id="", dhan_access_token="")
    monkeypatch.setattr(dhan_broker, "SETTINGS", empty)
    broker = dhan_broker.DhanBroker()
    assert broker.is_connected() is False
    object.__setattr__(empty, "dhan_client_id", "1000000001")
    object.__setattr__(empty, "dhan_access_token", "a.b.c")
    assert broker.is_connected() is True


# ---------------------------------------------------------------- 8. scans do not persist backtests
def test_pipeline_runs_do_not_write_research_backtests(monkeypatch):
    from quant_intelligence.backtesting import engine as engine_module
    from quant_intelligence.research import pipeline

    seen = []
    real = engine_module.run_backtest

    def spy(*a, **kw):
        seen.append(kw.get("persist", True))
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "run_backtest", spy)
    src = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "persist=False" in src.split("run_backtest(strategy, ohlcv, features, instrument")[1].split(")")[0]


def _prune_module():
    spec = importlib.util.spec_from_file_location("prune_research_tables", Path(__file__).resolve().parents[2] / "scripts" / "prune_research_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prune_script_removes_only_old_research_runs(tmp_path, monkeypatch, capsys):
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE backtest_runs (run_id TEXT, split TEXT, created_at TEXT);
        CREATE TABLE backtest_trades (run_id TEXT);
        CREATE TABLE strategy_observations (run_id TEXT);
        INSERT INTO backtest_runs VALUES ('old-r','RESEARCH','2020-01-01 00:00:00'),
                                         ('new-r','RESEARCH','2999-01-01 00:00:00'),
                                         ('lab','IN_SAMPLE','2020-01-01 00:00:00');
        INSERT INTO backtest_trades VALUES ('old-r'),('old-r'),('new-r'),('lab');
        INSERT INTO strategy_observations VALUES ('old-r'),('lab');
    """)
    c.commit()
    c.close()
    mod = _prune_module()

    monkeypatch.setattr(sys, "argv", ["prune", "--db", str(db)])
    assert mod.main() == 0 and "Dry run" in capsys.readouterr().out
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0] == 4

    monkeypatch.setattr(sys, "argv", ["prune", "--db", str(db), "--apply"])
    assert mod.main() == 0
    c = sqlite3.connect(db)
    assert c.execute("SELECT run_id FROM backtest_runs ORDER BY run_id").fetchall() == [("lab",), ("new-r",)]
    assert c.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0] == 2
    assert c.execute("SELECT COUNT(*) FROM strategy_observations").fetchone()[0] == 1
    assert len(list(tmp_path.glob("t.backup-*.db"))) == 1
