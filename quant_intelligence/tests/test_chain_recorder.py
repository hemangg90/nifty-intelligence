"""The live chain recorder: off by default, throttled, never raises, and round-trips what it saves."""
from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from quant_intelligence.options import chain_recorder as cr
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow, parse_option_chain

NOW = dt.datetime(2026, 10, 20, 11, 0)


def snap(underlying="NIFTY", spot=22000.0, partial=False):
    rows = [StrikeRow(strike=22000.0 + 50 * k, ce_ltp=100.0 - k, ce_oi=1000.0 + k, ce_oi_change=10.0, ce_volume=500.0, ce_iv=14.0,
                      ce_delta=0.5, ce_security_id="1", pe_ltp=90.0 + k, pe_oi=2000.0, pe_oi_change=None, pe_volume=400.0,
                      pe_iv=15.0, pe_delta=-0.5, pe_security_id="2") for k in range(-2, 3)]
    return ChainSnapshot(underlying, "2026-10-27", spot, rows, partial=partial)


@pytest.fixture(autouse=True)
def _fresh():
    cr.reset_throttle()
    yield
    cr.reset_throttle()


def enable(monkeypatch, **over):
    monkeypatch.setattr(cr, "SETTINGS", dataclasses.replace(cr.SETTINGS, chain_recorder=True, **over))


def test_nothing_is_written_while_the_flag_is_off(tmp_path):
    assert cr.record_snapshot(snap(), now=NOW, root=tmp_path) is False
    assert not list(tmp_path.rglob("*.parquet"))


def test_a_snapshot_round_trips_every_strike(monkeypatch, tmp_path):
    enable(monkeypatch)
    assert cr.record_snapshot(snap(), now=NOW, root=tmp_path) is True
    back = cr.load_recorded("NIFTY", NOW.date(), tmp_path)
    assert len(back) == 1 and back[0].source == "recorded" and not back[0].partial
    assert [s.strike for s in back[0].strikes] == [21900.0, 21950.0, 22000.0, 22050.0, 22100.0]
    assert back[0].total_pcr == pytest.approx(snap().total_pcr) and back[0].spot_price == 22000.0
    assert back[0].strikes[0].pe_oi_change is None                      # missing stays missing, not NaN
    assert back[0].atm_iv == pytest.approx(snap().atm_iv)


def test_snapshots_are_throttled_per_underlying(monkeypatch, tmp_path):
    enable(monkeypatch, chain_record_interval_sec=3600.0)
    assert cr.record_snapshot(snap("NIFTY"), now=NOW, root=tmp_path)
    assert not cr.record_snapshot(snap("NIFTY"), now=NOW + dt.timedelta(minutes=1), root=tmp_path)
    assert cr.record_snapshot(snap("BANKNIFTY"), now=NOW, root=tmp_path)           # a different underlying has its own clock
    assert len(cr.load_recorded("NIFTY", NOW.date(), tmp_path)) == 1


def test_later_snapshots_append_to_the_same_day_file(monkeypatch, tmp_path):
    enable(monkeypatch)
    cr.record_snapshot(snap(spot=22000.0), now=NOW, root=tmp_path, interval_sec=0)
    cr.record_snapshot(snap(spot=22040.0), now=NOW + dt.timedelta(minutes=5), root=tmp_path, interval_sec=0)
    got = cr.load_recorded("NIFTY", NOW.date(), tmp_path)
    assert [g.spot_price for g in got] == [22000.0, 22040.0]


def test_partial_and_empty_snapshots_are_never_recorded_as_chains(monkeypatch, tmp_path):
    enable(monkeypatch)
    assert not cr.record_snapshot(snap(partial=True), now=NOW, root=tmp_path, force=True)
    assert not cr.record_snapshot(ChainSnapshot("NIFTY", "x", 1.0), now=NOW, root=tmp_path)
    assert not cr.record_snapshot(None, now=NOW, root=tmp_path)


def test_a_write_failure_is_swallowed(monkeypatch, tmp_path):
    enable(monkeypatch)
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a folder is needed")
    assert cr.record_snapshot(snap(), now=NOW, root=blocker) is False


def test_nothing_recorded_means_an_empty_list(tmp_path):
    assert cr.load_recorded("NIFTY", NOW.date(), tmp_path) == []


def test_fetching_a_chain_records_it_when_enabled(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from quant_intelligence.options import option_selector

    enable(monkeypatch, chain_record_dir=str(tmp_path))
    raw = {"data": {"last_price": 22000.0, "oc": {"22000.000000": {"ce": {"last_price": 100, "oi": 10, "previous_oi": 8, "volume": 5,
                                                                         "implied_volatility": 14},
                                                                   "pe": {"last_price": 90, "oi": 20, "previous_oi": 18, "volume": 4,
                                                                          "implied_volatility": 15}}}}}
    client = SimpleNamespace(get_expiry_list=lambda sid, seg: ["2026-10-27"], get_option_chain=lambda sid, seg, exp: raw)
    monkeypatch.setattr(option_selector, "get_underlying_info", lambda u: {"security_id": 13, "seg": "IDX_I"})
    got = option_selector._fetch_chain_uncached(client, "NIFTY", "NEAREST")
    assert got.total_pcr == 2.0
    assert len(list((tmp_path / "NIFTY").glob("*.parquet"))) == 1
    assert parse_option_chain("NIFTY", "2026-10-27", raw).total_pcr == 2.0
