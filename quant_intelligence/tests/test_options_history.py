"""Dhan rolling-option history: request shape, parsing, resumable storage."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from quant_intelligence.data_adapters import dhan_options_history as oh
from quant_intelligence.data_adapters.dhan_options_history import (
    OptionsHistory,
    RollingResponseError,
    SeriesKey,
    backfill_options,
    fetch_series,
    parse_rolling_response,
    request_body,
    windows,
)
from quant_intelligence.tests.options_helpers import FakeRollingApi, day, make_history, option_frame, response_body, session_bars

KEY = SeriesKey("NIFTY", "WEEK", 1, "CALL", 0)


@pytest.fixture(scope="module")
def bars(synthetic_ohlcv):
    return session_bars(synthetic_ohlcv)


@pytest.fixture(scope="module")
def history(bars):
    return make_history(bars)


# ---- request shape ------------------------------------------------------------------------------------------------

def test_windows_are_contiguous_and_at_most_30_days():
    ws = windows(day("2024-01-01"), day("2024-04-15"), days=oh.MAX_DAYS_PER_REQUEST)
    assert ws[0][0] == day("2024-01-01") and ws[-1][1] == day("2024-04-15")
    assert all(a[1] == b[0] for a, b in zip(ws, ws[1:]))
    assert all((hi - lo).days <= 30 for lo, hi in ws) and len(ws) == 4
    assert windows(day("2024-01-01"), day("2024-01-01")) == []


def test_default_windows_are_small_because_the_endpoint_is_slow():
    ws = windows(day("2024-01-01"), day("2024-02-01"))
    assert all((hi - lo).days <= oh.DEFAULT_WINDOW_DAYS for lo, hi in ws) and oh.DEFAULT_WINDOW_DAYS < oh.MAX_DAYS_PER_REQUEST


def test_request_body_follows_the_documented_fields():
    b = request_body(13, "NSE_FNO", "OPTIDX", SeriesKey("NIFTY", "WEEK", 1, "PUT", -2), "5", day("2024-05-01"), day("2024-05-31"))
    assert b == {"exchangeSegment": "NSE_FNO", "interval": "5", "securityId": 13, "instrument": "OPTIDX", "expiryCode": 1,
                 "expiryFlag": "WEEK", "strike": "ATM-2", "drvOptionType": "PUT", "requiredData": oh.REQUIRED_DATA,
                 "fromDate": "2024-05-01", "toDate": "2024-05-31"}
    assert request_body(13, "NSE_FNO", "OPTIDX", SeriesKey("NIFTY", "WEEK", 1, "CALL", 0), "5", day("2024-05-01"), day("2024-05-02"))["strike"] == "ATM"
    assert request_body(13, "NSE_FNO", "OPTIDX", SeriesKey("NIFTY", "WEEK", 1, "CALL", 3), "5", day("2024-05-01"), day("2024-05-02"))["strike"] == "ATM+3"


def test_series_files_are_named_by_flag_code_type_and_offset():
    assert SeriesKey("NIFTY", "WEEK", 1, "CALL", -1).filename == "WEEK1_CALL_-1.parquet"
    assert SeriesKey("NIFTY", "MONTH", 2, "PUT", 0).filename == "MONTH2_PUT_+0.parquet"


# ---- parsing ------------------------------------------------------------------------------------------------------

def test_parse_returns_ist_timestamps_and_all_columns(bars):
    f = option_frame(bars.iloc[:50], "CALL", 0)
    out = parse_rolling_response(response_body(f, "CALL"), "CALL")
    assert list(out.columns) == oh.COLUMNS and len(out) == 50
    assert (out["timestamp"].to_numpy() == f["timestamp"].to_numpy()).all()       # UTC epoch shifted back to naive IST
    assert out["close"].tolist() == pytest.approx(f["close"].tolist())


def test_the_other_side_of_a_one_sided_request_is_empty(bars):
    body = response_body(option_frame(bars.iloc[:10], "CALL", 0), "CALL")
    assert len(parse_rolling_response(body, "PUT")) == 0


def test_empty_and_missing_data_are_empty_frames_not_errors():
    assert len(parse_rolling_response({"data": {"ce": None, "pe": None}}, "CALL")) == 0
    assert len(parse_rolling_response({"data": {"ce": {"timestamp": []}}}, "CALL")) == 0


def test_missing_optional_arrays_become_nan(bars):
    body = response_body(option_frame(bars.iloc[:10], "CALL", 0), "CALL")
    del body["data"]["ce"]["iv"]
    assert parse_rolling_response(body, "CALL")["iv"].isna().all()


def test_unrecognised_shapes_are_refused_not_guessed(bars):
    with pytest.raises(RollingResponseError):
        parse_rolling_response("nonsense", "CALL")
    body = response_body(option_frame(bars.iloc[:10], "CALL", 0), "CALL")
    body["data"]["ce"]["close"] = body["data"]["ce"]["close"][:-1]
    with pytest.raises(RollingResponseError):
        parse_rolling_response(body, "CALL")
    del body["data"]["ce"]["timestamp"]
    with pytest.raises(RollingResponseError):
        parse_rolling_response(body, "CALL")


def test_timestamps_outside_the_session_are_refused(bars):
    f = option_frame(bars.iloc[:10], "CALL", 0)
    body = response_body(f, "CALL")
    body["data"]["ce"]["timestamp"] = [t + 12 * 3600 for t in body["data"]["ce"]["timestamp"]]   # a clock that fits no known basis
    with pytest.raises(RollingResponseError):
        parse_rolling_response(body, "CALL")


# ---- fetching and storage -----------------------------------------------------------------------------------------

def span_of(bars):
    d = pd.to_datetime(bars["timestamp"]).dt.date
    return d.min(), d.max() + dt.timedelta(days=1)


def test_fetch_stores_the_series_and_reads_it_back(history, bars, tmp_path):
    lo, hi = span_of(bars)
    api = FakeRollingApi(history)
    res = fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=api)
    assert not res.errors and res.rows_total == len(history.frames[("CALL", 0)]) and res.windows_fetched == len(windows(lo, hi))
    stored = pd.read_parquet(oh.series_path(KEY, tmp_path))
    assert list(stored.columns) == oh.COLUMNS and len(stored) == res.rows_total


def test_fetching_again_makes_no_requests(history, bars, tmp_path):
    lo, hi = span_of(bars)
    fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=FakeRollingApi(history))
    again = FakeRollingApi(history)
    res = fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=again)
    assert again.calls == 0 and res.rows_added == 0


def test_only_windows_with_undone_days_are_requested(history, bars, tmp_path):
    lo, hi = span_of(bars)
    mid_lo, mid_hi = lo + dt.timedelta(days=10), lo + dt.timedelta(days=25)
    fetch_series(None, KEY, 13, mid_lo, mid_hi, root=tmp_path, post=FakeRollingApi(history))
    api = FakeRollingApi(history)
    res = fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=api)
    assert res.earliest.date() == lo and res.latest.date() == (hi - dt.timedelta(days=1))
    for b in api.bodies:                                   # every request still has something to fetch
        wk = oh._weekdays(day(b["fromDate"]), day(b["toDate"]))
        assert any(not (mid_lo <= d < mid_hi) for d in wk)
    assert res.rows_total == len(history.frames[("CALL", 0)])             # nothing duplicated at the seams


def test_windows_are_fetched_newest_first(history, bars, tmp_path):
    lo, hi = span_of(bars)
    api = FakeRollingApi(history)
    fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=api)
    froms = [day(b["fromDate"]) for b in api.bodies]
    assert froms == sorted(froms, reverse=True)


def test_a_hole_left_by_a_stopped_run_is_found_and_filled(history, bars, tmp_path):
    """The failure that cost a retry: oldest part fetched, newest part fetched, a gap between. First/last-date logic cannot see it."""
    lo, hi = span_of(bars)
    third = (hi - lo).days // 3
    a_end, b_start = lo + dt.timedelta(days=third), lo + dt.timedelta(days=2 * third)
    fetch_series(None, KEY, 13, lo, a_end, root=tmp_path, post=FakeRollingApi(history))
    fetch_series(None, KEY, 13, b_start, hi, root=tmp_path, post=FakeRollingApi(history))
    stored = pd.read_parquet(oh.series_path(KEY, tmp_path))
    assert len(stored) < len(history.frames[("CALL", 0)])
    api = FakeRollingApi(history)
    res = fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=api)
    assert res.rows_total == len(history.frames[("CALL", 0)]) and api.calls > 0
    again = FakeRollingApi(history)
    fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=again)
    assert again.calls == 0


def test_pre_listing_days_are_asked_for_once_then_remembered(history, bars, tmp_path):
    lo, hi = span_of(bars)
    early = lo - dt.timedelta(days=60)                      # before the fixture has any data
    api = FakeRollingApi(history)
    fetch_series(None, KEY, 13, early, hi, root=tmp_path, post=api)
    assert api.calls > len(windows(lo, hi))                  # the empty windows were asked for once
    again = FakeRollingApi(history)
    fetch_series(None, KEY, 13, early, hi, root=tmp_path, post=again)
    assert again.calls == 0                                  # and remembered as settled


def test_today_is_never_marked_done_so_a_partial_day_is_refetched(history, bars, tmp_path):
    today = dt.date.today()
    shift = pd.Timestamp(today) - bars["timestamp"].dt.normalize().iloc[-1]
    shifted = make_history(bars.assign(timestamp=bars["timestamp"] + shift))
    lo, hi = today - dt.timedelta(days=3), today + dt.timedelta(days=1)
    fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=FakeRollingApi(shifted))
    again = FakeRollingApi(shifted)
    fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=again)
    assert again.calls >= 1                                  # the window holding today is requested again


def test_a_failed_window_is_reported_and_keeps_what_was_stored(history, bars, tmp_path):
    lo, hi = span_of(bars)
    api = FakeRollingApi(history, fail_after=1)
    res = fetch_series(None, KEY, 13, lo, hi, root=tmp_path, post=api)
    assert res.errors and "token expired" in res.errors[0] and res.windows_fetched == 1
    assert res.rows_total > 0                                              # the first window survived


def test_backfill_stops_at_the_first_error_instead_of_repeating_it(history, bars, tmp_path):
    lo, hi = span_of(bars)
    api = FakeRollingApi(history, fail_after=0)
    out = backfill_options(None, "NIFTY", lo, hi, max_offset=3, root=tmp_path, info={"security_id": 13}, post=api)
    assert api.calls == 1 and out.errors and len(out.series) == 1


def test_backfill_fetches_calls_and_puts_for_every_offset(history, bars, tmp_path):
    lo, hi = span_of(bars)
    out = backfill_options(None, "NIFTY", lo, hi, max_offset=2, root=tmp_path, info={"security_id": 13}, post=FakeRollingApi(history))
    assert not out.errors and len(out.series) == 2 * 5
    loaded = OptionsHistory.load("NIFTY", root=tmp_path)
    assert loaded.offsets == [-2, -1, 0, 1, 2] and len(loaded.frames) == 10


def test_history_lookup_span_and_fingerprint(history, bars):
    ts = bars["timestamp"].iloc[10]
    row = history.row("CALL", 0, ts)
    assert row is not None and row["strike"] % 50 == 0 and history.row("CALL", 0, ts + pd.Timedelta(days=400)) is None
    assert history.row("CALL", 9, ts) is None
    lo, hi = history.span()
    assert lo == bars["timestamp"].min() and hi == bars["timestamp"].max()
    assert history.fingerprint() == history.fingerprint()
    changed = make_history(bars.assign(close=bars["close"] + 1.0))
    assert changed.fingerprint() != history.fingerprint()


def test_loading_a_symbol_with_nothing_stored_is_none(tmp_path):
    assert OptionsHistory.load("NOSUCH", root=tmp_path) is None


# ---- what the live endpoint taught us ---------------------------------------------------------------------------------

def test_stray_prints_outside_the_session_are_dropped(bars):
    f = option_frame(bars.iloc[:80], "CALL", 0)
    extra = f.iloc[-2:].copy()
    extra["timestamp"] = [f["timestamp"].iloc[-1].normalize() + pd.Timedelta(hours=15, minutes=30),
                          f["timestamp"].iloc[-1].normalize() + pd.Timedelta(hours=15, minutes=35)]
    out = parse_rolling_response(response_body(pd.concat([f, extra], ignore_index=True), "CALL"), "CALL")
    assert len(out) == len(f) and out["timestamp"].max() < out["timestamp"].max().normalize() + pd.Timedelta(hours=15, minutes=30)


class _FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.timeouts = list(outcomes), 0, []

    def _post(self, path, body, category="x", retry=True, timeout=15):
        self.calls += 1
        self.timeouts.append(timeout)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def gateway(status):
    from quant_intelligence.brokers.dhan_api_client import DhanApiError

    return DhanApiError(f"Dhan API error ({status}): upstream", status_code=status)


def test_gateway_errors_and_timeouts_are_retried_with_a_long_timeout():
    import requests

    waits = []
    c = _FakeClient([gateway(504), requests.exceptions.ReadTimeout("slow"), {"data": {"ce": None, "pe": None}}])
    assert oh.post_with_retry(c, {}, sleep=waits.append) == {"data": {"ce": None, "pe": None}}
    assert c.calls == 3 and waits == list(oh.RETRY_BACKOFF_SEC[:2]) and set(c.timeouts) == {oh.REQUEST_TIMEOUT_SEC}


def test_it_gives_up_after_the_retry_budget():
    c = _FakeClient([gateway(504)] * (oh.GATEWAY_RETRIES + 1))
    with pytest.raises(Exception, match="504"):
        oh.post_with_retry(c, {}, sleep=lambda s: None)
    assert c.calls == oh.GATEWAY_RETRIES + 1


@pytest.mark.parametrize("status", [401, 429, 400, 814])
def test_auth_rate_limit_and_validation_errors_are_not_retried(status):
    c = _FakeClient([gateway(status), {"data": {}}])
    with pytest.raises(Exception, match=str(status)):
        oh.post_with_retry(c, {}, sleep=lambda s: None)
    assert c.calls == 1


def test_the_api_client_passes_a_per_call_timeout(monkeypatch):
    from types import SimpleNamespace

    from quant_intelligence.brokers import dhan_api_client as mod
    from quant_intelligence.brokers.dhan_api_client import DhanApiClient

    seen = []
    monkeypatch.setattr(mod.requests, "post", lambda url, headers=None, json=None, timeout=None: seen.append(timeout) or
                        SimpleNamespace(ok=True, status_code=200, json=lambda: {"data": {}}, text="", headers={}))
    monkeypatch.setattr(mod, "token_status", lambda tok: {"state": "valid", "expires_at": None})
    c = DhanApiClient(client_id="1", access_token="tok")
    c._post("/x", {}, category="data")
    c._post("/x", {}, category="data", timeout=120)
    assert seen == [15, 120]                      # the default is unchanged for every other caller


def test_the_real_shape_of_a_day_is_accepted(bars):
    """The live response has 77 bars a day: the 75 session bars plus prints at 15:30 and 15:35 (about 2.6% outside the session)."""
    day_bars = bars.iloc[:75]
    f = option_frame(day_bars, "CALL", 0)
    midnight = f["timestamp"].iloc[0].normalize()
    extra = pd.concat([f.iloc[-1:], f.iloc[-1:]], ignore_index=True)
    extra["timestamp"] = [midnight + pd.Timedelta(hours=15, minutes=30), midnight + pd.Timedelta(hours=15, minutes=35)]
    out = parse_rolling_response(response_body(pd.concat([f, extra], ignore_index=True), "CALL"), "CALL")
    assert len(out) == 75 and out["timestamp"].iloc[0] == f["timestamp"].iloc[0]


def test_a_gateway_error_page_becomes_one_short_line():
    from quant_intelligence.brokers.dhan_api_client import _plain

    page = "<HTML><BODY><H1>ERROR</H1>\n<P>We can't connect to the server.</P><PRE>Request ID: abc</PRE></BODY></HTML>" + "<BR>" * 100
    out = _plain(page)
    assert "<" not in out and "\n" not in out and out.startswith("ERROR We can't connect") and len(out) < 200
    assert _plain('{"remarks": "bad request"}') == '{"remarks": "bad request"}'
