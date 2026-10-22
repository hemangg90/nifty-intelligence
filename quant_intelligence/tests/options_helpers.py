"""Test fixtures that look like Dhan's rolling-option responses. The prices are a deterministic function of a spot path
(intrinsic plus a time value that decays away from the money) - a stand-in for the endpoint, never evidence of anything."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant_intelligence.data_adapters.dhan_options_history import COLUMNS, OptionsHistory, SIDES

STEP = 50


def session_bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Only the bars inside the NSE session (09:15 to before 15:30), as cleaned real data would be."""
    minutes = ohlcv["timestamp"].dt.hour * 60 + ohlcv["timestamp"].dt.minute
    return ohlcv[(minutes >= 9 * 60 + 15) & (minutes < 15 * 60 + 30)].reset_index(drop=True)


def option_frame(bars: pd.DataFrame, kind: str, offset: int, step: int = STEP, oi_base: float = 10_000.0,
                 oi_scale: float = 1.0) -> pd.DataFrame:
    """One rolling series (COLUMNS) for `kind` at `offset` strikes from the money, from OHLC spot bars."""
    spot_c, spot_o = bars["close"].to_numpy(float), bars["open"].to_numpy(float)
    atm = np.round(spot_c / step) * step
    strike = atm + offset * step

    def value(spot):
        intrinsic = np.maximum(spot - strike, 0.0) if kind == "CALL" else np.maximum(strike - spot, 0.0)
        return intrinsic + 120.0 * np.exp(-np.abs(spot - strike) / 300.0)

    o, c = value(spot_o), value(spot_c)
    day_index = pd.factorize(bars["timestamp"].dt.date)[0]
    return pd.DataFrame({
        "timestamp": bars["timestamp"].to_numpy(), "open": o, "high": np.maximum(o, c) + 0.5, "low": np.minimum(o, c) - 0.5, "close": c,
        "iv": 15.0 + 0.5 * offset, "volume": 100.0 + 10 * (offset + 5), "strike": strike,
        # open interest belongs to the contract (its strike), not to the moving "offset" label
        "oi": (oi_base * (5 + (strike / step) % 20) + 100.0 * day_index) * oi_scale, "spot": spot_c,
    })[COLUMNS]


def make_history(bars: pd.DataFrame, offsets=range(-3, 4), symbol: str = "NIFTY", put_oi_scale: float = 2.0) -> OptionsHistory:
    frames = {}
    for kind in SIDES:
        for off in offsets:
            frames[(kind, off)] = option_frame(bars, kind, off, oi_scale=put_oi_scale if kind == "PUT" else 1.0)
    return OptionsHistory(symbol, frames)


def epoch_for(ts: pd.Series) -> list[int]:
    """Dhan-style epoch seconds for naive IST timestamps (UTC basis)."""
    return [int((t - pd.Timedelta(hours=5, minutes=30) - pd.Timestamp(0)).total_seconds()) for t in ts]


def response_body(frame: pd.DataFrame, kind: str) -> dict:
    side = {"timestamp": epoch_for(frame["timestamp"])}
    for col in COLUMNS[1:]:
        side[col] = frame[col].tolist()
    return {"data": {"ce": side if kind == "CALL" else None, "pe": side if kind == "PUT" else None}}


class FakeRollingApi:
    """Stands in for client._post(ENDPOINT, body): serves windows [fromDate, toDate) of a prepared history and counts calls."""

    def __init__(self, history: OptionsHistory, fail_after: int | None = None):
        self.history, self.calls, self.bodies, self.fail_after = history, 0, [], fail_after

    def __call__(self, body: dict) -> dict:
        self.calls += 1
        self.bodies.append(body)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("Dhan API error (401): token expired")
        kind = body["drvOptionType"]
        label = body["strike"]
        offset = 0 if label == "ATM" else int(label[3:])
        f = self.history.frames.get((kind, offset))
        if f is None:
            return {"data": {"ce": None, "pe": None}}
        lo, hi = pd.Timestamp(body["fromDate"]), pd.Timestamp(body["toDate"])
        window = f[(f.index >= lo) & (f.index < hi)].reset_index()
        return response_body(window, kind) if len(window) else {"data": {"ce": None, "pe": None}}


def day(d: str) -> dt.date:
    return dt.date.fromisoformat(d)
