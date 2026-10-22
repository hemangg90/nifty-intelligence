"""Real option prices for the backtest (behind OPTION_HISTORY_IN_BACKTEST), from Dhan's rolling-option history.

Same interface as `BacktestPremiumModel` (`price(trade)` returns the dict the engine stores as `bs_premium`), but the
numbers are traded prices instead of Black-Scholes, so theta, gamma, vega, skew and the real premium level are all in.

For a trade the model prices the BOUGHT ATM option (call for LONG, put for SHORT):
  * entry - the option's price at the fill: the OPEN of the bar after the signal under `next_open` timing, the CLOSE of the
    signal bar under the legacy `signal_close` timing. Its strike K is the ATM strike at that bar.
  * exit  - the CLOSE of the exit bar, for the SAME strike K (found among the stored offsets, since "ATM" moves with spot).
    A stop or target reached inside a bar is priced at the bar's close: bar resolution, stated limit.
  * risk  - the premium lost if the underlying reached the stop: a delta read from the stored neighbouring strikes at the
    entry bar (central difference, clipped to 0.1-0.95) times the stop distance. First order, so it ignores gamma.

A trade is priced only when everything it needs exists in the history; otherwise `price` returns None and the engine
keeps the modelled or assumed premium for that trade (counted separately). Reasons are tallied in `skipped_reasons`.
Not modelled: bid/ask spread (the data has none), overnight holds (the "same strike" cannot be followed across a roll),
contracts rolling to a new expiry mid-trade.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from quant_intelligence.data_adapters.dhan_options_history import OptionsHistory

MIN_PREMIUM = 0.05
DELTA_BOUNDS = (0.10, 0.95)


class RealPremiumModel:
    source = "dhan_history"

    def __init__(self, history: OptionsHistory, tf_minutes: int, timing: str = "signal_close") -> None:
        self.history = history
        self.tf = tf_minutes
        self.timing = timing
        self.priced = 0
        self.skipped = 0
        self.skipped_reasons: Counter = Counter()

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons[reason] += 1
        return None

    def _entry_row(self, kind: str, signal_ts: pd.Timestamp):
        if self.timing == "next_open":
            ts = signal_ts + pd.Timedelta(minutes=self.tf)
            return ts, self.history.row(kind, 0, ts), "open"
        return signal_ts, self.history.row(kind, 0, signal_ts), "close"

    def _delta(self, kind: str, ts, entry_row) -> float | None:
        up, down = self.history.row(kind, 1, ts), self.history.row(kind, -1, ts)
        if up is None or down is None:
            return None
        width = float(up["strike"]) - float(down["strike"])
        if not width or width != width:
            return None
        # Call value falls with strike, put value rises: delta is the signed slope, positive for a bought option.
        slope = (float(down["close"]) - float(up["close"])) / width if kind == "CALL" else (float(up["close"]) - float(down["close"])) / width
        if slope != slope:
            return None
        return min(max(slope, DELTA_BOUNDS[0]), DELTA_BOUNDS[1])

    def price(self, trade) -> dict | None:
        setup = trade.setup
        if trade.exit_price is None or trade.exit_timestamp is None:
            return self._skip("no exit")
        signal_ts, exit_ts = pd.Timestamp(setup.timestamp), pd.Timestamp(trade.exit_timestamp)
        kind = "CALL" if setup.direction == "LONG" else "PUT"
        entry_ts, row, field = self._entry_row(kind, signal_ts)
        if row is None:
            return self._skip("no option bar at entry")
        if entry_ts.date() != exit_ts.date():
            return self._skip("held overnight")
        entry = float(row[field])
        strike = float(row["strike"])
        if not (entry == entry and strike == strike) or entry <= 0:
            return self._skip("entry price missing")

        delta = self._delta(kind, entry_ts, row)
        if delta is None:
            return self._skip("no neighbouring strikes for delta")

        exit_row = None
        for offset in self.history.offsets:          # the same strike, wherever "ATM" has moved to by the exit bar
            r = self.history.row(kind, offset, exit_ts)
            if r is not None and float(r["strike"]) == strike:
                exit_row = r
                break
        if exit_row is None:
            return self._skip("held strike not in stored offsets at exit")
        exit_ = float(exit_row["close"])
        if exit_ != exit_:
            return self._skip("exit price missing")

        risk = delta * abs(float(setup.entry_price) - float(setup.stop_price))
        if risk <= 0:
            return self._skip("zero risk")
        self.priced += 1
        iv = row.get("iv")
        return {"entry": max(entry, MIN_PREMIUM), "exit": max(exit_, MIN_PREMIUM), "risk": risk,
                "sigma": None if iv is None or iv != iv else float(iv) / 100.0, "expiry": None, "kind": kind,
                "T_entry": None, "strike": strike, "delta": delta, "source": self.source}

    def coverage(self) -> dict:
        return {"priced": self.priced, "skipped": self.skipped, "skipped_reasons": dict(self.skipped_reasons)}


def history_covers(history: OptionsHistory | None, ohlcv: pd.DataFrame) -> bool:
    """Does the stored history overlap the data being backtested at all?"""
    if history is None or len(ohlcv) == 0:
        return False
    lo, hi = history.span()
    ts = pd.to_datetime(ohlcv["timestamp"])
    return bool(ts.max() >= lo and ts.min() <= hi)


def load_history_for(instrument: str) -> OptionsHistory | None:
    try:
        return OptionsHistory.load(instrument)
    except Exception:
        return None


__all__ = ["RealPremiumModel", "history_covers", "load_history_for"]
