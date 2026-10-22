"""
Rebuilding option-chain snapshots from stored rolling-strike history, so the chain-aware strategies can be backtested.

`HistoricalChain.snapshot_at(ts)` returns the `ChainSnapshot` as it could have been seen at bar `ts`, from the strikes
around spot that the history holds (see data_adapters/dhan_options_history.py). Everything is causal: a snapshot at `ts`
uses bars at or before `ts` only.

Fields, and how they differ from the live chain:
* ltp / iv / oi   - the bar's close, iv and oi for that strike.
* volume          - cumulative traded volume for that strike since the session open, summed over the bars in which the strike
                    was inside the stored window. A strike that drifted out of the window mid-session is under-counted.
* oi_change       - oi minus that strike's last oi of the previous session (None when it was not in the window then).
* the strikes     - only the offsets held (for example 7 strikes either side of spot), so `total_pcr` is a near-the-money
                    ratio, not the whole-chain PCR the strategy's thresholds were written for. Snapshots are marked
                    `partial=True`, and validation caps the evidence state of strategies run on them below VALIDATED.

`generate_chain_setups` replays a chain-aware strategy's own `check_chain_setup` bar by bar on those snapshots. Sell-side
setups are not generated: the backtest cost and risk model prices bought options only, and writing premium needs a margin
and premium-stop model that does not exist yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_intelligence.data_adapters.dhan_options_history import OptionsHistory
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow


class HistoricalChain:
    def __init__(self, history: OptionsHistory, expiry_label: str | None = None) -> None:
        self.history = history
        self.symbol = history.symbol
        self.expiry_label = expiry_label or f"rolling-{history.expiry_flag}{history.expiry_code}"
        parts = []
        for (kind, offset), f in history.frames.items():
            p = f.reset_index()[["timestamp", "strike", "close", "volume", "iv", "oi", "spot"]].copy()
            p["kind"], p["offset"] = kind, offset
            parts.append(p)
        long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["timestamp", "strike", "close", "volume", "iv", "oi", "spot", "kind", "offset"])
        long = long.dropna(subset=["strike"]).sort_values(["timestamp", "kind", "strike"])
        long = long.drop_duplicates(["timestamp", "kind", "strike"], keep="last").reset_index(drop=True)
        long["date"] = long["timestamp"].dt.date
        long["cum_volume"] = long.groupby(["kind", "strike", "date"])["volume"].transform(lambda s: s.fillna(0).cumsum())
        self._long = long
        self._ts = long["timestamp"].to_numpy()
        self._arr = {c: long[c].to_numpy() for c in ("kind", "strike", "close", "oi", "cum_volume", "iv", "spot", "date")}
        # last oi of every (kind, strike) per session, for the day-over-day change
        last = long.sort_values("timestamp").groupby(["date", "kind", "strike"])["oi"].last()
        self._last_oi: dict = {}
        for (d, kind, strike), oi in last.items():
            self._last_oi.setdefault(d, {})[(kind, strike)] = oi
        dates = sorted(self._last_oi)
        self._prev_date = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}

    def _previous_oi(self, date, kind: str, strike: float):
        prev = self._prev_date.get(date)
        return self._last_oi[prev].get((kind, strike)) if prev is not None else None

    def snapshot_at(self, ts) -> ChainSnapshot | None:
        ts64 = np.datetime64(pd.Timestamp(ts))
        lo, hi = np.searchsorted(self._ts, ts64, "left"), np.searchsorted(self._ts, ts64, "right")
        if hi <= lo:
            return None
        a = self._arr
        date = a["date"][lo]
        by_strike: dict[float, dict] = {}
        for k in range(lo, hi):
            kind, strike, oi = a["kind"][k], float(a["strike"][k]), a["oi"][k]
            prev = self._previous_oi(date, kind, strike)
            by_strike.setdefault(strike, {})["ce" if kind == "CALL" else "pe"] = (
                a["close"][k], oi, None if prev is None or oi != oi else oi - prev, a["cum_volume"][k], a["iv"][k])
        spots = a["spot"][lo:hi]
        spots = spots[~np.isnan(spots)]
        if not len(spots) or not by_strike:
            return None
        strikes = []
        for strike in sorted(by_strike):
            ce, pe = by_strike[strike].get("ce"), by_strike[strike].get("pe")
            strikes.append(StrikeRow(
                strike=strike,
                ce_ltp=ce[0] if ce else None, ce_oi=ce[1] if ce else None, ce_oi_change=ce[2] if ce else None,
                ce_volume=ce[3] if ce else None, ce_iv=ce[4] if ce else None, ce_delta=None, ce_security_id=None,
                pe_ltp=pe[0] if pe else None, pe_oi=pe[1] if pe else None, pe_oi_change=pe[2] if pe else None,
                pe_volume=pe[3] if pe else None, pe_iv=pe[4] if pe else None, pe_delta=None, pe_security_id=None))
        return ChainSnapshot(self.symbol, self.expiry_label, float(spots[0]), strikes, partial=True, source="dhan_history")

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.history.span()


class FeatureState:
    """The slice of MarketState a strategy's check_chain_setup reads: `.get(feature)` over one feature row."""

    def __init__(self, row: dict) -> None:
        self.features = row

    def get(self, key: str, default=None):
        v = self.features.get(key, default)
        if v is None or (isinstance(v, float) and v != v):
            return default
        return v


def generate_chain_setups(strategy, ohlcv: pd.DataFrame, features: pd.DataFrame, chain: HistoricalChain) -> list:
    """Every setup `strategy.check_chain_setup` would have produced bar by bar on the rebuilt snapshots (BUY-side only)."""
    lo, hi = chain.span()
    ts_all = pd.to_datetime(ohlcv["timestamp"]).to_numpy()
    feat = features.reset_index(drop=True)
    wanted = [c for c in getattr(strategy, "required_features", []) if c in feat.columns] or list(feat.columns)
    cols = {c: feat[c].to_numpy() for c in wanted}      # arrays once, not a pandas row per bar
    setups = []
    for i in range(len(ohlcv)):
        ts = pd.Timestamp(ts_all[i])
        if ts < lo or ts > hi:
            continue
        snap = chain.snapshot_at(ts)
        if snap is None:
            continue
        state = FeatureState({c: a[i] for c, a in cols.items()})
        recent = ohlcv.iloc[max(0, i - 4): i + 1]
        setup = strategy.check_chain_setup(state, recent, snap)
        if setup is None or (setup.meta or {}).get("transaction", "BUY") != "BUY":
            continue
        setup.timestamp = ts
        setups.append(setup)
    return setups
