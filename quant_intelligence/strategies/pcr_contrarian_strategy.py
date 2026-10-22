"""PCR Extreme Contrarian (options-chain-aware).

Setup: the whole-chain Put/Call Ratio (total PE OI / total CE OI) reaches an
extreme alongside a short-term RSI(2) reversal in the same direction. A very
high PCR means the crowd is heavily hedged/positioned bearish (contrarian
bullish read); a very low PCR means the crowd is heavily positioned bullish
(contrarian bearish read). Combining it with an RSI(2) extreme (the same
short-term-exhaustion filter used by RSI2MeanReversionStrategy) avoids acting
on PCR alone, which drifts slowly and is noisy in isolation.

Always BUYS the opposite-side option (defined risk = premium paid) - this is
a directional bet, not a hedge/spread.

LIMITATION: this strategy needs a live option chain (PCR, per-strike OI) that
this codebase does not have a historical data source for, so
`generate_historical_setups` returns [] (matching the "report zero, never
fabricate" convention used elsewhere, e.g. feature_engine's NaN derivatives
placeholders). It only produces setups live/paper via `check_chain_setup`.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class PCRContrarianStrategy(BaseStrategy):
    name = "PCR Extreme Contrarian"
    description = (
        "Fades chain-wide PCR extremes confirmed by an RSI(2) reversal, buying the "
        "opposite-side option (long calls when PCR is extremely high and oversold, "
        "long puts when PCR is extremely low and overbought)."
    )
    required_features = ["rsi_2", "atr_14"]
    required_chain_features = ["total_pcr"]
    default_parameters = {
        "pcr_bullish_threshold": 1.5,
        "pcr_bearish_threshold": 0.5,
        "rsi_oversold": 20.0,
        "rsi_overbought": 80.0,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 1.5,
        "moneyness_offset": 0,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        return self.historical_chain_setups(ohlcv, features)

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        return None

    def check_chain_setup(self, market_state, recent_ohlcv: pd.DataFrame, chain: ChainSnapshot) -> Setup | None:
        pcr = chain.total_pcr
        rsi2 = market_state.get("rsi_2")
        atr = market_state.get("atr_14")
        if pcr is None or rsi2 is None or atr is None:
            return None

        entry = chain.spot_price
        ts = recent_ohlcv.iloc[-1]["timestamp"] if len(recent_ohlcv) else None

        if pcr > self.parameters["pcr_bullish_threshold"] and rsi2 <= self.parameters["rsi_oversold"]:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"total_pcr": pcr, "transaction": "BUY"})
        if pcr < self.parameters["pcr_bearish_threshold"] and rsi2 >= self.parameters["rsi_overbought"]:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"total_pcr": pcr, "transaction": "BUY"})
        return None
