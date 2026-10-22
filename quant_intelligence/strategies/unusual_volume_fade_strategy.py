"""Unusual Volume Fade (options-chain-aware).

Setup: a strike near the money shows a statistically unusual volume/OI ratio
(see ChainSnapshot.unusual_volume_strikes) on one side without a matching
directional confirmation elsewhere in the chain - read as short-lived
speculative flow (retail chasing a spike) rather than fresh informed
positioning. Fades it by buying the cheaper opposite-side option, on the
view that speculative single-strike spikes mean-revert faster than genuine
OI buildups (contrast with OIBuildupStrategy, which requires the OI
confirmation this strategy explicitly treats as absent).

LIMITATION: no historical option-chain data source exists in this codebase,
so `generate_historical_setups` returns [] (same documented limitation as
the other chain-aware strategies).
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class UnusualVolumeFadeStrategy(BaseStrategy):
    name = "Unusual Volume Fade"
    description = (
        "Fades a statistically unusual single-strike volume spike near the money by "
        "buying the cheaper opposite-side option, expecting the speculative flow to reverse."
    )
    required_features = ["atr_14"]
    required_chain_features = ["unusual_volume_strikes", "atm_strike"]
    default_parameters = {
        "max_strikes_from_atm": 2,
        "stop_atr_mult": 0.75,
        "target_atr_mult": 1.0,
        "moneyness_offset": 0,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        return self.historical_chain_setups(ohlcv, features)

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        return None

    def check_chain_setup(self, market_state, recent_ohlcv: pd.DataFrame, chain: ChainSnapshot) -> Setup | None:
        atr = market_state.get("atr_14")
        atm_strike = chain.atm_strike
        flagged = chain.unusual_volume_strikes
        if atr is None or atm_strike is None or not flagged:
            return None

        sorted_strikes = sorted(s.strike for s in chain.strikes)
        step = sorted_strikes[1] - sorted_strikes[0] if len(sorted_strikes) > 1 else 1
        max_distance = self.parameters["max_strikes_from_atm"] * step

        near_flags = [f for f in flagged if abs(f - atm_strike) <= max_distance]
        if not near_flags:
            return None

        flagged_strike = min(near_flags, key=lambda f: abs(f - atm_strike))
        row = next(s for s in chain.strikes if s.strike == flagged_strike)

        ce_ratio = (row.ce_volume / row.ce_oi) if row.ce_oi else None
        pe_ratio = (row.pe_volume / row.pe_oi) if row.pe_oi else None

        entry = chain.spot_price
        ts = recent_ohlcv.iloc[-1]["timestamp"] if len(recent_ohlcv) else None

        # The spiking side is the one with the higher volume/OI ratio; fade it by buying
        # the opposite side (CE spike -> speculative bullish flow -> buy PE, and vice versa).
        if ce_ratio is not None and (pe_ratio is None or ce_ratio > pe_ratio):
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"flagged_strike": flagged_strike, "transaction": "BUY"})
        if pe_ratio is not None:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"flagged_strike": flagged_strike, "transaction": "BUY"})
        return None
