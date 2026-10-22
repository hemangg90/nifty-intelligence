"""OI Buildup Confirmation (options-chain-aware).

Setup: a Donchian-style 20-bar breakout/breakdown in the underlying,
confirmed by rising open interest at the near-ATM strike on the side that
agrees with the move ("long buildup" - rising OI + rising price - is the
classic Indian F&O reading of fresh directional conviction, as opposed to a
breakout on falling/flat OI which is more likely a short-covering fakeout).
Buys the confirming option outright.

LIMITATION: no historical option-chain data source exists in this codebase,
so `generate_historical_setups` returns [] - this strategy only produces
setups live/paper via `check_chain_setup` (see PCRContrarianStrategy for the
same, documented limitation).
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class OIBuildupStrategy(BaseStrategy):
    name = "OI Buildup Confirmation"
    description = (
        "Confirms a Donchian channel breakout/breakdown with rising open interest at "
        "the near-ATM strike on the agreeing side, buying the confirming option."
    )
    required_features = ["donchian_high_20", "donchian_low_20", "atr_14"]
    required_chain_features = ["atm_strike"]
    default_parameters = {
        "min_oi_change_pct": 5.0,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 2.0,
        "moneyness_offset": 0,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        return self.historical_chain_setups(ohlcv, features)

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        return None

    def check_chain_setup(self, market_state, recent_ohlcv: pd.DataFrame, chain: ChainSnapshot) -> Setup | None:
        high_ch = market_state.get("donchian_high_20")
        low_ch = market_state.get("donchian_low_20")
        atr = market_state.get("atr_14")
        atm_strike = chain.atm_strike
        if high_ch is None or low_ch is None or atr is None or atm_strike is None:
            return None

        atm_row = next((s for s in chain.strikes if s.strike == atm_strike), None)
        if atm_row is None:
            return None

        entry = chain.spot_price
        ts = recent_ohlcv.iloc[-1]["timestamp"] if len(recent_ohlcv) else None
        min_change_pct = self.parameters["min_oi_change_pct"]

        ce_oi_change_pct = _pct_change(atm_row.ce_oi_change, atm_row.ce_oi)
        pe_oi_change_pct = _pct_change(atm_row.pe_oi_change, atm_row.pe_oi)

        if entry > high_ch and ce_oi_change_pct is not None and ce_oi_change_pct >= min_change_pct:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"ce_oi_change_pct": ce_oi_change_pct, "transaction": "BUY"})
        if entry < low_ch and pe_oi_change_pct is not None and pe_oi_change_pct >= min_change_pct:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"pe_oi_change_pct": pe_oi_change_pct, "transaction": "BUY"})
        return None


def _pct_change(change: float | None, current: float | None) -> float | None:
    if change is None or not current:
        return None
    prior = current - change
    if not prior:
        return None
    return change / prior * 100.0
