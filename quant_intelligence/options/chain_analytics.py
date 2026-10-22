"""
Option Chain Analytics.

Turns a raw Dhan `/v2/optionchain` response into a clean, documented
snapshot - the options-market equivalent of `feature_engine.compute_features`.
Every derived field below has a precise definition so strategies never treat
it as a black box (same convention as FEATURE_DEFINITIONS in feature_engine).

Raw Dhan response shape (see docs/v2/option-chain):
    {"data": {"last_price": <spot>, "oc": {"<strike>": {"ce": {...}, "pe": {...}}, ...}}}
Each leg dict: last_price, oi, previous_oi, volume, implied_volatility,
top_bid_price, top_ask_price, greeks: {delta, gamma, theta, vega}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHAIN_FIELD_DEFINITIONS: dict[str, str] = {
    "total_pcr": "Put/Call ratio across the whole chain: sum(PE open interest) / sum(CE open interest)",
    "atm_strike": "Strike nearest to the current spot/underlying price",
    "atm_iv": "Average of CE and PE implied volatility at the ATM strike",
    "iv_skew": "OTM put IV minus OTM call IV at roughly equal distance from spot (25-delta proxy); "
    "positive skew = puts pricing more fear than calls",
    "max_pain_strike": "Strike at which total option-writer payout (CE+PE) is minimized at expiry",
    "unusual_volume_strikes": "Strikes where today's volume/OI ratio is a statistical outlier (z-score > 2) "
    "versus the rest of the chain, flagging concentrated speculative flow",
}


@dataclass
class StrikeRow:
    strike: float
    ce_ltp: float | None
    ce_oi: float | None
    ce_oi_change: float | None
    ce_volume: float | None
    ce_iv: float | None
    ce_delta: float | None
    ce_security_id: str | None
    pe_ltp: float | None
    pe_oi: float | None
    pe_oi_change: float | None
    pe_volume: float | None
    pe_iv: float | None
    pe_delta: float | None
    pe_security_id: str | None

    @property
    def pcr_at_strike(self) -> float | None:
        if not self.ce_oi:
            return None
        return (self.pe_oi or 0.0) / self.ce_oi


@dataclass
class ChainSnapshot:
    underlying: str
    expiry: str
    spot_price: float
    strikes: list[StrikeRow] = field(default_factory=list)
    # A snapshot rebuilt from rolling-strike history holds only the strikes near the money, so whole-chain totals such as
    # total_pcr are near-the-money figures, not the live whole-chain ones. `partial` says so; live snapshots are complete.
    partial: bool = False
    source: str = "live"

    @property
    def atm_strike(self) -> float | None:
        if not self.strikes:
            return None
        return min(self.strikes, key=lambda s: abs(s.strike - self.spot_price)).strike

    @property
    def total_pcr(self) -> float | None:
        total_ce_oi = sum(s.ce_oi or 0.0 for s in self.strikes)
        total_pe_oi = sum(s.pe_oi or 0.0 for s in self.strikes)
        if total_ce_oi == 0:
            return None
        return total_pe_oi / total_ce_oi

    @property
    def atm_iv(self) -> float | None:
        atm = self._row_at(self.atm_strike)
        if atm is None:
            return None
        ivs = [v for v in (atm.ce_iv, atm.pe_iv) if v is not None]
        return sum(ivs) / len(ivs) if ivs else None

    @property
    def iv_skew(self) -> float | None:
        """OTM put IV (below spot) minus OTM call IV (above spot), at the strikes nearest
        one strike-step away from ATM on each side - a simple proxy for the standard
        25-delta risk-reversal skew without needing per-contract delta interpolation."""
        atm = self.atm_strike
        if atm is None or len(self.strikes) < 3:
            return None
        below = [s for s in self.strikes if s.strike < atm]
        above = [s for s in self.strikes if s.strike > atm]
        if not below or not above:
            return None
        put_leg = max(below, key=lambda s: s.strike)
        call_leg = min(above, key=lambda s: s.strike)
        if put_leg.pe_iv is None or call_leg.ce_iv is None:
            return None
        return put_leg.pe_iv - call_leg.ce_iv

    @property
    def max_pain_strike(self) -> float | None:
        if not self.strikes:
            return None
        candidate_strikes = [s.strike for s in self.strikes]
        best_strike, best_payout = None, None
        for expiry_price in candidate_strikes:
            payout = 0.0
            for row in self.strikes:
                if row.ce_oi:
                    payout += max(expiry_price - row.strike, 0.0) * row.ce_oi
                if row.pe_oi:
                    payout += max(row.strike - expiry_price, 0.0) * row.pe_oi
            if best_payout is None or payout < best_payout:
                best_payout, best_strike = payout, expiry_price
        return best_strike

    @property
    def unusual_volume_strikes(self) -> list[float]:
        ratios = []
        for row in self.strikes:
            for oi, vol in ((row.ce_oi, row.ce_volume), (row.pe_oi, row.pe_volume)):
                if oi and vol is not None:
                    ratios.append(vol / oi)
        if len(ratios) < 4:
            return []
        mean = sum(ratios) / len(ratios)
        variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        std = variance**0.5
        if std == 0:
            return []
        flagged: list[float] = []
        for row in self.strikes:
            for oi, vol in ((row.ce_oi, row.ce_volume), (row.pe_oi, row.pe_volume)):
                if oi and vol is not None and (vol / oi - mean) / std > 2:
                    flagged.append(row.strike)
                    break
        return flagged

    def _row_at(self, strike: float | None) -> StrikeRow | None:
        if strike is None:
            return None
        for row in self.strikes:
            if row.strike == strike:
                return row
        return None


def parse_option_chain(underlying: str, expiry: str, raw_response: dict) -> ChainSnapshot:
    """Parse a raw Dhan `/v2/optionchain` response body into a ChainSnapshot."""
    data = raw_response.get("data", {})
    spot_price = data.get("last_price", 0.0)
    oc = data.get("oc", {})

    strikes: list[StrikeRow] = []
    for strike_str, legs in oc.items():
        ce = legs.get("ce", {}) or {}
        pe = legs.get("pe", {}) or {}
        strikes.append(
            StrikeRow(
                strike=float(strike_str),
                ce_ltp=ce.get("last_price"),
                ce_oi=ce.get("oi"),
                ce_oi_change=_oi_change(ce),
                ce_volume=ce.get("volume"),
                ce_iv=ce.get("implied_volatility"),
                ce_delta=(ce.get("greeks") or {}).get("delta"),
                ce_security_id=ce.get("security_id"),
                pe_ltp=pe.get("last_price"),
                pe_oi=pe.get("oi"),
                pe_oi_change=_oi_change(pe),
                pe_volume=pe.get("volume"),
                pe_iv=pe.get("implied_volatility"),
                pe_delta=(pe.get("greeks") or {}).get("delta"),
                pe_security_id=pe.get("security_id"),
            )
        )
    strikes.sort(key=lambda s: s.strike)
    return ChainSnapshot(underlying=underlying, expiry=expiry, spot_price=spot_price, strikes=strikes)


def _oi_change(leg: dict) -> float | None:
    oi, prev_oi = leg.get("oi"), leg.get("previous_oi")
    if oi is None or prev_oi is None:
        return None
    return oi - prev_oi
