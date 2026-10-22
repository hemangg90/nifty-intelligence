"""
Option Contract Selection.

Pure logic, no network calls - takes an already-fetched ChainSnapshot and
resolves a strategy's directional Setup into a concrete tradeable option
contract. Every underlying that can be traded must resolve via
`get_underlying_info` (security id + segment + strike step + lot size):
the two indices below are statically registered; any other symbol is
resolved dynamically as an NSE F&O stock via
`data_adapters.dhan_instrument_master.resolve_fno_stock` (backed by Dhan's
own scrip master, so the tradeable stock universe stays in sync with NSE's
actual listings without a hand-maintained list).

Direction mapping (LONG/SHORT are the strategy's view on the underlying,
independent of whether the resulting option trade is a BUY or a SELL):
  LONG  -> the bullish leg -> CE
  SHORT -> the bearish leg -> PE
A strategy that wants to profit from the underlying falling can equally do
so by SELLING a CE (transaction="SELL", option_type comes out as CE) - the
`transaction` parameter is independent of `direction`.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_intelligence.options.chain_analytics import ChainSnapshot, parse_option_chain

# Index underlyings: security_id/seg are Dhan's permanent instrument identifiers, static
# here. strike_step/lot_size are last-known-good fallbacks only - NSE revises index lot
# sizes periodically, so get_underlying_info() always tries to refresh them from the
# scrip master first (see resolve_derivative_lot_specs) and only falls back to these
# if that lookup fails (e.g. offline with no cached scrip master yet).
# `option_segment` overrides the exchange segment option orders/LTP go to (default: the
# symbol's MarketProfile, NSE_FNO) - SENSEX shares NSE's session hours but trades on BSE.
UNDERLYING_REGISTRY: dict[str, dict] = {
    "NIFTY": {"security_id": 13, "seg": "IDX_I", "strike_step": 50, "lot_size": 75},
    "BANKNIFTY": {"security_id": 25, "seg": "IDX_I", "strike_step": 100, "lot_size": 30},
    "FINNIFTY": {"security_id": 27, "seg": "IDX_I", "strike_step": 50, "lot_size": 60},
    "MIDCPNIFTY": {"security_id": 442, "seg": "IDX_I", "strike_step": 25, "lot_size": 120},
    "SENSEX": {"security_id": 51, "seg": "IDX_I", "strike_step": 100, "lot_size": 20, "option_segment": "BSE_FNO"},
}


def option_segment_for(symbol: str | None) -> str:
    """Dhan exchange segment for `symbol`'s option orders and LTP (BSE_FNO / NSE_FNO / MCX_COMM)."""
    from quant_intelligence.utils.market_profile import profile_for

    entry = UNDERLYING_REGISTRY.get((symbol or "").strip().upper(), {})
    return entry.get("option_segment") or profile_for(symbol).option_segment


class OptionSelectionError(ValueError):
    pass


def get_underlying_info(underlying: str) -> dict:
    """Returns {"security_id", "seg", "strike_step", "lot_size"} for any tradeable
    underlying - a statically registered index (lot_size/strike_step refreshed from
    the scrip master when available), or an NSE F&O stock resolved dynamically from
    Dhan's scrip master. Raises OptionSelectionError if neither resolves (i.e. the
    symbol has no listed options at all)."""
    from quant_intelligence.data_adapters.dhan_instrument_master import (
        resolve_derivative_lot_specs,
        resolve_fno_stock,
        resolve_mcx,
    )

    underlying = underlying.strip().upper()
    static_info = UNDERLYING_REGISTRY.get(underlying)
    if static_info is not None:
        fresh_specs = resolve_derivative_lot_specs(underlying)
        return {**static_info, **fresh_specs} if fresh_specs is not None else static_info

    info = resolve_fno_stock(underlying) or resolve_mcx(underlying)
    if info is None:
        raise OptionSelectionError(
            f"Underlying '{underlying}' has no registered index entry, no listed NSE F&O stock options "
            "and is not a supported MCX commodity with listed options"
        )
    return info


@dataclass
class OptionContract:
    security_id: str
    trading_symbol: str
    underlying: str
    strike: float
    option_type: str  # CE / PE
    expiry: str
    lot_size: int
    transaction: str  # BUY / SELL
    premium: float
    delta: float | None
    iv: float | None


def resolve_expiry(available_expiries: list[str], preference: str = "NEAREST") -> str:
    if not available_expiries:
        raise OptionSelectionError("No expiries available for this underlying")
    ordered = sorted(available_expiries)
    if preference == "NEAREST_MONTHLY":
        # The last listed expiry within each month is the monthly expiry; pick the
        # nearest month's monthly expiry that hasn't passed.
        by_month: dict[tuple[int, int], str] = {}
        for exp in ordered:
            y, m, _ = exp.split("-")
            by_month[(int(y), int(m))] = exp  # last write per month wins since ordered ascending
        return sorted(by_month.items())[0][1]
    return ordered[0]


def fetch_chain(
    client, underlying: str, expiry_preference: str = "NEAREST", max_age: float | None = None
) -> ChainSnapshot:
    """Fetch and resolve the current live option chain for `underlying` - a registered
    index or any NSE F&O stock (see get_underlying_info). Makes real network calls via
    `client` (a DhanApiClient), but at most one per `max_age` seconds per underlying: the result is
    shared by every caller (background runners and every browser tab) through a single-flight cache,
    which is what keeps the account under Dhan's option-chain limit. `max_age` defaults to
    SETTINGS.dhan.chain_cache_ttl_sec; pass 0 to force a fresh fetch (e.g. before a live order)."""
    from quant_intelligence.brokers.dhan_cache import CHAIN_CACHE
    from quant_intelligence.config.settings import SETTINGS

    ttl = SETTINGS.dhan.chain_cache_ttl_sec if max_age is None else max_age
    key = (getattr(client, "base_url", ""), underlying.strip().upper(), expiry_preference)
    return CHAIN_CACHE.get(key, lambda: _fetch_chain_uncached(client, underlying, expiry_preference), ttl)


def _fetch_chain_uncached(client, underlying: str, expiry_preference: str) -> ChainSnapshot:
    info = get_underlying_info(underlying)
    # MCX options sit on a futures contract; once the front-month future has no live option
    # expiry left, the next month's future carries the chain.
    for security_id in info.get("security_ids", [info["security_id"]])[:2]:
        expiries = client.get_expiry_list(security_id, info["seg"])
        if expiries:
            break
    expiry = resolve_expiry(expiries, expiry_preference)
    raw = client.get_option_chain(security_id, info["seg"], expiry)
    snapshot = parse_option_chain(underlying, expiry, raw)
    from quant_intelligence.options.chain_recorder import record_snapshot

    record_snapshot(snapshot)  # no-op unless CHAIN_RECORDER is on; never raises
    return snapshot


def select_contract(
    chain: ChainSnapshot,
    direction: str,
    transaction: str,
    moneyness_offset: int = 0,
    lot_size: int = 1,
) -> OptionContract:
    """Resolve a directional Setup into a concrete option contract from `chain`.

    moneyness_offset: 0 = ATM. For a BUY, positive values move OTM (cheaper premium,
    lower delta). For a SELL, positive values also move OTM (further from spot = safer
    to write, lower premium collected). The sign of the offset always means "away from
    spot" regardless of BUY/SELL, so callers don't have to reason about strike direction.
    """
    if direction not in ("LONG", "SHORT"):
        raise OptionSelectionError(f"Unknown direction: {direction}")
    if transaction not in ("BUY", "SELL"):
        raise OptionSelectionError(f"Unknown transaction: {transaction}")
    if not chain.strikes:
        raise OptionSelectionError("Option chain has no strikes")

    option_type = "CE" if direction == "LONG" else "PE"
    sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike)
    atm_strike = chain.atm_strike
    atm_idx = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i].strike - atm_strike))

    # "Away from spot" is toward higher strikes for calls, lower strikes for puts.
    step = 1 if option_type == "CE" else -1
    target_idx = atm_idx + step * moneyness_offset
    target_idx = max(0, min(len(sorted_strikes) - 1, target_idx))
    row = sorted_strikes[target_idx]

    if option_type == "CE":
        premium, delta, iv, security_id = row.ce_ltp, row.ce_delta, row.ce_iv, row.ce_security_id
    else:
        premium, delta, iv, security_id = row.pe_ltp, row.pe_delta, row.pe_iv, row.pe_security_id

    if premium is None or security_id is None:
        raise OptionSelectionError(f"No {option_type} quote/security_id available at strike {row.strike}")

    trading_symbol = f"{chain.underlying} {chain.expiry} {int(row.strike)} {option_type}"
    return OptionContract(
        security_id=str(security_id),
        trading_symbol=trading_symbol,
        underlying=chain.underlying,
        strike=row.strike,
        option_type=option_type,
        expiry=chain.expiry,
        lot_size=lot_size,
        transaction=transaction,
        premium=float(premium),
        delta=delta,
        iv=iv,
    )
