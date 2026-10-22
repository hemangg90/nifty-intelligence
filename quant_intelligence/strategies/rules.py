"""
Explicit, reviewable rules for every strategy in the library.

A backtest is only as unambiguous as the rules it simulates, so each strategy's entry, exit, stop, target,
sizing, session, time-exit and re-entry behaviour is written down here exactly as the code implements it
today. Nothing in this file changes behaviour; `tests/test_strategy_rules.py` keeps it honest (every registered
strategy has rules, every tunable default sits inside its permitted range, and ranges cover every numeric
default). When a strategy's logic changes, change its rules and bump its version in the same commit.

`PARAMETER_RANGES[name][param] = (low, high, step)` is the economically reasonable neighbourhood the
robustness engine may probe. It is NOT an optimisation search space: robustness testing reports how stable
results are across it and never picks the best point as unbiased evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ALL_TIMEFRAMES = ("5min",)  # the timeframe the library is researched and traded on
INDEX_AND_STOCKS = ("NSE indices", "NSE F&O stocks", "MCX commodities")
NSE_ONLY = ("NSE indices", "NSE F&O stocks")

_SIZING = (
    "Fixed lot-size quantity in the backtest; live/paper size is capped by the per-trade loss budget "
    "(execution/auto_trader.size_position) and scaled by the risk engine."
)
_SESSION_NONE = "No time-of-day restriction in the signal logic; any closed bar of the session can signal."
_TIME_EXIT = "Exit at the close of the max-holding bar (75 bars = one 5-min NSE session) if neither stop nor target was hit."


@dataclass(frozen=True)
class StrategyRules:
    entry: str
    exit: str
    stop: str
    target: str
    sizing: str = _SIZING
    session: str = _SESSION_NONE
    time_exit: str = _TIME_EXIT
    reentry: str = ""
    max_holding_bars: int = 75
    instruments: tuple[str, ...] = INDEX_AND_STOCKS
    timeframes: tuple[str, ...] = ALL_TIMEFRAMES
    backtestable: bool = True  # False: needs data the price history cannot provide (option chains)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


_CHAIN_LIMITATION = (
    "Needs historical option-chain data (PCR, OI, IV, strike volumes). Without it the strategy cannot be "
    "backtested and its evidence state is INSUFFICIENT DATA; no edge may be claimed from the underlying alone. "
    "With rolling-strike option history attached (options/historical_chain.py) the BUY-side ones can be replayed on "
    "near-the-money snapshots; that is a proxy for the whole chain, so they are capped below VALIDATED FOR RESEARCH."
)

# Chain-aware strategies that can be replayed on rebuilt near-the-money snapshots. IV Crush Premium Writer is excluded: it
# SELLS premium and the backtest has no sell-side cost, margin or premium-stop model.
CHAIN_HISTORY_BACKTESTABLE = frozenset({"PCR Extreme Contrarian", "OI Buildup Confirmation", "Unusual Volume Fade"})
PROXY_DATA_LIMITATION = ("backtested on near-the-money strikes from rolling-strike history, not the whole option chain the "
                         "strategy's thresholds were written for")

RULES: dict[str, StrategyRules] = {
    "Opening Range Breakout": StrategyRules(
        entry="Close crosses beyond the first-30-minute range: LONG when the previous close <= range high < close; "
              "SHORT when the previous close >= range low > close. Needs at least `min_minutes_since_open` elapsed.",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult from the entry price, on the opposite side.",
        target="ATR(14) x target_atr_mult from the entry price.",
        session="No signal before 31 minutes after the open (the opening range must be complete).",
        reentry="At most one LONG and one SHORT per session.",
        limitations=("Signal fires on the bar close; the live fill is the next bar's open.",),
    ),
    "Momentum": StrategyRules(
        entry="20-bar momentum above +threshold with positive trend slope (LONG) or below -threshold with negative "
              "slope (SHORT), and relative volume >= min_relative_volume.",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        reentry="`cooldown_bars` bars after any signal.",
        limitations=("Uses traded volume, so it is skipped on instruments without volume.",),
    ),
    "VWAP Mean Reversion": StrategyRules(
        entry="Price stretched beyond +/- stretch_threshold_pct from session VWAP while |trend slope| <= max_trend_slope, "
              "and the bar reverses (close below the previous close for SHORT, above for LONG).",
        exit="Stop, VWAP target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult beyond the entry.",
        target="Session VWAP as of the signal bar (frozen at signal time).",
        reentry="`cooldown_bars` bars after any signal.",
        limitations=("target_atr_mult is declared but unused: the target is the VWAP level.",),
    ),
    "RSI(2) Mean Reversion": StrategyRules(
        entry="With the trend (close above EMA200 for LONG, below for SHORT): RSI(2) <= oversold (LONG) or >= overbought (SHORT).",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        reentry="`cooldown_bars` bars after any signal.",
        limitations=("EMA200 on 5-minute bars spans roughly 2.7 sessions, so 'trend' is short-horizon.",),
    ),
    "Bollinger Band Mean Reversion": StrategyRules(
        entry="Previous close outside the 20-bar Bollinger band and the current close back inside it, while |trend slope| <= max_trend_slope. "
              "Above the upper band -> SHORT; below the lower band -> LONG.",
        exit="Stop, band-midline target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult beyond the entry.",
        target="Bollinger midline (20-bar mean) as of the signal bar.",
        reentry="`cooldown_bars` bars after any signal.",
        limitations=("target_atr_mult is declared but unused: the target is the band midline.",),
    ),
    "Supertrend Trend Following": StrategyRules(
        entry="Supertrend(10, 3) direction flips: to up -> LONG, to down -> SHORT.",
        exit="Stop (the Supertrend line at the flip), target, or time exit, whichever comes first.",
        stop="The Supertrend line value at the signal bar (not trailed in the backtest).",
        target="ATR(14) x target_atr_mult.",
        reentry="None beyond the next direction flip.",
        limitations=("The stop is static in the backtest even though the strategy is described as trailing.",),
    ),
    "EMA 50/200 Crossover": StrategyRules(
        entry="EMA(50) crosses above EMA(200) -> LONG; below -> SHORT.",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        reentry="None beyond the next crossover.",
        limitations=("Crossovers are rare on 5-minute bars; sample sizes are small and the 75-bar cap cuts most trades short.",),
    ),
    "Donchian Channel Breakout": StrategyRules(
        entry="Close above the 20-bar Donchian high -> LONG; below the 20-bar low -> SHORT.",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        reentry="`cooldown_bars` bars after any signal.",
    ),
    "MACD Signal Crossover": StrategyRules(
        entry="MACD line crosses its signal line, in the direction of the EMA200 trend (above EMA200 for LONG, below for SHORT).",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        reentry="`cooldown_bars` bars after any signal.",
    ),
    "CPR Breakout": StrategyRules(
        entry="Close crosses beyond the prior session's Central Pivot Range: previous close <= CPR high < close (LONG), "
              "or previous close >= CPR low > close (SHORT).",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult.",
        target="ATR(14) x target_atr_mult.",
        session="No signal before 16 minutes after the open.",
        reentry="`cooldown_bars` bars after any signal.",
    ),
    "Camarilla Pivot Reversal": StrategyRules(
        entry="Previous bar's high reached R3 and the current close is below the previous close (SHORT), or the previous low "
              "reached S3 and the current close is above the previous close (LONG).",
        exit="Stop, pivot target, or time exit, whichever comes first.",
        stop="The farther of the previous bar's extreme and ATR(14) x stop_atr_mult from the entry.",
        target="The prior session's pivot (CPR pivot).",
        session="No signal before 16 minutes after the open.",
        reentry="`cooldown_bars` bars after any signal.",
    ),
    "Inside Bar Breakout": StrategyRules(
        entry="The previous bar is an inside bar and the current close breaks its high (LONG) or low (SHORT).",
        exit="Stop, target, or time exit, whichever comes first.",
        stop="The opposite end of the inside bar.",
        target="ATR(14) x target_atr_mult from the entry.",
        reentry="`cooldown_bars` bars after any signal.",
    ),
    "Gap Fill Reversion": StrategyRules(
        entry="Opening gap between min_gap_pct and max_gap_pct; within `max_minutes_since_open` of the open the close reverses "
              "against the gap (below the previous close for a gap up -> SHORT; above for a gap down -> LONG).",
        exit="Stop, prior-close target, or time exit, whichever comes first.",
        stop="ATR(14) x stop_atr_mult beyond the entry.",
        target="The prior session's close, reconstructed from the opening gap.",
        session="Only in the first 20 minutes after the open.",
        reentry="One trade per session.",
        instruments=NSE_ONLY,
    ),
    "PCR Extreme Contrarian": StrategyRules(
        entry="Chain-wide PCR above pcr_bullish_threshold with RSI(2) oversold (buy CE), or below pcr_bearish_threshold with RSI(2) overbought (buy PE). Live only.",
        exit="Stop, target, or time exit on the underlying, mapped to the option.",
        stop="ATR(14) x stop_atr_mult on the underlying.",
        target="ATR(14) x target_atr_mult on the underlying.",
        reentry="Evaluated each cycle while flat.",
        instruments=NSE_ONLY,
        backtestable=False,
        limitations=(_CHAIN_LIMITATION,),
    ),
    "OI Buildup Confirmation": StrategyRules(
        entry="Donchian(20) breakout confirmed by near-ATM open interest rising at least min_oi_change_pct on the agreeing side. Live only.",
        exit="Stop, target, or time exit on the underlying, mapped to the option.",
        stop="ATR(14) x stop_atr_mult on the underlying.",
        target="ATR(14) x target_atr_mult on the underlying.",
        reentry="Evaluated each cycle while flat.",
        instruments=NSE_ONLY,
        backtestable=False,
        limitations=(_CHAIN_LIMITATION,),
    ),
    "IV Crush Premium Writer": StrategyRules(
        entry="ATM IV exceeds 20-bar realised volatility by at least min_iv_premium_pct_pts while |trend slope| <= max_trend_slope: sell one OTM option. Live only.",
        exit="Stop, target, or time exit; a mandatory premium stop applies (options/premium_model).",
        stop="ATR(14) x stop_atr_mult on the underlying, plus a mandatory premium stop.",
        target="ATR(14) x target_atr_mult on the underlying.",
        reentry="Evaluated each cycle while flat.",
        instruments=NSE_ONLY,
        backtestable=False,
        limitations=(_CHAIN_LIMITATION, "Short-premium: loss is not bounded by the stop alone in a gap; the premium stop is the backstop."),
    ),
    "Unusual Volume Fade": StrategyRules(
        entry="A statistically unusual volume spike within max_strikes_from_atm of the money: buy the cheaper opposite-side option. Live only.",
        exit="Stop, target, or time exit on the underlying, mapped to the option.",
        stop="ATR(14) x stop_atr_mult on the underlying.",
        target="ATR(14) x target_atr_mult on the underlying.",
        reentry="Evaluated each cycle while flat.",
        instruments=NSE_ONLY,
        backtestable=False,
        limitations=(_CHAIN_LIMITATION,),
    ),
}

# (low, high, step). Only parameters the logic actually reads; every default must lie inside its range.
_ATR_STOP = (0.5, 2.5, 0.25)
_ATR_TARGET = (1.0, 4.0, 0.5)
_COOLDOWN = (3, 20, 1)

PARAMETER_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    "Opening Range Breakout": {"stop_atr_mult": (0.5, 1.5, 0.25), "target_atr_mult": (1.0, 3.0, 0.5), "min_minutes_since_open": (31, 46, 5)},
    "Momentum": {"momentum_threshold": (0.0015, 0.006, 0.0005), "min_relative_volume": (1.0, 1.6, 0.1), "stop_atr_mult": _ATR_STOP,
                 "target_atr_mult": _ATR_TARGET, "cooldown_bars": _COOLDOWN},
    "VWAP Mean Reversion": {"stretch_threshold_pct": (0.2, 0.6, 0.05), "max_trend_slope": (0.0003, 0.0012, 0.0001), "stop_atr_mult": _ATR_STOP,
                            "cooldown_bars": _COOLDOWN},
    "RSI(2) Mean Reversion": {"oversold": (3, 20, 1), "overbought": (80, 97, 1), "stop_atr_mult": _ATR_STOP, "target_atr_mult": _ATR_TARGET,
                              "cooldown_bars": _COOLDOWN},
    "Bollinger Band Mean Reversion": {"max_trend_slope": (0.0003, 0.0012, 0.0001), "stop_atr_mult": _ATR_STOP, "cooldown_bars": _COOLDOWN},
    "Supertrend Trend Following": {"target_atr_mult": (1.5, 5.0, 0.5)},
    "EMA 50/200 Crossover": {"stop_atr_mult": (1.0, 3.0, 0.5), "target_atr_mult": (3.0, 9.0, 1.0)},
    "Donchian Channel Breakout": {"stop_atr_mult": (1.0, 3.0, 0.5), "target_atr_mult": (2.0, 6.0, 1.0), "cooldown_bars": (5, 30, 5)},
    "MACD Signal Crossover": {"stop_atr_mult": (0.75, 2.5, 0.25), "target_atr_mult": (1.5, 4.0, 0.5), "cooldown_bars": _COOLDOWN},
    "CPR Breakout": {"stop_atr_mult": _ATR_STOP, "target_atr_mult": _ATR_TARGET, "min_minutes_since_open": (16, 46, 5), "cooldown_bars": _COOLDOWN},
    "Camarilla Pivot Reversal": {"stop_atr_mult": (0.5, 1.5, 0.25), "min_minutes_since_open": (16, 46, 5), "cooldown_bars": _COOLDOWN},
    "Inside Bar Breakout": {"target_atr_mult": _ATR_TARGET, "cooldown_bars": (1, 10, 1)},
    "Gap Fill Reversion": {"min_gap_pct": (0.2, 0.5, 0.05), "max_gap_pct": (1.0, 2.5, 0.25), "max_minutes_since_open": (10, 40, 5),
                           "stop_atr_mult": _ATR_STOP},
    "PCR Extreme Contrarian": {"pcr_bullish_threshold": (1.2, 1.8, 0.1), "pcr_bearish_threshold": (0.3, 0.8, 0.1), "rsi_oversold": (10, 30, 5),
                               "rsi_overbought": (70, 90, 5), "stop_atr_mult": _ATR_STOP, "target_atr_mult": _ATR_TARGET},
    "OI Buildup Confirmation": {"min_oi_change_pct": (2.0, 10.0, 1.0), "stop_atr_mult": _ATR_STOP, "target_atr_mult": _ATR_TARGET},
    "IV Crush Premium Writer": {"min_iv_premium_pct_pts": (2.0, 8.0, 1.0), "max_trend_slope": (0.0002, 0.0008, 0.0001),
                                "stop_atr_mult": _ATR_STOP, "target_atr_mult": (0.5, 2.0, 0.25)},
    "Unusual Volume Fade": {"max_strikes_from_atm": (1, 3, 1), "stop_atr_mult": (0.5, 1.5, 0.25), "target_atr_mult": (0.5, 2.0, 0.25)},
}

# Bump a strategy's version whenever its signal, stop or target logic changes; every backtest records it.
STRATEGY_VERSIONS: dict[str, str] = {name: "1.0" for name in RULES}


def rules_for(name: str) -> StrategyRules | None:
    return RULES.get(name)


def ranges_for(name: str) -> dict[str, tuple[float, float, float]]:
    return PARAMETER_RANGES.get(name, {})


def version_for(name: str) -> str:
    return STRATEGY_VERSIONS.get(name, "0.0")
