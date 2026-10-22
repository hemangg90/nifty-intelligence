"""
Standard strategy interface.

Every strategy in the library implements this same interface so the ranking
engine, backtester, and setup detector can treat all strategies uniformly.
Strategies NEVER place broker orders directly - they only produce setups /
signals / entry-stop-target proposals that flow through the risk engine.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Setup:
    """A single historical or live occurrence of a strategy's entry conditions being met."""

    timestamp: pd.Timestamp
    direction: str  # LONG / SHORT
    entry_price: float
    stop_price: float
    target_price: float
    meta: dict = field(default_factory=dict)
    invalidation: str = ""  # human-readable condition under which the setup is void before entry
    strategy_version: str = ""

    @property
    def signal_timestamp(self) -> pd.Timestamp:
        """The bar whose close produced this signal (the decision time)."""
        return self.timestamp

    def to_record(self, strategy: "BaseStrategy", instrument: str, timeframe: str = "5min") -> dict:
        """Standard, strategy-independent description of this setup (the shared strategy output schema)."""
        return {
            "strategy": strategy.name,
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": self.timestamp,
            "signal": self.direction,
            "entry_condition_met": True,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_price,
            "target": self.target_price,
            "invalidation_condition": self.invalidation or strategy.invalidation_condition(self),
            "strategy_version": self.strategy_version or strategy.version,
        }


@dataclass
class TradeResult:
    setup: Setup
    exit_timestamp: pd.Timestamp | None
    exit_price: float | None
    r_multiple: float | None
    holding_period_bars: int
    mfe: float
    mae: float
    gross_pnl: float
    outcome: str  # WIN / LOSS / SCRATCH / OPEN


class BaseStrategy(abc.ABC):
    name: str = "base"
    description: str = ""
    required_features: list[str] = []
    default_parameters: dict = {}

    # Chain-aware strategies (see quant_intelligence/options/) set this to the
    # ChainSnapshot-derived fields they read (e.g. "total_pcr", "atm_iv") and
    # implement check_chain_setup below. Pure price-action strategies leave
    # this empty and are never asked for a chain-based setup.
    required_chain_features: list[str] = []
    _chain_history = None  # options.historical_chain.HistoricalChain, attached for research runs only

    def attach_chain_history(self, chain_history) -> "BaseStrategy":
        """Give a chain-aware strategy rebuilt option-chain history so `generate_historical_setups` can replay it. Without
        it (the default, and always in live trading) such a strategy produces no historical setups."""
        self._chain_history = chain_history
        return self

    def historical_chain_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list["Setup"]:
        if self._chain_history is None:
            return []
        from quant_intelligence.options.historical_chain import generate_chain_setups

        return generate_chain_setups(self, ohlcv, features, self._chain_history)

    @property
    def version(self) -> str:
        from quant_intelligence.strategies.rules import version_for

        return version_for(self.name)

    @property
    def rules(self):
        """StrategyRules from strategies/rules.py (None for a strategy that has not documented its rules)."""
        from quant_intelligence.strategies.rules import rules_for

        return rules_for(self.name)

    @property
    def parameter_ranges(self) -> dict:
        from quant_intelligence.strategies.rules import ranges_for

        return ranges_for(self.name)

    def invalidation_condition(self, setup: "Setup") -> str:
        """Default: the setup is void if the stop level trades before the entry fills."""
        side = "below" if setup.direction == "LONG" else "above"
        return f"price trades {side} the stop {setup.stop_price:.2f} before entry"

    def check_chain_setup(self, market_state, recent_ohlcv, chain) -> Setup | None:
        """Chain-aware strategies override this to combine option-chain analytics
        (a ChainSnapshot from quant_intelligence.options.chain_analytics) with price
        action. Default: no chain-based setup (pure price-action strategy)."""
        return None

    # Parameters that set how far the stop and target sit from the entry; STOP_ATR_SCALE multiplies their DEFAULTS.
    SCALED_PARAMETERS = ("stop_atr_mult", "target_atr_mult")

    def __init__(self, parameters: dict | None = None):
        from quant_intelligence.config.settings import SETTINGS

        defaults = dict(self.default_parameters)
        scale = SETTINGS.stop_atr_scale
        if scale != 1.0:
            for key in self.SCALED_PARAMETERS:
                if key in defaults:
                    defaults[key] = round(defaults[key] * scale, 4)
        # Explicit parameters (a user's own values) always win and are never scaled.
        self.parameters = {**defaults, **(parameters or {})}
        self.validate_parameters(self.parameters)

    def validate_parameters(self, parameters: dict) -> None:
        """Override to raise ValueError on invalid parameter combinations."""
        return None

    @abc.abstractmethod
    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        """Scan historical OHLCV+features and return every historical setup occurrence.

        `ohlcv` and `features` must be aligned (same length/index, features computed
        with no look-ahead). Implementations must only use data up to bar i when
        deciding whether a setup occurs at bar i.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        """Check whether the strategy's entry conditions are met RIGHT NOW, live/paper context."""
        raise NotImplementedError

    def generate_signal(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        """Default: a signal is just a live setup. Strategies can override for extra filtering."""
        return self.check_setup(market_state, recent_ohlcv)

    def calculate_entry_stop_target(self, direction: str, reference_price: float, atr: float) -> tuple[float, float, float]:
        """Default ATR-based stop/target; strategies may override with their own logic."""
        stop_mult = self.parameters.get("stop_atr_mult", 1.0)
        target_mult = self.parameters.get("target_atr_mult", 2.0)
        if direction == "LONG":
            stop = reference_price - stop_mult * atr
            target = reference_price + target_mult * atr
        else:
            stop = reference_price + stop_mult * atr
            target = reference_price - target_mult * atr
        return reference_price, stop, target

    def simulate_trade(self, setup: Setup, ohlcv: pd.DataFrame, entry_idx: int, max_holding_bars: int = 75,
                       execution_timing: str | None = None) -> TradeResult:
        """Bar-by-bar forward simulation from entry_idx: whichever of stop/target is hit first wins.
        If neither is hit within max_holding_bars, exit at the close of the last bar (time-based exit).
        This is intentionally simple and conservative (no intra-bar look-ahead: within a bar we
        check stop before target for adverse conservatism on longs, and vice versa for shorts).
        """
        from quant_intelligence.config.settings import SETTINGS

        if (execution_timing or SETTINGS.execution_timing) == "next_open":
            return self._simulate_next_open(setup, ohlcv, entry_idx, max_holding_bars)
        direction = setup.direction
        n = len(ohlcv)
        # Entry price is the CLOSE of entry_idx, so monitoring must start at the NEXT
        # bar - checking entry_idx's own high/low against a stop derived from its close
        # would double-count that bar's intrabar range and produce spuriously high
        # stop-out rates.
        monitor_start_idx = entry_idx + 1
        end_idx = min(entry_idx + max_holding_bars, n - 1)

        mfe = 0.0
        mae = 0.0
        exit_price = None
        exit_ts = None
        outcome = "OPEN"

        risk_per_unit = abs(setup.entry_price - setup.stop_price)
        if risk_per_unit == 0:
            risk_per_unit = 1e-6

        if monitor_start_idx > end_idx:
            last_bar = ohlcv.iloc[entry_idx]
            pnl_dir = 1 if direction == "LONG" else -1
            gross_pnl = pnl_dir * (last_bar["close"] - setup.entry_price)
            return TradeResult(
                setup=setup,
                exit_timestamp=last_bar["timestamp"],
                exit_price=last_bar["close"],
                r_multiple=gross_pnl / risk_per_unit,
                holding_period_bars=0,
                mfe=0.0,
                mae=0.0,
                gross_pnl=gross_pnl,
                outcome="WIN" if gross_pnl > 0 else ("LOSS" if gross_pnl < 0 else "SCRATCH"),
            )

        for idx in range(monitor_start_idx, end_idx + 1):
            bar = ohlcv.iloc[idx]
            if direction == "LONG":
                favorable = bar["high"] - setup.entry_price
                adverse = setup.entry_price - bar["low"]
            else:
                favorable = setup.entry_price - bar["low"]
                adverse = bar["high"] - setup.entry_price

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            if direction == "LONG":
                hit_stop = bar["low"] <= setup.stop_price
                hit_target = bar["high"] >= setup.target_price
            else:
                hit_stop = bar["high"] >= setup.stop_price
                hit_target = bar["low"] <= setup.target_price

            if hit_stop and hit_target:
                # Conservative assumption: stop is hit first within the bar.
                exit_price = setup.stop_price
                outcome = "LOSS"
                exit_ts = bar["timestamp"]
                break
            if hit_stop:
                exit_price = setup.stop_price
                outcome = "LOSS"
                exit_ts = bar["timestamp"]
                break
            if hit_target:
                exit_price = setup.target_price
                outcome = "WIN"
                exit_ts = bar["timestamp"]
                break

        if exit_price is None:
            # Time-based exit at the close of the last bar in the window.
            last_bar = ohlcv.iloc[end_idx]
            exit_price = last_bar["close"]
            exit_ts = last_bar["timestamp"]
            pnl_dir = 1 if direction == "LONG" else -1
            gross_pnl_check = pnl_dir * (exit_price - setup.entry_price)
            outcome = "WIN" if gross_pnl_check > 0 else ("LOSS" if gross_pnl_check < 0 else "SCRATCH")

        pnl_dir = 1 if direction == "LONG" else -1
        gross_pnl = pnl_dir * (exit_price - setup.entry_price)
        r_multiple = gross_pnl / risk_per_unit

        return TradeResult(
            setup=setup,
            exit_timestamp=exit_ts,
            exit_price=exit_price,
            r_multiple=r_multiple,
            holding_period_bars=end_idx - entry_idx,
            mfe=mfe,
            mae=mae,
            gross_pnl=gross_pnl,
            outcome=outcome,
        )

    def _simulate_next_open(self, setup: Setup, ohlcv: pd.DataFrame, signal_idx: int, max_holding_bars: int) -> TradeResult:
        """Fill at the NEXT bar's open, then walk forward from that same bar (its range is after the fill).

        The signal is only known once bar `signal_idx` has closed, so the earliest achievable fill is the next
        open. Stop and target stay at their absolute signal-time levels; R is measured from the actual fill.
        Within a bar the stop is checked before the target. An open position is flattened at the session's last
        bar, because the strategies are traded intraday. Callers must pre-check `next_open_fill`.
        """
        fill = next_open_fill(setup, ohlcv, signal_idx)
        if fill is None:
            raise ValueError("signal has no executable next-open fill; check next_open_fill() first")
        fill_idx = signal_idx + 1
        sign = 1 if setup.direction == "LONG" else -1
        filled = Setup(
            timestamp=setup.timestamp, direction=setup.direction, entry_price=fill, stop_price=setup.stop_price,
            target_price=setup.target_price, invalidation=setup.invalidation, strategy_version=setup.strategy_version,
            meta={**(setup.meta or {}), "execution": "next_open", "signal_price": setup.entry_price,
                  "fill_timestamp": ohlcv["timestamp"].iloc[fill_idx]},
        )
        risk_per_unit = abs(fill - setup.stop_price) or 1e-6

        day = pd.to_datetime(ohlcv["timestamp"]).dt.date
        last_idx = fill_idx
        limit = min(signal_idx + max_holding_bars, len(ohlcv) - 1)
        while last_idx < limit and day.iloc[last_idx + 1] == day.iloc[fill_idx]:
            last_idx += 1

        mfe = mae = 0.0
        exit_idx = last_idx
        exit_price = None
        outcome = "OPEN"
        for idx in range(fill_idx, last_idx + 1):
            bar = ohlcv.iloc[idx]
            mfe = max(mfe, sign * ((bar["high"] if sign == 1 else bar["low"]) - fill))
            mae = max(mae, sign * (fill - (bar["low"] if sign == 1 else bar["high"])))
            hit_stop = bar["low"] <= setup.stop_price if sign == 1 else bar["high"] >= setup.stop_price
            hit_target = bar["high"] >= setup.target_price if sign == 1 else bar["low"] <= setup.target_price
            if hit_stop:  # also when both are hit in one bar: assume the adverse move came first
                exit_price, outcome, exit_idx = setup.stop_price, "LOSS", idx
                break
            if hit_target:
                exit_price, outcome, exit_idx = setup.target_price, "WIN", idx
                break
        if exit_price is None:
            exit_price = float(ohlcv["close"].iloc[last_idx])
            pnl = sign * (exit_price - fill)
            outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "SCRATCH")

        gross_pnl = sign * (exit_price - fill)
        return TradeResult(
            setup=filled, exit_timestamp=ohlcv["timestamp"].iloc[exit_idx], exit_price=exit_price,
            r_multiple=gross_pnl / risk_per_unit, holding_period_bars=exit_idx - signal_idx,
            mfe=mfe, mae=mae, gross_pnl=gross_pnl, outcome=outcome,
        )


def next_open_fill(setup: Setup, ohlcv: pd.DataFrame, signal_idx: int) -> float | None:
    """The achievable fill for a signal on bar `signal_idx`: the next bar's open, or None when there is no
    honest fill. None means: no next bar, the next bar belongs to another session (an intraday signal is not
    carried overnight), the open is already through the stop or target, or the stop distance at the open has
    collapsed to under half the planned risk (R would be inflated by a barely-valid entry)."""
    if signal_idx + 1 >= len(ohlcv):
        return None
    ts = pd.to_datetime(ohlcv["timestamp"])
    if ts.iloc[signal_idx + 1].date() != ts.iloc[signal_idx].date():
        return None
    fill = float(ohlcv["open"].iloc[signal_idx + 1])
    if setup.direction == "LONG":
        if fill <= setup.stop_price or fill >= setup.target_price:
            return None
    elif fill >= setup.stop_price or fill <= setup.target_price:
        return None
    planned = abs(setup.entry_price - setup.stop_price)
    if planned > 0 and abs(fill - setup.stop_price) < 0.5 * planned:
        return None
    return fill
