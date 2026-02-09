"""Trading strategies module.

All strategies inherit from BaseStrategy and implement:
- on_market_data(): Process new bar data
- generate_signals(): Produce trading signals

Strategies are decoupled from execution - they only produce signals.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from core.events import (
    EventBus, EventType, MarketDataEvent, SignalEvent,
    SignalType, OrderEvent, OrderType, OrderSide,
)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Strategies consume MarketDataEvents and produce SignalEvents.
    They maintain their own internal state (price history, indicators).
    """

    def __init__(self, name: str, event_bus: EventBus, params: dict = None):
        self.name = name
        self.event_bus = event_bus
        self.params = params or {}
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._bar_history: Dict[str, List[dict]] = defaultdict(list)
        self._current_signal: Dict[str, SignalType] = {}
        self._is_active = True

        # Subscribe to market data
        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

    @abstractmethod
    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Core signal logic. Override this in subclasses."""
        pass

    def on_market_data(self, event: MarketDataEvent) -> None:
        """Handle incoming market data bar."""
        if not self._is_active:
            return

        symbol = event.symbol

        # Store price history
        self._price_history[symbol].append(event.close)
        self._bar_history[symbol].append({
            "timestamp": event.bar_timestamp or event.timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        })

        # Generate signal
        signal = self.calculate_signal(symbol)
        if signal is not None:
            self.event_bus.publish(signal)

    def get_prices(self, symbol: str, lookback: Optional[int] = None) -> np.ndarray:
        """Get price history as numpy array."""
        prices = self._price_history.get(symbol, [])
        if lookback:
            prices = prices[-lookback:]
        return np.array(prices)

    def get_bars_df(self, symbol: str, lookback: Optional[int] = None) -> pd.DataFrame:
        """Get bar history as DataFrame."""
        bars = self._bar_history.get(symbol, [])
        if lookback:
            bars = bars[-lookback:]
        return pd.DataFrame(bars)

    def stop(self):
        self._is_active = False

    def start(self):
        self._is_active = True


class PairsTradingStrategy(BaseStrategy):
    """Mean-reversion pairs trading strategy.

    Trades the spread between two cointegrated instruments.
    Uses z-score of the spread to generate entry/exit signals.

    This is a classic quant strategy that teaches:
    - Cointegration testing
    - Spread construction & hedge ratios
    - Z-score normalization
    - Mean reversion dynamics
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "pair": ["SPY", "IWM"],
            "lookback_period": 60,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.5,
            "hedge_ratio_method": "ols",  # ols or rolling
        }
        merged = {**default_params, **(params or {})}
        super().__init__("pairs_trading", event_bus, merged)

        self.symbol_a = self.params["pair"][0]
        self.symbol_b = self.params["pair"][1]
        self._hedge_ratio = None
        self._spread_history: List[float] = []
        self._z_scores: List[float] = []
        self._in_position = False
        self._position_side = None  # "long_spread" or "short_spread"

    def _calculate_hedge_ratio(self) -> Optional[float]:
        """Calculate hedge ratio using OLS regression."""
        lookback = self.params["lookback_period"]
        prices_a = self.get_prices(self.symbol_a, lookback)
        prices_b = self.get_prices(self.symbol_b, lookback)

        if len(prices_a) < lookback or len(prices_b) < lookback:
            return None

        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        if self.params["hedge_ratio_method"] == "ols":
            # OLS: A = beta * B + alpha + epsilon
            slope, intercept, r_value, p_value, std_err = stats.linregress(prices_b, prices_a)
            return slope
        else:
            # Rolling ratio
            return prices_a[-1] / prices_b[-1]

    def _calculate_spread(self) -> Optional[float]:
        """Calculate the spread: A - hedge_ratio * B."""
        prices_a = self.get_prices(self.symbol_a)
        prices_b = self.get_prices(self.symbol_b)

        if len(prices_a) == 0 or len(prices_b) == 0 or self._hedge_ratio is None:
            return None

        return prices_a[-1] - self._hedge_ratio * prices_b[-1]

    def _calculate_z_score(self) -> Optional[float]:
        """Z-score of current spread vs historical spread."""
        lookback = self.params["lookback_period"]
        if len(self._spread_history) < lookback:
            return None

        recent = np.array(self._spread_history[-lookback:])
        mean = recent.mean()
        std = recent.std()

        if std == 0:
            return 0.0

        return (self._spread_history[-1] - mean) / std

    def _check_cointegration(self) -> Optional[float]:
        """Run Engle-Granger cointegration test. Returns p-value."""
        lookback = self.params["lookback_period"]
        prices_a = self.get_prices(self.symbol_a, lookback)
        prices_b = self.get_prices(self.symbol_b, lookback)

        if len(prices_a) < lookback or len(prices_b) < lookback:
            return None

        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # OLS residuals
        slope, intercept, _, _, _ = stats.linregress(prices_b, prices_a)
        residuals = prices_a - (slope * prices_b + intercept)

        # ADF test on residuals (simplified - check if residuals are stationary)
        # For full implementation, use statsmodels.tsa.stattools.adfuller
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(residuals, maxlag=1, regression="c")
            return result[1]  # p-value
        except ImportError:
            # Simplified check: autocorrelation of residuals
            if len(residuals) > 1:
                autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                return 1.0 if autocorr > 0.9 else 0.05  # Rough estimate
            return None

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Generate pairs trading signal.

        Only generate signals when we have data for both symbols
        on the same bar (both must have updated).
        """
        # Only trigger on symbol_a updates (avoid double signals)
        if symbol != self.symbol_a:
            return None

        # Need both symbols to have data
        if not self._price_history.get(self.symbol_b):
            return None

        # Recalculate hedge ratio periodically
        if self._hedge_ratio is None or len(self._spread_history) % 20 == 0:
            self._hedge_ratio = self._calculate_hedge_ratio()

        if self._hedge_ratio is None:
            return None

        # Calculate spread and z-score
        spread = self._calculate_spread()
        if spread is None:
            return None

        self._spread_history.append(spread)
        z_score = self._calculate_z_score()

        if z_score is None:
            return None

        self._z_scores.append(z_score)

        entry_z = self.params["entry_z"]
        exit_z = self.params["exit_z"]
        stop_z = self.params["stop_z"]

        signal = None

        if not self._in_position:
            # Entry conditions
            if z_score > entry_z:
                # Spread is too high -> short spread (sell A, buy B)
                signal = SignalEvent(
                    symbol=self.symbol_a,
                    signal_type=SignalType.SHORT,
                    strength=min(abs(z_score) / stop_z, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "z_score": z_score,
                        "hedge_ratio": self._hedge_ratio,
                        "spread": spread,
                        "pair": [self.symbol_a, self.symbol_b],
                        "action": "short_spread",
                    },
                )
                self._in_position = True
                self._position_side = "short_spread"

            elif z_score < -entry_z:
                # Spread is too low -> long spread (buy A, sell B)
                signal = SignalEvent(
                    symbol=self.symbol_a,
                    signal_type=SignalType.LONG,
                    strength=min(abs(z_score) / stop_z, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "z_score": z_score,
                        "hedge_ratio": self._hedge_ratio,
                        "spread": spread,
                        "pair": [self.symbol_a, self.symbol_b],
                        "action": "long_spread",
                    },
                )
                self._in_position = True
                self._position_side = "long_spread"

        else:
            # Exit conditions
            should_exit = False

            if self._position_side == "short_spread":
                if z_score < exit_z:  # Mean reverted
                    should_exit = True
                elif z_score > stop_z:  # Stop loss
                    should_exit = True

            elif self._position_side == "long_spread":
                if z_score > -exit_z:  # Mean reverted
                    should_exit = True
                elif z_score < -stop_z:  # Stop loss
                    should_exit = True

            if should_exit:
                exit_type = (
                    SignalType.EXIT_SHORT if self._position_side == "short_spread"
                    else SignalType.EXIT_LONG
                )
                signal = SignalEvent(
                    symbol=self.symbol_a,
                    signal_type=exit_type,
                    strength=0.0,
                    strategy_name=self.name,
                    metadata={
                        "z_score": z_score,
                        "spread": spread,
                        "action": "exit",
                    },
                )
                self._in_position = False
                self._position_side = None

        return signal

    def get_diagnostics(self) -> Dict:
        """Return current strategy state for debugging/dashboard."""
        return {
            "hedge_ratio": self._hedge_ratio,
            "spread": self._spread_history[-1] if self._spread_history else None,
            "z_score": self._z_scores[-1] if self._z_scores else None,
            "in_position": self._in_position,
            "position_side": self._position_side,
            "spread_history_len": len(self._spread_history),
        }


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover with Volatility Regime Filter.

    Combines trend following (MA crossover) with regime detection
    (high/low volatility) to adapt behavior.

    In low-vol regimes: Standard MA crossover
    In high-vol regimes: Tighten stops, reduce position size, or skip trades
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["SPY"],
            "fast_period": 20,
            "slow_period": 50,
            "vol_lookback": 30,
            "vol_threshold": 0.20,  # Annualized vol threshold
        }
        merged = {**default_params, **(params or {})}
        super().__init__("ma_crossover", event_bus, merged)

        self._prev_fast_ma: Dict[str, Optional[float]] = {}
        self._prev_slow_ma: Dict[str, Optional[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def _moving_average(self, prices: np.ndarray, period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return prices[-period:].mean()

    def _realized_volatility(self, prices: np.ndarray, lookback: int) -> Optional[float]:
        if len(prices) < lookback + 1:
            return None
        returns = np.diff(np.log(prices[-lookback - 1:])) 
        return returns.std() * np.sqrt(252)  # Annualized

    def _get_regime(self, symbol: str) -> str:
        """Determine volatility regime: 'low' or 'high'."""
        prices = self.get_prices(symbol)
        vol = self._realized_volatility(prices, self.params["vol_lookback"])
        if vol is None:
            return "unknown"
        return "high" if vol > self.params["vol_threshold"] else "low"

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Generate MA crossover signal with regime filter."""
        if symbol not in self.params.get("symbols", []):
            return None

        prices = self.get_prices(symbol)
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        fast_ma = self._moving_average(prices, fast_period)
        slow_ma = self._moving_average(prices, slow_period)

        if fast_ma is None or slow_ma is None:
            return None

        prev_fast = self._prev_fast_ma.get(symbol)
        prev_slow = self._prev_slow_ma.get(symbol)

        # Update for next iteration
        self._prev_fast_ma[symbol] = fast_ma
        self._prev_slow_ma[symbol] = slow_ma

        if prev_fast is None or prev_slow is None:
            return None

        regime = self._get_regime(symbol)
        in_pos = self._in_position.get(symbol, False)

        signal = None

        # Bullish crossover: fast crosses above slow
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            if not in_pos:
                strength = 1.0 if regime == "low" else 0.5  # Reduce size in high-vol
                signal = SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    metadata={
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "regime": regime,
                        "crossover": "bullish",
                    },
                )
                self._in_position[symbol] = True

        # Bearish crossover: fast crosses below slow
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            if in_pos:
                signal = SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    strength=0.0,
                    strategy_name=self.name,
                    metadata={
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "regime": regime,
                        "crossover": "bearish",
                    },
                )
                self._in_position[symbol] = False

        return signal

    def get_diagnostics(self) -> Dict:
        diagnostics = {}
        for symbol in self.params.get("symbols", []):
            prices = self.get_prices(symbol)
            diagnostics[symbol] = {
                "fast_ma": self._moving_average(prices, self.params["fast_period"]),
                "slow_ma": self._moving_average(prices, self.params["slow_period"]),
                "regime": self._get_regime(symbol),
                "in_position": self._in_position.get(symbol, False),
                "price_count": len(prices),
            }
        return diagnostics


# Import BOS strategy
from strategies.bos_strategy import BOSStrategy

# Strategy registry for easy lookup
STRATEGY_REGISTRY = {
    "pairs_trading": PairsTradingStrategy,
    "ma_crossover": MACrossoverStrategy,
    "bos": BOSStrategy,
}


def create_strategy(name: str, event_bus: EventBus, params: dict = None) -> BaseStrategy:
    """Factory function to create strategies by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](event_bus, params)