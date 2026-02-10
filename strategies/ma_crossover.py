"""Moving Average Crossover Strategy with Volatility Regime Filter.

Combines trend following (MA crossover) with regime detection
(high/low volatility) to adapt position sizing.

In low-vol regimes:  Full size, standard MA crossover
In high-vol regimes: Half size or skip trades

Works on any timeframe — volatility annualization adapts to interval.
"""

from collections import defaultdict
from typing import Dict, Optional
import numpy as np

from core.events import EventBus, SignalEvent, SignalType


# Bars per trading day for each interval (US equity: 6.5hr/day)
_BARS_PER_DAY = {
    "1m": 390, "5m": 78, "15m": 26, "30m": 13,
    "1h": 7, "4h": 2, "1d": 1, "1wk": 0.2,
}

def _bars_per_year(interval: str) -> float:
    return _BARS_PER_DAY.get(interval, 1) * 252


class MACrossoverStrategy:
    """MA Crossover with volatility regime filter.

    Params:
        symbols:        List of symbols to trade
        fast_period:    Fast MA lookback (default 20)
        slow_period:    Slow MA lookback (default 50)
        vol_lookback:   Bars for realized vol calculation (default 30)
        vol_threshold:  Annualized vol above which regime is "high" (default 0.25)
        interval:       Bar interval for correct vol annualization (default "15m")
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        from strategies import BaseStrategy  # avoid circular at module level

        default_params = {
            "symbols": ["SPY"],
            "fast_period": 20,
            "slow_period": 50,
            "vol_lookback": 30,
            "vol_threshold": 0.25,   # Annualized vol threshold
            "interval": "15m",
        }
        merged = {**default_params, **(params or {})}

        self.name = "ma_crossover"
        self.event_bus = event_bus
        self.params = merged
        self._is_active = True

        # Internal state
        self._price_history: Dict[str, list] = defaultdict(list)
        self._bar_history: Dict[str, list] = defaultdict(list)
        self._prev_fast_ma: Dict[str, Optional[float]] = {}
        self._prev_slow_ma: Dict[str, Optional[float]] = {}
        self._in_position: Dict[str, bool] = {}

        # Signal funnel for diagnostics
        self._funnel = {
            "bars_processed": 0,
            "warmup_skip": 0,
            "no_crossover": 0,
            "bullish_crossover": 0,
            "bearish_crossover": 0,
            "already_in_position": 0,
            "not_in_position": 0,
            "entries_sent": 0,
            "exits_sent": 0,
            "regime_high": 0,
            "regime_low": 0,
        }

        # Subscribe
        from core.events import EventType, MarketDataEvent
        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

    def on_market_data(self, event) -> None:
        """Handle incoming bar."""
        if not self._is_active:
            return
        symbol = event.symbol
        self._price_history[symbol].append(event.close)
        self._bar_history[symbol].append({
            "timestamp": event.bar_timestamp or event.timestamp,
            "open": event.open, "high": event.high,
            "low": event.low, "close": event.close,
            "volume": event.volume,
        })
        signal = self.calculate_signal(symbol)
        if signal is not None:
            self.event_bus.publish(signal)

    def get_prices(self, symbol: str) -> np.ndarray:
        return np.array(self._price_history.get(symbol, []))

    # --- Core indicators ---

    def _moving_average(self, prices: np.ndarray, period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return float(prices[-period:].mean())

    def _realized_volatility(self, prices: np.ndarray, lookback: int) -> Optional[float]:
        """Compute annualized realized volatility, adapting to the bar interval."""
        if len(prices) < lookback + 1:
            return None
        log_returns = np.diff(np.log(prices[-lookback - 1:]))
        # Annualize using the correct factor for this interval
        bpy = _bars_per_year(self.params.get("interval", "15m"))
        return float(log_returns.std() * np.sqrt(bpy))

    def _get_regime(self, symbol: str) -> str:
        """Determine volatility regime: 'low' or 'high'."""
        prices = self.get_prices(symbol)
        vol = self._realized_volatility(prices, self.params["vol_lookback"])
        if vol is None:
            return "unknown"
        regime = "high" if vol > self.params["vol_threshold"] else "low"
        if regime == "high":
            self._funnel["regime_high"] += 1
        else:
            self._funnel["regime_low"] += 1
        return regime

    # --- Signal generation ---

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Generate MA crossover signal with regime filter."""
        if symbol not in self.params.get("symbols", []):
            return None

        self._funnel["bars_processed"] += 1

        prices = self.get_prices(symbol)
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        fast_ma = self._moving_average(prices, fast_period)
        slow_ma = self._moving_average(prices, slow_period)

        if fast_ma is None or slow_ma is None:
            self._funnel["warmup_skip"] += 1
            return None

        prev_fast = self._prev_fast_ma.get(symbol)
        prev_slow = self._prev_slow_ma.get(symbol)

        # Update for next iteration
        self._prev_fast_ma[symbol] = fast_ma
        self._prev_slow_ma[symbol] = slow_ma

        if prev_fast is None or prev_slow is None:
            self._funnel["warmup_skip"] += 1
            return None

        regime = self._get_regime(symbol)
        in_pos = self._in_position.get(symbol, False)

        # --- Bullish crossover: fast crosses above slow ---
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            self._funnel["bullish_crossover"] += 1
            if not in_pos:
                strength = 1.0 if regime == "low" else 0.5
                self._in_position[symbol] = True
                self._funnel["entries_sent"] += 1
                return SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    metadata={
                        "fast_ma": round(fast_ma, 4),
                        "slow_ma": round(slow_ma, 4),
                        "regime": regime,
                        "crossover": "bullish",
                    },
                )
            else:
                self._funnel["already_in_position"] += 1

        # --- Bearish crossover: fast crosses below slow ---
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            self._funnel["bearish_crossover"] += 1
            if in_pos:
                self._in_position[symbol] = False
                self._funnel["exits_sent"] += 1
                return SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    strength=0.0,
                    strategy_name=self.name,
                    metadata={
                        "fast_ma": round(fast_ma, 4),
                        "slow_ma": round(slow_ma, 4),
                        "regime": regime,
                        "crossover": "bearish",
                    },
                )
            else:
                self._funnel["not_in_position"] += 1
        else:
            self._funnel["no_crossover"] += 1

        return None

    # --- Diagnostics ---

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
        diagnostics["_funnel"] = self._funnel
        return diagnostics

    def print_funnel(self):
        """Print the signal funnel — shows where signals die at each stage."""
        f = self._funnel
        print(f"\n  SIGNAL FUNNEL — {self.name}")
        print(f"  {'─'*50}")
        print(f"  Bars processed:           {f['bars_processed']:>8,}")
        print(f"    ├─ Warmup (not enough): {f['warmup_skip']:>8,}")
        print(f"    ├─ No crossover:        {f['no_crossover']:>8,}")
        print(f"    ├─ Bullish crossover:   {f['bullish_crossover']:>8,}")
        print(f"    │   ├─ Already in pos:  {f['already_in_position']:>8,}")
        print(f"    │   └─ ENTRY sent:      {f['entries_sent']:>8,}")
        print(f"    └─ Bearish crossover:   {f['bearish_crossover']:>8,}")
        print(f"        ├─ Not in pos:      {f['not_in_position']:>8,}")
        print(f"        └─ EXIT sent:       {f['exits_sent']:>8,}")
        print(f"  {'─'*50}")
        print(f"  Vol regime — low:         {f['regime_low']:>8,}")
        print(f"  Vol regime — high:        {f['regime_high']:>8,}")
        print(f"  {'─'*50}")

    def stop(self):
        self._is_active = False

    def start(self):
        self._is_active = True