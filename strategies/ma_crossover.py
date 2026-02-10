"""Moving Average Crossover Strategy — Long AND Short.

Always-in-market trend following:
  Fast MA > Slow MA  →  be LONG
  Fast MA < Slow MA  →  be SHORT

Position flips take 2 bars for safety (exit bar + entry bar) because
the event queue processes signals before fills resolve. Sending EXIT + ENTRY
on the same bar causes the entry to be sized incorrectly.

Volatility regime scales position size:
  Low vol  → full size (strength=1.0)
  High vol → half size (strength=0.5)
"""

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType

_BARS_PER_DAY = {
    "1m": 390, "2m": 195, "3m": 130,
    "5m": 78, "10m": 39, "15m": 26,
    "20m": 20, "30m": 13,
    "1h": 7, "2h": 3, "3h": 2, "4h": 2, "8h": 1,
    "1d": 1, "1wk": 0.2,
}

def _bars_per_year(interval: str) -> float:
    return _BARS_PER_DAY.get(interval, 1) * 252


class MACrossoverStrategy:
    """MA Crossover — always-in-market long/short strategy."""

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["SPY"],
            "fast_period": 20,
            "slow_period": 50,
            "vol_lookback": 30,
            "vol_threshold": 0.25,
            "interval": "15m",
        }
        self.name = "ma_crossover"
        self.event_bus = event_bus
        self.params = {**default_params, **(params or {})}
        self._is_active = True

        self._price_history: Dict[str, list] = defaultdict(list)
        self._bar_history: Dict[str, list] = defaultdict(list)

        # "long", "short", or None (flat, between exit and next entry)
        self._position: Dict[str, Optional[str]] = {}

        self._funnel = {
            "bars_processed": 0,
            "warmup_skip": 0,
            "no_change": 0,
            "long_entries": 0,
            "short_entries": 0,
            "long_exits": 0,
            "short_exits": 0,
            "regime_high": 0,
            "regime_low": 0,
        }

        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

    def on_market_data(self, event: MarketDataEvent) -> None:
        if not self._is_active:
            return
        symbol = event.symbol
        if symbol not in self.params.get("symbols", []):
            return

        self._price_history[symbol].append(event.close)
        self._bar_history[symbol].append({
            "timestamp": event.bar_timestamp or event.timestamp,
            "open": event.open, "high": event.high,
            "low": event.low, "close": event.close,
            "volume": event.volume,
        })

        signal = self._decide(symbol)
        if signal is not None:
            self.event_bus.publish(signal)

    def _decide(self, symbol: str) -> Optional[SignalEvent]:
        """One signal per bar. Exit on flip bar, enter on next bar."""
        self._funnel["bars_processed"] += 1

        prices = np.array(self._price_history[symbol])
        fast_ma = self._ma(prices, self.params["fast_period"])
        slow_ma = self._ma(prices, self.params["slow_period"])

        if fast_ma is None or slow_ma is None:
            self._funnel["warmup_skip"] += 1
            return None

        desired = "long" if fast_ma > slow_ma else "short"
        current = self._position.get(symbol)

        regime = self._get_regime(symbol)
        strength = 1.0 if regime == "low" else 0.5

        meta = {
            "fast_ma": round(fast_ma, 4),
            "slow_ma": round(slow_ma, 4),
            "regime": regime,
            "desired": desired,
            "current": current or "flat",
        }

        # Already correctly positioned
        if desired == current:
            self._funnel["no_change"] += 1
            return None

        # Need to exit first before reversing (flip takes 2 bars)
        if current == "long" and desired == "short":
            self._position[symbol] = None
            self._funnel["long_exits"] += 1
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.EXIT_LONG,
                strength=0.0, strategy_name=self.name,
                metadata={**meta, "action": "exit_long_for_flip"},
            )

        if current == "short" and desired == "long":
            self._position[symbol] = None
            self._funnel["short_exits"] += 1
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.EXIT_SHORT,
                strength=0.0, strategy_name=self.name,
                metadata={**meta, "action": "exit_short_for_flip"},
            )

        # Flat → enter new position
        if current is None and desired == "long":
            self._position[symbol] = "long"
            self._funnel["long_entries"] += 1
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.LONG,
                strength=strength, strategy_name=self.name,
                metadata={**meta, "action": "enter_long"},
            )

        if current is None and desired == "short":
            self._position[symbol] = "short"
            self._funnel["short_entries"] += 1
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.SHORT,
                strength=strength, strategy_name=self.name,
                metadata={**meta, "action": "enter_short"},
            )

        self._funnel["no_change"] += 1
        return None

    def _ma(self, prices: np.ndarray, period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return float(prices[-period:].mean())

    def _get_regime(self, symbol: str) -> str:
        prices = np.array(self._price_history.get(symbol, []))
        lb = self.params["vol_lookback"]
        if len(prices) < lb + 1:
            return "unknown"
        log_ret = np.diff(np.log(prices[-lb - 1:]))
        bpy = _bars_per_year(self.params.get("interval", "15m"))
        vol = float(log_ret.std() * np.sqrt(bpy))
        regime = "high" if vol > self.params["vol_threshold"] else "low"
        self._funnel[f"regime_{regime}"] += 1
        return regime

    def get_prices(self, symbol: str) -> np.ndarray:
        return np.array(self._price_history.get(symbol, []))

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        return None

    def get_diagnostics(self) -> Dict:
        diag = {}
        for sym in self.params.get("symbols", []):
            prices = self.get_prices(sym)
            diag[sym] = {
                "fast_ma": self._ma(prices, self.params["fast_period"]),
                "slow_ma": self._ma(prices, self.params["slow_period"]),
                "position": self._position.get(sym, "flat"),
                "bars": len(prices),
            }
        diag["_funnel"] = self._funnel
        return diag

    def print_funnel(self):
        f = self._funnel
        total_entries = f["long_entries"] + f["short_entries"]
        total_exits = f["long_exits"] + f["short_exits"]
        total_signals = total_entries + total_exits
        print(f"\n  SIGNAL FUNNEL — {self.name}")
        print(f"  {'─'*50}")
        print(f"  Bars processed:           {f['bars_processed']:>8,}")
        print(f"    ├─ Warmup:              {f['warmup_skip']:>8,}")
        print(f"    ├─ Already positioned:  {f['no_change']:>8,}")
        print(f"    ├─ Enter LONG:          {f['long_entries']:>8,}")
        print(f"    ├─ Enter SHORT:         {f['short_entries']:>8,}")
        print(f"    ├─ Exit long (flip):    {f['long_exits']:>8,}")
        print(f"    └─ Exit short (flip):   {f['short_exits']:>8,}")
        print(f"  {'─'*50}")
        print(f"  Total signals:            {total_signals:>8,}  ({total_entries} entries + {total_exits} exits)")
        print(f"  Vol regime — low:         {f['regime_low']:>8,}")
        print(f"  Vol regime — high:        {f['regime_high']:>8,}")
        print(f"  {'─'*50}")

    def stop(self):
        self._is_active = False

    def start(self):
        self._is_active = True