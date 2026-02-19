"""Moving Average Crossover Strategy v2 — With Stop/Target.

v1 Problem: held trades forever waiting for MA re-cross. A $400 winner
would turn into a -$160 loser because the MAs are slow to react.

v2 Fix: ATR-based stop loss and take profit on every trade.
  - Stop: 2 ATR below entry (long) or above entry (short)
  - Target: 3 ATR in profit direction (1.5:1 risk-reward)
  - Trailing stop: move stop to breakeven after 1.5 ATR profit
  - Max hold: 20 bars — if no stop/target hit, exit at market

This means the strategy still uses MA crossover for DIRECTION, but
uses ATR for RISK MANAGEMENT. Entry timing from MAs, exit timing from
price action. This is how professional trend followers work.

Position flips take 2 bars for safety (exit bar + entry bar).
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
    """MA Crossover v2 — trend direction + ATR risk management."""

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["SPY"],
            "fast_period": 20,
            "slow_period": 50,
            "vol_lookback": 30,
            "vol_threshold": 0.25,
            "interval": "15m",
            "atr_period": 14,
            "stop_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "trail_trigger_atr": 1.5,
            "max_hold_bars": 20,
            # --- Issue #3 fix: confirmation before entry ---
            "confirm_bars": 2,          # MA must stay crossed for 2 bars
            # --- Issue #9 fix: minimum hold period ---
            "min_hold_bars": 3,         # Don't exit within 3 bars of entry
            # --- Cooldown after exit ---
            "cooldown_bars": 2,         # Wait 2 bars after exit before re-entering
        }
        self.name = "ma_crossover"
        self.event_bus = event_bus
        self.params = {**default_params, **(params or {})}
        self._is_active = True

        self._price_history: Dict[str, list] = defaultdict(list)
        self._bar_history: Dict[str, list] = defaultdict(list)

        self._position: Dict[str, Optional[str]] = {}
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._target_price: Dict[str, float] = {}
        self._original_stop: Dict[str, float] = {}
        self._bars_in_trade: Dict[str, int] = {}
        # Issue #3: confirmation tracking
        self._cross_direction: Dict[str, Optional[str]] = {}  # Which way MAs crossed
        self._cross_count: Dict[str, int] = {}  # How many bars the cross held
        # Issue #9: cooldown after exit
        self._cooldown: Dict[str, int] = {}  # Bars since last exit

        self._funnel = {
            "bars_processed": 0,
            "warmup_skip": 0,
            "no_change": 0,
            "long_entries": 0,
            "short_entries": 0,
            "long_exits": 0,
            "short_exits": 0,
            "exits_stop": 0,
            "exits_target": 0,
            "exits_trailing": 0,
            "exits_timeout": 0,
            "exits_flip": 0,
            "confirm_skip": 0,     # Issue #3: cross not confirmed
            "cooldown_skip": 0,    # Issue #9: in cooldown period
            "min_hold_skip": 0,    # Issue #9: too early to exit
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

        # Tick down cooldown
        if symbol in self._cooldown and self._cooldown[symbol] > 0:
            self._cooldown[symbol] -= 1

        signal = self._decide(symbol)
        if signal is not None:
            self.event_bus.publish(signal)

    def _decide(self, symbol: str) -> Optional[SignalEvent]:
        self._funnel["bars_processed"] += 1

        prices = np.array(self._price_history[symbol])
        fast_ma = self._ma(prices, self.params["fast_period"])
        slow_ma = self._ma(prices, self.params["slow_period"])

        if fast_ma is None or slow_ma is None:
            self._funnel["warmup_skip"] += 1
            return None

        current_price = prices[-1]
        current = self._position.get(symbol)
        desired = "long" if fast_ma > slow_ma else "short"

        bars = self._bar_history[symbol]
        atr_val = self._compute_atr(bars, self.params["atr_period"])

        regime = self._get_regime(symbol)
        strength = 1.0 if regime == "low" else 0.5

        # --- Check exits first (if in position) ---
        if current is not None:
            self._bars_in_trade[symbol] = self._bars_in_trade.get(symbol, 0) + 1

            # Issue #9: minimum hold period — don't exit too quickly
            bars_held = self._bars_in_trade.get(symbol, 0)
            min_hold = self.params["min_hold_bars"]
            if bars_held < min_hold:
                # Only allow exits for stop loss (hard risk limit)
                stop = self._stop_price.get(symbol, 0)
                if current == "long" and current_price <= stop:
                    pass  # Allow stop exit even during min hold
                elif current == "short" and current_price >= stop:
                    pass  # Allow stop exit even during min hold
                else:
                    self._funnel["min_hold_skip"] += 1
                    self._funnel["no_change"] += 1
                    return None

            exit_signal = self._check_exit(symbol, current_price, atr_val, desired)
            if exit_signal:
                # Start cooldown
                self._cooldown[symbol] = self.params["cooldown_bars"]
                return exit_signal
            self._funnel["no_change"] += 1
            return None

        # --- Issue #9: cooldown check ---
        if self._cooldown.get(symbol, 0) > 0:
            self._funnel["cooldown_skip"] += 1
            return None

        # --- Issue #3: confirmation check ---
        # Track how many consecutive bars the cross has held
        prev_cross = self._cross_direction.get(symbol)
        if desired != prev_cross:
            # New cross detected — start counting
            self._cross_direction[symbol] = desired
            self._cross_count[symbol] = 1
            self._funnel["confirm_skip"] += 1
            return None  # Don't enter yet, wait for confirmation
        else:
            self._cross_count[symbol] = self._cross_count.get(symbol, 0) + 1

        # Need N consecutive bars confirming the cross
        if self._cross_count.get(symbol, 0) < self.params["confirm_bars"]:
            self._funnel["confirm_skip"] += 1
            return None

        # --- Confirmed: enter new position ---
        stop_mult = self.params["stop_atr_mult"]
        target_mult = self.params["target_atr_mult"]

        if desired == "long":
            stop = current_price - stop_mult * atr_val
            target = current_price + target_mult * atr_val
            self._position[symbol] = "long"
            self._entry_price[symbol] = current_price
            self._stop_price[symbol] = stop
            self._original_stop[symbol] = stop
            self._target_price[symbol] = target
            self._bars_in_trade[symbol] = 0
            self._funnel["long_entries"] += 1

            return SignalEvent(
                symbol=symbol, signal_type=SignalType.LONG,
                strength=strength, strategy_name=self.name,
                metadata={
                    "fast_ma": round(fast_ma, 4), "slow_ma": round(slow_ma, 4),
                    "regime": regime, "action": "enter_long",
                    "stop": round(stop, 4), "target": round(target, 4),
                    "entry": round(current_price, 4),
                },
            )

        elif desired == "short":
            stop = current_price + stop_mult * atr_val
            target = current_price - target_mult * atr_val
            self._position[symbol] = "short"
            self._entry_price[symbol] = current_price
            self._stop_price[symbol] = stop
            self._original_stop[symbol] = stop
            self._target_price[symbol] = target
            self._bars_in_trade[symbol] = 0
            self._funnel["short_entries"] += 1

            return SignalEvent(
                symbol=symbol, signal_type=SignalType.SHORT,
                strength=strength, strategy_name=self.name,
                metadata={
                    "fast_ma": round(fast_ma, 4), "slow_ma": round(slow_ma, 4),
                    "regime": regime, "action": "enter_short",
                    "stop": round(stop, 4), "target": round(target, 4),
                    "entry": round(current_price, 4),
                },
            )

        self._funnel["no_change"] += 1
        return None

    def _check_exit(self, symbol, price, atr_val, desired):
        direction = self._position[symbol]
        stop = self._stop_price[symbol]
        target = self._target_price[symbol]
        entry = self._entry_price[symbol]
        bars_held = self._bars_in_trade.get(symbol, 0)

        # Trailing stop
        trigger = self.params["trail_trigger_atr"] * atr_val
        if direction == "long" and price > entry + trigger:
            new_stop = max(stop, entry + atr_val * 0.1)
            if new_stop > self._original_stop.get(symbol, stop):
                self._stop_price[symbol] = new_stop
                stop = new_stop
        elif direction == "short" and price < entry - trigger:
            new_stop = min(stop, entry - atr_val * 0.1)
            if new_stop < self._original_stop.get(symbol, stop):
                self._stop_price[symbol] = new_stop
                stop = new_stop

        should_exit = False
        exit_reason = ""

        if direction == "long":
            if price <= stop:
                should_exit = True
                exit_reason = "trailing" if stop > self._original_stop.get(symbol, stop) else "stop"
            elif price >= target:
                should_exit, exit_reason = True, "target"
        elif direction == "short":
            if price >= stop:
                should_exit = True
                exit_reason = "trailing" if stop < self._original_stop.get(symbol, stop) else "stop"
            elif price <= target:
                should_exit, exit_reason = True, "target"

        if not should_exit and bars_held >= self.params["max_hold_bars"]:
            should_exit, exit_reason = True, "timeout"

        if not should_exit and direction != desired:
            should_exit, exit_reason = True, "flip"

        if should_exit:
            exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT

            if exit_reason == "stop": self._funnel["exits_stop"] += 1
            elif exit_reason == "target": self._funnel["exits_target"] += 1
            elif exit_reason == "trailing": self._funnel["exits_trailing"] += 1
            elif exit_reason == "timeout": self._funnel["exits_timeout"] += 1
            elif exit_reason == "flip": self._funnel["exits_flip"] += 1

            if direction == "long": self._funnel["long_exits"] += 1
            else: self._funnel["short_exits"] += 1

            self._position[symbol] = None
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._original_stop.pop(symbol, None)
            self._target_price.pop(symbol, None)
            self._bars_in_trade.pop(symbol, None)

            return SignalEvent(
                symbol=symbol, signal_type=exit_type,
                strength=0.0, strategy_name=self.name,
                metadata={"exit_reason": exit_reason, "exit_price": round(price, 4),
                          "entry_price": round(entry, 4)},
            )
        return None

    def _compute_atr(self, bars, period):
        if len(bars) < period + 1:
            return 1.0
        trs = []
        for i in range(-period, 0):
            b = bars[i]
            prev_c = bars[i-1]["close"]
            tr = max(b["high"] - b["low"], abs(b["high"] - prev_c), abs(b["low"] - prev_c))
            trs.append(tr)
        return sum(trs) / len(trs)

    def _ma(self, prices, period):
        if len(prices) < period:
            return None
        return float(prices[-period:].mean())

    def _get_regime(self, symbol):
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

    def get_prices(self, symbol):
        return np.array(self._price_history.get(symbol, []))

    def calculate_signal(self, symbol):
        return None

    def get_diagnostics(self):
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
        print(f"    ├─ Confirm wait:        {f['confirm_skip']:>8,}  (cross not held {self.params['confirm_bars']} bars)")
        print(f"    ├─ Cooldown:            {f['cooldown_skip']:>8,}  ({self.params['cooldown_bars']} bar wait after exit)")
        print(f"    ├─ Min hold block:      {f['min_hold_skip']:>8,}  (< {self.params['min_hold_bars']} bars)")
        print(f"    ├─ Enter LONG:          {f['long_entries']:>8,}")
        print(f"    ├─ Enter SHORT:         {f['short_entries']:>8,}")
        print(f"    ├─ Exit long:           {f['long_exits']:>8,}")
        print(f"    └─ Exit short:          {f['short_exits']:>8,}")
        print(f"  {'─'*50}")
        print(f"  Total signals:            {total_signals:>8,}  ({total_entries} entries + {total_exits} exits)")
        print(f"  Exits — stop:             {f['exits_stop']:>8,}")
        print(f"  Exits — target:           {f['exits_target']:>8,}")
        print(f"  Exits — trailing:         {f['exits_trailing']:>8,}")
        print(f"  Exits — timeout:          {f['exits_timeout']:>8,}  (>{self.params['max_hold_bars']} bars)")
        print(f"  Exits — MA flip:          {f['exits_flip']:>8,}")
        print(f"  Vol regime — low:         {f['regime_low']:>8,}")
        print(f"  Vol regime — high:        {f['regime_high']:>8,}")
        print(f"  {'─'*50}")

    def stop(self):
        self._is_active = False

    def start(self):
        self._is_active = True