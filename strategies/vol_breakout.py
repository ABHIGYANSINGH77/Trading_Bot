"""Volatility Breakout Strategy — Profits from Vol Expansion.

This strategy is uncorrelated to both trend-following and mean-reversion.
It detects periods of low volatility (compression) and trades the breakout
when volatility expands.

Core idea (Bollinger Band Squeeze):
  1. When Bollinger bandwidth contracts below a threshold → market is coiling
  2. When price breaks out of the squeeze with volume → ride the expansion
  3. Direction determined by which band gets broken first

This is the strategy behind the "squeeze" setups that prop traders use.
It doesn't care about trend or range — it trades the TRANSITION between them.

Entry:
  - Detect squeeze: BB width < threshold (e.g., 20th percentile of recent width)
  - Wait for breakout: price closes above upper band OR below lower band
  - Confirm: volume > 1.2x average
  - Enter in direction of breakout

Exit:
  - Stop: opposite Bollinger Band at entry time
  - Target: 2x the Bollinger bandwidth at entry
  - Trailing stop after 1.5 ATR profit
"""

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType
from strategies import BaseStrategy
from features import atr, bollinger_bands, relative_volume


class VolatilityBreakoutStrategy(BaseStrategy):
    """Bollinger Squeeze Breakout — trades vol expansion.

    Params:
        symbols:              List of symbols
        bb_period:            Bollinger Band lookback (default 20)
        bb_std:               Standard deviations (default 2.0)
        squeeze_percentile:   Width percentile threshold for squeeze (default 20)
        squeeze_lookback:     Bars to compute width percentile (default 120)
        volume_confirm:       Min RVOL for breakout confirmation (default 1.0)
        atr_period:           ATR period (default 14)
        risk_reward:          Target as multiple of bandwidth (default 2.0)
        use_trailing:         Enable trailing stop (default True)
        trail_trigger_atr:    ATR profit to activate trailing (default 1.5)
        min_bars:             Warmup (default 60)
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            "bb_period": 20,
            "bb_std": 2.0,
            "squeeze_percentile": 20,
            "squeeze_lookback": 120,
            "volume_confirm": 1.0,
            "atr_period": 14,
            "risk_reward": 2.0,
            "use_trailing": True,
            "trail_trigger_atr": 1.5,
            "min_bars": 60,
        }
        merged = {**default_params, **(params or {})}
        super().__init__("vol_breakout", event_bus, merged)

        self._in_squeeze: Dict[str, bool] = {}
        self._in_position: Dict[str, str] = {}
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._original_stop: Dict[str, float] = {}
        self._target_price: Dict[str, float] = {}

        self._funnel = {
            "bars_processed": 0,
            "warmup_skip": 0,
            "in_position": 0,
            "not_in_squeeze": 0,
            "in_squeeze_waiting": 0,
            "no_breakout": 0,
            "filtered_volume": 0,
            "long_entries": 0,
            "short_entries": 0,
            "exits_target": 0,
            "exits_stop": 0,
            "exits_trailing": 0,
        }

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        if symbol not in self.params.get("symbols", []):
            return None

        bars = self._bar_history.get(symbol, [])
        if len(bars) < self.params["min_bars"]:
            self._funnel["warmup_skip"] += 1
            return None

        self._funnel["bars_processed"] += 1

        high = np.array([b["high"] for b in bars])
        low = np.array([b["low"] for b in bars])
        close = np.array([b["close"] for b in bars])
        volume = np.array([b.get("volume", 0) for b in bars], dtype=float)

        price = close[-1]
        upper_arr, middle_arr, lower_arr = bollinger_bands(close, self.params["bb_period"], self.params["bb_std"])
        upper = upper_arr[-1]
        lower = lower_arr[-1]
        middle = middle_arr[-1]

        # Bollinger bandwidth (normalized)
        bandwidth = (upper - lower) / middle if middle > 0 else 0

        atr_val = atr(high, low, close, self.params["atr_period"])[-1]
        rvol = relative_volume(volume, 20)[-1]

        if np.isnan(atr_val) or np.isnan(rvol):
            return None

        # --- Check exits first ---
        if symbol in self._in_position:
            self._funnel["in_position"] += 1
            return self._check_exit(symbol, price, atr_val)

        # --- Detect squeeze ---
        # Compute percentile of current bandwidth vs recent history
        lb = self.params["squeeze_lookback"]
        if len(close) >= lb:
            u_hist, m_hist, l_hist = bollinger_bands(close, self.params["bb_period"], self.params["bb_std"])
            widths = (u_hist - l_hist) / np.where(m_hist > 0, m_hist, 1.0)
            widths = widths[-lb:]
            widths = widths[~np.isnan(widths)]
            if len(widths) > 10:
                pct = np.percentile(widths, self.params["squeeze_percentile"])
                is_squeeze = bandwidth < pct
            else:
                is_squeeze = False
        else:
            is_squeeze = False

        was_in_squeeze = self._in_squeeze.get(symbol, False)
        self._in_squeeze[symbol] = is_squeeze

        # Need to have been in squeeze, now breaking out
        if not was_in_squeeze and not is_squeeze:
            self._funnel["not_in_squeeze"] += 1
            return None

        if is_squeeze:
            self._funnel["in_squeeze_waiting"] += 1
            return None  # Still in squeeze, waiting for breakout

        # --- Breakout detected (was in squeeze, now bandwidth expanding) ---
        # Determine direction
        if price > upper:
            direction = "long"
        elif price < lower:
            direction = "short"
        else:
            self._funnel["no_breakout"] += 1
            return None  # Expansion but price between bands — no clear direction

        # Volume confirmation
        if rvol < self.params["volume_confirm"]:
            self._funnel["filtered_volume"] += 1
            return None

        # Set up trade
        bw_distance = upper - lower  # Use bandwidth for target
        if direction == "long":
            stop = lower  # Stop at opposite band
            target = price + bw_distance * self.params["risk_reward"]
            signal_type = SignalType.LONG
            self._funnel["long_entries"] += 1
        else:
            stop = upper
            target = price - bw_distance * self.params["risk_reward"]
            signal_type = SignalType.SHORT
            self._funnel["short_entries"] += 1

        self._in_position[symbol] = direction
        self._entry_price[symbol] = price
        self._stop_price[symbol] = stop
        self._original_stop[symbol] = stop
        self._target_price[symbol] = target

        return SignalEvent(
            symbol=symbol, signal_type=signal_type,
            strength=min(rvol / 2.0, 1.0), strategy_name=self.name,
            metadata={"entry": price, "stop": stop, "target": target,
                      "bandwidth": bandwidth, "rvol": rvol, "direction": direction},
        )

    def _check_exit(self, symbol, price, atr_val):
        direction = self._in_position[symbol]
        stop = self._stop_price[symbol]
        target = self._target_price[symbol]
        entry = self._entry_price[symbol]

        # Trailing stop
        if self.params["use_trailing"]:
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
                exit_reason = "trailing" if stop > self._original_stop.get(symbol, stop) else "stop_loss"
            elif price >= target:
                should_exit, exit_reason = True, "take_profit"
        elif direction == "short":
            if price >= stop:
                should_exit = True
                exit_reason = "trailing" if stop < self._original_stop.get(symbol, stop) else "stop_loss"
            elif price <= target:
                should_exit, exit_reason = True, "take_profit"

        if should_exit:
            exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT
            if "trailing" in exit_reason:
                self._funnel["exits_trailing"] += 1
            elif exit_reason == "stop_loss":
                self._funnel["exits_stop"] += 1
            else:
                self._funnel["exits_target"] += 1

            del self._in_position[symbol]
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._original_stop.pop(symbol, None)
            self._target_price.pop(symbol, None)

            return SignalEvent(
                symbol=symbol, signal_type=exit_type,
                strength=0.0, strategy_name=self.name,
                metadata={"exit_reason": exit_reason, "exit_price": price, "entry_price": entry},
            )
        return None

    def get_diagnostics(self) -> Dict:
        return {"_funnel": self._funnel}

    def print_funnel(self):
        f = self._funnel
        entries = f["long_entries"] + f["short_entries"]
        exits = f["exits_target"] + f["exits_stop"] + f["exits_trailing"]
        print(f"\n  SIGNAL FUNNEL — {self.name}")
        print(f"  {'─'*50}")
        print(f"  Bars processed:           {f['bars_processed']:>8,}")
        print(f"    ├─ In position:         {f['in_position']:>8,}")
        print(f"    ├─ Not in squeeze:      {f['not_in_squeeze']:>8,}")
        print(f"    ├─ In squeeze (wait):   {f['in_squeeze_waiting']:>8,}")
        print(f"    ├─ No clear breakout:   {f['no_breakout']:>8,}")
        print(f"    ├─ Volume too low:      {f['filtered_volume']:>8,}")
        print(f"    ├─ Enter LONG:          {f['long_entries']:>8,}")
        print(f"    └─ Enter SHORT:         {f['short_entries']:>8,}")
        print(f"  {'─'*50}")
        print(f"  Exits — target:           {f['exits_target']:>8,}")
        print(f"  Exits — stop:             {f['exits_stop']:>8,}")
        print(f"  Exits — trailing:         {f['exits_trailing']:>8,}")
        print(f"  {'─'*50}")