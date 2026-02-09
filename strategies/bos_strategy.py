"""Break of Structure (BOS) Strategy.

An intraday strategy based on Smart Money Concepts:

1. Identify market structure using swing highs/lows
2. Detect Break of Structure (BOS) or Change of Character (CHoCH)
3. Confirm with volume and volatility filters
4. Enter on pullback to order block / support-resistance
5. Manage risk with ATR-based stops

This is a professional-grade implementation of concepts used by
institutional and smart money traders.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.events import (
    EventBus, EventType, MarketDataEvent, SignalEvent, SignalType,
)
from strategies import BaseStrategy

from features import (
    find_swing_points,
    detect_market_structure,
    find_support_resistance,
    nearest_support,
    nearest_resistance,
    atr,
    rsi,
    relative_volume,
    vwap,
    volume_profile,
    bollinger_bands,
    realized_volatility,
    StructureBreak,
    SwingType,
)


class BOSStrategy(BaseStrategy):
    """Break of Structure intraday strategy.

    Entry logic:
    - Wait for a clear BOS (price closes beyond recent swing)
    - Confirm with volume (RVOL > threshold) and momentum (RSI not extreme)
    - Enter on pullback toward the broken level
    - Stop loss below/above the structure level
    - Take profit at next support/resistance or risk multiple

    Params:
        symbols: List of symbols to trade
        swing_lookback: Bars for swing detection (default: 5)
        min_bars: Minimum bars before generating signals (warmup)
        rvol_threshold: Min relative volume for confirmation (default: 1.2)
        rsi_oversold: RSI oversold level (default: 30)
        rsi_overbought: RSI overbought level (default: 70)
        atr_period: ATR period for stops (default: 14)
        risk_reward: Take profit as multiple of stop distance (default: 2.0)
        pullback_pct: How much pullback to wait for before entry (0-1)
        use_volume_filter: Require volume confirmation (default: True)
        use_vwap_filter: Require price vs VWAP alignment (default: True)
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            "swing_lookback": 5,
            "min_bars": 60,
            "rvol_threshold": 1.2,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "risk_reward": 2.0,
            "pullback_pct": 0.3,
            "use_volume_filter": True,
            "use_vwap_filter": True,
        }
        merged = {**default_params, **(params or {})}
        super().__init__("bos", event_bus, merged)

        # Per-symbol state
        self._last_structure: Dict[str, Dict] = {}
        self._pending_entry: Dict[str, Dict] = {}  # Waiting for pullback
        self._in_position: Dict[str, str] = {}     # symbol -> "long" or "short"
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._target_price: Dict[str, float] = {}
        self._trade_count: Dict[str, int] = defaultdict(int)

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Main signal logic for BOS strategy."""
        if symbol not in self.params.get("symbols", []):
            return None

        bars = self._bar_history.get(symbol, [])
        if len(bars) < self.params["min_bars"]:
            return None

        # Build arrays from bar history
        high = np.array([b["high"] for b in bars])
        low = np.array([b["low"] for b in bars])
        close = np.array([b["close"] for b in bars])
        opn = np.array([b["open"] for b in bars])
        volume = np.array([b.get("volume", 0) for b in bars], dtype=float)

        current_price = close[-1]
        current_idx = len(close) - 1

        # --- Compute features ---
        swing_lb = self.params["swing_lookback"]
        swings = find_swing_points(high, low, swing_lb, swing_lb)
        structure = detect_market_structure(high, low, close, swings, swing_lb, 2)
        sr_levels = find_support_resistance(high, low, close, swings)
        atr_values = atr(high, low, close, self.params["atr_period"])
        rsi_values = rsi(close, 14)
        rvol_values = relative_volume(volume, 20)
        vwap_values = vwap(high, low, close, volume)

        current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else 0
        current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50
        current_rvol = rvol_values[-1] if not np.isnan(rvol_values[-1]) else 1.0
        current_vwap = vwap_values[-1]

        # Store structure for diagnostics
        self._last_structure[symbol] = structure

        # --- Check for exits first ---
        if symbol in self._in_position:
            return self._check_exit(symbol, current_price, current_atr, structure)

        # --- Check for pending pullback entries ---
        if symbol in self._pending_entry:
            return self._check_pullback_entry(
                symbol, current_price, current_rsi, current_rvol,
                current_vwap, current_atr
            )

        # --- Detect new BOS ---
        bos = structure["structure_break"]

        if bos == StructureBreak.NONE:
            return None

        # --- Volume confirmation ---
        if self.params["use_volume_filter"] and current_rvol < self.params["rvol_threshold"]:
            return None  # Not enough volume behind the break

        # --- VWAP confirmation ---
        if self.params["use_vwap_filter"]:
            if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH):
                if current_price < current_vwap:
                    return None  # Price below VWAP contradicts bullish break
            elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH):
                if current_price > current_vwap:
                    return None  # Price above VWAP contradicts bearish break

        # --- RSI filter: don't chase extremes ---
        if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH):
            if current_rsi > self.params["rsi_overbought"]:
                return None  # Too extended
        elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH):
            if current_rsi < self.params["rsi_oversold"]:
                return None  # Too extended

        # --- Set up pending entry (wait for pullback) ---
        last_sh = structure["last_swing_high"]
        last_sl = structure["last_swing_low"]

        if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH) and last_sh:
            # Bullish BOS: wait for pullback toward the broken swing high
            broken_level = last_sh.price
            pullback_target = current_price - (current_price - broken_level) * self.params["pullback_pct"]
            stop = last_sl.price if last_sl else broken_level - 2 * current_atr
            target = current_price + (current_price - stop) * self.params["risk_reward"]

            self._pending_entry[symbol] = {
                "direction": "long",
                "bos_type": bos.value,
                "broken_level": broken_level,
                "pullback_target": pullback_target,
                "stop": stop,
                "target": target,
                "atr_at_signal": current_atr,
                "bars_waiting": 0,
                "max_wait_bars": 10,
            }

        elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH) and last_sl:
            # Bearish BOS: wait for pullback toward the broken swing low
            broken_level = last_sl.price
            pullback_target = current_price + (broken_level - current_price) * self.params["pullback_pct"]
            stop = last_sh.price if last_sh else broken_level + 2 * current_atr
            target = current_price - (stop - current_price) * self.params["risk_reward"]

            self._pending_entry[symbol] = {
                "direction": "short",
                "bos_type": bos.value,
                "broken_level": broken_level,
                "pullback_target": pullback_target,
                "stop": stop,
                "target": target,
                "atr_at_signal": current_atr,
                "bars_waiting": 0,
                "max_wait_bars": 10,
            }

        return None  # Signal comes on pullback, not on the break itself

    def _check_pullback_entry(
        self, symbol: str, price: float, rsi_val: float,
        rvol: float, vwap_val: float, current_atr: float,
    ) -> Optional[SignalEvent]:
        """Check if pullback has reached entry zone."""
        pending = self._pending_entry[symbol]
        pending["bars_waiting"] += 1

        # Cancel if waited too long
        if pending["bars_waiting"] > pending["max_wait_bars"]:
            del self._pending_entry[symbol]
            return None

        if pending["direction"] == "long":
            # Enter if price pulled back enough
            if price <= pending["pullback_target"]:
                del self._pending_entry[symbol]

                self._in_position[symbol] = "long"
                self._entry_price[symbol] = price
                self._stop_price[symbol] = pending["stop"]
                self._target_price[symbol] = pending["target"]
                self._trade_count[symbol] += 1

                return SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=min(rvol / 2.0, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "bos_type": pending["bos_type"],
                        "entry": price,
                        "stop": pending["stop"],
                        "target": pending["target"],
                        "rsi": rsi_val,
                        "rvol": rvol,
                        "risk_reward": self.params["risk_reward"],
                    },
                )

        elif pending["direction"] == "short":
            if price >= pending["pullback_target"]:
                del self._pending_entry[symbol]

                self._in_position[symbol] = "short"
                self._entry_price[symbol] = price
                self._stop_price[symbol] = pending["stop"]
                self._target_price[symbol] = pending["target"]
                self._trade_count[symbol] += 1

                return SignalEvent(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=min(rvol / 2.0, 1.0),
                    strategy_name=self.name,
                    metadata={
                        "bos_type": pending["bos_type"],
                        "entry": price,
                        "stop": pending["stop"],
                        "target": pending["target"],
                        "rsi": rsi_val,
                        "rvol": rvol,
                        "risk_reward": self.params["risk_reward"],
                    },
                )

        return None

    def _check_exit(
        self, symbol: str, price: float, current_atr: float, structure: Dict,
    ) -> Optional[SignalEvent]:
        """Check stop loss, take profit, and structure-based exits."""
        direction = self._in_position[symbol]
        stop = self._stop_price[symbol]
        target = self._target_price[symbol]

        should_exit = False
        exit_reason = ""

        if direction == "long":
            if price <= stop:
                should_exit = True
                exit_reason = "stop_loss"
            elif price >= target:
                should_exit = True
                exit_reason = "take_profit"
            # Exit on bearish structure break while long
            elif structure["structure_break"] in (
                StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH
            ):
                should_exit = True
                exit_reason = "structure_reversal"

        elif direction == "short":
            if price >= stop:
                should_exit = True
                exit_reason = "stop_loss"
            elif price <= target:
                should_exit = True
                exit_reason = "take_profit"
            elif structure["structure_break"] in (
                StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH
            ):
                should_exit = True
                exit_reason = "structure_reversal"

        if should_exit:
            exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT
            entry = self._entry_price.get(symbol, price)
            pnl_pct = (price - entry) / entry if direction == "long" else (entry - price) / entry

            # Clean up state
            del self._in_position[symbol]
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._target_price.pop(symbol, None)

            return SignalEvent(
                symbol=symbol,
                signal_type=exit_type,
                strength=0.0,
                strategy_name=self.name,
                metadata={
                    "exit_reason": exit_reason,
                    "exit_price": price,
                    "entry_price": entry,
                    "pnl_pct": pnl_pct,
                },
            )

        return None

    def get_diagnostics(self) -> Dict:
        """Return current strategy state."""
        diag = {}
        for symbol in self.params.get("symbols", []):
            struct = self._last_structure.get(symbol, {})
            diag[symbol] = {
                "trend": struct.get("trend", "unknown"),
                "bos": struct.get("structure_break", StructureBreak.NONE).value
                       if hasattr(struct.get("structure_break", ""), "value") else "none",
                "hh": struct.get("higher_highs", 0),
                "hl": struct.get("higher_lows", 0),
                "lh": struct.get("lower_highs", 0),
                "ll": struct.get("lower_lows", 0),
                "in_position": self._in_position.get(symbol, "flat"),
                "pending_entry": symbol in self._pending_entry,
                "trades": self._trade_count.get(symbol, 0),
                "stop": self._stop_price.get(symbol),
                "target": self._target_price.get(symbol),
                "bars": len(self._bar_history.get(symbol, [])),
            }
        return diag
