"""Break of Structure (BOS) Strategy v2 — With Regime Detection.

Improvements over initial bos strategy:
  1. REGIME FILTER: Only trades when ADX-style trend strength is sufficient.
     Sits in cash during ranging/choppy markets (where v1 lost all its money).

  2. RELAXED FILTERS: Volume filter threshold lowered (1.2 → 0.8) because
     high-volume stocks like NVDA/AAPL have high baselines. VWAP filter
     disabled by default — on trending days price is often far from VWAP
     but the breakout is still valid.

  3. ADAPTIVE RISK-REWARD: 2.0x was too ambitious — only hit 11% of time.
     Now uses 1.5x default, and 1.2x in high-volatility regimes.

  4. LONGER PULLBACK WINDOW: 10 bars → 20 bars. Pullbacks on 15m charts
     can take 5+ hours. 10 bars (2.5 hrs) was losing 36% of valid signals.

  5. TRAILING STOP: Once trade is 1 ATR in profit, stop moves to breakeven.
     This cuts the "reversal exit" losses that were 63% of all exits.

Entry logic:
  - Detect BOS (price closes beyond recent swing)
  - Check regime: trend strength > threshold (skip ranging markets)
  - Confirm with volume (RVOL > 0.8) and momentum (RSI not extreme)
  - Enter on pullback toward the broken level
  - Stop loss below/above structure level
  - Take profit at risk_reward multiple
  - Trailing stop to breakeven after 1 ATR profit
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
    volatility_regime,
    realized_volatility,
    bollinger_bands,
    StructureBreak,
    SwingType,
)


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ADX (Average Directional Index) — measures trend strength.

    ADX > 25: trending market (trade breakouts)
    ADX < 20: ranging market (don't trade breakouts)

    This is the single most important filter for BOS.
    """
    n = len(close)
    adx_vals = np.full(n, np.nan)

    if n < period * 2 + 1:
        return adx_vals

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed averages (Wilder's smoothing)
    atr_smooth = np.zeros(n)
    plus_di_smooth = np.zeros(n)
    minus_di_smooth = np.zeros(n)

    # Seed with SMA
    atr_smooth[period] = tr[1:period+1].mean()
    plus_di_smooth[period] = plus_dm[1:period+1].mean()
    minus_di_smooth[period] = minus_dm[1:period+1].mean()

    for i in range(period + 1, n):
        atr_smooth[i] = (atr_smooth[i-1] * (period - 1) + tr[i]) / period
        plus_di_smooth[i] = (plus_di_smooth[i-1] * (period - 1) + plus_dm[i]) / period
        minus_di_smooth[i] = (minus_di_smooth[i-1] * (period - 1) + minus_dm[i]) / period

    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_di_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_di_smooth[i] / atr_smooth[i]
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

    # ADX = smoothed DX
    adx_start = period * 2
    if adx_start < n:
        adx_vals[adx_start] = dx[period+1:adx_start+1].mean()
        for i in range(adx_start + 1, n):
            adx_vals[i] = (adx_vals[i-1] * (period - 1) + dx[i]) / period

    return adx_vals


class BOSStrategy(BaseStrategy):
    """Break of Structure v2 — regime-aware with adaptive parameters."""

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            "swing_lookback": 5,
            "min_bars": 60,
            # --- Relaxed filters ---
            "rvol_threshold": 0.8,        # Was 1.2 — killed 54% of signals
            "rsi_oversold": 25,           # Was 30 — slightly wider band
            "rsi_overbought": 75,         # Was 70
            "atr_period": 14,
            "risk_reward": 1.5,           # Was 2.0 — hit only 11% of time
            "pullback_pct": 0.3,
            "max_wait_bars": 20,          # Was 10 — lost 36% of signals
            # --- New: regime detection ---
            "use_volume_filter": True,
            "use_vwap_filter": False,     # Was True — killed 27% on trending days
            "use_regime_filter": True,    # NEW: the most important change
            "adx_threshold": 20,          # ADX > 20 = trending enough to trade
            "adx_period": 14,
            # --- New: trailing stop ---
            "use_trailing_stop": True,
            "trail_trigger_atr": 1.0,     # Move stop to BE after 1 ATR profit
        }
        merged = {**default_params, **(params or {})}
        super().__init__("bos", event_bus, merged)

        # Per-symbol state
        self._last_structure: Dict[str, Dict] = {}
        self._pending_entry: Dict[str, Dict] = {}
        self._in_position: Dict[str, str] = {}     # symbol -> "long" or "short"
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._target_price: Dict[str, float] = {}
        self._original_stop: Dict[str, float] = {}  # For trailing stop
        self._trade_count: Dict[str, int] = defaultdict(int)

        # Signal funnel
        self._funnel = {
            "bars_processed": 0,
            "bos_detected": 0,
            "filtered_no_bos": 0,
            "filtered_in_position": 0,
            "filtered_pending": 0,
            "filtered_volume": 0,
            "filtered_vwap": 0,
            "filtered_rsi": 0,
            "filtered_regime": 0,      # NEW
            "pending_created": 0,
            "pullback_timeout": 0,
            "pullback_entered": 0,
            "exits_stop": 0,
            "exits_trailing": 0,       # NEW
            "exits_target": 0,
            "exits_reversal": 0,
            "signals_sent": 0,
        }

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Main signal logic."""
        if symbol not in self.params.get("symbols", []):
            return None

        bars = self._bar_history.get(symbol, [])
        if len(bars) < self.params["min_bars"]:
            return None

        # Build arrays
        high = np.array([b["high"] for b in bars])
        low = np.array([b["low"] for b in bars])
        close = np.array([b["close"] for b in bars])
        volume = np.array([b.get("volume", 0) for b in bars], dtype=float)

        current_price = close[-1]

        # Compute features
        swing_lb = self.params["swing_lookback"]
        swings = find_swing_points(high, low, swing_lb, swing_lb)
        structure = detect_market_structure(high, low, close, swings, swing_lb, 2)
        atr_values = atr(high, low, close, self.params["atr_period"])
        rsi_values = rsi(close, 14)
        rvol_values = relative_volume(volume, 20)
        vwap_values = vwap(high, low, close, volume)
        adx_values = _adx(high, low, close, self.params["adx_period"])

        current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else 0
        current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50
        current_rvol = rvol_values[-1] if not np.isnan(rvol_values[-1]) else 1.0
        current_vwap = vwap_values[-1]
        current_adx = adx_values[-1] if not np.isnan(adx_values[-1]) else 0

        self._last_structure[symbol] = structure
        self._funnel["bars_processed"] += 1

        # --- Check exits first ---
        if symbol in self._in_position:
            self._funnel["filtered_in_position"] += 1
            return self._check_exit(symbol, current_price, current_atr, structure)

        # --- Check pending pullback entries ---
        if symbol in self._pending_entry:
            self._funnel["filtered_pending"] += 1
            return self._check_pullback_entry(
                symbol, current_price, current_rsi, current_rvol,
                current_vwap, current_atr
            )

        # --- Detect new BOS ---
        bos = structure["structure_break"]
        if bos == StructureBreak.NONE:
            self._funnel["filtered_no_bos"] += 1
            return None

        self._funnel["bos_detected"] += 1

        # --- REGIME FILTER (the key improvement) ---
        if self.params["use_regime_filter"]:
            if current_adx < self.params["adx_threshold"]:
                self._funnel["filtered_regime"] += 1
                return None  # Market not trending enough — sit in cash

        # --- Volume filter ---
        if self.params["use_volume_filter"] and current_rvol < self.params["rvol_threshold"]:
            self._funnel["filtered_volume"] += 1
            return None

        # --- VWAP filter ---
        if self.params["use_vwap_filter"]:
            if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH):
                if current_price < current_vwap:
                    self._funnel["filtered_vwap"] += 1
                    return None
            elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH):
                if current_price > current_vwap:
                    self._funnel["filtered_vwap"] += 1
                    return None

        # --- RSI filter ---
        if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH):
            if current_rsi > self.params["rsi_overbought"]:
                self._funnel["filtered_rsi"] += 1
                return None
        elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH):
            if current_rsi < self.params["rsi_oversold"]:
                self._funnel["filtered_rsi"] += 1
                return None

        # --- Set up pending entry ---
        last_sh = structure["last_swing_high"]
        last_sl = structure["last_swing_low"]

        # Adaptive risk-reward based on volatility
        vol_regime_values = volatility_regime(close)
        vol_ratio = vol_regime_values[-1] if not np.isnan(vol_regime_values[-1]) else 1.0
        rr = self.params["risk_reward"]
        if vol_ratio > 1.3:
            rr = max(1.2, rr - 0.3)  # Tighter target in high vol

        max_wait = self.params.get("max_wait_bars", 20)

        if bos in (StructureBreak.BULLISH_BOS, StructureBreak.BULLISH_CHOCH) and last_sh:
            broken_level = last_sh.price
            pullback_target = current_price - (current_price - broken_level) * self.params["pullback_pct"]
            stop = last_sl.price if last_sl else broken_level - 2 * current_atr
            target = current_price + (current_price - stop) * rr

            self._pending_entry[symbol] = {
                "direction": "long",
                "bos_type": bos.value,
                "broken_level": broken_level,
                "pullback_target": pullback_target,
                "stop": stop,
                "target": target,
                "atr_at_signal": current_atr,
                "bars_waiting": 0,
                "max_wait_bars": max_wait,
                "adx": current_adx,
            }
            self._funnel["pending_created"] += 1

        elif bos in (StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH) and last_sl:
            broken_level = last_sl.price
            pullback_target = current_price + (broken_level - current_price) * self.params["pullback_pct"]
            stop = last_sh.price if last_sh else broken_level + 2 * current_atr
            target = current_price - (stop - current_price) * rr

            self._pending_entry[symbol] = {
                "direction": "short",
                "bos_type": bos.value,
                "broken_level": broken_level,
                "pullback_target": pullback_target,
                "stop": stop,
                "target": target,
                "atr_at_signal": current_atr,
                "bars_waiting": 0,
                "max_wait_bars": max_wait,
                "adx": current_adx,
            }
            self._funnel["pending_created"] += 1

        return None

    def _check_pullback_entry(
        self, symbol: str, price: float, rsi_val: float,
        rvol: float, vwap_val: float, current_atr: float,
    ) -> Optional[SignalEvent]:
        """Check if pullback has reached entry zone."""
        pending = self._pending_entry[symbol]
        pending["bars_waiting"] += 1

        if pending["bars_waiting"] > pending["max_wait_bars"]:
            del self._pending_entry[symbol]
            self._funnel["pullback_timeout"] += 1
            return None

        entered = False
        if pending["direction"] == "long" and price <= pending["pullback_target"]:
            entered = True
        elif pending["direction"] == "short" and price >= pending["pullback_target"]:
            entered = True

        if not entered:
            return None

        direction = pending["direction"]
        del self._pending_entry[symbol]

        self._in_position[symbol] = direction
        self._entry_price[symbol] = price
        self._stop_price[symbol] = pending["stop"]
        self._original_stop[symbol] = pending["stop"]
        self._target_price[symbol] = pending["target"]
        self._trade_count[symbol] += 1
        self._funnel["pullback_entered"] += 1
        self._funnel["signals_sent"] += 1

        signal_type = SignalType.LONG if direction == "long" else SignalType.SHORT

        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            strength=min(rvol / 2.0, 1.0),
            strategy_name=self.name,
            metadata={
                "bos_type": pending["bos_type"],
                "entry": price,
                "stop": pending["stop"],
                "target": pending["target"],
                "rsi": rsi_val,
                "rvol": rvol,
                "adx": pending.get("adx", 0),
                "risk_reward": self.params["risk_reward"],
            },
        )

    def _check_exit(
        self, symbol: str, price: float, current_atr: float, structure: Dict,
    ) -> Optional[SignalEvent]:
        """Check stop loss, trailing stop, take profit, and structure exits."""
        direction = self._in_position[symbol]
        stop = self._stop_price[symbol]
        target = self._target_price[symbol]
        entry = self._entry_price.get(symbol, price)

        # --- Trailing stop: move to breakeven after 1 ATR profit ---
        if self.params["use_trailing_stop"]:
            trigger = self.params["trail_trigger_atr"] * current_atr
            if direction == "long" and price > entry + trigger:
                new_stop = max(stop, entry + current_atr * 0.1)  # Just above breakeven
                if new_stop > self._original_stop[symbol]:
                    self._stop_price[symbol] = new_stop
                    stop = new_stop
            elif direction == "short" and price < entry - trigger:
                new_stop = min(stop, entry - current_atr * 0.1)
                if new_stop < self._original_stop[symbol]:
                    self._stop_price[symbol] = new_stop
                    stop = new_stop

        should_exit = False
        exit_reason = ""

        if direction == "long":
            if price <= stop:
                should_exit = True
                # Determine if this was the original stop or trailing
                exit_reason = "trailing_stop" if stop > self._original_stop.get(symbol, stop) else "stop_loss"
            elif price >= target:
                should_exit = True
                exit_reason = "take_profit"
            elif structure["structure_break"] in (
                StructureBreak.BEARISH_BOS, StructureBreak.BEARISH_CHOCH
            ):
                should_exit = True
                exit_reason = "structure_reversal"

        elif direction == "short":
            if price >= stop:
                should_exit = True
                exit_reason = "trailing_stop" if stop < self._original_stop.get(symbol, stop) else "stop_loss"
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
            pnl_pct = (price - entry) / entry if direction == "long" else (entry - price) / entry

            if exit_reason == "stop_loss":
                self._funnel["exits_stop"] += 1
            elif exit_reason == "trailing_stop":
                self._funnel["exits_trailing"] += 1
            elif exit_reason == "take_profit":
                self._funnel["exits_target"] += 1
            elif exit_reason == "structure_reversal":
                self._funnel["exits_reversal"] += 1
            self._funnel["signals_sent"] += 1

            # Clean up
            del self._in_position[symbol]
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._original_stop.pop(symbol, None)
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
        diag["_funnel"] = self._funnel
        return diag

    def print_funnel(self):
        f = self._funnel
        print(f"\n  SIGNAL FUNNEL — {self.name}")
        print(f"  {'─'*50}")
        print(f"  Bars processed:           {f['bars_processed']:>8,}")
        print(f"    ├─ No BOS detected:     {f['filtered_no_bos']:>8,}")
        print(f"    ├─ In position (exit):   {f['filtered_in_position']:>8,}")
        print(f"    ├─ Pending pullback:     {f['filtered_pending']:>8,}")
        print(f"    └─ BOS detected:        {f['bos_detected']:>8,}  ← potential signals")
        print(f"        ├─ Regime filtered:  {f['filtered_regime']:>8,}  (ADX < {self.params['adx_threshold']})")
        print(f"        ├─ Volume filtered:  {f['filtered_volume']:>8,}  (RVOL < {self.params['rvol_threshold']})")
        if self.params['use_vwap_filter']:
            print(f"        ├─ VWAP filtered:    {f['filtered_vwap']:>8,}")
        print(f"        ├─ RSI filtered:     {f['filtered_rsi']:>8,}")
        print(f"        └─ Pending created:  {f['pending_created']:>8,}  ← waiting for pullback")
        print(f"            ├─ Timed out:    {f['pullback_timeout']:>8,}  (no pullback in {self.params.get('max_wait_bars', 20)} bars)")
        print(f"            └─ ENTERED:      {f['pullback_entered']:>8,}  ← actual entries")
        print(f"  {'─'*50}")
        print(f"  Total signals sent:       {f['signals_sent']:>8,}  (entries + exits)")
        print(f"  Exits — stop loss:        {f['exits_stop']:>8,}")
        print(f"  Exits — trailing stop:    {f['exits_trailing']:>8,}")
        print(f"  Exits — take profit:      {f['exits_target']:>8,}")
        print(f"  Exits — reversal:         {f['exits_reversal']:>8,}")
        print(f"  {'─'*50}")