"""Mean Reversion Strategy — Profits in Ranging Markets.

This is the complement to BOS and MA Crossover. When they lose money
(choppy, range-bound markets), mean reversion thrives.

Core idea: prices oscillate around a mean. When price deviates too far
(measured by Bollinger Bands + RSI), bet on it reverting back.

Entry:
  LONG:  Price touches lower Bollinger Band AND RSI < 30 AND ADX < 25 (ranging)
  SHORT: Price touches upper Bollinger Band AND RSI > 70 AND ADX < 25 (ranging)

Exit:
  - Target: middle Bollinger Band (the mean)
  - Stop: 2 ATR beyond entry (in case it breaks out of range)
  - Timeout: close after max_hold_bars if neither target nor stop hit

The ADX < 25 filter is KEY — this strategy ONLY trades in ranging markets.
When ADX > 25 (trending), it sits in cash and lets BOS/MA Crossover work.
"""

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType
from strategies import BaseStrategy
from features import atr, rsi, bollinger_bands


def _adx(high, low, close, period=14):
    """ADX calculation (same as BOS v2)."""
    n = len(close)
    adx_vals = np.full(n, np.nan)
    if n < period * 2 + 1:
        return adx_vals

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0: plus_dm[i] = up
        if down > up and down > 0: minus_dm[i] = down

    atr_s = np.zeros(n)
    pdi_s = np.zeros(n)
    mdi_s = np.zeros(n)
    atr_s[period] = tr[1:period+1].mean()
    pdi_s[period] = plus_dm[1:period+1].mean()
    mdi_s[period] = minus_dm[1:period+1].mean()

    for i in range(period+1, n):
        atr_s[i] = (atr_s[i-1]*(period-1) + tr[i]) / period
        pdi_s[i] = (pdi_s[i-1]*(period-1) + plus_dm[i]) / period
        mdi_s[i] = (mdi_s[i-1]*(period-1) + minus_dm[i]) / period

    pdi = np.zeros(n)
    mdi = np.zeros(n)
    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] > 0:
            pdi[i] = 100 * pdi_s[i] / atr_s[i]
            mdi[i] = 100 * mdi_s[i] / atr_s[i]
        d = pdi[i] + mdi[i]
        if d > 0: dx[i] = 100 * abs(pdi[i] - mdi[i]) / d

    s = period * 2
    if s < n:
        adx_vals[s] = dx[period+1:s+1].mean()
        for i in range(s+1, n):
            adx_vals[i] = (adx_vals[i-1]*(period-1) + dx[i]) / period
    return adx_vals


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI mean reversion — only in ranging markets.

    Params:
        symbols:          List of symbols
        bb_period:        Bollinger Band lookback (default 20)
        bb_std:           Standard deviations for bands (default 2.0)
        rsi_period:       RSI lookback (default 14)
        rsi_oversold:     RSI level for long entry (default 30)
        rsi_overbought:   RSI level for short entry (default 70)
        adx_max:          Maximum ADX to trade (only range-bound) (default 25)
        atr_stop_mult:    ATR multiplier for stop loss (default 2.0)
        max_hold_bars:    Max bars to hold before timeout exit (default 40)
        min_bars:         Warmup period (default 50)
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 35,       # Was 30 — too strict, missed entries
            "rsi_overbought": 65,     # Was 70 — too strict
            "adx_max": 30,            # Was 25 — filtered 48% of bars
            "atr_stop_mult": 2.0,
            "max_hold_bars": 40,
            "min_bars": 50,
        }
        merged = {**default_params, **(params or {})}
        super().__init__("mean_reversion", event_bus, merged)

        self._in_position: Dict[str, str] = {}
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

        self._funnel = {
            "bars_processed": 0,
            "warmup_skip": 0,
            "in_position": 0,
            "no_signal": 0,
            "filtered_adx": 0,
            "filtered_rsi": 0,
            "long_entries": 0,
            "short_entries": 0,
            "exits_target": 0,
            "exits_stop": 0,
            "exits_timeout": 0,
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

        price = close[-1]
        upper_arr, middle_arr, lower_arr = bollinger_bands(close, self.params["bb_period"], self.params["bb_std"])
        upper_band = upper_arr[-1]
        lower_band = lower_arr[-1]
        middle_band = middle_arr[-1]
        rsi_val = rsi(close, self.params["rsi_period"])[-1]
        atr_val = atr(high, low, close, 14)[-1]
        adx_val = _adx(high, low, close, 14)[-1]

        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(adx_val):
            return None

        # --- Check exits first ---
        if symbol in self._in_position:
            self._funnel["in_position"] += 1
            self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1
            return self._check_exit(symbol, price, middle_band, atr_val)

        # --- Regime filter: only trade in ranges ---
        if adx_val > self.params["adx_max"]:
            self._funnel["filtered_adx"] += 1
            return None

        # --- Entry signals ---
        # Use proximity to band (within 10% of band distance) instead of exact touch
        band_width = upper_band - lower_band
        band_tolerance = band_width * 0.10  # Within 10% of the band

        # --- TREND-AWARE DIRECTION ---
        # The ensemble pushes HTF trend to us via _htf_trend[symbol].
        # In a downtrend, "buying the dip" is catching a falling knife.
        # In an uptrend, "shorting the rip" fights momentum.
        # Only take the direction aligned with macro trend.
        #
        # Bearish trend: only SHORT at upper band (price will revert DOWN)
        # Bullish trend: only LONG at lower band (price will revert UP)
        # Neutral: take both directions
        htf_trend = getattr(self, '_htf_trend', {}).get(symbol, "neutral")

        # Long: price near lower band + RSI oversold
        # ONLY in bullish or neutral trend — NOT in bearish (falling knife!)
        if (price <= lower_band + band_tolerance
                and rsi_val < self.params["rsi_oversold"]
                and htf_trend != "bearish"):
            stop = price - self.params["atr_stop_mult"] * atr_val
            self._in_position[symbol] = "long"
            self._entry_price[symbol] = price
            self._stop_price[symbol] = stop
            self._bars_held[symbol] = 0
            self._funnel["long_entries"] += 1

            return SignalEvent(
                symbol=symbol, signal_type=SignalType.LONG,
                strength=0.9 if htf_trend == "bullish" else 0.7,
                strategy_name=self.name,
                metadata={"entry": price, "stop": stop, "target": middle_band,
                          "rsi": rsi_val, "adx": adx_val, "band": "lower",
                          "htf_trend": htf_trend},
            )

        # Short: price near upper band + RSI overbought
        # ONLY in bearish or neutral trend — NOT in bullish (fighting momentum!)
        if (price >= upper_band - band_tolerance
                and rsi_val > self.params["rsi_overbought"]
                and htf_trend != "bullish"):
            stop = price + self.params["atr_stop_mult"] * atr_val
            self._in_position[symbol] = "short"
            self._entry_price[symbol] = price
            self._stop_price[symbol] = stop
            self._bars_held[symbol] = 0
            self._funnel["short_entries"] += 1

            return SignalEvent(
                symbol=symbol, signal_type=SignalType.SHORT,
                strength=0.9 if htf_trend == "bearish" else 0.7,
                strategy_name=self.name,
                metadata={"entry": price, "stop": stop, "target": middle_band,
                          "rsi": rsi_val, "adx": adx_val, "band": "upper",
                          "htf_trend": htf_trend},
            )

        # Track: if we had a band touch but trend blocked the wrong direction
        if price <= lower_band + band_tolerance and rsi_val < self.params["rsi_oversold"] and htf_trend == "bearish":
            self._funnel["trend_filtered"] = self._funnel.get("trend_filtered", 0) + 1
        if price >= upper_band - band_tolerance and rsi_val > self.params["rsi_overbought"] and htf_trend == "bullish":
            self._funnel["trend_filtered"] = self._funnel.get("trend_filtered", 0) + 1

        self._funnel["no_signal"] += 1
        return None

    def _check_exit(self, symbol, price, middle_band, atr_val):
        direction = self._in_position[symbol]
        stop = self._stop_price[symbol]
        entry = self._entry_price[symbol]
        bars_held = self._bars_held.get(symbol, 0)

        should_exit = False
        exit_reason = ""

        if direction == "long":
            if price >= middle_band:
                should_exit, exit_reason = True, "target"
            elif price <= stop:
                should_exit, exit_reason = True, "stop"
        elif direction == "short":
            if price <= middle_band:
                should_exit, exit_reason = True, "target"
            elif price >= stop:
                should_exit, exit_reason = True, "stop"

        if bars_held >= self.params["max_hold_bars"]:
            should_exit, exit_reason = True, "timeout"

        if should_exit:
            exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT
            self._funnel[f"exits_{exit_reason}"] += 1

            del self._in_position[symbol]
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._bars_held.pop(symbol, None)

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
        exits = f["exits_target"] + f["exits_stop"] + f["exits_timeout"]
        tf = f.get("trend_filtered", 0)
        print(f"\n  SIGNAL FUNNEL — {self.name}")
        print(f"  {'─'*50}")
        print(f"  Bars processed:           {f['bars_processed']:>8,}")
        print(f"    ├─ In position:         {f['in_position']:>8,}")
        print(f"    ├─ ADX too high:        {f['filtered_adx']:>8,}  (trending → skip)")
        print(f"    ├─ No band touch:       {f['no_signal']:>8,}")
        if tf > 0:
            print(f"    ├─ Trend aligned:       {tf:>8,}  (wrong dir for HTF trend)")
        print(f"    ├─ Enter LONG:          {f['long_entries']:>8,}  (lower band + RSI)")
        print(f"    └─ Enter SHORT:         {f['short_entries']:>8,}  (upper band + RSI)")
        print(f"  {'─'*50}")
        print(f"  Exits — target (mean):    {f['exits_target']:>8,}")
        print(f"  Exits — stop loss:        {f['exits_stop']:>8,}")
        print(f"  Exits — timeout:          {f['exits_timeout']:>8,}")
        print(f"  {'─'*50}")