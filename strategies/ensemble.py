"""Strategy Allocator — Level 3 Meta-Strategy.

This is the "fund manager" layer that sits above individual strategies.
It doesn't generate its own signals — instead it:

  1. Detects the current market regime (trending / ranging / volatile)
  2. Allocates capital to the strategy best suited for that regime
  3. Adjusts signal strength to scale position sizes

Market Regimes:
  TRENDING   (ADX > 25, vol normal)   → BOS gets 60%, MA Crossover 30%, others 10%
  RANGING    (ADX < 20, vol normal)   → Mean Reversion gets 70%, others 10%
  VOLATILE   (vol expanding rapidly)  → Vol Breakout gets 50%, reduced sizing overall
  CRISIS     (vol > 2x normal, DD)    → All strategies get 0% — sit in cash

This is what separates retail traders from institutional quants.
A single strategy can't work in all regimes. The allocator ensures
you're always running the RIGHT strategy for current conditions.

Usage:
    python main.py backtest -s ensemble -i 15m -d ibkr --start 2025-06-01 --end 2025-12-01

The ensemble strategy wraps all 4 sub-strategies, intercepts their signals,
and scales them based on the allocator's regime detection.
"""

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType
from strategies import BaseStrategy
from features import atr, bollinger_bands, realized_volatility


def _adx(high, low, close, period=14):
    """ADX — trend strength indicator."""
    n = len(close)
    adx_vals = np.full(n, np.nan)
    if n < period * 2 + 1:
        return adx_vals

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    plus_dm, minus_dm = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        up, down = high[i] - high[i-1], low[i-1] - low[i]
        if up > down and up > 0: plus_dm[i] = up
        if down > up and down > 0: minus_dm[i] = down

    atr_s, pdi_s, mdi_s = np.zeros(n), np.zeros(n), np.zeros(n)
    atr_s[period] = tr[1:period+1].mean()
    pdi_s[period] = plus_dm[1:period+1].mean()
    mdi_s[period] = minus_dm[1:period+1].mean()

    for i in range(period+1, n):
        atr_s[i] = (atr_s[i-1]*(period-1) + tr[i]) / period
        pdi_s[i] = (pdi_s[i-1]*(period-1) + plus_dm[i]) / period
        mdi_s[i] = (mdi_s[i-1]*(period-1) + minus_dm[i]) / period

    pdi, mdi, dx = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(period, n):
        if atr_s[i] > 0:
            pdi[i] = 100*pdi_s[i]/atr_s[i]
            mdi[i] = 100*mdi_s[i]/atr_s[i]
        d = pdi[i] + mdi[i]
        if d > 0: dx[i] = 100*abs(pdi[i]-mdi[i])/d

    s = period * 2
    if s < n:
        adx_vals[s] = dx[period+1:s+1].mean()
        for i in range(s+1, n):
            adx_vals[i] = (adx_vals[i-1]*(period-1) + dx[i]) / period
    return adx_vals


def _higher_tf_trend(bars: list, multiplier: int = 4) -> str:
    """Build higher-timeframe bars and determine trend direction.

    For 1h bars with multiplier=4, this creates ~4h bars.
    Uses multiple timeframes for confirmation:
      - Short-term: 10-bar vs 20-bar MA (recent direction)
      - Medium-term: 20-bar vs 50-bar MA (established trend)
    Both must agree for a confident trend call.
    Returns "bullish", "bearish", or "neutral".
    """
    if len(bars) < multiplier * 50:
        return "neutral"

    # Aggregate bars into higher timeframe
    htf_close = []
    for i in range(0, len(bars) - multiplier + 1, multiplier):
        chunk = bars[i:i+multiplier]
        htf_close.append(chunk[-1]["close"])

    if len(htf_close) < 50:
        return "neutral"

    htf = np.array(htf_close)

    # Short-term trend
    fast_s = htf[-10:].mean()
    slow_s = htf[-20:].mean()

    # Medium-term trend (the macro trend)
    fast_m = htf[-20:].mean()
    slow_m = htf[-50:].mean()

    # Also check: is current price above or below the 50-period MA?
    price_vs_50 = htf[-1] / slow_m - 1.0

    short_bull = fast_s > slow_s * 1.001
    short_bear = fast_s < slow_s * 0.999
    medium_bull = fast_m > slow_m * 1.002
    medium_bear = fast_m < slow_m * 0.998

    # Strong bullish: both timeframes agree + price above 50 MA
    if medium_bull and (short_bull or price_vs_50 > 0.02):
        return "bullish"
    # Strong bearish: both agree + price below 50 MA
    elif medium_bear and (short_bear or price_vs_50 < -0.02):
        return "bearish"
    # Weak trend: only short-term signal
    elif short_bull and price_vs_50 > 0:
        return "bullish"
    elif short_bear and price_vs_50 < 0:
        return "bearish"
    return "neutral"


class StrategyAllocator:
    """Detects market regime and computes allocation weights."""

    # Regime definitions
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"

    # Allocation weights per regime: {strategy_name: weight}
    # With trend direction filter, BOS and MA only trade WITH the trend.
    # Weights below 0.15 on $10k accounts cause min_order drops.
    # Strategies get 0% in wrong-regime to avoid micro-positions.
    ALLOCATIONS = {
        "trending": {
            "bos": 0.35,
            "ma_crossover": 0.20,       # Reduced from 35% — still trades but less dominant
            "mean_reversion": 0.0,       # Don't mean-revert in trends
            "vol_breakout": 0.30,        # Increased — best performer
        },
        "ranging": {
            "bos": 0.0,
            "ma_crossover": 0.0,
            "mean_reversion": 0.50,      # Reduced from 65% — leave room
            "vol_breakout": 0.50,        # Increased from 35%
        },
        "volatile": {
            "bos": 0.15,
            "ma_crossover": 0.0,
            "mean_reversion": 0.25,      # Increased from 20%
            "vol_breakout": 0.45,
            "_size_scale": 0.6,
        },
        "crisis": {
            "bos": 0.0,
            "ma_crossover": 0.0,
            "mean_reversion": 0.0,
            "vol_breakout": 0.0,
            "_size_scale": 0.0,
        },
    }

    def __init__(self, adx_trending: float = 25, adx_ranging: float = 18,
                 vol_crisis_mult: float = 2.5):
        self.adx_trending = adx_trending
        self.adx_ranging = adx_ranging
        self.vol_crisis_mult = vol_crisis_mult
        self.current_regime = "unknown"
        self.regime_history = []

    def detect_regime(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> str:
        """Detect current market regime from price data."""
        if len(close) < 60:
            return self.RANGING

        adx_vals = _adx(high, low, close, 14)
        current_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 15

        # Volatility: compare short-term to long-term
        short_vol = realized_volatility(close, 10)[-1]
        long_vol = realized_volatility(close, 50)[-1] if len(close) >= 51 else short_vol

        if np.isnan(short_vol): short_vol = 0
        if np.isnan(long_vol) or long_vol == 0: long_vol = short_vol if short_vol > 0 else 0.01

        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

        # Crisis: vol expanding rapidly beyond normal
        if vol_ratio > self.vol_crisis_mult:
            regime = self.CRISIS
        # Volatile: significant vol expansion (was 1.8, too high — NVDA regularly hits 1.4)
        elif vol_ratio > 1.4:
            # Can be volatile AND trending — check ADX too
            if current_adx > self.adx_trending:
                # High trend + high vol: reduce sizing, keep trend strategies
                regime = self.VOLATILE  # Will use volatile weights with _size_scale=0.6
            else:
                regime = self.VOLATILE
        # Trending: strong directional movement
        elif current_adx > self.adx_trending:
            regime = self.TRENDING
        # Ranging: low directional movement
        elif current_adx < self.adx_ranging:
            regime = self.RANGING
        else:
            # In between — use vol as tiebreaker
            regime = self.TRENDING if vol_ratio > 1.2 else self.RANGING

        self.current_regime = regime
        self.regime_history.append(regime)
        return regime

    def get_weights(self, regime: str = None) -> Dict[str, float]:
        """Get strategy allocation weights for a regime."""
        r = regime or self.current_regime
        return self.ALLOCATIONS.get(r, self.ALLOCATIONS[self.RANGING])

    def scale_signal(self, signal: SignalEvent, regime: str = None) -> SignalEvent:
        """Scale a signal's strength based on regime allocation."""
        r = regime or self.current_regime
        weights = self.get_weights(r)
        strategy_weight = weights.get(signal.strategy_name, 0.1)
        size_scale = weights.get("_size_scale", 1.0)

        # Scale signal strength: original_strength × strategy_weight × regime_scale
        new_strength = signal.strength * strategy_weight * size_scale
        signal.strength = max(0.01, min(new_strength, 1.0))

        # Store regime info in metadata
        signal.metadata["regime"] = r
        signal.metadata["allocation_weight"] = strategy_weight
        signal.metadata["original_strength"] = signal.strength

        return signal


class EnsembleStrategy(BaseStrategy):
    """Multi-strategy ensemble with regime-based allocation.

    Runs BOS, MA Crossover, Mean Reversion, and Vol Breakout simultaneously.
    The StrategyAllocator detects the regime each bar and scales signals.

    In TRENDING regime:  BOS and MA Crossover get most capital
    In RANGING regime:   Mean Reversion dominates
    In VOLATILE regime:  Vol Breakout dominates, sizing reduced
    In CRISIS regime:    Everything goes to cash

    This is the key insight: you don't need a strategy that works in
    all conditions. You need multiple strategies + an intelligent switch.
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            "interval": "15m",
            # Sub-strategy parameters can be overridden
            "bos_params": {},
            "ma_params": {},
            "mr_params": {},
            "vb_params": {},
        }
        merged = {**default_params, **(params or {})}

        # Don't call BaseStrategy.__init__ with event_bus subscription
        # because sub-strategies will subscribe themselves
        self.name = "ensemble"
        self.event_bus = event_bus
        self.params = merged
        self._is_active = True
        self._price_history = defaultdict(list)
        self._bar_history = defaultdict(list)

        symbols = merged.get("symbols", ["AAPL"])
        interval = merged.get("interval", "15m")

        # Create sub-strategies
        from strategies.bos_strategy import BOSStrategy
        from strategies.ma_crossover import MACrossoverStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.vol_breakout import VolatilityBreakoutStrategy

        bos_p = {"symbols": symbols, "interval": interval, **merged.get("bos_params", {})}
        ma_p = {"symbols": symbols, "interval": interval, **merged.get("ma_params", {})}
        mr_p = {"symbols": symbols, "interval": interval, **merged.get("mr_params", {})}
        vb_p = {"symbols": symbols, "interval": interval, **merged.get("vb_params", {})}

        self._sub_strategies = {
            "bos": BOSStrategy(event_bus, bos_p),
            "ma_crossover": MACrossoverStrategy(event_bus, ma_p),
            "mean_reversion": MeanReversionStrategy(event_bus, mr_p),
            "vol_breakout": VolatilityBreakoutStrategy(event_bus, vb_p),
        }

        # Allocator
        self.allocator = StrategyAllocator()

        # Intercept signals: unsubscribe sub-strategies' signal publishing
        # and instead route through the allocator
        self._original_publish = event_bus.publish
        event_bus.publish = self._intercept_publish

        # Track our own bars for regime detection
        event_bus.subscribe(EventType.MARKET_DATA, self._on_market_data_ensemble)

        # Diagnostics
        self._regime_counts = defaultdict(int)
        self._signals_by_regime = defaultdict(lambda: defaultdict(int))
        self._signals_scaled = 0
        self._signals_blocked = 0
        self._signals_trend_blocked = 0
        self._htf_trend_per_symbol: Dict[str, str] = {}

        # --- CONFLUENCE FILTER STATE ---
        # VWAP: session-based, resets daily (institutional benchmark)
        self._current_vwap: Dict[str, float] = {}
        self._vwap_session_date: Dict[str, object] = {}
        self._vwap_cum_tp_vol: Dict[str, float] = {}
        self._vwap_cum_vol: Dict[str, float] = {}
        # Volume: 20-bar rolling average for confirmation
        self._volume_history: Dict[str, list] = defaultdict(list)
        self._avg_volume: Dict[str, float] = {}
        # Time tracking for time-of-day filter
        self._current_bar_time: Dict[str, object] = {}
        # Confluence diagnostics
        self._confluence_vwap_blocked = 0
        self._confluence_volume_blocked = 0
        self._confluence_lunch_blocked = 0
        self._confluence_rr_blocked = 0        # NEW: bad risk:reward ratio
        self._confluence_passed = 0

    def _on_market_data_ensemble(self, event: MarketDataEvent):
        """Track bars for regime detection, trend direction, and confluence filters."""
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

        # --- CONFLUENCE: Compute session VWAP per symbol ---
        # Reset at start of each trading day (session-based, like institutions)
        bars = self._bar_history[symbol]
        ts = event.bar_timestamp or event.timestamp
        bar_date = getattr(ts, 'date', lambda: None)()

        if bar_date and bar_date != self._vwap_session_date.get(symbol):
            # New trading day — reset VWAP accumulators
            self._vwap_session_date[symbol] = bar_date
            self._vwap_cum_tp_vol[symbol] = 0.0
            self._vwap_cum_vol[symbol] = 0.0

        tp = (event.high + event.low + event.close) / 3.0
        self._vwap_cum_tp_vol[symbol] = self._vwap_cum_tp_vol.get(symbol, 0.0) + tp * event.volume
        self._vwap_cum_vol[symbol] = self._vwap_cum_vol.get(symbol, 0.0) + event.volume

        if self._vwap_cum_vol[symbol] > 0:
            self._current_vwap[symbol] = self._vwap_cum_tp_vol[symbol] / self._vwap_cum_vol[symbol]
        else:
            self._current_vwap[symbol] = event.close

        # --- CONFLUENCE: Track rolling volume average (20-bar) ---
        vol_history = self._volume_history[symbol]
        vol_history.append(event.volume)
        if len(vol_history) > 20:
            vol_history.pop(0)
        self._avg_volume[symbol] = sum(vol_history) / len(vol_history) if vol_history else 1.0

        # Store current bar time for time-of-day filter
        self._current_bar_time[symbol] = ts

        # Detect regime
        if len(bars) >= 60:
            high = np.array([b["high"] for b in bars])
            low = np.array([b["low"] for b in bars])
            close = np.array([b["close"] for b in bars])
            regime = self.allocator.detect_regime(high, low, close)
            self._regime_counts[regime] += 1

        # Track HTF trend direction per symbol
        htf_trend = _higher_tf_trend(bars, multiplier=4)
        self._htf_trend_per_symbol[symbol] = htf_trend

        # --- PUSH HTF TREND TO SUB-STRATEGIES ---
        # Instead of blocking counter-trend signals at the ensemble level,
        # let each strategy KNOW the trend so it generates the RIGHT direction.
        # This is the key insight: don't filter — INFORM.
        for strat in self._sub_strategies.values():
            if not hasattr(strat, '_htf_trend'):
                strat._htf_trend = {}
            strat._htf_trend[symbol] = htf_trend

    def _intercept_publish(self, event):
        """Intercept signal events and scale them based on regime."""
        if (hasattr(event, 'event_type') and event.event_type == EventType.SIGNAL
                and hasattr(event, 'strategy_name')
                and event.strategy_name in self._sub_strategies):

            regime = self.allocator.current_regime
            weights = self.allocator.get_weights(regime)
            strategy_weight = weights.get(event.strategy_name, 0.1)

            # Block signal if strategy has 0 allocation in this regime
            if strategy_weight <= 0.01:
                self._signals_blocked += 1
                return  # Don't publish

            # Scale signal
            event = self.allocator.scale_signal(event, regime)
            self._signals_scaled += 1
            self._signals_by_regime[regime][event.strategy_name] += 1

            # --- TREND DIRECTION FILTER ---
            # For trend-following strategies (BOS, MA Crossover):
            # HARD BLOCK signals that go against the macro trend
            # Mean Reversion and Vol Breakout are exempt (they can counter-trend)
            symbol = event.symbol
            bars = self._bar_history.get(symbol, [])
            htf_trend = self._htf_trend_per_symbol.get(symbol, "neutral")
            event.metadata["htf_trend"] = htf_trend

            is_long = event.signal_type in (SignalType.LONG,)
            is_short = event.signal_type in (SignalType.SHORT,)
            is_trend_strategy = event.strategy_name in ("bos", "ma_crossover")

            if is_trend_strategy and htf_trend != "neutral":
                # BLOCK counter-trend entries entirely
                if (is_long and htf_trend == "bearish") or (is_short and htf_trend == "bullish"):
                    self._signals_trend_blocked += 1
                    return  # Don't publish — this is the key fix

                # BOOST trend-aligned signals
                if (is_long and htf_trend == "bullish") or (is_short and htf_trend == "bearish"):
                    event.strength = min(event.strength * 1.3, 1.0)
                    event.metadata["htf_aligned"] = True

            # For non-trend strategies (mean_reversion, vol_breakout):
            # These strategies now have HTF trend passed to them and handle
            # direction alignment internally. We don't block here — we let
            # them generate the RIGHT direction signal in the first place.
            # Boost trend-aligned signals from non-trend strategies.
            elif not is_trend_strategy and htf_trend != "neutral":
                if (is_long and htf_trend == "bullish") or (is_short and htf_trend == "bearish"):
                    event.strength = min(event.strength * 1.3, 1.0)
                    event.metadata["htf_aligned"] = True

            # ═══════════════════════════════════════════════════════════
            #  CONFLUENCE FILTERS — Applied to ALL entry signals
            #  These are what prop firms use to filter out low-quality entries.
            #  Exits are ALWAYS allowed through (never block closing a position).
            # ═══════════════════════════════════════════════════════════
            is_entry = event.signal_type in (SignalType.LONG, SignalType.SHORT)

            if is_entry:
                # --- 1. VWAP DIRECTIONAL GATE ---
                # The #1 institutional intraday filter:
                #   LONG only if price > VWAP (buyers in control)
                #   SHORT only if price < VWAP (sellers in control)
                # This alone filters ~30% of bad signals.
                current_vwap = self._current_vwap.get(symbol, 0)
                if current_vwap > 0:
                    current_price = self._bar_history[symbol][-1]["close"] if self._bar_history[symbol] else 0
                    if current_price > 0:
                        # Allow a small tolerance (0.1% of VWAP) to avoid
                        # blocking signals right at the VWAP line
                        vwap_tolerance = current_vwap * 0.001

                        if is_long and current_price < (current_vwap - vwap_tolerance):
                            self._confluence_vwap_blocked += 1
                            event.metadata["blocked_reason"] = "vwap_gate"
                            return  # BLOCKED: trying to go long below VWAP

                        if is_short and current_price > (current_vwap + vwap_tolerance):
                            self._confluence_vwap_blocked += 1
                            event.metadata["blocked_reason"] = "vwap_gate"
                            return  # BLOCKED: trying to go short above VWAP

                # --- 2. VOLUME CONFIRMATION ---
                # Only enter when current bar volume > 1.2x 20-bar average.
                # Low volume = thin market, signals are noise.
                # Prop firms never trade in thin conditions.
                if self._bar_history[symbol]:
                    current_vol = self._bar_history[symbol][-1]["volume"]
                    avg_vol = self._avg_volume.get(symbol, 0)
                    vol_threshold = 1.2  # 20% above average

                    if avg_vol > 0 and current_vol < avg_vol * vol_threshold:
                        self._confluence_volume_blocked += 1
                        event.metadata["blocked_reason"] = "low_volume"
                        return  # BLOCKED: insufficient volume conviction

                # --- 3. TIME-OF-DAY FILTER ---
                # Avoid entries during lunch lull (12:00-13:00 ET).
                # Low participation = unreliable signals.
                # Best entries: 09:30-11:30 (open drive) and 14:00-15:15 (close drive)
                bar_time = self._current_bar_time.get(symbol)
                if bar_time is not None:
                    bar_hour = getattr(bar_time, 'hour', None)
                    if bar_hour is not None and bar_hour == 12:
                        self._confluence_lunch_blocked += 1
                        event.metadata["blocked_reason"] = "lunch_lull"
                        return  # BLOCKED: lunch hour — low volume, choppy

                # All confluence checks passed
                self._confluence_passed += 1
                event.metadata["confluence_ok"] = True
                event.metadata["vwap"] = round(self._current_vwap.get(symbol, 0), 2)

                # --- 4. MINIMUM RISK:REWARD GATE ---
                # Don't enter trades where the potential reward doesn't
                # justify the risk. Require at least 1.5:1 R:R.
                # This catches mean_reversion trades where target (middle band)
                # is closer than the stop — inverted R:R.
                entry_price = self._bar_history[symbol][-1]["close"] if self._bar_history[symbol] else 0
                stop_price = event.metadata.get("stop", 0)
                target_price = event.metadata.get("target", 0)
                if entry_price > 0 and stop_price > 0 and target_price > 0:
                    risk = abs(entry_price - stop_price)
                    reward = abs(target_price - entry_price)
                    if risk > 0:
                        rr_ratio = reward / risk
                        event.metadata["rr_ratio"] = round(rr_ratio, 2)
                        if rr_ratio < 1.5:
                            self._confluence_rr_blocked += 1
                            event.metadata["blocked_reason"] = f"bad_rr_{rr_ratio:.1f}"
                            # Undo the passed count
                            self._confluence_passed -= 1
                            return  # BLOCKED: risk > reward

        # Publish via original method
        self._original_publish(event)

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """Not used — sub-strategies generate their own signals."""
        return None

    def get_diagnostics(self) -> Dict:
        return {
            "regime": self.allocator.current_regime,
            "regime_counts": dict(self._regime_counts),
            "signals_scaled": self._signals_scaled,
            "signals_blocked": self._signals_blocked,
            "signals_by_regime": {r: dict(s) for r, s in self._signals_by_regime.items()},
            "confluence_vwap_blocked": self._confluence_vwap_blocked,
            "confluence_volume_blocked": self._confluence_volume_blocked,
            "confluence_lunch_blocked": self._confluence_lunch_blocked,
            "confluence_rr_blocked": self._confluence_rr_blocked,
            "confluence_passed": self._confluence_passed,
        }

    def print_funnel(self):
        total_bars = sum(self._regime_counts.values())
        print(f"\n  ENSEMBLE ALLOCATOR")
        print(f"  {'─'*55}")
        print(f"  Regime Detection ({total_bars} bars):")
        for regime in ["trending", "ranging", "volatile", "crisis", "unknown"]:
            count = self._regime_counts.get(regime, 0)
            pct = count / total_bars * 100 if total_bars > 0 else 0
            bar = "█" * int(pct / 2)
            if count > 0:
                print(f"    {regime:>10}: {count:>6} ({pct:>5.1f}%) {bar}")

        print(f"\n  Signal Routing:")
        print(f"    Signals scaled:      {self._signals_scaled:>6}")
        print(f"    Signals blocked:     {self._signals_blocked:>6}  (wrong regime)")
        print(f"    Trend blocked:       {self._signals_trend_blocked:>6}  (counter-trend BOS/MA)")

        # Confluence filter stats
        total_conf_blocked = (self._confluence_vwap_blocked +
                              self._confluence_volume_blocked +
                              self._confluence_lunch_blocked +
                              self._confluence_rr_blocked)
        if total_conf_blocked > 0 or self._confluence_passed > 0:
            print(f"\n  Confluence Filters (prop firm quality gate):")
            print(f"    VWAP gate blocked:   {self._confluence_vwap_blocked:>6}  (long below VWAP / short above)")
            print(f"    Volume blocked:      {self._confluence_volume_blocked:>6}  (vol < 1.2x avg)")
            print(f"    Lunch lull blocked:  {self._confluence_lunch_blocked:>6}  (12:00-13:00 ET)")
            print(f"    Bad R:R blocked:     {self._confluence_rr_blocked:>6}  (reward/risk < 1.5)")
            print(f"    ──────────────────────────────")
            print(f"    Total blocked:       {total_conf_blocked:>6}")
            print(f"    Entries passed:      {self._confluence_passed:>6}  (high-quality signals)")
            if self._confluence_passed + total_conf_blocked > 0:
                pass_rate = self._confluence_passed / (self._confluence_passed + total_conf_blocked) * 100
                print(f"    Pass rate:           {pass_rate:>5.1f}%")

        # Show current trend direction per symbol
        if self._htf_trend_per_symbol:
            trends = ", ".join(f"{s}={t}" for s, t in sorted(self._htf_trend_per_symbol.items()))
            print(f"    HTF trend:           {trends}")

        if self._signals_by_regime:
            print(f"\n  Signals by Regime × Strategy:")
            for regime, strats in sorted(self._signals_by_regime.items()):
                strat_list = ", ".join(f"{s}={c}" for s, c in sorted(strats.items()))
                print(f"    {regime:>10}: {strat_list}")

        # Print each sub-strategy's funnel
        for name, strat in self._sub_strategies.items():
            if hasattr(strat, "print_funnel"):
                strat.print_funnel()

        print(f"\n  ALLOCATION WEIGHTS:")
        print(f"  {'Strategy':<18} {'Trending':>10} {'Ranging':>10} {'Volatile':>10} {'Crisis':>10}")
        print(f"  {'─'*58}")
        for strat in ["bos", "ma_crossover", "mean_reversion", "vol_breakout"]:
            t = StrategyAllocator.ALLOCATIONS["trending"].get(strat, 0)
            r = StrategyAllocator.ALLOCATIONS["ranging"].get(strat, 0)
            v = StrategyAllocator.ALLOCATIONS["volatile"].get(strat, 0)
            c = StrategyAllocator.ALLOCATIONS["crisis"].get(strat, 0)
            print(f"  {strat:<18} {t:>9.0%} {r:>10.0%} {v:>10.0%} {c:>10.0%}")
        print(f"  {'─'*58}")

    def stop(self):
        self._is_active = False
        for s in self._sub_strategies.values():
            if hasattr(s, 'stop'):
                s.stop()

    def start(self):
        self._is_active = True
        for s in self._sub_strategies.values():
            if hasattr(s, 'start'):
                s.start()