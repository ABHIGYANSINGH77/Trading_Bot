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

    For 15m bars with multiplier=4, this creates 1h bars.
    Returns "bullish", "bearish", or "neutral".
    """
    if len(bars) < multiplier * 10:
        return "neutral"

    # Aggregate bars into higher timeframe
    htf_close = []
    for i in range(0, len(bars) - multiplier + 1, multiplier):
        chunk = bars[i:i+multiplier]
        htf_close.append(chunk[-1]["close"])

    if len(htf_close) < 20:
        return "neutral"

    htf = np.array(htf_close)

    # Simple trend: is 10-period MA above or below 20-period MA?
    fast = htf[-10:].mean()
    slow = htf[-20:].mean()

    if fast > slow * 1.002:
        return "bullish"
    elif fast < slow * 0.998:
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
    # IMPORTANT: On small accounts ($10k), weights below 0.15 cause position
    # sizes to fall below min_order_value and get dropped. The minimum
    # weight for a strategy that should still trade is 0.15 (15%).
    # Strategies that should NOT trade in a regime get 0.0.
    ALLOCATIONS = {
        "trending": {
            "bos": 0.45,
            "ma_crossover": 0.35,
            "mean_reversion": 0.0,       # Don't trade mean reversion in trends
            "vol_breakout": 0.20,
        },
        "ranging": {
            "bos": 0.0,                  # Don't trade breakouts in ranges
            "ma_crossover": 0.0,         # Don't trade trend-following in ranges
            "mean_reversion": 0.65,
            "vol_breakout": 0.35,
        },
        "volatile": {
            "bos": 0.15,
            "ma_crossover": 0.0,
            "mean_reversion": 0.20,
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
        # Volatile: significant vol expansion
        elif vol_ratio > 1.8:
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

    def _on_market_data_ensemble(self, event: MarketDataEvent):
        """Track bars for regime detection."""
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

        # Detect regime
        bars = self._bar_history[symbol]
        if len(bars) >= 60:
            high = np.array([b["high"] for b in bars])
            low = np.array([b["low"] for b in bars])
            close = np.array([b["close"] for b in bars])
            regime = self.allocator.detect_regime(high, low, close)
            self._regime_counts[regime] += 1

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

            # Add higher-timeframe context to metadata
            symbol = event.symbol
            bars = self._bar_history.get(symbol, [])
            htf_trend = _higher_tf_trend(bars, multiplier=4)
            event.metadata["htf_trend"] = htf_trend

            # Additional filter: if signal direction conflicts with HTF trend, reduce strength
            if htf_trend != "neutral":
                is_long = event.signal_type in (SignalType.LONG,)
                is_short = event.signal_type in (SignalType.SHORT,)

                if (is_long and htf_trend == "bearish") or (is_short and htf_trend == "bullish"):
                    event.strength *= 0.3  # Heavily reduce counter-trend signals
                    event.metadata["htf_conflict"] = True
                elif (is_long and htf_trend == "bullish") or (is_short and htf_trend == "bearish"):
                    event.strength = min(event.strength * 1.3, 1.0)  # Boost aligned signals
                    event.metadata["htf_aligned"] = True

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
        print(f"    Signals scaled:  {self._signals_scaled:>6}")
        print(f"    Signals blocked: {self._signals_blocked:>6}  (wrong regime)")

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