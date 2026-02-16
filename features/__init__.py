"""Feature engineering module.

Computes derived features from raw OHLCV data that strategies use
for signal generation. This is the quantitative core of the system.

Features:
- Swing highs/lows (pivot points)
- Support & resistance levels
- Break of Structure (BOS) detection
- Volume analysis (relative volume, VWAP, volume profile)
- Volatility features (ATR, Bollinger, realized vol, vol regime)
- Trend features (ADX, higher highs/lows)
- Momentum (RSI, rate of change)

All functions work on numpy arrays or pandas DataFrames for speed.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum, auto


# ================================================================
#  Data structures
# ================================================================

class SwingType(Enum):
    HIGH = "swing_high"
    LOW = "swing_low"


class StructureBreak(Enum):
    BULLISH_BOS = "bullish_bos"     # Price breaks above a swing high
    BEARISH_BOS = "bearish_bos"     # Price breaks below a swing low
    BULLISH_CHOCH = "bullish_choch" # Change of character - bearish to bullish
    BEARISH_CHOCH = "bearish_choch" # Change of character - bullish to bearish
    NONE = "none"


@dataclass
class SwingPoint:
    """A swing high or swing low."""
    index: int
    price: float
    swing_type: SwingType
    timestamp: object = None
    strength: int = 1  # How many bars on each side confirm it


@dataclass
class SRLevel:
    """Support or resistance level."""
    price: float
    level_type: str  # "support" or "resistance"
    touches: int = 1       # Times price has tested this level
    strength: float = 0.0  # Strength score
    first_seen: int = 0    # Bar index when first identified
    last_tested: int = 0   # Bar index of last test


# ================================================================
#  Swing Point Detection
# ================================================================

def find_swing_points(
    high: np.ndarray,
    low: np.ndarray,
    left_bars: int = 5,
    right_bars: int = 5,
) -> List[SwingPoint]:
    """Detect swing highs and swing lows using left/right bar confirmation.

    A swing high is a bar whose high is higher than the highs of
    `left_bars` bars before it AND `right_bars` bars after it.
    Vice versa for swing lows.

    This is the foundation for market structure analysis.

    Args:
        high: Array of high prices
        low: Array of low prices
        left_bars: Bars to the left that must be lower/higher
        right_bars: Bars to the right that must be lower/higher

    Returns:
        List of SwingPoint objects sorted by index
    """
    n = len(high)
    swings = []

    for i in range(left_bars, n - right_bars):
        # Check swing high
        is_swing_high = True
        for j in range(1, left_bars + 1):
            if high[i - j] >= high[i]:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(1, right_bars + 1):
                if high[i + j] >= high[i]:
                    is_swing_high = False
                    break

        if is_swing_high:
            swings.append(SwingPoint(
                index=i, price=high[i],
                swing_type=SwingType.HIGH,
                strength=left_bars,
            ))

        # Check swing low
        is_swing_low = True
        for j in range(1, left_bars + 1):
            if low[i - j] <= low[i]:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(1, right_bars + 1):
                if low[i + j] <= low[i]:
                    is_swing_low = False
                    break

        if is_swing_low:
            swings.append(SwingPoint(
                index=i, price=low[i],
                swing_type=SwingType.LOW,
                strength=left_bars,
            ))

    return sorted(swings, key=lambda s: s.index)


def find_swing_points_adaptive(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    min_bars: int = 3,
    max_bars: int = 10,
) -> List[SwingPoint]:
    """Adaptive swing detection that uses ATR to adjust sensitivity.

    In high volatility: requires more confirmation bars (fewer swings)
    In low volatility: requires fewer bars (more responsive)
    """
    if len(atr) == 0:
        return find_swing_points(high, low, min_bars, min_bars)

    avg_atr = np.mean(atr[-20:]) if len(atr) >= 20 else np.mean(atr)
    current_atr = atr[-1]

    # Scale bars based on relative volatility
    vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
    bars = int(np.clip(min_bars * vol_ratio, min_bars, max_bars))

    return find_swing_points(high, low, bars, bars)


# ================================================================
#  Support & Resistance
# ================================================================

def find_support_resistance(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_points: List[SwingPoint] = None,
    tolerance_pct: float = 0.005,
    min_touches: int = 2,
    lookback: int = 100,
) -> List[SRLevel]:
    """Identify support and resistance levels from swing points.

    Groups nearby swing points into levels and scores them by:
    - Number of touches (more touches = stronger)
    - Recency (recent levels matter more)
    - Whether they're above (resistance) or below (support) current price

    Args:
        tolerance_pct: Price tolerance for grouping (0.5% default)
        min_touches: Minimum touches to qualify as a level
        lookback: How far back to look for swings
    """
    if swing_points is None:
        swing_points = find_swing_points(high, low)

    if not swing_points:
        return []

    current_price = close[-1]
    n = len(close)

    # Filter to lookback window
    recent_swings = [s for s in swing_points if s.index >= n - lookback]
    if not recent_swings:
        return []

    # Cluster nearby swing prices into levels
    swing_prices = sorted([s.price for s in recent_swings])
    levels = []
    used = set()

    for i, price in enumerate(swing_prices):
        if i in used:
            continue

        cluster = [price]
        cluster_indices = [i]

        for j in range(i + 1, len(swing_prices)):
            if j in used:
                continue
            if abs(swing_prices[j] - price) / price <= tolerance_pct:
                cluster.append(swing_prices[j])
                cluster_indices.append(j)
                used.add(j)

        used.add(i)

        if len(cluster) >= min_touches:
            avg_price = np.mean(cluster)

            # Find matching swings for metadata
            matching_swings = [
                s for s in recent_swings
                if abs(s.price - avg_price) / avg_price <= tolerance_pct
            ]

            level_type = "resistance" if avg_price > current_price else "support"

            # Strength = touches * recency_weight
            recency_weight = np.mean([
                1.0 / (1.0 + (n - s.index) / lookback)
                for s in matching_swings
            ])

            levels.append(SRLevel(
                price=avg_price,
                level_type=level_type,
                touches=len(cluster),
                strength=len(cluster) * recency_weight,
                first_seen=min(s.index for s in matching_swings),
                last_tested=max(s.index for s in matching_swings),
            ))

    return sorted(levels, key=lambda l: l.strength, reverse=True)


def nearest_support(levels: List[SRLevel], price: float) -> Optional[SRLevel]:
    """Find the nearest support level below current price."""
    supports = [l for l in levels if l.level_type == "support" and l.price < price]
    if not supports:
        return None
    return max(supports, key=lambda l: l.price)


def nearest_resistance(levels: List[SRLevel], price: float) -> Optional[SRLevel]:
    """Find the nearest resistance level above current price."""
    resistances = [l for l in levels if l.level_type == "resistance" and l.price > price]
    if not resistances:
        return None
    return min(resistances, key=lambda l: l.price)


# ================================================================
#  Break of Structure (BOS) Detection
# ================================================================

def detect_market_structure(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_points: List[SwingPoint] = None,
    left_bars: int = 5,
    right_bars: int = 2,
) -> Dict:
    """Analyze market structure and detect Break of Structure (BOS).

    Market structure concepts:
    - Uptrend: Higher Highs (HH) and Higher Lows (HL)
    - Downtrend: Lower Highs (LH) and Lower Lows (LL)
    - BOS: When price breaks the most recent swing in trend direction
    - CHoCH: Change of Character - first break against the trend

    Returns dict with:
    - trend: "bullish", "bearish", or "ranging"
    - structure_break: StructureBreak enum
    - swing_highs: recent swing highs
    - swing_lows: recent swing lows
    - higher_highs, higher_lows, lower_highs, lower_lows counts
    """
    if swing_points is None:
        swing_points = find_swing_points(high, low, left_bars, right_bars)

    if len(swing_points) < 4:
        return {
            "trend": "ranging",
            "structure_break": StructureBreak.NONE,
            "swing_highs": [],
            "swing_lows": [],
            "last_swing_high": None,
            "last_swing_low": None,
            "higher_highs": 0,
            "higher_lows": 0,
            "lower_highs": 0,
            "lower_lows": 0,
        }

    swing_highs = [s for s in swing_points if s.swing_type == SwingType.HIGH]
    swing_lows = [s for s in swing_points if s.swing_type == SwingType.LOW]

    # Count structure patterns in recent swings
    hh_count = 0  # Higher highs
    hl_count = 0  # Higher lows
    lh_count = 0  # Lower highs
    ll_count = 0  # Lower lows

    for i in range(1, len(swing_highs)):
        if swing_highs[i].price > swing_highs[i - 1].price:
            hh_count += 1
        else:
            lh_count += 1

    for i in range(1, len(swing_lows)):
        if swing_lows[i].price > swing_lows[i - 1].price:
            hl_count += 1
        else:
            ll_count += 1

    # Determine trend
    bullish_score = hh_count + hl_count
    bearish_score = lh_count + ll_count

    if bullish_score > bearish_score * 1.5:
        trend = "bullish"
    elif bearish_score > bullish_score * 1.5:
        trend = "bearish"
    else:
        trend = "ranging"

    # Detect Break of Structure
    # Use HIGH for bullish breaks and LOW for bearish breaks (wick counts)
    # but require close to be at least halfway past the level for confirmation
    # This catches genuine breakouts while filtering fakeout wicks
    current_close = close[-1]
    current_high = high[-1]
    current_low = low[-1]

    last_sh = swing_highs[-1] if swing_highs else None
    last_sl = swing_lows[-1] if swing_lows else None

    structure_break = StructureBreak.NONE

    if last_sh:
        level = last_sh.price
        # Bullish break: high pierces above swing high AND close confirms
        # Close must be above the midpoint between level and high
        # This filters wick-only fakeouts while still detecting legitimate breaks
        if current_high > level and current_close > level - (current_high - level) * 0.3:
            if trend == "bearish":
                structure_break = StructureBreak.BULLISH_CHOCH
            else:
                structure_break = StructureBreak.BULLISH_BOS

    if structure_break == StructureBreak.NONE and last_sl:
        level = last_sl.price
        # Bearish break: low pierces below swing low AND close confirms
        if current_low < level and current_close < level + (level - current_low) * 0.3:
            if trend == "bullish":
                structure_break = StructureBreak.BEARISH_CHOCH
            else:
                structure_break = StructureBreak.BEARISH_BOS

    return {
        "trend": trend,
        "structure_break": structure_break,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "last_swing_high": last_sh,
        "last_swing_low": last_sl,
        "higher_highs": hh_count,
        "higher_lows": hl_count,
        "lower_highs": lh_count,
        "lower_lows": ll_count,
    }


# ================================================================
#  Volatility Features
# ================================================================

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range — measures volatility."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr_values = np.zeros(n)
    atr_values[:period] = np.nan

    # First ATR is simple average
    if n >= period:
        atr_values[period - 1] = np.mean(tr[:period])

        # Smoothed ATR
        for i in range(period, n):
            atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    return atr_values


def bollinger_bands(
    close: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands — volatility envelope around moving average.

    Returns (upper_band, middle_band, lower_band)
    """
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        ma = window.mean()
        std = window.std()
        middle[i] = ma
        upper[i] = ma + std_dev * std
        lower[i] = ma - std_dev * std

    return upper, middle, lower


def realized_volatility(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Annualized realized volatility from log returns."""
    n = len(close)
    vol = np.full(n, np.nan)
    log_returns = np.diff(np.log(close))

    for i in range(period, n):
        window = log_returns[i - period: i]
        vol[i] = window.std() * np.sqrt(252)

    return vol


def volatility_regime(
    close: np.ndarray,
    short_period: int = 10,
    long_period: int = 50,
) -> np.ndarray:
    """Volatility regime: ratio of short-term to long-term vol.

    > 1.0 = expanding volatility (high regime)
    < 1.0 = contracting volatility (low regime)
    """
    short_vol = realized_volatility(close, short_period)
    long_vol = realized_volatility(close, long_period)

    with np.errstate(divide="ignore", invalid="ignore"):
        regime = np.where(long_vol > 0, short_vol / long_vol, 1.0)

    return regime


# ================================================================
#  Volume Features
# ================================================================

def relative_volume(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Relative volume — current volume vs average.

    RVOL > 1.5 = high volume (institutional interest)
    RVOL < 0.5 = low volume (lack of conviction)
    """
    n = len(volume)
    rvol = np.full(n, np.nan)

    for i in range(period, n):
        avg = np.mean(volume[i - period: i])
        rvol[i] = volume[i] / avg if avg > 0 else 1.0

    return rvol


def vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    timestamps: np.ndarray = None
) -> np.ndarray:
    """Volume Weighted Average Price — institutional benchmark.

    VWAP resets at the start of each trading session (9:30 AM ET).
    Without timestamps, it falls back to a rolling 7-bar window
    (approximating 1 trading day on 1h bars).

    Price above VWAP = bullish bias
    Price below VWAP = bearish bias
    """
    typical_price = (high + low + close) / 3.0
    n = len(close)
    vwap_values = np.full(n, np.nan)

    if timestamps is not None:
        # Session-based VWAP: reset at market open each day
        # Detect session boundaries: new session when date changes
        # or when hour resets to 9 (9:30 open)
        import pandas as pd
        ts = pd.DatetimeIndex(timestamps)
        dates = ts.date

        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_date = None

        for i in range(n):
            current_date = dates[i]

            # Reset at start of new trading day
            if current_date != prev_date:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_date = current_date

            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]

            if cum_vol > 0:
                vwap_values[i] = cum_tp_vol / cum_vol
            else:
                vwap_values[i] = close[i]
    else:
        # Fallback: rolling 7-bar window (approx 1 day on 1h bars)
        window = min(7, n)
        for i in range(n):
            start = max(0, i - window + 1)
            tp_slice = typical_price[start:i+1]
            vol_slice = volume[start:i+1]
            total_vol = vol_slice.sum()
            if total_vol > 0:
                vwap_values[i] = (tp_slice * vol_slice).sum() / total_vol
            else:
                vwap_values[i] = close[i]

    return vwap_values


def volume_profile(
    close: np.ndarray,
    volume: np.ndarray,
    n_bins: int = 20,
    lookback: int = 50,
) -> Dict:
    """Volume profile — distribution of volume at price levels.

    Returns:
        poc: Point of Control (price with most volume)
        value_area_high: Upper boundary of 70% volume area
        value_area_low: Lower boundary of 70% volume area
        profile: dict of price_level -> total_volume
    """
    recent_close = close[-lookback:]
    recent_vol = volume[-lookback:]

    price_min, price_max = recent_close.min(), recent_close.max()
    if price_max == price_min:
        return {"poc": price_min, "value_area_high": price_max,
                "value_area_low": price_min, "profile": {}}

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_volumes = np.zeros(n_bins)

    for i in range(len(recent_close)):
        bin_idx = np.searchsorted(bins, recent_close[i]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_volumes[bin_idx] += recent_vol[i]

    # Point of Control
    poc_idx = np.argmax(bin_volumes)
    poc = bin_centers[poc_idx]

    # Value Area (70% of volume)
    total_vol = bin_volumes.sum()
    target = total_vol * 0.7
    sorted_idx = np.argsort(bin_volumes)[::-1]
    cumulative = 0
    va_indices = []
    for idx in sorted_idx:
        va_indices.append(idx)
        cumulative += bin_volumes[idx]
        if cumulative >= target:
            break

    va_prices = bin_centers[va_indices]
    value_area_high = va_prices.max()
    value_area_low = va_prices.min()

    profile = {float(bin_centers[i]): float(bin_volumes[i]) for i in range(n_bins)}

    return {
        "poc": float(poc),
        "value_area_high": float(value_area_high),
        "value_area_low": float(value_area_low),
        "profile": profile,
    }


# ================================================================
#  Momentum / Trend Features
# ================================================================

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    n = len(close)
    rsi_values = np.full(n, np.nan)
    deltas = np.diff(close)

    if len(deltas) < period:
        return rsi_values

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values


def rate_of_change(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of change — momentum indicator."""
    n = len(close)
    roc = np.full(n, np.nan)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period]
    return roc


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(data)
    result = np.full(n, np.nan)
    if n < period:
        return result

    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])

    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]

    return result


# ================================================================
#  Convenience: Compute All Features at Once
# ================================================================

def compute_all_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Compute all features for a DataFrame with OHLCV columns.

    Input DataFrame must have: open, high, low, close, volume
    Returns DataFrame with all original + computed columns.

    This is the main entry point for feature engineering.
    """
    cfg = config or {}
    out = df.copy()

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values.astype(float)

    # --- Volatility ---
    out["atr_14"] = atr(h, l, c, 14)
    out["atr_7"] = atr(h, l, c, 7)
    bb_upper, bb_mid, bb_lower = bollinger_bands(c, 20, 2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1.0)
    out["realized_vol_20"] = realized_volatility(c, 20)
    out["vol_regime"] = volatility_regime(c, 10, 50)

    # --- Volume ---
    out["rvol_20"] = relative_volume(v, 20)
    # Pass timestamps for session-based VWAP if available
    ts = df.index.values if hasattr(df.index, 'values') else None
    out["vwap"] = vwap(h, l, c, v, ts)
    out["price_vs_vwap"] = c - out["vwap"].values

    # --- Momentum ---
    out["rsi_14"] = rsi(c, 14)
    out["rsi_7"] = rsi(c, 7)
    out["roc_10"] = rate_of_change(c, 10)
    out["ema_9"] = ema(c, 9)
    out["ema_21"] = ema(c, 21)
    out["ema_50"] = ema(c, 50)

    # --- Trend ---
    ema_9 = out["ema_9"].values
    ema_21 = out["ema_21"].values
    out["ema_cross"] = np.where(ema_9 > ema_21, 1, np.where(ema_9 < ema_21, -1, 0))

    # --- Candle features ---
    body = c - o
    full_range = h - l
    out["body_pct"] = np.where(full_range > 0, body / full_range, 0)
    out["upper_wick_pct"] = np.where(full_range > 0, (h - np.maximum(o, c)) / full_range, 0)
    out["lower_wick_pct"] = np.where(full_range > 0, (np.minimum(o, c) - l) / full_range, 0)

    return out