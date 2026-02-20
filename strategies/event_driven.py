"""Event-Driven Intraday Strategy.

instead of scanning every bar with indicators, we WAIT for specific market events, then CONTEXTUALIZE and SCORE them.

Events (the TRIGGER — rare, structural):
  1. Opening Range Breakout/Failure (ORB)
  2. Prior Session High/Low Sweep + Rejection

Context Features (the FILTER — repurposed indicators):
  - MA alignment, RSI, VWAP position, volume ratio
  - ATR percentile, time of day, gap direction
  - These DON'T generate signals. They score events.

Usage:
    python main.py backtest -s event_driven -i 15m -d ibkr --start 2025-06-01 --end 2025-12-01
"""

from collections import defaultdict
from datetime import time, date, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType
from strategies import BaseStrategy
from features import atr, rsi, ema, relative_volume


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE — tracks daily levels per symbol
# ═══════════════════════════════════════════════════════════════

class SessionState:
    """Tracks per-symbol, per-day session data."""

    def __init__(self):
        self.current_date: Optional[date] = None

        # Prior day levels (set at start of new day)
        self.prior_day_high: float = 0.0
        self.prior_day_low: float = 0.0
        self.prior_day_close: float = 0.0
        self.prior_day_range: float = 0.0

        # Today's session tracking
        self.today_bars: List[dict] = []
        self.today_high: float = 0.0
        self.today_low: float = float('inf')

        # Opening range (first N minutes)
        self.orb_high: float = 0.0
        self.orb_low: float = 0.0
        self.orb_defined: bool = False
        self.orb_volume: float = 0.0
        self.orb_bar_count: int = 0

        # Event flags (fire once per day)
        self.orb_breakout_fired: bool = False
        self.orb_failure_fired: bool = False
        self.pdh_sweep_fired: bool = False
        self.pdl_sweep_fired: bool = False

        # Sweep tracking
        self.pdh_sweep_detected: bool = False
        self.pdh_sweep_bar_idx: int = 0
        self.pdh_sweep_extreme: float = 0.0
        self.pdh_sweep_confirmed: bool = False
        self.pdl_sweep_detected: bool = False
        self.pdl_sweep_bar_idx: int = 0
        self.pdl_sweep_extreme: float = 0.0
        self.pdl_sweep_confirmed: bool = False

        # VWAP
        self.vwap_cum_tp_vol: float = 0.0
        self.vwap_cum_vol: float = 0.0
        self.current_vwap: float = 0.0

        # Historical ORB widths for relative comparison
        self.orb_history: List[float] = []

    def new_day(self, prior_bars: List[dict]):
        """Reset for new trading day. prior_bars = yesterday's bars."""
        if prior_bars:
            highs = [b["high"] for b in prior_bars]
            lows = [b["low"] for b in prior_bars]
            self.prior_day_high = max(highs)
            self.prior_day_low = min(lows)
            self.prior_day_close = prior_bars[-1]["close"]
            self.prior_day_range = self.prior_day_high - self.prior_day_low

            # Store ORB if it was defined
            if self.orb_defined:
                self.orb_history.append(self.orb_high - self.orb_low)
                if len(self.orb_history) > 20:
                    self.orb_history.pop(0)

        self.today_bars = []
        self.today_high = 0.0
        self.today_low = float('inf')
        self.orb_high = 0.0
        self.orb_low = 0.0
        self.orb_defined = False
        self.orb_volume = 0.0
        self.orb_bar_count = 0
        self.orb_breakout_fired = False
        self.orb_failure_fired = False
        self.pdh_sweep_fired = False
        self.pdl_sweep_fired = False
        self.pdh_sweep_detected = False
        self.pdl_sweep_detected = False
        self.vwap_cum_tp_vol = 0.0
        self.vwap_cum_vol = 0.0

    def update_bar(self, bar: dict):
        """Update session state with new bar."""
        self.today_bars.append(bar)
        self.today_high = max(self.today_high, bar["high"])
        self.today_low = min(self.today_low, bar["low"])

        # Update VWAP
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        self.vwap_cum_tp_vol += tp * bar["volume"]
        self.vwap_cum_vol += bar["volume"]
        if self.vwap_cum_vol > 0:
            self.current_vwap = self.vwap_cum_tp_vol / self.vwap_cum_vol

        # ── Track sweep state (always, even during position) ──
        # Phase 1: detect wick through PDH/PDL
        price = bar["close"]
        bar_idx = len(self.today_bars) - 1

        if self.prior_day_high > 0:
            # PDH sweep: wick above PDH, close below
            if (not self.pdh_sweep_detected and not self.pdh_sweep_fired
                    and bar["high"] > self.prior_day_high
                    and price <= self.prior_day_high):
                self.pdh_sweep_detected = True
                self.pdh_sweep_bar_idx = bar_idx
                self.pdh_sweep_extreme = bar["high"]

            # PDL sweep: wick below PDL, close above
            if (not self.pdl_sweep_detected and not self.pdl_sweep_fired
                    and bar["low"] < self.prior_day_low
                    and price >= self.prior_day_low):
                self.pdl_sweep_detected = True
                self.pdl_sweep_bar_idx = bar_idx
                self.pdl_sweep_extreme = bar["low"]

            # Phase 2: Check sweep rejection confirmation
            rejection_bars = 2  # default, overridden from strategy params later
            if self.pdh_sweep_detected:
                bars_since = bar_idx - self.pdh_sweep_bar_idx
                if bars_since > rejection_bars:
                    self.pdh_sweep_detected = False
                elif bars_since >= 1:
                    if (bar["high"] < self.pdh_sweep_extreme
                            and price < self.prior_day_high
                            and price < bar["open"]):
                        self.pdh_sweep_confirmed = True

            if self.pdl_sweep_detected:
                bars_since = bar_idx - self.pdl_sweep_bar_idx
                if bars_since > rejection_bars:
                    self.pdl_sweep_detected = False
                elif bars_since >= 1:
                    if (bar["low"] > self.pdl_sweep_extreme
                            and price > self.prior_day_low
                            and price > bar["open"]):
                        self.pdl_sweep_confirmed = True


# ═══════════════════════════════════════════════════════════════
#  EVENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════

class MarketEvent:
    """Base class for detected market events."""
    def __init__(self, name: str, direction: str, symbol: str,
                 trigger_price: float, **kwargs):
        self.name = name
        self.direction = direction  # "LONG" or "SHORT"
        self.symbol = symbol
        self.trigger_price = trigger_price
        self.features = kwargs  # event-specific features


# ═══════════════════════════════════════════════════════════════
#  THE STRATEGY
# ═══════════════════════════════════════════════════════════════

class EventDrivenStrategy(BaseStrategy):
    """
    Event-driven intraday strategy.

    On most bars: does NOTHING.
    When an event fires: contextualizes → scores → trades if high confidence.
    """

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL"],
            # Opening range
            "orb_minutes": 30,          # First 30 min = opening range
            "orb_break_pct": 0.001,     # Close must exceed range by 0.1%
            "orb_failure_bars": 3,      # Bars to confirm failure
            # Session sweep
            "sweep_rejection_bars": 2,  # Bars to confirm rejection
            "sweep_min_overshoot_atr": 0.1,  # Min wick beyond level
            # Scoring
            "score_threshold": 6,       # Out of 10 — trade only >= this
            # Trade management
            "risk_atr_mult": 1.5,       # Stop = 1.5 ATR
            "target_rr": 2.5,           # Target = 2.5× risk
            "trail_trigger_rr": 1.5,    # Trail after 1.5× risk gained
            "max_hold_bars": 20,        # Max 20 bars (5 hours on 15m)
            "entry_cutoff": time(15, 0),  # No new entries after 3 PM
            # Indicator periods (for context features)
            "ma_fast": 20,
            "ma_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "min_bars": 60,             # Warmup
        }
        merged = {**default_params, **(params or {})}
        super().__init__("event_driven", event_bus, merged)

        # Per-symbol state
        self._session: Dict[str, SessionState] = {}
        self._in_position: Dict[str, str] = {}      # symbol -> "long"/"short"
        self._entry_price: Dict[str, float] = {}
        self._stop_price: Dict[str, float] = {}
        self._target_price: Dict[str, float] = {}
        self._entry_bar_idx: Dict[str, int] = {}
        self._bar_count: Dict[str, int] = defaultdict(int)
        self._entry_event: Dict[str, str] = {}      # symbol -> event name

        # Diagnostics
        self._events_detected = defaultdict(int)
        self._events_scored = defaultdict(lambda: {"passed": 0, "failed": 0})
        self._trades_by_event = defaultdict(lambda: {"count": 0, "wins": 0})
        self._total_bars = 0
        self._total_events = 0
        self._total_trades = 0
        self._score_distribution = defaultdict(int)

    def _get_session(self, symbol: str) -> SessionState:
        if symbol not in self._session:
            self._session[symbol] = SessionState()
        return self._session[symbol]

    # ───────────────────────────────────────────────────────
    #  MAIN SIGNAL LOOP — mostly does NOTHING
    # ───────────────────────────────────────────────────────

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        if symbol not in self.params.get("symbols", []):
            return None

        bars = self._bar_history.get(symbol, [])
        if not bars:
            return None

        self._total_bars += 1
        self._bar_count[symbol] += 1
        bar = bars[-1]
        bar_time = self._get_bar_time(bar)
        bar_date = self._get_bar_date(bar)
        session = self._get_session(symbol)

        # ── New day? Reset session state ──
        # MUST happen before warmup check so PDH/PDL get set
        if bar_date and bar_date != session.current_date:
            prior_bars = [b for b in bars[:-1]
                          if self._get_bar_date(b) == session.current_date]
            session.new_day(prior_bars)
            session.current_date = bar_date

        session.update_bar(bar)

        # ── Build opening range even during warmup ──
        if bar_time and not session.orb_defined:
            orb_end = self._add_minutes(time(9, 30), self.params["orb_minutes"])
            if bar_time < orb_end:
                if session.orb_bar_count == 0:
                    session.orb_high = bar["high"]
                    session.orb_low = bar["low"]
                else:
                    session.orb_high = max(session.orb_high, bar["high"])
                    session.orb_low = min(session.orb_low, bar["low"])
                session.orb_volume += bar.get("volume", 0)
                session.orb_bar_count += 1
            elif bar_time >= orb_end:
                session.orb_defined = True

        # ── Warmup: need enough bars for indicators ──
        if len(bars) < self.params["min_bars"]:
            return None

        # ── Check exits first ──
        if symbol in self._in_position:
            return self._check_exit(symbol, bars)

        # ── No prior day data yet? Skip event detection ──
        if session.prior_day_high == 0:
            return None

        # ── Still building opening range? No events yet ──
        if not session.orb_defined:
            return None

        # ── Entry cutoff ──
        if bar_time and bar_time >= self.params["entry_cutoff"]:
            return None

        # ═══════════════════════════════════════════════════
        #  EVENT DETECTION — the heart of the system
        #  Each detector returns a MarketEvent or None.
        #  On most bars, ALL return None. That's correct.
        # ═══════════════════════════════════════════════════

        event = None

        # Priority 1: ORB Breakout
        if event is None:
            event = self._detect_orb_breakout(symbol, bar, bars, session)

        # Priority 2: ORB Failure (reverse)
        if event is None:
            event = self._detect_orb_failure(symbol, bar, bars, session)

        # Priority 3: Session High/Low Sweep + Rejection
        if event is None:
            event = self._detect_session_sweep(symbol, bar, bars, session)

        # ── No event? Do nothing. This is most bars. ──
        if event is None:
            return None

        # ═══════════════════════════════════════════════════
        #  EVENT FIRED — Now contextualize and score
        # ═══════════════════════════════════════════════════

        self._total_events += 1
        self._events_detected[event.name] += 1

        context = self._build_context(event, bar, bars, session)
        score = self._score_event(event, context)
        self._score_distribution[score] += 1

        if score < self.params["score_threshold"]:
            self._events_scored[event.name]["failed"] += 1
            return None  # Event fired but context doesn't support it

        self._events_scored[event.name]["passed"] += 1

        # ═══════════════════════════════════════════════════
        #  HIGH-CONFIDENCE EVENT — Enter the trade
        # ═══════════════════════════════════════════════════

        self._total_trades += 1
        self._trades_by_event[event.name]["count"] += 1

        # Compute ATR for stop/target
        high_arr = np.array([b["high"] for b in bars])
        low_arr = np.array([b["low"] for b in bars])
        close_arr = np.array([b["close"] for b in bars])
        atr_val = atr(high_arr, low_arr, close_arr, self.params["atr_period"])[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        price = bar["close"]
        stop_dist = self.params["risk_atr_mult"] * atr_val
        target_dist = stop_dist * self.params["target_rr"]

        if event.direction == "LONG":
            stop = price - stop_dist
            target = price + target_dist
            signal_type = SignalType.LONG
        else:
            stop = price + stop_dist
            target = price - target_dist
            signal_type = SignalType.SHORT

        # Record position state
        direction = "long" if event.direction == "LONG" else "short"
        self._in_position[symbol] = direction
        self._entry_price[symbol] = price
        self._stop_price[symbol] = stop
        self._target_price[symbol] = target
        self._entry_bar_idx[symbol] = self._bar_count[symbol]
        self._entry_event[symbol] = event.name

        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            strength=min(score / 10.0, 1.0),
            strategy_name=self.name,
            metadata={
                "entry": round(price, 2),
                "stop": round(stop, 2),
                "target": round(target, 2),
                "event": event.name,
                "direction": event.direction,
                "score": score,
                "context": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in context.items()},
            },
        )

    # ───────────────────────────────────────────────────────
    #  EVENT DETECTOR 1: Opening Range Breakout
    # ───────────────────────────────────────────────────────

    def _detect_orb_breakout(self, symbol, bar, bars, session) -> Optional[MarketEvent]:
        """
        Event: Price closes beyond the opening range after it's defined.

        Why this works:
        - The first 30 min captures overnight order flow resolution
        - A breakout means one side has won the opening battle
        - Well-documented edge: Crabel (1990s), still used by institutions

        Frequency: 0-1 per day per symbol (perfect sweet spot)
        """
        if not session.orb_defined or session.orb_breakout_fired:
            return None

        price = bar["close"]
        orb_range = session.orb_high - session.orb_low
        if orb_range <= 0:
            return None

        min_break = orb_range * self.params["orb_break_pct"]

        # LONG breakout: close above ORB high
        if price > session.orb_high + min_break:
            session.orb_breakout_fired = True
            return MarketEvent(
                name="orb_breakout",
                direction="LONG",
                symbol=symbol,
                trigger_price=price,
                orb_high=session.orb_high,
                orb_low=session.orb_low,
                orb_range=orb_range,
                orb_volume=session.orb_volume,
            )

        # SHORT breakout: close below ORB low
        if price < session.orb_low - min_break:
            session.orb_breakout_fired = True
            return MarketEvent(
                name="orb_breakout",
                direction="SHORT",
                symbol=symbol,
                trigger_price=price,
                orb_high=session.orb_high,
                orb_low=session.orb_low,
                orb_range=orb_range,
                orb_volume=session.orb_volume,
            )

        return None

    # ───────────────────────────────────────────────────────
    #  EVENT DETECTOR 2: Opening Range Failure
    # ───────────────────────────────────────────────────────

    def _detect_orb_failure(self, symbol, bar, bars, session) -> Optional[MarketEvent]:
        """
        Event: Price breaks ORB, then reverses back inside within N bars.

        Why this works:
        - Traders who entered the breakout are now TRAPPED
        - Their stop-loss exits accelerate the reversal
        - Failed breakouts are often stronger than breakouts themselves

        Frequency: ~2-3 per week per symbol
        """
        if not session.orb_defined or session.orb_failure_fired:
            return None
        if not session.orb_breakout_fired:
            return None  # Need a breakout first to have a failure

        price = bar["close"]
        orb_mid = (session.orb_high + session.orb_low) / 2

        # Get recent bars since ORB was defined
        recent = session.today_bars[-self.params["orb_failure_bars"]:]
        if len(recent) < self.params["orb_failure_bars"]:
            return None

        # Check if any recent bar broke above then current close is inside
        broke_high = any(b["high"] > session.orb_high for b in recent[:-1])
        broke_low = any(b["low"] < session.orb_low for b in recent[:-1])

        # Failed upside breakout: broke above but now closing below ORB mid
        if broke_high and price < orb_mid:
            session.orb_failure_fired = True
            return MarketEvent(
                name="orb_failure",
                direction="SHORT",  # Fade the failed breakout
                symbol=symbol,
                trigger_price=price,
                orb_high=session.orb_high,
                orb_low=session.orb_low,
                orb_range=session.orb_high - session.orb_low,
                failed_direction="long",
            )

        # Failed downside breakout: broke below but now closing above ORB mid
        if broke_low and price > orb_mid:
            session.orb_failure_fired = True
            return MarketEvent(
                name="orb_failure",
                direction="LONG",  # Fade the failed breakout
                symbol=symbol,
                trigger_price=price,
                orb_high=session.orb_high,
                orb_low=session.orb_low,
                orb_range=session.orb_high - session.orb_low,
                failed_direction="short",
            )

        return None

    # ───────────────────────────────────────────────────────
    #  EVENT DETECTOR 3: Session Level Sweep + Rejection
    # ───────────────────────────────────────────────────────

    def _detect_session_sweep(self, symbol, bar, bars, session) -> Optional[MarketEvent]:
        """
        Event: Price sweeps prior day high/low (takes liquidity) then rejects.

        Why this works:
        - Stops cluster above PDH and below PDL
        - Institutions push price to trigger those stops (liquidity grab)
        - Once stops are hit, institutions fill large orders and price reverses
        - This is directly from market microstructure theory

        Sweep tracking happens in SessionState.update_bar() so it works
        even when we're in a position from another trade.

        Frequency: 2-5 per week per symbol (perfect sweet spot)
        """
        price = bar["close"]

        # PDH sweep confirmed → SHORT
        if session.pdh_sweep_confirmed and not session.pdh_sweep_fired:
            session.pdh_sweep_confirmed = False
            session.pdh_sweep_fired = True
            session.pdh_sweep_detected = False
            overshoot = session.pdh_sweep_extreme - session.prior_day_high
            return MarketEvent(
                name="session_sweep",
                direction="SHORT",
                symbol=symbol,
                trigger_price=price,
                level_swept=session.prior_day_high,
                sweep_extreme=session.pdh_sweep_extreme,
                overshoot=overshoot,
                sweep_type="pdh",
            )

        # PDL sweep confirmed → LONG
        if session.pdl_sweep_confirmed and not session.pdl_sweep_fired:
            session.pdl_sweep_confirmed = False
            session.pdl_sweep_fired = True
            session.pdl_sweep_detected = False
            overshoot = session.prior_day_low - session.pdl_sweep_extreme
            return MarketEvent(
                name="session_sweep",
                direction="LONG",
                symbol=symbol,
                trigger_price=price,
                level_swept=session.prior_day_low,
                sweep_extreme=session.pdl_sweep_extreme,
                overshoot=overshoot,
                sweep_type="pdl",
            )

        return None

    # ───────────────────────────────────────────────────────
    #  CONTEXT BUILDER — repurposes existing indicators
    # ───────────────────────────────────────────────────────

    def _build_context(self, event: MarketEvent, bar: dict,
                       bars: list, session: SessionState) -> Dict[str, Any]:
        """
        Gather contextualizing features around the event.
        These are what your friend calls "better interpretation."
        Our existing indicators become features, not signals.
        """
        close_arr = np.array([b["close"] for b in bars])
        high_arr = np.array([b["high"] for b in bars])
        low_arr = np.array([b["low"] for b in bars])
        vol_arr = np.array([b.get("volume", 0) for b in bars], dtype=float)

        price = bar["close"]
        atr_val = atr(high_arr, low_arr, close_arr, self.params["atr_period"])[-1]
        rsi_val = rsi(close_arr, self.params["rsi_period"])[-1]
        rvol = relative_volume(vol_arr, 20)[-1]
        ma_f = ema(close_arr, self.params["ma_fast"])[-1]
        ma_s = ema(close_arr, self.params["ma_slow"])[-1]

        is_long = event.direction == "LONG"
        context = {}

        # 1. MA alignment — trend agrees with event direction?
        if not np.isnan(ma_f) and not np.isnan(ma_s):
            ma_bull = ma_f > ma_s
            context["ma_aligned"] = (is_long and ma_bull) or (not is_long and not ma_bull)
        else:
            context["ma_aligned"] = False

        # 2. RSI — not overextended in event direction?
        if not np.isnan(rsi_val):
            context["rsi"] = rsi_val
            context["rsi_ok"] = (
                (is_long and 30 < rsi_val < 70) or
                (not is_long and 30 < rsi_val < 70)
            )
        else:
            context["rsi"] = 50.0
            context["rsi_ok"] = True

        # 3. VWAP position — institutional flow agrees?
        context["above_vwap"] = price > session.current_vwap
        context["vwap_aligned"] = (
            (is_long and price > session.current_vwap) or
            (not is_long and price < session.current_vwap)
        )

        # 4. Volume confirmation — above average participation?
        context["rvol"] = rvol if not np.isnan(rvol) else 1.0
        context["volume_strong"] = context["rvol"] > 1.2

        # 5. ATR context — enough movement potential?
        context["atr"] = atr_val if not np.isnan(atr_val) else 0.0

        # 6. Time of day — morning is higher conviction
        bar_time = self._get_bar_time(bar)
        if bar_time:
            context["is_morning"] = bar_time < time(11, 30)
            context["is_power_hour"] = bar_time >= time(14, 0)
            context["is_lunch"] = time(12, 0) <= bar_time < time(13, 0)
        else:
            context["is_morning"] = False
            context["is_power_hour"] = False
            context["is_lunch"] = False

        # 7. Gap from prior close — gap aligns with direction?
        if session.prior_day_close > 0 and len(session.today_bars) > 0:
            first_open = session.today_bars[0]["open"]
            gap_pct = (first_open - session.prior_day_close) / session.prior_day_close
            context["gap_pct"] = gap_pct
            context["gap_aligned"] = (
                (is_long and gap_pct > 0.001) or
                (not is_long and gap_pct < -0.001)
            )
        else:
            context["gap_pct"] = 0.0
            context["gap_aligned"] = False

        # 8. Prior day close position — closed strong or weak?
        if session.prior_day_range > 0:
            pdc_position = ((session.prior_day_close - session.prior_day_low)
                            / session.prior_day_range)
            context["prior_close_strong"] = (
                (is_long and pdc_position > 0.6) or
                (not is_long and pdc_position < 0.4)
            )
        else:
            context["prior_close_strong"] = False

        # 9. ORB width relative to average (for ORB events)
        if session.orb_history:
            avg_orb = np.mean(session.orb_history)
            orb_range = event.features.get("orb_range", 0)
            if avg_orb > 0 and orb_range > 0:
                context["orb_width_ratio"] = orb_range / avg_orb
                context["orb_narrow"] = context["orb_width_ratio"] < 0.8
            else:
                context["orb_width_ratio"] = 1.0
                context["orb_narrow"] = False
        else:
            context["orb_width_ratio"] = 1.0
            context["orb_narrow"] = False

        # 10. Sweep overshoot relative to ATR (for sweep events)
        overshoot = event.features.get("overshoot", 0)
        if context["atr"] > 0 and overshoot > 0:
            context["overshoot_atr_ratio"] = overshoot / context["atr"]
            context["overshoot_meaningful"] = 0.15 < context["overshoot_atr_ratio"] < 1.5
        else:
            context["overshoot_atr_ratio"] = 0.0
            context["overshoot_meaningful"] = False

        return context

    # ───────────────────────────────────────────────────────
    #  SCORER — rule-based, debuggable, tunable
    # ───────────────────────────────────────────────────────

    def _score_event(self, event: MarketEvent, context: Dict) -> int:
        """
        Score 0-10 based on contextual features.
        Each feature adds 0-2 points.

        Why rule-based first (not ML):
        - You can see WHY a trade was taken
        - You can debug failures one rule at a time
        - ML needs hundreds of samples; we'll have 30-50
        - Move to ML later once you have labeled data
        """
        score = 0

        # ── Universal context (all events) ──

        # MA alignment: trend agrees with direction (+1)
        if context.get("ma_aligned"):
            score += 1

        # RSI not overextended (+1)
        if context.get("rsi_ok"):
            score += 1

        # VWAP position confirms direction (+2 — this is the #1 institutional filter)
        if context.get("vwap_aligned"):
            score += 2

        # Volume above average (+1)
        if context.get("volume_strong"):
            score += 1

        # Time of day: morning or power hour (+1), lunch penalty (-1)
        if context.get("is_morning") or context.get("is_power_hour"):
            score += 1
        if context.get("is_lunch"):
            score -= 1

        # Gap aligns with direction (+1)
        if context.get("gap_aligned"):
            score += 1

        # Prior day close supports direction (+1)
        if context.get("prior_close_strong"):
            score += 1

        # ── Event-specific scoring ──

        if event.name == "orb_breakout":
            # Narrow ORB = compressed energy (+1)
            if context.get("orb_narrow"):
                score += 1

        elif event.name == "orb_failure":
            # Failed breakouts are inherently high-probability
            # Give them a base bonus (+1)
            score += 1

        elif event.name == "session_sweep":
            # Clean overshoot size (+1)
            if context.get("overshoot_meaningful"):
                score += 1

        return max(0, score)  # Floor at 0

    # ───────────────────────────────────────────────────────
    #  EXIT MANAGEMENT
    # ───────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, bars: list) -> Optional[SignalEvent]:
        """Manage open position: stop, target, trailing, timeout."""
        direction = self._in_position[symbol]
        price = bars[-1]["close"]
        entry = self._entry_price[symbol]
        stop = self._stop_price[symbol]
        target = self._target_price[symbol]
        bars_held = self._bar_count[symbol] - self._entry_bar_idx[symbol]

        # Compute current ATR for trailing
        high_arr = np.array([b["high"] for b in bars])
        low_arr = np.array([b["low"] for b in bars])
        close_arr = np.array([b["close"] for b in bars])
        atr_val = atr(high_arr, low_arr, close_arr, self.params["atr_period"])[-1]
        if np.isnan(atr_val):
            atr_val = abs(entry - stop) / self.params["risk_atr_mult"]

        should_exit = False
        exit_reason = ""

        # Trailing stop: after 1.5R profit, move stop to breakeven + 0.2 ATR
        risk = abs(entry - stop)
        trail_trigger = risk * self.params["trail_trigger_rr"]

        if direction == "long":
            current_profit = price - entry
            if current_profit >= trail_trigger:
                new_stop = max(stop, entry + atr_val * 0.2)
                if new_stop > stop:
                    self._stop_price[symbol] = new_stop
                    stop = new_stop

            if price <= stop:
                should_exit, exit_reason = True, "stop"
            elif price >= target:
                should_exit, exit_reason = True, "target"

        elif direction == "short":
            current_profit = entry - price
            if current_profit >= trail_trigger:
                new_stop = min(stop, entry - atr_val * 0.2)
                if new_stop < stop:
                    self._stop_price[symbol] = new_stop
                    stop = new_stop

            if price >= stop:
                should_exit, exit_reason = True, "stop"
            elif price <= target:
                should_exit, exit_reason = True, "target"

        # Timeout
        if bars_held >= self.params["max_hold_bars"]:
            should_exit, exit_reason = True, "timeout"

        if should_exit:
            pnl_pct = ((price - entry) / entry * 100) if direction == "long" else ((entry - price) / entry * 100)
            is_win = pnl_pct > 0
            event_name = self._entry_event.get(symbol, "unknown")
            if is_win:
                self._trades_by_event[event_name]["wins"] += 1

            exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT

            # Clean up
            del self._in_position[symbol]
            self._entry_price.pop(symbol, None)
            self._stop_price.pop(symbol, None)
            self._target_price.pop(symbol, None)
            self._entry_bar_idx.pop(symbol, None)
            self._entry_event.pop(symbol, None)

            return SignalEvent(
                symbol=symbol,
                signal_type=exit_type,
                strength=0.0,
                strategy_name=self.name,
                metadata={"exit_reason": exit_reason, "exit_price": round(price, 2),
                          "entry_price": round(entry, 2)},
            )

        return None

    # ───────────────────────────────────────────────────────
    #  UTILITIES
    # ───────────────────────────────────────────────────────

    def _get_bar_time(self, bar: dict) -> Optional[time]:
        ts = bar.get("timestamp")
        if ts is not None and hasattr(ts, 'time'):
            return ts.time()
        return None

    def _get_bar_date(self, bar: dict) -> Optional[date]:
        ts = bar.get("timestamp")
        if ts is not None and hasattr(ts, 'date'):
            return ts.date() if callable(ts.date) else ts.date
        return None

    def _add_minutes(self, t: time, minutes: int) -> time:
        total = t.hour * 60 + t.minute + minutes
        return time(total // 60, total % 60)

    # ───────────────────────────────────────────────────────
    #  DIAGNOSTICS
    # ───────────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict:
        return {
            "total_bars": self._total_bars,
            "total_events": self._total_events,
            "total_trades": self._total_trades,
            "events_detected": dict(self._events_detected),
            "events_scored": {k: dict(v) for k, v in self._events_scored.items()},
            "trades_by_event": {k: dict(v) for k, v in self._trades_by_event.items()},
            "score_distribution": dict(self._score_distribution),
        }

    def print_funnel(self):
        print(f"\n  EVENT-DRIVEN STRATEGY")
        print(f"  {'═'*55}")
        print(f"  Architecture: Event → Context → Score → Trade")
        print(f"  {'─'*55}")
        print(f"  Total bars scanned:       {self._total_bars:>8,}")
        print(f"  Events detected:          {self._total_events:>8,}  "
              f"({self._total_events / max(self._total_bars, 1) * 100:.1f}% of bars)")
        print(f"  Events traded:            {self._total_trades:>8,}  "
              f"(score >= {self.params['score_threshold']})")
        no_event_bars = self._total_bars - self._total_events
        print(f"  Bars with NO action:      {no_event_bars:>8,}  "
              f"({no_event_bars / max(self._total_bars, 1) * 100:.1f}%)")

        print(f"\n  Events by Type:")
        for ename, count in sorted(self._events_detected.items()):
            scored = self._events_scored.get(ename, {"passed": 0, "failed": 0})
            traded = self._trades_by_event.get(ename, {"count": 0, "wins": 0})
            win_rate = (traded["wins"] / traded["count"] * 100
                        if traded["count"] > 0 else 0)
            print(f"    {ename:>20}: {count:>4} detected │ "
                  f"{scored['passed']:>3} traded │ "
                  f"{traded['wins']}/{traded['count']} won ({win_rate:.0f}%)")

        print(f"\n  Score Distribution:")
        for score in sorted(self._score_distribution.keys()):
            count = self._score_distribution[score]
            threshold_marker = " ← threshold" if score == self.params["score_threshold"] else ""
            bar = "█" * count
            traded = "✓" if score >= self.params["score_threshold"] else "✗"
            print(f"    Score {score:>2}: {count:>4} events {traded} {bar}{threshold_marker}")

        print(f"  {'═'*55}")
