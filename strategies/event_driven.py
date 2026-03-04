"""Event-Driven Intraday Strategy v10 — Research-Validated.

Three strategies validated through independent Phase 0→4 research pipeline
(Jan 2024 – Dec 2025, 5 large-cap equities at 15-min bars):

  SWEEP_HQ          NVDA PDL liquidity sweep + gap-up day + prior-up
                    OOS Sharpe +0.933, 4/4 WF pass, n≈50 OOS trades/yr
                    Filters: NVDA + prior_up + gap_pos + not_friday + avoid_13_14h
                    Entry : sweep-bar close (M1) | Stop: sweep wick
                    Exit  : Fixed +1R target

  ORBFAIL_REGIME    Failed SHORT ORB breakout on gap-up days, above VWAP
                    OOS Sharpe +0.704, 4/4 WF pass, n≈38 OOS trades/yr
                    Filters: not_goog + gap_up(>+0.2%) + fail_above_vwap
                    Entry : failure-bar close (M1) | Stop: min-low of BO move
                    Exit  : 50%@1R → activate ATR trail on remainder; EOD fallback

  SWEEP_PRIMARY     All-symbol PDL sweep, midweek, near intraday extreme
                    OOS Sharpe ~+0.29, 2/4 WF pass — MARGINAL (half position)
                    Filters: LONG + prior_up + midweek(Tue/Wed/Thu) + near_extreme + RVOL≥1.0x
                    Entry : confirmation-bar close (M2) | Stop: sweep wick
                    Exit  : EOD (hold all session)

Vol-compression event: DROPPED — Phase 0 showed no edge (-0.007R, WR=18%).
ORB-breakout momentum: not implemented — Phase 4 WF marginal, superseded by
  ORBFAIL_REGIME which uses the same events more profitably via fade.
"""

from collections import defaultdict, deque
from datetime import time, date
from typing import Dict, List, Optional

import numpy as np

from core.events import EventBus, EventType, MarketDataEvent, SignalEvent, SignalType
from strategies import BaseStrategy
from features import atr as _atr_arr

# Must match SimulatedExecution.slippage_pct so targets are exactly 1R from fill.
# Entry fills at close × (1 + ENTRY_SLIPPAGE) for longs,
#               close × (1 - ENTRY_SLIPPAGE) for shorts.
ENTRY_SLIPPAGE = 0.001


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE — per-symbol, per-day intraday tracking
# ═══════════════════════════════════════════════════════════════

class SessionState:
    """Tracks all intraday levels and event flags needed by the three strategies."""

    def __init__(self):
        self.current_date: Optional[date] = None

        # ── Prior day levels ──
        self.prior_day_high: float = 0.0
        self.prior_day_low: float = 0.0
        self.prior_day_open: float = 0.0    # first bar open of prior day
        self.prior_day_close: float = 0.0   # last bar close of prior day
        self.prior_day_up: bool = False      # prior-day close >= open

        # ── Today's session ──
        self.today_bars: List[dict] = []
        self.today_open: float = 0.0
        self.today_high: float = 0.0
        self.today_low: float = float("inf")

        # ── Gap ──
        self.gap_pct: float = 0.0
        self.gap_pos: bool = False   # today's open > prior close by > +0.2%

        # ── Opening range (ORB = first 2 bars, first 30 min at 15-min resolution) ──
        self.orb_high: float = 0.0
        self.orb_low: float = 0.0
        self.orb_defined: bool = False
        self.orb_bar_count: int = 0

        # ── ORB breakout tracking (used by ORBFAIL_REGIME) ──
        # Only SHORT_BO (close below ORB_LOW) → potential LONG fade is tracked.
        self.orb_bo_detected: bool = False
        self.orb_bo_direction: str = ""     # "SHORT_BO" only
        self.orb_bo_extreme: float = 0.0    # running min-low from BO+1 bar onwards
        self.orb_bo_bar_idx: int = 0        # bar index of BO detection (0-indexed)
        self.orb_fail_fired: bool = False   # one failure signal per day

        # ── Running VWAP (reset each session) ──
        self.vwap_num: float = 0.0
        self.vwap_den: float = 0.0
        self.current_vwap: float = 0.0

        # ── PDL sweep tracking ──
        self.pdl_sweep_fired: bool = False   # any PDL sweep signal fired today
        # M2 confirmation state (SWEEP_PRIMARY)
        self.pdl_sweep_pending: bool = False
        self.pdl_sweep_bar_idx: int = 0      # bar index when sweep detected
        self.pdl_sweep_extreme: float = 0.0  # sweep wick (absolute low at detection)
        self.pdl_near_extreme: bool = False  # near_extreme flag saved at detection time

    def new_day(self, prior_bars: List[dict]) -> None:
        """Reset for a new trading day, seeding prior-day levels from yesterday's bars."""
        if prior_bars:
            self.prior_day_high = max(b["high"] for b in prior_bars)
            self.prior_day_low = min(b["low"] for b in prior_bars)
            self.prior_day_open = prior_bars[0]["open"]
            self.prior_day_close = prior_bars[-1]["close"]
            self.prior_day_up = self.prior_day_close >= self.prior_day_open

        self.today_bars = []
        self.today_open = 0.0
        self.today_high = 0.0
        self.today_low = float("inf")
        self.gap_pct = 0.0
        self.gap_pos = False
        self.orb_high = 0.0
        self.orb_low = 0.0
        self.orb_defined = False
        self.orb_bar_count = 0
        self.orb_bo_detected = False
        self.orb_bo_direction = ""
        self.orb_bo_extreme = 0.0
        self.orb_bo_bar_idx = 0
        self.orb_fail_fired = False
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.current_vwap = 0.0
        self.pdl_sweep_fired = False
        self.pdl_sweep_pending = False
        self.pdl_sweep_bar_idx = 0
        self.pdl_sweep_extreme = 0.0
        self.pdl_near_extreme = False

    def update_bar(self, bar: dict) -> None:
        """Update session state with the incoming bar (call before signal logic)."""
        self.today_bars.append(bar)
        self.today_high = max(self.today_high, bar["high"])
        self.today_low = min(self.today_low, bar["low"])

        # Gap is computed once on the first bar of the day
        if len(self.today_bars) == 1:
            self.today_open = bar["open"]
            if self.prior_day_close > 0:
                self.gap_pct = (bar["open"] - self.prior_day_close) / self.prior_day_close
                self.gap_pos = self.gap_pct > 0.002   # > +0.2% (threshold applied below)

        # Running VWAP (session-cumulative, no look-ahead)
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        vol = max(bar.get("volume", 0.0), 0.0)
        if vol > 0:
            self.vwap_num += tp * vol
            self.vwap_den += vol
            self.current_vwap = self.vwap_num / self.vwap_den


# ═══════════════════════════════════════════════════════════════
#  STRATEGY
# ═══════════════════════════════════════════════════════════════

class EventDrivenStrategy(BaseStrategy):
    """Event-driven intraday strategy v10 — three research-validated setups.

    Architecture: Data → Session Update → Exit Check → Entry Detection
    No scoring engine; each strategy uses hard binary filters validated
    through the Phase 0→4 walk-forward pipeline.
    """

    name = "event_driven"

    # ── ORB failure parameters (from Phase 0 research) ──
    _ORB_BO_MIN_BAR = 2    # earliest valid BO: bar index 2 (10:00 bar), 0-indexed
    _ORB_BO_MAX_BAR = 7    # latest valid BO: bar index 7 (11:15 bar)
    _ORB_FAIL_WINDOW = 6   # failure must occur within 6 bars of BO

    # ── Sweep parameters ──
    _SWEEP_CONF_WINDOW = 3  # bars available for M2 confirmation

    # ── Time constants ──
    _ORB_END_TIME = time(10, 0)     # ORB defined when bar_time >= 10:00
    _ENTRY_CUTOFF = time(15, 15)    # no new entries after this
    _EOD_TIME = time(15, 30)        # EOD exit trigger (SWEEP_PRIMARY + ORBFAIL fallback)

    def __init__(self, event_bus: EventBus, params: dict = None):
        default_params = {
            "symbols": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOG"],
            "min_bars": 30,
            "atr_period": 14,
            # Tunable filter thresholds — override in config or precompute_grid.py
            "gap_threshold": 0.002,       # min abs gap for gap_pos flag (default 0.2%)
            "rvol_threshold_sweep": 1.0,  # min RVOL for SWEEP_PRIMARY (0.0 = off)
            # Strategy toggles — set False to disable a sub-strategy entirely
            "enable_sweep_hq": True,
            "enable_orbfail": True,
            "enable_sweep_primary": True,
        }
        merged = {**default_params, **(params or {})}
        super().__init__("event_driven", event_bus, merged)

        # Session state (one per symbol)
        self._sessions: Dict[str, SessionState] = {}
        self._bar_count: Dict[str, int] = defaultdict(int)

        # ── Open position state (one trade per symbol at a time) ──
        self._in_position: Dict[str, str] = {}      # symbol → strategy name
        self._pos_direction: Dict[str, str] = {}    # symbol → "long" | "short"
        self._pos_entry: Dict[str, float] = {}
        self._pos_stop: Dict[str, float] = {}
        self._pos_risk: Dict[str, float] = {}       # initial risk distance
        self._pos_target: Dict[str, float] = {}     # SWEEP_HQ fixed 1R target
        self._pos_partial: Dict[str, bool] = {}     # ORBFAIL: has 1R been reached?
        self._pos_best: Dict[str, float] = {}       # ORBFAIL: high-water for ATR trail

        # ── RVOL tracking (Zarattini 2024): first-bar relative volume filter ──
        # Stores up to 15 days of first-bar volumes for the rolling 14-day average.
        self._first_bar_vol_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=15))
        self._daily_rvol: Dict[str, float] = {}

        # ── Diagnostics ──
        self._total_trades = 0
        self._trades_by_strategy: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "wins": 0})
        self._exit_reasons: Dict[str, int] = defaultdict(int)

    # ───────────────────────────────────────────────────────────
    #  SESSION ACCESSOR
    # ───────────────────────────────────────────────────────────

    def _get_session(self, symbol: str) -> SessionState:
        if symbol not in self._sessions:
            self._sessions[symbol] = SessionState()
        return self._sessions[symbol]

    # ───────────────────────────────────────────────────────────
    #  MAIN BAR LOOP
    # ───────────────────────────────────────────────────────────

    def calculate_signal(self, symbol: str) -> Optional[SignalEvent]:
        if symbol not in self.params.get("symbols", []):
            return None

        bars = self._bar_history.get(symbol, [])
        if not bars or len(bars) < self.params["min_bars"]:
            return None

        self._bar_count[symbol] += 1
        bar = bars[-1]
        bar_time = self._get_bar_time(bar)
        bar_date = self._get_bar_date(bar)
        session = self._get_session(symbol)

        # ── New day: reset session, seed prior-day levels ──
        if bar_date and bar_date != session.current_date:
            prior_bars = [b for b in bars[:-1]
                          if self._get_bar_date(b) == session.current_date]
            session.new_day(prior_bars)
            session.current_date = bar_date

            # ── RVOL: compute today's relative volume from prior history ──
            # Use history of prior days only (no look-ahead) — matches rvol_phase0.py.
            vol_history = list(self._first_bar_vol_history[symbol])
            first_vol = bar.get("volume", 0.0)
            if len(vol_history) >= 5:
                avg_vol = float(np.mean(vol_history[-14:]))
                self._daily_rvol[symbol] = (first_vol / avg_vol) if avg_vol > 0 else 1.0
            else:
                self._daily_rvol[symbol] = 1.0  # insufficient history → no filter
            # Append current first-bar volume for future days' RVOL computation.
            self._first_bar_vol_history[symbol].append(first_vol)

        session.update_bar(bar)

        # ── Apply configurable gap threshold (overrides SessionState default 0.2%) ──
        gap_thr = self.params.get("gap_threshold", 0.002)
        if len(session.today_bars) == 1 and session.prior_day_close > 0:
            session.gap_pos = session.gap_pct > gap_thr

        # ── Build ORB (first 2 bars = 9:30 and 9:45) ──
        if not session.orb_defined and bar_time is not None:
            if bar_time < self._ORB_END_TIME:
                if session.orb_bar_count == 0:
                    session.orb_high = bar["high"]
                    session.orb_low = bar["low"]
                else:
                    session.orb_high = max(session.orb_high, bar["high"])
                    session.orb_low = min(session.orb_low, bar["low"])
                session.orb_bar_count += 1
            else:
                session.orb_defined = True

        # ── Exit check: always first, before any entry logic ──
        if symbol in self._in_position:
            return self._check_exit(symbol, bar, bars, session, bar_time)

        # ── Entry guards ──
        if session.prior_day_high == 0:
            return None
        if bar_time is not None and bar_time >= self._ENTRY_CUTOFF:
            return None

        # ── Update ORB breakout tracking (runs every bar after ORB defined) ──
        if session.orb_defined and not session.orb_fail_fired:
            self._update_orb_bo_tracking(bar, session)

        # ── Strategy 1: SWEEP_HQ (NVDA only) ──
        if (self.params.get("enable_sweep_hq", True)
                and symbol == "NVDA" and bar_time is not None):
            sig = self._check_sweep_hq(symbol, bar, session, bar_time, bar_date)
            if sig:
                return sig

        # ── Strategy 2: ORBFAIL_REGIME (all except GOOG) ──
        if (self.params.get("enable_orbfail", True)
                and symbol != "GOOG" and session.orb_defined):
            sig = self._check_orbfail_regime(symbol, bar, bars, session)
            if sig:
                return sig

        # ── Strategy 3: SWEEP_PRIMARY (all symbols, LONG only) ──
        if self.params.get("enable_sweep_primary", True):
            sig = self._check_sweep_primary(symbol, bar, session, bar_date)
            if sig:
                return sig

        return None

    # ───────────────────────────────────────────────────────────
    #  ORB BREAKOUT TRACKING (for ORBFAIL_REGIME)
    # ───────────────────────────────────────────────────────────

    def _update_orb_bo_tracking(self, bar: dict, session: SessionState) -> None:
        """Detect and track SHORT_BO (close below ORB low) in the valid window.

        The running extreme (min low) is accumulated from the bar AFTER detection
        through the failure bar, matching the research's stop_fn calculation.
        """
        bar_idx = len(session.today_bars) - 1

        if not session.orb_bo_detected:
            # Only detect within the validated window (bar indices 2-7)
            if self._ORB_BO_MIN_BAR <= bar_idx <= self._ORB_BO_MAX_BAR:
                if bar["close"] < session.orb_low:
                    session.orb_bo_detected = True
                    session.orb_bo_direction = "SHORT_BO"
                    session.orb_bo_extreme = float("inf")  # updated from next bar
                    session.orb_bo_bar_idx = bar_idx
        else:
            # Accumulate running min-low from bars AFTER the BO bar
            if bar_idx > session.orb_bo_bar_idx:
                session.orb_bo_extreme = min(session.orb_bo_extreme, bar["low"])

    # ───────────────────────────────────────────────────────────
    #  STRATEGY 1: SWEEP_HQ
    # ───────────────────────────────────────────────────────────

    def _check_sweep_hq(self, symbol: str, bar: dict, session: SessionState,
                         bar_time: time, bar_date: Optional[date]) -> Optional[SignalEvent]:
        """SWEEP_HQ — NVDA PDL sweep, M1 entry, Fixed +1R target.

        Research result: 4/4 WF pass, OOS Sharpe +0.933, CI [+0.437R, +0.867R]
        Entry on the sweep bar itself (M1 = sweep-bar close, no waiting for confirmation).
        """
        if session.pdl_sweep_fired or session.prior_day_low == 0:
            return None

        # Validated time/day filters (from Phase 3 regime scan)
        if bar_time.hour in (13, 14):
            return None
        if bar_date is not None and bar_date.weekday() == 4:   # Friday
            return None

        # Validated day-regime filters
        if not session.prior_day_up:
            return None
        if not session.gap_pos:
            return None

        price = bar["close"]

        # PDL sweep: bar wicks below PDL AND closes back inside (>= PDL)
        if bar["low"] < session.prior_day_low and price >= session.prior_day_low:
            stop = bar["low"]               # sweep wick extreme
            # Use expected fill (incl. slippage) so target is exactly 1R from fill,
            # not from the close. Without this the effective R:R is ~0.5-0.7:1.
            fill = price * (1 + ENTRY_SLIPPAGE)
            risk = abs(fill - stop)
            if risk <= 0:
                return None
            target = fill + risk            # Fixed +1R from fill

            session.pdl_sweep_fired = True
            self._enter_position(symbol, "sweep_hq", "long", fill, stop, risk,
                                  target=target)
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.LONG,
                strength=1.0, strategy_name=self.name,
                metadata={
                    "strategy": "sweep_hq",
                    "entry": round(fill, 2),
                    "stop": round(stop, 2),
                    "target": round(target, 2),
                    "risk": round(risk, 2),
                    "position_scale": 1.0,
                },
            )
        return None

    # ───────────────────────────────────────────────────────────
    #  STRATEGY 2: ORBFAIL_REGIME
    # ───────────────────────────────────────────────────────────

    def _check_orbfail_regime(self, symbol: str, bar: dict, bars: list,
                               session: SessionState) -> Optional[SignalEvent]:
        """ORBFAIL_REGIME — failed SHORT ORB on gap-up day above VWAP, M1 entry.

        Research result: 4/4 WF pass, OOS Sharpe +0.704, CI [+0.224R, +0.504R]
        Fades a SHORT ORB breakout that fails (bullish reversal bar above ORB mid)
        when gap_up and price still above VWAP → trapped shorts fuel the move.
        """
        if session.orb_fail_fired:
            return None
        if not session.orb_bo_detected or session.orb_bo_direction != "SHORT_BO":
            return None

        # Check failure window (1 to FAIL_WINDOW bars after BO detection)
        bar_idx = len(session.today_bars) - 1
        bars_since_bo = bar_idx - session.orb_bo_bar_idx
        if bars_since_bo < 1 or bars_since_bo > self._ORB_FAIL_WINDOW:
            return None

        # Day-regime filter: gap-up day only
        if not session.gap_pos:
            return None

        price = bar["close"]
        orb_mid = (session.orb_high + session.orb_low) / 2.0

        # Failure conditions: bullish reversal bar (close > open) above ORB midpoint
        if not (price > bar["open"] and price > orb_mid):
            return None

        # fail_above_vwap: price still above VWAP at failure bar
        # (elevated price despite BO failure → strong reversal signal)
        if session.current_vwap > 0 and price <= session.current_vwap:
            return None

        # Stop = min low of [BO+1 bar .. failure bar] (running extreme accumulated)
        stop = session.orb_bo_extreme
        if stop == float("inf"):
            stop = bar["low"]   # edge case: no bars between BO and failure
        # Use expected fill so 1R target check in exit is relative to actual fill.
        fill = price * (1 + ENTRY_SLIPPAGE)
        risk = abs(fill - stop)
        if risk <= 0:
            return None

        session.orb_fail_fired = True
        self._enter_position(symbol, "orbfail_regime", "long", fill, stop, risk,
                              best_price=fill)
        return SignalEvent(
            symbol=symbol, signal_type=SignalType.LONG,
            strength=1.0, strategy_name=self.name,
            metadata={
                "strategy": "orbfail_regime",
                "entry": round(fill, 2),
                "stop": round(stop, 2),
                "risk": round(risk, 2),
                "orb_mid": round(orb_mid, 2),
                "vwap": round(session.current_vwap, 2),
                "position_scale": 1.0,
            },
        )

    # ───────────────────────────────────────────────────────────
    #  STRATEGY 3: SWEEP_PRIMARY
    # ───────────────────────────────────────────────────────────

    def _check_sweep_primary(self, symbol: str, bar: dict, session: SessionState,
                              bar_date: Optional[date]) -> Optional[SignalEvent]:
        """SWEEP_PRIMARY — all-symbol PDL sweep, M2 confirmation entry.

        Research result: 2/4 WF pass (MARGINAL) — half position size.
        Waits for a bullish confirmation bar (close > open) within 3 bars
        of the PDL sweep. All regime filters applied at confirmation time.
        """
        if session.prior_day_low == 0:
            return None

        bar_idx = len(session.today_bars) - 1
        price = bar["close"]

        # Phase 1: detect PDL sweep bar (set pending flag, do NOT fire yet)
        if not session.pdl_sweep_pending and not session.pdl_sweep_fired:
            if bar["low"] < session.prior_day_low and price >= session.prior_day_low:
                # RVOL gate: above-average opening volume required
                rvol_thr = self.params.get("rvol_threshold_sweep", 1.0)
                if rvol_thr > 0 and self._daily_rvol.get(symbol, 1.0) < rvol_thr:
                    return None
                session.pdl_sweep_pending = True
                session.pdl_sweep_bar_idx = bar_idx
                session.pdl_sweep_extreme = bar["low"]
                # Compute near_extreme at detection time (uses today_bars up to this bar)
                session.pdl_near_extreme = self._is_near_intraday_extreme(
                    session, bar["low"])
                return None   # Wait for confirmation bar

        # Phase 2: check for M2 confirmation
        if not session.pdl_sweep_pending:
            return None

        bars_since = bar_idx - session.pdl_sweep_bar_idx

        # Expire if no confirmation within window
        if bars_since > self._SWEEP_CONF_WINDOW:
            session.pdl_sweep_pending = False
            return None

        if bars_since < 1:
            return None   # Same bar as sweep detection — skip

        # Apply validated regime filters at confirmation time
        if not session.prior_day_up:
            session.pdl_sweep_pending = False
            return None

        if bar_date is not None and bar_date.weekday() not in (1, 2, 3):  # not Tue/Wed/Thu
            session.pdl_sweep_pending = False
            return None

        if not session.pdl_near_extreme:
            session.pdl_sweep_pending = False
            return None

        # Confirmation: bullish bar (close > open)
        if price > bar["open"]:
            session.pdl_sweep_pending = False
            session.pdl_sweep_fired = True
            stop = session.pdl_sweep_extreme
            fill = price * (1 + ENTRY_SLIPPAGE)
            risk = abs(fill - stop)
            if risk <= 0:
                return None

            self._enter_position(symbol, "sweep_primary", "long", fill, stop, risk)
            return SignalEvent(
                symbol=symbol, signal_type=SignalType.LONG,
                strength=0.5,   # Half position — 2/4 WF pass (marginal)
                strategy_name=self.name,
                metadata={
                    "strategy": "sweep_primary",
                    "entry": round(fill, 2),
                    "stop": round(stop, 2),
                    "risk": round(risk, 2),
                    "exit": "eod",
                    "position_scale": 0.5,
                },
            )

        return None

    # ───────────────────────────────────────────────────────────
    #  POSITION MANAGEMENT
    # ───────────────────────────────────────────────────────────

    def _enter_position(self, symbol: str, strategy: str, direction: str,
                         entry: float, stop: float, risk: float,
                         target: Optional[float] = None,
                         best_price: Optional[float] = None) -> None:
        self._in_position[symbol] = strategy
        self._pos_direction[symbol] = direction
        self._pos_entry[symbol] = entry
        self._pos_stop[symbol] = stop
        self._pos_risk[symbol] = risk
        self._pos_partial[symbol] = False
        self._pos_best[symbol] = best_price if best_price is not None else entry

        if target is not None:
            self._pos_target[symbol] = target
        else:
            self._pos_target.pop(symbol, None)

        self._total_trades += 1
        self._trades_by_strategy[strategy]["count"] += 1

    def _clear_position(self, symbol: str) -> None:
        for d in (self._in_position, self._pos_direction, self._pos_entry,
                  self._pos_stop, self._pos_risk, self._pos_target,
                  self._pos_partial, self._pos_best):
            d.pop(symbol, None)

    # ───────────────────────────────────────────────────────────
    #  EXIT MANAGEMENT
    # ───────────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, bar: dict, bars: list,
                    session: SessionState, bar_time: Optional[time]) -> Optional[SignalEvent]:
        """Route to strategy-specific exit logic."""
        strategy = self._in_position.get(symbol, "")
        if strategy == "sweep_hq":
            return self._exit_sweep_hq(symbol, bar)
        if strategy == "orbfail_regime":
            return self._exit_orbfail_regime(symbol, bar, bars, bar_time)
        if strategy == "sweep_primary":
            return self._exit_sweep_primary(symbol, bar, bar_time)
        return None

    def _exit_sweep_hq(self, symbol: str, bar: dict) -> Optional[SignalEvent]:
        """SWEEP_HQ exit: Fixed +1R target or stop.

        Validated exit X3 (Fixed 1R) — best Sharpe (+0.899) for NVDA morning sweeps.
        Fill at exact stop/target price (matches research script assumption).
        """
        direction = self._pos_direction[symbol]
        stop = self._pos_stop[symbol]
        risk = self._pos_risk[symbol]
        entry = self._pos_entry[symbol]
        target = self._pos_target.get(symbol, entry + risk)

        if direction == "long":
            if bar["low"] <= stop:
                return self._emit_exit(symbol, "sweep_hq", direction,
                                       stop, entry, "stop")
            if bar["high"] >= target:
                return self._emit_exit(symbol, "sweep_hq", direction,
                                       target, entry, "target_1r")
        else:
            if bar["high"] >= stop:
                return self._emit_exit(symbol, "sweep_hq", direction,
                                       stop, entry, "stop")
            if bar["low"] <= target:
                return self._emit_exit(symbol, "sweep_hq", direction,
                                       target, entry, "target_1r")
        return None

    def _exit_orbfail_regime(self, symbol: str, bar: dict, bars: list,
                              bar_time: Optional[time]) -> Optional[SignalEvent]:
        """ORBFAIL_REGIME exit: 50%@1R → ATR trail on remainder, EOD fallback.

        X6 partial logic (matches validated research exactly):
          Phase A (partial_done=False):
            - Stop hit → full exit (stop signal).
            - 1R hit   → emit 50% partial close signal to risk manager,
                         activate ATR trail, keep position alive (partial_done=True).
          Phase B (partial_done=True):
            - Trail stop advances from intrabar high-water each bar.
            - Trail hit → full exit of remaining 50% (trail signal).
            - EOD 15:30 → full exit of remaining 50% (eod signal).

        The risk manager closes qty = int(position * partial_pct) on the
        partial signal, and closes 100% of remaining qty on the final exit.
        """
        direction = self._pos_direction[symbol]
        entry = self._pos_entry[symbol]
        stop = self._pos_stop[symbol]
        risk = self._pos_risk[symbol]
        partial_done = self._pos_partial.get(symbol, False)

        atr_val = self._compute_atr(bars)

        if direction == "long":
            # Update high-water mark (wick-based, for ATR trail anchoring)
            best = max(self._pos_best.get(symbol, entry), bar["high"])
            self._pos_best[symbol] = best

            if partial_done:
                # Advance ATR trail from the high-water
                new_trail = best - atr_val
                if new_trail > self._pos_stop[symbol]:
                    self._pos_stop[symbol] = new_trail
                stop = self._pos_stop[symbol]

            # 1. Stop (or trail stop) hit → exit whatever remains
            if bar["low"] <= stop:
                reason = "trail" if partial_done else "stop"
                return self._emit_exit(symbol, "orbfail_regime", direction,
                                       stop, entry, reason)

            # 2. 1R target reached → emit 50% close, activate ATR trail on rest
            if not partial_done and bar["high"] >= entry + risk:
                self._pos_partial[symbol] = True
                trail_stop = max(stop, entry + risk - atr_val)
                self._pos_stop[symbol] = trail_stop
                # Emit the partial close at exactly 1R level
                return self._emit_partial_exit(symbol, "orbfail_regime", direction,
                                               entry + risk, entry)

            # 3. EOD fallback
            elif bar_time is not None and bar_time >= self._EOD_TIME:
                return self._emit_exit(symbol, "orbfail_regime", direction,
                                       bar["close"], entry, "eod")

        else:  # short (directional mirror — not currently triggered by research)
            best = min(self._pos_best.get(symbol, entry), bar["low"])
            self._pos_best[symbol] = best

            if partial_done:
                new_trail = best + atr_val
                if new_trail < self._pos_stop[symbol]:
                    self._pos_stop[symbol] = new_trail
                stop = self._pos_stop[symbol]

            if bar["high"] >= stop:
                reason = "trail" if partial_done else "stop"
                return self._emit_exit(symbol, "orbfail_regime", direction,
                                       stop, entry, reason)

            if not partial_done and bar["low"] <= entry - risk:
                self._pos_partial[symbol] = True
                trail_stop = min(stop, entry - risk + atr_val)
                self._pos_stop[symbol] = trail_stop
                return self._emit_partial_exit(symbol, "orbfail_regime", direction,
                                               entry - risk, entry)

            elif bar_time is not None and bar_time >= self._EOD_TIME:
                return self._emit_exit(symbol, "orbfail_regime", direction,
                                       bar["close"], entry, "eod")

        return None

    def _exit_sweep_primary(self, symbol: str, bar: dict,
                             bar_time: Optional[time]) -> Optional[SignalEvent]:
        """SWEEP_PRIMARY exit: EOD hold (X1) or stop.

        Validated exit X1 (EOD) — sweep trades run all session, taking the
        full reversal move from PDL back toward PDH.
        """
        direction = self._pos_direction[symbol]
        entry = self._pos_entry[symbol]
        stop = self._pos_stop[symbol]

        if direction == "long":
            if bar["low"] <= stop:
                return self._emit_exit(symbol, "sweep_primary", direction,
                                       stop, entry, "stop")
            if bar_time is not None and bar_time >= self._EOD_TIME:
                return self._emit_exit(symbol, "sweep_primary", direction,
                                       bar["close"], entry, "eod")
        else:
            if bar["high"] >= stop:
                return self._emit_exit(symbol, "sweep_primary", direction,
                                       stop, entry, "stop")
            if bar_time is not None and bar_time >= self._EOD_TIME:
                return self._emit_exit(symbol, "sweep_primary", direction,
                                       bar["close"], entry, "eod")
        return None

    def _emit_exit(self, symbol: str, strategy: str, direction: str,
                   price: float, entry: float, reason: str) -> SignalEvent:
        """Record P&L diagnostics, clean up state, return exit signal.

        `price` is the intended fill price (stop/target/close). Passed through
        to SimulatedExecution via `exit_fill_price` metadata so fills match
        the research scripts' assumption (exit at exact stop/target level).
        """
        pnl_pct = ((price - entry) / entry * 100 if direction == "long"
                   else (entry - price) / entry * 100)
        if pnl_pct > 0:
            self._trades_by_strategy[strategy]["wins"] += 1
        self._exit_reasons[reason] += 1
        self._clear_position(symbol)

        exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT
        return SignalEvent(
            symbol=symbol, signal_type=exit_type, strength=0.0,
            strategy_name=self.name,
            metadata={
                "exit_reason": reason, "strategy": strategy,
                "exit_price": round(price, 2), "entry_price": round(entry, 2),
                "exit_fill_price": round(price, 4),
            },
        )

    def _emit_partial_exit(self, symbol: str, strategy: str, direction: str,
                           price: float, entry: float) -> SignalEvent:
        """Emit a 50% partial close WITHOUT clearing position state.

        Position remains active so ATR trail continues on the remaining 50%.
        Risk manager reads metadata['partial_pct'] = 0.5 and closes half.
        `price` should be the 1R level (entry ± risk) to fill at exact target.
        """
        exit_type = SignalType.EXIT_LONG if direction == "long" else SignalType.EXIT_SHORT
        return SignalEvent(
            symbol=symbol, signal_type=exit_type, strength=0.0,
            strategy_name=self.name,
            metadata={
                "exit_reason": "partial_1r",
                "strategy": strategy,
                "exit_price": round(price, 2),
                "entry_price": round(entry, 2),
                "partial_pct": 0.5,
                "exit_fill_price": round(price, 4),
            },
        )

    # ───────────────────────────────────────────────────────────
    #  HELPER FUNCTIONS
    # ───────────────────────────────────────────────────────────

    def _is_near_intraday_extreme(self, session: SessionState,
                                   sweep_wick: float) -> bool:
        """True if sweep wick is within bottom 5% of recent 20-bar intraday range.

        Computed at sweep-bar detection time (today_bars includes sweep bar).
        """
        n = min(20, len(session.today_bars))
        if n < 2:
            return True
        recent = session.today_bars[-n:]
        intra_high = max(b["high"] for b in recent)
        intra_low = min(b["low"] for b in recent)
        intra_range = intra_high - intra_low
        if intra_range <= 0:
            return True
        pct_from_low = (sweep_wick - intra_low) / intra_range
        return pct_from_low <= 0.05

    def _compute_atr(self, bars: list, period: int = 14) -> float:
        """Compute ATR from recent bar history using the features module."""
        n = min(period * 3, len(bars))
        if n < 2:
            val = abs(bars[-1]["high"] - bars[-1]["low"]) if bars else 0.01
            return max(val, 0.01)
        recent = bars[-n:]
        high_arr = np.array([b["high"] for b in recent])
        low_arr = np.array([b["low"] for b in recent])
        close_arr = np.array([b["close"] for b in recent])
        vals = _atr_arr(high_arr, low_arr, close_arr, min(period, n - 1))
        last = vals[-1] if len(vals) > 0 else np.nan
        if np.isnan(last) or last <= 0:
            return max(abs(bars[-1]["high"] - bars[-1]["low"]), 0.01)
        return float(last)

    def _get_bar_time(self, bar: dict) -> Optional[time]:
        ts = bar.get("timestamp")
        if ts is not None and hasattr(ts, "time"):
            return ts.time()
        return None

    def _get_bar_date(self, bar: dict) -> Optional[date]:
        ts = bar.get("timestamp")
        if ts is not None and hasattr(ts, "date"):
            return ts.date() if callable(ts.date) else ts.date
        return None

    # ───────────────────────────────────────────────────────────
    #  DIAGNOSTICS
    # ───────────────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        return {
            "total_trades": self._total_trades,
            "trades_by_strategy": {k: dict(v)
                                    for k, v in self._trades_by_strategy.items()},
            "exit_reasons": dict(self._exit_reasons),
        }

    def print_funnel(self) -> None:
        """Print strategy summary (compatible with demo_backtest.py call site)."""
        print(f"\n  EVENT-DRIVEN STRATEGY v10 (Research-Validated)")
        print(f"  {'═' * 65}")
        print(f"  Three strategies, all validated via Phase 0→4 walk-forward pipeline")
        print(f"  {'─' * 65}")

        rows = [
            ("sweep_hq",
             "NVDA PDL + gap_pos + prior_up + not_fri",
             "4/4 PASS", "+0.933", "Fixed +1R", "Full"),
            ("orbfail_regime",
             "not_goog + gap_up + fail_above_vwap",
             "4/4 PASS", "+0.704", "50%@1R + ATR trail", "Full"),
            ("sweep_primary",
             "all + prior_up + midweek + near_extreme",
             "2/4 MARGINAL", "~+0.29", "EOD", "Half"),
        ]

        print(f"\n  {'Strategy':<18} {'WF':>12} {'OOS Sharpe':>11} "
              f"{'Exit':>20} {'Size':>5}  Trades/WR")
        print(f"  {'─' * 75}")
        for strat, desc, wf, sharpe, exit_m, sz in rows:
            tr = self._trades_by_strategy[strat]
            count, wins = tr["count"], tr["wins"]
            wr = f"{wins/count*100:.0f}%" if count > 0 else "—"
            print(f"  {strat:<18} {wf:>12} {sharpe:>11} {exit_m:>20} "
                  f"{sz:>5}  {count} trades, WR {wr}")
            print(f"    Filter: {desc}")

        print(f"\n  Total trades  : {self._total_trades}")
        if self._exit_reasons:
            print(f"  Exit reasons  :")
            for reason, count in sorted(self._exit_reasons.items(),
                                        key=lambda x: -x[1]):
                pct = count / max(1, self._total_trades) * 100
                print(f"    {reason:>15}: {count:>5}  ({pct:.1f}%)")
