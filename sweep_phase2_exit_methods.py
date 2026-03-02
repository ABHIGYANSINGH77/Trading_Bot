"""sweep_phase2_exit_methods.py — Event Type 2: Session Sweep + Rejection

Phase 2: Exit Method Optimization

Question: Given a confirmed sweep entry, what is the best way to exit?

Phase 1 conclusions (locked in):
  - Primary cohort: M2 conf_close + Filter B (pdl_long + prior_up, n=176) → robust sample
  - Secondary cohort: M1 sweep_close + Filter A (nvda + avoid_13_14 + prior_up, n=113) → best Sharpe
  - Entry method locked: M2 conf_close for primary analysis (100% fill, clean stop rate)

Phase 0 baseline exit: EOD or stop (held until end of session)
  - Filter B M2: +0.467R, +0.17 Sharpe, 39% stop rate, 45% WR

Why exits matter for sweeps:
  - Sweeps are mean-reversion plays. The target is recovery to the prior level OR a measured move.
  - EOD holds capture full reversals but give back profits on partial recoveries.
  - VWAP provides a natural mean-reversion target (price was rejected at PDH/PDL → VWAP = equilibrium)
  - Fixed R targets lock in profits but may exit too early on strong moves.
  - Trailing stops let winners run while protecting profit.

Exit methods tested:

  X1. EOD or stop           — baseline (Phase 0 default)
                               Hold until EOD or stop hit

  X2. VWAP exit             — exit when price crosses VWAP (in our favour)
                               Hypothesis: sweep reverses to VWAP = full mean-reversion target
                               With hard -1R cap on EOD holds
                               (my addition based on Phase 0.5 VWAP importance)

  X3. Fixed 1R target       — exit at +1R profit  (1:1 R:R)
                               Simple, keeps WR high, protects gains

  X4. Fixed 2R target       — exit at +2R profit  (1:2 R:R)
                               Lets winners run more

  X5. Fixed 3R target       — exit at +3R profit  (1:3 R:R)
                               Aggressive, lower WR but bigger winners

  X6. ATR trail (1x)        — trailing stop at 1x ATR(14) below entry (LONG) / above (SHORT)
                               Activated once price moves +0.5R in favour
                               Hypothesis: lets winners run while protecting against choppy reversals

  X7. ATR trail (1.5x)      — trailing stop at 1.5x ATR(14) below entry
                               Wider trail, more forgiving, higher final P&L on strong moves

  X8. PDH/PDL target + stop  — for LONG: target = PDH (prior day high); for SHORT: target = PDL
                               The "full reversal" to the opposite prior level
                               Hypothesis: strongest setups will run the full range

  X9. Partial exit combo     — 50% at +1R, trail remaining 50% with 1x ATR
                               Locks in baseline profit, lets runner work

  X10. Time-based exit       — exit at SPECIFIC hours (test: 12pm, 2pm, 3pm)
                               Sweeps often complete within 90-120 minutes
                               After that they become chop. Best time-cutoff?

Each exit is tested on:
  Primary:   M2 conf_close + Filter B (pdl_long + prior_up, n=176)
  Secondary: M1 sweep_close + Filter A (nvda + avoid_13_14 + prior_up, n=113)
  Baseline:  M2 conf_close, all events (n=1252)

Usage:
  python3 sweep_phase2_exit_methods.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
                                        ./data/cache/NVDA_2024-01-01_2025-12-31_15_mins.csv \\
                                        ./data/cache/MSFT_2024-01-01_2025-12-31_15_mins.csv \\
                                        ./data/cache/AMZN_2024-01-01_2025-12-31_15_mins.csv \\
                                        ./data/cache/GOOG_2024-01-01_2025-12-31_15_mins.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import time as dtime

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MAX_CONF_BARS     = 3      # max bars to find confirmation
ATR_PERIOD        = 14
TRAIL_ACTIVATE_R  = 0.5    # activate trail once trade is +0.5R in profit
PARTIAL_TAKE_R    = 1.0    # partial exit at +1R (X9)
VWAP_HARD_CAP_R   = -1.0  # hard stop on VWAP exit if held to EOD without VWAP cross
MIN_TRADES        = 15

# Filters from Phase 0.5
FILTER_A = lambda ev: ev.symbol == "NVDA" and ev.sweep_hour not in (13, 14) and ev.prior_day_up
FILTER_B = lambda ev: ev.direction == "LONG" and ev.prior_day_up


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SweepSetup:
    """Everything known about a sweep entry (locked from Phase 1)."""
    symbol:          str
    date:            str
    direction:       str      # "LONG" or "SHORT"

    pd_level:        float    # PDH (SHORT) or PDL (LONG)
    pd_high:         float
    pd_low:          float
    pd_range:        float
    sweep_extreme:   float    # wick tip = stop for all methods
    overshoot_abs:   float
    overshoot_pct:   float

    sweep_bar_iloc:  int
    sweep_bar_close: float    # M1 entry
    sweep_hour:      int

    conf_bar_num:    int
    conf_bar_close:  float    # M2 entry (baseline)
    conf_bar_iloc:   int

    prior_day_up:    bool
    gap_pct:         float
    prior_close:     float

    # Full day DataFrame for simulation
    day_df:          object


@dataclass
class ExitResult:
    method:      str
    pnl_r:       float
    mfe_r:       float
    mae_r:       float
    exit_reason: str
    hit_stop:    bool
    bars_held:   int


# ── CSV Loading ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"date": "timestamp"})
    df["symbol"] = Path(path).stem.split("_")[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    df = df[(df["time"] >= dtime(MARKET_OPEN_HOUR, MARKET_OPEN_MIN)) &
            (df["time"] < dtime(16, 0))].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


# ── ATR Computation ───────────────────────────────────────────────────────────

def compute_atr(day_df: pd.DataFrame) -> float:
    """Compute ATR(14) from the current day's bars (intraday rolling)."""
    if len(day_df) < 2:
        return float(day_df["high"].iloc[0] - day_df["low"].iloc[0]) if len(day_df) == 1 else 0.01
    highs  = day_df["high"].values
    lows   = day_df["low"].values
    closes = day_df["close"].values
    trs = [highs[0] - lows[0]]
    for i in range(1, len(day_df)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        trs.append(tr)
    trs = np.array(trs)
    n   = min(ATR_PERIOD, len(trs))
    return float(np.mean(trs[-n:]))


# ── VWAP Computation ─────────────────────────────────────────────────────────

def compute_vwap(day_df: pd.DataFrame, up_to_iloc: int) -> float:
    """Compute VWAP from session open up to (and including) bar at up_to_iloc."""
    sub = day_df.iloc[:up_to_iloc + 1]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    return float((tp * sub["volume"]).sum() / sub["volume"].sum()) if sub["volume"].sum() > 0 else float(tp.mean())


# ── Event Detection (same as Phase 1) ────────────────────────────────────────

def find_setups(df: pd.DataFrame) -> List[SweepSetup]:
    symbol  = df["symbol"].iloc[0]
    setups: List[SweepSetup] = []
    pd_high = pd_low = pd_close = pd_open = prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if pd_high is None:
            pd_high = day["high"].max(); pd_low = day["low"].min()
            pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
            prior_close = day["close"].iloc[-1]
            continue

        pd_range     = pd_high - pd_low
        prior_day_up = pd_close > pd_open
        gap_pct      = ((day["open"].iloc[0] - prior_close) / prior_close
                        if prior_close and prior_close > 0 else 0.0)
        found = False

        for i in range(len(day) - 1):
            if found:
                break
            bar = day.iloc[i]

            # PDH Sweep → SHORT
            if bar["high"] > pd_high and bar["close"] <= pd_high:
                sweep_ext = float(bar["high"])
                ov_abs    = sweep_ext - pd_high
                ov_pct    = ov_abs / pd_high if pd_high > 0 else 0.0
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] < conf["open"] and conf["close"] < pd_high:
                        setups.append(SweepSetup(
                            symbol=symbol, date=str(day_date), direction="SHORT",
                            pd_level=pd_high, pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                            sweep_extreme=sweep_ext, overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                            sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                            sweep_hour=bar["time"].hour,
                            conf_bar_num=k, conf_bar_close=float(conf["close"]),
                            conf_bar_iloc=i + k,
                            prior_day_up=prior_day_up, gap_pct=gap_pct,
                            prior_close=float(prior_close), day_df=day,
                        ))
                        found = True
                        break

            # PDL Sweep → LONG
            elif bar["low"] < pd_low and bar["close"] >= pd_low:
                sweep_ext = float(bar["low"])
                ov_abs    = pd_low - sweep_ext
                ov_pct    = ov_abs / pd_low if pd_low > 0 else 0.0
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] > conf["open"] and conf["close"] > pd_low:
                        setups.append(SweepSetup(
                            symbol=symbol, date=str(day_date), direction="LONG",
                            pd_level=pd_low, pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                            sweep_extreme=sweep_ext, overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                            sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                            sweep_hour=bar["time"].hour,
                            conf_bar_num=k, conf_bar_close=float(conf["close"]),
                            conf_bar_iloc=i + k,
                            prior_day_up=prior_day_up, gap_pct=gap_pct,
                            prior_close=float(prior_close), day_df=day,
                        ))
                        found = True
                        break

        pd_high = day["high"].max(); pd_low = day["low"].min()
        pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
        prior_close = day["close"].iloc[-1]

    return setups


# ── Exit Simulators ───────────────────────────────────────────────────────────

def _favourable(bar: pd.Series, entry: float, direction: str) -> float:
    """Max favourable excursion in this bar."""
    return (bar["high"] - entry) if direction == "LONG" else (entry - bar["low"])

def _adverse(bar: pd.Series, entry: float, direction: str) -> float:
    """Max adverse excursion in this bar (positive = bad)."""
    return (entry - bar["low"]) if direction == "LONG" else (bar["high"] - entry)

def _stop_hit(bar: pd.Series, stop: float, direction: str) -> bool:
    return (bar["low"] <= stop) if direction == "LONG" else (bar["high"] >= stop)

def _target_hit(bar: pd.Series, target: float, direction: str) -> bool:
    return (bar["high"] >= target) if direction == "LONG" else (bar["low"] <= target)


def x1_eod_or_stop(s: SweepSetup, entry: float) -> ExitResult:
    """X1: Baseline — hold to EOD or stop."""
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X1_eod_stop", 0, 0, 0, "zero_risk", False, 0)
    day  = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X1_eod_stop", -1.0, mf, ma, "stop", True, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult("X1_eod_stop", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x2_vwap_exit(s: SweepSetup, entry: float) -> ExitResult:
    """X2: Exit at VWAP cross + hard -1R cap if held to EOD."""
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X2_vwap", 0, 0, 0, "zero_risk", False, 0)
    day  = s.day_df
    mf = ma = 0.0
    for b_num, (idx, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        iloc_abs = s.conf_bar_iloc + 1 + b_num
        vwap     = compute_vwap(day, iloc_abs)
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X2_vwap", -1.0, mf, ma, "stop", True, b_num + 1)
        # VWAP cross: bar close crosses VWAP in favourable direction
        if s.direction == "LONG" and bar["close"] >= vwap:
            pnl = (bar["close"] - entry) / risk
            return ExitResult("X2_vwap", pnl, mf, ma, "vwap", False, b_num + 1)
        if s.direction == "SHORT" and bar["close"] <= vwap:
            pnl = (entry - bar["close"]) / risk
            return ExitResult("X2_vwap", pnl, mf, ma, "vwap", False, b_num + 1)
    # Held to EOD without VWAP cross — apply hard cap
    eod = float(day["close"].iloc[-1])
    raw_pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    pnl = max(raw_pnl, VWAP_HARD_CAP_R)
    return ExitResult("X2_vwap", pnl, mf, ma, "eod_cap", False, len(day) - s.conf_bar_iloc - 1)


def x3_fixed_1r(s: SweepSetup, entry: float) -> ExitResult:
    """X3: Fixed 1R target."""
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X3_1R", 0, 0, 0, "zero_risk", False, 0)
    target = (entry + risk) if s.direction == "LONG" else (entry - risk)
    day    = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X3_1R", -1.0, mf, ma, "stop", True, b_num + 1)
        if _target_hit(bar, target, s.direction):
            return ExitResult("X3_1R", 1.0, mf, ma, "target", False, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult("X3_1R", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x4_fixed_2r(s: SweepSetup, entry: float) -> ExitResult:
    """X4: Fixed 2R target."""
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X4_2R", 0, 0, 0, "zero_risk", False, 0)
    target = (entry + 2 * risk) if s.direction == "LONG" else (entry - 2 * risk)
    day    = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X4_2R", -1.0, mf, ma, "stop", True, b_num + 1)
        if _target_hit(bar, target, s.direction):
            return ExitResult("X4_2R", 2.0, mf, ma, "target", False, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult("X4_2R", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x5_fixed_3r(s: SweepSetup, entry: float) -> ExitResult:
    """X5: Fixed 3R target."""
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X5_3R", 0, 0, 0, "zero_risk", False, 0)
    target = (entry + 3 * risk) if s.direction == "LONG" else (entry - 3 * risk)
    day    = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X5_3R", -1.0, mf, ma, "stop", True, b_num + 1)
        if _target_hit(bar, target, s.direction):
            return ExitResult("X5_3R", 3.0, mf, ma, "target", False, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult("X5_3R", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x6_atr_trail_1x(s: SweepSetup, entry: float) -> ExitResult:
    """X6: ATR trailing stop at 1x ATR, activated after +0.5R."""
    return _atr_trail(s, entry, "X6_atr1x", multiplier=1.0)


def x7_atr_trail_15x(s: SweepSetup, entry: float) -> ExitResult:
    """X7: ATR trailing stop at 1.5x ATR, activated after +0.5R."""
    return _atr_trail(s, entry, "X7_atr15x", multiplier=1.5)


def _atr_trail(s: SweepSetup, entry: float, label: str, multiplier: float) -> ExitResult:
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult(label, 0, 0, 0, "zero_risk", False, 0)
    day    = s.day_df
    atr    = compute_atr(day.iloc[:s.conf_bar_iloc + 1])
    trail  = atr * multiplier
    # Trail tracks the best close seen, activated once +0.5R ahead
    activated    = False
    trail_stop   = stop   # starts at original stop until activated
    best_price   = entry  # highest close seen (LONG) / lowest (SHORT)
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        # Check activation
        if not activated:
            curr_profit = fav
            if curr_profit >= TRAIL_ACTIVATE_R:
                activated = True
        # Update trailing stop
        if activated:
            if s.direction == "LONG":
                best_price = max(best_price, float(bar["close"]))
                trail_stop = max(trail_stop, best_price - trail)
            else:
                best_price = min(best_price, float(bar["close"]))
                trail_stop = min(trail_stop, best_price + trail)
        # Check stop hit (use trailing stop once activated, else original)
        eff_stop = trail_stop if activated else stop
        if _stop_hit(bar, eff_stop, s.direction):
            pnl_r = (eff_stop - entry) / risk if s.direction == "LONG" else (entry - eff_stop) / risk
            return ExitResult(label, pnl_r, mf, ma, "trail_stop", eff_stop == stop, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult(label, pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x8_pd_level_target(s: SweepSetup, entry: float) -> ExitResult:
    """X8: Full reversal target — LONG targets PDH; SHORT targets PDL."""
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X8_pd_target", 0, 0, 0, "zero_risk", False, 0)
    # Target: full reversal to opposite prior level
    target = s.pd_high if s.direction == "LONG" else s.pd_low
    day    = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult("X8_pd_target", -1.0, mf, ma, "stop", True, b_num + 1)
        if _target_hit(bar, target, s.direction):
            pnl = (target - entry) / risk if s.direction == "LONG" else (entry - target) / risk
            return ExitResult("X8_pd_target", pnl, mf, ma, "target", False, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult("X8_pd_target", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x9_partial_1r_trail(s: SweepSetup, entry: float) -> ExitResult:
    """X9: Partial exit — take 50% at +1R, trail remaining 50% with 1x ATR.
    Reported as blended P&L = 0.5 * first_exit + 0.5 * trail_exit.
    """
    stop   = s.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return ExitResult("X9_partial", 0, 0, 0, "zero_risk", False, 0)
    target1  = (entry + risk) if s.direction == "LONG" else (entry - risk)
    day      = s.day_df
    atr      = compute_atr(day.iloc[:s.conf_bar_iloc + 1])
    trail    = atr * 1.0
    partial_taken   = False
    partial_pnl     = 0.0
    partial_bar     = 0
    activated       = False
    trail_stop      = stop
    best_price      = entry
    mf = ma = 0.0

    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)

        # Check stop (full stop if partial not yet taken)
        eff_stop = trail_stop if (partial_taken and activated) else stop
        if _stop_hit(bar, eff_stop, s.direction):
            stop_pnl = (eff_stop - entry) / risk if s.direction == "LONG" else (entry - eff_stop) / risk
            if partial_taken:
                blended = 0.5 * partial_pnl + 0.5 * stop_pnl
            else:
                blended = stop_pnl
            reason  = "trail_stop" if (partial_taken and activated) else "stop"
            return ExitResult("X9_partial", blended, mf, ma, reason, not partial_taken or not activated, b_num + 1)

        # Partial exit at +1R
        if not partial_taken and _target_hit(bar, target1, s.direction):
            partial_taken = True
            partial_pnl   = 1.0
            partial_bar   = b_num
            # Activate trail from here
            activated     = True
            best_price    = target1

        # Update trail
        if activated:
            if s.direction == "LONG":
                best_price = max(best_price, float(bar["close"]))
                trail_stop = max(trail_stop, best_price - trail)
            else:
                best_price = min(best_price, float(bar["close"]))
                trail_stop = min(trail_stop, best_price + trail)

    eod     = float(day["close"].iloc[-1])
    eod_pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    if partial_taken:
        blended = 0.5 * partial_pnl + 0.5 * eod_pnl
    else:
        blended = eod_pnl
    return ExitResult("X9_partial", blended, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


def x10_time_exit(s: SweepSetup, entry: float, cutoff_hour: int) -> ExitResult:
    """X10: Exit at specified hour (or stop)."""
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    if risk <= 0:
        return ExitResult(f"X10_t{cutoff_hour}h", 0, 0, 0, "zero_risk", False, 0)
    day   = s.day_df
    mf = ma = 0.0
    for b_num, (_, bar) in enumerate(day.iloc[s.conf_bar_iloc + 1:].iterrows()):
        fav = _favourable(bar, entry, s.direction) / risk
        adv = _adverse(bar, entry, s.direction) / risk
        mf = max(mf, fav); ma = min(ma, -adv)
        if _stop_hit(bar, stop, s.direction):
            return ExitResult(f"X10_t{cutoff_hour}h", -1.0, mf, ma, "stop", True, b_num + 1)
        bar_hour = bar.name  # index — need time from day_df
        # bar is the row; use iloc to get the time
        iloc_abs = s.conf_bar_iloc + 1 + b_num
        bar_time = day.iloc[iloc_abs]["time"] if "time" in day.columns else dtime(16, 0)
        if bar_time.hour >= cutoff_hour:
            pnl = (bar["close"] - entry) / risk if s.direction == "LONG" else (entry - bar["close"]) / risk
            return ExitResult(f"X10_t{cutoff_hour}h", pnl, mf, ma, "time", False, b_num + 1)
    eod = float(day["close"].iloc[-1])
    pnl = (eod - entry) / risk if s.direction == "LONG" else (entry - eod) / risk
    return ExitResult(f"X10_t{cutoff_hour}h", pnl, mf, ma, "eod", False, len(day) - s.conf_bar_iloc - 1)


# ── Statistics ────────────────────────────────────────────────────────────────

def sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    if arr.std() < 1e-9:
        return 0.0
    return float(arr.mean() / arr.std())


def profit_factor(pnls: List[float]) -> float:
    wins  = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def print_exit_row(label: str, results: List[ExitResult]) -> None:
    if not results:
        return
    pnls   = [r.pnl_r for r in results]
    exp    = float(np.mean(pnls))
    wr     = sum(1 for p in pnls if p > 0) / len(pnls)
    sh     = sharpe(pnls)
    pf     = profit_factor(pnls)
    sr     = sum(1 for r in results if r.hit_stop) / len(results)
    mfe    = float(np.mean([r.mfe_r for r in results]))
    mae    = float(np.mean([r.mae_r for r in results]))
    avg_bars = float(np.mean([r.bars_held for r in results]))
    print(f"  {label:<22s}  n={len(results):4d}  exp={exp:+.3f}R  WR={wr:4.0%}  "
          f"Sharpe={sh:+.3f}  PF={pf:5.2f}  stop={sr:3.0%}  "
          f"MFE={mfe:+.3f}R  MAE={mae:+.3f}R  bars={avg_bars:.1f}")


def exit_reason_breakdown(results: List[ExitResult]) -> None:
    reasons: dict = {}
    for r in results:
        reasons[r.exit_reason] = reasons.get(r.exit_reason, 0) + 1
    total = len(results)
    parts = [f"{k}: {v/total:.0%}" for k, v in sorted(reasons.items())]
    print(f"    Exits → {' | '.join(parts)}")


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_cohort(setups: List[SweepSetup], label: str, entry_fn) -> None:
    if len(setups) < MIN_TRADES:
        print(f"\n  {label}: only {len(setups)} setups, skipping")
        return

    print(f"\n  {'═'*115}")
    print(f"  {label}  (n={len(setups)} setups)")
    print(f"  {'═'*115}")
    print(f"\n  {'Method':<22s}  {'n':>4}  {'Exp':>8}  {'WR':>4}  {'Sharpe':>7}  "
          f"{'PF':>6}  {'Stop%':>6}  {'MFE':>8}  {'MAE':>8}  {'Bars':>5}")
    print(f"  {'─'*115}")

    # Run each exit on each setup
    all_exits = {
        "X1 EOD/stop      ★BL": [x1_eod_or_stop(s, entry_fn(s)) for s in setups],
        "X2 VWAP exit        ": [x2_vwap_exit(s, entry_fn(s)) for s in setups],
        "X3 Fixed 1R         ": [x3_fixed_1r(s, entry_fn(s)) for s in setups],
        "X4 Fixed 2R         ": [x4_fixed_2r(s, entry_fn(s)) for s in setups],
        "X5 Fixed 3R         ": [x5_fixed_3r(s, entry_fn(s)) for s in setups],
        "X6 ATR trail 1x     ": [x6_atr_trail_1x(s, entry_fn(s)) for s in setups],
        "X7 ATR trail 1.5x   ": [x7_atr_trail_15x(s, entry_fn(s)) for s in setups],
        "X8 PD opposite level": [x8_pd_level_target(s, entry_fn(s)) for s in setups],
        "X9 50% at 1R+trail  ": [x9_partial_1r_trail(s, entry_fn(s)) for s in setups],
        "X10 time exit 12pm  ": [x10_time_exit(s, entry_fn(s), cutoff_hour=12) for s in setups],
        "X10 time exit 14pm  ": [x10_time_exit(s, entry_fn(s), cutoff_hour=14) for s in setups],
        "X10 time exit 15pm  ": [x10_time_exit(s, entry_fn(s), cutoff_hour=15) for s in setups],
    }

    best_sharpe = -999.0
    best_label  = ""
    best_exp    = -999.0

    for lbl, results in all_exits.items():
        pnls = [r.pnl_r for r in results]
        sh   = sharpe(pnls)
        exp  = float(np.mean(pnls))
        print_exit_row(lbl, results)
        if len(results) >= MIN_TRADES and sh > best_sharpe:
            best_sharpe = sh
            best_label  = lbl
            best_exp    = exp

    print(f"\n  Exit reason breakdown:")
    for lbl, results in all_exits.items():
        print(f"    {lbl.strip():<22s}: ", end="")
        exit_reason_breakdown(results)

    print(f"\n  VERDICT: Best exit by Sharpe → {best_label.strip()}  "
          f"exp={best_exp:+.3f}R  Sharpe={best_sharpe:+.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase2_exit_methods.py <csv_files...>")
        sys.exit(1)

    all_setups: List[SweepSetup] = []
    for path in sys.argv[1:]:
        df   = load_csv(path)
        sym  = Path(path).stem.split("_")[0]
        evs  = find_setups(df)
        print(f"  {sym:6s}  :  {len(evs):4d} setups")
        all_setups.extend(evs)

    # Entry functions (locked from Phase 1)
    def m2_entry(s: SweepSetup) -> float:
        return s.conf_bar_close  # M2: conf bar close

    def m1_entry(s: SweepSetup) -> float:
        return s.sweep_bar_close  # M1: sweep bar close

    print()
    print("=" * 120)
    print("  EVENT TYPE 2: SWEEP + REJECTION — PHASE 2 EXIT METHOD OPTIMIZATION")
    print("  Question: Given a confirmed sweep entry, what is the best way to exit?")
    print("  Baseline exit (X1): EOD or stop — all other methods tested relative to this")
    print("=" * 120)

    # Cohort 1: M2 entry, all events
    run_cohort(all_setups, "ALL EVENTS — M2 conf_close (Phase 0 baseline)", m2_entry)

    # Cohort 2: M2 entry + Filter B (pdl_long + prior_up) — primary
    b_setups = [s for s in all_setups if FILTER_B(s)]
    run_cohort(b_setups, "PRIMARY — M2 conf_close + Filter B (pdl_long+prior_up)", m2_entry)

    # Cohort 3: M1 entry + Filter A (nvda + avoid_13_14 + prior_up) — secondary
    a_setups = [s for s in all_setups if FILTER_A(s)]
    run_cohort(a_setups, "SECONDARY — M1 sweep_close + Filter A (nvda+avoid13_14+prior_up)", m1_entry)

    # Cohort 4: M2 entry + Filter A (to isolate exit method from entry method on Filter A)
    run_cohort(a_setups, "FILTER A — M2 conf_close + Filter A (apples-to-apples)", m2_entry)

    print()
    print("=" * 120)
    print("  OVERALL EXIT RECOMMENDATION")
    print("=" * 120)
    print("""
  Phase 2 purpose: lock in the best exit method for Phase 3 (regime filter).
  Key questions answered:
    1. Does VWAP serve as a natural mean-reversion target? (X2)
    2. Is there a meaningful R target (1R/2R/3R) that outperforms EOD? (X3/X4/X5)
    3. Do trailing stops protect gains on strong reversals? (X6/X7)
    4. Is the full PD-range target realistic? (X8)
    5. Does the partial exit hybrid (X9) give best risk-adjusted return?
    6. Is there a time-of-day where exiting is clearly better? (X10)

  → Winner for Phase 3 and Phase 4: best Sharpe across PRIMARY and SECONDARY cohorts
  → If split decision, prefer PRIMARY (Filter B) as it has more trades (n≈176 vs n≈113)
    """)


if __name__ == "__main__":
    main()
