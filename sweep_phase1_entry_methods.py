"""sweep_phase1_entry_methods.py — Event Type 2: Session Sweep + Rejection

Phase 1: Entry Method Optimization

Question: What is the best way to enter a confirmed PDH/PDL sweep?

Context from Phase 0.5:
  Best filter (n≥100): nvda + avoid_13_14 + prior_up → n=111, exp=+0.589R
  Broad filter:        pdl_long + prior_up           → n=176, exp=+0.467R
  Baseline (all):      n=1235,                          exp=+0.080R

The sweep setup is FUNDAMENTALLY different from ORB:
  ORB:   price breaks → CONTINUE → limit entries have adverse selection (best moves never pull back)
  Sweep: price breaks → REJECT  → retest of the level is a natural part of the move
         The question is whether entering at the level retest gives a better R:R than
         entering at the confirmation bar close.

Methods tested (all use same stop: sweep_extreme):

  M1. Sweep bar close   — enter at close of sweep bar, NO confirmation required
                          (aggressive: tests whether confirmation step is even needed)
                          Stop = sweep_extreme | Fill = 100%

  M2. Confirmation close — baseline (Phase 0 entry): close of the confirmation bar
                           Stop = sweep_extreme | Fill = 100%

  M3. Next bar open      — open of the bar AFTER confirmation bar
                           Tests whether delay improves price vs costs slippage
                           Stop = sweep_extreme | Fill = 100%

  M4. Limit at level     — after confirmation, place limit order AT the PDH/PDL level
                           Fill when price touches the level within next 5 bars
                           Stop = sweep_extreme | Risk = overshoot_abs (very tight!)
                           Hypothesis: retest of PDH/PDL as S/R gives best R:R
                           Key question: adverse selection? (do best reversals never retest?)

  M5. Retest close       — after confirmation, wait for a bar to CLOSE within 0.15% of
                           the level (from inside: close ≤ PDH for SHORT, ≥ PDL for LONG)
                           Fill at close of retest bar | Stop = sweep_extreme
                           More conservative than M4: requires confirmed close near level

  M6. Level-touch + confirm — (my addition) after confirmation, wait for price to touch
                              the level AND the next bar to be directional (close in
                              reversal direction). Entry = close of that confirming bar.
                              Two-step confirmation: sweep → conf → retest → second conf.
                              Hypothesis: highest quality, lowest fill rate.

Each method is run on:
  A. All events (baseline, no filter)
  B. Phase 0.5 filter A: nvda + avoid_13_14 + prior_up  (n≈111)
  C. Phase 0.5 filter B: pdl_long + prior_up             (n≈176)

Adverse selection test:
  For limit/retest methods (M4, M5, M6): trades that DID NOT fill are simulated at
  confirmation bar close — if unfilled trades earn MORE than filled trades, the limit
  entry has adverse selection (best reversals never return to the level).

Usage:
  python3 sweep_phase1_entry_methods.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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

MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
MAX_CONF_BARS      = 3     # max bars to look for confirmation
RETEST_WINDOW      = 5     # bars to scan for retest after confirmation
RETEST_TOLERANCE   = 0.0015   # 0.15% — how close price must close to the level
MIN_TRADES         = 15

# Phase 0.5 best filters
FILTER_A = lambda ev: ev.symbol == "NVDA" and ev.sweep_hour not in (13, 14) and ev.prior_day_up
FILTER_B = lambda ev: ev.direction == "LONG" and ev.prior_day_up


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SweepSetup:
    """Everything known about a sweep before any entry decision."""
    symbol:          str
    date:            str
    direction:       str     # "LONG" or "SHORT"

    pd_level:        float   # PDH (SHORT) or PDL (LONG)
    sweep_extreme:   float   # wick tip = stop for all methods
    overshoot_abs:   float   # sweep_extreme - pd_level  (very tight risk for limit entries)
    overshoot_pct:   float

    # Sweep bar
    sweep_bar_iloc:  int
    sweep_bar_close: float   # used for M1 entry
    sweep_hour:      int

    # Confirmation bar
    conf_bar_num:    int
    conf_bar_close:  float   # used for M2 entry
    conf_bar_iloc:   int

    # Prior day context
    pd_high:         float
    pd_low:          float
    pd_range:        float
    prior_day_up:    bool
    gap_pct:         float
    prior_close:     float

    # Full day DataFrame (for post-entry simulation)
    day_df:          object


@dataclass
class EntryResult:
    method:       str
    filled:       bool
    entry_price:  float
    stop_price:   float
    risk:         float       # abs(entry - stop)
    risk_vs_conf: float       # risk / conf_bar_risk  (1.0 = same, < 1 = tighter)
    bars_to_fill: int         # 0 = immediate
    pnl_r:        float       # 0.0 if not filled
    mfe_r:        float
    mae_r:        float
    exit_reason:  str
    hit_stop:     bool
    # For adverse selection: unfilled trades only
    conf_pnl_r:   float = 0.0  # what conf-close entry would have earned (for unfilled only)


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


# ── Event Detection ───────────────────────────────────────────────────────────

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

        pd_range    = pd_high - pd_low
        prior_day_up = pd_close > pd_open
        gap_pct     = ((day["open"].iloc[0] - prior_close) / prior_close
                       if prior_close and prior_close > 0 else 0.0)
        found = False

        for i in range(len(day) - 1):
            if found:
                break
            bar = day.iloc[i]

            # PDH Sweep → SHORT
            if bar["high"] > pd_high and bar["close"] <= pd_high:
                sweep_ext  = float(bar["high"])
                ov_abs     = sweep_ext - pd_high
                ov_pct     = ov_abs / pd_high if pd_high > 0 else 0.0
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] < conf["open"] and conf["close"] < pd_high:
                        setups.append(SweepSetup(
                            symbol=symbol, date=str(day_date), direction="SHORT",
                            pd_level=pd_high, sweep_extreme=sweep_ext,
                            overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                            sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                            sweep_hour=bar["time"].hour,
                            conf_bar_num=k, conf_bar_close=float(conf["close"]),
                            conf_bar_iloc=i + k,
                            pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                            prior_day_up=prior_day_up, gap_pct=gap_pct,
                            prior_close=float(prior_close), day_df=day,
                        ))
                        found = True
                        break

            # PDL Sweep → LONG
            elif bar["low"] < pd_low and bar["close"] >= pd_low:
                sweep_ext  = float(bar["low"])
                ov_abs     = pd_low - sweep_ext
                ov_pct     = ov_abs / pd_low if pd_low > 0 else 0.0
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] > conf["open"] and conf["close"] > pd_low:
                        setups.append(SweepSetup(
                            symbol=symbol, date=str(day_date), direction="LONG",
                            pd_level=pd_low, sweep_extreme=sweep_ext,
                            overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                            sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                            sweep_hour=bar["time"].hour,
                            conf_bar_num=k, conf_bar_close=float(conf["close"]),
                            conf_bar_iloc=i + k,
                            pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                            prior_day_up=prior_day_up, gap_pct=gap_pct,
                            prior_close=float(prior_close), day_df=day,
                        ))
                        found = True
                        break

        pd_high = day["high"].max(); pd_low = day["low"].min()
        pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
        prior_close = day["close"].iloc[-1]

    return setups


# ── Trade Exit Simulation ─────────────────────────────────────────────────────

def sim_exit(entry: float, stop: float, risk: float, direction: str,
             remaining: pd.DataFrame, eod_price: float) -> Tuple[float, float, float, str, bool]:
    """Simulate EOD-or-stop exit. Returns (pnl_r, mfe_r, mae_r, exit_reason, hit_stop)."""
    mf = ma = 0.0
    for i, (_, bar) in enumerate(remaining.iterrows()):
        fav = (bar["high"] - entry) / risk if direction == "LONG" else (entry - bar["low"])  / risk
        adv = (bar["low"]  - entry) / risk if direction == "LONG" else (entry - bar["high"]) / risk
        mf = max(mf, fav); ma = min(ma, adv)
        if direction == "LONG"  and bar["low"]  <= stop:
            return (stop - entry) / risk, mf, ma, "stop", True
        if direction == "SHORT" and bar["high"] >= stop:
            return (entry - stop) / risk, mf, ma, "stop", True
    pnl = ((eod_price - entry) / risk if direction == "LONG"
           else (entry - eod_price) / risk)
    return pnl, mf, ma, "eod", False


def conf_close_pnl(s: SweepSetup) -> float:
    """P&L if entered at confirmation bar close (for unfilled trade comparison)."""
    entry = s.conf_bar_close
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    if risk <= 0:
        return 0.0
    remaining = s.day_df.iloc[s.conf_bar_iloc + 1:]
    eod       = float(s.day_df["close"].iloc[-1])
    pnl, *_   = sim_exit(entry, stop, risk, s.direction, remaining, eod)
    return pnl


# ── Entry Simulators ──────────────────────────────────────────────────────────

def m1_sweep_close(s: SweepSetup) -> EntryResult:
    """M1: Enter at close of sweep bar — no confirmation required."""
    entry     = s.sweep_bar_close
    stop      = s.sweep_extreme
    risk      = abs(entry - stop)
    conf_risk = abs(s.conf_bar_close - s.sweep_extreme)
    if risk <= 0:
        return EntryResult("M1_sweep_close", False, 0, stop, 0, 1, 0, 0, 0, 0, "n/a", False)
    remaining = s.day_df.iloc[s.sweep_bar_iloc + 1:]
    eod       = float(s.day_df["close"].iloc[-1])
    pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
    return EntryResult("M1_sweep_close", True, entry, stop, risk,
                       risk / conf_risk if conf_risk > 0 else 1.0,
                       0, pnl, mf, ma, reason, hs)


def m2_conf_close(s: SweepSetup) -> EntryResult:
    """M2: Enter at confirmation bar close — baseline."""
    entry     = s.conf_bar_close
    stop      = s.sweep_extreme
    risk      = abs(entry - stop)
    if risk <= 0:
        return EntryResult("M2_conf_close", False, 0, stop, 0, 1, 0, 0, 0, 0, "n/a", False)
    remaining = s.day_df.iloc[s.conf_bar_iloc + 1:]
    eod       = float(s.day_df["close"].iloc[-1])
    pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
    return EntryResult("M2_conf_close", True, entry, stop, risk, 1.0,
                       0, pnl, mf, ma, reason, hs)


def m3_next_open(s: SweepSetup) -> EntryResult:
    """M3: Enter at open of bar after confirmation."""
    day       = s.day_df
    next_iloc = s.conf_bar_iloc + 1
    if next_iloc >= len(day):
        return EntryResult("M3_next_open", False, 0, s.sweep_extreme, 0, 1, 0, 0, 0, 0, "no_bar", False)
    entry     = float(day.iloc[next_iloc]["open"])
    stop      = s.sweep_extreme
    risk      = abs(entry - stop)
    conf_risk = abs(s.conf_bar_close - s.sweep_extreme)
    if risk <= 0:
        return EntryResult("M3_next_open", False, 0, stop, 0, 1, 1, 0, 0, 0, "n/a", False)
    remaining = day.iloc[next_iloc + 1:]
    eod       = float(day["close"].iloc[-1])
    pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
    return EntryResult("M3_next_open", True, entry, stop, risk,
                       risk / conf_risk if conf_risk > 0 else 1.0,
                       1, pnl, mf, ma, reason, hs)


def m4_limit_at_level(s: SweepSetup) -> EntryResult:
    """M4: Limit order at PDH/PDL after confirmation — fill when price touches level."""
    day    = s.day_df
    entry  = s.pd_level          # limit price = the level itself
    stop   = s.sweep_extreme
    risk   = s.overshoot_abs     # very tight: only the wick overshoot
    conf_risk = abs(s.conf_bar_close - s.sweep_extreme)
    if risk <= 0:
        return EntryResult("M4_limit_level", False, 0, stop, 0, 1, 0, 0, 0, 0, "n/a", False)

    # Scan RETEST_WINDOW bars after confirmation for a touch of the level
    scan_start = s.conf_bar_iloc + 1
    for j in range(RETEST_WINDOW):
        idx = scan_start + j
        if idx >= len(day):
            break
        bar = day.iloc[idx]
        # SHORT: price must rally back up to PDH (bar high >= pdh)
        # LONG:  price must drop back down to PDL (bar low <= pdl)
        filled = ((s.direction == "SHORT" and bar["high"] >= entry) or
                  (s.direction == "LONG"  and bar["low"]  <= entry))
        if filled:
            remaining = day.iloc[idx + 1:]
            eod       = float(day["close"].iloc[-1])
            pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
            return EntryResult("M4_limit_level", True, entry, stop, risk,
                               risk / conf_risk if conf_risk > 0 else 1.0,
                               j + 1, pnl, mf, ma, reason, hs)

    # Not filled — compute what conf-close would have earned (adverse selection check)
    cp = conf_close_pnl(s)
    return EntryResult("M4_limit_level", False, 0, stop, 0,
                       risk / conf_risk if conf_risk > 0 else 1.0,
                       RETEST_WINDOW, 0, 0, 0, "not_filled", False, conf_pnl_r=cp)


def m5_retest_close(s: SweepSetup) -> EntryResult:
    """M5: Wait for a bar to CLOSE within RETEST_TOLERANCE of PDH/PDL from inside."""
    day       = s.day_df
    level     = s.pd_level
    stop      = s.sweep_extreme
    conf_risk = abs(s.conf_bar_close - s.sweep_extreme)

    scan_start = s.conf_bar_iloc + 1
    for j in range(RETEST_WINDOW):
        idx = scan_start + j
        if idx >= len(day):
            break
        bar = day.iloc[idx]
        close = float(bar["close"])
        # SHORT: close within tolerance ABOVE/AT level but not above stop
        # LONG:  close within tolerance BELOW/AT level but not below stop
        if s.direction == "SHORT":
            near = (close >= level * (1 - RETEST_TOLERANCE) and close <= level)
        else:
            near = (close <= level * (1 + RETEST_TOLERANCE) and close >= level)
        if near:
            entry = close
            risk  = abs(entry - stop)
            if risk <= 0:
                continue
            remaining = day.iloc[idx + 1:]
            eod       = float(day["close"].iloc[-1])
            pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
            return EntryResult("M5_retest_close", True, entry, stop, risk,
                               risk / conf_risk if conf_risk > 0 else 1.0,
                               j + 1, pnl, mf, ma, reason, hs)

    cp = conf_close_pnl(s)
    return EntryResult("M5_retest_close", False, 0, stop, 0,
                       abs(s.overshoot_abs) / conf_risk if conf_risk > 0 else 1.0,
                       RETEST_WINDOW, 0, 0, 0, "not_filled", False, conf_pnl_r=cp)


def m6_touch_then_confirm(s: SweepSetup) -> EntryResult:
    """M6: Wait for level touch AND a subsequent directional bar close.
    Two-step filter: sweep → conf → retest touch → second directional bar.
    Hypothesis: highest quality setup, lowest fill rate.
    """
    day       = s.day_df
    level     = s.pd_level
    stop      = s.sweep_extreme
    conf_risk = abs(s.conf_bar_close - s.sweep_extreme)

    scan_start = s.conf_bar_iloc + 1
    touched    = False
    touch_iloc = -1

    for j in range(RETEST_WINDOW):
        idx = scan_start + j
        if idx >= len(day):
            break
        bar = day.iloc[idx]
        # Detect touch of level
        if not touched:
            touch = ((s.direction == "SHORT" and bar["high"] >= level) or
                     (s.direction == "LONG"  and bar["low"]  <= level))
            if touch:
                touched    = True
                touch_iloc = idx
        else:
            # Next bar after touch: requires directional close
            conf2 = day.iloc[idx]
            if s.direction == "SHORT":
                valid = conf2["close"] < conf2["open"] and conf2["close"] < level
            else:
                valid = conf2["close"] > conf2["open"] and conf2["close"] > level
            if valid:
                entry = float(conf2["close"])
                risk  = abs(entry - stop)
                if risk <= 0:
                    break
                remaining = day.iloc[idx + 1:]
                eod       = float(day["close"].iloc[-1])
                pnl, mf, ma, reason, hs = sim_exit(entry, stop, risk, s.direction, remaining, eod)
                return EntryResult("M6_touch_confirm", True, entry, stop, risk,
                                   risk / conf_risk if conf_risk > 0 else 1.0,
                                   j + 1, pnl, mf, ma, reason, hs)
            else:
                # Touch but no directional bar — reset and keep scanning
                touched = False

    cp = conf_close_pnl(s)
    return EntryResult("M6_touch_confirm", False, 0, stop, 0,
                       abs(s.overshoot_abs) / conf_risk if conf_risk > 0 else 1.0,
                       RETEST_WINDOW, 0, 0, 0, "not_filled", False, conf_pnl_r=cp)


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(pnls: List[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "sharpe": 0.0, "avg_w": 0.0, "avg_l": 0.0}
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    exp  = float(np.mean(pnls))
    std  = float(np.std(pnls))
    return {
        "n":     len(pnls),
        "exp":   exp,
        "wr":    len(wins) / len(pnls) * 100,
        "sharpe": exp / std if std > 0 else 0.0,
        "avg_w": float(np.mean(wins)) if wins else 0.0,
        "avg_l": float(np.mean(loss)) if loss else 0.0,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def method_row(label: str, entries: List[EntryResult], conf_baseline: float) -> None:
    filled    = [e for e in entries if e.filled]
    unfilled  = [e for e in entries if not e.filled]
    fill_rate = len(filled) / len(entries) * 100 if entries else 0

    pnls = [e.pnl_r for e in filled]
    if not pnls:
        print(f"  {label:<22s}  fill=  0%  n=  0  (no fills)")
        return

    s    = stats(pnls)
    flag = "★★" if s["exp"] > 0.40 else ("★ " if s["exp"] > 0.15 else "  ")
    diff = s["exp"] - conf_baseline
    # Risk relative to conf close (1.0 = same, <1 = tighter stop)
    avg_risk_ratio = float(np.mean([e.risk_vs_conf for e in filled])) if filled else 1.0
    print(f"  {label:<22s}  fill={fill_rate:>4.0f}%  n={s['n']:>4}  "
          f"exp={s['exp']:>+.3f}R  WR={s['wr']:>4.0f}%  Sharpe={s['sharpe']:>+.2f}  "
          f"Δ={diff:>+.3f}R  risk×{avg_risk_ratio:.2f}  {flag}")


def adverse_selection_row(label: str, entries: List[EntryResult]) -> None:
    filled   = [e for e in entries if e.filled]
    unfilled = [e for e in entries if not e.filled]
    if not unfilled:
        return
    fill_pnls = [e.pnl_r for e in filled]
    unf_pnls  = [e.conf_pnl_r for e in unfilled]  # what conf-close earned on missed trades
    fs = stats(fill_pnls)
    us = stats(unf_pnls)
    adverse = us["exp"] > fs["exp"] + 0.05  # unfilled trades clearly better = adverse selection
    verdict = "⚠ ADVERSE SELECTION (best trades never retest)" if adverse else "✓ No adverse selection"
    print(f"  {label:<22s}  Filled: exp={fs['exp']:>+.3f}R n={fs['n']:>3} | "
          f"Unfilled: exp={us['exp']:>+.3f}R n={us['n']:>3} | {verdict}")


def run_and_print(setups: List[SweepSetup], label: str) -> None:
    if len(setups) < MIN_TRADES:
        print(f"  {label}: too few setups ({len(setups)})")
        return

    W = 115
    print(f"\n  {'═' * W}")
    print(f"  {label}  (n={len(setups)} setups)")
    print(f"  {'═' * W}")

    simulators = [
        ("M1 sweep_close    ", m1_sweep_close),
        ("M2 conf_close ★BL ", m2_conf_close),
        ("M3 next_open      ", m3_next_open),
        ("M4 limit_at_level ", m4_limit_at_level),
        ("M5 retest_close   ", m5_retest_close),
        ("M6 touch_confirm  ", m6_touch_then_confirm),
    ]

    # Compute baseline (M2)
    m2_results  = [m2_conf_close(s) for s in setups]
    conf_base   = float(np.mean([e.pnl_r for e in m2_results if e.filled]))

    all_results: dict = {}
    print(f"\n  {'Method':<22s}  {'Fill':>6}  {'n':>4}  {'Exp':>8}  {'WR':>5}  "
          f"{'Sharpe':>6}  {'Δ vs M2':>8}  {'Risk×':>6}")
    print(f"  {'─' * W}")

    for name, sim_fn in simulators:
        results = [sim_fn(s) for s in setups]
        all_results[name.strip()] = results
        method_row(name, results, conf_base)

    # ── MFE / MAE comparison ─────────────────────────────────────────────────
    print(f"\n  MFE / MAE (avg, filled trades only):")
    for name, sim_fn in simulators:
        results = all_results[name.strip()]
        filled  = [e for e in results if e.filled]
        if not filled:
            continue
        mfe = np.mean([e.mfe_r for e in filled])
        mae = np.mean([e.mae_r for e in filled])
        stop_rate = sum(1 for e in filled if e.hit_stop) / len(filled) * 100
        print(f"  {name:<22s}  MFE={mfe:>+.3f}R  MAE={mae:>+.3f}R  stop_rate={stop_rate:.0f}%")

    # ── Adverse selection check ───────────────────────────────────────────────
    print(f"\n  ADVERSE SELECTION CHECK (limit/retest methods):")
    print(f"  If unfilled trade P&L >> filled P&L → best moves never retested the level")
    for name in ("M4 limit_at_level", "M5 retest_close", "M6 touch_confirm"):
        key = name.strip()
        if key in all_results:
            adverse_selection_row(name, all_results[key])

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  VERDICT for {label}:")
    # Find best filled expectancy
    best_exp = ("M2_conf_close", conf_base)
    for name, sim_fn in simulators:
        filled = [e for e in all_results.get(name.strip(), []) if e.filled]
        if len(filled) >= MIN_TRADES:
            exp = float(np.mean([e.pnl_r for e in filled]))
            if exp > best_exp[1]:
                best_exp = (name.strip(), exp)
    print(f"  Best method: {best_exp[0]}  exp={best_exp[1]:+.3f}R")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase1_entry_methods.py ./data/cache/*.csv")
        sys.exit(1)

    all_setups: List[SweepSetup] = []
    for path in sys.argv[1:]:
        sym = Path(path).stem.split("_")[0]
        try:
            df = load_csv(path)
            ss = find_setups(df)
            all_setups.extend(ss)
            print(f"  {sym:6s}: {len(ss):4d} setups")
        except Exception as ex:
            print(f"  ERROR {path}: {ex}")

    if not all_setups:
        print("No setups found.")
        sys.exit(1)

    W = 115
    print("\n" + "=" * W)
    print("  EVENT TYPE 2: SWEEP + REJECTION — PHASE 1 ENTRY METHOD OPTIMIZATION")
    print(f"  Question: What is the best way to enter a confirmed PDH/PDL sweep?")
    print(f"  Stop for all methods = sweep bar extreme (wick tip)")
    print(f"  Exit for all methods = EOD or stop (Phase 2 will optimize exits)")
    print("=" * W)

    # ── Run on all events (baseline) ──────────────────────────────────────────
    run_and_print(all_setups, "ALL EVENTS  (no filter, Phase 0 baseline)")

    # ── Phase 0.5 filter A: nvda + avoid_13_14 + prior_up ────────────────────
    fa_setups = [s for s in all_setups if FILTER_A(s)]
    run_and_print(fa_setups, "FILTER A: nvda + avoid_13_14 + prior_up  (Phase 0.5 best, n≈111)")

    # ── Phase 0.5 filter B: pdl_long + prior_up ───────────────────────────────
    fb_setups = [s for s in all_setups if FILTER_B(s)]
    run_and_print(fb_setups, "FILTER B: pdl_long + prior_up  (n≈176, most robust)")

    # ── Combined final recommendation ─────────────────────────────────────────
    print(f"\n  {'═' * W}")
    print(f"  OVERALL ENTRY RECOMMENDATION")
    print(f"  {'═' * W}")
    print(f"  Context from Phase 0.5:")
    print(f"    LONG sweeps work better below VWAP (+0.279R) — direction matters")
    print(f"    Prior day UP strongly amplifies PDL sweeps (+0.467R)")
    print(f"    Confirmation bar 2 (+30 min) outperforms bar 1")
    print()
    print(f"  Key question answered by this phase:")
    print(f"    Does the sweep's mean-reversion nature make RETEST entries better than ORB?")
    print(f"    (ORB: limit entries had adverse selection — best moves never pulled back)")
    print(f"    (Sweep: best reversals may naturally return to the level before continuing)")
    print()
    print(f"  → Recommended entry for Phase 2 exit optimization: see VERDICT rows above")
    print(f"  → Use Filter B (pdl_long + prior_up, n=176) for broader validation")
    print(f"  {'═' * W}\n")


if __name__ == "__main__":
    main()
