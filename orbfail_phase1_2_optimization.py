"""orbfail_phase1_2_optimization.py — Event Type 3: ORB Failure

Combined Phase 1 (Entry) + Phase 2 (Exit) Optimization

Phase 0.5 locked filters:
  Filter A (best Sharpe): gap_up + fail_11h       → n=72,  exp=+0.223R, Sharpe=+0.244
  Filter B (most robust): gap_up + not_goog        → n=123, exp=+0.218R, Sharpe=+0.207
  Baseline (gap_up only): gap_up                   → n=153, exp=+0.181R, Sharpe=+0.177

Combining Phase 1+2 for ORB Failure because:
  - Sample sizes are smaller than sweep (n=72-153 vs n=176 for sweep primary)
  - Running separate scripts would produce thin windows
  - The best entry+exit combination is selected together for Phase 3+4

Entry methods tested (same as sweep Phase 1):
  M1. Failure bar close           — baseline (confirmed by failure bar)
  M2. Next bar open               — delayed entry
  M3. Limit at ORB boundary      — wait for price to retest ORB level after failure
                                    Stop = extreme of failed move, risk = overshoot
  M4. Midpoint retest close       — wait for close at ORB midpoint
  M5. Tight stop at midpoint      — entry at failure close but stop at ORB midpoint
                                    Smaller risk, tests if tighter stop improves Sharpe

Exit methods tested:
  X1. EOD or stop                 — baseline
  X2. VWAP exit                   — exit when price crosses VWAP (natural mean-reversion target)
  X3. Fixed 1R target
  X4. Fixed 2R target
  X5. ATR trail 1x                — activated at +0.5R
  X6. Partial 50% at 1R + trail
  X7. Time exit at 12pm / 14pm
  X8. Opposite ORB boundary target — target = ORB low for SHORT / ORB high for LONG

Tested on:
  - All events (no filter, baseline)
  - Filter A (gap_up + fail_11h, n=72)
  - Filter B (gap_up + not_goog, n=123)

Usage:
  python3 orbfail_phase1_2_optimization.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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
from typing import List, Tuple, Callable
from datetime import time as dtime

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
ORB_BARS          = 2
MAX_BO_BAR        = 8
FAIL_WINDOW       = 6
RETEST_WINDOW     = 5
ATR_PERIOD        = 14
TRAIL_ACTIVATE_R  = 0.5
MIN_TRADES        = 10

# Phase 0.5 locked filters
FILTER_A = lambda e: e.gap_pos and e.fail_hour == 11
FILTER_B = lambda e: e.gap_pos and e.symbol != "GOOG"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBFailEvent:
    symbol:           str
    date:             str
    direction:        str    # "SHORT" or "LONG" (what we trade)

    orb_high:         float
    orb_low:          float
    orb_mid:          float
    orb_range:        float

    breakout_bar:     int
    breakout_extreme: float
    breakout_overshoot: float

    fail_bar:         int
    fail_hour:        int
    fail_bar_close:   float  # M1 entry
    vwap_at_fail:     float

    stop_from_extreme: float  # stop = breakout extreme
    risk_from_extreme: float  # abs(entry - extreme)

    gap_pct:          float
    gap_pos:          bool

    pnl_r:            float   # placeholder
    day_df:           object


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_vwap(day: pd.DataFrame, up_to: int) -> float:
    sub = day.iloc[:up_to + 1]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol = sub["volume"]
    return float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(tp.mean())


def compute_atr(day: pd.DataFrame, up_to: int) -> float:
    bars = day.iloc[:up_to + 1]
    if len(bars) < 2: return float(bars.iloc[-1]["high"] - bars.iloc[-1]["low"])
    trs = [bars.iloc[0]["high"] - bars.iloc[0]["low"]]
    for j in range(1, len(bars)):
        tr = max(bars.iloc[j]["high"] - bars.iloc[j]["low"],
                 abs(bars.iloc[j]["high"] - bars.iloc[j-1]["close"]),
                 abs(bars.iloc[j]["low"]  - bars.iloc[j-1]["close"]))
        trs.append(tr)
    n = min(ATR_PERIOD, len(trs))
    return float(np.mean(trs[-n:]))


def sim_stop_eod(entry: float, stop: float, risk: float, direction: str,
                 day: pd.DataFrame, from_iloc: int) -> float:
    if risk <= 0: return 0.0
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG"  and bar["low"]  <= stop: return -1.0
        if direction == "SHORT" and bar["high"] >= stop: return -1.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


# ── Event Detection ───────────────────────────────────────────────────────────

def find_events(df: pd.DataFrame) -> List[ORBFailEvent]:
    symbol = df["symbol"].iloc[0]
    events: List[ORBFailEvent] = []
    prior_close = None; pd_close = None; pd_open = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))
        if prior_close is None:
            prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
            pd_open = float(day["open"].iloc[0]); continue

        gap_pct = (float(day["open"].iloc[0]) - prior_close) / prior_close if prior_close > 0 else 0.0
        gap_pos = gap_pct > 0.002
        if len(day) < ORB_BARS + 1:
            prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
            pd_open = float(day["open"].iloc[0]); continue

        orb = day.iloc[:ORB_BARS]
        orb_high = float(orb["high"].max()); orb_low = float(orb["low"].min())
        orb_mid  = (orb_high + orb_low) / 2.0; orb_range = orb_high - orb_low
        found = False

        for i in range(ORB_BARS, min(MAX_BO_BAR + 1, len(day))):
            if found: break
            bar = day.iloc[i]
            for bo_dir, bo_check, fail_check, entry_dir in [
                ("LONG_BO",
                 lambda b, h=orb_high: b["close"] > h,
                 lambda b, m=orb_mid: b["close"] < b["open"] and b["close"] < m,
                 "SHORT"),
                ("SHORT_BO",
                 lambda b, l=orb_low: b["close"] < l,
                 lambda b, m=orb_mid: b["close"] > b["open"] and b["close"] > m,
                 "LONG"),
            ]:
                if not bo_check(bar): continue
                running_ext = float(bar["high"]) if bo_dir == "LONG_BO" else float(bar["low"])
                scan_end    = min(i + FAIL_WINDOW + 1, len(day))
                for j_off, (_, fb) in enumerate(day.iloc[i + 1:scan_end].iterrows()):
                    if bo_dir == "LONG_BO": running_ext = max(running_ext, float(fb["high"]))
                    else:                   running_ext = min(running_ext, float(fb["low"]))
                    if not fail_check(fb): continue
                    fail_iloc = i + 1 + j_off
                    entry     = float(fb["close"])
                    risk      = abs(entry - running_ext)
                    overshoot = ((running_ext - orb_high) if bo_dir == "LONG_BO"
                                 else (orb_low - running_ext))
                    vwap_at   = compute_vwap(day, fail_iloc)
                    events.append(ORBFailEvent(
                        symbol=symbol, date=str(day_date), direction=entry_dir,
                        orb_high=orb_high, orb_low=orb_low, orb_mid=orb_mid, orb_range=orb_range,
                        breakout_bar=i, breakout_extreme=running_ext,
                        breakout_overshoot=overshoot,
                        fail_bar=fail_iloc, fail_hour=fb["time"].hour,
                        fail_bar_close=entry, vwap_at_fail=vwap_at,
                        stop_from_extreme=running_ext, risk_from_extreme=risk,
                        gap_pct=gap_pct, gap_pos=gap_pos,
                        pnl_r=0.0, day_df=day,
                    ))
                    found = True; break
                if found: break
        prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
        pd_open = float(day["open"].iloc[0])
    return events


# ── Exit Methods ──────────────────────────────────────────────────────────────

def sim_x1(entry: float, stop: float, risk: float, direction: str,
           day: pd.DataFrame, from_iloc: int) -> float:
    return sim_stop_eod(entry, stop, risk, direction, day, from_iloc)


def sim_x2_vwap(entry: float, stop: float, risk: float, direction: str,
                day: pd.DataFrame, from_iloc: int, cap: float = -1.0) -> float:
    if risk <= 0: return 0.0
    for b_num, (idx, bar) in enumerate(day.iloc[from_iloc:].iterrows()):
        iloc_abs = from_iloc + b_num
        vwap     = compute_vwap(day, iloc_abs)
        if direction == "LONG"  and bar["low"]  <= stop: return -1.0
        if direction == "SHORT" and bar["high"] >= stop: return -1.0
        if direction == "LONG"  and bar["close"] >= vwap:
            return (bar["close"] - entry) / risk
        if direction == "SHORT" and bar["close"] <= vwap:
            return (entry - bar["close"]) / risk
    eod = float(day["close"].iloc[-1])
    raw = (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk
    return max(raw, cap)


def sim_x3_1r(entry: float, stop: float, risk: float, direction: str,
              day: pd.DataFrame, from_iloc: int) -> float:
    if risk <= 0: return 0.0
    target = (entry + risk) if direction == "LONG" else (entry - risk)
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG":
            if bar["low"] <= stop: return -1.0
            if bar["high"] >= target: return 1.0
        else:
            if bar["high"] >= stop: return -1.0
            if bar["low"] <= target: return 1.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


def sim_x4_2r(entry: float, stop: float, risk: float, direction: str,
              day: pd.DataFrame, from_iloc: int) -> float:
    if risk <= 0: return 0.0
    target = (entry + 2 * risk) if direction == "LONG" else (entry - 2 * risk)
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG":
            if bar["low"] <= stop: return -1.0
            if bar["high"] >= target: return 2.0
        else:
            if bar["high"] >= stop: return -1.0
            if bar["low"] <= target: return 2.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


def sim_x5_atr_trail(entry: float, stop: float, risk: float, direction: str,
                     day: pd.DataFrame, from_iloc: int) -> float:
    if risk <= 0: return 0.0
    atr   = compute_atr(day, from_iloc)
    trail = atr * 1.0
    activated  = False
    trail_stop = stop
    best_price = entry
    for b_num, (_, bar) in enumerate(day.iloc[from_iloc:].iterrows()):
        fav = (bar["high"] - entry) / risk if direction == "LONG" else (entry - bar["low"]) / risk
        if not activated and fav >= TRAIL_ACTIVATE_R:
            activated = True
        if activated:
            if direction == "LONG":
                best_price = max(best_price, float(bar["close"]))
                trail_stop = max(trail_stop, best_price - trail)
            else:
                best_price = min(best_price, float(bar["close"]))
                trail_stop = min(trail_stop, best_price + trail)
        eff_stop = trail_stop if activated else stop
        if direction == "LONG"  and bar["low"]  <= eff_stop:
            return (eff_stop - entry) / risk
        if direction == "SHORT" and bar["high"] >= eff_stop:
            return (entry - eff_stop) / risk
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


def sim_x6_partial(entry: float, stop: float, risk: float, direction: str,
                   day: pd.DataFrame, from_iloc: int) -> float:
    """50% at +1R, trail remaining."""
    if risk <= 0: return 0.0
    target1 = (entry + risk) if direction == "LONG" else (entry - risk)
    atr   = compute_atr(day, from_iloc)
    trail = atr * 1.0
    partial_taken = False; partial_pnl = 0.0
    activated = False; trail_stop = stop; best_price = entry
    for _, bar in day.iloc[from_iloc:].iterrows():
        eff_stop = trail_stop if (partial_taken and activated) else stop
        if direction == "LONG"  and bar["low"]  <= eff_stop:
            sp = (eff_stop - entry) / risk
            return 0.5 * partial_pnl + 0.5 * sp if partial_taken else sp
        if direction == "SHORT" and bar["high"] >= eff_stop:
            sp = (entry - eff_stop) / risk
            return 0.5 * partial_pnl + 0.5 * sp if partial_taken else sp
        if not partial_taken:
            hit = (bar["high"] >= target1) if direction == "LONG" else (bar["low"] <= target1)
            if hit:
                partial_taken = True; partial_pnl = 1.0; activated = True
                best_price = target1
        if activated:
            if direction == "LONG":
                best_price = max(best_price, float(bar["close"]))
                trail_stop = max(trail_stop, best_price - trail)
            else:
                best_price = min(best_price, float(bar["close"]))
                trail_stop = min(trail_stop, best_price + trail)
    eod = float(day["close"].iloc[-1])
    ep  = (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk
    return 0.5 * partial_pnl + 0.5 * ep if partial_taken else ep


def sim_x7_orb_target(entry: float, stop: float, risk: float, direction: str,
                      day: pd.DataFrame, from_iloc: int,
                      orb_low: float, orb_high: float) -> float:
    """Target = ORB opposite boundary (SHORT targets ORB low, LONG targets ORB high)."""
    if risk <= 0: return 0.0
    target = orb_low if direction == "SHORT" else orb_high
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG":
            if bar["low"]  <= stop:  return -1.0
            if bar["high"] >= target: return (target - entry) / risk
        else:
            if bar["high"] >= stop:  return -1.0
            if bar["low"]  <= target: return (entry - target) / risk
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


def sim_x8_time(entry: float, stop: float, risk: float, direction: str,
                day: pd.DataFrame, from_iloc: int, cutoff_h: int) -> float:
    if risk <= 0: return 0.0
    for b_num, (_, bar) in enumerate(day.iloc[from_iloc:].iterrows()):
        iloc_abs = from_iloc + b_num
        if direction == "LONG"  and bar["low"]  <= stop: return -1.0
        if direction == "SHORT" and bar["high"] >= stop: return -1.0
        btime = day.iloc[iloc_abs]["time"] if "time" in day.columns else dtime(16, 0)
        if btime.hour >= cutoff_h:
            return (bar["close"] - entry) / risk if direction == "LONG" else (entry - bar["close"]) / risk
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


# ── Combined Entry × Exit Simulation ─────────────────────────────────────────

def run_entry_exit(events: List[ORBFailEvent], label: str) -> None:
    if len(events) < MIN_TRADES:
        print(f"\n  {label}: only {len(events)} events, skipping")
        return

    print(f"\n  {'═'*105}")
    print(f"  {label}  (n={len(events)})")
    print(f"  {'═'*105}")
    print(f"\n  {'Entry+Exit':<32s}  {'n':>4}  {'Exp':>8}  {'WR':>4}  {'Sharpe':>7}  {'Stop%':>6}  note")
    print(f"  {'─'*90}")

    results: dict = {}
    best_sh = -999.0; best_key = ""; best_exp = 0.0

    for entry_lbl, get_entry, get_from, get_risk, get_stop in [
        # M1: failure bar close, stop = extreme
        ("M1_fail_close ",
         lambda e: e.fail_bar_close,
         lambda e: e.fail_bar + 1,
         lambda e: e.risk_from_extreme,
         lambda e: e.stop_from_extreme),
        # M2: next bar open, stop = extreme
        ("M2_next_open   ",
         lambda e: float(e.day_df.iloc[e.fail_bar + 1]["open"]) if e.fail_bar + 1 < len(e.day_df) else None,
         lambda e: e.fail_bar + 2,
         lambda e: abs((float(e.day_df.iloc[e.fail_bar + 1]["open"]) if e.fail_bar + 1 < len(e.day_df) else 0)
                       - e.stop_from_extreme),
         lambda e: e.stop_from_extreme),
        # M3: limit at ORB boundary, stop = extreme, tight risk = overshoot
        # (fill = price touches ORB level within RETEST_WINDOW bars)
        ("M3_limit_ORB_lvl",
         None,   # special handling below
         None, None, None),
    ]:
        if entry_lbl.startswith("M3"):
            # Limit at ORB level (SHORT at ORB high, LONG at ORB low)
            filled_results = []
            for e in events:
                limit_price = e.orb_high if e.direction == "SHORT" else e.orb_low
                overshoot   = e.breakout_overshoot
                risk        = overshoot if overshoot > 0 else abs(e.fail_bar_close - limit_price)
                if risk <= 0: continue
                scan_start  = e.fail_bar + 1
                scan_end    = min(scan_start + RETEST_WINDOW, len(e.day_df))
                filled      = False
                for k in range(scan_start, scan_end):
                    bar = e.day_df.iloc[k]
                    if e.direction == "SHORT" and bar["high"] >= limit_price:
                        entry   = limit_price
                        stop    = e.breakout_extreme
                        r       = abs(entry - stop)
                        if r <= 0: break
                        remaining = e.day_df.iloc[k + 1:]
                        eod       = float(e.day_df["close"].iloc[-1])
                        pnl       = sim_stop_eod(entry, stop, r, e.direction, e.day_df, k + 1)
                        filled_results.append(pnl); filled = True; break
                    elif e.direction == "LONG" and bar["low"] <= limit_price:
                        entry   = limit_price
                        stop    = e.breakout_extreme
                        r       = abs(entry - stop)
                        if r <= 0: break
                        pnl     = sim_stop_eod(entry, stop, r, e.direction, e.day_df, k + 1)
                        filled_results.append(pnl); filled = True; break

            if len(filled_results) >= MIN_TRADES:
                arr  = np.array(filled_results)
                exp  = float(arr.mean())
                wr   = float((arr > 0).mean())
                sh   = float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0
                sr   = float((arr <= -0.99).mean())
                fr   = len(filled_results) / len(events)
                flag = "★★" if exp > 0.20 and sh > 0.20 else ("★" if exp > 0.10 else "")
                print(f"  {'M3_limit_ORB_lvl (eod)':<32s}  n={len(filled_results):4d} ({fr:.0%})  "
                      f"exp={exp:+.3f}R  WR={wr:.0%}  Sharpe={sh:+.3f}  stop={sr:.0%}  {flag}")
                if sh > best_sh: best_sh = sh; best_key = "M3_EOD"; best_exp = exp
            continue

        for exit_lbl, exit_fn_factory in [
            ("X1_EOD",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x1(ent, stp, rsk, dir, day, fr)),
            ("X2_VWAP",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x2_vwap(ent, stp, rsk, dir, day, fr)),
            ("X3_1R",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x3_1r(ent, stp, rsk, dir, day, fr)),
            ("X4_2R",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x4_2r(ent, stp, rsk, dir, day, fr)),
            ("X5_ATR_trail",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x5_atr_trail(ent, stp, rsk, dir, day, fr)),
            ("X6_partial",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x6_partial(ent, stp, rsk, dir, day, fr)),
            ("X7_ORB_target",
             lambda ent, stp, rsk, dir, day, fr, ev: sim_x7_orb_target(
                 ent, stp, rsk, dir, day, fr, ev.orb_low, ev.orb_high)),
            ("X8_12pm",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x8_time(ent, stp, rsk, dir, day, fr, 12)),
            ("X8_14pm",
             lambda ent, stp, rsk, dir, day, fr, e=None: sim_x8_time(ent, stp, rsk, dir, day, fr, 14)),
        ]:
            pnls = []
            for ev in events:
                en = get_entry(ev)
                if en is None: continue
                fr  = get_from(ev)
                if fr >= len(ev.day_df): continue
                rsk = get_risk(ev)
                stp = get_stop(ev)
                if rsk <= 0: continue
                pnls.append(exit_fn_factory(en, stp, rsk, ev.direction, ev.day_df, fr, ev))

            if len(pnls) < MIN_TRADES: continue
            arr  = np.array(pnls)
            exp  = float(arr.mean())
            wr   = float((arr > 0).mean())
            sh   = float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0
            sr   = float((arr <= -0.99).mean())
            flag = "★★" if exp > 0.20 and sh > 0.20 else ("★" if exp > 0.10 else "")
            key  = f"{entry_lbl.strip()}+{exit_lbl}"
            print(f"  {key:<32s}  n={len(pnls):4d}  exp={exp:+.3f}R  WR={wr:.0%}  "
                  f"Sharpe={sh:+.3f}  stop={sr:.0%}  {flag}")
            results[key] = (exp, sh, len(pnls))
            if sh > best_sh: best_sh = sh; best_key = key; best_exp = exp

    print(f"\n  VERDICT: Best entry+exit → {best_key}  exp={best_exp:+.3f}R  Sharpe={best_sh:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orbfail_phase1_2_optimization.py <csv_files...>")
        sys.exit(1)

    all_events: List[ORBFailEvent] = []
    for path in sys.argv[1:]:
        df  = load_csv(path)
        sym = Path(path).stem.split("_")[0]
        evs = find_events(df)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    print()
    print("=" * 110)
    print("  EVENT TYPE 3: ORB FAILURE — PHASE 1+2 ENTRY × EXIT OPTIMIZATION")
    print("  Entry methods: M1 (fail close), M2 (next open), M3 (limit at ORB level)")
    print("  Exit methods: X1 EOD, X2 VWAP, X3 1R, X4 2R, X5 ATR trail, X6 partial, X7 ORB target, X8 time")
    print("=" * 110)

    # All events (baseline)
    run_entry_exit(all_events, "ALL EVENTS — no filter (Phase 0 baseline)")

    # Filter A
    a_events = [e for e in all_events if FILTER_A(e)]
    run_entry_exit(a_events, "FILTER A — gap_up + fail_11h")

    # Filter B
    b_events = [e for e in all_events if FILTER_B(e)]
    run_entry_exit(b_events, "FILTER B — gap_up + not_goog (primary, n≈123)")

    print()
    print("=" * 110)
    print("  RECOMMENDATION: Lock best entry+exit for Phase 3 regime filter testing")
    print("  Primary → Filter B (n≈123): choose highest Sharpe combination")
    print("  Phase 3 will test day-level regime conditions on top of locked entry+exit")
    print("=" * 110)


if __name__ == "__main__":
    main()
