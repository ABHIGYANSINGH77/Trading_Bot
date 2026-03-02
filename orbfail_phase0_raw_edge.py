"""orbfail_phase0_raw_edge.py — Event Type 3: ORB Failure (Fade the Failed Breakout)

Phase 0: Raw Edge Test

Hypothesis:
  When an ORB (Opening Range Breakout) occurs and then FAILS — price re-enters the ORB
  and crosses the midpoint in the opposite direction — the trapped breakout traders' stops
  provide fuel for a mean-reversion move back towards the opposite ORB boundary.

  "Failed breakout = trapped traders = forced covering = fuel for reversal"

Setup logic:
  1. ORB defined as the high/low of the first 30 minutes (bars 1-2 at 15-min resolution)
  2. ORB breakout: price CLOSES outside the ORB (close > ORB high for LONG, close < ORB low for SHORT)
     → must happen within the first 8 bars of the session (2 hours)
  3. ORB failure: within FAIL_WINDOW bars after the breakout, price:
     - Closes BACK inside the ORB (crosses the breakout level)
     - AND closes past the ORB MIDPOINT in the opposite direction
  4. Entry: close of the failure bar
  5. Stop: extreme of the failed breakout move (highest high for failed LONG, lowest low for failed SHORT)
  6. Target / exit: EOD or stop (Phase 2 will optimize exits)

Why this is different from regular ORB:
  - Regular ORB: trade WITH the breakout direction
  - ORB Failure: trade AGAINST the failed breakout, fading the trapped crowd
  - Expected frequency: lower than sweeps (~1-2 events/week vs 4-5/week for sweeps)

Risk notes:
  - If sample < 30 events: flag as insufficient, do not proceed past Phase 0
  - Min sample for meaningful statistics: 50 events

Analysis dimensions:
  - Direction: LONG failure (SHORT fade) vs SHORT failure (LONG fade)
  - Symbol: per-stock breakdown
  - Time: how quickly the failure happens (breakout bar #, failure bar #)
  - Gap: gapped days vs flat opens
  - Prior day direction: prior_up vs prior_down
  - ORB range: large vs small ORBs that fail
  - Overshoot: how far beyond the ORB did price go before failing?

Usage:
  python3 orbfail_phase0_raw_edge.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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
ORB_BARS          = 2       # number of 15-min bars forming the ORB (30 min)
MAX_BO_BAR        = 8       # latest bar # at which breakout can occur (2h window)
FAIL_WINDOW       = 6       # bars after breakout to look for failure
MIN_EVENTS        = 20      # minimum to report a cut; drop if total < this

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBFailEvent:
    symbol:           str
    date:             str

    direction:        str    # "SHORT" = fade failed LONG breakout, "LONG" = fade failed SHORT

    orb_high:         float
    orb_low:          float
    orb_mid:          float
    orb_range:        float

    breakout_bar:     int    # bar number of breakout (0-indexed)
    breakout_price:   float  # close of breakout bar
    breakout_extreme: float  # highest high (LONG BO) or lowest low (SHORT BO)
    breakout_overshoot: float  # how far beyond ORB boundary price went

    fail_bar:         int    # bar number of failure bar
    fail_hour:        int

    entry_price:      float  # close of failure bar
    stop_price:       float  # extreme of failed move
    risk:             float

    gap_pct:          float
    prior_day_up:     bool
    prior_close:      float

    mfe:              float  # max favourable R
    mae:              float  # max adverse R
    pnl_r:            float
    exit_reason:      str
    hit_stop:         bool

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


# ── Trade Simulation ──────────────────────────────────────────────────────────

def sim_exit(entry: float, stop: float, risk: float, direction: str,
             remaining: pd.DataFrame, eod_price: float) -> Tuple[float, float, float, str, bool]:
    mf = ma = 0.0
    for _, bar in remaining.iterrows():
        fav = ((bar["high"] - entry) / risk if direction == "LONG"
               else (entry - bar["low"]) / risk)
        adv = ((entry - bar["low"]) / risk if direction == "LONG"
               else (bar["high"] - entry) / risk)
        mf = max(mf, fav); ma = min(ma, -adv)
        if direction == "LONG"  and bar["low"]  <= stop:
            return -1.0, mf, ma, "stop", True
        if direction == "SHORT" and bar["high"] >= stop:
            return -1.0, mf, ma, "stop", True
    pnl = ((eod_price - entry) / risk if direction == "LONG"
           else (entry - eod_price) / risk)
    return pnl, mf, ma, "eod", False


# ── Event Detection ───────────────────────────────────────────────────────────

def find_events(df: pd.DataFrame) -> List[ORBFailEvent]:
    symbol  = df["symbol"].iloc[0]
    events: List[ORBFailEvent] = []
    prior_close = None
    pd_close    = None
    pd_open     = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if prior_close is None:
            prior_close = float(day["close"].iloc[-1])
            pd_close    = float(day["close"].iloc[-1])
            pd_open     = float(day["open"].iloc[0])
            continue

        prior_day_up = pd_close > pd_open
        gap_pct      = (float(day["open"].iloc[0]) - prior_close) / prior_close if prior_close > 0 else 0.0

        # Define ORB
        if len(day) < ORB_BARS + 1:
            prior_close = float(day["close"].iloc[-1])
            pd_close    = prior_close
            pd_open     = float(day["open"].iloc[0])
            continue

        orb      = day.iloc[:ORB_BARS]
        orb_high = float(orb["high"].max())
        orb_low  = float(orb["low"].min())
        orb_mid  = (orb_high + orb_low) / 2.0
        orb_range = orb_high - orb_low
        eod      = float(day["close"].iloc[-1])

        # Scan for breakout starting at bar ORB_BARS
        found = False
        for i in range(ORB_BARS, min(MAX_BO_BAR + 1, len(day))):
            if found:
                break
            bar = day.iloc[i]

            for bo_dir, bo_check, fail_check, stop_fn, entry_dir in [
                # LONG breakout → fade when it fails
                ("LONG_BO",
                 lambda b, h=orb_high: b["close"] > h,
                 lambda b, m=orb_mid: b["close"] < b["open"] and b["close"] < m,
                 lambda bars: float(bars["high"].max()),
                 "SHORT"),   # we go SHORT to fade the failed LONG breakout
                # SHORT breakout → fade when it fails
                ("SHORT_BO",
                 lambda b, l=orb_low: b["close"] < l,
                 lambda b, m=orb_mid: b["close"] > b["open"] and b["close"] > m,
                 lambda bars: float(bars["low"].min()),
                 "LONG"),    # we go LONG to fade the failed SHORT breakout
            ]:
                if not bo_check(bar):
                    continue

                # Breakout confirmed at bar i
                bo_close = float(bar["close"])
                bo_level = orb_high if bo_dir == "LONG_BO" else orb_low

                # Track extreme from breakout bar onwards (for stop)
                # Scan FAIL_WINDOW bars for failure
                scan_end = min(i + FAIL_WINDOW + 1, len(day))
                scan     = day.iloc[i + 1:scan_end]

                # Running extreme (to set stop = extreme of failed move)
                running_ext = float(bar["high"]) if bo_dir == "LONG_BO" else float(bar["low"])

                for j_off, (_, fail_bar) in enumerate(scan.iterrows()):
                    if bo_dir == "LONG_BO":
                        running_ext = max(running_ext, float(fail_bar["high"]))
                    else:
                        running_ext = min(running_ext, float(fail_bar["low"]))

                    if not fail_check(fail_bar):
                        continue

                    # Failure confirmed
                    fail_iloc   = i + 1 + j_off
                    entry       = float(fail_bar["close"])
                    stop        = running_ext
                    risk        = abs(entry - stop)
                    if risk <= 0:
                        continue

                    # Overshoot = how far beyond ORB boundary the breakout went
                    if bo_dir == "LONG_BO":
                        overshoot = running_ext - orb_high
                    else:
                        overshoot = orb_low - running_ext

                    remaining = day.iloc[fail_iloc + 1:]
                    pnl, mf, ma, reason, hs = sim_exit(
                        entry, stop, risk, entry_dir, remaining, eod)

                    events.append(ORBFailEvent(
                        symbol=symbol, date=str(day_date),
                        direction=entry_dir,
                        orb_high=orb_high, orb_low=orb_low, orb_mid=orb_mid,
                        orb_range=orb_range,
                        breakout_bar=i, breakout_price=bo_close,
                        breakout_extreme=running_ext,
                        breakout_overshoot=overshoot,
                        fail_bar=fail_iloc,
                        fail_hour=fail_bar["time"].hour,
                        entry_price=entry, stop_price=stop, risk=risk,
                        gap_pct=gap_pct, prior_day_up=prior_day_up,
                        prior_close=float(prior_close),
                        mfe=mf, mae=ma, pnl_r=pnl,
                        exit_reason=reason, hit_stop=hs,
                        day_df=day,
                    ))
                    found = True
                    break
                if found:
                    break

        prior_close = float(day["close"].iloc[-1])
        pd_close    = prior_close
        pd_open     = float(day["open"].iloc[0])

    return events


# ── Analysis ──────────────────────────────────────────────────────────────────

def stats_row(label: str, events: List[ORBFailEvent],
              indent: str = "  ") -> None:
    if len(events) < MIN_EVENTS:
        print(f"{indent}{label:<40s}  n={len(events):4d}  [too few]")
        return
    pnls  = [e.pnl_r for e in events]
    exp   = float(np.mean(pnls))
    wr    = float(np.mean([p > 0 for p in pnls]))
    sh    = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 1e-9 else 0.0
    sr    = float(np.mean([e.hit_stop for e in events]))
    mfe   = float(np.mean([e.mfe for e in events]))
    flag  = "★★" if exp > 0.15 and sh > 0.1 else ("★" if exp > 0.05 else "")
    print(f"{indent}{label:<40s}  n={len(events):4d}  exp={exp:+.3f}R  "
          f"WR={wr:4.0%}  Sharpe={sh:+.3f}  stop={sr:.0%}  MFE={mfe:+.3f}R  {flag}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orbfail_phase0_raw_edge.py <csv_files...>")
        sys.exit(1)

    all_events: List[ORBFailEvent] = []
    for path in sys.argv[1:]:
        df   = load_csv(path)
        sym  = Path(path).stem.split("_")[0]
        evs  = find_events(df)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    n_total = len(all_events)
    print()
    print("=" * 100)
    print("  EVENT TYPE 3: ORB FAILURE — PHASE 0 RAW EDGE TEST")
    print(f"  Total events detected: {n_total}")
    print("=" * 100)

    if n_total < MIN_EVENTS:
        print(f"\n  INSUFFICIENT DATA: only {n_total} events detected.")
        print(f"  Need at least {MIN_EVENTS} to proceed. DROP this event type.")
        return

    # Overall baseline
    print(f"\n  ── OVERALL BASELINE ──────────────────────────────────────────────")
    stats_row("All events", all_events)

    # Direction
    print(f"\n  ── D1: DIRECTION (fade which breakout?) ──────────────────────────")
    stats_row("SHORT (fade failed LONG BO)", [e for e in all_events if e.direction == "SHORT"])
    stats_row("LONG  (fade failed SHORT BO)", [e for e in all_events if e.direction == "LONG"])

    # Symbol
    print(f"\n  ── D2: SYMBOL ────────────────────────────────────────────────────")
    for sym in sorted(set(e.symbol for e in all_events)):
        stats_row(sym, [e for e in all_events if e.symbol == sym])

    # Breakout speed (how early the breakout occurred)
    print(f"\n  ── D3: BREAKOUT TIMING ───────────────────────────────────────────")
    stats_row("Early BO (bar 2-3, first 30min)", [e for e in all_events if e.breakout_bar <= 3])
    stats_row("Mid BO (bar 4-6, 30-90min)",      [e for e in all_events if 4 <= e.breakout_bar <= 6])
    stats_row("Late BO (bar 7-8, 90-120min)",    [e for e in all_events if e.breakout_bar >= 7])

    # Failure timing
    print(f"\n  ── D4: FAILURE HOUR ──────────────────────────────────────────────")
    for h in sorted(set(e.fail_hour for e in all_events)):
        stats_row(f"Failure at {h:02d}h", [e for e in all_events if e.fail_hour == h])

    # Prior day direction
    print(f"\n  ── D5: PRIOR DAY DIRECTION ───────────────────────────────────────")
    stats_row("Prior day UP",   [e for e in all_events if e.prior_day_up])
    stats_row("Prior day DOWN", [e for e in all_events if not e.prior_day_up])

    # Gap
    print(f"\n  ── D6: GAP ───────────────────────────────────────────────────────")
    stats_row("Gap UP  (>+0.2%)", [e for e in all_events if e.gap_pct >  0.002])
    stats_row("Gap DOWN (<-0.2%)",[e for e in all_events if e.gap_pct < -0.002])
    stats_row("No gap  (flat)",   [e for e in all_events if abs(e.gap_pct) <= 0.002])

    # Overshoot size (how far the BO went before failing)
    med_os = float(np.median([e.breakout_overshoot for e in all_events]))
    print(f"\n  ── D7: BREAKOUT OVERSHOOT (median={med_os:.4f}) ─────────────────────")
    stats_row("Small overshoot (<median)",  [e for e in all_events if e.breakout_overshoot <  med_os])
    stats_row("Large overshoot (>=median)", [e for e in all_events if e.breakout_overshoot >= med_os])

    # ORB range size
    med_orb = float(np.median([e.orb_range for e in all_events]))
    print(f"\n  ── D8: ORB RANGE SIZE (median={med_orb:.4f}) ─────────────────────────")
    stats_row("Small ORB (<median)", [e for e in all_events if e.orb_range <  med_orb])
    stats_row("Large ORB (>=median)",[e for e in all_events if e.orb_range >= med_orb])

    # MFE distribution insight
    all_pnls = [e.pnl_r for e in all_events]
    eod_surv = [e for e in all_events if not e.hit_stop]
    stop_hit = [e for e in all_events if e.hit_stop]
    print(f"\n  ── EXIT BREAKDOWN ────────────────────────────────────────────────")
    print(f"    Reached EOD: n={len(eod_surv)}  avg P&L={np.mean([e.pnl_r for e in eod_surv]):+.3f}R")
    print(f"    Hit stop:    n={len(stop_hit)}  avg P&L={np.mean([e.pnl_r for e in stop_hit]):+.3f}R  (expected -1.0R)")
    print(f"    Avg MFE all: {np.mean([e.mfe for e in all_events]):+.3f}R")
    print(f"    Avg MAE all: {np.mean([e.mae for e in all_events]):+.3f}R")

    # Verdict
    overall_pnls = [e.pnl_r for e in all_events]
    overall_exp  = float(np.mean(overall_pnls))
    overall_sh   = (float(np.mean(overall_pnls) / np.std(overall_pnls))
                    if np.std(overall_pnls) > 1e-9 else 0.0)
    print(f"\n  ── VERDICT ───────────────────────────────────────────────────────")
    if overall_exp > 0.05 and overall_sh > 0.05:
        print(f"    PROCEED TO PHASE 0.5  (exp={overall_exp:+.3f}R, Sharpe={overall_sh:+.3f})")
        print(f"    Edge detected in ORB failure. Decompose dimensions in Phase 0.5.")
    elif overall_exp > 0:
        print(f"    WEAK EDGE (exp={overall_exp:+.3f}R, Sharpe={overall_sh:+.3f})")
        print(f"    Some positive signal but marginal. Check if any dimension shows strong edge.")
    else:
        print(f"    NO EDGE DETECTED (exp={overall_exp:+.3f}R, Sharpe={overall_sh:+.3f})")
        print(f"    DROP this event type. Do not proceed past Phase 0.")

    print(f"\n  Event count threshold for next phase: minimum 50 events total required.")
    print(f"  Current: {n_total} events {'✓' if n_total >= 50 else '✗'}")


if __name__ == "__main__":
    main()
