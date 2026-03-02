"""orbfail_phase0_5_decomposition.py — Event Type 3: ORB Failure

Phase 0.5: Edge Decomposition

Phase 0 findings (locked):
  - Overall: +0.024R (weak), n=330
  - Gap UP (>+0.2%) → +0.181R, Sharpe +0.177, n=153 ★★
  - Late BO (bar 7-8, 90-120min) → +0.153R, Sharpe +0.201, n=48 ★★
  - Gap DOWN: -0.170R — must EXCLUDE
  - GOOG: -0.127R — likely to exclude
  - Best dimension cross: Gap UP + Late BO = key hypothesis

Questions for Phase 0.5:
  1. Does Gap UP + Late BO combine multiplicatively? (expected: strong edge in intersection)
  2. Which direction fades better on gap-up days? (SHORT fades failed LONG BO, or LONG fades SHORT?)
  3. Does prior day direction interact with gap direction?
  4. Does VWAP position at breakout time predict which breakouts will fail?
  5. Does the size of the overshoot predict failure quality?
  6. Are there time-of-failure windows that matter (fail at 11h vs 12h)?
  7. Symbol-specific: does NVDA + gap_up dominate?

Design rule: ALL filters must be observable BEFORE or AT the time of the failure bar close.
  - Gap is known at market open ✓
  - Breakout bar number is known ✓
  - VWAP at failure time is known ✓
  - Prior day direction is known ✓

Usage:
  python3 orbfail_phase0_5_decomposition.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
                                             ./data/cache/NVDA_2024-01-01_2025-12-31_15_mins.csv \\
                                             ./data/cache/MSFT_2024-01-01_2025-12-31_15_mins.csv \\
                                             ./data/cache/AMZN_2024-01-01_2025-12-31_15_mins.csv \\
                                             ./data/cache/GOOG_2024-01-01_2025-12-31_15_mins.csv
"""

import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple
from datetime import time as dtime

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
ORB_BARS          = 2
MAX_BO_BAR        = 8
FAIL_WINDOW       = 6
MIN_TRADES        = 20

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBFailEvent:
    symbol:           str
    date:             str
    direction:        str     # "SHORT" = fade failed LONG BO; "LONG" = fade failed SHORT BO
    bo_direction:     str     # "LONG_BO" or "SHORT_BO"

    orb_high:         float
    orb_low:          float
    orb_mid:          float
    orb_range:        float

    breakout_bar:     int
    breakout_price:   float
    breakout_extreme: float
    breakout_overshoot: float
    overshoot_pct:    float   # overshoot / orb_range

    fail_bar:         int
    fail_hour:        int
    fail_bar_close:   float
    vwap_at_fail:     float   # VWAP up to failure bar
    price_vs_vwap:    str     # "above" or "below"

    entry_price:      float
    stop_price:       float
    risk:             float

    gap_pct:          float
    gap_pos:          bool
    gap_neg:          bool
    gap_abs:          bool
    prior_day_up:     bool

    late_bo:          bool    # breakout bar >= 7
    early_fail:       bool    # fail within 1-2 bars of breakout
    fail_at_11:       bool
    fail_at_12:       bool

    nvda:             bool
    not_goog:         bool

    mfe:              float
    mae:              float
    pnl_r:            float
    hit_stop:         bool


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


# ── VWAP ──────────────────────────────────────────────────────────────────────

def compute_vwap(day: pd.DataFrame, up_to_iloc: int) -> float:
    sub = day.iloc[:up_to_iloc + 1]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol = sub["volume"]
    return float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(tp.mean())


# ── Trade Simulation ──────────────────────────────────────────────────────────

def sim_exit(entry: float, stop: float, risk: float, direction: str,
             remaining: pd.DataFrame, eod: float) -> Tuple[float, float, float, bool]:
    mf = ma = 0.0
    for _, bar in remaining.iterrows():
        fav = ((bar["high"] - entry) / risk if direction == "LONG"
               else (entry - bar["low"]) / risk)
        adv = ((entry - bar["low"]) / risk if direction == "LONG"
               else (bar["high"] - entry) / risk)
        mf = max(mf, fav); ma = min(ma, -adv)
        if direction == "LONG"  and bar["low"]  <= stop:
            return -1.0, mf, ma, True
        if direction == "SHORT" and bar["high"] >= stop:
            return -1.0, mf, ma, True
    pnl = (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk
    return pnl, mf, ma, False


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
            prior_close = float(day["close"].iloc[-1])
            pd_close    = prior_close; pd_open = float(day["open"].iloc[0])
            continue

        prior_day_up = pd_close > pd_open
        gap_pct      = (float(day["open"].iloc[0]) - prior_close) / prior_close if prior_close > 0 else 0.0
        gap_pos      = gap_pct > 0.002
        gap_neg      = gap_pct < -0.002
        gap_abs      = abs(gap_pct) > 0.003

        if len(day) < ORB_BARS + 1:
            prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
            pd_open = float(day["open"].iloc[0]); continue

        orb       = day.iloc[:ORB_BARS]
        orb_high  = float(orb["high"].max())
        orb_low   = float(orb["low"].min())
        orb_mid   = (orb_high + orb_low) / 2.0
        orb_range = orb_high - orb_low
        eod       = float(day["close"].iloc[-1])
        found     = False

        for i in range(ORB_BARS, min(MAX_BO_BAR + 1, len(day))):
            if found: break
            bar = day.iloc[i]

            for bo_dir, bo_check, fail_check, stop_fn, entry_dir in [
                ("LONG_BO",
                 lambda b, h=orb_high: b["close"] > h,
                 lambda b, m=orb_mid: b["close"] < b["open"] and b["close"] < m,
                 lambda brs: float(brs["high"].max()),
                 "SHORT"),
                ("SHORT_BO",
                 lambda b, l=orb_low: b["close"] < l,
                 lambda b, m=orb_mid: b["close"] > b["open"] and b["close"] > m,
                 lambda brs: float(brs["low"].min()),
                 "LONG"),
            ]:
                if not bo_check(bar): continue

                bo_level  = orb_high if bo_dir == "LONG_BO" else orb_low
                scan_end  = min(i + FAIL_WINDOW + 1, len(day))
                running_ext = float(bar["high"]) if bo_dir == "LONG_BO" else float(bar["low"])

                for j_off, (_, fb) in enumerate(day.iloc[i + 1:scan_end].iterrows()):
                    if bo_dir == "LONG_BO":
                        running_ext = max(running_ext, float(fb["high"]))
                    else:
                        running_ext = min(running_ext, float(fb["low"]))

                    if not fail_check(fb): continue

                    fail_iloc  = i + 1 + j_off
                    entry      = float(fb["close"])
                    stop       = running_ext
                    risk       = abs(entry - stop)
                    if risk <= 0: continue

                    overshoot    = (running_ext - orb_high) if bo_dir == "LONG_BO" else (orb_low - running_ext)
                    overshoot_pct = overshoot / orb_range if orb_range > 0 else 0.0
                    vwap_at_fail = compute_vwap(day, fail_iloc)
                    pvwap        = "above" if float(fb["close"]) > vwap_at_fail else "below"

                    remaining    = day.iloc[fail_iloc + 1:]
                    pnl, mf, ma, hs = sim_exit(entry, stop, risk, entry_dir, remaining, eod)

                    events.append(ORBFailEvent(
                        symbol=symbol, date=str(day_date),
                        direction=entry_dir, bo_direction=bo_dir,
                        orb_high=orb_high, orb_low=orb_low, orb_mid=orb_mid, orb_range=orb_range,
                        breakout_bar=i, breakout_price=float(bar["close"]),
                        breakout_extreme=running_ext, breakout_overshoot=overshoot,
                        overshoot_pct=overshoot_pct,
                        fail_bar=fail_iloc, fail_hour=fb["time"].hour,
                        fail_bar_close=entry, vwap_at_fail=vwap_at_fail, price_vs_vwap=pvwap,
                        entry_price=entry, stop_price=stop, risk=risk,
                        gap_pct=gap_pct, gap_pos=gap_pos, gap_neg=gap_neg, gap_abs=gap_abs,
                        prior_day_up=prior_day_up,
                        late_bo=(i >= 7),
                        early_fail=(j_off <= 1),
                        fail_at_11=(fb["time"].hour == 11),
                        fail_at_12=(fb["time"].hour == 12),
                        nvda=(symbol == "NVDA"),
                        not_goog=(symbol != "GOOG"),
                        mfe=mf, mae=ma, pnl_r=pnl, hit_stop=hs,
                    ))
                    found = True; break
                if found: break

        prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
        pd_open = float(day["open"].iloc[0])

    return events


# ── Filters ───────────────────────────────────────────────────────────────────

def build_filters() -> Dict[str, Callable]:
    return {
        # Gap direction (Phase 0 key finding)
        "gap_up":           lambda e: e.gap_pos,
        "gap_down":         lambda e: e.gap_neg,
        "gap_flat":         lambda e: not e.gap_abs,
        "avoid_gap_down":   lambda e: not e.gap_neg,

        # Breakout timing (Phase 0 key finding)
        "late_bo":          lambda e: e.late_bo,
        "early_bo":         lambda e: not e.late_bo,

        # Failure timing
        "fail_11h":         lambda e: e.fail_at_11,
        "fail_12h":         lambda e: e.fail_at_12,
        "fail_not_10h":     lambda e: e.fail_hour != 10,
        "fail_11_or_12":    lambda e: e.fail_at_11 or e.fail_at_12,

        # Direction
        "fade_long_bo":     lambda e: e.direction == "SHORT",
        "fade_short_bo":    lambda e: e.direction == "LONG",

        # VWAP context at failure
        "fail_above_vwap":  lambda e: e.price_vs_vwap == "above",
        "fail_below_vwap":  lambda e: e.price_vs_vwap == "below",
        # SHORT (fading LONG BO) above VWAP = overextended fade
        "short_above_vwap": lambda e: e.direction == "SHORT" and e.price_vs_vwap == "above",
        "long_below_vwap":  lambda e: e.direction == "LONG" and e.price_vs_vwap == "below",

        # Prior day
        "prior_up":         lambda e: e.prior_day_up,
        "prior_down":       lambda e: not e.prior_day_up,

        # Symbol
        "nvda":             lambda e: e.nvda,
        "not_goog":         lambda e: e.not_goog,
        "aapl_amzn_nvda":   lambda e: e.symbol in ("AAPL", "AMZN", "NVDA"),

        # Overshoot size
        "small_overshoot":  lambda e: e.overshoot_pct < 0.3,
        "large_overshoot":  lambda e: e.overshoot_pct >= 0.3,

        # Early fail (quick reversal within 1-2 bars)
        "early_fail":       lambda e: e.early_fail,
        "late_fail":        lambda e: not e.early_fail,
    }


CONFLICTS = {
    frozenset({"gap_up", "gap_down"}),
    frozenset({"gap_up", "gap_flat"}),
    frozenset({"gap_down", "gap_flat"}),
    frozenset({"gap_up", "avoid_gap_down"}),
    frozenset({"late_bo", "early_bo"}),
    frozenset({"fade_long_bo", "fade_short_bo"}),
    frozenset({"fail_above_vwap", "fail_below_vwap"}),
    frozenset({"short_above_vwap", "long_below_vwap"}),
    frozenset({"prior_up", "prior_down"}),
    frozenset({"early_fail", "late_fail"}),
    frozenset({"small_overshoot", "large_overshoot"}),
}


# ── Stats ─────────────────────────────────────────────────────────────────────

def sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    return float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0


def print_row(label: str, events: List[ORBFailEvent], indent: str = "  ") -> None:
    if len(events) < MIN_TRADES:
        print(f"{indent}{label:<36s}  n={len(events):4d}  [too few]")
        return
    pnls = [e.pnl_r for e in events]
    exp  = float(np.mean(pnls))
    wr   = float(np.mean([p > 0 for p in pnls]))
    sh   = sharpe(pnls)
    sr   = float(np.mean([e.hit_stop for e in events]))
    flag = "★★" if exp > 0.15 and sh > 0.10 else ("★" if exp > 0.05 else "")
    print(f"{indent}{label:<36s}  n={len(events):4d}  exp={exp:+.3f}R  "
          f"WR={wr:.0%}  Sharpe={sh:+.3f}  stop={sr:.0%}  {flag}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orbfail_phase0_5_decomposition.py <csv_files...>")
        sys.exit(1)

    all_events: List[ORBFailEvent] = []
    for path in sys.argv[1:]:
        df   = load_csv(path)
        sym  = Path(path).stem.split("_")[0]
        evs  = find_events(df)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    n = len(all_events)
    print()
    print("=" * 100)
    print("  EVENT TYPE 3: ORB FAILURE — PHASE 0.5 EDGE DECOMPOSITION")
    print(f"  n={n} events  |  Baseline exp={np.mean([e.pnl_r for e in all_events]):+.3f}R  "
          f"Sharpe={sharpe([e.pnl_r for e in all_events]):+.3f}")
    print("=" * 100)

    filters = build_filters()

    # ── Key Cross: Gap UP × Late BO ──────────────────────────────────────────
    print(f"\n  ── KEY HYPOTHESIS: Gap UP × Late BO ─────────────────────────────")
    print_row("All events (baseline)",   all_events)
    print_row("gap_up only",             [e for e in all_events if e.gap_pos])
    print_row("late_bo only",            [e for e in all_events if e.late_bo])
    print_row("gap_up + late_bo",        [e for e in all_events if e.gap_pos and e.late_bo])
    print_row("gap_up + avoid 10h fail", [e for e in all_events if e.gap_pos and e.fail_hour != 10])
    print_row("gap_up + fade_long_bo",   [e for e in all_events if e.gap_pos and e.direction == "SHORT"])

    # ── Singles ──────────────────────────────────────────────────────────────
    print(f"\n  ── ALL SINGLES (ranked by Sharpe) ───────────────────────────────")
    singles = []
    for fname, fn in filters.items():
        sub  = [e for e in all_events if fn(e)]
        if len(sub) < MIN_TRADES: continue
        pnls = [e.pnl_r for e in sub]
        sh   = sharpe(pnls)
        singles.append((sh, fname, float(np.mean(pnls)), len(sub),
                        float(np.mean([p > 0 for p in pnls])),
                        float(np.mean([e.hit_stop for e in sub]))))

    singles.sort(key=lambda x: x[0], reverse=True)
    for sh, fname, exp, n_sub, wr, sr in singles[:15]:
        flag = "★★" if exp > 0.15 and sh > 0.10 else ("★" if exp > 0.05 else "")
        print(f"  {fname:<30s}  n={n_sub:4d}  exp={exp:+.3f}R  WR={wr:.0%}  "
              f"Sharpe={sh:+.3f}  stop={sr:.0%}  {flag}")

    top_names = [s[1] for s in singles[:8]]

    # ── Pair combinations ────────────────────────────────────────────────────
    print(f"\n  ── TOP PAIRS (top-8 singles) ─────────────────────────────────────")
    pairs = []
    for n1, n2 in itertools.combinations(top_names, 2):
        if frozenset({n1, n2}) in CONFLICTS: continue
        fn1  = filters[n1]; fn2 = filters[n2]
        sub  = [e for e in all_events if fn1(e) and fn2(e)]
        if len(sub) < MIN_TRADES: continue
        pnls = [e.pnl_r for e in sub]
        sh   = sharpe(pnls)
        pairs.append((sh, f"{n1} + {n2}", float(np.mean(pnls)), len(sub),
                      float(np.mean([p > 0 for p in pnls])),
                      float(np.mean([e.hit_stop for e in sub]))))

    pairs.sort(key=lambda x: x[0], reverse=True)
    for sh, fname, exp, n_sub, wr, sr in pairs[:12]:
        flag = "★★" if exp > 0.15 and sh > 0.10 else ("★" if exp > 0.05 else "")
        print(f"  {fname:<44s}  n={n_sub:4d}  exp={exp:+.3f}R  WR={wr:.0%}  "
              f"Sharpe={sh:+.3f}  stop={sr:.0%}  {flag}")

    # ── Best n≥50 for Phase 1 ────────────────────────────────────────────────
    print(f"\n  ── BEST FILTERS WITH n≥50 (recommended for Phase 1) ─────────────")
    all_combos = [(sh, fn, exp, n_s, wr, sr) for sh, fn, exp, n_s, wr, sr in singles + pairs
                  if n_s >= 50]
    all_combos.sort(key=lambda x: x[0], reverse=True)
    for sh, fname, exp, n_sub, wr, sr in all_combos[:8]:
        flag = "★★" if exp > 0.15 and sh > 0.10 else ("★" if exp > 0.05 else "")
        print(f"  {fname:<44s}  n={n_sub:4d}  exp={exp:+.3f}R  WR={wr:.0%}  "
              f"Sharpe={sh:+.3f}  {flag}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n  ── VERDICT ──────────────────────────────────────────────────────")
    top_n50 = [(sh, fn, exp, n_s) for sh, fn, exp, n_s, *_ in all_combos if n_s >= 50]
    if top_n50:
        best_sh, best_fn, best_exp, best_n = top_n50[0]
        if best_sh > 0.10 and best_exp > 0.05:
            print(f"    PROCEED TO PHASE 1")
            print(f"    Best filter (n≥50): {best_fn}")
            print(f"    n={best_n}  exp={best_exp:+.3f}R  Sharpe={best_sh:+.3f}")
            print(f"    Lock this filter for Phase 1 entry method testing.")
        else:
            print(f"    WEAK EDGE — best filter exp={best_exp:+.3f}R, Sharpe={best_sh:+.3f}")
            print(f"    Marginal: proceed with caution or drop if no clear signal in pairs.")
    else:
        print(f"    NO ROBUST FILTER FOUND (no filter with n≥50 and Sharpe > 0.10)")
        print(f"    Consider dropping ORB Failure or use gap_up only (n={len([e for e in all_events if e.gap_pos])})")


if __name__ == "__main__":
    main()
