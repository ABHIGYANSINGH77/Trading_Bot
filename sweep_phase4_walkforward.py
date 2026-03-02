"""sweep_phase4_walkforward.py — Event Type 2: Session Sweep + Rejection

Phase 4: Walk-Forward Validation

Question: "Does the sweep strategy, trained on Phase 0-3 findings, hold up on unseen data?"

Locked strategies from Phase 1-3:
  PRIMARY:      M2 conf_close + Filter B (pdl_long + prior_up)
                + Regime: midweek + near_extreme
                + Exit X1 (EOD or stop)
                IS baseline: n≈55/2yr, exp=+0.690R, Sharpe=+0.333

  HIGH-QUALITY: M1 sweep_close + Filter A (nvda + avoid_13_14 + prior_up)
                + Regime: gap_pos + not_friday
                + Exit X3 (Fixed 1R)
                IS baseline: n≈50/2yr, exp=+0.620R, Sharpe=+0.841

Method: Rolling walk-forward — 3-month train, 1-month test, step 1 month.
  Fixed mode:   Locked Phase 3 filters applied as-is to OOS window.
  Naive bench:  Same base event detection, NO filters — is the filter actually adding value?

Success criteria:
  1. Avg OOS expectancy > +0.03R
  2. > 60% of windows show positive OOS expectancy
  3. IS→OOS degradation ratio > 0.40 (OOS retains ≥40% of IS edge)
  4. Combined OOS t-stat: p < 0.10

Additional analyses:
  - Per-symbol OOS breakdown
  - Bootstrap 90% CI on OOS expectancy
  - Comparison of PRIMARY vs HIGH-QUALITY OOS
  - Naive (unfiltered M2) benchmark

Usage:
  python3 sweep_phase4_walkforward.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
                                       ./data/cache/NVDA_2024-01-01_2025-12-31_15_mins.csv \\
                                       ./data/cache/MSFT_2024-01-01_2025-12-31_15_mins.csv \\
                                       ./data/cache/AMZN_2024-01-01_2025-12-31_15_mins.csv \\
                                       ./data/cache/GOOG_2024-01-01_2025-12-31_15_mins.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import time as dtime, date as date_type
from dateutil.relativedelta import relativedelta
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MAX_CONF_BARS     = 3
ATR_PERIOD        = 14
RVOL_LOOKBACK     = 10
RANGE_LOOKBACK    = 20

TRAIN_MONTHS      = 3
TEST_MONTHS       = 1
MIN_OOS_TRADES    = 3    # minimum trades in an OOS window to count it


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SweepEvent:
    """All signals needed for filtering, computed at trade entry time."""
    symbol:          str
    date:            str
    date_obj:        object   # date for window splitting
    direction:       str

    pd_level:        float
    pd_high:         float
    pd_low:          float
    pd_range:        float
    sweep_extreme:   float
    overshoot_abs:   float
    overshoot_pct:   float

    sweep_bar_iloc:  int
    sweep_bar_close: float
    sweep_hour:      int

    conf_bar_iloc:   int
    conf_bar_close:  float
    conf_bar_num:    int

    prior_day_up:    bool
    gap_pct:         float

    # Regime signals
    gap_pos:         bool
    gap_neg:         bool
    gap_abs:         bool
    gap_aligned:     bool
    orb_up:          bool
    orb_down:        bool
    rvol_high:       bool
    rvol_low:        bool
    atr_expanding:   bool
    atr_quiet:       bool
    atr_normal:      bool
    vwap_aligned:    bool
    dow_monday:      bool
    dow_friday:      bool
    dow_midweek:     bool
    near_extreme:    bool
    pd_range_large:  bool
    pd_range_small:  bool

    day_df:          object


@dataclass
class WindowResult:
    window_num:    int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    is_exp:        float
    is_n:          int
    oos_pnls:      List[float]
    oos_symbols:   List[str]

    @property
    def oos_exp(self) -> float:
        return float(np.mean(self.oos_pnls)) if self.oos_pnls else 0.0

    @property
    def oos_n(self) -> int:
        return len(self.oos_pnls)

    @property
    def oos_positive(self) -> bool:
        return self.oos_exp > 0


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

def compute_vwap(day_df: pd.DataFrame, up_to_iloc: int) -> float:
    sub = day_df.iloc[:up_to_iloc + 1]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol = sub["volume"]
    return float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(tp.mean())


# ── Event Detection ───────────────────────────────────────────────────────────

def find_events(df: pd.DataFrame) -> List[SweepEvent]:
    symbol  = df["symbol"].iloc[0]
    events: List[SweepEvent] = []
    pd_high = pd_low = pd_close = pd_open = None
    pd_range_rolling: List[float] = []

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if pd_high is None:
            pd_high = float(day["high"].max())
            pd_low  = float(day["low"].min())
            pd_close = float(day["close"].iloc[-1])
            pd_open  = float(day["open"].iloc[0])
            pd_range_rolling.append(pd_high - pd_low)
            continue

        pd_range     = pd_high - pd_low
        prior_day_up = pd_close > pd_open
        gap_pct      = (float(day["open"].iloc[0]) - pd_close) / pd_close if pd_close > 0 else 0.0

        # ORB direction (first 2 bars)
        orb_bars = day.iloc[:min(2, len(day))]
        orb_up   = bool(orb_bars.iloc[-1]["close"] > orb_bars.iloc[0]["open"]) if len(orb_bars) >= 1 else False
        orb_down = not orb_up

        # Prior range percentile
        if len(pd_range_rolling) >= RANGE_LOOKBACK:
            pct25 = float(np.percentile(pd_range_rolling[-RANGE_LOOKBACK:], 25))
            pct75 = float(np.percentile(pd_range_rolling[-RANGE_LOOKBACK:], 75))
        else:
            pct25, pct75 = -1.0, 1e9
        pd_range_large = pd_range > pct75
        pd_range_small = pd_range < pct25

        # Day of week
        dow         = day_date.weekday()
        dow_monday  = (dow == 0)
        dow_friday  = (dow == 4)
        dow_midweek = (dow in (1, 2, 3))

        found = False
        for i in range(len(day) - 1):
            if found:
                break
            bar = day.iloc[i]

            for direction, sweep_check, conf_check, lvl in [
                ("SHORT",
                 lambda b, h=pd_high: b["high"] > h and b["close"] <= h,
                 lambda c, h=pd_high: c["close"] < c["open"] and c["close"] < h,
                 pd_high),
                ("LONG",
                 lambda b, l=pd_low: b["low"] < l and b["close"] >= l,
                 lambda c, l=pd_low: c["close"] > c["open"] and c["close"] > l,
                 pd_low),
            ]:
                if not sweep_check(bar):
                    continue

                sweep_ext = float(bar["high"]) if direction == "SHORT" else float(bar["low"])
                ov_abs    = (sweep_ext - lvl) if direction == "SHORT" else (lvl - sweep_ext)
                ov_pct    = ov_abs / lvl if lvl > 0 else 0.0

                # ── Regime signals ──
                # ATR
                atr_vals = []
                for j in range(1, i):
                    tr = max(day.iloc[j]["high"] - day.iloc[j]["low"],
                             abs(day.iloc[j]["high"] - day.iloc[j-1]["close"]),
                             abs(day.iloc[j]["low"]  - day.iloc[j-1]["close"]))
                    atr_vals.append(tr)
                atr_n   = min(ATR_PERIOD, len(atr_vals))
                atr_avg = float(np.mean(atr_vals[-atr_n:])) if atr_vals else 1.0
                br      = float(bar["high"] - bar["low"])
                atr_exp = br > 2.0 * atr_avg
                atr_qui = br < 0.5 * atr_avg
                atr_nor = not atr_exp and not atr_qui

                # RVOL
                vol_hist = day.iloc[:i]["volume"].values
                vol_avg  = float(np.mean(vol_hist[-RVOL_LOOKBACK:])) if len(vol_hist) >= 3 else 1.0
                bvol     = float(bar["volume"])
                rvol_hi  = bvol > 1.5 * vol_avg
                rvol_lo  = bvol < 0.7 * vol_avg

                # VWAP
                vwap_at = compute_vwap(day, i)
                if direction == "LONG":
                    vwap_al = bar["close"] < vwap_at
                else:
                    vwap_al = bar["close"] > vwap_at

                # Near intraday extreme
                lb = day.iloc[max(0, i - 20):i + 1]
                if len(lb) >= 3:
                    lb_h = lb["high"].max(); lb_l = lb["low"].min()
                    lb_r = lb_h - lb_l
                    if lb_r > 0:
                        near_ext = ((lb_l - bar["low"]) / lb_r < 0.05 if direction == "LONG"
                                    else (bar["high"] - lb_h) / lb_r < 0.05)
                    else:
                        near_ext = True
                else:
                    near_ext = False

                # Gap alignment
                gap_al = (gap_pct < -0.002) if direction == "LONG" else (gap_pct > 0.002)

                # Find confirmation bar
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if not conf_check(conf):
                        continue
                    events.append(SweepEvent(
                        symbol=symbol, date=str(day_date), date_obj=day_date,
                        direction=direction,
                        pd_level=lvl, pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                        sweep_extreme=sweep_ext, overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                        sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                        sweep_hour=bar["time"].hour,
                        conf_bar_iloc=i + k, conf_bar_close=float(conf["close"]),
                        conf_bar_num=k,
                        prior_day_up=prior_day_up, gap_pct=gap_pct,
                        gap_pos=(gap_pct > 0.002), gap_neg=(gap_pct < -0.002),
                        gap_abs=(abs(gap_pct) > 0.003), gap_aligned=gap_al,
                        orb_up=orb_up, orb_down=orb_down,
                        rvol_high=rvol_hi, rvol_low=rvol_lo,
                        atr_expanding=atr_exp, atr_quiet=atr_qui, atr_normal=atr_nor,
                        vwap_aligned=vwap_al,
                        dow_monday=dow_monday, dow_friday=dow_friday, dow_midweek=dow_midweek,
                        near_extreme=near_ext,
                        pd_range_large=pd_range_large, pd_range_small=pd_range_small,
                        day_df=day,
                    ))
                    found = True
                    break
                if found:
                    break

        pd_range_rolling.append(pd_high - pd_low)
        pd_high = float(day["high"].max())
        pd_low  = float(day["low"].min())
        pd_close = float(day["close"].iloc[-1])
        pd_open  = float(day["open"].iloc[0])

    return events


# ── Exit Simulation ───────────────────────────────────────────────────────────

def x1_eod(ev: SweepEvent) -> float:
    """X1: EOD or stop — Primary locked exit."""
    entry = ev.conf_bar_close
    stop  = ev.sweep_extreme
    risk  = abs(entry - stop)
    if risk <= 0:
        return 0.0
    for _, bar in ev.day_df.iloc[ev.conf_bar_iloc + 1:].iterrows():
        if ev.direction == "LONG" and bar["low"] <= stop:
            return -1.0
        if ev.direction == "SHORT" and bar["high"] >= stop:
            return -1.0
    eod = float(ev.day_df["close"].iloc[-1])
    return (eod - entry) / risk if ev.direction == "LONG" else (entry - eod) / risk


def x3_fixed_1r(ev: SweepEvent) -> float:
    """X3: Fixed 1R target — High-quality locked exit, M1 entry."""
    entry  = ev.sweep_bar_close
    stop   = ev.sweep_extreme
    risk   = abs(entry - stop)
    if risk <= 0:
        return 0.0
    target = (entry + risk) if ev.direction == "LONG" else (entry - risk)
    for _, bar in ev.day_df.iloc[ev.sweep_bar_iloc + 1:].iterrows():
        if ev.direction == "LONG":
            if bar["low"] <= stop:
                return -1.0
            if bar["high"] >= target:
                return 1.0
        else:
            if bar["high"] >= stop:
                return -1.0
            if bar["low"] <= target:
                return 1.0
    eod = float(ev.day_df["close"].iloc[-1])
    return (eod - entry) / risk if ev.direction == "LONG" else (entry - eod) / risk


def m2_eod_no_filter(ev: SweepEvent) -> float:
    """Naive benchmark: M2 entry, X1 EOD, NO filter."""
    return x1_eod(ev)


# ── Filters (locked from Phase 0.5 + Phase 3) ────────────────────────────────

def primary_filter(ev: SweepEvent) -> bool:
    """Phase 0.5 Filter B + Phase 3 regime: pdl_long + prior_up + midweek + near_extreme."""
    return (ev.direction == "LONG"
            and ev.prior_day_up
            and ev.dow_midweek
            and ev.near_extreme)


def hq_filter(ev: SweepEvent) -> bool:
    """Phase 0.5 Filter A + Phase 3 regime: nvda + avoid_13_14 + prior_up + gap_pos + not_friday."""
    return (ev.symbol == "NVDA"
            and ev.sweep_hour not in (13, 14)
            and ev.prior_day_up
            and ev.gap_pos
            and not ev.dow_friday)


def naive_filter(_: SweepEvent) -> bool:
    return True


# ── Walk-Forward Engine ───────────────────────────────────────────────────────

def define_windows(all_events: List[SweepEvent]) -> List[Tuple]:
    """Generate rolling (train_start, train_end, test_start, test_end) date tuples."""
    if not all_events:
        return []
    dates   = sorted(set(ev.date_obj for ev in all_events))
    min_d   = dates[0]
    max_d   = dates[-1]
    windows = []
    # Start from earliest date, step by TEST_MONTHS
    t_start = min_d
    while True:
        t_end   = t_start + relativedelta(months=TRAIN_MONTHS) - relativedelta(days=1)
        oos_s   = t_start + relativedelta(months=TRAIN_MONTHS)
        oos_e   = oos_s   + relativedelta(months=TEST_MONTHS)  - relativedelta(days=1)
        if oos_e > max_d:
            break
        windows.append((t_start, t_end, oos_s, oos_e))
        t_start = t_start + relativedelta(months=TEST_MONTHS)
    return windows


def run_window(events: List[SweepEvent], window: Tuple,
               filt: callable, sim_fn: callable) -> WindowResult:
    t_start, t_end, oos_s, oos_e = window
    is_evs  = [ev for ev in events if t_start <= ev.date_obj <= t_end   and filt(ev)]
    oos_evs = [ev for ev in events if oos_s  <= ev.date_obj <= oos_e    and filt(ev)]
    is_pnls  = [sim_fn(ev) for ev in is_evs]
    oos_pnls = [sim_fn(ev) for ev in oos_evs]
    is_exp   = float(np.mean(is_pnls)) if is_pnls else 0.0
    return WindowResult(
        window_num=0,
        train_start=str(t_start), train_end=str(t_end),
        test_start=str(oos_s), test_end=str(oos_e),
        is_exp=is_exp, is_n=len(is_pnls),
        oos_pnls=oos_pnls,
        oos_symbols=[ev.symbol for ev in oos_evs],
    )


# ── Statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(pnls: List[float], n_boot: int = 2000,
                 ci: float = 0.90) -> Tuple[float, float]:
    if len(pnls) < 2:
        return (0.0, 0.0)
    arr  = np.array(pnls)
    boot = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo   = (1 - ci) / 2
    return float(np.percentile(boot, lo * 100)), float(np.percentile(boot, (1 - lo) * 100))


def t_test(pnls: List[float]) -> Tuple[float, float]:
    if len(pnls) < 3:
        return (0.0, 1.0)
    t, p = scipy_stats.ttest_1samp(pnls, 0.0)
    return float(t), float(p)


def sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    if arr.std() < 1e-9:
        return 0.0
    return float(arr.mean() / arr.std())


# ── Strategy Runner ───────────────────────────────────────────────────────────

def run_strategy(all_events: List[SweepEvent], windows: List[Tuple],
                 filt: callable, sim_fn: callable,
                 label: str, naive_label: str = "") -> None:
    print(f"\n  {'═'*110}")
    print(f"  {label}")
    print(f"  {'═'*110}")

    results: List[WindowResult] = []
    for i, w in enumerate(windows):
        r = run_window(all_events, w, filt, sim_fn)
        r.window_num = i + 1
        results.append(r)

    # Full-sample IS (using all data — just for reference)
    all_is = [ev for ev in all_events if filt(ev)]
    all_is_pnls = [sim_fn(ev) for ev in all_is]
    full_is_exp = float(np.mean(all_is_pnls)) if all_is_pnls else 0.0
    full_is_sh  = sharpe(all_is_pnls)
    print(f"\n  Full-sample IS check: n={len(all_is_pnls)}  "
          f"exp={full_is_exp:+.3f}R  Sharpe={full_is_sh:+.3f}")

    # Window table
    print(f"\n  {'Win':>3}  {'Train period':<24}  {'OOS period':<24}  "
          f"{'IS exp':>8}  {'IS n':>4}  {'OOS exp':>9}  {'OOS n':>5}  {'✓'}  ")
    print(f"  {'─'*105}")

    all_oos: List[float] = []
    all_oos_syms: List[str] = []
    counted = 0

    for r in results:
        flag = "+"  if r.oos_positive else "-"
        skip = "" if r.oos_n >= MIN_OOS_TRADES else " [skip]"
        print(f"  {r.window_num:>3}  "
              f"[{r.train_start} – {r.train_end}]  "
              f"[{r.test_start} – {r.test_end}]  "
              f"IS={r.is_exp:+.3f}R  n={r.is_n:>3}  "
              f"OOS={r.oos_exp:+.3f}R  n={r.oos_n:>4}  {flag}{skip}")
        if r.oos_n >= MIN_OOS_TRADES:
            all_oos.extend(r.oos_pnls)
            all_oos_syms.extend(r.oos_symbols)
            counted += 1

    if not all_oos:
        print(f"\n  No OOS windows with ≥{MIN_OOS_TRADES} trades — insufficient data.")
        return

    # Aggregate stats
    counted_results = [r for r in results if r.oos_n >= MIN_OOS_TRADES]
    oos_exps        = [r.oos_exp for r in counted_results]
    avg_oos         = float(np.mean(oos_exps))
    pct_pos         = sum(1 for e in oos_exps if e > 0) / len(oos_exps)
    avg_is          = float(np.mean([r.is_exp for r in counted_results]))
    ratio           = avg_oos / avg_is if abs(avg_is) > 1e-9 else 0.0
    t_val, p_val    = t_test(all_oos)
    ci_lo, ci_hi    = bootstrap_ci(all_oos)
    oos_sh          = sharpe(all_oos)

    print(f"\n  Aggregate OOS ({counted} windows, {len(all_oos)} trades):")
    print(f"    Avg OOS expectancy:  {avg_oos:+.3f}R  (>+0.03R required)")
    print(f"    % positive windows:  {pct_pos:.0%}            (>60% required)")
    print(f"    IS→OOS ratio:        {ratio:+.3f}           (>0.40 required)")
    print(f"    t-stat: {t_val:+.3f}  p={p_val:.3f}  (p<0.10 required)")
    print(f"    Bootstrap 90% CI:    [{ci_lo:+.3f}R, {ci_hi:+.3f}R]")
    print(f"    OOS Sharpe:          {oos_sh:+.3f}")

    # Success criteria
    c1 = avg_oos  > 0.03
    c2 = pct_pos  > 0.60
    c3 = ratio    > 0.40
    c4 = p_val    < 0.10
    passed = sum([c1, c2, c3, c4])
    print(f"\n  Success criteria:")
    print(f"    [{'✓' if c1 else '✗'}] Avg OOS exp > +0.03R      → {avg_oos:+.3f}R")
    print(f"    [{'✓' if c2 else '✗'}] >60% positive windows     → {pct_pos:.0%}")
    print(f"    [{'✓' if c3 else '✗'}] IS/OOS ratio > 0.40       → {ratio:.2f}")
    print(f"    [{'✓' if c4 else '✗'}] p < 0.10                  → p={p_val:.3f}")
    print(f"\n    → {passed}/4 criteria passed")

    # Per-symbol
    sym_pnls: Dict[str, List[float]] = {}
    for pnl, sym in zip(all_oos, all_oos_syms):
        sym_pnls.setdefault(sym, []).append(pnl)
    print(f"\n  Per-symbol OOS:")
    for sym in sorted(sym_pnls):
        pnls = sym_pnls[sym]
        flag = "✓" if np.mean(pnls) > 0.03 else ("~" if np.mean(pnls) > 0 else "✗")
        print(f"    {sym:6s}  n={len(pnls):3d}  exp={np.mean(pnls):+.3f}R  "
              f"WR={np.mean([p>0 for p in pnls]):.0%}  {flag}")

    # Verdict
    print(f"\n  VERDICT: ", end="")
    if passed >= 3:
        print(f"PASS ({passed}/4) — strategy holds up OOS, proceed to Event Type 3")
    elif passed == 2:
        print(f"MARGINAL ({passed}/4) — edge present but weak; acceptable to include in ensemble "
              f"with reduced position sizing")
    else:
        print(f"FAIL ({passed}/4) — edge does not survive walk-forward; do not include "
              f"this strategy variant")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase4_walkforward.py <csv_files...>")
        sys.exit(1)

    all_events: List[SweepEvent] = []
    for path in sys.argv[1:]:
        df   = load_csv(path)
        sym  = Path(path).stem.split("_")[0]
        evs  = find_events(df)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    windows = define_windows(all_events)

    print()
    print("=" * 115)
    print("  EVENT TYPE 2: SWEEP + REJECTION — PHASE 4 WALK-FORWARD VALIDATION")
    print(f"  {len(windows)} walk-forward windows  ({TRAIN_MONTHS}-month train / {TEST_MONTHS}-month test)")
    print("=" * 115)

    # ── 1. Naive benchmark: M2, no filter ────────────────────────────────────
    run_strategy(all_events, windows, naive_filter, m2_eod_no_filter,
                 "NAIVE BENCHMARK — M2 conf_close, no filter, X1 EOD")

    # ── 2. Primary: M2 + Filter B + midweek + near_extreme + X1 EOD ──────────
    run_strategy(all_events, windows, primary_filter, x1_eod,
                 "PRIMARY — M2 + pdl_long+prior_up + midweek+near_extreme + X1 EOD")

    # ── 3. High-quality: M1 + Filter A + gap_pos + not_friday + X3 1R ────────
    run_strategy(all_events, windows, hq_filter, x3_fixed_1r,
                 "HIGH-QUALITY — M1 + nvda+avoid13_14+prior_up+gap_pos+not_friday + X3 1R")

    print()
    print("=" * 115)
    print("  FINAL SUMMARY")
    print("=" * 115)
    print("""
  Interpretation guide:
    PRIMARY strategy tests the broad PDL LONG + midweek + near_extreme setup.
    HIGH-QUALITY tests the tighter NVDA morning sweep with gap filter.

    A PASS on either strategy → include it in the final event_driven.py
    A MARGINAL result → include with half position size
    A FAIL → do not include (strategy didn't generalise to new data)

  Next: Event Type 3 — ORB Failure (Fade the Failed Breakout)
    """)


if __name__ == "__main__":
    main()
