"""orbfail_phase3_4_validation.py — Event Type 3: ORB Failure

Combined Phase 3 (Regime Filter) + Phase 4 (Walk-Forward Validation)

Locked from Phase 1+2:
  Filter B primary: gap_up + not_goog   → n=123, exp=+0.218R EOD / +0.196R X6_partial, Sharpe=+0.256
  Filter A secondary: gap_up + fail_11h → n=72,  exp=+0.223R EOD / +0.220R X6_partial, Sharpe=+0.298
  Entry: M1 fail_close
  Exit:  X6 partial (50% at 1R, trail remainder with 1x ATR)
  Baseline exit also tested: X1 EOD (simpler, higher absolute exp)

Phase 3 — Regime filter scan:
  Tests additional conditions on top of locked base filter.
  All signals must be observable at or before the failure bar close.

Phase 4 — Walk-forward:
  3-month train, 1-month test rolling windows.
  Tests locked strategy on unseen data.
  Note: smaller n than sweep — some windows will be skipped (< MIN_OOS_TRADES=2).

Usage:
  python3 orbfail_phase3_4_validation.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
                                          ./data/cache/NVDA_2024-01-01_2025-12-31_15_mins.csv \\
                                          ./data/cache/MSFT_2024-01-01_2025-12-31_15_mins.csv \\
                                          ./data/cache/AMZN_2024-01-01_2025-12-31_15_mins.csv \\
                                          ./data/cache/GOOG_2024-01-01_2025-12-31_15_mins.csv
"""

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from datetime import time as dtime
from dateutil.relativedelta import relativedelta
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
ORB_BARS          = 2
MAX_BO_BAR        = 8
FAIL_WINDOW       = 6
ATR_PERIOD        = 14
TRAIL_ACTIVATE_R  = 0.5
TRAIN_MONTHS      = 3
TEST_MONTHS       = 1
MIN_OOS_TRADES    = 2
MIN_TRADES_P3     = 15

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBFailEvent:
    symbol:            str
    date:              str
    date_obj:          object
    direction:         str

    orb_high:          float
    orb_low:           float
    orb_mid:           float
    orb_range:         float

    breakout_bar:      int
    breakout_extreme:  float
    fail_bar:          int
    fail_hour:         int
    fail_bar_close:    float
    vwap_at_fail:      float

    gap_pct:           float
    gap_pos:           bool
    gap_neg:           bool
    prior_day_up:      bool

    # Regime signals
    orb_large:         bool    # ORB range > 75th pct of rolling ORBs
    orb_small:         bool
    rvol_high:         bool    # breakout bar volume > 1.5x avg
    rvol_normal:       bool
    atr_normal:        bool    # breakout bar range is normal (not expansion)
    fail_above_vwap:   bool    # at failure, price still above VWAP (SHORT fades from elevated)
    dow_monday:        bool
    dow_friday:        bool
    dow_midweek:       bool

    day_df:            object


@dataclass
class WFResult:
    window_num:  int
    train_start: str; train_end: str
    test_start:  str; test_end:  str
    is_exp:      float; is_n: int
    oos_pnls:    List[float]
    oos_symbols: List[str]

    @property
    def oos_exp(self) -> float: return float(np.mean(self.oos_pnls)) if self.oos_pnls else 0.0
    @property
    def oos_n(self) -> int: return len(self.oos_pnls)
    @property
    def oos_positive(self) -> bool: return self.oos_exp > 0


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


def compute_atr_from_iloc(day: pd.DataFrame, up_to: int) -> float:
    bars = day.iloc[:up_to + 1]
    if len(bars) < 2: return float(bars.iloc[-1]["high"] - bars.iloc[-1]["low"])
    trs = [bars.iloc[0]["high"] - bars.iloc[0]["low"]]
    for j in range(1, len(bars)):
        tr = max(bars.iloc[j]["high"] - bars.iloc[j]["low"],
                 abs(bars.iloc[j]["high"] - bars.iloc[j-1]["close"]),
                 abs(bars.iloc[j]["low"]  - bars.iloc[j-1]["close"]))
        trs.append(tr)
    return float(np.mean(trs[-min(ATR_PERIOD, len(trs)):]))


# ── Event Detection ───────────────────────────────────────────────────────────

def find_events(df: pd.DataFrame) -> List[ORBFailEvent]:
    symbol = df["symbol"].iloc[0]
    events: List[ORBFailEvent] = []
    prior_close = None; pd_close = None; pd_open = None
    orb_range_hist: List[float] = []

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if prior_close is None:
            prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
            pd_open = float(day["open"].iloc[0]); continue

        prior_day_up = pd_close > pd_open
        gap_pct      = (float(day["open"].iloc[0]) - prior_close) / prior_close if prior_close > 0 else 0.0
        gap_pos      = gap_pct > 0.002; gap_neg = gap_pct < -0.002

        if len(day) < ORB_BARS + 1:
            prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
            pd_open = float(day["open"].iloc[0]); continue

        orb = day.iloc[:ORB_BARS]
        orb_high = float(orb["high"].max()); orb_low = float(orb["low"].min())
        orb_mid  = (orb_high + orb_low) / 2.0; orb_range = orb_high - orb_low

        # ORB range percentile
        if len(orb_range_hist) >= 20:
            p25 = float(np.percentile(orb_range_hist[-20:], 25))
            p75 = float(np.percentile(orb_range_hist[-20:], 75))
        else:
            p25, p75 = -1.0, 1e9
        orb_large = orb_range > p75; orb_small = orb_range < p25

        # Day of week
        dow = day_date.weekday()
        dow_monday = (dow == 0); dow_friday = (dow == 4); dow_midweek = (dow in (1, 2, 3))
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

                # Breakout bar regime signals
                vol_hist = day.iloc[:i]["volume"].values
                vol_avg  = float(np.mean(vol_hist[-10:])) if len(vol_hist) >= 3 else 1.0
                rvol_hi  = float(bar["volume"]) > 1.5 * vol_avg
                rvol_nor = not rvol_hi

                atr_avg  = compute_atr_from_iloc(day, i)
                br       = float(bar["high"] - bar["low"])
                atr_nor  = (br < 2.0 * atr_avg) and (br > 0.5 * atr_avg)

                running_ext = float(bar["high"]) if bo_dir == "LONG_BO" else float(bar["low"])
                scan_end    = min(i + FAIL_WINDOW + 1, len(day))

                for j_off, (_, fb) in enumerate(day.iloc[i + 1:scan_end].iterrows()):
                    if bo_dir == "LONG_BO": running_ext = max(running_ext, float(fb["high"]))
                    else:                   running_ext = min(running_ext, float(fb["low"]))
                    if not fail_check(fb): continue

                    fail_iloc   = i + 1 + j_off
                    vwap_at     = compute_vwap(day, fail_iloc)
                    fail_above  = float(fb["close"]) > vwap_at

                    events.append(ORBFailEvent(
                        symbol=symbol, date=str(day_date), date_obj=day_date,
                        direction=entry_dir,
                        orb_high=orb_high, orb_low=orb_low, orb_mid=orb_mid, orb_range=orb_range,
                        breakout_bar=i, breakout_extreme=running_ext,
                        fail_bar=fail_iloc, fail_hour=fb["time"].hour,
                        fail_bar_close=float(fb["close"]),
                        vwap_at_fail=vwap_at,
                        gap_pct=gap_pct, gap_pos=gap_pos, gap_neg=gap_neg,
                        prior_day_up=prior_day_up,
                        orb_large=orb_large, orb_small=orb_small,
                        rvol_high=rvol_hi, rvol_normal=rvol_nor,
                        atr_normal=atr_nor,
                        fail_above_vwap=fail_above,
                        dow_monday=dow_monday, dow_friday=dow_friday, dow_midweek=dow_midweek,
                        day_df=day,
                    ))
                    found = True; break
                if found: break

        orb_range_hist.append(orb_range)
        prior_close = float(day["close"].iloc[-1]); pd_close = prior_close
        pd_open = float(day["open"].iloc[0])

    return events


# ── Simulation ────────────────────────────────────────────────────────────────

def sim_m1_x6(ev: ORBFailEvent) -> float:
    """M1 entry (fail close) + X6 partial (50%@1R + ATR trail)."""
    entry = ev.fail_bar_close
    stop  = ev.breakout_extreme
    risk  = abs(entry - stop)
    if risk <= 0: return 0.0
    day   = ev.day_df
    atr   = compute_atr_from_iloc(day, ev.fail_bar)
    trail = atr
    target1    = (entry + risk) if ev.direction == "LONG" else (entry - risk)
    partial    = False; partial_pnl = 0.0
    activated  = False; trail_stop = stop; best_price = entry
    for _, bar in day.iloc[ev.fail_bar + 1:].iterrows():
        eff_stop = trail_stop if (partial and activated) else stop
        if ev.direction == "LONG"  and bar["low"]  <= eff_stop:
            sp = (eff_stop - entry) / risk
            return 0.5 * partial_pnl + 0.5 * sp if partial else sp
        if ev.direction == "SHORT" and bar["high"] >= eff_stop:
            sp = (entry - eff_stop) / risk
            return 0.5 * partial_pnl + 0.5 * sp if partial else sp
        if not partial:
            hit = (bar["high"] >= target1) if ev.direction == "LONG" else (bar["low"] <= target1)
            if hit:
                partial = True; partial_pnl = 1.0; activated = True; best_price = target1
        if activated:
            if ev.direction == "LONG":
                best_price = max(best_price, float(bar["close"]))
                trail_stop = max(trail_stop, best_price - trail)
            else:
                best_price = min(best_price, float(bar["close"]))
                trail_stop = min(trail_stop, best_price + trail)
    eod = float(day["close"].iloc[-1])
    ep  = (eod - entry) / risk if ev.direction == "LONG" else (entry - eod) / risk
    return 0.5 * partial_pnl + 0.5 * ep if partial else ep


def sim_m1_x1_eod(ev: ORBFailEvent) -> float:
    """M1 + X1 EOD (simpler baseline for comparison)."""
    entry = ev.fail_bar_close; stop = ev.breakout_extreme; risk = abs(entry - stop)
    if risk <= 0: return 0.0
    day = ev.day_df
    for _, bar in day.iloc[ev.fail_bar + 1:].iterrows():
        if ev.direction == "LONG"  and bar["low"]  <= stop: return -1.0
        if ev.direction == "SHORT" and bar["high"] >= stop: return -1.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if ev.direction == "LONG" else (entry - eod) / risk


# ── Phase 3: Regime Filters ───────────────────────────────────────────────────

FILTER_B = lambda e: e.gap_pos and e.symbol != "GOOG"
FILTER_A = lambda e: e.gap_pos and e.fail_hour == 11

def build_regime_filters() -> Dict[str, Callable]:
    return {
        "prior_up":         lambda e: e.prior_day_up,
        "prior_down":       lambda e: not e.prior_day_up,
        "orb_large":        lambda e: e.orb_large,
        "orb_small":        lambda e: e.orb_small,
        "rvol_high":        lambda e: e.rvol_high,
        "rvol_normal":      lambda e: e.rvol_normal,
        "atr_normal":       lambda e: e.atr_normal,
        "fail_above_vwap":  lambda e: e.fail_above_vwap,
        "fail_below_vwap":  lambda e: not e.fail_above_vwap,
        "monday":           lambda e: e.dow_monday,
        "midweek":          lambda e: e.dow_midweek,
        "not_friday":       lambda e: not e.dow_friday,
        "fade_long_bo":     lambda e: e.direction == "SHORT",  # fade failed LONG breakout
        "fade_short_bo":    lambda e: e.direction == "LONG",
        "fail_11h":         lambda e: e.fail_hour == 11,
        "fail_12h":         lambda e: e.fail_hour == 12,
        "fail_not_10h":     lambda e: e.fail_hour != 10,
        "nvda":             lambda e: e.symbol == "NVDA",
        "aapl_amzn_nvda":   lambda e: e.symbol in ("AAPL", "AMZN", "NVDA"),
    }


def sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    return float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0


def run_phase3(events: List[ORBFailEvent], base_filter: Callable,
               base_label: str, sim_fn: Callable, label: str) -> Tuple[str, Callable, float, float]:
    """Returns (best_filter_name, best_filter_fn, best_sharpe, best_exp) for Phase 4."""
    base   = [e for e in events if base_filter(e)]
    b_pnls = [sim_fn(e) for e in base]
    b_exp  = float(np.mean(b_pnls)); b_sh = sharpe(b_pnls)

    print(f"\n  {'═'*100}")
    print(f"  {label}")
    print(f"  {'═'*100}")
    print(f"  Baseline ({base_label}): n={len(base)}  exp={b_exp:+.3f}R  Sharpe={b_sh:+.3f}")

    filters  = build_regime_filters()
    MIN_IMP  = 0.05

    # Singles
    singles = []
    for fn, ff in filters.items():
        sub  = [e for e in base if ff(e)]
        if len(sub) < MIN_TRADES_P3: continue
        pnls = [sim_fn(e) for e in sub]
        sh   = sharpe(pnls); exp = float(np.mean(pnls))
        singles.append((sh, fn, exp, len(sub), ff))
    singles.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Top singles:")
    for sh, fn, exp, n, _ in singles[:8]:
        flag = "★★" if sh - b_sh > MIN_IMP and exp > 0 else ""
        print(f"  {fn:<26s}  n={n:4d}  exp={exp:+.3f}R  Sharpe={sh:+.3f}  Δ={sh-b_sh:+.3f}  {flag}")

    # Pairs
    pair_results = []
    for (sh1, fn1, e1, n1, ff1), (sh2, fn2, e2, n2, ff2) in itertools.combinations(
            [(s[0], s[1], s[2], s[3], s[4]) for s in singles[:6]], 2):
        sub  = [e for e in base if ff1(e) and ff2(e)]
        if len(sub) < MIN_TRADES_P3: continue
        pnls = [sim_fn(e) for e in sub]
        sh   = sharpe(pnls); exp = float(np.mean(pnls))
        pair_results.append((sh, f"{fn1}+{fn2}", exp, len(sub), lambda e, a=ff1, b=ff2: a(e) and b(e)))
    pair_results.sort(key=lambda x: x[0], reverse=True)

    if pair_results:
        print(f"\n  Top pairs:")
        for sh, fn, exp, n, _ in pair_results[:6]:
            flag = "★★" if sh - b_sh > MIN_IMP and exp > 0 else ""
            print(f"  {fn:<36s}  n={n:4d}  exp={exp:+.3f}R  Sharpe={sh:+.3f}  Δ={sh-b_sh:+.3f}  {flag}")

    # Best filter for Phase 4
    all_scored = [(sh, fn, exp, n, ff) for sh, fn, exp, n, ff in singles + pair_results
                  if sh - b_sh > MIN_IMP and exp > 0 and n >= 30]
    all_scored.sort(key=lambda x: x[0], reverse=True)

    if all_scored:
        best = all_scored[0]
        print(f"\n  BEST for Phase 4: {best[1]}  n={best[3]}  exp={best[2]:+.3f}R  Sharpe={best[0]:+.3f}")
        return best[1], best[4], best[0], best[2]
    else:
        print(f"\n  No regime filter improves Sharpe by >{MIN_IMP}. Using base filter as-is for Phase 4.")
        return base_label, base_filter, b_sh, b_exp


# ── Phase 4: Walk-Forward ─────────────────────────────────────────────────────

def define_windows(events: List[ORBFailEvent]) -> List[Tuple]:
    dates = sorted(set(e.date_obj for e in events))
    if not dates: return []
    windows = []
    t_start = dates[0]
    max_d   = dates[-1]
    while True:
        t_end  = t_start + relativedelta(months=TRAIN_MONTHS) - relativedelta(days=1)
        oos_s  = t_start + relativedelta(months=TRAIN_MONTHS)
        oos_e  = oos_s   + relativedelta(months=TEST_MONTHS)  - relativedelta(days=1)
        if oos_e > max_d: break
        windows.append((t_start, t_end, oos_s, oos_e))
        t_start = t_start + relativedelta(months=TEST_MONTHS)
    return windows


def bootstrap_ci(pnls: List[float], n_boot: int = 2000, ci: float = 0.90) -> Tuple[float, float]:
    if len(pnls) < 2: return (0.0, 0.0)
    arr  = np.array(pnls)
    boot = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo   = (1 - ci) / 2
    return float(np.percentile(boot, lo * 100)), float(np.percentile(boot, (1 - lo) * 100))


def run_walkforward(events: List[ORBFailEvent], windows: List[Tuple],
                    filt: Callable, sim_fn: Callable, label: str) -> None:
    print(f"\n  {'═'*100}")
    print(f"  PHASE 4 WALK-FORWARD: {label}")
    print(f"  {'═'*100}")

    # Full IS check
    all_filt = [e for e in events if filt(e)]
    all_pnls = [sim_fn(e) for e in all_filt]
    print(f"\n  Full IS check: n={len(all_pnls)}  exp={np.mean(all_pnls):+.3f}R  "
          f"Sharpe={sharpe(all_pnls):+.3f}")

    print(f"\n  {'Win':>3}  {'Train':^24}  {'Test':^24}  {'IS exp':>8}  {'IS n':>4}  "
          f"{'OOS exp':>9}  {'OOS n':>5}  {'✓'}")
    print(f"  {'─'*100}")

    results: List[WFResult] = []
    for i, w in enumerate(windows):
        t_s, t_e, os_s, os_e = w
        is_ev  = [e for e in events if t_s <= e.date_obj <= t_e and filt(e)]
        oos_ev = [e for e in events if os_s <= e.date_obj <= os_e and filt(e)]
        is_p   = [sim_fn(e) for e in is_ev]
        oos_p  = [sim_fn(e) for e in oos_ev]
        is_exp = float(np.mean(is_p)) if is_p else 0.0
        r = WFResult(window_num=i+1, train_start=str(t_s), train_end=str(t_e),
                     test_start=str(os_s), test_end=str(os_e),
                     is_exp=is_exp, is_n=len(is_p),
                     oos_pnls=oos_p, oos_symbols=[e.symbol for e in oos_ev])
        results.append(r)
        skip = " [skip]" if r.oos_n < MIN_OOS_TRADES else ""
        print(f"  {i+1:>3}  [{t_s} – {t_e}]  [{os_s} – {os_e}]  "
              f"IS={is_exp:+.3f}R  n={r.is_n:>3}  OOS={r.oos_exp:+.3f}R  n={r.oos_n:>4}  "
              f"{'+'if r.oos_positive else '-'}{skip}")

    counted = [r for r in results if r.oos_n >= MIN_OOS_TRADES]
    if not counted:
        print(f"\n  No windows with ≥{MIN_OOS_TRADES} OOS trades. Data too thin for WF.")
        return

    all_oos   = [p for r in counted for p in r.oos_pnls]
    all_syms  = [s for r in counted for s in r.oos_symbols]
    oos_exps  = [r.oos_exp for r in counted]
    avg_oos   = float(np.mean(oos_exps))
    pct_pos   = sum(1 for e in oos_exps if e > 0) / len(oos_exps)
    avg_is    = float(np.mean([r.is_exp for r in counted]))
    ratio     = avg_oos / avg_is if abs(avg_is) > 1e-9 else 0.0
    t_v, p_v  = scipy_stats.ttest_1samp(all_oos, 0.0) if len(all_oos) >= 3 else (0.0, 1.0)
    ci_lo, ci_hi = bootstrap_ci(all_oos)
    oos_sh    = sharpe(all_oos)

    print(f"\n  Aggregate ({len(counted)} windows, {len(all_oos)} trades):")
    print(f"    Avg OOS exp:    {avg_oos:+.3f}R  (>+0.03R needed)")
    print(f"    % pos windows:  {pct_pos:.0%}     (>60% needed)")
    print(f"    IS/OOS ratio:   {ratio:.2f}      (>0.40 needed)")
    print(f"    p={p_v:.3f}               (p<0.10 needed)")
    print(f"    Bootstrap 90% CI: [{ci_lo:+.3f}R, {ci_hi:+.3f}R]")
    print(f"    OOS Sharpe: {oos_sh:+.3f}")

    c1=avg_oos>0.03; c2=pct_pos>0.60; c3=ratio>0.40; c4=p_v<0.10
    passed = sum([c1,c2,c3,c4])
    for check, crit in [(c1,f"OOS exp>{'+0.03R'} → {avg_oos:+.3f}R"),
                        (c2,f">60% pos windows → {pct_pos:.0%}"),
                        (c3,f"IS/OOS>0.40 → {ratio:.2f}"),
                        (c4,f"p<0.10 → p={p_v:.3f}")]:
        print(f"    [{'✓'if check else'✗'}] {crit}")

    # Per-symbol
    sym_d: Dict[str, List[float]] = {}
    for p, s in zip(all_oos, all_syms):
        sym_d.setdefault(s, []).append(p)
    print(f"\n  Per-symbol:")
    for sym in sorted(sym_d):
        p = sym_d[sym]
        print(f"    {sym:6s}  n={len(p):3d}  exp={np.mean(p):+.3f}R  WR={np.mean([x>0 for x in p]):.0%}  "
              f"{'✓'if np.mean(p)>0.03 else('~'if np.mean(p)>0 else'✗')}")

    print(f"\n  VERDICT: ", end="")
    if passed >= 3: print(f"PASS ({passed}/4)")
    elif passed == 2: print(f"MARGINAL ({passed}/4) — include at half position")
    else: print(f"FAIL ({passed}/4) — do not include")
    print(f"  Note: thin data ({len(all_oos)} OOS trades) — treat result as indicative only")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orbfail_phase3_4_validation.py <csv_files...>")
        sys.exit(1)

    all_events: List[ORBFailEvent] = []
    for path in sys.argv[1:]:
        df  = load_csv(path)
        sym = Path(path).stem.split("_")[0]
        evs = find_events(df)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    windows = define_windows(all_events)

    print()
    print("=" * 105)
    print("  EVENT TYPE 3: ORB FAILURE — PHASE 3 REGIME FILTER + PHASE 4 WALK-FORWARD")
    print(f"  {len(windows)} walk-forward windows  ({TRAIN_MONTHS}M train / {TEST_MONTHS}M test)")
    print("=" * 105)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print("\n" + "─"*105)
    print("  PHASE 3: REGIME FILTER SCAN")
    print("─"*105)

    p3b_name, p3b_filter, p3b_sh, p3b_exp = run_phase3(
        all_events, FILTER_B, "gap_up+not_goog",
        sim_m1_x6, "FILTER B (gap_up+not_goog) + M1+X6_partial")

    p3a_name, p3a_filter, p3a_sh, p3a_exp = run_phase3(
        all_events, FILTER_A, "gap_up+fail_11h",
        sim_m1_x6, "FILTER A (gap_up+fail_11h) + M1+X6_partial")

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    print("\n" + "─"*105)
    print("  PHASE 4: WALK-FORWARD VALIDATION")
    print("─"*105)

    # Naive (no filter)
    run_walkforward(all_events, windows, lambda _: True, sim_m1_x1_eod,
                    "NAIVE — M1 fail_close + X1 EOD, no filter")

    # Filter B primary
    def fb_final(e): return FILTER_B(e)
    run_walkforward(all_events, windows, fb_final, sim_m1_x6,
                    "PRIMARY — Filter B (gap_up+not_goog) + M1+X6_partial")

    # Filter B + best regime filter (from Phase 3)
    def fb_regime(e): return FILTER_B(e) and p3b_filter(e)
    run_walkforward(all_events, windows, fb_regime, sim_m1_x6,
                    f"PRIMARY + REGIME — {p3b_name} + M1+X6_partial")

    print()
    print("=" * 105)
    print("  FINAL SUMMARY: EVENT TYPE 3")
    print("=" * 105)
    print("""
  Decision:
    PASS (≥3/4)    → include in event_driven.py with full position
    MARGINAL (2/4) → include at half position
    FAIL (<2/4)    → do not include

  Note: ORB Failure has fewer trades than Sweep (~6 OOS/window vs ~10+ for sweeps).
  Walk-forward is indicative. If PASS, use conservative position sizing.
    """)


if __name__ == "__main__":
    main()
