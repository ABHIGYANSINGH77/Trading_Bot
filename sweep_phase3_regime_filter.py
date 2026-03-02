"""sweep_phase3_regime_filter.py — Event Type 2: Session Sweep + Rejection

Phase 3: Regime Filter Validation

Question: Are there day-level regime conditions that further improve the already-filtered strategy?

Locked from Phase 1 + 2:
  Primary:      M2 conf_close + Filter B (pdl_long + prior_up) + X1 EOD exit
                Baseline: n=176, exp=+0.467R, WR=45%, Sharpe=+0.171
  High-quality: M1 sweep_close + Filter A (nvda+avoid13_14+prior_up) + X3 Fixed 1R
                Baseline: n=113, exp=+0.636R, WR=81%, Sharpe=+0.899

Important constraint:
  ALL filters must use ONLY information available at the TIME of the sweep bar
  (i.e., prior day data, premarket gap, opening bar, intraday bars BEFORE the sweep).
  No look-ahead of any kind.

Regime dimensions tested:

  R1. Premarket gap direction / size
        gap_pos    = gap > +0.2%   (market opens above prior close — bias UP)
        gap_neg    = gap < -0.2%   (market opens below prior close — bias DOWN)
        gap_abs    = |gap| > 0.3%  (any strong gap)
        Hypothesis: PDL LONG sweeps work better when day opens WITH a gap down
        (price swept below PDL on a flush → immediate rejection more likely)
        Also test: gap_aligned = gap direction aligned with sweep direction
                   (gap_down for LONG sweeps / gap_up for SHORT sweeps)

  R2. ORB (Opening Range Breakout) direction
        ORB = first 30-min bar (bars 1-2 at 15-min resolution)
        orb_up   = ORB high > ORB low × 1.001 AND first close > first open
        orb_down = first close < first open
        Hypothesis: if ORB is directional DOWN and then PDL is swept → liquidity grab
        on a down-biased day, more likely to hold

  R3. Prior day range percentile
        pd_range_large = pd_range > 90th percentile of prior 20-day ranges
        pd_range_small = pd_range < 10th percentile
        Hypothesis: after large prior day ranges, PDH/PDL levels are more "respected"
        After small prior ranges, levels may be thin and prone to breakout

  R4. ATR condition at sweep time
        atr_expanding = current bar's range > 2× prior 14-bar avg (expansion)
        atr_quiet     = current bar's range < 0.5× prior avg (compression)
        Hypothesis: the sweep bar itself should be an expansion bar (rejection bar)
        A quiet sweep (small wick) may not signal real liquidity grab

  R5. Relative volume at sweep
        rvol_high = bar's volume > 1.5× prior 10-bar avg volume
        rvol_low  = bar's volume < 0.7× prior 10-bar avg volume
        Hypothesis: high volume on sweep bar = institutional rejection, stronger signal

  R6. VWAP position at sweep (from Phase 0.5 D9, now confirmed)
        long_below_vwap  = LONG sweep with price below VWAP at sweep
        short_above_vwap = SHORT sweep with price above VWAP at sweep
        Hypothesis (Phase 0.5): LONG sweeps work better below VWAP (+0.279R confirmed)
        Now test if this adds to the already-filtered cohort (Filter B is LONG only)

  R7. Day of week
        monday, friday — known for reversals and low follow-through
        Hypothesis: Monday sweeps may be more reliable (fresh weekly opening range)
        Friday sweeps may not complete (early close behaviour)

  R8. Prior swing structure (momentum)
        price_vs_20bar_low  = close within 20 bars before sweep at/near 20-bar low (LONG)
        price_vs_20bar_high = close within 20 bars at/near 20-bar high (SHORT)
        Hypothesis: PDL sweeps work best when intraday price is already weak
        (more trapped longs who will panic → fuel the recovery)

  R9. Consecutive sweep direction
        first_sweep = this is the first sweep of PDL/PDH this day (already enforced in detection)
        Tests if time since prior day's sweep matters (not implemented here — left for future)

Combinations tested systematically:
  - Singles: all R1-R8 filters individually
  - Pairs:   promising singles combined
  - Best pair locked → if any materially improve Sharpe on Primary cohort

Usage:
  python3 sweep_phase3_regime_filter.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple
from datetime import time as dtime

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MAX_CONF_BARS     = 3
ATR_PERIOD        = 14
RVOL_LOOKBACK     = 10   # bars for relative volume
RANGE_LOOKBACK    = 20   # days for prior range percentile
TOP_N_SINGLES     = 8    # top singles to combine into pairs
MIN_TRADES        = 20   # minimum n to report a filter

# Regime filter: minimum Sharpe improvement to be "material"
MIN_SHARPE_IMPROVEMENT = 0.05

# ── Phase 2 exit methods (locked) ─────────────────────────────────────────────

def x1_eod_exit(entry: float, stop: float, risk: float, direction: str,
                day: pd.DataFrame, from_iloc: int) -> float:
    """X1: EOD or stop — locked for Primary cohort."""
    if risk <= 0:
        return 0.0
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG" and bar["low"] <= stop:
            return -1.0
        if direction == "SHORT" and bar["high"] >= stop:
            return -1.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


def x3_fixed_1r_exit(entry: float, stop: float, risk: float, direction: str,
                     day: pd.DataFrame, from_iloc: int) -> float:
    """X3: Fixed 1R target or stop — locked for High-quality cohort."""
    if risk <= 0:
        return 0.0
    target = (entry + risk) if direction == "LONG" else (entry - risk)
    for _, bar in day.iloc[from_iloc:].iterrows():
        if direction == "LONG":
            if bar["low"] <= stop:
                return -1.0
            if bar["high"] >= target:
                return 1.0
        else:
            if bar["high"] >= stop:
                return -1.0
            if bar["low"] <= target:
                return 1.0
    eod = float(day["close"].iloc[-1])
    return (eod - entry) / risk if direction == "LONG" else (entry - eod) / risk


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RegimeSetup:
    """Enriched sweep setup with all regime signals computed at trade time."""
    # Core
    symbol:          str
    date:            str
    direction:       str
    pd_level:        float
    pd_high:         float
    pd_low:          float
    pd_range:        float
    sweep_extreme:   float
    overshoot_abs:   float
    overshoot_pct:   float

    # Entry
    sweep_bar_iloc:  int
    sweep_bar_close: float     # M1 entry
    sweep_hour:      int
    conf_bar_iloc:   int
    conf_bar_close:  float     # M2 entry
    conf_bar_num:    int

    # Prior day context
    prior_day_up:    bool
    gap_pct:         float
    prior_close:     float

    # Regime signals (computed at sweep bar time)
    gap_pos:         bool      # gap > +0.2%
    gap_neg:         bool      # gap < -0.2%
    gap_abs:         bool      # |gap| > 0.3%
    gap_aligned:     bool      # gap direction aligned with sweep (down for LONG, up for SHORT)

    orb_up:          bool      # ORB directionally up (first bar bullish)
    orb_down:        bool      # ORB directionally down

    rvol_high:       bool      # sweep bar volume > 1.5x prior 10-bar avg
    rvol_low:        bool      # sweep bar volume < 0.7x prior 10-bar avg

    atr_expanding:   bool      # sweep bar range > 2x prior ATR
    atr_quiet:       bool      # sweep bar range < 0.5x prior ATR

    vwap_aligned:    bool      # LONG below VWAP or SHORT above VWAP at sweep

    dow_monday:      bool
    dow_friday:      bool
    dow_midweek:     bool      # Tue/Wed/Thu

    near_intraday_extreme: bool  # price within 5% of 20-bar intraday range extreme

    pd_range_large:  bool      # prior day range > 75th pct of rolling 20-day ranges
    pd_range_small:  bool      # prior day range < 25th pct

    # For simulation
    day_df:          object


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

def compute_vwap(day_df: pd.DataFrame, up_to_iloc: int) -> float:
    sub = day_df.iloc[:up_to_iloc + 1]
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol = sub["volume"]
    return float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(tp.mean())


# ── Event Detection + Regime Enrichment ───────────────────────────────────────

def find_setups(df: pd.DataFrame, pd_range_history: Dict) -> List[RegimeSetup]:
    """Detect sweep setups and enrich with regime signals."""
    symbol  = df["symbol"].iloc[0]
    setups: List[RegimeSetup] = []
    pd_high = pd_low = pd_close = pd_open = prior_close = None
    pd_range_rolling: List[float] = []

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if pd_high is None:
            pd_high = day["high"].max(); pd_low = day["low"].min()
            pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
            prior_close = day["close"].iloc[-1]
            pd_range_rolling.append(pd_high - pd_low)
            continue

        pd_range     = pd_high - pd_low
        prior_day_up = pd_close > pd_open
        gap_pct      = ((day["open"].iloc[0] - prior_close) / prior_close
                        if prior_close and prior_close > 0 else 0.0)

        # ORB (first 2 bars = 30 min)
        orb_bars     = day.iloc[:min(2, len(day))]
        orb_up       = bool(orb_bars.iloc[-1]["close"] > orb_bars.iloc[0]["open"]) if len(orb_bars) >= 1 else False
        orb_down     = not orb_up

        # Prior range percentile
        if len(pd_range_rolling) >= RANGE_LOOKBACK:
            pct25 = float(np.percentile(pd_range_rolling[-RANGE_LOOKBACK:], 25))
            pct75 = float(np.percentile(pd_range_rolling[-RANGE_LOOKBACK:], 75))
        else:
            pct25, pct75 = -1.0, 1e9
        pd_range_large = pd_range > pct75
        pd_range_small = pd_range < pct25

        # Day of week
        dow = day_date.weekday()   # 0=Mon, 4=Fri
        dow_monday  = (dow == 0)
        dow_friday  = (dow == 4)
        dow_midweek = (dow in (1, 2, 3))

        found = False
        for i in range(len(day) - 1):
            if found:
                break
            bar = day.iloc[i]

            for direction, check_sweep, check_conf in [
                ("SHORT",
                 lambda b, pdh=pd_high: b["high"] > pdh and b["close"] <= pdh,
                 lambda c, pdh=pd_high: c["close"] < c["open"] and c["close"] < pdh),
                ("LONG",
                 lambda b, pdl=pd_low: b["low"] < pdl and b["close"] >= pdl,
                 lambda c, pdl=pd_low: c["close"] > c["open"] and c["close"] > pdl),
            ]:
                if not check_sweep(bar):
                    continue

                # Sweep bar attributes
                if direction == "SHORT":
                    sweep_ext = float(bar["high"])
                    ov_abs    = sweep_ext - pd_high
                else:
                    sweep_ext = float(bar["low"])
                    ov_abs    = pd_low - sweep_ext
                ov_pct = ov_abs / float(pd_high if direction == "SHORT" else pd_low)

                # ── Regime signals at sweep bar time ──────────────────────
                # ATR
                past_bars   = day.iloc[:i]
                atr_vals    = []
                for j in range(1, len(past_bars)):
                    tr = max(past_bars.iloc[j]["high"] - past_bars.iloc[j]["low"],
                             abs(past_bars.iloc[j]["high"] - past_bars.iloc[j-1]["close"]),
                             abs(past_bars.iloc[j]["low"]  - past_bars.iloc[j-1]["close"]))
                    atr_vals.append(tr)
                atr_n       = min(ATR_PERIOD, len(atr_vals))
                atr_avg     = float(np.mean(atr_vals[-atr_n:])) if atr_vals else 1.0
                bar_range   = float(bar["high"] - bar["low"])
                atr_expanding = bar_range > 2.0 * atr_avg
                atr_quiet     = bar_range < 0.5 * atr_avg

                # RVOL
                vol_hist    = day.iloc[:i]["volume"].values
                vol_avg     = float(np.mean(vol_hist[-RVOL_LOOKBACK:])) if len(vol_hist) >= 3 else 1.0
                bar_vol     = float(bar["volume"])
                rvol_high   = bar_vol > 1.5 * vol_avg
                rvol_low    = bar_vol < 0.7 * vol_avg

                # VWAP
                vwap_at     = compute_vwap(day, i)
                if direction == "LONG":
                    vwap_aligned = bar["close"] < vwap_at   # swept below VWAP
                else:
                    vwap_aligned = bar["close"] > vwap_at   # swept above VWAP

                # Intraday extreme proximity (20-bar lookback)
                lookback_bars = day.iloc[max(0, i - 20):i + 1]
                if len(lookback_bars) >= 3:
                    lb_high = lookback_bars["high"].max()
                    lb_low  = lookback_bars["low"].min()
                    lb_range = lb_high - lb_low
                    if lb_range > 0:
                        if direction == "LONG":
                            near_extreme = (lb_low - bar["low"]) / lb_range < 0.05
                        else:
                            near_extreme = (bar["high"] - lb_high) / lb_range < 0.05
                    else:
                        near_extreme = True
                else:
                    near_extreme = False

                # Gap alignment
                if direction == "LONG":
                    gap_aligned = gap_pct < -0.002   # gap DOWN = aligned with LONG sweep
                else:
                    gap_aligned = gap_pct > 0.002    # gap UP = aligned with SHORT sweep

                # Find confirmation bar
                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if not check_conf(conf):
                        continue

                    setups.append(RegimeSetup(
                        symbol=symbol, date=str(day_date), direction=direction,
                        pd_level=(pd_high if direction == "SHORT" else pd_low),
                        pd_high=pd_high, pd_low=pd_low, pd_range=pd_range,
                        sweep_extreme=sweep_ext, overshoot_abs=ov_abs, overshoot_pct=ov_pct,
                        sweep_bar_iloc=i, sweep_bar_close=float(bar["close"]),
                        sweep_hour=bar["time"].hour,
                        conf_bar_iloc=i + k, conf_bar_close=float(conf["close"]),
                        conf_bar_num=k,
                        prior_day_up=prior_day_up, gap_pct=gap_pct, prior_close=float(prior_close),
                        # Regime signals
                        gap_pos=(gap_pct > 0.002),
                        gap_neg=(gap_pct < -0.002),
                        gap_abs=(abs(gap_pct) > 0.003),
                        gap_aligned=gap_aligned,
                        orb_up=orb_up, orb_down=orb_down,
                        rvol_high=rvol_high, rvol_low=rvol_low,
                        atr_expanding=atr_expanding, atr_quiet=atr_quiet,
                        vwap_aligned=vwap_aligned,
                        dow_monday=dow_monday, dow_friday=dow_friday, dow_midweek=dow_midweek,
                        near_intraday_extreme=near_extreme,
                        pd_range_large=pd_range_large, pd_range_small=pd_range_small,
                        day_df=day,
                    ))
                    found = True
                    break
                if found:
                    break

        pd_range_rolling.append(pd_high - pd_low)
        pd_high = day["high"].max(); pd_low = day["low"].min()
        pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
        prior_close = day["close"].iloc[-1]

    return setups


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_primary(s: RegimeSetup) -> float:
    """M2 + X1: conf_close entry, EOD exit — Primary cohort."""
    entry = s.conf_bar_close
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    return x1_eod_exit(entry, stop, risk, s.direction, s.day_df, s.conf_bar_iloc + 1)


def simulate_hq(s: RegimeSetup) -> float:
    """M1 + X3: sweep_close entry, Fixed 1R — High-quality cohort."""
    entry = s.sweep_bar_close
    stop  = s.sweep_extreme
    risk  = abs(entry - stop)
    return x3_fixed_1r_exit(entry, stop, risk, s.direction, s.day_df, s.sweep_bar_iloc + 1)


# ── Filter Definitions ────────────────────────────────────────────────────────

def build_filters() -> Dict[str, Callable]:
    """All regime filters — all use only pre-trade information."""
    f: Dict[str, Callable] = {
        # R1: Gap
        "gap_pos":          lambda s: s.gap_pos,
        "gap_neg":          lambda s: s.gap_neg,
        "gap_abs":          lambda s: s.gap_abs,
        "gap_flat":         lambda s: not s.gap_abs,
        "gap_aligned":      lambda s: s.gap_aligned,
        "gap_counter":      lambda s: not s.gap_aligned and s.gap_abs,
        # R2: ORB
        "orb_up":           lambda s: s.orb_up,
        "orb_down":         lambda s: s.orb_down,
        "orb_aligned_long": lambda s: s.orb_down and s.direction == "LONG",   # ORB down before PDL sweep
        "orb_aligned_shrt": lambda s: s.orb_up   and s.direction == "SHORT",  # ORB up before PDH sweep
        # R3: Prior day range
        "pd_range_large":   lambda s: s.pd_range_large,
        "pd_range_small":   lambda s: s.pd_range_small,
        "pd_range_mid":     lambda s: not s.pd_range_large and not s.pd_range_small,
        # R4: ATR expansion
        "atr_expanding":    lambda s: s.atr_expanding,
        "atr_quiet":        lambda s: s.atr_quiet,
        "atr_normal":       lambda s: not s.atr_expanding and not s.atr_quiet,
        # R5: Relative volume
        "rvol_high":        lambda s: s.rvol_high,
        "rvol_low":         lambda s: s.rvol_low,
        "rvol_normal":      lambda s: not s.rvol_high and not s.rvol_low,
        # R6: VWAP alignment
        "vwap_aligned":     lambda s: s.vwap_aligned,
        "vwap_counter":     lambda s: not s.vwap_aligned,
        # R7: Day of week
        "monday":           lambda s: s.dow_monday,
        "friday":           lambda s: s.dow_friday,
        "midweek":          lambda s: s.dow_midweek,
        "not_friday":       lambda s: not s.dow_friday,
        # R8: Intraday structure
        "near_extreme":     lambda s: s.near_intraday_extreme,
        "not_near_extreme": lambda s: not s.near_intraday_extreme,
    }
    return f


# Phase 0.5 base filters (applied before regime filters)
FILTER_A = lambda s: s.symbol == "NVDA" and s.sweep_hour not in (13, 14) and s.prior_day_up
FILTER_B = lambda s: s.direction == "LONG" and s.prior_day_up


# ── Statistics ────────────────────────────────────────────────────────────────

def stats(pnls: List[float]) -> Tuple[float, float, float]:
    """Returns (exp, wr, sharpe)."""
    arr = np.array(pnls)
    exp = float(arr.mean())
    wr  = float((arr > 0).mean())
    sh  = float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0
    return exp, wr, sh


def run_filter_scan(setups: List[RegimeSetup], sim_fn: Callable,
                    base_filter: Callable, base_label: str,
                    filters: Dict[str, Callable],
                    label: str) -> None:
    """Run all filters on the base cohort, print ranked results."""
    base_setups = [s for s in setups if base_filter(s)]
    base_pnls   = [sim_fn(s) for s in base_setups]
    b_exp, b_wr, b_sh = stats(base_pnls)
    n_base = len(base_pnls)

    print(f"\n  {'═'*110}")
    print(f"  {label}")
    print(f"  {'═'*110}")
    print(f"  Baseline ({base_label}): n={n_base}  exp={b_exp:+.3f}R  WR={b_wr:.0%}  Sharpe={b_sh:+.3f}")
    print(f"\n  ── Single Regime Filters ──────────────────────────────────────────────────────────────────────────")
    print(f"  {'Filter':<24s}  {'n':>4}  {'Exp':>8}  {'WR':>4}  {'Sharpe':>7}  {'ΔExp':>8}  {'ΔSharpe':>9}  {'note'}")
    print(f"  {'─'*110}")

    scored: List[Tuple[float, str, float, int, List[float]]] = []
    for fname, fn in filters.items():
        sub   = [s for s in base_setups if fn(s)]
        if len(sub) < MIN_TRADES:
            continue
        pnls  = [sim_fn(s) for s in sub]
        e, w, sh = stats(pnls)
        delta_e  = e - b_exp
        delta_sh = sh - b_sh
        scored.append((sh, fname, e, len(sub), pnls, delta_e, delta_sh, w))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_singles = []
    for rank, (sh, fname, e, n, pnls, de, dsh, wr) in enumerate(scored[:20]):
        flag = ""
        if dsh > MIN_SHARPE_IMPROVEMENT and de > 0:
            flag = "★ BETTER"
        elif dsh < -MIN_SHARPE_IMPROVEMENT:
            flag = "✗ WORSE"
        print(f"  {fname:<24s}  n={n:4d}  exp={e:+.3f}R  WR={wr:.0%}  Sharpe={sh:+.3f}  "
              f"ΔExp={de:+.3f}R  ΔSharpe={dsh:+.3f}  {flag}")
        if rank < TOP_N_SINGLES:
            top_singles.append((fname, sh, e, pnls))

    # ── Pair combinations ────────────────────────────────────────────────────
    print(f"\n  ── Top Pair Combinations (from top-{TOP_N_SINGLES} singles) ─────────────────────────────────────────")
    print(f"  {'Filter pair':<40s}  {'n':>4}  {'Exp':>8}  {'WR':>4}  {'Sharpe':>7}  {'ΔExp':>8}  {'ΔSharpe':>9}")
    print(f"  {'─'*110}")

    pair_scored = []
    for (n1, sh1, e1, _), (n2, sh2, e2, __) in itertools.combinations(
            [(n, sh, e, p) for n, sh, e, p in top_singles], 2):
        fn1 = filters[n1]; fn2 = filters[n2]
        sub = [s for s in base_setups if fn1(s) and fn2(s)]
        if len(sub) < MIN_TRADES:
            continue
        pnls = [sim_fn(s) for s in sub]
        e, w, sh = stats(pnls)
        pair_scored.append((sh, f"{n1} + {n2}", e, len(sub), w))

    pair_scored.sort(key=lambda x: x[0], reverse=True)
    best_pair_sharpe = b_sh
    best_pair_label  = ""
    best_pair_exp    = b_exp
    for sh, fname, e, n, wr in pair_scored[:12]:
        de  = e - b_exp
        dsh = sh - b_sh
        flag = "★★" if dsh > MIN_SHARPE_IMPROVEMENT and de > 0 else ""
        print(f"  {fname:<40s}  n={n:4d}  exp={e:+.3f}R  WR={wr:.0%}  Sharpe={sh:+.3f}  "
              f"ΔExp={de:+.3f}R  ΔSharpe={dsh:+.3f}  {flag}")
        if dsh > MIN_SHARPE_IMPROVEMENT and de > 0 and sh > best_pair_sharpe:
            best_pair_sharpe = sh
            best_pair_label  = fname
            best_pair_exp    = e

    print(f"\n  VERDICT:")
    if best_pair_label:
        dsh = best_pair_sharpe - b_sh
        print(f"    Best regime filter: {best_pair_label}")
        print(f"    Sharpe: {best_pair_sharpe:+.3f}  (+{dsh:.3f} vs baseline {b_sh:+.3f})")
        print(f"    Exp:    {best_pair_exp:+.3f}R (baseline {b_exp:+.3f}R)")
        print(f"    → Lock this filter for Phase 4 walk-forward validation")
    else:
        top_sh = scored[0][0] if scored else b_sh
        top_single_improvement = top_sh - b_sh
        if top_single_improvement > MIN_SHARPE_IMPROVEMENT:
            print(f"    No pair significantly improves on baseline. Best single: {scored[0][1]}  "
                  f"Sharpe={top_sh:+.3f}  (+{top_single_improvement:.3f})")
            print(f"    → Use best single if Sharpe improvement > {MIN_SHARPE_IMPROVEMENT}")
        else:
            print(f"    No regime filter materially improves the base filter.")
            print(f"    → Base filter (Phase 0.5) is sufficient. No additional regime filter needed.")
            print(f"    → This is GOOD: the strategy is robust without extra conditions.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase3_regime_filter.py <csv_files...>")
        sys.exit(1)

    all_setups: List[RegimeSetup] = []
    pd_range_history: Dict = {}
    for path in sys.argv[1:]:
        df   = load_csv(path)
        sym  = Path(path).stem.split("_")[0]
        evs  = find_setups(df, pd_range_history)
        print(f"  {sym:6s}  :  {len(evs):4d} setups")
        all_setups.extend(evs)

    print()
    print("=" * 115)
    print("  EVENT TYPE 2: SWEEP + REJECTION — PHASE 3 REGIME FILTER VALIDATION")
    print("  Question: Are there day-level conditions that further improve already-filtered strategies?")
    print("=" * 115)

    filters = build_filters()

    # Cohort 1: Primary — M2 + Filter B + X1 EOD
    run_filter_scan(
        all_setups, simulate_primary, FILTER_B,
        "pdl_long + prior_up", filters,
        "PRIMARY — M2 conf_close + Filter B + X1 EOD  (locked from Phase 1+2)"
    )

    # Cohort 2: High-quality — M1 + Filter A + X3 1R
    run_filter_scan(
        all_setups, simulate_hq, FILTER_A,
        "nvda + avoid_13_14 + prior_up", filters,
        "HIGH-QUALITY — M1 sweep_close + Filter A + X3 Fixed 1R  (locked from Phase 1+2)"
    )

    print()
    print("=" * 115)
    print("  OVERALL PHASE 3 SUMMARY")
    print("=" * 115)
    print("""
  Decision framework:
    1. If a regime filter improves Sharpe by > 0.05 AND maintains exp > 0 → LOCK IT
    2. If no filter clears the bar → proceed with base filter only (robust result)
    3. Locked regime filter (or none) + Phase 0.5 base filter = FINAL STRATEGY for Phase 4

  Phase 4 will validate the locked strategy via walk-forward on the 2-year dataset.
    """)


if __name__ == "__main__":
    main()
