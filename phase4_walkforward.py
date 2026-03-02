"""Phase 4: Walk-Forward Validation

Question: "Does the Phase 3 strategy work on data it hasn't seen?"

Method: Rolling walk-forward — 3-month training window, 1-month test window,
        stepped forward by 1 month.

  Fixed mode:    Lock in Phase 3 filter (has_gap + prior_day_up), no refitting.
                 This is the purest OOS test — no information from future windows.
  Adaptive mode: Refit the best filter combo on each training window, then apply
                 that exact filter (with training-derived thresholds) to test.
                 Shows whether refitting adds value or just adds overfitting.

  With current data (Jul–Dec 2025): ~2–3 windows — flagged as preliminary.
  Run fetch_alpaca_data.py for 2 years of data → 21 windows.

Success criteria (user's plan + additions):
  1.  Avg OOS expectancy > +0.03R
  2.  > 60% of windows show positive OOS expectancy
  3.  IS→OOS degradation ratio > 0.40  (OOS retains ≥40% of IS edge)
  4.  Combined OOS t-stat: p < 0.10

Additional analyses:
  - Naive benchmark: unfiltered ORB (baseline to beat)
  - Per-symbol OOS breakdown (is the edge concentrated in one stock?)
  - Bootstrap 90% CI on OOS expectancy
  - Fixed vs adaptive comparison (overfitting diagnostic)

Usage:
  # With current data only:
  python3 phase4_walkforward.py ./data/cache/*_15_mins.csv

  # After fetching 2 years of data:
  python3 fetch_alpaca_data.py
  python3 phase4_walkforward.py ./data/cache/*_15_mins.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from datetime import time as dtime
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
ATR_LOOKBACK       = 14
TRENDING_THRESHOLD = 3.8   # Phase 2 finding: day_range / session_ATR

TRAIN_MONTHS = 3
TEST_MONTHS  = 1
MIN_IS_TRADES = 10   # minimum trades in training window for adaptive filter to be valid

# Phase 3's best real-time filter — used in fixed mode
PHASE3_FILTER = "has_gap+prior_up"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBEvent:
    symbol:               str
    date:                 str
    direction:            str
    orb_high:             float
    orb_low:              float
    orb_range:            float
    orb_volume:           float
    gap_pct:              float
    breakout_close:       float
    breakout_time:        str
    breakout_bar_iloc:    int
    # Pre-market proxies
    prior_day_range:      float
    prior_day_atr:        float
    prior_day_up:         bool
    prior_day_trending:   bool
    # At-ORB proxies (real-time only — no look-ahead)
    orb_vs_prior_atr:     float
    orb_vs_prior_range:   float
    orb_vol_vs_prior_day: float   # orb_vol / prior_day_total_vol (real-time)
    breakout_bar_strength: float
    # Ground truth (EOD — never used as filter)
    actual_day_range:     float
    session_atr:          float
    is_trending:          bool
    # Structural levels
    prev_day_high:        float
    prev_day_low:         float
    # Raw bars for exit simulation
    day_df:               object


@dataclass
class ORBTrade:
    event:        ORBEvent
    entry_price:  float
    stop_price:   float
    risk:         float
    remaining:    pd.DataFrame
    eod_price:    float
    atr:          float
    vwap:         np.ndarray


@dataclass
class ExitResult:
    method:      str
    pnl_r:       float
    bars_held:   int
    exit_reason: str
    mfe_r:       float
    mae_r:       float


@dataclass
class WindowResult:
    window_idx:  int
    train_start: str
    train_end:   str
    test_month:  str        # "YYYY-MM"
    is_filter:   str
    is_exp:      float
    is_n:        int
    oos_exp:     float
    oos_n:       int
    oos_pnls:    List[float]   = field(default_factory=list)
    oos_symbols: List[str]     = field(default_factory=list)
    is_positive: bool          = False


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

def compute_atr(bars: pd.DataFrame, n: int = ATR_LOOKBACK) -> float:
    if len(bars) == 0:
        return 0.01
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    trs = [h[0] - l[0]]
    for i in range(1, len(bars)):
        trs.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    return float(np.mean(trs[-n:])) if trs else 0.01


def compute_vwap(day_df: pd.DataFrame) -> np.ndarray:
    tp      = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
    cum_vol = day_df["volume"].cumsum()
    vwap    = (tp * day_df["volume"]).cumsum() / cum_vol
    return vwap.values


def r_val(entry: float, exit_p: float, risk: float, direction: str) -> float:
    return (exit_p - entry) / risk if direction == "LONG" else (entry - exit_p) / risk


def excursion(bars: pd.DataFrame, entry: float,
              risk: float, direction: str) -> Tuple[float, float]:
    if bars.empty:
        return 0.0, 0.0
    if direction == "LONG":
        return ((bars["high"].max() - entry) / risk,
                (bars["low"].min() - entry) / risk)
    return ((entry - bars["low"].min()) / risk,
            (entry - bars["high"].max()) / risk)


# ── Event Detection ───────────────────────────────────────────────────────────

def find_orb_events(df: pd.DataFrame) -> List[ORBEvent]:
    orb_end  = dtime(10, 0)
    symbol   = df["symbol"].iloc[0]
    events: List[ORBEvent] = []

    pd_high = pd_low = pd_close = pd_open = pd_volume = pd_df = prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        pd_atr      = compute_atr(pd_df) if (pd_df is not None and len(pd_df) > 2) else 0.01
        pd_range    = (pd_high - pd_low) if (pd_high is not None and pd_low is not None) else 0.01
        pd_up       = (pd_close > pd_open) if (pd_close is not None and pd_open is not None) else True
        pd_trending = (pd_range / pd_atr) >= 3.0 if pd_atr > 0 else False
        pd_vol      = pd_volume if (pd_volume and pd_volume > 0) else 1.0

        def _save_prior():
            nonlocal pd_high, pd_low, pd_close, pd_open, pd_volume, pd_df, prior_close
            pd_high   = day["high"].max()
            pd_low    = day["low"].min()
            pd_close  = day["close"].iloc[-1]
            pd_open   = day["open"].iloc[0]
            pd_volume = day["volume"].sum()
            pd_df     = day
            prior_close = day["close"].iloc[-1]

        if len(day) < 4:
            if len(day) > 0:
                _save_prior()
            continue

        orb_bars = day[day["time"] < orb_end]
        if len(orb_bars) < 1:
            _save_prior()
            continue

        orb_high   = orb_bars["high"].max()
        orb_low    = orb_bars["low"].min()
        orb_range  = orb_high - orb_low
        orb_volume = orb_bars["volume"].sum()

        if orb_range <= 0:
            _save_prior()
            continue

        day_open   = day["open"].iloc[0]
        day_volume = day["volume"].sum()
        gap_pct    = ((day_open - prior_close) / prior_close
                      if prior_close and prior_close > 0 else 0.0)

        post_orb = day[day["time"] >= orb_end]
        if len(post_orb) < 2:
            _save_prior()
            continue

        for idx, row in post_orb.iterrows():
            close = row["close"]
            if close > orb_high:
                direction = "LONG"
            elif close < orb_low:
                direction = "SHORT"
            else:
                continue

            b_iloc      = day.index.get_loc(idx)
            session_atr = compute_atr(day.iloc[:b_iloc], n=ATR_LOOKBACK)
            if session_atr <= 0:
                session_atr = pd_atr if pd_atr > 0 else orb_range / 2

            actual_day_range = day["high"].max() - day["low"].min()
            is_trending      = (actual_day_range / session_atr) >= TRENDING_THRESHOLD

            bar_range = row["high"] - row["low"]
            if direction == "LONG" and bar_range > 0:
                bb_strength = (close - row["low"]) / bar_range
            elif direction == "SHORT" and bar_range > 0:
                bb_strength = (row["high"] - close) / bar_range
            else:
                bb_strength = 0.5

            orb_vol_rt = orb_volume / pd_vol if pd_vol > 0 else 0.5

            events.append(ORBEvent(
                symbol=symbol, date=str(day_date), direction=direction,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_volume=orb_volume, gap_pct=gap_pct,
                breakout_close=close, breakout_time=str(row["time"]),
                breakout_bar_iloc=b_iloc,
                prior_day_range=pd_range, prior_day_atr=pd_atr,
                prior_day_up=pd_up, prior_day_trending=pd_trending,
                orb_vs_prior_atr=orb_range / pd_atr if pd_atr > 0 else 1.0,
                orb_vs_prior_range=orb_range / pd_range if pd_range > 0 else 0.5,
                orb_vol_vs_prior_day=orb_vol_rt,
                breakout_bar_strength=bb_strength,
                actual_day_range=actual_day_range,
                session_atr=session_atr,
                is_trending=is_trending,
                prev_day_high=pd_high if pd_high else orb_high + orb_range,
                prev_day_low=pd_low   if pd_low  else orb_low  - orb_range,
                day_df=day,
            ))
            break   # one event per day per symbol

        _save_prior()

    return events


def build_trade(ev: ORBEvent) -> Optional[ORBTrade]:
    day   = ev.day_df
    b     = ev.breakout_bar_iloc
    entry = ev.breakout_close
    stop  = ev.orb_low if ev.direction == "LONG" else ev.orb_high
    risk  = abs(entry - stop)
    if risk <= 0:
        return None
    remaining = day.iloc[b + 1:]
    if remaining.empty:
        return None
    atr = compute_atr(day.iloc[:b], n=ATR_LOOKBACK)
    if atr <= 0:
        atr = ev.orb_range / 2
    return ORBTrade(
        event=ev, entry_price=entry, stop_price=stop, risk=risk,
        remaining=remaining, eod_price=day["close"].iloc[-1],
        atr=atr, vwap=compute_vwap(day),
    )


# ── Exit Simulators ───────────────────────────────────────────────────────────

def exit_vwap_capped(t: ORBTrade) -> ExitResult:
    """Phase 3 primary exit: VWAP cross with −1R hard stop cap."""
    e, rk, d  = t.entry_price, t.risk, t.event.direction
    hard_stop = e - rk if d == "LONG" else e + rk
    day       = t.event.day_df

    for i, (idx, bar) in enumerate(t.remaining.iterrows()):
        bar_iloc = day.index.get_loc(idx)

        if d == "LONG" and bar["low"] <= hard_stop:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap+1Rcap", r_val(e, hard_stop, rk, d), i+1, "stop", mfe, mae)
        if d == "SHORT" and bar["high"] >= hard_stop:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap+1Rcap", r_val(e, hard_stop, rk, d), i+1, "stop", mfe, mae)

        if bar_iloc < len(t.vwap):
            vwap_val = t.vwap[bar_iloc]
            if (d == "LONG"  and bar["close"] < vwap_val) or \
               (d == "SHORT" and bar["close"] > vwap_val):
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult("vwap+1Rcap", r_val(e, bar["close"], rk, d), i+1, "vwap", mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("vwap+1Rcap", r_val(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


# ── Filter Definitions ────────────────────────────────────────────────────────

# Fixed-threshold filters (no training-data context needed)
FIXED_FILTERS: Dict[str, Callable[[ORBEvent], bool]] = {
    "unfiltered":         lambda e: True,
    "has_gap":            lambda e: abs(e.gap_pct) > 0.002,
    "prior_up":           lambda e: e.prior_day_up,
    "prior_trending":     lambda e: e.prior_day_trending,
    "strong_breakout":    lambda e: e.breakout_bar_strength > 0.65,
    "has_gap+prior_up":   lambda e: abs(e.gap_pct) > 0.002 and e.prior_day_up,
    "has_gap+prior_trend":lambda e: abs(e.gap_pct) > 0.002 and e.prior_day_trending,
    "prior_up+strong_bb": lambda e: e.prior_day_up and e.breakout_bar_strength > 0.65,
    "gap+trend+strong_bb":lambda e: (abs(e.gap_pct) > 0.002 and e.prior_day_trending
                                     and e.breakout_bar_strength > 0.65),
}

# Context-dependent filters (threshold derived from training data)
CONTEXT_FILTER_NAMES = ["low_orb_vol_rt", "has_gap+prior_up+low_vol"]


def _apply_filter(ev: ORBEvent, name: str, ctx: Dict) -> bool:
    if name in FIXED_FILTERS:
        return FIXED_FILTERS[name](ev)
    med_vol = ctx.get("orb_vol_median", 1.0)
    if name == "low_orb_vol_rt":
        return ev.orb_vol_vs_prior_day < med_vol
    if name == "has_gap+prior_up+low_vol":
        return abs(ev.gap_pct) > 0.002 and ev.prior_day_up and ev.orb_vol_vs_prior_day < med_vol
    return True  # fallback


def filter_events(events: List[ORBEvent], name: str, ctx: Dict) -> List[ORBEvent]:
    return [e for e in events if _apply_filter(e, name, ctx)]


def build_context(train_events: List[ORBEvent]) -> Dict:
    """Compute dynamic thresholds from training events only."""
    if not train_events:
        return {}
    vols   = [e.orb_vol_vs_prior_day for e in train_events]
    ranges = [e.orb_range for e in train_events]
    return {
        "orb_vol_median": float(np.median(vols)),
        "orb_range_p20":  float(np.percentile(ranges, 20)),
    }


ALL_FILTER_NAMES = list(FIXED_FILTERS.keys()) + CONTEXT_FILTER_NAMES


def evaluate_filter_on(events: List[ORBEvent], name: str, ctx: Dict) -> Tuple[float, int]:
    """Returns (expectancy, n) for a filter applied to events."""
    filtered = filter_events(events, name, ctx)
    trades   = [t for e in filtered for t in [build_trade(e)] if t]
    pnls     = [exit_vwap_capped(t).pnl_r for t in trades]
    if len(pnls) < MIN_IS_TRADES:
        return -999.0, len(pnls)
    return float(np.mean(pnls)), len(pnls)


def select_best_filter(train_events: List[ORBEvent]) -> Tuple[str, float, int, Dict]:
    """Adaptive mode: pick the best filter on training data.

    Returns (filter_name, IS_expectancy, IS_n, context_dict).
    The context_dict must be passed when applying the filter to test data
    so dynamic thresholds (derived from training data) are preserved.
    """
    ctx      = build_context(train_events)
    best_name = "unfiltered"
    best_exp, best_n = evaluate_filter_on(train_events, "unfiltered", ctx)

    for name in ALL_FILTER_NAMES:
        if name == "unfiltered":
            continue
        exp, n = evaluate_filter_on(train_events, name, ctx)
        if n >= MIN_IS_TRADES and exp > best_exp:
            best_exp, best_n, best_name = exp, n, name

    return best_name, best_exp, best_n, ctx


# ── Walk-Forward Engine ───────────────────────────────────────────────────────

def define_windows(all_events: List[ORBEvent]) -> List[Tuple[str, str, str, str]]:
    """Generate (train_start, train_end, test_start, test_end) date strings.

    Rolling: step by TEST_MONTHS.  Each window: TRAIN_MONTHS train + TEST_MONTHS test.
    """
    if not all_events:
        return []

    all_dates = sorted(set(e.date for e in all_events))
    first = pd.Timestamp(all_dates[0])
    last  = pd.Timestamp(all_dates[-1])

    windows = []
    cur = pd.Timestamp(first.year, first.month, 1)  # start of first full calendar month

    while True:
        train_start = cur
        test_start  = cur + pd.DateOffset(months=TRAIN_MONTHS)
        test_end    = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

        if test_end > last:
            break

        train_end = test_start - pd.DateOffset(days=1)

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        cur += pd.DateOffset(months=TEST_MONTHS)

    return windows


def run_window(win_idx: int, all_events: List[ORBEvent],
               window: Tuple[str, str, str, str],
               mode: str) -> Optional[WindowResult]:
    """Execute one walk-forward window.  mode = 'fixed' | 'adaptive'."""
    tr_start, tr_end, te_start, te_end = window

    train_evs = [e for e in all_events if tr_start <= e.date <= tr_end]
    test_evs  = [e for e in all_events if te_start <= e.date <= te_end]

    if not train_evs or not test_evs:
        return None

    if mode == "fixed":
        filt_name = PHASE3_FILTER
        ctx       = {}   # Phase 3 fixed filter uses no dynamic thresholds
        is_exp, is_n = evaluate_filter_on(train_evs, filt_name, ctx)
    else:
        filt_name, is_exp, is_n, ctx = select_best_filter(train_evs)

    # OOS: apply filter with training-derived context → never touch test data for selection
    oos_filtered = filter_events(test_evs, filt_name, ctx)
    oos_trades   = [t for e in oos_filtered for t in [build_trade(e)] if t]

    if not oos_trades:
        return None

    oos_pnls    = [exit_vwap_capped(t).pnl_r for t in oos_trades]
    oos_symbols = [t.event.symbol for t in oos_trades]
    oos_exp     = float(np.mean(oos_pnls))
    test_month  = te_start[:7]   # "YYYY-MM"

    return WindowResult(
        window_idx=win_idx,
        train_start=tr_start, train_end=tr_end,
        test_month=test_month,
        is_filter=filt_name,
        is_exp=is_exp, is_n=is_n,
        oos_exp=oos_exp, oos_n=len(oos_pnls),
        oos_pnls=oos_pnls, oos_symbols=oos_symbols,
        is_positive=(oos_exp > 0),
    )


def walk_forward(all_events: List[ORBEvent], mode: str) -> List[WindowResult]:
    windows = define_windows(all_events)
    return [r for i, w in enumerate(windows)
            for r in [run_window(i + 1, all_events, w, mode)] if r]


# ── Statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(pnls: List[float], n_boot: int = 2000, ci: float = 0.90) -> Tuple[float, float]:
    if len(pnls) < 2:
        return 0.0, 0.0
    arr   = np.array(pnls)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def t_test(pnls: List[float]) -> Tuple[float, float]:
    """One-sample t-test vs H0: mean = 0. Returns (t, p)."""
    if len(pnls) < 3:
        return 0.0, 1.0
    arr = np.array(pnls)
    t   = float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(len(arr))))
    p   = float(2 * scipy_stats.t.sf(abs(t), df=len(arr) - 1))
    return t, p


# ── Printing ──────────────────────────────────────────────────────────────────

W = 100   # line width

def hdr(text: str):
    print(f"\n  {'═' * W}")
    print(f"  {text}")
    print(f"  {'═' * W}")


def print_window_table(results: List[WindowResult], label: str):
    print(f"\n  {'─' * W}")
    print(f"  WINDOW TABLE  [{label}]")
    print(f"  {'─' * W}")
    print(f"  {'#':>3}  {'Train':22s}  {'Test':>7}  {'Filter':26s}  "
          f"{'IS Exp':>7} {'IS n':>5}   {'OOS Exp':>7} {'OOS n':>5}  Pos")
    print(f"  {'─' * W}")

    for r in results:
        is_str  = f"{r.is_exp:+.3f}R" if r.is_exp > -100 else "  n/a  "
        oos_str = f"{r.oos_exp:+.3f}R"
        pos     = "✓" if r.is_positive else "✗"
        print(f"  {r.window_idx:>3}  {r.train_start} → {r.train_end}  {r.test_month}  "
              f"  {r.is_filter:26s}  {is_str:>7} {r.is_n:>5}   {oos_str:>7} {r.oos_n:>5}  {pos}")

    print(f"  {'─' * W}")


def print_aggregate(results: List[WindowResult], label: str) -> dict:
    all_pnls = [p for r in results for p in r.oos_pnls]
    if not all_pnls:
        print("  No OOS trades.")
        return {}

    avg_oos = float(np.mean(all_pnls))
    std_oos = float(np.std(all_pnls))
    pos_win = sum(1 for r in results if r.is_positive)
    pct_pos = pos_win / len(results) * 100

    valid_is = [r.is_exp for r in results if r.is_exp > -100]
    avg_is   = float(np.mean(valid_is)) if valid_is else 0.0
    ratio    = avg_oos / avg_is if avg_is > 0.001 else 0.0

    lo, hi   = bootstrap_ci(all_pnls)
    t, p     = t_test(all_pnls)

    c1 = avg_oos > 0.03
    c2 = pct_pos > 60
    c3 = ratio   > 0.40
    c4 = p       < 0.10

    print(f"\n  AGGREGATE  [{label}]")
    print(f"  {'─' * 60}")
    print(f"  Windows:              {len(results)}")
    print(f"  Total OOS trades:     {len(all_pnls)}")
    print(f"  Avg OOS expectancy:   {avg_oos:+.3f}R  (std {std_oos:.3f})")
    print(f"  Avg IS  expectancy:   {avg_is:+.3f}R")
    print(f"  IS→OOS ratio:         {ratio:.2f}  ({ratio*100:.0f}% of IS edge survives OOS)")
    print(f"  Positive windows:     {pos_win} / {len(results)}  ({pct_pos:.0f}%)")
    print(f"  Bootstrap 90% CI:     [{lo:+.3f}R, {hi:+.3f}R]")
    print(f"  t-stat (OOS≠0):       t={t:+.2f},  p={p:.3f}  (n={len(all_pnls)} trades)")
    print()
    print(f"  SUCCESS CRITERIA")
    print(f"  {'─' * 60}")
    print(f"  [{'✓' if c1 else '✗'}]  Avg OOS > +0.03R           {avg_oos:+.3f}R")
    print(f"  [{'✓' if c2 else '✗'}]  > 60% positive windows     {pct_pos:.0f}%  ({pos_win}/{len(results)})")
    print(f"  [{'✓' if c3 else '✗'}]  IS→OOS ratio > 0.40        {ratio:.2f}")
    print(f"  [{'✓' if c4 else '✗'}]  p < 0.10                   p={p:.3f}  (need more windows for significance)")
    passed = sum([c1, c2, c3, c4])
    print(f"\n  Criteria passed: {passed}/4")

    return {"avg_oos": avg_oos, "pct_pos": pct_pos, "ratio": ratio,
            "t": t, "p": p, "passed": passed, "n": len(all_pnls),
            "n_windows": len(results)}


def print_per_symbol(results: List[WindowResult]):
    sym_pnls: Dict[str, List[float]] = {}
    for r in results:
        for sym, pnl in zip(r.oos_symbols, r.oos_pnls):
            sym_pnls.setdefault(sym, []).append(pnl)

    if not sym_pnls:
        return

    print(f"\n  PER-SYMBOL OOS BREAKDOWN (all windows combined)")
    print(f"  {'─' * 55}")
    print(f"  {'Symbol':<8}  {'n':>4}  {'Exp':>7}  {'WR':>5}  {'Std':>5}  OK?")
    print(f"  {'─' * 55}")
    for sym in sorted(sym_pnls):
        p = sym_pnls[sym]
        if not p:
            continue
        exp = np.mean(p)
        wr  = sum(1 for x in p if x > 0) / len(p) * 100
        std = np.std(p) if len(p) > 1 else 0.0
        flag = "✓" if exp > 0.02 else ("✗" if exp < -0.02 else "~")
        print(f"  {sym:<8}  {len(p):>4}  {exp:+.3f}R  {wr:>4.0f}%  {std:.3f}  {flag}")
    print(f"  {'─' * 55}")


def print_verdict(fixed: dict, adaptive: dict, n_win: int, n_months: int):
    hdr("VERDICT")

    if n_win < 6:
        print(f"\n  ⚠  PRELIMINARY  —  only {n_win} windows from ~{n_months} months of data.")
        print(f"     Statistical significance requires 12+ windows (≥16 months).")
        print(f"     Run fetch_alpaca_data.py to get 2 years of data → 21 windows:")
        print(f"       python3 fetch_alpaca_data.py")
        print(f"       python3 phase4_walkforward.py ./data/cache/*.csv")
        print()

    fp = fixed.get("passed", 0)
    ap = adaptive.get("passed", 0)

    if fp >= 3:
        verdict = "PASS — Phase 3 filter generalises to unseen data"
    elif fp == 2:
        verdict = "MARGINAL — partial OOS evidence, needs more windows"
    else:
        verdict = "INCONCLUSIVE / FAIL — insufficient OOS evidence"

    print(f"  Fixed filter ({PHASE3_FILTER}): {fp}/4 criteria  →  {verdict}")
    print(f"  Adaptive filter:                 {ap}/4 criteria", end="")
    if ap > fp:
        print(f"  ← adaptive outperforms fixed (unexpected, may indicate over-test)")
    elif ap == fp:
        print(f"  ← no benefit from adaptive refitting")
    else:
        print(f"  ← fixed filter outperforms adaptive (adaptive overfits train window)")

    print(f"\n  ─ Confirmed strategy definition ─")
    print(f"  Entry:  Close of first 15-min bar beyond ORB range  (bar-close, market order)")
    print(f"  Filter: abs(gap_pct) > 0.2%  AND  prior_day_up")
    print(f"  Stop:   Opposite ORB boundary  (max −1R cap)")
    print(f"  Exit:   VWAP cross  +  hard −1R stop")
    print(f"  {'═' * W}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase4_walkforward.py ./data/cache/*.csv")
        sys.exit(1)

    np.random.seed(42)

    # ── Load data ──────────────────────────────────────────────────────────────
    hdr("PHASE 4: WALK-FORWARD VALIDATION")

    all_events: List[ORBEvent] = []
    loaded: List[str] = []

    for path in sys.argv[1:]:
        sym = Path(path).stem.split("_")[0]
        try:
            df  = load_csv(path)
            evs = find_orb_events(df)
            all_events.extend(evs)
            loaded.append(sym)
            print(f"  Loaded {sym:6s}: {len(evs):4d} ORB events")
        except Exception as ex:
            print(f"  ERROR loading {path}: {ex}")

    if not all_events:
        print("No events found. Check data files.")
        sys.exit(1)

    all_dates = sorted(set(e.date for e in all_events))
    windows   = define_windows(all_events)

    # Estimate months spanned
    first_ts  = pd.Timestamp(all_dates[0])
    last_ts   = pd.Timestamp(all_dates[-1])
    n_months  = (last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month) + 1

    print(f"\n  Total ORB events: {len(all_events)}")
    print(f"  Date range:       {all_dates[0]} → {all_dates[-1]}  (~{n_months} months)")
    print(f"  Symbols:          {', '.join(sorted(set(loaded)))}")
    print(f"  WF windows:       {len(windows)}  ({TRAIN_MONTHS}mo train · {TEST_MONTHS}mo test · rolling)")

    if len(windows) == 0:
        print(f"\n  ERROR: Not enough data for even one walk-forward window.")
        print(f"  Need ≥ {TRAIN_MONTHS + TEST_MONTHS} calendar months.  Currently have ~{n_months}.")
        print(f"  Run: python3 fetch_alpaca_data.py   (fetches Jan 2024 – Dec 2025)")
        sys.exit(1)

    if len(windows) < 4:
        print(f"\n  ⚠  Only {len(windows)} windows — results are preliminary.")
        print(f"     Run fetch_alpaca_data.py for 2 years of data (21 windows).")

    # ── Naive benchmark ────────────────────────────────────────────────────────
    hdr("NAIVE BENCHMARK  (unfiltered ORB — baseline to beat)")
    all_trades  = [t for e in all_events for t in [build_trade(e)] if t]
    naive_pnls  = [exit_vwap_capped(t).pnl_r for t in all_trades]
    t_n, p_n    = t_test(naive_pnls)
    if naive_pnls:
        avg_n = np.mean(naive_pnls)
        wr_n  = sum(1 for p in naive_pnls if p > 0) / len(naive_pnls) * 100
        print(f"  All {len(naive_pnls)} events  |  exp={avg_n:+.3f}R  WR={wr_n:.0f}%  "
              f"t={t_n:+.2f} p={p_n:.3f}")
        print(f"  (The Phase 3 filter should show meaningfully higher expectancy than this.)")

    # ── Phase 3 filter check on full sample ───────────────────────────────────
    hdr(f"FULL-SAMPLE CHECK  (Phase 3 filter on all data — for reference)")
    fs_evs    = filter_events(all_events, PHASE3_FILTER, {})
    fs_trades = [t for e in fs_evs for t in [build_trade(e)] if t]
    fs_pnls   = [exit_vwap_capped(t).pnl_r for t in fs_trades]
    t_fs, p_fs = t_test(fs_pnls)
    if fs_pnls:
        print(f"  Filter: {PHASE3_FILTER}  |  n={len(fs_pnls)}  "
              f"exp={np.mean(fs_pnls):+.3f}R  WR={sum(1 for p in fs_pnls if p>0)/len(fs_pnls)*100:.0f}%  "
              f"t={t_fs:+.2f} p={p_fs:.3f}")
        print(f"  Note: this is IN-SAMPLE — walk-forward OOS results below are the honest test.")

    # ── Fixed mode walk-forward ───────────────────────────────────────────────
    hdr(f"FIXED FILTER WALK-FORWARD  (Phase 3: {PHASE3_FILTER}, no refitting)")
    fixed_results = walk_forward(all_events, mode="fixed")

    if not fixed_results:
        print("  No walk-forward windows produced results.")
    else:
        print_window_table(fixed_results, f"Fixed: {PHASE3_FILTER}")
        fixed_stats = print_aggregate(fixed_results, "Fixed Filter")
        print_per_symbol(fixed_results)

    # ── Adaptive mode walk-forward ────────────────────────────────────────────
    hdr("ADAPTIVE FILTER WALK-FORWARD  (refit best filter on each training window)")
    adaptive_results = walk_forward(all_events, mode="adaptive")

    if not adaptive_results:
        print("  No walk-forward windows produced results.")
        adaptive_stats = {}
    else:
        print_window_table(adaptive_results, "Adaptive (refit per window)")
        adaptive_stats = print_aggregate(adaptive_results, "Adaptive Filter")

        # Show which filter each window picked
        print(f"\n  FILTER SELECTION STABILITY (adaptive mode)")
        print(f"  {'─' * 55}")
        for r in adaptive_results:
            same = "same as Phase3" if r.is_filter == PHASE3_FILTER else "different"
            print(f"    Win {r.window_idx:>2}  test={r.test_month}  "
                  f"best IS filter: {r.is_filter:30s}  [{same}]")
        print(f"  {'─' * 55}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print_verdict(
        fixed_stats    if fixed_results    else {},
        adaptive_stats if adaptive_results else {},
        len(fixed_results),
        n_months,
    )


if __name__ == "__main__":
    main()
