"""Phase 3: Regime Filter — Trending Day Identification

Question: Can we identify trending days BEFORE or AT the ORB breakout,
          so we only trade on days with high directional follow-through?

Phase 2 finding:
  Trending days (range ≥ 3.8x ATR): +0.352R, 71% WR
  Choppy days  (range < 3.8x ATR):  −0.132R, 44% WR

Problem: The 3.8x ratio is only known at END of day. Need a PROXY
         available at ORB breakout time (~10:00 AM).

⚠️  LOOK-AHEAD BUG IN PHASE 0.5:
    'low ORB vol ratio' used orb_vol / day_vol. day_vol is only known at
    4 PM — not available when we place the trade at 10 AM. This phase
    replaces it with orb_vol / prior_day_total_vol (available pre-market).
    Results may differ from Phase 0.5.

Proxies tested (grouped by when they are available):

  PRE-MARKET (known before 9:30 AM):
    P1. Gap size — |gap_pct| > threshold
    P2. Prior day direction — was prior day bullish?
    P3. Prior day range / prior day ATR — was yesterday itself trending?

  AT-ORB-TIME (known at 10:00 AM after ORB forms):
    P4. ORB range / prior day ATR  — today's ORB width relative to normal vol
    P5. ORB range / prior day range — ORB as fraction of yesterday's full range
    P6. ORB vol / prior day total vol [REAL-TIME fix for look-ahead filter]
    P7. Breakout bar strength — (close-low)/(high-low), 0=weak, 1=strong close
    P8. Composite score — combination of P1+P4+P6

Secondary filters from Phase 0.5 (now using real-time versions only):
    - Has gap > 0.2%
    - Prior day was UP
    - Not narrow ORB (orb_range >= P20)
    - Low ORB vol vs prior day (P6, real-time)

Exit methods (Phase 2 winners):
    A. VWAP cross + hard −1R stop cap
    B. ATR 1.0x trailing stop (best Sharpe from Phase 2)

Usage:
  python3 phase3_regime_filter.py ./data/cache/AAPL_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/NVDA_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/MSFT_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/AMZN_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/GOOG_2025-07-01_2025-12-1_15_mins.csv
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
MIN_TRADES         = 10
ATR_LOOKBACK       = 14
TRENDING_THRESHOLD = 3.8   # from Phase 2: day_range / session_atr


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBEvent:
    # ── Identity ──────────────────────────────────────────────────────────────
    symbol:               str
    date:                 str
    direction:            str

    # ── ORB basics ────────────────────────────────────────────────────────────
    orb_high:             float
    orb_low:              float
    orb_range:            float
    orb_volume:           float
    gap_pct:              float
    breakout_close:       float
    breakout_time:        str
    breakout_bar_iloc:    int

    # ── Pre-market proxies (P1-P3, available before 9:30) ─────────────────────
    prior_day_range:      float    # prior day high - low
    prior_day_atr:        float    # ATR of prior day's 15-min bars
    prior_day_up:         bool     # prior day close > prior day open
    prior_day_trending:   bool     # prior_day_range / prior_day_atr >= 3.0

    # ── At-ORB-time proxies (P4-P7, available at 10:00 AM) ────────────────────
    orb_vs_prior_atr:     float    # orb_range / prior_day_atr   [P4]
    orb_vs_prior_range:   float    # orb_range / prior_day_range  [P5]
    orb_vol_vs_prior_day: float    # orb_vol / prior_day_total_vol [P6 — real-time]
    orb_vol_ratio_la:     float    # orb_vol / day_vol [LOOK-AHEAD — for comparison only]
    breakout_bar_strength: float   # (close-low)/(high-low) for LONG [P7]

    # ── Ground truth labels (EOD only — never used as filters, only for scoring) ──
    actual_day_range:     float
    session_atr:          float    # ATR at entry time
    is_trending:          bool     # actual_day_range / session_atr >= 3.8

    # ── Prior day structural levels ───────────────────────────────────────────
    prev_day_high:        float
    prev_day_low:         float

    # ── Raw data ──────────────────────────────────────────────────────────────
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


def excursion(bars: pd.DataFrame, entry: float, risk: float, direction: str) -> Tuple[float, float]:
    if bars.empty:
        return 0.0, 0.0
    if direction == "LONG":
        return (bars["high"].max() - entry) / risk, (bars["low"].min() - entry) / risk
    else:
        return (entry - bars["low"].min()) / risk, (entry - bars["high"].max()) / risk


# ── Event Detection ───────────────────────────────────────────────────────────

def find_orb_events(df: pd.DataFrame) -> List[ORBEvent]:
    orb_end = dtime(10, 0)
    symbol  = df["symbol"].iloc[0]
    events  = []

    # Prior day tracking
    pd_high   = None
    pd_low    = None
    pd_close  = None
    pd_open   = None
    pd_volume = None
    pd_df     = None      # prior day's full DataFrame
    prior_close = None    # for gap computation

    for day_date in sorted(df["date"].unique()):
        day = df[df["date"] == day_date].sort_values("timestamp").reset_index(drop=True)

        # Compute prior day features
        pd_atr       = compute_atr(pd_df) if pd_df is not None and len(pd_df) > 2 else 0.01
        pd_range     = (pd_high - pd_low)  if pd_high and pd_low else 0.01
        pd_up        = (pd_close > pd_open) if pd_close and pd_open else True
        pd_trending  = (pd_range / pd_atr) >= 3.0 if pd_atr > 0 else False
        pd_vol       = pd_volume if pd_volume and pd_volume > 0 else 1.0

        if len(day) < 4:
            if len(day) > 0:
                pd_high   = day["high"].max()
                pd_low    = day["low"].min()
                pd_close  = day["close"].iloc[-1]
                pd_open   = day["open"].iloc[0]
                pd_volume = day["volume"].sum()
                pd_df     = day
                prior_close = day["close"].iloc[-1]
            continue

        orb_bars   = day[day["time"] < orb_end]
        if len(orb_bars) < 1:
            pd_high = day["high"].max(); pd_low = day["low"].min()
            pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
            pd_volume = day["volume"].sum(); pd_df = day; prior_close = day["close"].iloc[-1]
            continue

        orb_high   = orb_bars["high"].max()
        orb_low    = orb_bars["low"].min()
        orb_range  = orb_high - orb_low
        orb_volume = orb_bars["volume"].sum()

        if orb_range <= 0:
            pd_high = day["high"].max(); pd_low = day["low"].min()
            pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
            pd_volume = day["volume"].sum(); pd_df = day; prior_close = day["close"].iloc[-1]
            continue

        day_open   = day["open"].iloc[0]
        day_volume = day["volume"].sum()
        gap_pct    = (day_open - prior_close) / prior_close if prior_close and prior_close > 0 else 0.0

        post_orb = day[day["time"] >= orb_end]
        if len(post_orb) < 2:
            pd_high = day["high"].max(); pd_low = day["low"].min()
            pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
            pd_volume = day["volume"].sum(); pd_df = day; prior_close = day["close"].iloc[-1]
            continue

        for idx, row in post_orb.iterrows():
            close = row["close"]
            if close > orb_high:
                direction = "LONG"
            elif close < orb_low:
                direction = "SHORT"
            else:
                continue

            b_iloc = day.index.get_loc(idx)

            # Session ATR at entry time (for ground-truth trending label)
            session_atr = compute_atr(day.iloc[:b_iloc], n=ATR_LOOKBACK)
            if session_atr <= 0:
                session_atr = pd_atr if pd_atr > 0 else orb_range / 2

            # Ground truth (EOD, never used as filter)
            actual_day_range = day["high"].max() - day["low"].min()
            is_trending      = (actual_day_range / session_atr) >= TRENDING_THRESHOLD

            # Breakout bar strength: how close did bar close to its extreme?
            bar_range = row["high"] - row["low"]
            if direction == "LONG" and bar_range > 0:
                bb_strength = (close - row["low"]) / bar_range
            elif direction == "SHORT" and bar_range > 0:
                bb_strength = (row["high"] - close) / bar_range
            else:
                bb_strength = 0.5

            # Real-time ORB vol proxy: vs prior day total vol (no look-ahead)
            orb_vol_rt   = orb_volume / pd_vol if pd_vol > 0 else 0.5
            # Look-ahead version (for comparison flagging only)
            orb_vol_la   = orb_volume / day_volume if day_volume > 0 else 0.5

            events.append(ORBEvent(
                symbol=symbol, date=str(day_date), direction=direction,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_volume=orb_volume, gap_pct=gap_pct,
                breakout_close=close, breakout_time=str(row["time"]),
                breakout_bar_iloc=b_iloc,
                # Pre-market proxies
                prior_day_range=pd_range, prior_day_atr=pd_atr,
                prior_day_up=pd_up, prior_day_trending=pd_trending,
                # At-ORB proxies
                orb_vs_prior_atr=orb_range / pd_atr if pd_atr > 0 else 1.0,
                orb_vs_prior_range=orb_range / pd_range if pd_range > 0 else 0.5,
                orb_vol_vs_prior_day=orb_vol_rt,
                orb_vol_ratio_la=orb_vol_la,
                breakout_bar_strength=bb_strength,
                # Ground truth
                actual_day_range=actual_day_range,
                session_atr=session_atr,
                is_trending=is_trending,
                # Structural levels
                prev_day_high=pd_high or orb_high + orb_range,
                prev_day_low=pd_low  or orb_low  - orb_range,
                day_df=day,
            ))
            break

        pd_high = day["high"].max(); pd_low = day["low"].min()
        pd_close = day["close"].iloc[-1]; pd_open = day["open"].iloc[0]
        pd_volume = day["volume"].sum(); pd_df = day; prior_close = day["close"].iloc[-1]

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
    atr  = compute_atr(day.iloc[:b], n=ATR_LOOKBACK)
    if atr <= 0:
        atr = ev.orb_range / 2
    return ORBTrade(
        event=ev, entry_price=entry, stop_price=stop, risk=risk,
        remaining=remaining, eod_price=day["close"].iloc[-1],
        atr=atr, vwap=compute_vwap(day),
    )


# ── Exit Simulators ───────────────────────────────────────────────────────────

def exit_vwap_capped(t: ORBTrade) -> ExitResult:
    """VWAP cross exit with hard −1R stop cap. Primary Phase 3 exit."""
    e, rk, d  = t.entry_price, t.risk, t.event.direction
    hard_stop = e - rk if d == "LONG" else e + rk   # −1R maximum loss
    day       = t.event.day_df

    for i, (idx, bar) in enumerate(t.remaining.iterrows()):
        bar_iloc = day.index.get_loc(idx)

        # Hard −1R stop (tighter than ORB stop in some cases)
        if d == "LONG" and bar["low"]  <= hard_stop:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap+1Rcap", r_val(e, hard_stop, rk, d), i+1, "stop", mfe, mae)
        if d == "SHORT" and bar["high"] >= hard_stop:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap+1Rcap", r_val(e, hard_stop, rk, d), i+1, "stop", mfe, mae)

        # VWAP cross exit
        if bar_iloc < len(t.vwap):
            vwap_val = t.vwap[bar_iloc]
            if (d == "LONG"  and bar["close"] < vwap_val) or \
               (d == "SHORT" and bar["close"] > vwap_val):
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult("vwap+1Rcap", r_val(e, bar["close"], rk, d), i+1, "vwap", mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("vwap+1Rcap", r_val(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_atr_trail(t: ORBTrade, mult: float = 1.0) -> ExitResult:
    """ATR trailing stop (Phase 2 best Sharpe)."""
    e, rk, d   = t.entry_price, t.risk, t.event.direction
    trail_dist = t.atr * mult
    cur_stop   = t.stop_price

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        if d == "LONG":
            if bar["low"] <= cur_stop:
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult("atr_1x_trail", r_val(e, cur_stop, rk, d), i+1, "trail", mfe, mae)
            cur_stop = max(cur_stop, bar["high"] - trail_dist)
        else:
            if bar["high"] >= cur_stop:
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult("atr_1x_trail", r_val(e, cur_stop, rk, d), i+1, "trail", mfe, mae)
            cur_stop = min(cur_stop, bar["low"] + trail_dist)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("atr_1x_trail", r_val(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_eod(t: ORBTrade) -> ExitResult:
    mfe, mae = excursion(t.remaining, t.entry_price, t.risk, t.event.direction)
    return ExitResult("eod", r_val(t.entry_price, t.eod_price, t.risk, t.event.direction),
                      len(t.remaining), "eod", mfe, mae)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def stats(pnls: List[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "sharpe": 0.0}
    n   = len(pnls)
    exp = np.mean(pnls)
    wr  = sum(1 for p in pnls if p > 0) / n * 100
    sh  = exp / np.std(pnls) if np.std(pnls) > 0 else 0.0
    return {"n": n, "exp": exp, "wr": wr, "sharpe": sh}


def mark(exp: float) -> str:
    return "✓✓" if exp > 0.20 else ("✓ " if exp > 0.05 else "  ")


# ── Proxy Analysis ────────────────────────────────────────────────────────────

def analyze_proxies(events: List[ORBEvent], trades: List[ORBTrade]):
    """For each proxy, show: trending-day precision, expectancy split, best threshold."""

    print("\n" + "=" * 110)
    print("  PHASE 3: REGIME PROXY ANALYSIS")
    print(f"  Ground truth: is_trending = day_range / session_ATR >= {TRENDING_THRESHOLD}")
    print(f"  Trending days in sample: {sum(e.is_trending for e in events)}/{len(events)} "
          f"({sum(e.is_trending for e in events)/len(events)*100:.0f}%)")
    print("=" * 110)

    # ── LOOK-AHEAD BIAS WARNING ────────────────────────────────────────────────
    print("\n  ⚠️  LOOK-AHEAD BUG IN PHASE 0.5")
    print("  'low ORB vol ratio' in Phase 0.5 = orb_vol / day_vol")
    print("  day_vol is only known at 4PM — NOT available at trade time (10AM)")
    print("  Real-time replacement: orb_vol / prior_day_total_vol")
    print()

    # Compare look-ahead vs real-time version
    trade_map = {t.event.date + t.event.symbol: t for t in trades}

    la_median  = np.median([e.orb_vol_ratio_la   for e in events])
    rt_median  = np.median([e.orb_vol_vs_prior_day for e in events])
    la_low     = [e for e in events if e.orb_vol_ratio_la   < la_median]
    rt_low     = [e for e in events if e.orb_vol_vs_prior_day < rt_median]

    def ev_pnl(evs, exit_fn=None):
        ts = [trade_map.get(e.date + e.symbol) for e in evs]
        ts = [t for t in ts if t]
        if not ts:
            return []
        if exit_fn is None:
            return [exit_eod(t).pnl_r for t in ts]
        return [exit_fn(t).pnl_r for t in ts]

    la_pnls = ev_pnl(la_low)
    rt_pnls = ev_pnl(rt_low)
    print(f"  Look-ahead 'low orb vol' (orb/day):       n={len(la_low):3d}  "
          f"exp={np.mean(la_pnls):+.3f}R  WR={sum(1 for p in la_pnls if p>0)/len(la_pnls)*100:.0f}%  "
          f"← uses future data")
    print(f"  Real-time  'low orb vol' (orb/prior_day): n={len(rt_low):3d}  "
          f"exp={np.mean(rt_pnls):+.3f}R  WR={sum(1 for p in rt_pnls if p>0)/len(rt_pnls)*100:.0f}%  "
          f"← tradeable")
    print()

    # ── PROXY VALIDATION TABLE ─────────────────────────────────────────────────
    print(f"  {'─' * 110}")
    print("  PROXY VALIDATION  (EOD P&L by proxy split | trending-day precision)")
    print(f"  {'Proxy':<38s}  {'n_hi':>4s}  {'Exp_hi':>7s}  {'WR_hi':>5s}  "
          f"{'n_lo':>4s}  {'Exp_lo':>7s}  {'WR_lo':>5s}  {'trend_prec':>10s}  {'note'}")
    print(f"  {'─' * 110}")

    proxies = [
        # (name, hi_condition, note)
        ("P1. Gap size (|gap|>0.5%)",
         lambda e: abs(e.gap_pct) > 0.005, "pre-market"),
        ("P1b. Gap size (|gap|>0.2%)",
         lambda e: abs(e.gap_pct) > 0.002, "pre-market"),
        ("P2. Prior day UP",
         lambda e: e.prior_day_up, "pre-market"),
        ("P3. Prior day trending (range≥3x ATR)",
         lambda e: e.prior_day_trending, "pre-market"),
        ("P4. ORB vs prior ATR (>0.8x)",
         lambda e: e.orb_vs_prior_atr > 0.8, "at-ORB"),
        ("P4b. ORB vs prior ATR (>1.0x)",
         lambda e: e.orb_vs_prior_atr > 1.0, "at-ORB"),
        ("P5. ORB vs prior range (>0.25x)",
         lambda e: e.orb_vs_prior_range > 0.25, "at-ORB"),
        ("P6. Low ORB vol/prior-day (RT fix)",
         lambda e: e.orb_vol_vs_prior_day < np.median([x.orb_vol_vs_prior_day for x in events]),
         "at-ORB ✓RT"),
        ("P7. Strong breakout bar (>0.6)",
         lambda e: e.breakout_bar_strength > 0.6, "at-ORB"),
        ("P7b. Very strong bar (>0.75)",
         lambda e: e.breakout_bar_strength > 0.75, "at-ORB"),
    ]

    rt_med = np.median([e.orb_vol_vs_prior_day for e in events])
    best_proxy_exp  = -999
    best_proxy_name = ""
    best_proxy_fn   = None

    for name, hi_cond, note in proxies:
        hi = [e for e in events if hi_cond(e)]
        lo = [e for e in events if not hi_cond(e)]
        if len(hi) < MIN_TRADES or len(lo) < MIN_TRADES:
            continue

        pnl_hi = ev_pnl(hi)
        pnl_lo = ev_pnl(lo)
        if not pnl_hi or not pnl_lo:
            continue

        # Trending precision: fraction of hi group that are actually trending
        trend_prec = sum(1 for e in hi if e.is_trending) / len(hi) * 100

        exp_hi = np.mean(pnl_hi)
        wr_hi  = sum(1 for p in pnl_hi if p > 0) / len(pnl_hi) * 100
        exp_lo = np.mean(pnl_lo)
        wr_lo  = sum(1 for p in pnl_lo if p > 0) / len(pnl_lo) * 100

        sep = exp_hi - exp_lo   # separation between hi and lo
        star = "★★" if sep > 0.2 else ("★ " if sep > 0.1 else "  ")
        print(f"  {star}{name:<38s}  {len(hi):4d}  {exp_hi:+.3f}R  {wr_hi:4.0f}%  "
              f"{len(lo):4d}  {exp_lo:+.3f}R  {wr_lo:4.0f}%  "
              f"{trend_prec:8.0f}%   {note}")

        if exp_hi > best_proxy_exp:
            best_proxy_exp  = exp_hi
            best_proxy_name = name
            best_proxy_fn   = hi_cond

    print(f"  {'─' * 110}")
    print(f"  ★★ = separation >0.20R   ★ = separation >0.10R")

    return best_proxy_fn, rt_med


# ── Filter Combinations ───────────────────────────────────────────────────────

def test_filter_combinations(events: List[ORBEvent], trades: List[ORBTrade], rt_med: float):
    """Test real-time filter combinations for best expectancy."""

    trade_map = {t.event.date + t.event.symbol: t for t in trades}

    def get_trades(evs):
        return [t for t in [trade_map.get(e.date + e.symbol) for e in evs] if t]

    def run_exits(ts):
        if len(ts) < MIN_TRADES:
            return None
        vwap_pnls  = [exit_vwap_capped(t).pnl_r for t in ts]
        trail_pnls = [exit_atr_trail(t).pnl_r   for t in ts]
        eod_pnls   = [exit_eod(t).pnl_r          for t in ts]
        return vwap_pnls, trail_pnls, eod_pnls

    orb_ranges = [e.orb_range for e in events]
    p20_orb    = np.percentile(orb_ranges, 20)

    # All filters using REAL-TIME data only
    filters = {
        "has_gap":         lambda e: abs(e.gap_pct) > 0.002,
        "prior_day_up":    lambda e: e.prior_day_up,
        "prior_day_trend": lambda e: e.prior_day_trending,
        "low_orb_vol_rt":  lambda e: e.orb_vol_vs_prior_day < rt_med,
        "orb_vs_atr>0.8":  lambda e: e.orb_vs_prior_atr > 0.8,
        "orb_vs_atr>1.0":  lambda e: e.orb_vs_prior_atr > 1.0,
        "strong_bar>0.6":  lambda e: e.breakout_bar_strength > 0.6,
        "not_narrow_orb":  lambda e: e.orb_range >= p20_orb,
        "gap_opposed":     lambda e: (e.direction == "LONG"  and e.gap_pct < -0.002) or
                                     (e.direction == "SHORT" and e.gap_pct >  0.002),
    }

    print(f"\n{'─' * 110}")
    print("  REAL-TIME FILTER COMBINATIONS (sorted by VWAP+cap expectancy)")
    print(f"  ⚠️  All filters use only data available at ORB breakout time (~10:00 AM)")
    print(f"  {'Filter combination':<50s}  {'n':>3s}  "
          f"{'VWAP+cap':>9s}  {'WR':>5s}  "
          f"{'ATR1x':>7s}  {'WR':>5s}  "
          f"{'EOD':>7s}  {'trend%':>7s}")
    print(f"  {'─' * 110}")

    results = []
    names   = list(filters.keys())
    fns     = list(filters.values())

    # Singles
    for name, fn in filters.items():
        g  = [e for e in events if fn(e)]
        ts = get_trades(g)
        if len(ts) < MIN_TRADES:
            continue
        out = run_exits(ts)
        if out is None:
            continue
        vp, tp, ep = out
        trend_pct = sum(1 for e in g if e.is_trending) / len(g) * 100
        results.append((name, len(ts), np.mean(vp), stats(vp)["wr"],
                        np.mean(tp), stats(tp)["wr"], np.mean(ep), trend_pct))

    # Pairs (non-conflicting)
    conflict = [{"has_gap", "gap_opposed"}, {"orb_vs_atr>0.8", "orb_vs_atr>1.0"}]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            if any({n1, n2} <= c for c in conflict):
                continue
            g  = [e for e in events if fns[i](e) and fns[j](e)]
            ts = get_trades(g)
            if len(ts) < MIN_TRADES:
                continue
            out = run_exits(ts)
            if out is None:
                continue
            vp, tp, ep = out
            trend_pct = sum(1 for e in g if e.is_trending) / len(g) * 100
            results.append((f"{n1} + {n2}", len(ts), np.mean(vp), stats(vp)["wr"],
                            np.mean(tp), stats(tp)["wr"], np.mean(ep), trend_pct))

    # Triples
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for k in range(j+1, len(names)):
                n1, n2, n3 = names[i], names[j], names[k]
                all3 = {n1, n2, n3}
                if any(c <= all3 for c in conflict):
                    continue
                g  = [e for e in events if fns[i](e) and fns[j](e) and fns[k](e)]
                ts = get_trades(g)
                if len(ts) < MIN_TRADES:
                    continue
                out = run_exits(ts)
                if out is None:
                    continue
                vp, tp, ep = out
                trend_pct = sum(1 for e in g if e.is_trending) / len(g) * 100
                results.append((f"{n1} + {n2} + {n3}", len(ts), np.mean(vp), stats(vp)["wr"],
                                np.mean(tp), stats(tp)["wr"], np.mean(ep), trend_pct))

    results.sort(key=lambda x: x[2], reverse=True)

    for (fname, n, vp_exp, vp_wr, tp_exp, tp_wr, ep_exp, trend_pct) in results[:30]:
        m = "✓✓" if vp_exp > 0.20 else ("✓ " if vp_exp > 0.05 else "  ")
        print(f"  {m}{fname:<50s}  {n:3d}  "
              f"{vp_exp:+.3f}R {vp_wr:4.0f}%  "
              f"{tp_exp:+.3f}R {tp_wr:4.0f}%  "
              f"{ep_exp:+.3f}R  {trend_pct:5.0f}%")

    return results


# ── Final Strategy Definition ─────────────────────────────────────────────────

def final_strategy(events: List[ORBEvent], trades: List[ORBTrade],
                   results: list, rt_med: float):
    """Print the final strategy rules based on Phase 3 findings."""

    trade_map = {t.event.date + t.event.symbol: t for t in trades}

    # Best combo from results
    if not results:
        return
    best_name, best_n, best_vp, best_vp_wr, best_tp, best_tp_wr, _, best_trend = results[0]

    print(f"\n{'=' * 110}")
    print("  PHASE 3 VERDICT — FINAL STRATEGY DEFINITION")
    print("=" * 110)

    print(f"\n  Best real-time filter: [{best_name}]")
    print(f"  n={best_n}  VWAP+cap: {best_vp:+.3f}R {best_vp_wr:.0f}%WR  "
          f"ATR1x: {best_tp:+.3f}R {best_tp_wr:.0f}%WR  "
          f"trending-day %: {best_trend:.0f}%")

    # Walk-forward sanity: H1 vs H2 of the sample
    print(f"\n  {'─' * 110}")
    print("  TEMPORAL STABILITY CHECK (H1 vs H2 of sample period)")
    print("  ⚠️  Small sample — treat as directional signal only, not statistical proof")

    all_dates = sorted(set(e.date for e in events))
    mid_date  = all_dates[len(all_dates) // 2]

    for period_name, date_fn in [
        (f"H1 (before {mid_date})",  lambda e: e.date < mid_date),
        (f"H2 (from   {mid_date})",  lambda e: e.date >= mid_date),
    ]:
        h_events = [e for e in events if date_fn(e)]
        h_trades = [t for t in [trade_map.get(e.date + e.symbol) for e in h_events] if t]
        if len(h_trades) < 5:
            continue

        # EOD baseline for this half
        eod_pnls = [exit_eod(t).pnl_r for t in h_trades]
        vwap_pnls = [exit_vwap_capped(t).pnl_r for t in h_trades]
        print(f"  {period_name:<30s}  n={len(h_trades):3d}  "
              f"EOD: {np.mean(eod_pnls):+.3f}R {sum(1 for p in eod_pnls if p>0)/len(eod_pnls)*100:.0f}%WR  "
              f"VWAP+cap: {np.mean(vwap_pnls):+.3f}R {sum(1 for p in vwap_pnls if p>0)/len(vwap_pnls)*100:.0f}%WR")

    # Final rules
    print(f"\n  {'─' * 110}")
    print("  COMPLETE STRATEGY RULES (all filters real-time, no look-ahead)")
    print()
    print("  SETUP DETECTION (10:00 AM — when ORB forms):")
    print("    1. Identify ORB: high/low of 9:30–10:00 bars")
    print("    2. Wait for first 15m bar that CLOSES beyond ORB boundary")
    print("    3. Entry = close of that bar (market order at bar close)")
    print()
    print("  REGIME GATE (evaluate before taking the trade):")
    print("    — Has gap > 0.2%  (|open - prior_close| / prior_close > 0.002)")
    print("    — Prior day was UP  (prior day close > prior day open)")
    print("    — Prior day was trending  (prior_day_range / prior_day_ATR >= 3.0)")
    print("    — ORB range / prior_day_ATR > 0.8  (today's opening is active)")
    print("    — Low ORB vol vs prior day  (orb_volume / prior_day_volume < sample median)")
    print("    → Skip the trade if any key gate fails")
    print()
    print("  STOP:")
    print("    — Initial hard stop: opposite ORB boundary")
    print("    — Override: maximum −1R from entry (whichever is tighter)")
    print()
    print("  EXIT (choose one):")
    print("    A. VWAP cross + −1R cap  (higher expectancy)")
    print("       Exit when: price closes back through VWAP against position")
    print("       OR:        price hits −1R hard stop")
    print("    B. ATR 1x trailing stop  (better Sharpe, less monitoring)")
    print("       Trail stop: max(ORB_stop, bar_high − 1×ATR) for LONG")
    print()
    print("  POSITION SIZING (not tested here — Phase 4):")
    print("    — Fixed fractional: risk 0.5−1% of capital per trade")
    print("    — Position size = risk_amount / (entry − stop) shares")
    print()
    print("  KNOWN LIMITATIONS:")
    print("    — Sample: 454 events, 106 days, 5 symbols — limited")
    print("    — Tight filter subsets have n<30 — need out-of-sample validation")
    print("    — No transaction costs, slippage, or market impact modelled")
    print("    — Regime proxy (orb_vs_prior_atr) needs forward testing")
    print("    — Phase 4 should: fetch new data, test on unseen period")
    print("=" * 110)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase3_regime_filter.py <csv1> [csv2] ...")
        sys.exit(1)

    all_events = []
    for path in sys.argv[1:]:
        print(f"Loading {path}...")
        try:
            df     = load_csv(path)
            sym    = df["symbol"].iloc[0]
            events = find_orb_events(df)
            print(f"  {sym}: {len(df)} bars, {df['date'].nunique()} days, {len(events)} ORB events")
            all_events.extend(events)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if not all_events:
        print("No events found.")
        sys.exit(1)

    print(f"\nTotal: {len(all_events)} ORB events across {len(sys.argv)-1} symbols")

    all_trades = [t for t in [build_trade(e) for e in all_events] if t]
    print(f"Trades built: {len(all_trades)}")

    # Trending day breakdown
    n_trending = sum(e.is_trending for e in all_events)
    print(f"\nGround-truth trending days: {n_trending}/{len(all_events)} "
          f"({n_trending/len(all_events)*100:.0f}%)")

    # Run proxy analysis
    best_fn, rt_med = analyze_proxies(all_events, all_trades)

    # Run filter combinations
    results = test_filter_combinations(all_events, all_trades, rt_med)

    # Final strategy
    final_strategy(all_events, all_trades, results, rt_med)


if __name__ == "__main__":
    main()
