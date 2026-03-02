"""Phase 2: Exit Method Optimization

Question: What exit method produces the best R:R given our winning entry?

Entry (from Phase 1): bar close of first breakout bar  (100% fill, no adverse selection)
Filter (from Phase 0.5): low ORB vol ratio + has gap   (154 events, +0.110R baseline)

Exit methods tested (18 total):
  Baseline:    EOD close
  Fixed R:R:   1.0R, 1.5R, 2.0R, 3.0R targets
  ATR trail:   0.5x, 1.0x, 1.5x, 2.0x ATR trailing stop
  Structural:  PDH/PDL level, VWAP cross exit
  Time-based:  exit at 13:00, 14:00, 15:00
  Partials:    50%@1R→breakeven trail,  50%@1.5R→breakeven trail,
               50%@2R→stop locked at +1R,  33%@1R + 33%@2R + 34%@EOD

Only the exit varies. Same entries, same initial stop (opposite ORB boundary).

Also includes regime proxy analysis:
  - Performance by prior day direction
  - Performance by gap alignment / trending-day proxy
  - Intraday range proxy for trending vs choppy days

Usage:
  python3 phase2_exit_methods.py ./data/cache/AAPL_2025-07-01_2025-12-1_15_mins.csv \\
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
from typing import List, Tuple, Optional
from datetime import time as dtime


# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN  = 30
MIN_TRADES       = 10
ATR_LOOKBACK     = 10   # bars for ATR computation


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBEvent:
    symbol:             str
    date:               str
    direction:          str
    orb_high:           float
    orb_low:            float
    orb_range:          float
    orb_volume:         float
    day_volume:         float
    gap_pct:            float
    breakout_close:     float
    breakout_time:      str
    breakout_bar_iloc:  int
    prev_day_high:      float
    prev_day_low:       float
    prev_day_up:        bool    # was previous day bullish?
    day_df:             object


@dataclass
class ORBTrade:
    event:        ORBEvent
    entry_price:  float
    stop_price:   float
    risk:         float
    remaining:    pd.DataFrame   # bars strictly after entry bar
    eod_price:    float
    atr:          float
    vwap:         np.ndarray     # running VWAP for each bar in day_df


@dataclass
class ExitResult:
    method:       str
    pnl_r:        float
    bars_held:    int
    exit_reason:  str            # target | stop | trail | eod | time | vwap
    mfe_r:        float
    mae_r:        float


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

def compute_vwap(day_df: pd.DataFrame) -> np.ndarray:
    tp      = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
    cum_vol = day_df["volume"].cumsum()
    vwap    = (tp * day_df["volume"]).cumsum() / cum_vol
    return vwap.values


def compute_atr(bars: pd.DataFrame, n: int = ATR_LOOKBACK) -> float:
    if len(bars) == 0:
        return 0.01
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    trs = [h[0] - l[0]]
    for i in range(1, len(bars)):
        trs.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    return float(np.mean(trs[-n:])) if trs else 0.01


def r(entry: float, exit_p: float, risk: float, direction: str) -> float:
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
    orb_end        = dtime(10, 0)
    symbol         = df["symbol"].iloc[0]
    events         = []
    prev_day_high  = None
    prev_day_low   = None
    prev_day_close = None
    prev_day_open  = None
    prior_close    = None

    for day_date in sorted(df["date"].unique()):
        day = df[df["date"] == day_date].sort_values("timestamp").reset_index(drop=True)

        prev_up = (prev_day_close > prev_day_open) if (prev_day_close and prev_day_open) else True

        if len(day) < 4:
            if len(day) > 0:
                prev_day_high  = day["high"].max()
                prev_day_low   = day["low"].min()
                prev_day_close = day["close"].iloc[-1]
                prev_day_open  = day["open"].iloc[0]
                prior_close    = day["close"].iloc[-1]
            continue

        orb_bars   = day[day["time"] < orb_end]
        if len(orb_bars) < 1:
            prev_day_high  = day["high"].max()
            prev_day_low   = day["low"].min()
            prev_day_close = day["close"].iloc[-1]
            prev_day_open  = day["open"].iloc[0]
            prior_close    = day["close"].iloc[-1]
            continue

        orb_high   = orb_bars["high"].max()
        orb_low    = orb_bars["low"].min()
        orb_range  = orb_high - orb_low
        orb_volume = orb_bars["volume"].sum()

        if orb_range <= 0:
            prev_day_high  = day["high"].max()
            prev_day_low   = day["low"].min()
            prev_day_close = day["close"].iloc[-1]
            prev_day_open  = day["open"].iloc[0]
            prior_close    = day["close"].iloc[-1]
            continue

        day_open   = day["open"].iloc[0]
        day_volume = day["volume"].sum()
        gap_pct    = (day_open - prior_close) / prior_close if prior_close and prior_close > 0 else 0.0

        post_orb = day[day["time"] >= orb_end]
        if len(post_orb) < 2:
            prev_day_high  = day["high"].max()
            prev_day_low   = day["low"].min()
            prev_day_close = day["close"].iloc[-1]
            prev_day_open  = day["open"].iloc[0]
            prior_close    = day["close"].iloc[-1]
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
            pdh    = prev_day_high if prev_day_high else orb_high + orb_range
            pdl    = prev_day_low  if prev_day_low  else orb_low  - orb_range

            events.append(ORBEvent(
                symbol=symbol, date=str(day_date), direction=direction,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_volume=orb_volume, day_volume=day_volume, gap_pct=gap_pct,
                breakout_close=close, breakout_time=str(row["time"]),
                breakout_bar_iloc=b_iloc,
                prev_day_high=pdh, prev_day_low=pdl,
                prev_day_up=prev_up, day_df=day,
            ))
            break

        prev_day_high  = day["high"].max()
        prev_day_low   = day["low"].min()
        prev_day_close = day["close"].iloc[-1]
        prev_day_open  = day["open"].iloc[0]
        prior_close    = day["close"].iloc[-1]

    return events


def build_trade(ev: ORBEvent) -> Optional[ORBTrade]:
    """Construct an ORBTrade using bar-close entry (Phase 1 winner)."""
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

    vwap = compute_vwap(day)

    return ORBTrade(
        event=ev, entry_price=entry, stop_price=stop, risk=risk,
        remaining=remaining, eod_price=day["close"].iloc[-1],
        atr=atr, vwap=vwap,
    )


# ── Exit Simulators ───────────────────────────────────────────────────────────

def _bar_step(bar, entry, stop, target, direction):
    """Check stop/target hit for one bar. Returns ('stop'|'target'|None, exit_price)."""
    if direction == "LONG":
        stop_hit   = bar["low"]  <= stop   if stop   is not None else False
        target_hit = bar["high"] >= target if target is not None else False
    else:
        stop_hit   = bar["high"] >= stop   if stop   is not None else False
        target_hit = bar["low"]  <= target if target is not None else False

    if stop_hit and target_hit:
        return "stop", stop          # conservative: stop first
    if target_hit:
        return "target", target
    if stop_hit:
        return "stop", stop
    return None, None


def exit_eod(t: ORBTrade) -> ExitResult:
    mfe, mae = excursion(t.remaining, t.entry_price, t.risk, t.event.direction)
    return ExitResult("eod", r(t.entry_price, t.eod_price, t.risk, t.event.direction),
                      len(t.remaining), "eod", mfe, mae)


def exit_fixed(t: ORBTrade, target_r: float, label: str) -> ExitResult:
    e, s, rk, d = t.entry_price, t.stop_price, t.risk, t.event.direction
    tgt = e + target_r * rk if d == "LONG" else e - target_r * rk

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        reason, exit_p = _bar_step(bar, e, s, tgt, d)
        if reason:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult(label, r(e, exit_p, rk, d), i+1, reason, mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult(label, r(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_atr_trail(t: ORBTrade, mult: float, label: str) -> ExitResult:
    e, rk, d    = t.entry_price, t.risk, t.event.direction
    trail_dist  = t.atr * mult
    cur_stop    = t.stop_price

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        # Check stop BEFORE updating trail (avoids same-bar ambiguity)
        if d == "LONG":
            if bar["low"] <= cur_stop:
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, r(e, cur_stop, rk, d), i+1, "trail", mfe, mae)
            cur_stop = max(cur_stop, bar["high"] - trail_dist)
        else:
            if bar["high"] >= cur_stop:
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, r(e, cur_stop, rk, d), i+1, "trail", mfe, mae)
            cur_stop = min(cur_stop, bar["low"] + trail_dist)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult(label, r(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_vwap_cross(t: ORBTrade) -> ExitResult:
    """Exit when price closes back through VWAP against the position."""
    e, s, rk, d = t.entry_price, t.stop_price, t.risk, t.event.direction
    day = t.event.day_df

    for i, (idx, bar) in enumerate(t.remaining.iterrows()):
        bar_iloc = day.index.get_loc(idx)
        if bar_iloc >= len(t.vwap):
            break
        vwap_val = t.vwap[bar_iloc]

        # Stop check first
        stop_hit = bar["low"] <= s if d == "LONG" else bar["high"] >= s
        if stop_hit:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap_cross", r(e, s, rk, d), i+1, "stop", mfe, mae)

        # VWAP cross against position
        if (d == "LONG"  and bar["close"] < vwap_val) or \
           (d == "SHORT" and bar["close"] > vwap_val):
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("vwap_cross", r(e, bar["close"], rk, d), i+1, "vwap", mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("vwap_cross", r(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_pdh_pdl(t: ORBTrade) -> ExitResult:
    """Target = PDH for LONG, PDL for SHORT. Falls back to EOD if already past target."""
    e, s, rk, d = t.entry_price, t.stop_price, t.risk, t.event.direction
    tgt = t.event.prev_day_high if d == "LONG" else t.event.prev_day_low

    # If target is invalid (below entry for LONG, above for SHORT), use EOD
    if (d == "LONG" and tgt <= e) or (d == "SHORT" and tgt >= e):
        return exit_eod(t)

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        reason, exit_p = _bar_step(bar, e, s, tgt, d)
        if reason:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("pdh_pdl", r(e, exit_p, rk, d), i+1, reason, mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("pdh_pdl", r(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_time(t: ORBTrade, exit_at: dtime, label: str) -> ExitResult:
    """Exit at close of first bar at or after exit_at time."""
    e, s, rk, d = t.entry_price, t.stop_price, t.risk, t.event.direction

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        stop_hit = bar["low"] <= s if d == "LONG" else bar["high"] >= s
        if stop_hit:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult(label, r(e, s, rk, d), i+1, "stop", mfe, mae)
        if bar["time"] >= exit_at:
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult(label, r(e, bar["close"], rk, d), i+1, "time", mfe, mae)

    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult(label, r(e, t.eod_price, rk, d), len(t.remaining), "eod", mfe, mae)


def exit_partial_be_trail(t: ORBTrade, first_r: float, label: str) -> ExitResult:
    """50% exits at first_r target → stop moves to breakeven → trail rest at 1x ATR."""
    e, s, rk, d   = t.entry_price, t.stop_price, t.risk, t.event.direction
    tgt1           = e + first_r * rk if d == "LONG" else e - first_r * rk
    trail_dist     = t.atr * 1.0
    partial_done   = False
    realized       = 0.0
    cur_stop       = s

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        if not partial_done:
            reason, exit_p = _bar_step(bar, e, s, tgt1, d)
            if reason == "stop":
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, r(e, exit_p, rk, d), i+1, "stop", mfe, mae)
            if reason == "target":
                realized     = 0.5 * first_r   # 50% locked
                cur_stop     = e               # move to breakeven
                partial_done = True
                # Update trail immediately on this bar
                cur_stop = max(cur_stop, bar["high"] - trail_dist) if d == "LONG" else \
                           min(cur_stop, bar["low"]  + trail_dist)
        else:
            # Trailing the remaining 50%
            if (d == "LONG" and bar["low"] <= cur_stop) or \
               (d == "SHORT" and bar["high"] >= cur_stop):
                total    = realized + 0.5 * r(e, cur_stop, rk, d)
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, total, i+1, "trail", mfe, mae)
            cur_stop = max(cur_stop, bar["high"] - trail_dist) if d == "LONG" else \
                       min(cur_stop, bar["low"]  + trail_dist)

    rem_pnl  = r(e, t.eod_price, rk, d)
    total    = (realized + 0.5 * rem_pnl) if partial_done else rem_pnl
    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult(label, total, len(t.remaining), "eod", mfe, mae)


def exit_partial_lock(t: ORBTrade, first_r: float, lock_r: float, label: str) -> ExitResult:
    """50% at first_r → lock stop at +lock_r → hold rest to EOD or stop."""
    e, s, rk, d = t.entry_price, t.stop_price, t.risk, t.event.direction
    tgt1         = e + first_r * rk if d == "LONG" else e - first_r * rk
    locked_stop  = e + lock_r * rk  if d == "LONG" else e - lock_r * rk
    partial_done = False
    realized     = 0.0
    cur_stop     = s

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        if not partial_done:
            reason, exit_p = _bar_step(bar, e, s, tgt1, d)
            if reason == "stop":
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, r(e, exit_p, rk, d), i+1, "stop", mfe, mae)
            if reason == "target":
                realized     = 0.5 * first_r
                cur_stop     = locked_stop
                partial_done = True
        else:
            if (d == "LONG" and bar["low"] <= cur_stop) or \
               (d == "SHORT" and bar["high"] >= cur_stop):
                total    = realized + 0.5 * r(e, cur_stop, rk, d)
                mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
                return ExitResult(label, total, i+1, "stop", mfe, mae)

    rem_pnl  = r(e, t.eod_price, rk, d)
    total    = (realized + 0.5 * rem_pnl) if partial_done else rem_pnl
    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult(label, total, len(t.remaining), "eod", mfe, mae)


def exit_thirds(t: ORBTrade) -> ExitResult:
    """33% at 1R, 33% at 2R, 34% at EOD. Stop on all remaining."""
    e, s, rk, d  = t.entry_price, t.stop_price, t.risk, t.event.direction
    t1 = e + 1.0 * rk if d == "LONG" else e - 1.0 * rk
    t2 = e + 2.0 * rk if d == "LONG" else e - 2.0 * rk
    realized  = 0.0
    rem_size  = 1.0
    t1_done   = False
    t2_done   = False

    for i, (_, bar) in enumerate(t.remaining.iterrows()):
        stop_hit = bar["low"] <= s if d == "LONG" else bar["high"] >= s
        if stop_hit:
            total    = realized + rem_size * r(e, s, rk, d)
            mfe, mae = excursion(t.remaining.iloc[:i+1], e, rk, d)
            return ExitResult("thirds", total, i+1, "stop", mfe, mae)
        if not t1_done and (bar["high"] >= t1 if d == "LONG" else bar["low"] <= t1):
            realized += 0.33 * 1.0
            rem_size -= 0.33
            t1_done   = True
        if not t2_done and (bar["high"] >= t2 if d == "LONG" else bar["low"] <= t2):
            realized += 0.33 * 2.0
            rem_size -= 0.33
            t2_done   = True

    total    = realized + rem_size * r(e, t.eod_price, rk, d)
    mfe, mae = excursion(t.remaining, e, rk, d)
    return ExitResult("thirds", total, len(t.remaining), "eod", mfe, mae)


# ── All exit methods in order ─────────────────────────────────────────────────

def run_all_exits(trade: ORBTrade) -> List[ExitResult]:
    return [
        exit_eod(trade),
        exit_fixed(trade, 1.0,  "fixed_1R"),
        exit_fixed(trade, 1.5,  "fixed_1.5R"),
        exit_fixed(trade, 2.0,  "fixed_2R"),
        exit_fixed(trade, 3.0,  "fixed_3R"),
        exit_atr_trail(trade, 0.5, "atr_0.5x"),
        exit_atr_trail(trade, 1.0, "atr_1.0x"),
        exit_atr_trail(trade, 1.5, "atr_1.5x"),
        exit_atr_trail(trade, 2.0, "atr_2.0x"),
        exit_vwap_cross(trade),
        exit_pdh_pdl(trade),
        exit_time(trade, dtime(13, 0), "time_13:00"),
        exit_time(trade, dtime(14, 0), "time_14:00"),
        exit_time(trade, dtime(15, 0), "time_15:00"),
        exit_partial_be_trail(trade, 1.0,  "50%@1R→BE+trail"),
        exit_partial_be_trail(trade, 1.5,  "50%@1.5R→BE+trail"),
        exit_partial_lock(trade,   2.0, 1.0, "50%@2R→lock+1R"),
        exit_thirds(trade),
    ]


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(trades: List[ORBTrade], label: str):
    n = len(trades)
    if n < MIN_TRADES:
        return

    # Collect results per method
    method_results: dict = {}
    for t in trades:
        for res in run_all_exits(t):
            if res.method not in method_results:
                method_results[res.method] = []
            method_results[res.method].append(res)

    print(f"\n  {label}  ({n} trades)")
    print(f"  {'─' * 113}")
    print(f"  {'Exit Method':<22s}  {'Exp':>7s}  {'WR':>5s}  {'Sharpe':>6s}  "
          f"{'AvgW':>6s}  {'AvgL':>6s}  {'MFE':>6s}  {'MAE':>6s}  "
          f"{'Bars':>5s}  {'Tgt%':>5s}  {'Stp%':>5s}  {'EOD%':>5s}")
    print(f"  {'─' * 113}")

    all_rows = []
    for method, results in method_results.items():
        pnls     = [res.pnl_r        for res in results]
        mfes     = [res.mfe_r        for res in results]
        maes     = [res.mae_r        for res in results]
        bars     = [res.bars_held    for res in results]
        reasons  = [res.exit_reason  for res in results]

        exp    = np.mean(pnls)
        wr     = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sharpe = exp / np.std(pnls) if np.std(pnls) > 0 else 0.0
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_w  = np.mean(wins)   if wins   else 0.0
        avg_l  = np.mean(losses) if losses else 0.0
        tgt_pct = reasons.count("target") / n * 100
        stp_pct = (reasons.count("stop") + reasons.count("trail")) / n * 100
        eod_pct = (reasons.count("eod") + reasons.count("time") + reasons.count("vwap")) / n * 100
        avg_bars = np.mean(bars)

        all_rows.append((method, exp, wr, sharpe, avg_w, avg_l,
                         np.mean(mfes), np.mean(maes), avg_bars,
                         tgt_pct, stp_pct, eod_pct))

    # Sort by expectancy
    all_rows.sort(key=lambda x: x[1], reverse=True)

    for (method, exp, wr, sharpe, avg_w, avg_l, mfe, mae,
         avg_bars, tgt_pct, stp_pct, eod_pct) in all_rows:
        mark = "✓✓" if exp > 0.20 else ("✓ " if exp > 0.05 else "  ")
        print(f"  {mark}{method:<22s}  {exp:+.3f}R  {wr:4.0f}%  {sharpe:+.4f}  "
              f"{avg_w:+.3f}R  {avg_l:+.3f}R  {mfe:+.3f}R  {mae:+.3f}R  "
              f"{avg_bars:4.1f}  {tgt_pct:4.0f}%  {stp_pct:4.0f}%  {eod_pct:4.0f}%")

    print(f"  {'─' * 113}")

    # P&L distribution for top 5 methods
    top5 = [row[0] for row in all_rows[:5]]
    print(f"\n  P&L distribution — top 5 methods (filled trades):")
    print(f"  {'Method':<22s}  {'P10':>6s}  {'P25':>6s}  {'P50':>6s}  "
          f"{'P75':>6s}  {'P90':>6s}  {'worst':>7s}  {'best':>7s}")
    print(f"  {'─' * 80}")
    for method in top5:
        pnls = [res.pnl_r for res in method_results[method]]
        p    = np.percentile(pnls, [10, 25, 50, 75, 90])
        print(f"  {method:<22s}  {p[0]:+.3f}R  {p[1]:+.3f}R  {p[2]:+.3f}R  "
              f"{p[3]:+.3f}R  {p[4]:+.3f}R  {min(pnls):+.3f}R  {max(pnls):+.3f}R")


# ── Filter Helpers ────────────────────────────────────────────────────────────

def filter_events(events: List[ORBEvent], **flags) -> List[ORBEvent]:
    g = list(events)
    if flags.get("low_orb_vol"):
        ratios = [e.orb_volume / e.day_volume for e in g if e.day_volume > 0]
        med    = np.median(ratios) if ratios else 0.19
        g = [e for e in g if e.day_volume > 0 and e.orb_volume / e.day_volume < med]
    if flags.get("has_gap"):
        g = [e for e in g if abs(e.gap_pct) > 0.002]
    if flags.get("not_narrow_orb"):
        all_ranges = [e.orb_range for e in events]
        p20 = np.percentile(all_ranges, 20)
        g = [e for e in g if e.orb_range >= p20]
    if flags.get("gap_opposed"):
        g = [e for e in g if
             (e.direction == "LONG"  and e.gap_pct < -0.002) or
             (e.direction == "SHORT" and e.gap_pct >  0.002)]
    if flags.get("mid_day"):
        g = [e for e in g if dtime.fromisoformat(e.breakout_time) >= dtime(11, 0)]
    return g


def events_to_trades(events: List[ORBEvent]) -> List[ORBTrade]:
    trades = []
    for ev in events:
        t = build_trade(ev)
        if t:
            trades.append(t)
    return trades


# ── Regime Proxy Analysis ─────────────────────────────────────────────────────

def regime_analysis(all_trades: List[ORBTrade]):
    """Does intraday regime (trending vs choppy) predict ORB exit success?"""
    print(f"\n{'─' * 115}")
    print("  REGIME PROXY ANALYSIS")
    print("  Question: does the type of day predict which exit works?")
    print("  Proxy: intraday range / ATR ratio as trending-day indicator")
    print()

    for t in all_trades:
        day    = t.event.day_df
        day_range = day["high"].max() - day["low"].min()
        t._day_range = day_range
        t._atr       = t.atr

    # Trending day proxy: day's high-low range vs ATR at entry
    # High ratio = trending day;  Low ratio = choppy day
    ratios   = [t._day_range / t._atr for t in all_trades if t._atr > 0]
    med_ratio = np.median(ratios)

    trending = [t for t in all_trades if t._atr > 0 and t._day_range / t._atr >= med_ratio]
    choppy   = [t for t in all_trades if t._atr > 0 and t._day_range / t._atr <  med_ratio]

    print(f"  Day range / ATR median: {med_ratio:.1f}x")
    print(f"  Trending days (range >= {med_ratio:.1f}x ATR): n={len(trending)}")
    print(f"  Choppy days   (range <  {med_ratio:.1f}x ATR): n={len(choppy)}")
    print()

    for group_label, group in [("Trending days", trending), ("Choppy days", choppy)]:
        if len(group) < MIN_TRADES:
            continue
        print(f"  {group_label}:")
        # Compare EOD vs fixed 2R vs ATR trail 1x on these days
        for sim_fn, label in [
            (exit_eod,                              "EOD"),
            (lambda t: exit_fixed(t, 2.0, "x"),    "Fixed 2R"),
            (lambda t: exit_atr_trail(t, 1.0, "x"),"ATR trail 1x"),
        ]:
            pnls = [sim_fn(t).pnl_r for t in group]
            exp  = np.mean(pnls)
            wr   = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"    {label:<18s}  exp {exp:+.3f}R  WR {wr:.0f}%")
        print()

    # Prior day direction
    print(f"  By prior day direction:")
    prev_up   = [t for t in all_trades if t.event.prev_day_up]
    prev_down = [t for t in all_trades if not t.event.prev_day_up]
    for group_label, group in [("Prior day UP", prev_up), ("Prior day DOWN", prev_down)]:
        if len(group) < MIN_TRADES:
            continue
        pnls = [exit_eod(t).pnl_r for t in group]
        exp  = np.mean(pnls)
        wr   = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"    {group_label:<20s}  exp {exp:+.3f}R  WR {wr:.0f}%  n={len(group)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase2_exit_methods.py <csv1> [csv2] ...")
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
    print("\n" + "=" * 115)
    print("  PHASE 2: EXIT METHOD COMPARISON")
    print("  Entry: bar close of first breakout bar  |  Stop: opposite ORB boundary")
    print("  ✓✓ = exp > +0.20R    ✓ = exp > +0.05R")
    print("  Columns: Exp | WR | Sharpe | AvgWin | AvgLoss | AvgMFE | AvgMAE | AvgBars | Tgt% | Stp% | EOD%")
    print("=" * 115)

    # Build trade sets
    f_best   = filter_events(all_events, low_orb_vol=True, has_gap=True)
    f_tight  = filter_events(all_events, low_orb_vol=True, gap_opposed=True, mid_day=True)
    f_full_f = filter_events(all_events, low_orb_vol=True, has_gap=True, not_narrow_orb=True)

    all_trades    = events_to_trades(all_events)
    best_trades   = events_to_trades(f_best)
    tight_trades  = events_to_trades(f_tight)
    full_f_trades = events_to_trades(f_full_f)

    print("\n" + "─" * 115)
    print("  [A] FULL SAMPLE  — no filters")
    report(all_trades, "Full sample (454 events)")

    print("\n" + "─" * 115)
    print("  [B] PHASE 0.5 BEST FILTER  — low ORB vol + has gap")
    report(best_trades, "low ORB vol + has gap (154 events)")

    print("\n" + "─" * 115)
    print("  [C] EXTENDED FILTER  — low ORB vol + has gap + not narrow ORB")
    report(full_f_trades, "low ORB vol + has gap + not narrow ORB (117 events)")

    print("\n" + "─" * 115)
    print("  [D] TIGHTEST FILTER  — low ORB vol + gap opposed + mid-day (Phase 0.5 best)")
    report(tight_trades, "low ORB vol + gap opposed + mid-day (28 events)")

    # Regime analysis on best filter set
    regime_analysis(best_trades)

    # Final verdict
    print(f"\n{'=' * 115}")
    print("  VERDICT")
    print()

    for label, trades in [
        ("Full sample",              all_trades),
        ("low ORB vol + has gap",    best_trades),
    ]:
        if len(trades) < MIN_TRADES:
            continue
        best_exp, best_method = -999, ""
        method_results = {}
        for t in trades:
            for res in run_all_exits(t):
                if res.method not in method_results:
                    method_results[res.method] = []
                method_results[res.method].append(res)
        for method, results in method_results.items():
            exp = np.mean([res.pnl_r for res in results])
            if exp > best_exp:
                best_exp, best_method = exp, method
        sharpe_vals = [res.pnl_r for res in method_results[best_method]]
        sharpe = np.mean(sharpe_vals) / np.std(sharpe_vals) if np.std(sharpe_vals) > 0 else 0
        print(f"  {label:<30s}  Best exit: {best_method:<22s}  "
              f"exp {best_exp:+.3f}R  Sharpe {sharpe:+.4f}")

    print()
    print("  Key takeaways:")
    print("  • Fixed targets cap your winners — only useful if ORB has strong directional follow-through")
    print("  • ATR trailing keeps you in trending moves while cutting losses early")
    print("  • Partial exits (50%@target + trail) give asymmetric payoff: guaranteed partial win + free runner")
    print("  • VWAP cross is a momentum exhaustion signal — exits when ORB breakout loses steam")
    print("  • PDH/PDL: structural resistance/support — clean target for the breakout to run to")
    print("  • Regime matters (see analysis above) — trending days favor trail/partial, choppy days favor fixed")
    print("  Next step → Phase 3: regime filter (trending vs choppy day identification pre-market)")
    print("=" * 115)


if __name__ == "__main__":
    main()
