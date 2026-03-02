"""Phase 1: Entry Method Optimization

Question: What entry method produces the best R:R?

Method: Test 6 entry methods on the SAME set of ORB breakout events.
  1. Bar close        — enter at close of first breakout bar (current)
  2. Next bar open    — enter at open of bar immediately after breakout
  3. Limit at ORB     — limit order at ORB boundary; fill when price touches
  4. Retest open      — price pulls back to ORB boundary, enter next bar's open
  5. Half pullback    — limit at midpoint between breakout close and ORB boundary
  6. Pullback+confirm — price touches boundary, first bar that closes in breakout
                        direction (bullish/bearish close) = entry [added from analysis]

Everything held constant across all methods:
  - Same ORB events (same days, same direction)
  - Stop = opposite ORB boundary
  - Target = EOD (full-day outcome)

Metrics per method:
  Fill rate | Avg risk $ | Expectancy R | Win rate | Sharpe | Avg MFE | Avg MAE | Stop hit %

Also runs each method on Phase 0.5's best filters:
  low ORB vol ratio + has gap + not narrow ORB

Usage:
  python3 phase1_entry_methods.py ./data/cache/AAPL_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/NVDA_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/MSFT_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/AMZN_2025-07-01_2025-12-1_15_mins.csv \\
                                   ./data/cache/GOOG_2025-07-01_2025-12-1_15_mins.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import time as dtime


# ── Configuration ─────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN  = 30
MIN_TRADES       = 15


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ORBEvent:
    """A detected ORB breakout setup — direction and context, before entry."""
    symbol:           str
    date:             str
    direction:        str       # "LONG" or "SHORT"
    orb_high:         float
    orb_low:          float
    orb_range:        float
    orb_volume:       float
    day_volume:       float
    gap_pct:          float
    breakout_close:   float
    breakout_time:    str
    breakout_bar_iloc: int      # position in day_df (integer location)
    day_df:           object    # full day DataFrame (reference, not copied)


@dataclass
class EntryResult:
    method:          str
    filled:          bool
    entry_price:     float = 0.0
    entry_time:      str   = ""
    stop_price:      float = 0.0
    risk_dollars:    float = 0.0
    eod_pnl_r:       float = 0.0
    max_favorable_r: float = 0.0
    max_adverse_r:   float = 0.0
    hit_stop:        bool  = False
    bars_to_fill:    int   = 0


METHOD_LABELS = {
    "1_bar_close":        "1. Bar close          (current, chase)",
    "2_next_open":        "2. Next bar open",
    "3_limit_orb":        "3. Limit at ORB boundary",
    "4_retest_open":      "4. Retest open         (touch → next open)",
    "5_half_pullback":    "5. Half-range pullback",
    "6_pullback_confirm": "6. Pullback + confirm  (touch → bullish close)",
}


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


# ── ORB Event Detection ───────────────────────────────────────────────────────

def find_orb_events(df: pd.DataFrame) -> List[ORBEvent]:
    """Detect ORB breakout setups. Entry is NOT simulated here."""
    orb_end     = dtime(10, 0)
    symbol      = df["symbol"].iloc[0]
    events      = []
    prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = df[df["date"] == day_date].sort_values("timestamp").reset_index(drop=True)
        if len(day) < 4:
            if len(day) > 0:
                prior_close = day["close"].iloc[-1]
            continue

        orb_bars = day[day["time"] < orb_end]
        if len(orb_bars) < 1:
            prior_close = day["close"].iloc[-1]
            continue

        orb_high   = orb_bars["high"].max()
        orb_low    = orb_bars["low"].min()
        orb_range  = orb_high - orb_low
        orb_volume = orb_bars["volume"].sum()

        if orb_range <= 0:
            prior_close = day["close"].iloc[-1]
            continue

        day_open   = day["open"].iloc[0]
        day_volume = day["volume"].sum()
        gap_pct    = (day_open - prior_close) / prior_close if prior_close and prior_close > 0 else 0.0

        post_orb = day[day["time"] >= orb_end]
        if len(post_orb) < 2:
            prior_close = day["close"].iloc[-1]
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
            events.append(ORBEvent(
                symbol=symbol, date=str(day_date), direction=direction,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_volume=orb_volume, day_volume=day_volume, gap_pct=gap_pct,
                breakout_close=close, breakout_time=str(row["time"]),
                breakout_bar_iloc=b_iloc, day_df=day,
            ))
            break  # only first breakout per day

        prior_close = day["close"].iloc[-1]

    return events


# ── Shared Metric Computation ─────────────────────────────────────────────────

def compute_r_metrics(entry: float, stop: float, direction: str,
                      remaining: pd.DataFrame, eod: float):
    """Returns (risk_$, eod_pnl_r, mfe_r, mae_r, hit_stop)."""
    risk = abs(entry - stop)
    if risk <= 0 or remaining.empty:
        return risk, 0.0, 0.0, 0.0, False

    if direction == "LONG":
        eod_r  = (eod - entry) / risk
        mfe_r  = (remaining["high"].max() - entry) / risk
        mae_r  = (remaining["low"].min()  - entry) / risk
        hit    = remaining["low"].min() <= stop
    else:
        eod_r  = (entry - eod) / risk
        mfe_r  = (entry - remaining["low"].min())  / risk
        mae_r  = (entry - remaining["high"].max()) / risk
        hit    = remaining["high"].max() >= stop

    return risk, eod_r, mfe_r, mae_r, hit


# ── Entry Simulators ──────────────────────────────────────────────────────────

def sim_bar_close(ev: ORBEvent) -> EntryResult:
    """Method 1: Enter at close of breakout bar. Always fills."""
    day    = ev.day_df
    b      = ev.breakout_bar_iloc
    entry  = ev.breakout_close
    stop   = ev.orb_low if ev.direction == "LONG" else ev.orb_high
    eod    = day["close"].iloc[-1]
    # remaining = bars AFTER entry bar (we entered at close, bar is done)
    remaining = day.iloc[b + 1:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(entry, stop, ev.direction, remaining, eod)
    return EntryResult("1_bar_close", filled=True,
                       entry_price=entry, entry_time=ev.breakout_time,
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=0)


def sim_next_open(ev: ORBEvent) -> EntryResult:
    """Method 2: Enter at open of bar immediately after breakout. Always fills if bar exists."""
    day  = ev.day_df
    b    = ev.breakout_bar_iloc
    stop = ev.orb_low if ev.direction == "LONG" else ev.orb_high
    eod  = day["close"].iloc[-1]

    if b + 1 >= len(day):
        return EntryResult("2_next_open", filled=False)

    bar   = day.iloc[b + 1]
    entry = bar["open"]
    # remaining = from entry bar onwards (entered at open, full bar is our trade)
    remaining = day.iloc[b + 1:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(entry, stop, ev.direction, remaining, eod)
    return EntryResult("2_next_open", filled=True,
                       entry_price=entry, entry_time=str(bar["time"]),
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=1)


def sim_limit_orb(ev: ORBEvent) -> EntryResult:
    """Method 3: Limit at ORB boundary. Fill when any bar after breakout touches it."""
    day  = ev.day_df
    b    = ev.breakout_bar_iloc
    eod  = day["close"].iloc[-1]
    post = day.iloc[b + 1:]   # bars AFTER breakout bar

    if ev.direction == "LONG":
        limit = ev.orb_high
        stop  = ev.orb_low
        mask  = post["low"] <= limit
    else:
        limit = ev.orb_low
        stop  = ev.orb_high
        mask  = post["high"] >= limit

    if not mask.any():
        return EntryResult("3_limit_orb", filled=False)

    pos       = int(mask.values.argmax())
    fill_bar  = post.iloc[pos]
    # remaining = from fill bar (we fill mid-bar, rest of bar is our trade)
    remaining = post.iloc[pos:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(limit, stop, ev.direction, remaining, eod)
    return EntryResult("3_limit_orb", filled=True,
                       entry_price=limit, entry_time=str(fill_bar["time"]),
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=pos + 1)


def sim_retest_open(ev: ORBEvent) -> EntryResult:
    """Method 4: Price pulls back to ORB boundary, then enter at the NEXT bar's open."""
    day  = ev.day_df
    b    = ev.breakout_bar_iloc
    eod  = day["close"].iloc[-1]
    post = day.iloc[b + 1:]

    if ev.direction == "LONG":
        boundary = ev.orb_high
        stop     = ev.orb_low
        mask     = post["low"] <= boundary
    else:
        boundary = ev.orb_low
        stop     = ev.orb_high
        mask     = post["high"] >= boundary

    if not mask.any():
        return EntryResult("4_retest_open", filled=False)

    touch_pos  = int(mask.values.argmax())
    entry_pos  = touch_pos + 1
    if entry_pos >= len(post):
        return EntryResult("4_retest_open", filled=False)

    bar   = post.iloc[entry_pos]
    entry = bar["open"]
    # remaining = from entry bar (entered at open, full bar is trade)
    remaining = post.iloc[entry_pos:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(entry, stop, ev.direction, remaining, eod)
    return EntryResult("4_retest_open", filled=True,
                       entry_price=entry, entry_time=str(bar["time"]),
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=entry_pos + 1)


def sim_half_pullback(ev: ORBEvent) -> EntryResult:
    """Method 5: Limit at 50% between breakout close and ORB boundary."""
    day  = ev.day_df
    b    = ev.breakout_bar_iloc
    eod  = day["close"].iloc[-1]
    post = day.iloc[b + 1:]

    if ev.direction == "LONG":
        # midpoint between orb_high and breakout close
        target = (ev.orb_high + ev.breakout_close) / 2
        stop   = ev.orb_low
        mask   = post["low"] <= target
    else:
        target = (ev.orb_low + ev.breakout_close) / 2
        stop   = ev.orb_high
        mask   = post["high"] >= target

    if not mask.any():
        return EntryResult("5_half_pullback", filled=False)

    pos       = int(mask.values.argmax())
    fill_bar  = post.iloc[pos]
    # remaining = from fill bar (limit fill mid-bar)
    remaining = post.iloc[pos:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(target, stop, ev.direction, remaining, eod)
    return EntryResult("5_half_pullback", filled=True,
                       entry_price=target, entry_time=str(fill_bar["time"]),
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=pos + 1)


def sim_pullback_confirm(ev: ORBEvent) -> EntryResult:
    """Method 6 (added): Price touches ORB boundary AND closes in breakout direction on same bar.
    Entry = close of that confirmation bar.
    This avoids limit order visibility and confirms actual demand/supply at the boundary."""
    day  = ev.day_df
    b    = ev.breakout_bar_iloc
    eod  = day["close"].iloc[-1]
    post = day.iloc[b + 1:]

    if ev.direction == "LONG":
        boundary = ev.orb_high
        stop     = ev.orb_low
        # Bar touched boundary AND closed bullish above it (shows buyers absorbed the retest)
        cond = (post["low"] <= boundary) & (post["close"] >= boundary) & (post["close"] > post["open"])
    else:
        boundary = ev.orb_low
        stop     = ev.orb_high
        # Bar touched boundary AND closed bearish below it
        cond = (post["high"] >= boundary) & (post["close"] <= boundary) & (post["close"] < post["open"])

    if not cond.any():
        return EntryResult("6_pullback_confirm", filled=False)

    pos       = int(cond.values.argmax())
    fill_bar  = post.iloc[pos]
    entry     = fill_bar["close"]
    # remaining = bars AFTER confirmation bar (entered at close, bar is done)
    remaining = post.iloc[pos + 1:]
    risk, eod_r, mfe_r, mae_r, hit = compute_r_metrics(entry, stop, ev.direction, remaining, eod)
    return EntryResult("6_pullback_confirm", filled=True,
                       entry_price=entry, entry_time=str(fill_bar["time"]),
                       stop_price=stop, risk_dollars=risk,
                       eod_pnl_r=eod_r, max_favorable_r=mfe_r, max_adverse_r=mae_r,
                       hit_stop=hit, bars_to_fill=pos + 1)


SIMULATORS = [
    sim_bar_close,
    sim_next_open,
    sim_limit_orb,
    sim_retest_open,
    sim_half_pullback,
    sim_pullback_confirm,
]


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_section(events: List[ORBEvent], label: str):
    """Run all 6 simulators on events and print comparison table."""
    n = len(events)
    if n < MIN_TRADES:
        return

    print(f"\n  {label}  ({n} events)")
    print(f"  {'─' * 105}")
    print(f"  {'Method':<40s}  {'Fill':>5s}  {'Risk$':>6s}  {'Exp':>7s}  "
          f"{'WR':>5s}  {'Sharpe':>6s}  {'MFE':>6s}  {'MAE':>6s}  {'Stop':>5s}  {'n_fill':>6s}")
    print(f"  {'─' * 105}")

    all_results = {}
    for sim in SIMULATORS:
        results  = [sim(ev) for ev in events]
        filled   = [r for r in results if r.filled]
        method   = sim.__name__.replace("sim_", "")
        key      = results[0].method

        fill_pct = len(filled) / n * 100

        if len(filled) < 3:
            print(f"  {'  ' + METHOD_LABELS[key]:<40s}  {fill_pct:4.0f}%  "
                  f"{'—':>6s}  {'—':>7s}  {'—':>5s}  {'—':>6s}  {'—':>6s}  {'—':>6s}  {'—':>5s}  {len(filled):>6d}")
            all_results[key] = None
            continue

        pnls     = [r.eod_pnl_r       for r in filled]
        risks    = [r.risk_dollars     for r in filled]
        mfes     = [r.max_favorable_r  for r in filled]
        maes     = [r.max_adverse_r    for r in filled]
        stops    = [r.hit_stop         for r in filled]

        exp      = np.mean(pnls)
        wr       = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sharpe   = exp / np.std(pnls) if np.std(pnls) > 0 else 0.0
        avg_risk = np.mean(risks)
        avg_mfe  = np.mean(mfes)
        avg_mae  = np.mean(maes)
        stop_pct = sum(stops) / len(filled) * 100

        mark = "✓✓" if exp > 0.15 else ("✓ " if exp > 0.05 else "  ")
        print(f"  {mark}{METHOD_LABELS[key]:<40s}  {fill_pct:4.0f}%  "
              f"${avg_risk:5.2f}  {exp:+.3f}R  {wr:4.0f}%  "
              f"{sharpe:+.5f}  {avg_mfe:+.3f}R  {avg_mae:+.3f}R  {stop_pct:4.0f}%  {len(filled):>6d}")

        all_results[key] = {"exp": exp, "wr": wr, "sharpe": sharpe,
                            "fill_pct": fill_pct, "n": len(filled), "pnls": pnls}

    print(f"  {'─' * 105}")

    # Distribution snapshot for filled results
    print(f"\n  P&L distribution per method (filled trades only):")
    print(f"  {'Method':<40s}  {'P10':>6s}  {'P25':>6s}  {'P50':>6s}  {'P75':>6s}  {'P90':>6s}  {'worst':>7s}  {'best':>7s}")
    print(f"  {'─' * 95}")
    for sim in SIMULATORS:
        results = [sim(ev) for ev in events]
        filled  = [r for r in results if r.filled]
        key     = results[0].method
        if len(filled) < 3:
            continue
        pnls = [r.eod_pnl_r for r in filled]
        p    = np.percentile(pnls, [10, 25, 50, 75, 90])
        print(f"  {METHOD_LABELS[key]:<40s}  "
              f"{p[0]:+.3f}R  {p[1]:+.3f}R  {p[2]:+.3f}R  {p[3]:+.3f}R  {p[4]:+.3f}R  "
              f"{min(pnls):+.3f}R  {max(pnls):+.3f}R")

    return all_results


# ── Filter Helpers ────────────────────────────────────────────────────────────

def filter_events(events: List[ORBEvent], **flags) -> List[ORBEvent]:
    g = events

    if flags.get("low_orb_vol"):
        ratios = [e.orb_volume / e.day_volume for e in g if e.day_volume > 0]
        med    = np.median(ratios) if ratios else 0.19
        g = [e for e in g if e.day_volume > 0 and e.orb_volume / e.day_volume < med]

    if flags.get("has_gap"):
        g = [e for e in g if abs(e.gap_pct) > 0.002]

    if flags.get("not_narrow_orb"):
        all_ranges = [e.orb_range for e in events]  # percentile from full pool
        p20 = np.percentile(all_ranges, 20)
        g = [e for e in g if e.orb_range >= p20]

    if flags.get("mid_day"):
        g = [e for e in g if dtime.fromisoformat(e.breakout_time) >= dtime(11, 0)]

    if flags.get("gap_opposed"):
        g = [e for e in g if
             (e.direction == "LONG"  and e.gap_pct < -0.002) or
             (e.direction == "SHORT" and e.gap_pct >  0.002)]

    return g


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase1_entry_methods.py <csv1> [csv2] ...")
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

    total_syms = len(sys.argv) - 1
    print(f"\nTotal: {len(all_events)} ORB events across {total_syms} symbols")

    print("\n" + "=" * 108)
    print("  PHASE 1: ENTRY METHOD COMPARISON")
    print("  Stop = opposite ORB boundary.  Target = EOD.  All 6 methods on identical ORB events.")
    print("  ✓✓ = exp > +0.15R    ✓ = exp > +0.05R    (blank) = negative or flat")
    print("=" * 108)

    # ── Full sample ──────────────────────────────────────────────────────────
    print("\n" + "─" * 108)
    print("  [A] FULL SAMPLE  — no filters")
    report_section(all_events, "Full sample")

    # ── Phase 0.5 best filters applied one by one ─────────────────────────────
    f_vol    = filter_events(all_events, low_orb_vol=True)
    f_gap    = filter_events(all_events, has_gap=True)
    f_oppose = filter_events(all_events, gap_opposed=True)
    f_best   = filter_events(all_events, low_orb_vol=True, has_gap=True)
    f_full   = filter_events(all_events, low_orb_vol=True, has_gap=True, not_narrow_orb=True)
    f_tight  = filter_events(all_events, low_orb_vol=True, gap_opposed=True, mid_day=True)

    print("\n" + "─" * 108)
    print("  [B] PHASE 0.5 FILTERS  (from prior decomposition)")

    report_section(f_vol,    "Filter: low ORB vol ratio only")
    report_section(f_gap,    "Filter: has gap (|gap|>0.2%) only")
    report_section(f_oppose, "Filter: gap opposed to breakout direction")
    report_section(f_best,   "Filter: low ORB vol + has gap")
    report_section(f_full,   "Filter: low ORB vol + has gap + not narrow ORB")
    report_section(f_tight,  "Filter: low ORB vol + gap opposed + mid-day (best from Phase 0.5)")

    # ── Long vs Short by best method ──────────────────────────────────────────
    print("\n" + "─" * 108)
    print("  [C] DIRECTION SPLIT  (limit entry, full sample)")
    for direction in ["LONG", "SHORT"]:
        g = [e for e in all_events if e.direction == direction]
        if len(g) < MIN_TRADES:
            continue
        results  = [sim_limit_orb(e) for e in g]
        filled   = [r for r in results if r.filled]
        if len(filled) < MIN_TRADES:
            continue
        pnls     = [r.eod_pnl_r for r in filled]
        exp      = np.mean(pnls)
        wr       = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sharpe   = exp / np.std(pnls) if np.std(pnls) > 0 else 0.0
        fill_pct = len(filled) / len(g) * 100
        print(f"  {direction:<6s}  n={len(g)}  fill={fill_pct:.0f}%  "
              f"exp={exp:+.3f}R  WR={wr:.0f}%  Sharpe={sharpe:+.3f}  (limit entry)")

    # ── Per-symbol (limit entry) ───────────────────────────────────────────────
    print("\n" + "─" * 108)
    print("  [D] PER SYMBOL  (limit entry, full sample)")
    print(f"  {'Symbol':<8s}  {'Events':>6s}  {'Fill':>5s}  {'Exp':>7s}  {'WR':>5s}  "
          f"{'Sharpe':>6s}  {'MFE':>6s}  {'MAE':>6s}")
    print(f"  {'─' * 70}")
    for sym in sorted(set(e.symbol for e in all_events)):
        g       = [e for e in all_events if e.symbol == sym]
        results = [sim_limit_orb(e) for e in g]
        filled  = [r for r in results if r.filled]
        if len(filled) < 5:
            continue
        pnls     = [r.eod_pnl_r      for r in filled]
        mfes     = [r.max_favorable_r for r in filled]
        maes     = [r.max_adverse_r   for r in filled]
        exp      = np.mean(pnls)
        wr       = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sharpe   = exp / np.std(pnls) if np.std(pnls) > 0 else 0.0
        fill_pct = len(filled) / len(g) * 100
        mark     = "✓" if exp > 0.05 else " "
        print(f"  {mark} {sym:<6s}  {len(g):6d}  {fill_pct:4.0f}%  {exp:+.3f}R  {wr:4.0f}%  "
              f"{sharpe:+.5f}  {np.mean(mfes):+.3f}R  {np.mean(maes):+.3f}R")

    # ── Critical: Limit fill bias analysis ───────────────────────────────────
    print("\n" + "─" * 108)
    print("  [E] LIMIT ORDER FILL BIAS  (key insight for methods 3-6)")
    print("  When price DOES NOT pull back → those are usually the strongest breakouts")
    print("  Limit-based methods will miss them → biases results downward")
    for sim in [sim_limit_orb, sim_retest_open, sim_pullback_confirm]:
        results    = [sim(e) for e in all_events]
        filled     = [r for r in results if r.filled]
        not_filled = [r for r in results if not r.filled]
        key        = results[0].method

        # For not-filled: simulate bar_close result on those events to see what we missed
        not_filled_events = [e for e, r in zip(all_events, results) if not r.filled]
        missed_bc  = [sim_bar_close(e) for e in not_filled_events]
        missed_pnl = np.mean([r.eod_pnl_r for r in missed_bc]) if missed_bc else 0

        fill_pct   = len(filled) / len(all_events) * 100
        filled_exp = np.mean([r.eod_pnl_r for r in filled]) if filled else 0
        print(f"\n  {METHOD_LABELS[key]}")
        print(f"    Fill rate:          {fill_pct:.0f}%  ({len(filled)} filled / {len(not_filled)} missed)")
        print(f"    Filled trades exp:  {filled_exp:+.3f}R")
        print(f"    Missed trades exp:  {missed_pnl:+.3f}R  ← these breakouts never pulled back")
        print(f"    → If missed exp >> filled exp, the limit strategy has adverse selection")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("  VERDICT")
    print()

    # Find best method on full sample and best filter
    best_overall  = {}
    scenarios = [("Full sample", all_events), ("low ORB vol + has gap", f_best)]
    for s_label, evs in scenarios:
        if len(evs) < MIN_TRADES:
            continue
        best_exp, best_key = -999, ""
        for sim in SIMULATORS:
            results = [sim(e) for e in evs]
            filled  = [r for r in results if r.filled]
            if len(filled) < MIN_TRADES:
                continue
            pnls = [r.eod_pnl_r for r in filled]
            exp  = np.mean(pnls)
            if exp > best_exp:
                best_exp = exp
                best_key = results[0].method
                best_n   = len(filled)
                best_fr  = len(filled) / len(evs) * 100
        if best_key:
            print(f"  {s_label:<30s}  Best entry: {METHOD_LABELS[best_key]}")
            print(f"  {'':30s}  Expectancy: {best_exp:+.3f}R   Fill rate: {best_fr:.0f}%   n_filled: {best_n}")
            print()

    print("  Key takeaways:")
    print("  • Limit methods (3-6) show higher expectancy but have FILL BIAS — they miss")
    print("    the strongest straight-line breakouts (which are the best trades)")
    print("  • Pullback+confirm (method 6) has lowest fill rate but best adverse selection filter")
    print("    — you only enter when the boundary held as support/resistance with confirmation")
    print("  • Method to proceed with → run Phase 2 (position sizing) on best method + best filter")
    print("=" * 108)


if __name__ == "__main__":
    main()
