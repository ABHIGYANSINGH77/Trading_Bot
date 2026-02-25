"""Phase 0: Raw ORB Edge Test.

Question: Does an ORB breakout on AAPL/NVDA/MSFT/AMZN/GOOG at 15m
          predict directional movement AT ALL?

Method:
  - No indicators, no scoring, no regime, no filters.
  - ORB = high/low of first 30 minutes (bars 9:30 + 9:45).
  - Breakout = first bar that CLOSES beyond ORB boundary.
  - Entry = close of breakout bar (current method — we measure how bad this is).
  - Stop = opposite side of ORB range.
  - Target = none (we measure where price goes by EOD).
  - We record everything and decompose by every dimension.

"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ─── Configuration ───────────────────────────────────────

ORB_MINUTES = 30          # First 30 min = opening range
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 45     # Last usable bar close (15:45 for 15m bars → 16:00)


# ─── Data Structures ────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    date: str
    direction: str            # "LONG" or "SHORT"
    orb_high: float
    orb_low: float
    orb_range: float
    orb_volume: float         # Total volume during ORB period
    entry_price: float        # Close of breakout bar
    entry_time: str
    stop_price: float         # Opposite side of ORB
    risk: float               # |entry - stop|
    chase: float              # How far past ORB boundary we entered
    chase_pct: float          # chase / orb_range
    eod_price: float          # Last bar close of the day
    eod_pnl_r: float          # (eod - entry) / risk, signed for direction
    max_favorable_r: float    # Best price reached (in R)
    max_adverse_r: float      # Worst price reached (in R, negative)
    hit_stop: bool            # Did price reach the stop level?
    gap_pct: float            # Overnight gap %
    bars_after_entry: int     # How many bars remained in the day
    breakout_bar_range: float # High-low of the breakout bar itself
    day_volume: float         # Total volume for the full day
    prior_close: float        # Previous day's close


# ─── Load and Parse CSV ─────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a cached CSV file, parse timestamps, filter market hours."""
    df = pd.read_csv(path)

    # Rename 'date' column to 'timestamp' for internal use
    df = df.rename(columns={"date": "timestamp"})

    # Extract symbol from filename
    fname = Path(path).stem
    symbol = fname.split("_")[0]
    df["symbol"] = symbol

    # Parse timestamps (handles timezone-aware strings like "2025-07-01 09:30:00-04:00")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")

    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    # Filter to market hours only
    from datetime import time as dtime
    market_open = dtime(MARKET_OPEN_HOUR, MARKET_OPEN_MIN)
    market_close = dtime(16, 0)
    df = df[(df["time"] >= market_open) & (df["time"] < market_close)].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─── Core Logic ──────────────────────────────────────────

def find_orb_trades(df: pd.DataFrame) -> List[Trade]:
    """Find all ORB breakout trades in the data.

    For each trading day:
    1. Define ORB from first 30 minutes
    2. Find first bar that closes beyond ORB boundary
    3. Record everything about what happens after
    """
    from datetime import time as dtime

    symbol = df["symbol"].iloc[0]
    trades = []
    dates = sorted(df["date"].unique())

    orb_end_time = dtime(10, 0)  # 9:30 + 30 min = 10:00

    prior_close = None

    for day_date in dates:
        day_df = df[df["date"] == day_date].sort_values("timestamp").reset_index(drop=True)
        if len(day_df) < 4:
            if len(day_df) > 0:
                prior_close = day_df["close"].iloc[-1]
            continue

        # ── Define ORB ──
        orb_bars = day_df[day_df["time"] < orb_end_time]
        if len(orb_bars) < 1:
            prior_close = day_df["close"].iloc[-1]
            continue

        orb_high = orb_bars["high"].max()
        orb_low = orb_bars["low"].min()
        orb_range = orb_high - orb_low
        orb_volume = orb_bars["volume"].sum()

        if orb_range <= 0:
            prior_close = day_df["close"].iloc[-1]
            continue

        # ── Gap ──
        day_open = day_df["open"].iloc[0]
        gap_pct = (day_open - prior_close) / prior_close if prior_close and prior_close > 0 else 0.0

        # ── Find first breakout after ORB ──
        post_orb = day_df[day_df["time"] >= orb_end_time]
        if len(post_orb) < 2:
            prior_close = day_df["close"].iloc[-1]
            continue

        breakout_found = False
        for idx, row in post_orb.iterrows():
            close = row["close"]

            if close > orb_high:
                # LONG breakout
                entry_price = close
                stop_price = orb_low
                risk = entry_price - stop_price
                chase = entry_price - orb_high
                direction = "LONG"
                breakout_found = True

            elif close < orb_low:
                # SHORT breakout
                entry_price = close
                stop_price = orb_high
                risk = stop_price - entry_price
                chase = orb_low - entry_price
                direction = "SHORT"
                breakout_found = True

            if breakout_found:
                if risk <= 0:
                    break

                # ── Measure what happens after entry ──
                entry_iloc = day_df.index.get_loc(idx)
                remaining = day_df.iloc[entry_iloc:]
                if len(remaining) < 1:
                    break

                eod_price = day_df["close"].iloc[-1]
                day_volume = day_df["volume"].sum()

                if direction == "LONG":
                    eod_pnl_r = (eod_price - entry_price) / risk
                    mfe_price = remaining["high"].max()
                    mae_price = remaining["low"].min()
                    max_favorable_r = (mfe_price - entry_price) / risk
                    max_adverse_r = (mae_price - entry_price) / risk  # negative if below entry
                    hit_stop = mae_price <= stop_price
                else:
                    eod_pnl_r = (entry_price - eod_price) / risk
                    mfe_price = remaining["low"].min()
                    mae_price = remaining["high"].max()
                    max_favorable_r = (entry_price - mfe_price) / risk
                    max_adverse_r = (entry_price - mae_price) / risk  # negative if above entry
                    hit_stop = mae_price >= stop_price

                trades.append(Trade(
                    symbol=symbol,
                    date=str(day_date),
                    direction=direction,
                    orb_high=orb_high,
                    orb_low=orb_low,
                    orb_range=orb_range,
                    orb_volume=orb_volume,
                    entry_price=entry_price,
                    entry_time=str(row["time"]),
                    stop_price=stop_price,
                    risk=risk,
                    chase=chase,
                    chase_pct=chase / orb_range if orb_range > 0 else 0,
                    eod_price=eod_price,
                    eod_pnl_r=eod_pnl_r,
                    max_favorable_r=max_favorable_r,
                    max_adverse_r=max_adverse_r,
                    hit_stop=hit_stop,
                    gap_pct=gap_pct,
                    bars_after_entry=len(remaining) - 1,
                    breakout_bar_range=row["high"] - row["low"],
                    day_volume=day_volume,
                    prior_close=prior_close or day_open,
                ))
                break  # Only first breakout per day

        prior_close = day_df["close"].iloc[-1]

    return trades


# ─── Reporting ───────────────────────────────────────────

def report_group(trades: List[Trade], label: str, indent: str = ""):
    """Print stats for a group of trades."""
    if not trades:
        print(f"{indent}{label}: (no trades)")
        return

    n = len(trades)
    eod_pnls = [t.eod_pnl_r for t in trades]
    wins = sum(1 for p in eod_pnls if p > 0)
    losses = sum(1 for p in eod_pnls if p <= 0)
    avg_r = np.mean(eod_pnls)
    med_r = np.median(eod_pnls)
    std_r = np.std(eod_pnls)
    win_rate = wins / n * 100

    avg_win = np.mean([p for p in eod_pnls if p > 0]) if wins > 0 else 0
    avg_loss = np.mean([p for p in eod_pnls if p <= 0]) if losses > 0 else 0

    mfe = [t.max_favorable_r for t in trades]
    mae = [t.max_adverse_r for t in trades]
    chases = [t.chase_pct for t in trades]
    stop_hits = sum(1 for t in trades if t.hit_stop)

    print(f"{indent}{label}: {n} trades")
    print(f"{indent}  Expectancy:   {avg_r:+.3f}R  (median {med_r:+.3f}R, std {std_r:.3f}R)")
    print(f"{indent}  Win rate:     {win_rate:.1f}%  ({wins}W / {losses}L)")
    print(f"{indent}  Avg win:      {avg_win:+.3f}R   Avg loss: {avg_loss:+.3f}R")
    print(f"{indent}  Avg MFE:      {np.mean(mfe):+.3f}R   Avg MAE: {np.mean(mae):+.3f}R")
    print(f"{indent}  Stop hit:     {stop_hits}/{n} ({stop_hits/n*100:.0f}%)")
    print(f"{indent}  Chase:        avg {np.mean(chases):.2f}x ORB  "
          f"(med {np.median(chases):.2f}x, max {np.max(chases):.2f}x)")


def decompose(trades: List[Trade]):
    """Break down results by every dimension."""
    from datetime import time as dtime

    print("\n" + "=" * 65)
    print("  PHASE 0: RAW ORB BREAKOUT EDGE TEST")
    print("=" * 65)

    report_group(trades, "ALL TRADES")

    # ── By Symbol ──
    print(f"\n{'─' * 55}")
    print("  BY SYMBOL")
    symbols = sorted(set(t.symbol for t in trades))
    for sym in symbols:
        group = [t for t in trades if t.symbol == sym]
        report_group(group, sym, "    ")

    # ── By Direction ──
    print(f"\n{'─' * 55}")
    print("  BY DIRECTION")
    for d in ["LONG", "SHORT"]:
        group = [t for t in trades if t.direction == d]
        report_group(group, d, "    ")

    # ── By Time of Day (entry time) ──
    print(f"\n{'─' * 55}")
    print("  BY ENTRY TIME")
    time_buckets = [
        ("10:00-10:30", dtime(10, 0), dtime(10, 30)),
        ("10:30-11:00", dtime(10, 30), dtime(11, 0)),
        ("11:00-12:00", dtime(11, 0), dtime(12, 0)),
        ("12:00-13:00", dtime(12, 0), dtime(13, 0)),
        ("13:00-14:00", dtime(13, 0), dtime(14, 0)),
        ("14:00-15:00", dtime(14, 0), dtime(15, 0)),
    ]
    for label, t_start, t_end in time_buckets:
        group = [t for t in trades
                 if dtime.fromisoformat(t.entry_time) >= t_start
                 and dtime.fromisoformat(t.entry_time) < t_end]
        report_group(group, label, "    ")

    # ── By ORB Width (narrow vs wide) ──
    print(f"\n{'─' * 55}")
    print("  BY ORB WIDTH")
    orb_ranges = [t.orb_range for t in trades]
    if orb_ranges:
        p25, p50, p75 = np.percentile(orb_ranges, [25, 50, 75])
        for label, lo, hi in [("Narrow (Q1)", 0, p25), ("Medium (Q2-Q3)", p25, p75),
                               ("Wide (Q4)", p75, float('inf'))]:
            group = [t for t in trades if lo <= t.orb_range < hi]
            if group:
                avg_range = np.mean([t.orb_range for t in group])
                report_group(group, f"{label} (avg ${avg_range:.2f})", "    ")

    # ── By ORB Volume ──
    print(f"\n{'─' * 55}")
    print("  BY ORB VOLUME")
    orb_vols = [t.orb_volume for t in trades]
    if orb_vols:
        med_vol = np.median(orb_vols)
        for label, filt in [("Below median vol", lambda t: t.orb_volume < med_vol),
                             ("Above median vol", lambda t: t.orb_volume >= med_vol)]:
            group = [t for t in trades if filt(t)]
            report_group(group, label, "    ")

    # ── By Gap Direction ──
    print(f"\n{'─' * 55}")
    print("  BY GAP")
    for label, filt in [
        ("Gap up (>0.2%)", lambda t: t.gap_pct > 0.002),
        ("Flat gap", lambda t: -0.002 <= t.gap_pct <= 0.002),
        ("Gap down (<-0.2%)", lambda t: t.gap_pct < -0.002),
    ]:
        group = [t for t in trades if filt(t)]
        report_group(group, label, "    ")

    # ── Gap Aligned With Direction ──
    print(f"\n{'─' * 55}")
    print("  BY GAP ALIGNMENT WITH DIRECTION")
    for label, filt in [
        ("Gap aligned", lambda t: (t.direction == "LONG" and t.gap_pct > 0.001) or
                                   (t.direction == "SHORT" and t.gap_pct < -0.001)),
        ("Gap opposed", lambda t: (t.direction == "LONG" and t.gap_pct < -0.001) or
                                   (t.direction == "SHORT" and t.gap_pct > 0.001)),
        ("Gap neutral", lambda t: -0.001 <= t.gap_pct <= 0.001),
    ]:
        group = [t for t in trades if filt(t)]
        report_group(group, label, "    ")

    # ── Chase Analysis (the key entry timing question) ──
    print(f"\n{'─' * 55}")
    print("  CHASE ANALYSIS (entry timing problem)")
    chases = [t.chase_pct for t in trades]
    print(f"  Chase = how far past ORB boundary the entry is, as fraction of ORB range")
    print(f"  Chase 0.0 = entered exactly at boundary")
    print(f"  Chase 0.5 = entered half a range past boundary")
    print(f"  Chase 1.0 = entered a full ORB range past boundary")
    print(f"")
    for label, lo, hi in [
        ("Low chase (<0.2x)", 0, 0.2),
        ("Med chase (0.2-0.5x)", 0.2, 0.5),
        ("High chase (0.5-1.0x)", 0.5, 1.0),
        ("Extreme chase (>1.0x)", 1.0, float('inf')),
    ]:
        group = [t for t in trades if lo <= t.chase_pct < hi]
        report_group(group, label, "    ")

    # ── MFE/MAE Distribution ──
    print(f"\n{'─' * 55}")
    print("  MFE/MAE DISTRIBUTION (max excursion before EOD)")
    mfes = [t.max_favorable_r for t in trades]
    maes = [t.max_adverse_r for t in trades]
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct:02d}:  MFE {np.percentile(mfes, pct):+.2f}R   "
              f"MAE {np.percentile(maes, pct):+.2f}R")
    print(f"  Mean: MFE {np.mean(mfes):+.2f}R   MAE {np.mean(maes):+.2f}R")

    # ── Stop Analysis ──
    print(f"\n{'─' * 55}")
    print("  STOP ANALYSIS")
    stop_hit = [t for t in trades if t.hit_stop]
    no_stop = [t for t in trades if not t.hit_stop]
    print(f"  Stop hit: {len(stop_hit)}/{len(trades)} ({len(stop_hit)/len(trades)*100:.0f}%)")
    if stop_hit:
        # Among trades that hit stop, what was max favorable first?
        mfe_before_stop = [t.max_favorable_r for t in stop_hit]
        print(f"  Trades that hit stop — avg MFE before stopping: {np.mean(mfe_before_stop):+.2f}R")
        print(f"    → These trades went {np.mean(mfe_before_stop):.2f}R in your favor before reversing")
    if no_stop:
        print(f"  Trades that survived — avg EOD P&L: {np.mean([t.eod_pnl_r for t in no_stop]):+.2f}R")

    # ── Hypothetical: What if we entered at ORB boundary instead? ──
    print(f"\n{'─' * 55}")
    print("  HYPOTHETICAL: ENTRY AT ORB BOUNDARY (0 chase)")
    # If entry was at ORB boundary, risk = ORB range, everything else same
    hypo_pnls = []
    for t in trades:
        if t.direction == "LONG":
            hypo_entry = t.orb_high
            hypo_risk = t.orb_range
            hypo_pnl = (t.eod_price - hypo_entry) / hypo_risk
        else:
            hypo_entry = t.orb_low
            hypo_risk = t.orb_range
            hypo_pnl = (hypo_entry - t.eod_price) / hypo_risk
        hypo_pnls.append(hypo_pnl)

    hypo_wins = sum(1 for p in hypo_pnls if p > 0)
    print(f"  Hypothetical expectancy: {np.mean(hypo_pnls):+.3f}R  "
          f"(actual: {np.mean([t.eod_pnl_r for t in trades]):+.3f}R)")
    print(f"  Hypothetical win rate:   {hypo_wins/len(trades)*100:.1f}%  "
          f"(actual: {sum(1 for t in trades if t.eod_pnl_r > 0)/len(trades)*100:.1f}%)")
    print(f"  Entry improvement:       {np.mean(hypo_pnls) - np.mean([t.eod_pnl_r for t in trades]):+.3f}R")

    print(f"\n{'=' * 65}")
    pass_fail = "✓ EDGE EXISTS" if np.mean([t.eod_pnl_r for t in trades]) > 0.05 else "✗ NO EDGE (expectancy ≤ +0.05R)"
    print(f"  VERDICT: {pass_fail}")
    print(f"  Next step: {'Phase 0.5 decomposition' if 'EXISTS' in pass_fail else 'Try different instrument/timeframe/event'}")
    print(f"{'=' * 65}")


# ─── Main ────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase0_raw_edge.py <csv_file1> [csv_file2] ...")
        print("       python3 phase0_raw_edge.py ./data/cache/*_15_mins.csv")
        sys.exit(1)

    all_trades = []
    for path in sys.argv[1:]:
        print(f"Loading {path}...")
        try:
            df = load_csv(path)
            symbol = df["symbol"].iloc[0]
            print(f"  {symbol}: {len(df)} bars, {df['date'].nunique()} days")
            trades = find_orb_trades(df)
            print(f"  {symbol}: {len(trades)} breakout trades found")
            all_trades.extend(trades)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")

    if not all_trades:
        print("No trades found. Check data files.")
        sys.exit(1)

    print(f"\nTotal: {len(all_trades)} trades across {len(sys.argv)-1} symbols")
    decompose(all_trades)


if __name__ == "__main__":
    main()