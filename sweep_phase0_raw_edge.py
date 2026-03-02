"""sweep_phase0_raw_edge.py — Event Type 2: Session Sweep + Rejection

Phase 0: Raw Edge Test

Hypothesis:
  When price wicks through the prior day's high (PDH) or prior day's low (PDL)
  and then rejects back inside, the liquidity grab is complete and price reverses.

This is fundamentally DIFFERENT from ORB:
  ORB = momentum play (break → continue)
  Sweep = mean reversion play (break → reverse)
  → entry logic, exit logic, and useful filters may be the opposite of ORB

Setup rules:
  PDH Sweep → SHORT:
    1. A bar's HIGH exceeds PDH (wick above — stops triggered above PDH)
    2. That bar CLOSES at or below PDH (rejection candle — bulls rejected)
    3. Within the next 1–3 bars: bearish confirmation bar
       (close < open  AND  close < PDH)
    Entry: close of confirmation bar
    Stop:  highest high of the sweep bar (wick extreme)

  PDL Sweep → LONG:
    1. A bar's LOW goes below PDL (wick below — stops triggered below PDL)
    2. That bar CLOSES at or above PDL (rejection candle — bears rejected)
    3. Within the next 1–3 bars: bullish confirmation bar
       (close > open  AND  close > PDL)
    Entry: close of confirmation bar
    Stop:  lowest low of the sweep bar (wick extreme)

Exit: stop hit OR end of day.  No targets, no trailing — raw edge only.
Filter: none at this stage.

One sweep event per symbol per day (first chronological sweep found).

Usage:
  python3 sweep_phase0_raw_edge.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN  = 30
MAX_CONF_BARS    = 3     # max bars after sweep to look for confirmation (1–3)
MIN_TRADES       = 20    # minimum events to draw conclusions


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SweepEvent:
    # Identity
    symbol:         str
    date:           str
    direction:      str    # "LONG" (PDL sweep) or "SHORT" (PDH sweep)

    # Levels
    pd_level:       float  # PDH (for short) or PDL (for long)
    pd_high:        float
    pd_low:         float
    pd_range:       float  # prior day high - low

    # Sweep bar
    sweep_bar_time: str
    sweep_bar_high: float
    sweep_bar_low:  float
    sweep_extreme:  float  # furthest wick extent (highest high / lowest low)
    overshoot_abs:  float  # wick distance beyond level (absolute $)
    overshoot_pct:  float  # overshoot_abs / pd_level

    # Confirmation bar
    conf_bar_num:   int    # 1, 2, or 3
    conf_bar_time:  str

    # Trade parameters
    entry_price:    float
    stop_price:     float
    risk:           float  # abs(entry - stop)

    # Context
    gap_pct:        float  # today's gap vs prior close
    prior_close:    float
    sweep_hour:     int    # hour of sweep bar (9–15)

    # For simulation
    remaining:      object  # pd.DataFrame slice (bars after confirmation)
    eod_price:      float


@dataclass
class TradeResult:
    pnl_r:       float
    exit_reason: str    # "stop" or "eod"
    bars_held:   int
    mfe_r:       float  # max favorable excursion in R
    mae_r:       float  # max adverse excursion in R (negative)
    hit_stop:    bool


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


# ── Event Detection ───────────────────────────────────────────────────────────

def find_sweep_events(df: pd.DataFrame) -> List[SweepEvent]:
    """Scan each day for the first PDH or PDL sweep + rejection setup."""
    symbol  = df["symbol"].iloc[0]
    events: List[SweepEvent] = []

    pd_high = pd_low = pd_close = pd_open = prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        # Need prior day to have PDH/PDL
        if pd_high is None:
            pd_high     = day["high"].max()
            pd_low      = day["low"].min()
            pd_close    = day["close"].iloc[-1]
            pd_open     = day["open"].iloc[0]
            prior_close = day["close"].iloc[-1]
            continue

        pd_range = pd_high - pd_low
        day_open = day["open"].iloc[0]
        gap_pct  = ((day_open - prior_close) / prior_close
                    if prior_close and prior_close > 0 else 0.0)

        found_event = False

        # Need at least 2 bars total (sweep bar + at least 1 confirmation bar)
        for i in range(len(day) - 1):
            if found_event:
                break

            bar = day.iloc[i]

            # ── PDH Sweep → SHORT ─────────────────────────────────────────────
            if bar["high"] > pd_high and bar["close"] <= pd_high:
                sweep_extreme = float(bar["high"])
                overshoot_abs = sweep_extreme - pd_high
                overshoot_pct = overshoot_abs / pd_high if pd_high > 0 else 0.0

                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    # Bearish confirmation: close < open, still inside (below PDH)
                    if conf["close"] < conf["open"] and conf["close"] < pd_high:
                        entry = float(conf["close"])
                        stop  = sweep_extreme
                        risk  = stop - entry   # for SHORT: stop above entry
                        if risk <= 0:
                            break
                        remaining = day.iloc[i + k + 1:]
                        if remaining.empty:
                            break
                        events.append(SweepEvent(
                            symbol=symbol, date=str(day_date), direction="SHORT",
                            pd_level=pd_high, pd_high=pd_high, pd_low=pd_low,
                            pd_range=pd_range,
                            sweep_bar_time=str(bar["time"]),
                            sweep_bar_high=float(bar["high"]),
                            sweep_bar_low=float(bar["low"]),
                            sweep_extreme=sweep_extreme,
                            overshoot_abs=overshoot_abs,
                            overshoot_pct=overshoot_pct,
                            conf_bar_num=k,
                            conf_bar_time=str(conf["time"]),
                            entry_price=entry, stop_price=stop, risk=risk,
                            gap_pct=gap_pct, prior_close=float(prior_close),
                            sweep_hour=bar["time"].hour,
                            remaining=remaining,
                            eod_price=float(day["close"].iloc[-1]),
                        ))
                        found_event = True
                        break

            # ── PDL Sweep → LONG ──────────────────────────────────────────────
            elif bar["low"] < pd_low and bar["close"] >= pd_low:
                sweep_extreme = float(bar["low"])
                overshoot_abs = pd_low - sweep_extreme
                overshoot_pct = overshoot_abs / pd_low if pd_low > 0 else 0.0

                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    # Bullish confirmation: close > open, still inside (above PDL)
                    if conf["close"] > conf["open"] and conf["close"] > pd_low:
                        entry = float(conf["close"])
                        stop  = sweep_extreme
                        risk  = entry - stop   # for LONG: stop below entry
                        if risk <= 0:
                            break
                        remaining = day.iloc[i + k + 1:]
                        if remaining.empty:
                            break
                        events.append(SweepEvent(
                            symbol=symbol, date=str(day_date), direction="LONG",
                            pd_level=pd_low, pd_high=pd_high, pd_low=pd_low,
                            pd_range=pd_range,
                            sweep_bar_time=str(bar["time"]),
                            sweep_bar_high=float(bar["high"]),
                            sweep_bar_low=float(bar["low"]),
                            sweep_extreme=sweep_extreme,
                            overshoot_abs=overshoot_abs,
                            overshoot_pct=overshoot_pct,
                            conf_bar_num=k,
                            conf_bar_time=str(conf["time"]),
                            entry_price=entry, stop_price=stop, risk=risk,
                            gap_pct=gap_pct, prior_close=float(prior_close),
                            sweep_hour=bar["time"].hour,
                            remaining=remaining,
                            eod_price=float(day["close"].iloc[-1]),
                        ))
                        found_event = True
                        break

        # Update prior day
        pd_high     = day["high"].max()
        pd_low      = day["low"].min()
        pd_close    = day["close"].iloc[-1]
        pd_open     = day["open"].iloc[0]
        prior_close = day["close"].iloc[-1]

    return events


# ── Trade Simulation ──────────────────────────────────────────────────────────

def simulate_trade(ev: SweepEvent) -> TradeResult:
    """Simulate the trade: stop hit or EOD exit, no targets, no trailing."""
    e   = ev.entry_price
    s   = ev.stop_price
    rk  = ev.risk
    d   = ev.direction
    rem = ev.remaining

    max_fav = 0.0
    max_adv = 0.0

    for i, (_, bar) in enumerate(rem.iterrows()):
        # Track MFE / MAE
        if d == "LONG":
            fav = (bar["high"] - e) / rk
            adv = (bar["low"]  - e) / rk
        else:
            fav = (e - bar["low"])  / rk
            adv = (e - bar["high"]) / rk
        max_fav = max(max_fav, fav)
        max_adv = min(max_adv, adv)

        # Stop check (conservative: stop fills at stop price exactly)
        if d == "LONG"  and bar["low"]  <= s:
            pnl = (s - e) / rk     # -1.0R at stop
            return TradeResult(pnl_r=pnl, exit_reason="stop",
                               bars_held=i+1, mfe_r=max_fav, mae_r=max_adv, hit_stop=True)
        if d == "SHORT" and bar["high"] >= s:
            pnl = (e - s) / rk     # -1.0R at stop
            return TradeResult(pnl_r=pnl, exit_reason="stop",
                               bars_held=i+1, mfe_r=max_fav, mae_r=max_adv, hit_stop=True)

    # EOD exit at last bar close
    pnl = ((ev.eod_price - e) / rk if d == "LONG"
           else (e - ev.eod_price) / rk)
    return TradeResult(pnl_r=pnl, exit_reason="eod",
                       bars_held=len(rem), mfe_r=max_fav, mae_r=max_adv, hit_stop=False)


# ── Stats & Display ───────────────────────────────────────────────────────────

def stats(pnls: List[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "sharpe": 0.0}
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    exp    = np.mean(pnls)
    wr     = len(wins) / len(pnls) * 100
    std    = np.std(pnls)
    sharpe = exp / std if std > 0 else 0.0
    return {
        "n":        len(pnls),
        "exp":      exp,
        "wr":       wr,
        "avg_win":  np.mean(wins)   if wins   else 0.0,
        "avg_loss": np.mean(losses) if losses else 0.0,
        "sharpe":   sharpe,
    }


def row(label: str, pnls: List[float],
        min_n: int = MIN_TRADES, indent: str = "  ") -> None:
    if len(pnls) < min_n:
        print(f"{indent}{label:<40s}  n={len(pnls):>3d}  (too few)")
        return
    s    = stats(pnls)
    flag = "★★" if s["exp"] > 0.20 else ("★ " if s["exp"] > 0.10 else "  ")
    print(f"{indent}{label:<40s}  n={s['n']:>4d}  "
          f"exp={s['exp']:+.3f}R  WR={s['wr']:>4.0f}%  "
          f"avgW={s['avg_win']:+.2f}R  avgL={s['avg_loss']:+.2f}R  "
          f"Sharpe={s['sharpe']:+.2f}  {flag}")


# ── Main Output ───────────────────────────────────────────────────────────────

def print_results(events: List[SweepEvent], results: List[TradeResult],
                  date_range: str) -> None:
    pnls = [r.pnl_r for r in results]
    W    = 110

    print("\n" + "=" * W)
    print("  EVENT TYPE 2: SESSION SWEEP + REJECTION — PHASE 0 RAW EDGE TEST")
    print(f"  Hypothesis: PDH/PDL liquidity grab → mean reversion (OPPOSITE of ORB momentum)")
    print(f"  Data range: {date_range}")
    print("=" * W)

    if not events:
        print("\n  No sweep events found. Check data.")
        return

    stops = sum(1 for r in results if r.hit_stop)
    print(f"\n  Sweep events detected:   {len(events)}")
    print(f"  Trades executed:         {len(results)} (100% fill — bar close entry)")
    print(f"  Stop rate:               {stops}/{len(results)}  ({stops/len(results)*100:.0f}% hit stop)")

    # ── OVERALL ───────────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  OVERALL")
    print(f"  {'─' * W}")
    row("ALL SWEEPS", pnls, indent="  ")

    s = stats(pnls)
    if len(pnls) >= MIN_TRADES:
        print(f"\n  MFE avg: {np.mean([r.mfe_r for r in results]):+.3f}R  "
              f"MAE avg: {np.mean([r.mae_r for r in results]):+.3f}R")
        print(f"  Avg hold: {np.mean([r.bars_held for r in results]):.1f} bars "
              f"({np.mean([r.bars_held for r in results]) * 15:.0f} min)")

    # ── BY DIRECTION ──────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY DIRECTION")
    print(f"  {'─' * W}")
    longs  = [(ev, r) for ev, r in zip(events, results) if ev.direction == "LONG"]
    shorts = [(ev, r) for ev, r in zip(events, results) if ev.direction == "SHORT"]
    row("PDL Sweep → LONG  (bears stopped out below PDL)",
        [r.pnl_r for _, r in longs], indent="  ")
    row("PDH Sweep → SHORT (bulls stopped out above PDH)",
        [r.pnl_r for _, r in shorts], indent="  ")

    # ── BY SYMBOL ─────────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY SYMBOL")
    print(f"  {'─' * W}")
    for sym in sorted(set(ev.symbol for ev in events)):
        sym_pnls = [r.pnl_r for ev, r in zip(events, results) if ev.symbol == sym]
        row(sym, sym_pnls, indent="  ")

    # ── BY TIME OF DAY ────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY TIME OF DAY  (hour of sweep bar)")
    print(f"  {'─' * W}")
    for hour in range(9, 16):
        h_pnls = [r.pnl_r for ev, r in zip(events, results) if ev.sweep_hour == hour]
        label  = f"{hour:02d}:00 – {hour+1:02d}:00"
        row(label, h_pnls, min_n=10, indent="  ")

    # ── BY CONFIRMATION SPEED ─────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY CONFIRMATION SPEED  (how quickly the reversal was confirmed)")
    print(f"  {'─' * W}")
    for k in range(1, MAX_CONF_BARS + 1):
        k_pnls = [r.pnl_r for ev, r in zip(events, results) if ev.conf_bar_num == k]
        row(f"Conf bar {k}  (confirmation +{k*15} min after sweep)", k_pnls, indent="  ")

    # ── BY OVERSHOOT SIZE ─────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY WICK OVERSHOOT  (how far wick extended beyond PDH/PDL)")
    print(f"  {'─' * W}")
    small  = [r.pnl_r for ev, r in zip(events, results) if ev.overshoot_pct < 0.001]
    medium = [r.pnl_r for ev, r in zip(events, results)
              if 0.001 <= ev.overshoot_pct < 0.003]
    large  = [r.pnl_r for ev, r in zip(events, results) if ev.overshoot_pct >= 0.003]
    row("Small  wick  (<0.1%  beyond level)",  small,  indent="  ")
    row("Medium wick  (0.1–0.3% beyond)",      medium, indent="  ")
    row("Large  wick  (>0.3%  beyond level)",  large,  indent="  ")

    # ── BY PRIOR DAY RANGE ────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  BY PRIOR DAY RANGE  (narrow vs wide prior day)")
    print(f"  {'─' * W}")
    ranges = [ev.pd_range for ev in events]
    p33, p67 = np.percentile(ranges, [33, 67])
    narrow = [r.pnl_r for ev, r in zip(events, results) if ev.pd_range < p33]
    mid    = [r.pnl_r for ev, r in zip(events, results) if p33 <= ev.pd_range < p67]
    wide   = [r.pnl_r for ev, r in zip(events, results) if ev.pd_range >= p67]
    row(f"Narrow prior day  (<P33={p33:.2f})", narrow, indent="  ")
    row(f"Medium prior day  (P33–P67)",        mid,    indent="  ")
    row(f"Wide   prior day  (>P67={p67:.2f})", wide,   indent="  ")

    # ── GAP DIRECTION VS SWEEP DIRECTION ─────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  GAP DIRECTION vs SWEEP DIRECTION")
    print(f"  (Did today's gap make the level more or less likely to hold?)")
    print(f"  {'─' * W}")
    # Gap TOWARD the level = more likely to be a legit sweep
    gap_toward = [r.pnl_r for ev, r in zip(events, results)
                  if (ev.direction == "SHORT" and ev.gap_pct > 0.001)   # gap up → PDH sweep
                  or (ev.direction == "LONG"  and ev.gap_pct < -0.001)]  # gap down → PDL sweep
    gap_away   = [r.pnl_r for ev, r in zip(events, results)
                  if (ev.direction == "SHORT" and ev.gap_pct < -0.001)  # gap down → PDH sweep
                  or (ev.direction == "LONG"  and ev.gap_pct > 0.001)]  # gap up → PDL sweep
    flat_gap   = [r.pnl_r for ev, r in zip(events, results)
                  if abs(ev.gap_pct) <= 0.001]
    row("Gap TOWARD the level (gap amplifies sweep)", gap_toward, indent="  ")
    row("Gap AWAY from level  (counter-gap sweep)",   gap_away,   indent="  ")
    row("Flat gap             (<0.1% gap)",           flat_gap,   indent="  ")

    # ── EXIT ANALYSIS ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  EXIT ANALYSIS")
    print(f"  {'─' * W}")
    stop_r = [r.pnl_r for r in results if r.exit_reason == "stop"]
    eod_r  = [r.pnl_r for r in results if r.exit_reason == "eod"]
    print(f"  Stop hit: {len(stop_r):>4d} ({len(stop_r)/len(results)*100:>4.0f}%)  "
          f"avg={np.mean(stop_r):+.3f}R" if stop_r else f"  Stop hit: {len(stop_r):>4d}  (0%)")
    print(f"  EOD exit: {len(eod_r):>4d} ({len(eod_r)/len(results)*100:>4.0f}%)  "
          f"avg={np.mean(eod_r):+.3f}R  "
          f"WR={sum(1 for p in eod_r if p>0)/len(eod_r)*100:.0f}%" if eod_r else "")

    # ── P&L DISTRIBUTION ─────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  P&L DISTRIBUTION  (R-multiples, all trades)")
    print(f"  {'─' * W}")
    arr    = np.array(pnls)
    ptiles = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f"  P10={ptiles[0]:+.2f}R  P25={ptiles[1]:+.2f}R  P50={ptiles[2]:+.2f}R  "
          f"P75={ptiles[3]:+.2f}R  P90={ptiles[4]:+.2f}R")

    bins = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5]
    hist, _ = np.histogram(arr, bins=bins)
    scale   = max(max(hist) // 35, 1)
    print()
    for i in range(len(bins) - 1):
        bar_str = "█" * (hist[i] // scale)
        print(f"  [{bins[i]:>+5.1f}, {bins[i+1]:>+5.1f})  {hist[i]:>4d}  {bar_str}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    s = stats(pnls)
    print(f"\n  {'═' * W}")
    print(f"  VERDICT")
    print(f"  {'═' * W}")

    c1 = s["exp"] > 0.03
    c2 = s["wr"]  > 40
    c3 = len(pnls) >= 100

    print(f"  [{'✓' if c1 else '✗'}]  Positive expectancy (> +0.03R):  {s['exp']:+.3f}R")
    print(f"  [{'✓' if c2 else '✗'}]  Win rate > 40%:                  {s['wr']:.0f}%")
    print(f"  [{'✓' if c3 else '✗'}]  Sample size ≥ 100:               n={len(pnls)}")

    passed = sum([c1, c2, c3])
    if passed >= 2 and c3:
        print(f"\n  → PROCEED TO PHASE 0.5  (edge decomposition — WHERE does the sweep edge live?)")
        print(f"  Key questions for Phase 0.5:")
        print(f"    1. PDH sweeps vs PDL sweeps — is one direction significantly better?")
        print(f"    2. Time of day — morning sweeps (9:30-11) vs afternoon (14-16)?")
        print(f"    3. Wick size — does a larger overshoot predict better reversal?")
        print(f"    4. Confirmation speed — bar 1 vs bar 2 vs bar 3?")
        print(f"    5. Gap toward vs gap away from the level?")
        print(f"    6. Prior day range — narrow levels cleaner?")
    elif not c3:
        print(f"\n  → SAMPLE TOO SMALL ({len(pnls)} < 100).  Consider expanding universe or date range.")
    else:
        print(f"\n  → NO EDGE FOUND — DROP this event type, move to Event Type 3 (ORB Failure).")

    print(f"  {'═' * W}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase0_raw_edge.py ./data/cache/*.csv")
        sys.exit(1)

    all_events: List[SweepEvent]  = []
    paths = sys.argv[1:]

    for path in paths:
        sym = Path(path).stem.split("_")[0]
        try:
            df  = load_csv(path)
            evs = find_sweep_events(df)
            all_events.extend(evs)
            print(f"  {sym:6s}: {len(evs):4d} sweep events")
        except Exception as ex:
            print(f"  ERROR loading {path}: {ex}")

    if not all_events:
        print("No sweep events found. Check data files.")
        sys.exit(1)

    results = [simulate_trade(ev) for ev in all_events]

    # Date range string for display
    all_dates  = sorted(set(ev.date for ev in all_events))
    date_range = f"{all_dates[0]} → {all_dates[-1]}  ({len(all_dates)} trading days across "  \
                 f"{len(set(ev.symbol for ev in all_events))} symbols)"

    print_results(all_events, results, date_range)


if __name__ == "__main__":
    main()
