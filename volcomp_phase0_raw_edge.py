"""volcomp_phase0_raw_edge.py — Event Type 4: Volatility Compression → Expansion

Phase 0: Raw Edge Test

Hypothesis:
  An extended period of low volatility (compression) stores directional energy that
  is eventually released as a sharp move (expansion). The direction of the expansion
  bar signals the breakout direction.

  "Low ATR coiling → expansion bar body > 40% range → trade the first expansion bar"

Setup logic:
  1. Compression: ATR(14) on a rolling basis. Calculate the rolling ATR percentile
     over the prior 20 sessions. Compression = ATR < 30th percentile for at LEAST
     COMPRESS_MIN_BARS consecutive bars within the current session.
  2. Expansion bar: the first bar AFTER the compression phase where:
     a. Bar range > EXPAND_MULT × prior compression avg bar range (default 1.5x)
     b. Bar body (|close - open|) > BODY_RATIO × bar range (default 40%) — directional
     c. Bar closes in the direction of the body (bullish or bearish bar)
  3. Entry: close of expansion bar
  4. Stop: LOW of expansion bar (LONG) or HIGH of expansion bar (SHORT)
  5. Exit: EOD or stop (Phase 2 will optimize exits)

Key design choices:
  - All ATR/percentile calculations use ONLY intraday bars seen so far → no look-ahead
  - Only one setup per day (first qualifying expansion)
  - Compression phase must occur before the expansion (within same session)

Why this is different from ORB/Sweep:
  - ORB: trade breakout of the first 30-min range
  - Sweep: fade the liquidity grab beyond a structural level
  - Vol compression: trade the first expansion after a quiet period (can happen any time)
  - Works in trending AND mean-reverting markets — direction comes from expansion bar

Risk notes:
  - Smallest expected sample size of all 4 event types
  - If n < 20 events: flag insufficient, do not proceed
  - Even n=20-40 events: treat as "indicative only", not statistically meaningful

Analysis dimensions:
  - Direction: LONG (bullish expansion) vs SHORT (bearish expansion)
  - Symbol: per-stock breakdown
  - Time of day: morning compression vs midday compression
  - Gap: does a gap morning affect compression quality?
  - Compression duration: 8-12 bars vs 12+ bars
  - Body ratio: just-passing (40-50%) vs high-body (>60%) expansion bars
  - Prior day direction: does trend context matter?

Usage:
  python3 volcomp_phase0_raw_edge.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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

MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
ATR_PERIOD         = 14       # bars for ATR
ATR_PCT_LOOKBACK   = 20       # sessions for rolling ATR percentile
COMPRESS_PERCENTILE = 30      # ATR < 30th pct = compressed
COMPRESS_MIN_BARS  = 5        # min consecutive compressed bars to qualify
EXPAND_MULT        = 1.5      # expansion bar range > 1.5× compression avg
BODY_RATIO         = 0.40     # bar body must be > 40% of range
MIN_EVENTS         = 15       # minimum to report a cut

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class VolCompEvent:
    symbol:             str
    date:               str
    direction:          str    # "LONG" or "SHORT"

    # Compression phase
    compress_start_bar: int    # bar iloc where compression started
    compress_end_bar:   int    # bar iloc of last compressed bar (= expand_bar - 1)
    compress_n_bars:    int    # number of consecutive compressed bars
    compress_avg_range: float  # avg bar range during compression phase
    atr_percentile:     float  # ATR percentile at time of expansion

    # Expansion bar
    expand_bar_iloc:    int
    expand_hour:        int
    expand_range:       float
    expand_body:        float
    expand_body_ratio:  float  # body / range
    expand_ratio:       float  # expand_range / compress_avg_range

    # Entry / stop
    entry_price:        float  # close of expansion bar
    stop_price:         float  # low (LONG) or high (SHORT) of expansion bar
    risk:               float

    # Context
    gap_pct:            float
    prior_day_up:       bool
    prior_close:        float

    # Results
    mfe:                float
    mae:                float
    pnl_r:              float
    exit_reason:        str
    hit_stop:           bool


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

def intraday_atr(bars: pd.DataFrame) -> float:
    """ATR(14) from a sequence of intraday bars."""
    if len(bars) < 2:
        return float(bars["high"].iloc[0] - bars["low"].iloc[0]) if len(bars) == 1 else 0.01
    trs = [bars.iloc[0]["high"] - bars.iloc[0]["low"]]
    for i in range(1, len(bars)):
        tr = max(bars.iloc[i]["high"] - bars.iloc[i]["low"],
                 abs(bars.iloc[i]["high"] - bars.iloc[i-1]["close"]),
                 abs(bars.iloc[i]["low"]  - bars.iloc[i-1]["close"]))
        trs.append(tr)
    n = min(ATR_PERIOD, len(trs))
    return float(np.mean(trs[-n:]))


def sim_exit(entry: float, stop: float, risk: float, direction: str,
             remaining: pd.DataFrame, eod_price: float) -> Tuple[float, float, float, str, bool]:
    mf = ma = 0.0
    for _, bar in remaining.iterrows():
        fav = ((bar["high"] - entry) / risk if direction == "LONG"
               else (entry - bar["low"]) / risk)
        adv = ((entry - bar["low"]) / risk if direction == "LONG"
               else (bar["high"] - entry) / risk)
        mf = max(mf, fav); ma = min(ma, -adv)
        if direction == "LONG"  and bar["low"]  <= stop:
            return -1.0, mf, ma, "stop", True
        if direction == "SHORT" and bar["high"] >= stop:
            return -1.0, mf, ma, "stop", True
    pnl = ((eod_price - entry) / risk if direction == "LONG"
           else (entry - eod_price) / risk)
    return pnl, mf, ma, "eod", False


# ── Event Detection ───────────────────────────────────────────────────────────

def find_events(df: pd.DataFrame, session_atrs: List[float]) -> List[VolCompEvent]:
    """Detect compression→expansion setups. session_atrs accumulates across calls."""
    symbol  = df["symbol"].iloc[0]
    events: List[VolCompEvent] = []
    prior_close = None
    pd_close    = None
    pd_open     = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if prior_close is None:
            prior_close = float(day["close"].iloc[-1])
            pd_close    = prior_close
            pd_open     = float(day["open"].iloc[0])
            session_atrs.append(intraday_atr(day))
            continue

        prior_day_up = pd_close > pd_open
        gap_pct      = (float(day["open"].iloc[0]) - prior_close) / prior_close if prior_close > 0 else 0.0
        eod          = float(day["close"].iloc[-1])

        # ATR percentile threshold for this session (from prior sessions)
        if len(session_atrs) >= ATR_PCT_LOOKBACK:
            atr_threshold = float(np.percentile(session_atrs[-ATR_PCT_LOOKBACK:], COMPRESS_PERCENTILE))
        else:
            atr_threshold = 1e9   # can't compute yet — skip compression test (treat all as non-compressed)

        # Scan for compression → expansion
        n = len(day)
        compress_count = 0
        compress_start = -1
        compress_ranges: List[float] = []
        found = False

        for i in range(1, n):
            bar = day.iloc[i]
            bar_range = float(bar["high"] - bar["low"])

            # Compute running intraday ATR up to this bar
            running_atr = intraday_atr(day.iloc[:i + 1])
            is_compressed = running_atr < atr_threshold and bar_range < atr_threshold

            if is_compressed:
                if compress_count == 0:
                    compress_start = i
                compress_count += 1
                compress_ranges.append(bar_range)
            else:
                # Check if we have enough compression and this bar is the expansion
                if compress_count >= COMPRESS_MIN_BARS and compress_ranges:
                    avg_comp_range = float(np.mean(compress_ranges))
                    body           = abs(float(bar["close"]) - float(bar["open"]))
                    body_ratio     = body / bar_range if bar_range > 0 else 0.0
                    expand_ratio   = bar_range / avg_comp_range if avg_comp_range > 0 else 0.0

                    # Is this a qualifying expansion bar?
                    if (expand_ratio >= EXPAND_MULT and body_ratio >= BODY_RATIO):
                        # Direction from bar body
                        is_bullish = bar["close"] > bar["open"]
                        direction  = "LONG" if is_bullish else "SHORT"

                        entry = float(bar["close"])
                        stop  = float(bar["low"]) if direction == "LONG" else float(bar["high"])
                        risk  = abs(entry - stop)

                        if risk > 0:
                            # ATR percentile of this session's ATR vs history
                            if len(session_atrs) >= 5:
                                atr_pct = float(np.percentile(
                                    session_atrs[-ATR_PCT_LOOKBACK:],
                                    np.searchsorted(
                                        np.sort(session_atrs[-ATR_PCT_LOOKBACK:]),
                                        running_atr
                                    ) * 100 / len(session_atrs[-ATR_PCT_LOOKBACK:])
                                ))
                            else:
                                atr_pct = 50.0

                            remaining = day.iloc[i + 1:]
                            pnl, mf, ma, reason, hs = sim_exit(
                                entry, stop, risk, direction, remaining, eod)

                            events.append(VolCompEvent(
                                symbol=symbol, date=str(day_date), direction=direction,
                                compress_start_bar=compress_start,
                                compress_end_bar=i - 1,
                                compress_n_bars=compress_count,
                                compress_avg_range=avg_comp_range,
                                atr_percentile=atr_pct,
                                expand_bar_iloc=i,
                                expand_hour=bar["time"].hour,
                                expand_range=bar_range,
                                expand_body=body,
                                expand_body_ratio=body_ratio,
                                expand_ratio=expand_ratio,
                                entry_price=entry, stop_price=stop, risk=risk,
                                gap_pct=gap_pct, prior_day_up=prior_day_up,
                                prior_close=float(prior_close),
                                mfe=mf, mae=ma, pnl_r=pnl,
                                exit_reason=reason, hit_stop=hs,
                            ))
                            found = True
                            break  # one setup per day

                # Reset compression
                compress_count = 0
                compress_start = -1
                compress_ranges = []

        session_atrs.append(intraday_atr(day))
        prior_close = float(day["close"].iloc[-1])
        pd_close    = prior_close
        pd_open     = float(day["open"].iloc[0])

    return events


# ── Analysis ──────────────────────────────────────────────────────────────────

def stats_row(label: str, events: List[VolCompEvent], indent: str = "  ") -> None:
    if len(events) < MIN_EVENTS:
        print(f"{indent}{label:<40s}  n={len(events):4d}  [too few]")
        return
    pnls = [e.pnl_r for e in events]
    exp  = float(np.mean(pnls))
    wr   = float(np.mean([p > 0 for p in pnls]))
    sh   = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 1e-9 else 0.0
    sr   = float(np.mean([e.hit_stop for e in events]))
    mfe  = float(np.mean([e.mfe for e in events]))
    flag = "★★" if exp > 0.15 and sh > 0.1 else ("★" if exp > 0.05 else "")
    print(f"{indent}{label:<40s}  n={len(events):4d}  exp={exp:+.3f}R  "
          f"WR={wr:4.0%}  Sharpe={sh:+.3f}  stop={sr:.0%}  MFE={mfe:+.3f}R  {flag}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 volcomp_phase0_raw_edge.py <csv_files...>")
        sys.exit(1)

    all_events: List[VolCompEvent] = []
    for path in sys.argv[1:]:
        df          = load_csv(path)
        sym         = Path(path).stem.split("_")[0]
        session_atrs: List[float] = []
        evs         = find_events(df, session_atrs)
        print(f"  {sym:6s}  :  {len(evs):4d} events")
        all_events.extend(evs)

    n_total = len(all_events)
    print()
    print("=" * 100)
    print("  EVENT TYPE 4: VOL COMPRESSION → EXPANSION — PHASE 0 RAW EDGE TEST")
    print(f"  Total events detected: {n_total}")
    print(f"  Config: compress ≥{COMPRESS_MIN_BARS} bars at ATR<{COMPRESS_PERCENTILE}th pct, "
          f"expand ratio ≥{EXPAND_MULT}x, body ≥{BODY_RATIO:.0%}")
    print("=" * 100)

    if n_total < MIN_EVENTS:
        print(f"\n  INSUFFICIENT DATA: only {n_total} events detected.")
        print(f"  Try loosening parameters (lower COMPRESS_MIN_BARS or COMPRESS_PERCENTILE).")
        print(f"  If still < 20 with looser params → DROP this event type.")
        return

    # Overall
    print(f"\n  ── OVERALL BASELINE ──────────────────────────────────────────────")
    stats_row("All events", all_events)

    # Direction
    print(f"\n  ── D1: DIRECTION ─────────────────────────────────────────────────")
    stats_row("LONG  (bullish expansion)", [e for e in all_events if e.direction == "LONG"])
    stats_row("SHORT (bearish expansion)", [e for e in all_events if e.direction == "SHORT"])

    # Symbol
    print(f"\n  ── D2: SYMBOL ────────────────────────────────────────────────────")
    for sym in sorted(set(e.symbol for e in all_events)):
        stats_row(sym, [e for e in all_events if e.symbol == sym])

    # Expansion hour
    print(f"\n  ── D3: EXPANSION TIME ────────────────────────────────────────────")
    for h in sorted(set(e.expand_hour for e in all_events)):
        stats_row(f"Expansion at {h:02d}h", [e for e in all_events if e.expand_hour == h])

    # Prior day direction
    print(f"\n  ── D4: PRIOR DAY DIRECTION ───────────────────────────────────────")
    stats_row("Prior day UP",   [e for e in all_events if e.prior_day_up])
    stats_row("Prior day DOWN", [e for e in all_events if not e.prior_day_up])

    # Compression duration
    med_n = float(np.median([e.compress_n_bars for e in all_events]))
    print(f"\n  ── D5: COMPRESSION DURATION (median={med_n:.0f} bars) ────────────────")
    stats_row(f"Short compress ({COMPRESS_MIN_BARS}-{int(med_n)} bars)",
              [e for e in all_events if e.compress_n_bars <= med_n])
    stats_row(f"Long compress (>{int(med_n)} bars)",
              [e for e in all_events if e.compress_n_bars > med_n])

    # Expansion body ratio quality
    print(f"\n  ── D6: EXPANSION BAR QUALITY ─────────────────────────────────────")
    stats_row("Body 40-60% (marginal)",   [e for e in all_events if 0.40 <= e.expand_body_ratio < 0.60])
    stats_row("Body 60-80% (good)",       [e for e in all_events if 0.60 <= e.expand_body_ratio < 0.80])
    stats_row("Body >80% (strong)",       [e for e in all_events if e.expand_body_ratio >= 0.80])

    # Expansion ratio (how strong the expansion relative to compression)
    med_er = float(np.median([e.expand_ratio for e in all_events]))
    print(f"\n  ── D7: EXPANSION STRENGTH (median={med_er:.2f}x) ───────────────────")
    stats_row(f"Moderate expand (1.5-{med_er:.1f}x)",
              [e for e in all_events if 1.5 <= e.expand_ratio <= med_er])
    stats_row(f"Strong expand (>{med_er:.1f}x)",
              [e for e in all_events if e.expand_ratio > med_er])

    # Gap
    print(f"\n  ── D8: GAP ───────────────────────────────────────────────────────")
    stats_row("Gap UP  (>+0.2%)", [e for e in all_events if e.gap_pct >  0.002])
    stats_row("Gap DOWN (<-0.2%)",[e for e in all_events if e.gap_pct < -0.002])
    stats_row("No gap",           [e for e in all_events if abs(e.gap_pct) <= 0.002])

    # Exit breakdown
    eod_surv = [e for e in all_events if not e.hit_stop]
    stop_hit = [e for e in all_events if e.hit_stop]
    print(f"\n  ── EXIT BREAKDOWN ────────────────────────────────────────────────")
    if eod_surv:
        print(f"    Reached EOD: n={len(eod_surv)}  avg P&L={np.mean([e.pnl_r for e in eod_surv]):+.3f}R")
    if stop_hit:
        print(f"    Hit stop:    n={len(stop_hit)}  avg P&L=-1.000R")
    print(f"    Avg MFE: {np.mean([e.mfe for e in all_events]):+.3f}R  |  "
          f"Avg MAE: {np.mean([e.mae for e in all_events]):+.3f}R")

    # Compression stats
    print(f"\n  ── COMPRESSION STATISTICS ────────────────────────────────────────")
    print(f"    Avg compress duration: {np.mean([e.compress_n_bars for e in all_events]):.1f} bars "
          f"({np.mean([e.compress_n_bars for e in all_events]) * 15:.0f} min)")
    print(f"    Avg expand ratio:      {np.mean([e.expand_ratio for e in all_events]):.2f}x")
    print(f"    Avg body ratio:        {np.mean([e.expand_body_ratio for e in all_events]):.2%}")
    print(f"    Avg expand hour:       {np.mean([e.expand_hour for e in all_events]):.1f}")

    # Verdict
    all_pnls = [e.pnl_r for e in all_events]
    exp  = float(np.mean(all_pnls))
    sh   = float(np.mean(all_pnls) / np.std(all_pnls)) if np.std(all_pnls) > 1e-9 else 0.0
    print(f"\n  ── VERDICT ───────────────────────────────────────────────────────")
    if n_total < 50:
        print(f"    WARNING: only {n_total} events — results are indicative, not statistically robust.")
    if exp > 0.05 and sh > 0.05:
        print(f"    PROCEED TO PHASE 0.5  (exp={exp:+.3f}R, Sharpe={sh:+.3f})")
    elif exp > 0:
        print(f"    WEAK EDGE (exp={exp:+.3f}R, Sharpe={sh:+.3f})")
        print(f"    Proceed with caution. Check if parameter tuning reveals stronger signal.")
    else:
        print(f"    NO EDGE DETECTED (exp={exp:+.3f}R, Sharpe={sh:+.3f})")
        print(f"    Consider loosening compression parameters before dropping.")


if __name__ == "__main__":
    main()
