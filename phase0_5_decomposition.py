"""Phase 0.5: Edge Decomposition

Question: WHERE does the edge live?

Method:
  - Same ORB breakout logic as Phase 0
  - Simulates LIMIT entry at ORB boundary (no chase) AND checks if it would have filled
  - Decomposes by every dimension: time, direction, ORB width, gap, volume, symbol
  - Tests multi-factor filter combinations (single, pairs, triples)
  - Ranks subsets by limit expectancy

Output:
  - Per-dimension breakdown: actual entry vs limit entry expectancy
  - Fill rate for limit order strategy (did price pull back to boundary?)
  - Top filter combinations by limit expectancy
  - Verdict: "The edge is specifically in [X]"

Usage:
  python3 phase0_5_decomposition.py ./data/cache/AAPL_2025-07-01_2025-12-1_15_mins.csv \\
                                    ./data/cache/NVDA_2025-07-01_2025-12-1_15_mins.csv \\
                                    ./data/cache/MSFT_2025-07-01_2025-12-1_15_mins.csv \\
                                    ./data/cache/AMZN_2025-07-01_2025-12-1_15_mins.csv \\
                                    ./data/cache/GOOG_2025-07-01_2025-12-1_15_mins.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List


# ─── Configuration ───────────────────────────────────────

MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
MIN_TRADES         = 15   # minimum trades to report a subset


# ─── Data Structures ────────────────────────────────────

@dataclass
class Trade:
    symbol:             str
    date:               str
    direction:          str
    orb_high:           float
    orb_low:            float
    orb_range:          float
    orb_volume:         float
    entry_price:        float   # actual: close of breakout bar
    entry_time:         str
    stop_price:         float
    risk:               float
    chase_pct:          float   # (entry - boundary) / orb_range
    eod_price:          float
    eod_pnl_r:          float   # actual entry P&L in R
    max_favorable_r:    float
    max_adverse_r:      float
    hit_stop:           bool
    gap_pct:            float
    day_volume:         float
    prior_close:        float
    # ── Limit entry simulation ──
    limit_entry:        float   # ORB boundary price
    limit_pnl_r:        float   # P&L if entered at ORB boundary (vs EOD)
    limit_filled:       bool    # did price pull back to boundary after breakout bar?
    limit_mfe_r:        float
    limit_mae_r:        float
    limit_hit_stop:     bool


# ─── Load CSV ────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    from datetime import time as dtime
    df = pd.read_csv(path)
    df = df.rename(columns={"date": "timestamp"})
    df["symbol"] = Path(path).stem.split("_")[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    market_open  = dtime(MARKET_OPEN_HOUR, MARKET_OPEN_MIN)
    market_close = dtime(16, 0)
    df = df[(df["time"] >= market_open) & (df["time"] < market_close)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


# ─── Core Logic ──────────────────────────────────────────

def find_orb_trades(df: pd.DataFrame) -> List[Trade]:
    from datetime import time as dtime
    symbol      = df["symbol"].iloc[0]
    orb_end     = dtime(10, 0)
    trades      = []
    prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = df[df["date"] == day_date].sort_values("timestamp").reset_index(drop=True)
        if len(day) < 4:
            if len(day) > 0:
                prior_close = day["close"].iloc[-1]
            continue

        orb_bars   = day[day["time"] < orb_end]
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

        day_open = day["open"].iloc[0]
        gap_pct  = (day_open - prior_close) / prior_close if prior_close and prior_close > 0 else 0.0

        post_orb = day[day["time"] >= orb_end]
        if len(post_orb) < 2:
            prior_close = day["close"].iloc[-1]
            continue

        for idx, row in post_orb.iterrows():
            close = row["close"]
            if close > orb_high:
                direction   = "LONG"
                entry_price = close
                stop_price  = orb_low
                risk        = entry_price - stop_price
                chase_pct   = (entry_price - orb_high) / orb_range
                limit_entry = orb_high
            elif close < orb_low:
                direction   = "SHORT"
                entry_price = close
                stop_price  = orb_high
                risk        = stop_price - entry_price
                chase_pct   = (orb_low - entry_price) / orb_range
                limit_entry = orb_low
            else:
                continue  # no breakout yet

            if risk <= 0:
                break

            entry_iloc   = day.index.get_loc(idx)
            remaining    = day.iloc[entry_iloc:]
            post_breakout= day.iloc[entry_iloc + 1:]
            eod_price    = day["close"].iloc[-1]
            day_volume   = day["volume"].sum()
            limit_risk   = orb_range

            # ── Actual entry metrics ──
            if direction == "LONG":
                eod_pnl_r       = (eod_price - entry_price) / risk
                mfe_price       = remaining["high"].max()
                mae_price       = remaining["low"].min()
                max_favorable_r = (mfe_price - entry_price) / risk
                max_adverse_r   = (mae_price - entry_price) / risk
                hit_stop        = mae_price <= stop_price
            else:
                eod_pnl_r       = (entry_price - eod_price) / risk
                mfe_price       = remaining["low"].min()
                mae_price       = remaining["high"].max()
                max_favorable_r = (entry_price - mfe_price) / risk
                max_adverse_r   = (entry_price - mae_price) / risk
                hit_stop        = mae_price >= stop_price

            # ── Limit entry simulation ──
            if direction == "LONG":
                limit_filled    = len(post_breakout) > 0 and post_breakout["low"].min() <= orb_high
                limit_pnl_r     = (eod_price - limit_entry) / limit_risk
                limit_mfe_r     = (remaining["high"].max() - limit_entry) / limit_risk
                limit_mae_r     = (remaining["low"].min()  - limit_entry) / limit_risk
                limit_hit_stop  = remaining["low"].min() <= orb_low
            else:
                limit_filled    = len(post_breakout) > 0 and post_breakout["high"].max() >= orb_low
                limit_pnl_r     = (limit_entry - eod_price) / limit_risk
                limit_mfe_r     = (limit_entry - remaining["low"].min())  / limit_risk
                limit_mae_r     = (limit_entry - remaining["high"].max()) / limit_risk
                limit_hit_stop  = remaining["high"].max() >= orb_high

            trades.append(Trade(
                symbol=symbol, date=str(day_date), direction=direction,
                orb_high=orb_high, orb_low=orb_low, orb_range=orb_range,
                orb_volume=orb_volume, entry_price=entry_price,
                entry_time=str(row["time"]), stop_price=stop_price, risk=risk,
                chase_pct=chase_pct, eod_price=eod_price, eod_pnl_r=eod_pnl_r,
                max_favorable_r=max_favorable_r, max_adverse_r=max_adverse_r,
                hit_stop=hit_stop, gap_pct=gap_pct,
                day_volume=day_volume, prior_close=prior_close or day_open,
                limit_entry=limit_entry, limit_pnl_r=limit_pnl_r,
                limit_filled=limit_filled, limit_mfe_r=limit_mfe_r,
                limit_mae_r=limit_mae_r, limit_hit_stop=limit_hit_stop,
            ))
            break  # only first breakout per day

        prior_close = day["close"].iloc[-1]

    return trades


# ─── Stats & Display ─────────────────────────────────────

def stats(pnls: List[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "med": 0.0, "wr": 0.0}
    n    = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {"n": n, "exp": np.mean(pnls), "med": np.median(pnls), "wr": wins / n * 100}


def row(label: str, trades: List[Trade], min_n: int = MIN_TRADES, indent: str = "  "):
    if len(trades) < min_n:
        return
    a  = stats([t.eod_pnl_r   for t in trades])
    lm = stats([t.limit_pnl_r for t in trades])
    fill_rate = sum(1 for t in trades if t.limit_filled) / len(trades) * 100

    def mark(exp): return "✓" if exp > 0.05 else ("~" if exp > 0 else "✗")

    print(f"{indent}{label:42s}"
          f"  n={a['n']:3d}"
          f"  actual {mark(a['exp'])} {a['exp']:+.3f}R {a['wr']:4.0f}%WR"
          f"  limit {mark(lm['exp'])} {lm['exp']:+.3f}R {lm['wr']:4.0f}%WR"
          f"  fill {fill_rate:3.0f}%")


# ─── Decomposition ───────────────────────────────────────

def decompose(trades: List[Trade]):
    from datetime import time as dtime

    orb_ranges     = [t.orb_range for t in trades]
    p20, p40, p60, p80 = np.percentile(orb_ranges, [20, 40, 60, 80])

    vol_ratios     = [t.orb_volume / t.day_volume for t in trades if t.day_volume > 0]
    med_vol_ratio  = np.median(vol_ratios)
    p33_ratio, p67_ratio = np.percentile(vol_ratios, [33, 67])

    header = ("  {:<42s}  {:>5s}  {:>20s}  {:>20s}  {:>7s}"
              .format("Filter", "n", "actual [E WR]", "limit [E WR]", "fill"))

    print("\n" + "=" * 100)
    print("  PHASE 0.5: WHERE DOES THE EDGE LIVE?")
    print("  Columns: n | actual [✓/~/✗ expectancy WR] | limit [✓/~/✗ expectancy WR] | fill rate")
    print("  ✓ = exp > +0.05R   ~ = exp > 0   ✗ = negative")
    print("=" * 100)
    row("ALL TRADES", trades, min_n=1)

    # ── By Symbol ──────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY SYMBOL")
    for sym in sorted(set(t.symbol for t in trades)):
        row(sym, [t for t in trades if t.symbol == sym])

    # ── By Direction ───────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY DIRECTION")
    for d in ["LONG", "SHORT"]:
        row(d, [t for t in trades if t.direction == d])

    # ── By Entry Time ──────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY ENTRY TIME")
    time_buckets = [
        ("10:00-10:30", dtime(10, 0),  dtime(10, 30)),
        ("10:30-11:00", dtime(10, 30), dtime(11, 0)),
        ("11:00-12:00", dtime(11, 0),  dtime(12, 0)),
        ("12:00-13:00", dtime(12, 0),  dtime(13, 0)),
        ("13:00-14:00", dtime(13, 0),  dtime(14, 0)),
        ("14:00-16:00", dtime(14, 0),  dtime(16, 0)),
    ]
    for label, t0, t1 in time_buckets:
        g = [t for t in trades if t0 <= dtime.fromisoformat(t.entry_time) < t1]
        row(label, g)

    # ── By ORB Width (quintiles) ────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY ORB WIDTH (quintiles)")
    orb_bins = [
        (f"Q1 Narrow  (<${p20:.2f})",            0,    p20),
        (f"Q2         (${p20:.2f}-${p40:.2f})",  p20,  p40),
        (f"Q3         (${p40:.2f}-${p60:.2f})",  p40,  p60),
        (f"Q4         (${p60:.2f}-${p80:.2f})",  p60,  p80),
        (f"Q5 Wide    (>${p80:.2f})",             p80,  float("inf")),
    ]
    for label, lo, hi in orb_bins:
        row(label, [t for t in trades if lo <= t.orb_range < hi])

    # ── By Gap ─────────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY GAP SIZE & DIRECTION")
    gap_filters = [
        ("Strong gap up   (>0.5%)",          lambda t: t.gap_pct >  0.005),
        ("Mild gap up     (0.2-0.5%)",        lambda t: 0.002  < t.gap_pct <=  0.005),
        ("Flat gap        (-0.2% to 0.2%)",   lambda t: -0.002 <= t.gap_pct <=  0.002),
        ("Mild gap down   (-0.5% to -0.2%)",  lambda t: -0.005 <= t.gap_pct <  -0.002),
        ("Strong gap down (<-0.5%)",          lambda t: t.gap_pct < -0.005),
    ]
    for label, fn in gap_filters:
        row(label, [t for t in trades if fn(t)])

    # ── Gap alignment with breakout direction ──────────
    print(f"\n{'─' * 100}")
    print("  BY GAP ALIGNMENT WITH BREAKOUT DIRECTION")
    align_filters = [
        ("Gap aligned  (gap & breakout same dir)",
         lambda t: (t.direction == "LONG"  and t.gap_pct >  0.002) or
                   (t.direction == "SHORT" and t.gap_pct < -0.002)),
        ("Gap opposed  (gap & breakout opposite)",
         lambda t: (t.direction == "LONG"  and t.gap_pct < -0.002) or
                   (t.direction == "SHORT" and t.gap_pct >  0.002)),
        ("Gap neutral  (no significant gap)",
         lambda t: -0.002 <= t.gap_pct <= 0.002),
    ]
    for label, fn in align_filters:
        row(label, [t for t in trades if fn(t)])

    # ── By ORB Volume Ratio ─────────────────────────────
    print(f"\n{'─' * 100}")
    print(f"  BY ORB VOLUME AS % OF DAY VOLUME  (med ratio = {med_vol_ratio:.2f})")
    for label, fn in [
        (f"Low   ORB vol ratio (<{p33_ratio:.2f})",
         lambda t: t.day_volume > 0 and t.orb_volume / t.day_volume < p33_ratio),
        (f"Med   ORB vol ratio ({p33_ratio:.2f}-{p67_ratio:.2f})",
         lambda t: t.day_volume > 0 and p33_ratio <= t.orb_volume / t.day_volume < p67_ratio),
        (f"High  ORB vol ratio (>{p67_ratio:.2f})",
         lambda t: t.day_volume > 0 and t.orb_volume / t.day_volume >= p67_ratio),
    ]:
        row(label, [t for t in trades if fn(t)])

    # ── By Chase ───────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  BY CHASE (how far past ORB boundary the actual entry was)")
    for label, lo, hi in [
        ("No chase    (<0.1x ORB)",    0.0, 0.1),
        ("Low chase   (0.1-0.2x ORB)", 0.1, 0.2),
        ("Med chase   (0.2-0.5x ORB)", 0.2, 0.5),
        ("High chase  (>0.5x ORB)",    0.5, float("inf")),
    ]:
        row(label, [t for t in trades if lo <= t.chase_pct < hi])

    # ── Multi-Factor Combination Search ────────────────
    print(f"\n{'─' * 100}")
    print("  MULTI-FACTOR FILTER SEARCH  (ranked by limit entry expectancy)")
    print("  Searching all single, pair, and triple filter combinations...")

    named_filters = [
        ("mid-day (12-14h)",   lambda t: dtime(12,0) <= dtime.fromisoformat(t.entry_time) < dtime(14,0)),
        ("morning (10-11h)",   lambda t: dtime(10,0) <= dtime.fromisoformat(t.entry_time) < dtime(11,0)),
        ("late (11h+)",        lambda t: dtime.fromisoformat(t.entry_time) >= dtime(11,0)),
        ("not narrow ORB",     lambda t: t.orb_range >= p20),
        ("med ORB (Q2-Q4)",    lambda t: p20 <= t.orb_range < p80),
        ("wide ORB (Q5)",      lambda t: t.orb_range >= p80),
        ("gap aligned",        lambda t: (t.direction=="LONG" and t.gap_pct>0.002) or
                                         (t.direction=="SHORT" and t.gap_pct<-0.002)),
        ("gap opposed",        lambda t: (t.direction=="LONG" and t.gap_pct<-0.002) or
                                         (t.direction=="SHORT" and t.gap_pct>0.002)),
        ("has gap (|gap|>0.2%)", lambda t: abs(t.gap_pct) > 0.002),
        ("no flat gap",        lambda t: abs(t.gap_pct) > 0.002),
        ("low ORB vol ratio",  lambda t: t.day_volume>0 and t.orb_volume/t.day_volume < med_vol_ratio),
        ("high ORB vol ratio", lambda t: t.day_volume>0 and t.orb_volume/t.day_volume >= med_vol_ratio),
        ("LONG only",          lambda t: t.direction == "LONG"),
        ("SHORT only",         lambda t: t.direction == "SHORT"),
    ]

    # Conflict groups — filters within a group are mutually exclusive
    conflict_groups = [
        {"mid-day (12-14h)", "morning (10-11h)"},
        {"LONG only", "SHORT only"},
        {"gap aligned", "gap opposed"},
        {"low ORB vol ratio", "high ORB vol ratio"},
        {"med ORB (Q2-Q4)", "wide ORB (Q5)"},
    ]

    def conflicts(names):
        for group in conflict_groups:
            if len(group & set(names)) > 1:
                return True
        return False

    results = []

    # Singles
    for n, fn in named_filters:
        g = [t for t in trades if fn(t)]
        if len(g) >= MIN_TRADES:
            s = stats([t.limit_pnl_r for t in g])
            results.append(([n], len(g), s["exp"], s["wr"]))

    # Pairs
    nf = named_filters
    for i in range(len(nf)):
        for j in range(i+1, len(nf)):
            names = [nf[i][0], nf[j][0]]
            if conflicts(names):
                continue
            g = [t for t in trades if nf[i][1](t) and nf[j][1](t)]
            if len(g) >= MIN_TRADES:
                s = stats([t.limit_pnl_r for t in g])
                results.append((names, len(g), s["exp"], s["wr"]))

    # Triples
    for i in range(len(nf)):
        for j in range(i+1, len(nf)):
            for k in range(j+1, len(nf)):
                names = [nf[i][0], nf[j][0], nf[k][0]]
                if conflicts(names):
                    continue
                g = [t for t in trades if nf[i][1](t) and nf[j][1](t) and nf[k][1](t)]
                if len(g) >= MIN_TRADES:
                    s = stats([t.limit_pnl_r for t in g])
                    results.append((names, len(g), s["exp"], s["wr"]))

    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {'Filter combination':<58s}  {'n':>3s}  {'Exp(limit)':>10s}  {'WR':>5s}")
    print(f"  {'─'*58}  {'─'*3}  {'─'*10}  {'─'*5}")
    for names, n, exp, wr in results[:25]:
        label  = " + ".join(names)
        marker = "✓✓" if exp > 0.20 else ("✓ " if exp > 0.05 else "  ")
        print(f"  {marker} {label:<56s}  {n:3d}  {exp:+.3f}R    {wr:.0f}%")

    # ── Limit Fill Rate Analysis ────────────────────────
    print(f"\n{'─' * 100}")
    print("  LIMIT ENTRY FILL RATE ANALYSIS")
    print("  Question: if we place a limit at ORB boundary, does price pull back to fill us?")
    filled     = [t for t in trades if t.limit_filled]
    not_filled = [t for t in trades if not t.limit_filled]
    fill_pct   = len(filled) / len(trades) * 100

    print(f"  Overall fill rate:   {len(filled)}/{len(trades)} = {fill_pct:.1f}%")
    if filled:
        fs = stats([t.limit_pnl_r for t in filled])
        print(f"  Filled trades:       exp {fs['exp']:+.3f}R   WR {fs['wr']:.0f}%   n={len(filled)}")
    if not_filled:
        nfs = stats([t.limit_pnl_r for t in not_filled])
        print(f"  Missed trades:       exp {nfs['exp']:+.3f}R   WR {nfs['wr']:.0f}%   n={len(not_filled)}")
        print(f"  → Missed = strong straight-line breakouts that never pull back")
        print(f"  → These are often the best moves — worth noting the bias")

    # Fill rate by time of day
    print(f"\n  Fill rate by entry time:")
    for label, t0, t1 in time_buckets:
        g = [t for t in trades if t0 <= dtime.fromisoformat(t.entry_time) < t1]
        if len(g) >= 5:
            fr = sum(1 for t in g if t.limit_filled) / len(g) * 100
            print(f"    {label:15s}  {fr:4.0f}% fill  (n={len(g)})")

    # ── Final Verdict ───────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  VERDICT")
    overall_actual = stats([t.eod_pnl_r   for t in trades])
    overall_limit  = stats([t.limit_pnl_r for t in trades])
    print(f"  Overall actual entry:  {overall_actual['exp']:+.3f}R  WR {overall_actual['wr']:.0f}%")
    print(f"  Overall limit entry:   {overall_limit['exp']:+.3f}R  WR {overall_limit['wr']:.0f}%")
    print()

    if results:
        best_names, best_n, best_exp, best_wr = results[0]
        best_label = " + ".join(best_names)
        print(f"  Best filtered subset (limit entry):  {best_exp:+.3f}R  WR {best_wr:.0f}%")
        print(f"  Filter: [{best_label}]  (n={best_n})")
        print()
        if best_exp > 0.05:
            print("  ✓ EDGE EXISTS in filtered subset")
            print("  Next step → Phase 1: build strategy around these filters")
            print(f"  Recommended filters: {best_label}")
        else:
            print("  ✗ NO EDGE found in any filtered subset")
            print("  Next step → reconsider: different timeframe, instrument, or setup type")

    print("=" * 100)


# ─── Main ────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase0_5_decomposition.py <csv1> [csv2] ...")
        sys.exit(1)

    all_trades = []
    for path in sys.argv[1:]:
        print(f"Loading {path}...")
        try:
            df     = load_csv(path)
            symbol = df["symbol"].iloc[0]
            print(f"  {symbol}: {len(df)} bars, {df['date'].nunique()} days")
            trades = find_orb_trades(df)
            print(f"  {symbol}: {len(trades)} trades")
            all_trades.extend(trades)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if not all_trades:
        print("No trades found.")
        sys.exit(1)

    print(f"\nTotal: {len(all_trades)} trades across {len(sys.argv)-1} symbols")
    decompose(all_trades)


if __name__ == "__main__":
    main()
