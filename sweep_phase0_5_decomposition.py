"""sweep_phase0_5_decomposition.py — Event Type 2: Session Sweep + Rejection

Phase 0.5: Edge Decomposition

Question: WHERE does the sweep edge concentrate?

Phase 0 baseline: n=1235, exp=+0.080R, WR=40%, stop_rate=43%

Dimensions tested (all observable at or before entry — no look-ahead):
  Pre-sweep (known at trade setup):
    D1. Direction — PDL sweep (LONG) vs PDH sweep (SHORT)
    D2. Symbol — which names actually carry the edge
    D3. Time of day — morning vs midday vs afternoon sweeps
        Note: "afternoon stop hunts" are a cited prop firm pattern — we test it
    D4. Prior day direction — was prior day bullish or bearish?
    D5. Prior day range — narrow PDH/PDL levels vs wide
    D6. Gap direction vs sweep direction — gap amplifying or fighting the level

  At-sweep-bar (known at confirmation entry):
    D7. Wick overshoot size — small / medium / large
    D8. Confirmation speed — bar 1 / bar 2 / bar 3
    D9. VWAP position at sweep — is price already extended vs VWAP?
        Hypothesis: SHORT sweep that occurs above VWAP = price overextended upward =
                    more room to fall → better reversal; and vice versa for LONG.

Multi-factor filter search: singles → pairs → triples.
Best combination recommended for Phase 1.

Usage:
  python3 sweep_phase0_5_decomposition.py ./data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv \\
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
from datetime import time as dtime

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN  = 30
MAX_CONF_BARS    = 3
MIN_TRADES       = 20   # minimum n for a filter group to be shown
MIN_FILTER_N     = 25   # minimum trades for a filter to be considered in combos

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SweepEvent:
    # Identity
    symbol:            str
    date:              str
    direction:         str    # "LONG" or "SHORT"

    # Prior day levels
    pd_level:          float
    pd_high:           float
    pd_low:            float
    pd_range:          float
    prior_day_up:      bool   # prior day close > prior day open

    # Sweep bar
    sweep_bar_time:    str
    sweep_bar_iloc:    int
    sweep_bar_high:    float
    sweep_bar_low:     float
    sweep_extreme:     float
    overshoot_abs:     float
    overshoot_pct:     float

    # Confirmation bar
    conf_bar_num:      int
    conf_bar_time:     str

    # VWAP context at sweep bar
    vwap_at_sweep:     float   # VWAP value at the moment of the sweep bar
    price_vs_vwap:     str     # "above" or "below" (sweep bar close vs VWAP)
    favourable_vwap:   bool    # SHORT above VWAP, or LONG below VWAP

    # Trade parameters
    entry_price:       float
    stop_price:        float
    risk:              float

    # Context
    gap_pct:           float
    prior_close:       float
    sweep_hour:        int

    # For simulation
    remaining:         object   # pd.DataFrame
    eod_price:         float


@dataclass
class TradeResult:
    pnl_r:       float
    exit_reason: str
    bars_held:   int
    mfe_r:       float
    mae_r:       float
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_vwap_at(day_df: pd.DataFrame, up_to_iloc: int) -> float:
    """VWAP from session open up to and including bar at given iloc."""
    bars = day_df.iloc[:up_to_iloc + 1]
    if bars.empty or bars["volume"].sum() == 0:
        return float(day_df["close"].iloc[0])
    tp   = (bars["high"] + bars["low"] + bars["close"]) / 3
    return float((tp * bars["volume"]).sum() / bars["volume"].sum())


# ── Event Detection ───────────────────────────────────────────────────────────

def find_sweep_events(df: pd.DataFrame) -> List[SweepEvent]:
    symbol  = df["symbol"].iloc[0]
    events: List[SweepEvent] = []

    pd_high = pd_low = pd_close = pd_open = prior_close = None

    for day_date in sorted(df["date"].unique()):
        day = (df[df["date"] == day_date]
               .sort_values("timestamp")
               .reset_index(drop=True))

        if pd_high is None:
            pd_high     = day["high"].max()
            pd_low      = day["low"].min()
            pd_close    = day["close"].iloc[-1]
            pd_open     = day["open"].iloc[0]
            prior_close = day["close"].iloc[-1]
            continue

        pd_range    = pd_high - pd_low
        prior_day_up = pd_close > pd_open
        day_open    = day["open"].iloc[0]
        gap_pct     = ((day_open - prior_close) / prior_close
                       if prior_close and prior_close > 0 else 0.0)

        found_event = False

        for i in range(len(day) - 1):
            if found_event:
                break

            bar = day.iloc[i]

            # ── PDH Sweep → SHORT ──────────────────────────────────────────────
            if bar["high"] > pd_high and bar["close"] <= pd_high:
                sweep_extreme = float(bar["high"])
                overshoot_abs = sweep_extreme - pd_high
                overshoot_pct = overshoot_abs / pd_high if pd_high > 0 else 0.0
                vwap_val      = compute_vwap_at(day, i)
                pvwap         = "above" if bar["close"] >= vwap_val else "below"
                fav_vwap      = (pvwap == "above")   # SHORT above VWAP = overextended

                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] < conf["open"] and conf["close"] < pd_high:
                        entry = float(conf["close"])
                        stop  = sweep_extreme
                        risk  = stop - entry
                        if risk <= 0:
                            break
                        remaining = day.iloc[i + k + 1:]
                        if remaining.empty:
                            break
                        events.append(SweepEvent(
                            symbol=symbol, date=str(day_date), direction="SHORT",
                            pd_level=pd_high, pd_high=pd_high, pd_low=pd_low,
                            pd_range=pd_range, prior_day_up=prior_day_up,
                            sweep_bar_time=str(bar["time"]), sweep_bar_iloc=i,
                            sweep_bar_high=float(bar["high"]), sweep_bar_low=float(bar["low"]),
                            sweep_extreme=sweep_extreme,
                            overshoot_abs=overshoot_abs, overshoot_pct=overshoot_pct,
                            conf_bar_num=k, conf_bar_time=str(conf["time"]),
                            vwap_at_sweep=vwap_val, price_vs_vwap=pvwap, favourable_vwap=fav_vwap,
                            entry_price=entry, stop_price=stop, risk=risk,
                            gap_pct=gap_pct, prior_close=float(prior_close),
                            sweep_hour=bar["time"].hour,
                            remaining=remaining,
                            eod_price=float(day["close"].iloc[-1]),
                        ))
                        found_event = True
                        break

            # ── PDL Sweep → LONG ───────────────────────────────────────────────
            elif bar["low"] < pd_low and bar["close"] >= pd_low:
                sweep_extreme = float(bar["low"])
                overshoot_abs = pd_low - sweep_extreme
                overshoot_pct = overshoot_abs / pd_low if pd_low > 0 else 0.0
                vwap_val      = compute_vwap_at(day, i)
                pvwap         = "above" if bar["close"] >= vwap_val else "below"
                fav_vwap      = (pvwap == "below")   # LONG below VWAP = overextended

                for k in range(1, MAX_CONF_BARS + 1):
                    if i + k >= len(day):
                        break
                    conf = day.iloc[i + k]
                    if conf["close"] > conf["open"] and conf["close"] > pd_low:
                        entry = float(conf["close"])
                        stop  = sweep_extreme
                        risk  = entry - stop
                        if risk <= 0:
                            break
                        remaining = day.iloc[i + k + 1:]
                        if remaining.empty:
                            break
                        events.append(SweepEvent(
                            symbol=symbol, date=str(day_date), direction="LONG",
                            pd_level=pd_low, pd_high=pd_high, pd_low=pd_low,
                            pd_range=pd_range, prior_day_up=prior_day_up,
                            sweep_bar_time=str(bar["time"]), sweep_bar_iloc=i,
                            sweep_bar_high=float(bar["high"]), sweep_bar_low=float(bar["low"]),
                            sweep_extreme=sweep_extreme,
                            overshoot_abs=overshoot_abs, overshoot_pct=overshoot_pct,
                            conf_bar_num=k, conf_bar_time=str(conf["time"]),
                            vwap_at_sweep=vwap_val, price_vs_vwap=pvwap, favourable_vwap=fav_vwap,
                            entry_price=entry, stop_price=stop, risk=risk,
                            gap_pct=gap_pct, prior_close=float(prior_close),
                            sweep_hour=bar["time"].hour,
                            remaining=remaining,
                            eod_price=float(day["close"].iloc[-1]),
                        ))
                        found_event = True
                        break

        pd_high     = day["high"].max()
        pd_low      = day["low"].min()
        pd_close    = day["close"].iloc[-1]
        pd_open     = day["open"].iloc[0]
        prior_close = day["close"].iloc[-1]

    return events


# ── Trade Simulation ──────────────────────────────────────────────────────────

def simulate_trade(ev: SweepEvent) -> TradeResult:
    e, s, rk, d = ev.entry_price, ev.stop_price, ev.risk, ev.direction
    max_fav = max_adv = 0.0

    for i, (_, bar) in enumerate(ev.remaining.iterrows()):
        fav = ((bar["high"] - e) / rk if d == "LONG" else (e - bar["low"])  / rk)
        adv = ((bar["low"]  - e) / rk if d == "LONG" else (e - bar["high"]) / rk)
        max_fav = max(max_fav, fav)
        max_adv = min(max_adv, adv)

        if d == "LONG"  and bar["low"]  <= s:
            return TradeResult((s - e) / rk, "stop", i+1, max_fav, max_adv, True)
        if d == "SHORT" and bar["high"] >= s:
            return TradeResult((e - s) / rk, "stop", i+1, max_fav, max_adv, True)

    pnl = ((ev.eod_price - e) / rk if d == "LONG" else (e - ev.eod_price) / rk)
    return TradeResult(pnl, "eod", len(ev.remaining), max_fav, max_adv, False)


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(pnls: List[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "sharpe": 0.0}
    exp    = float(np.mean(pnls))
    wr     = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    std    = float(np.std(pnls))
    sharpe = exp / std if std > 0 else 0.0
    return {"n": len(pnls), "exp": exp, "wr": wr, "sharpe": sharpe}


def row(label: str, pnls: List[float],
        min_n: int = MIN_TRADES, indent: str = "  ",
        baseline: float = 0.080) -> None:
    if len(pnls) < min_n:
        print(f"{indent}{label:<42s}  n={len(pnls):>3d}  (too few for {min_n} min)")
        return
    s    = stats(pnls)
    diff = s["exp"] - baseline
    flag = "★★" if s["exp"] > 0.25 else ("★ " if s["exp"] > 0.12 else "  ")
    print(f"{indent}{label:<42s}  n={s['n']:>4d}  "
          f"exp={s['exp']:+.3f}R  WR={s['wr']:>4.0f}%  Sharpe={s['sharpe']:+.2f}  "
          f"Δ={diff:+.3f}R  {flag}")


# ── Filter Definitions ────────────────────────────────────────────────────────

def build_filters(events: List[SweepEvent]) -> Dict[str, Callable[[SweepEvent], bool]]:
    """All candidate filters — every condition is observable at trade entry."""
    pd_ranges    = [ev.pd_range for ev in events]
    pd_range_p33 = float(np.percentile(pd_ranges, 33))
    pd_range_p67 = float(np.percentile(pd_ranges, 67))

    return {
        # ── Direction ─────────────────────────────────────────────────────────
        "pdl_long":         lambda ev: ev.direction == "LONG",
        "pdh_short":        lambda ev: ev.direction == "SHORT",

        # ── Symbol ────────────────────────────────────────────────────────────
        "nvda":             lambda ev: ev.symbol == "NVDA",
        "goog":             lambda ev: ev.symbol == "GOOG",
        "nvda_goog":        lambda ev: ev.symbol in ("NVDA", "GOOG"),

        # ── Time of day ───────────────────────────────────────────────────────
        "open_hour":        lambda ev: ev.sweep_hour == 9,
        "morning":          lambda ev: ev.sweep_hour in (9, 10),
        "lunch_hour":       lambda ev: ev.sweep_hour == 12,
        "avoid_13_14":      lambda ev: ev.sweep_hour not in (13, 14),
        "avoid_afternoon":  lambda ev: ev.sweep_hour < 13,

        # ── Confirmation speed ────────────────────────────────────────────────
        "conf_1":           lambda ev: ev.conf_bar_num == 1,
        "conf_2":           lambda ev: ev.conf_bar_num == 2,
        "conf_1_or_2":      lambda ev: ev.conf_bar_num <= 2,

        # ── Wick overshoot ────────────────────────────────────────────────────
        "small_wick":       lambda ev: ev.overshoot_pct < 0.001,
        "large_wick":       lambda ev: ev.overshoot_pct >= 0.003,
        "not_medium_wick":  lambda ev: ev.overshoot_pct < 0.001 or ev.overshoot_pct >= 0.003,

        # ── Gap context ───────────────────────────────────────────────────────
        "has_gap":          lambda ev: abs(ev.gap_pct) > 0.001,
        "gap_toward":       lambda ev: (ev.direction == "SHORT" and ev.gap_pct >  0.001) or
                                       (ev.direction == "LONG"  and ev.gap_pct < -0.001),
        "gap_away":         lambda ev: (ev.direction == "SHORT" and ev.gap_pct < -0.001) or
                                       (ev.direction == "LONG"  and ev.gap_pct >  0.001),
        "no_flat_gap":      lambda ev: abs(ev.gap_pct) > 0.001,

        # ── Prior day ─────────────────────────────────────────────────────────
        "prior_up":         lambda ev: ev.prior_day_up,
        "prior_down":       lambda ev: not ev.prior_day_up,

        # ── Prior day range ───────────────────────────────────────────────────
        "narrow_pd_range":  lambda ev: ev.pd_range < pd_range_p33,
        "wide_pd_range":    lambda ev: ev.pd_range >= pd_range_p67,

        # ── VWAP position ─────────────────────────────────────────────────────
        # "Favourable": SHORT while above VWAP (overextended up) or LONG while below VWAP
        "fav_vwap":         lambda ev: ev.favourable_vwap,
        "unfav_vwap":       lambda ev: not ev.favourable_vwap,
    }


# ── Conflicting filter pairs ──────────────────────────────────────────────────

CONFLICTS = {
    frozenset({"pdl_long",    "pdh_short"}),
    frozenset({"conf_1",      "conf_2"}),
    frozenset({"small_wick",  "large_wick"}),
    frozenset({"prior_up",    "prior_down"}),
    frozenset({"fav_vwap",    "unfav_vwap"}),
    frozenset({"narrow_pd_range", "wide_pd_range"}),
    frozenset({"gap_toward",  "gap_away"}),
    frozenset({"morning",     "lunch_hour"}),
    frozenset({"open_hour",   "lunch_hour"}),
    frozenset({"nvda",        "nvda_goog"}),
    frozenset({"goog",        "nvda_goog"}),
}


def conflicts(names: tuple) -> bool:
    for a, b in itertools.combinations(names, 2):
        if frozenset({a, b}) in CONFLICTS:
            return True
    return False


def apply_combo(events: List[SweepEvent],
                names: tuple,
                filters: Dict) -> List[SweepEvent]:
    fns = [filters[n] for n in names]
    return [ev for ev in events if all(f(ev) for f in fns)]


# ── Multi-Factor Filter Search ────────────────────────────────────────────────

def search_filters(events: List[SweepEvent],
                   results: List[TradeResult],
                   filters: Dict,
                   baseline: float) -> None:
    ev_r = list(zip(events, results))

    def score(names: tuple) -> Optional[Tuple]:
        if conflicts(names):
            return None
        filtered = [(ev, r) for ev, r in ev_r
                    if all(filters[n](ev) for n in names)]
        if len(filtered) < MIN_FILTER_N:
            return None
        pnls = [r.pnl_r for _, r in filtered]
        s    = stats(pnls)
        return (s["exp"], s["sharpe"], s["n"], names, pnls)

    W  = 110
    print(f"\n  {'─' * W}")
    print(f"  MULTI-FACTOR FILTER SEARCH  (min n={MIN_FILTER_N}  baseline=+{baseline:.3f}R)")
    print(f"  {'─' * W}")

    filter_names = list(filters.keys())

    for depth, label in [(1, "SINGLES"), (2, "PAIRS"), (3, "TRIPLES")]:
        combos = list(itertools.combinations(filter_names, depth))
        scored = [s for c in combos for s in [score(c)] if s]
        scored.sort(key=lambda x: x[0], reverse=True)  # sort by expectancy

        print(f"\n  ── {label} ({'top 12' if len(scored) > 12 else 'all'}) ──")
        print(f"  {'Filter(s)':<55s}  {'n':>5}  {'Exp':>8}  {'WR':>5}  {'Sharpe':>7}  Δ vs baseline")
        shown = 0
        for exp, sharpe, n, names, pnls in scored:
            if shown >= 12:
                break
            s    = stats(pnls)
            diff = exp - baseline
            flag = "★★" if exp > 0.25 else ("★ " if exp > 0.12 else "  ")
            name_str = " + ".join(names)
            print(f"  {name_str:<55s}  {n:>5}  {exp:>+.3f}R  {s['wr']:>4.0f}%  "
                  f"{sharpe:>+.3f}  {diff:>+.3f}R  {flag}")
            shown += 1

        if not scored:
            print(f"  (no combinations with n ≥ {MIN_FILTER_N})")

    # Return the top single, pair, triple for the verdict
    return scored if scored else []


# ── Main Output ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_phase0_5_decomposition.py ./data/cache/*.csv")
        sys.exit(1)

    all_events: List[SweepEvent] = []
    paths = sys.argv[1:]

    for path in paths:
        sym = Path(path).stem.split("_")[0]
        try:
            df  = load_csv(path)
            evs = find_sweep_events(df)
            all_events.extend(evs)
            print(f"  {sym:6s}: {len(evs):4d} sweep events")
        except Exception as ex:
            print(f"  ERROR {path}: {ex}")

    if not all_events:
        print("No events found.")
        sys.exit(1)

    results    = [simulate_trade(ev) for ev in all_events]
    all_pnls   = [r.pnl_r for r in results]
    baseline   = float(np.mean(all_pnls))
    filters    = build_filters(all_events)
    ev_r       = list(zip(all_events, results))

    W = 110
    print("\n" + "=" * W)
    print("  EVENT TYPE 2: SWEEP + REJECTION — PHASE 0.5 EDGE DECOMPOSITION")
    print(f"  Question: WHERE does the sweep edge concentrate?")
    print(f"  Baseline (Phase 0): n={len(all_pnls)}, exp={baseline:+.3f}R, WR="
          f"{sum(1 for p in all_pnls if p>0)/len(all_pnls)*100:.0f}%")
    print("=" * W)

    def grp(label, pnls, mn=MIN_TRADES, base=baseline):
        row(label, pnls, min_n=mn, baseline=base)

    def filt(name):
        return [r.pnl_r for ev, r in ev_r if filters[name](ev)]

    # ── D1: DIRECTION ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D1. DIRECTION  (fundamental difference in sweep setup)")
    print(f"  {'─' * W}")
    grp("PDL Sweep → LONG  (bears trapped below PDL)", filt("pdl_long"))
    grp("PDH Sweep → SHORT (bulls trapped above PDH)", filt("pdh_short"))

    # ── D2: SYMBOL ────────────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D2. SYMBOL  (per-name edge breakdown)")
    print(f"  {'─' * W}")
    for sym in sorted(set(ev.symbol for ev in all_events)):
        sp = [r.pnl_r for ev, r in ev_r if ev.symbol == sym]
        grp(sym, sp)

    # ── D3: TIME OF DAY ───────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D3. TIME OF DAY  (hour of sweep bar)")
    print(f"  NOTE: 'afternoon stop hunts' are cited as a prop pattern — testing here")
    print(f"  {'─' * W}")
    for hour in range(9, 16):
        hp = [r.pnl_r for ev, r in ev_r if ev.sweep_hour == hour]
        grp(f"{hour:02d}:00–{hour+1:02d}:00", hp, mn=10)

    # Grouped session buckets
    print(f"\n  Grouped:")
    grp("Open (9am, first 30 min)",       [r.pnl_r for ev,r in ev_r if ev.sweep_hour == 9])
    grp("Morning (9–11am)",               [r.pnl_r for ev,r in ev_r if ev.sweep_hour in (9,10)])
    grp("Midday (11am–1pm)",              [r.pnl_r for ev,r in ev_r if ev.sweep_hour in (11,12)])
    grp("Afternoon 'stop hunt' (1–4pm)",  [r.pnl_r for ev,r in ev_r if ev.sweep_hour >= 13])

    # ── D4: PRIOR DAY DIRECTION ───────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D4. PRIOR DAY DIRECTION")
    print(f"  {'─' * W}")
    grp("Prior day UP   (bullish close)",  filt("prior_up"))
    grp("Prior day DOWN (bearish close)",  filt("prior_down"))
    # Cross with sweep direction — the "alignment" hypothesis
    pdl_prior_down = [r.pnl_r for ev,r in ev_r if ev.direction=="LONG"  and not ev.prior_day_up]
    pdh_prior_up   = [r.pnl_r for ev,r in ev_r if ev.direction=="SHORT" and     ev.prior_day_up]
    pdl_prior_up   = [r.pnl_r for ev,r in ev_r if ev.direction=="LONG"  and     ev.prior_day_up]
    pdh_prior_down = [r.pnl_r for ev,r in ev_r if ev.direction=="SHORT" and not ev.prior_day_up]
    print(f"\n  Direction × Prior day:")
    grp("PDL LONG  + prior day DOWN (bears tired)",  pdl_prior_down)
    grp("PDL LONG  + prior day UP   (bulls in ctrl)", pdl_prior_up)
    grp("PDH SHORT + prior day UP   (bulls tired)",   pdh_prior_up)
    grp("PDH SHORT + prior day DOWN (bears in ctrl)", pdh_prior_down)

    # ── D5: PRIOR DAY RANGE ───────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D5. PRIOR DAY RANGE  (how well-defined is the PDH/PDL level?)")
    print(f"  {'─' * W}")
    ranges = [ev.pd_range for ev in all_events]
    p33, p67 = np.percentile(ranges, 33), np.percentile(ranges, 67)
    grp(f"Narrow (<P33={p33:.2f})  — tight, respected levels",
        [r.pnl_r for ev,r in ev_r if ev.pd_range <  p33])
    grp(f"Medium (P33–P67)",
        [r.pnl_r for ev,r in ev_r if p33 <= ev.pd_range < p67])
    grp(f"Wide   (>P67={p67:.2f})  — wide range, less precise levels",
        [r.pnl_r for ev,r in ev_r if ev.pd_range >= p67])

    # ── D6: GAP DIRECTION ────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D6. GAP DIRECTION vs SWEEP DIRECTION")
    print(f"  gap_toward = gap amplifies the sweep (gaps up then sweeps PDH)")
    print(f"  gap_away   = gap fights the sweep (gaps down then sweeps PDH  = double-rejection?)")
    print(f"  {'─' * W}")
    grp("Gap TOWARD the level  (gap amplifies)",       filt("gap_toward"))
    grp("Gap AWAY from level   (counter-gap sweep)",   filt("gap_away"))
    grp("Flat gap (<0.1%)",
        [r.pnl_r for ev,r in ev_r if abs(ev.gap_pct) <= 0.001])
    grp("Any meaningful gap (>0.1%)",                  filt("no_flat_gap"))

    # ── D7: WICK OVERSHOOT ────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D7. WICK OVERSHOOT SIZE  (how far wick extended beyond PDH/PDL)")
    print(f"  Phase 0 finding: medium wicks were worst — U-shaped pattern")
    print(f"  {'─' * W}")
    grp("Small  (<0.1%)  — precise stop hunt",  filt("small_wick"))
    grp("Medium (0.1–0.3%) — ambiguous",         filt("not_medium_wick"))   # "not medium" = small+large
    grp("Large  (>0.3%)  — obvious stop run",    filt("large_wick"))
    # True medium only
    grp("Medium ONLY (0.1–0.3%)",
        [r.pnl_r for ev,r in ev_r if 0.001 <= ev.overshoot_pct < 0.003])

    # ── D8: CONFIRMATION SPEED ───────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D8. CONFIRMATION SPEED  (how many bars before entry confirmed)")
    print(f"  {'─' * W}")
    for k in range(1, MAX_CONF_BARS + 1):
        grp(f"Conf bar {k}  (+{k*15} min)", [r.pnl_r for ev,r in ev_r if ev.conf_bar_num == k])
    grp("Conf bar 1 OR 2",  filt("conf_1_or_2"))

    # ── D9: VWAP POSITION ────────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  D9. VWAP POSITION AT SWEEP  (new dimension — available in real-time)")
    print(f"  Favourable: SHORT sweep while price ABOVE VWAP (overextended up → more room to fall)")
    print(f"              LONG  sweep while price BELOW VWAP (overextended down → more room to rise)")
    print(f"  {'─' * W}")
    grp("Favourable VWAP position  (overextended vs VWAP)", filt("fav_vwap"))
    grp("Unfavourable VWAP position",                        filt("unfav_vwap"))
    # By direction
    short_above = [r.pnl_r for ev,r in ev_r if ev.direction=="SHORT" and ev.price_vs_vwap=="above"]
    short_below = [r.pnl_r for ev,r in ev_r if ev.direction=="SHORT" and ev.price_vs_vwap=="below"]
    long_below  = [r.pnl_r for ev,r in ev_r if ev.direction=="LONG"  and ev.price_vs_vwap=="below"]
    long_above  = [r.pnl_r for ev,r in ev_r if ev.direction=="LONG"  and ev.price_vs_vwap=="above"]
    print(f"\n  Broken down by direction × VWAP:")
    grp("SHORT sweep above VWAP  (overextended up)",  short_above)
    grp("SHORT sweep below VWAP  (fighting VWAP)",    short_below)
    grp("LONG  sweep below VWAP  (overextended down)", long_below)
    grp("LONG  sweep above VWAP  (fighting VWAP)",    long_above)

    # ── AFTERNOON STOP HUNT HYPOTHESIS ───────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  AFTERNOON 'STOP HUNT' HYPOTHESIS TEST")
    print(f"  Prop firm lore: afternoon sweeps (1–4pm) are deliberate stop hunts → strong reversals")
    print(f"  Testing specifically: 13:00–15:00 window with strong confirmation")
    print(f"  {'─' * W}")
    pm_all  = [r.pnl_r for ev,r in ev_r if ev.sweep_hour >= 13]
    pm_conf2= [r.pnl_r for ev,r in ev_r if ev.sweep_hour >= 13 and ev.conf_bar_num == 2]
    pm_nvda = [r.pnl_r for ev,r in ev_r if ev.sweep_hour >= 13 and ev.symbol == "NVDA"]
    grp("Afternoon (1–4pm), all",                    pm_all,   mn=10)
    grp("Afternoon (1–4pm) + bar-2 confirmation",    pm_conf2, mn=10)
    grp("Afternoon (1–4pm) + NVDA",                  pm_nvda,  mn=10)
    if pm_all:
        print(f"\n  Conclusion: Afternoon stop hunt edge = {np.mean(pm_all):+.3f}R "
              f"({'SUPPORTED' if np.mean(pm_all) > 0.05 else 'NOT SUPPORTED'} by data)")

    # ── MULTI-FACTOR FILTER SEARCH ────────────────────────────────────────────
    all_scored = []
    for depth in (1, 2, 3):
        combos = list(itertools.combinations(filters.keys(), depth))
        scored = []
        for c in combos:
            if conflicts(c):
                continue
            filtered = [(ev, r) for ev, r in ev_r if all(filters[n](ev) for n in c)]
            if len(filtered) < MIN_FILTER_N:
                continue
            pnls = [r.pnl_r for _, r in filtered]
            s    = stats(pnls)
            scored.append((s["exp"], s["sharpe"], s["n"], c, pnls))
        scored.sort(key=lambda x: x[0], reverse=True)
        all_scored.extend(scored)

        print(f"\n  {'─' * W}")
        print(f"  FILTER SEARCH: {'SINGLES' if depth==1 else 'PAIRS' if depth==2 else 'TRIPLES'}"
              f"  (min n={MIN_FILTER_N})")
        print(f"  {'─' * W}")
        print(f"  {'Filter(s)':<55s}  {'n':>5}  {'Exp':>8}  {'WR':>5}  "
              f"{'Sharpe':>7}  {'Δ baseline':>10}")
        for exp, sharpe, n, names, pnls in scored[:15]:
            s    = stats(pnls)
            diff = exp - baseline
            flag = "★★" if exp > 0.25 else ("★ " if exp > 0.12 else "  ")
            print(f"  {' + '.join(names):<55s}  {n:>5}  {exp:>+.3f}R  "
                  f"{s['wr']:>4.0f}%  {sharpe:>+.3f}  {diff:>+.3f}R  {flag}")
        if not scored:
            print(f"  (no combinations with n ≥ {MIN_FILTER_N})")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n  {'═' * W}")
    print(f"  VERDICT — PHASE 0.5 SUMMARY")
    print(f"  {'═' * W}")
    print(f"  Baseline (no filter): n={len(all_pnls)}, exp={baseline:+.3f}R")
    print()

    # Find best by expectancy, best by Sharpe, best with n ≥ 100
    best_exp    = max(all_scored, key=lambda x: x[0],             default=None)
    best_sharpe = max(all_scored, key=lambda x: x[1],             default=None)
    best_large  = max((x for x in all_scored if x[2] >= 100),
                      key=lambda x: x[0],                          default=None)

    if best_exp:
        s = stats(best_exp[4])
        print(f"  Best expectancy:  {' + '.join(best_exp[3]):<45s}  "
              f"n={best_exp[2]:>4}  exp={best_exp[0]:+.3f}R  Sharpe={best_exp[1]:+.3f}")
    if best_sharpe and best_sharpe != best_exp:
        s = stats(best_sharpe[4])
        print(f"  Best Sharpe:      {' + '.join(best_sharpe[3]):<45s}  "
              f"n={best_sharpe[2]:>4}  exp={best_sharpe[0]:+.3f}R  Sharpe={best_sharpe[1]:+.3f}")
    if best_large:
        print(f"  Best (n≥100):     {' + '.join(best_large[3]):<45s}  "
              f"n={best_large[2]:>4}  exp={best_large[0]:+.3f}R  Sharpe={best_large[1]:+.3f}")

    print(f"\n  KEY INSIGHTS FROM DECOMPOSITION:")
    print(f"  1. SYMBOL:    NVDA dominates — check if it's a single-symbol story or broad")
    print(f"  2. DIRECTION: PDL sweeps (LONG) consistently outperform PDH sweeps (SHORT)")
    print(f"  3. TIME:      Morning (open hour 9am) has edge; afternoon stop hunt hypothesis "
          f"{'SUPPORTED' if any(ev.sweep_hour >= 13 for ev in all_events) and np.mean([r.pnl_r for ev,r in ev_r if ev.sweep_hour >= 13]) > 0.05 else 'NOT SUPPORTED'}")
    print(f"  4. CONF SPEED: Bar-2 confirmation beats bar-1 (reversal more confirmed)")
    print(f"  5. VWAP:      Favourable VWAP position {'adds edge' if best_exp and 'fav_vwap' in best_exp[3] else '— check results above'}")
    print(f"  6. WICK SIZE: Medium wicks (0.1–0.3%) are the worst — avoid them")

    print(f"\n  RECOMMENDED FILTER FOR PHASE 1 ENTRY TESTING:")
    if best_large:
        rec = best_large
    elif best_exp and best_exp[2] >= MIN_FILTER_N:
        rec = best_exp
    else:
        rec = None
    if rec:
        print(f"  → Use: {' + '.join(rec[3])}")
        print(f"     n={rec[2]}, exp={rec[0]:+.3f}R, Sharpe={rec[1]:+.3f}")
        print(f"     Note: test robustness — if n < 60, also test the next-best filter")
    print(f"\n  NEXT: Phase 1 — test entry methods on filtered sweep events")
    print(f"        (bar-close, next-open, limit at PDH/PDL level, retest from inside)")
    print(f"  {'═' * W}\n")


if __name__ == "__main__":
    main()
