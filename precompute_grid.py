"""Precompute parameter grid for the interactive showcase.

Runs N backtests across gap_threshold * rvol_threshold_sweep and saves
results to data/interactive_grid.json for instant client-side filtering
in the Streamlit dashboard (no re-computation on the cloud server).

Usage:
    python3 precompute_grid.py               # full 30-combination grid
    python3 precompute_grid.py --dry-run     # 4 combinations for quick test
    python3 precompute_grid.py --gap 0.002 0.003 --rvol 0.0 1.0

Grid (default):
    gap_threshold       : [0.001, 0.002, 0.003, 0.004, 0.005]
    rvol_threshold_sweep: [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    Total               : 30 combinations × ~15s each ≈ 7-8 minutes

Output:
    data/interactive_grid.json  (~5-10 MB committed to repo for cloud)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest import BacktestEngine

# Grid definition 
GAP_THRESHOLDS  = [0.001, 0.002, 0.003, 0.004, 0.005]
RVOL_THRESHOLDS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]

START       = "2024-01-01"
END         = "2025-12-31"
INTERVAL    = "15m"
DATA_SOURCE = "ibkr"          # reads from data/cache/ CSV files
OUTPUT_PATH = PROJECT_ROOT / "data" / "interactive_grid.json"

# Max equity-curve points to keep per run (keeps JSON lean)
MAX_EQUITY_POINTS = 600


#  Helpers 

def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _equity_curve_points(portfolio_history: pd.DataFrame) -> list:
    """Downsample portfolio history to ≤MAX_EQUITY_POINTS rows."""
    if portfolio_history is None or portfolio_history.empty:
        return []
    step = max(1, len(portfolio_history) // MAX_EQUITY_POINTS)
    sampled = portfolio_history.iloc[::step]
    points = []
    for idx, row in sampled.iterrows():
        equity_val = row.get("equity", row.get("total_value", None))
        if equity_val is None and len(row) > 0:
            equity_val = float(row.iloc[0])
        points.append({
            "ts": str(idx)[:19],
            "equity": _safe_float(equity_val, 10_000.0),
        })
    return points


def _trade_records(paired_trades: pd.DataFrame) -> list:
    """Convert paired_trades DataFrame to a list of compact dicts."""
    if paired_trades is None or paired_trades.empty:
        return []
    records = []
    for _, row in paired_trades.iterrows():
        records.append({
            "symbol":      str(row["symbol"]),
            "strategy":    str(row.get("strategy", "")),
            "date":        str(row["entry_time"])[:10],
            "entry_time":  str(row["entry_time"])[:19],
            "exit_time":   str(row["exit_time"])[:19],
            "direction":   str(row["direction"]),
            "qty":         int(row["quantity"]),
            "entry_px":    round(_safe_float(row["entry_price"]), 4),
            "exit_px":     round(_safe_float(row["exit_price"]), 4),
            "net_pnl":     round(_safe_float(row["net_pnl"]), 2),
            "pnl_pct":     round(_safe_float(row["pnl_pct"]), 3),
            "win":         bool(row["win"]),
        })
    return records


def _by_group(paired_trades: pd.DataFrame, col: str) -> dict:
    """Aggregate paired_trades by a grouping column (strategy or symbol)."""
    if paired_trades is None or paired_trades.empty or col not in paired_trades.columns:
        return {}
    out = {}
    for key, grp in paired_trades.groupby(col):
        n = len(grp)
        n_wins = int(grp["win"].sum())
        out[str(key)] = {
            "n":         n,
            "n_wins":    n_wins,
            "win_rate":  round(n_wins / n, 4) if n > 0 else 0.0,
            "total_pnl": round(_safe_float(grp["net_pnl"].sum()), 2),
            "avg_pnl":   round(_safe_float(grp["net_pnl"].mean()), 2),
        }
    return out


# ── Single backtest runner ─────────────────────────────────────────────────────

def run_single(config: dict, gap_threshold: float, rvol_threshold: float) -> dict:
    """Run one full backtest and return a grid-point dict."""
    engine = BacktestEngine(config)
    engine.add_strategy("event_driven", params={
        "gap_threshold":        gap_threshold,
        "rvol_threshold_sweep": rvol_threshold,
    })

    results = engine.run(
        start=START,
        end=END,
        interval=INTERVAL,
        data_source=DATA_SOURCE,
    )

    metrics       = results["metrics"]
    paired_trades = results["paired_trades"]
    port_history  = results["portfolio_history"]

    n_trades = int(metrics.get("total_round_trips", 0))
    sharpe   = _safe_float(metrics.get("sharpe_ratio", 0))
    win_rate = _safe_float(metrics.get("win_rate", 0))
    tot_ret  = _safe_float(metrics.get("total_return", 0))
    max_dd   = _safe_float(metrics.get("max_drawdown", 0))
    exp      = _safe_float(metrics.get("expectancy", 0))
    n_wins   = int(metrics.get("n_wins", 0))

    return {
        "gap_threshold":  gap_threshold,
        "rvol_threshold": rvol_threshold,
        "summary": {
            "n_trades":     n_trades,
            "n_wins":       n_wins,
            "win_rate":     round(win_rate, 4),
            "sharpe":       round(sharpe, 4),
            "total_return": round(tot_ret, 6),
            "max_drawdown": round(max_dd, 6),
            "total_pnl":    round(exp * n_trades, 2),
            "avg_pnl":      round(exp, 2),
        },
        "by_strategy": _by_group(paired_trades, "strategy"),
        "by_symbol":   _by_group(paired_trades, "symbol"),
        "trades":      _trade_records(paired_trades),
        "equity_curve": _equity_curve_points(port_history),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute parameter grid backtests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 4 combinations only (2×2 corner subset)")
    parser.add_argument("--gap",  type=float, nargs="+",
                        help="Override gap thresholds (e.g. --gap 0.001 0.002)")
    parser.add_argument("--rvol", type=float, nargs="+",
                        help="Override rvol thresholds (e.g. --rvol 0.0 1.0)")
    args = parser.parse_args()

    gap_grid  = args.gap  or GAP_THRESHOLDS
    rvol_grid = args.rvol or RVOL_THRESHOLDS

    if args.dry_run:
        gap_grid  = [gap_grid[0],  gap_grid[-1]]
        rvol_grid = [rvol_grid[0], rvol_grid[-1]]
        print("DRY RUN — using 2×2 corner grid for speed check")

    config = _load_config()
    total  = len(gap_grid) * len(rvol_grid)

    print(f"\nPrecomputing {len(gap_grid)} × {len(rvol_grid)} = {total} backtests")
    print(f"  Gap thresholds : {gap_grid}")
    print(f"  RVOL thresholds: {rvol_grid}")
    print(f"  Period         : {START} → {END}  interval={INTERVAL}")
    print(f"  Output         : {OUTPUT_PATH}\n")

    grid_results = []
    done = 0
    t0   = time.time()

    for gap in gap_grid:
        for rvol in rvol_grid:
            done += 1
            label = f"[{done:>2}/{total}] gap={gap:.3f}  rvol={rvol:.2f}"
            print(f"{label} ... ", end="", flush=True)
            t1 = time.time()
            try:
                result = run_single(config, gap, rvol)
                elapsed = time.time() - t1
                s = result["summary"]
                print(
                    f"OK  {s['n_trades']:3d} trades  "
                    f"Sharpe {s['sharpe']:+.3f}  "
                    f"WR {s['win_rate']:.0%}  "
                    f"({elapsed:.1f}s)"
                )
                grid_results.append(result)
            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                traceback.print_exc()

    total_elapsed = time.time() - t0
    print(f"\n  Finished {len(grid_results)}/{total} runs in "
          f"{total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ── Serialize ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "meta": {
            "start":           START,
            "end":             END,
            "interval":        INTERVAL,
            "gap_thresholds":  list(gap_grid),
            "rvol_thresholds": list(rvol_grid),
        },
        "grid": grid_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))   # compact — no spaces

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"  Saved {OUTPUT_PATH}  ({size_mb:.1f} MB, {len(grid_results)} grid points)")
    print("\nNext: streamlit run dashboard_app.py → Interactive Explorer tab")


if __name__ == "__main__":
    main()
