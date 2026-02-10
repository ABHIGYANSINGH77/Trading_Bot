#!/usr/bin/env python3
"""QuantBot Strategy Comparator.

Compare multiple backtest reports side-by-side to determine
which strategy performs best under which conditions.

Usage:
    # Compare two strategies
    python compare.py bos_report.json ma_report.json

    # Compare many
    python compare.py reports/*.json

    # Save comparison to CSV
    python compare.py bos_report.json ma_report.json --csv comparison.csv

    # Export as HTML dashboard
    python compare.py bos_report.json ma_report.json --html comparison.html
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


def load_report(path: str) -> dict:
    """Load and extract key metrics from a backtest report."""
    with open(path) as f:
        r = json.load(f)

    meta = r.get("meta", {})
    metrics = r.get("metrics", {})
    paired = r.get("paired_trades", [])
    signals = r.get("signals", [])

    # Compute trade-level stats
    wins = [t for t in paired if t.get("win")]
    losses = [t for t in paired if not t.get("win")]
    pnls = [t.get("pnl_pct", 0) for t in paired]
    win_pnls = [t.get("pnl_pct", 0) for t in wins]
    loss_pnls = [t.get("pnl_pct", 0) for t in losses]

    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    best = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0
    total_pnl_dollar = sum(t.get("net_pnl", 0) for t in paired)

    # Win/loss streaks
    max_win_streak = max_loss_streak = cur_streak = 0
    streak_type = None
    for t in paired:
        if t.get("win"):
            if streak_type == "win":
                cur_streak += 1
            else:
                cur_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, cur_streak)
        else:
            if streak_type == "loss":
                cur_streak += 1
            else:
                cur_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, cur_streak)

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else (99 if gross_profit > 0 else 0)

    return {
        "file": Path(path).name,
        "strategy": ", ".join(meta.get("strategies", ["?"])),
        "symbol": ", ".join(meta.get("symbols", ["?"])),
        "interval": meta.get("interval", "?"),
        "period": f"{meta.get('start', '?')} → {meta.get('end', '?')}",
        "bars": metrics.get("total_bars", 0),
        "initial_capital": metrics.get("initial_capital", 100000),
        "final_value": metrics.get("final_value", 0),
        "total_return": metrics.get("total_return", 0),
        "annualized_return": metrics.get("annualized_return", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "sortino_ratio": metrics.get("sortino_ratio", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "volatility": metrics.get("volatility", 0),
        "total_trades": len(paired),
        "total_signals": len(signals),
        "win_rate": len(wins) / len(paired) if paired else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best,
        "worst_trade": worst,
        "profit_factor": pf,
        "expectancy": sum(pnls) / len(pnls) if pnls else 0,
        "total_pnl": total_pnl_dollar,
        "commission": metrics.get("total_commission", 0),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "risk_rejected": metrics.get("risk_rejected", 0),
    }


def fmt(v, kind=""):
    """Format a value for display."""
    if kind == "pct":
        return f"{v:+.2%}" if isinstance(v, (int, float)) else str(v)
    elif kind == "pct2":
        return f"{v:.2f}%" if isinstance(v, (int, float)) else str(v)
    elif kind == "dollar":
        return f"${v:,.2f}" if isinstance(v, (int, float)) else str(v)
    elif kind == "ratio":
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    elif kind == "int":
        return f"{int(v):,}" if isinstance(v, (int, float)) else str(v)
    return str(v)


def print_comparison(reports: list):
    """Print side-by-side comparison to terminal."""
    if not reports:
        print("No reports to compare.")
        return

    # Determine column widths
    label_width = 22
    col_width = max(18, max(len(r["strategy"]) + len(r["symbol"]) + 3 for r in reports))

    # Header
    print()
    print("=" * (label_width + 2 + col_width * len(reports) + len(reports)))
    print("  STRATEGY COMPARISON")
    print("=" * (label_width + 2 + col_width * len(reports) + len(reports)))

    def row(label, key, kind=""):
        parts = [f"  {label:<{label_width}}"]
        vals = [r[key] for r in reports]
        best_idx = None

        # Highlight best value
        if kind in ("pct", "ratio") and any(isinstance(v, (int, float)) for v in vals):
            nums = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
            if key == "max_drawdown":
                best_idx = min(nums, key=lambda x: abs(x[1]))[0]
            elif key in ("avg_loss", "worst_trade", "max_loss_streak"):
                best_idx = min(nums, key=lambda x: abs(x[1]))[0]
            else:
                best_idx = max(nums, key=lambda x: x[1])[0]

        for i, r in enumerate(reports):
            val = fmt(r[key], kind)
            if i == best_idx:
                val = f"★ {val}"
            parts.append(f"{val:>{col_width}}")

        print(" │ ".join(parts))

    # Strategy info
    print(f"  {'':─<{label_width}}─┼─" + "─┼─".join(["─" * col_width] * len(reports)))
    for r in reports:
        pass
    row("Strategy", "strategy")
    row("Symbol", "symbol")
    row("Interval", "interval")
    row("Period", "period")
    row("Bars", "bars", "int")

    print(f"  {'':─<{label_width}}─┼─" + "─┼─".join(["─" * col_width] * len(reports)))

    # Returns
    row("Total Return", "total_return", "pct")
    row("Annualized Return", "annualized_return", "pct")
    row("Final Value", "final_value", "dollar")
    row("Total P&L", "total_pnl", "dollar")

    print(f"  {'':─<{label_width}}─┼─" + "─┼─".join(["─" * col_width] * len(reports)))

    # Risk
    row("Sharpe Ratio", "sharpe_ratio", "ratio")
    row("Sortino Ratio", "sortino_ratio", "ratio")
    row("Max Drawdown", "max_drawdown", "pct")
    row("Volatility", "volatility", "pct")

    print(f"  {'':─<{label_width}}─┼─" + "─┼─".join(["─" * col_width] * len(reports)))

    # Trades
    row("Total Trades", "total_trades", "int")
    row("Win Rate", "win_rate", "pct")
    row("Avg Win", "avg_win", "pct2")
    row("Avg Loss", "avg_loss", "pct2")
    row("Best Trade", "best_trade", "pct2")
    row("Worst Trade", "worst_trade", "pct2")
    row("Profit Factor", "profit_factor", "ratio")
    row("Expectancy", "expectancy", "pct2")
    row("Max Win Streak", "max_win_streak", "int")
    row("Max Loss Streak", "max_loss_streak", "int")

    print(f"  {'':─<{label_width}}─┼─" + "─┼─".join(["─" * col_width] * len(reports)))

    # Costs
    row("Commission", "commission", "dollar")
    row("Risk Rejected", "risk_rejected", "int")
    row("Signals Generated", "total_signals", "int")

    print("=" * (label_width + 2 + col_width * len(reports) + len(reports)))

    # Verdict
    print("\n  VERDICT:")
    if len(reports) >= 2:
        by_sharpe = sorted(reports, key=lambda r: r["sharpe_ratio"], reverse=True)
        by_return = sorted(reports, key=lambda r: r["total_return"] if isinstance(r["total_return"], (int, float)) else 0, reverse=True)
        by_dd = sorted(reports, key=lambda r: abs(r["max_drawdown"]) if isinstance(r["max_drawdown"], (int, float)) else 1)
        by_pf = sorted(reports, key=lambda r: r["profit_factor"], reverse=True)

        print(f"    Best risk-adjusted (Sharpe):  {by_sharpe[0]['strategy']} on {by_sharpe[0]['symbol']} ({by_sharpe[0]['interval']})")
        print(f"    Best raw return:              {by_return[0]['strategy']} on {by_return[0]['symbol']} ({by_return[0]['interval']})")
        print(f"    Lowest drawdown:              {by_dd[0]['strategy']} on {by_dd[0]['symbol']} ({by_dd[0]['interval']})")
        print(f"    Best profit factor:           {by_pf[0]['strategy']} on {by_pf[0]['symbol']} ({by_pf[0]['interval']})")

        # Overall score (weighted)
        for r in reports:
            score = 0
            if isinstance(r["sharpe_ratio"], (int, float)):
                score += r["sharpe_ratio"] * 30  # Sharpe matters most
            if isinstance(r["win_rate"], (int, float)):
                score += r["win_rate"] * 20
            if isinstance(r["profit_factor"], (int, float)):
                score += min(r["profit_factor"], 5) * 10
            if isinstance(r["max_drawdown"], (int, float)):
                score -= abs(r["max_drawdown"]) * 50  # Penalize drawdown
            if isinstance(r["expectancy"], (int, float)):
                score += r["expectancy"] * 5
            r["_score"] = score

        best = max(reports, key=lambda r: r["_score"])
        print(f"\n    ★ RECOMMENDED: {best['strategy']} on {best['symbol']} ({best['interval']})")
        print(f"      Score: {best['_score']:.1f} (weighted: Sharpe×30, WinRate×20, PF×10, DD×-50)")

    print()


def export_csv(reports: list, path: str):
    """Export comparison to CSV."""
    import csv
    if not reports:
        return

    keys = list(reports[0].keys())
    keys = [k for k in keys if not k.startswith("_")]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in reports:
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            # Format percentages
            for k in ("total_return", "annualized_return", "max_drawdown", "volatility", "win_rate"):
                if k in row and isinstance(row[k], float):
                    row[k] = f"{row[k]:.4f}"
            w.writerow(row)

    print(f"  CSV saved: {path}")


def export_html(reports: list, path: str):
    """Export comparison as a standalone HTML file."""
    rows_html = ""
    metrics = [
        ("Strategy", "strategy", ""),
        ("Symbol", "symbol", ""),
        ("Interval", "interval", ""),
        ("Period", "period", ""),
        ("Total Return", "total_return", "pct"),
        ("Sharpe Ratio", "sharpe_ratio", "ratio"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Win Rate", "win_rate", "pct"),
        ("Total Trades", "total_trades", "int"),
        ("Profit Factor", "profit_factor", "ratio"),
        ("Avg Win", "avg_win", "pct2"),
        ("Avg Loss", "avg_loss", "pct2"),
        ("Expectancy", "expectancy", "pct2"),
        ("Best Trade", "best_trade", "pct2"),
        ("Worst Trade", "worst_trade", "pct2"),
        ("Commission", "commission", "dollar"),
    ]

    for label, key, kind in metrics:
        row = f"<tr><td style='color:#9ca3af;font-weight:600'>{label}</td>"
        for r in reports:
            val = fmt(r[key], kind)
            color = "#d1d5db"
            if kind == "pct" and isinstance(r[key], (int, float)):
                if key == "max_drawdown":
                    color = "#dc2626"
                else:
                    color = "#16a34a" if r[key] > 0 else "#dc2626"
            row += f"<td style='color:{color};text-align:right'>{val}</td>"
        row += "</tr>"
        rows_html += row

    headers = "".join(f"<th style='color:#06b6d4;text-align:right'>{r['strategy']}<br><span style=\"color:#4b5563;font-size:10px\">{r['symbol']} {r['interval']}</span></th>" for r in reports)

    html = f"""<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
body{{background:#080c14;color:#d1d5db;font-family:'JetBrains Mono',monospace;padding:24px}}
h1{{color:#06b6d4;font-size:18px;letter-spacing:3px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;font-size:11px}}
th,td{{padding:8px 12px;border-bottom:1px solid #1f2937}}
th{{text-align:left;font-size:12px}}
tr:hover{{background:#111827}}
.ts{{color:#4b5563;font-size:10px;margin-top:24px}}
</style></head><body>
<h1>◆ QUANTBOT — STRATEGY COMPARISON</h1>
<table><thead><tr><th>Metric</th>{headers}</tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="ts">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""

    with open(path, "w") as f:
        f.write(html)
    print(f"  HTML saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare backtest strategies")
    parser.add_argument("reports", nargs="+", help="Backtest report JSON files")
    parser.add_argument("--csv", default=None, help="Export to CSV file")
    parser.add_argument("--html", default=None, help="Export to HTML file")
    args = parser.parse_args()

    reports = []
    for path in args.reports:
        try:
            r = load_report(path)
            reports.append(r)
            print(f"  ✓ Loaded: {path} ({r['strategy']} on {r['symbol']})")
        except Exception as e:
            print(f"  ✗ Failed: {path} — {e}")

    if not reports:
        print("No valid reports found.")
        sys.exit(1)

    print_comparison(reports)

    if args.csv:
        export_csv(reports, args.csv)

    if args.html:
        export_html(reports, args.html)


if __name__ == "__main__":
    main()