"""Backtest & Validation performance dashboards — dark-themed PNG figures.

Two public functions:

  plot_backtest_dashboard(source, save_path)
      source = path to backtest_report.json  OR  engine._results dict
      Produces a 6-panel figure:
        1. Equity curve + trade markers + green/red shading
        2. Drawdown fill
        3. Per-trade net P&L bars with cumulative overlay
        4. Trade return distribution histogram
        5. Monthly net P&L bars
        6. Performance metrics scorecard

  plot_validation_dashboard(validation_results, save_path)
      validation_results = dict returned by run_validation()
      Produces a 6-panel figure:
        1. Walk-forward IS vs OOS grouped bar chart
        2. Out-of-sample cumulative equity line
        3. Standard bootstrap MC return distribution
        4. Markov-chain MC fan chart (regime-switching paths)
        5. Regime analysis: transition matrix heatmap + stats
        6. Validation verdict scorecard

Usage:
    from backtest.dashboard import plot_backtest_dashboard, plot_validation_dashboard
    plot_backtest_dashboard("backtest_report.json", "backtest_dashboard.png")
    plot_validation_dashboard(run_validation(...), "validation_dashboard.png")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Dark colour palette (GitHub-dark inspired) ─────────────────────────────
_BG    = "#0d1117"   # figure background
_PANEL = "#161b22"   # axes face
_GRID  = "#30363d"   # gridlines / spine
_GREEN = "#3fb950"   # win / profit / positive
_RED   = "#f85149"   # loss / drawdown / negative
_BLUE  = "#58a6ff"   # primary line (equity, in-sample)
_ORG   = "#d29922"   # warning / secondary accent
_PURP  = "#a371f7"   # alternate accent
_TEXT  = "#c9d1d9"   # primary text
_MUTED = "#8b949e"   # secondary / label text


# ── Shared helpers ─────────────────────────────────────────────────────────

def _ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent dark styling to an axes object."""
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=9, fontweight="bold", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color=_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color=_MUTED)


def _load(source: Union[str, Dict]) -> Dict:
    """Load a backtest report from a JSON file path or pass-through a dict."""
    if isinstance(source, (str, Path)):
        with open(source) as f:
            return json.load(f)
    return source


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_backtest_dashboard(
    source: Union[str, Dict],
    save_path: str = "./backtest_dashboard.png",
) -> str:
    """Generate a comprehensive 6-panel backtest performance dashboard.

    Args:
        source:    Path to backtest_report.json  OR  dict from BacktestEngine.
        save_path: Output PNG path.

    Returns:
        save_path
    """
    report  = _load(source)
    meta    = report.get("meta", {})
    metrics = report.get("metrics", {})
    eq_raw  = report.get("equity_curve", [])
    tr_raw  = report.get("paired_trades", [])

    # ── Parse equity curve ────────────────────────────────────────────────
    if eq_raw:
        ec = pd.DataFrame(eq_raw)
        # Handle both "timestamp" (JSON export) and "ts" (dict)
        ts_col = "timestamp" if "timestamp" in ec.columns else "ts"
        if ts_col in ec.columns:
            ec["ts"] = pd.to_datetime(ec[ts_col], errors="coerce", utc=True)
            ec = ec.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        else:
            ec = pd.DataFrame(columns=["ts", "value", "drawdown"])
    else:
        ec = pd.DataFrame(columns=["ts", "value", "drawdown"])

    # ── Parse paired trades ───────────────────────────────────────────────
    if isinstance(tr_raw, list) and tr_raw:
        tr = pd.DataFrame(tr_raw)
    elif isinstance(tr_raw, pd.DataFrame):
        tr = tr_raw.copy()
    else:
        tr = pd.DataFrame()

    if not tr.empty and "entry_time" in tr.columns:
        tr["edt"] = pd.to_datetime(tr["entry_time"], errors="coerce", utc=True)
        tr["xdt"] = pd.to_datetime(tr["exit_time"],  errors="coerce", utc=True)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15), facecolor=_BG)
    gs  = gridspec.GridSpec(
        4, 3,
        height_ratios=[2.8, 1.0, 1.7, 1.7],
        hspace=0.44, wspace=0.30,
        left=0.05, right=0.97, top=0.93, bottom=0.04,
    )
    ax_eq  = fig.add_subplot(gs[0, :])     # equity — full width
    ax_dd  = fig.add_subplot(gs[1, :])     # drawdown — full width
    ax_tr  = fig.add_subplot(gs[2, :2])    # per-trade P&L bars
    ax_hi  = fig.add_subplot(gs[2, 2])     # return histogram
    ax_mo  = fig.add_subplot(gs[3, :2])    # monthly P&L
    ax_sc  = fig.add_subplot(gs[3, 2])     # metrics scorecard

    # ── Title ─────────────────────────────────────────────────────────────
    parts: List[str] = []
    if meta.get("strategies"):  parts.append(" + ".join(meta["strategies"]).upper())
    if meta.get("symbols"):     parts.append(", ".join(meta["symbols"]))
    if meta.get("interval"):    parts.append(meta["interval"])
    if meta.get("start") and meta.get("end"):
        parts.append(f"{meta['start']} → {meta['end']}")
    fig.suptitle(
        "  ·  ".join(parts) if parts else "Backtest Performance Dashboard",
        fontsize=13, fontweight="bold", color=_TEXT, y=0.97,
    )

    # ── Panel 1: Equity curve ─────────────────────────────────────────────
    _ax(ax_eq, "Equity Curve", ylabel="Portfolio Value ($)")
    if not ec.empty and "value" in ec.columns:
        init  = metrics.get("initial_capital", float(ec["value"].iloc[0]))
        ts    = ec["ts"].values
        vals  = ec["value"].values
        ax_eq.plot(ts, vals, lw=1.8, color=_BLUE, label="Strategy", zorder=3)
        ax_eq.axhline(init, color=_MUTED, lw=0.7, ls="--", alpha=0.6,
                      label=f"Initial ${init:,.0f}")
        ax_eq.fill_between(ts, init, vals,
                           where=vals >= init, alpha=0.15, color=_GREEN, interpolate=True)
        ax_eq.fill_between(ts, init, vals,
                           where=vals < init,  alpha=0.15, color=_RED,   interpolate=True)
        if not tr.empty and "edt" in tr.columns:
            for _, t in tr.iterrows():
                c = _GREEN if t.get("win") else _RED
                try:
                    ax_eq.axvline(t["edt"], color=c, alpha=0.22, lw=0.7)
                    ax_eq.axvline(t["xdt"], color=c, alpha=0.10, lw=0.7, ls="--")
                except Exception:
                    pass
    ax_eq.legend(fontsize=8, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    _ax(ax_dd, "Drawdown from Peak", ylabel="Drawdown (%)")
    if not ec.empty and "drawdown" in ec.columns:
        dd = ec["drawdown"].values * 100.0
        ax_dd.fill_between(ec["ts"], -dd, 0.0, color=_RED, alpha=0.50, lw=0)
        ax_dd.plot(ec["ts"], -dd, color=_RED, lw=0.8, alpha=0.75)
        max_dd = metrics.get("max_drawdown", float(dd.max() / 100.0))
        ax_dd.axhline(-max_dd * 100, color=_ORG, lw=0.9, ls=":",
                      label=f"Max DD {max_dd:.1%}")
        ax_dd.set_ylim(bottom=min(-dd.max() * 1.25, -0.5), top=0.5)
        ax_dd.legend(fontsize=8, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Panel 3: Per-trade P&L bars + cumulative overlay ─────────────────
    _ax(ax_tr, "Per-Trade Net P&L", xlabel="Trade #", ylabel="Net P&L ($)")
    if not tr.empty and "net_pnl" in tr.columns:
        pnls   = tr["net_pnl"].values
        wins   = tr["win"].values if "win" in tr.columns else (pnls > 0)
        colors = [_GREEN if w else _RED for w in wins]
        ax_tr.bar(range(len(pnls)), pnls, color=colors, alpha=0.85, width=0.85)
        ax_tr.axhline(0, color=_GRID, lw=0.9)
        ax2 = ax_tr.twinx()
        ax2.plot(range(len(pnls)), np.cumsum(pnls),
                 color=_BLUE, lw=1.6, alpha=0.85, label="Cumulative")
        ax2.set_facecolor("none")
        ax2.tick_params(colors=_MUTED, labelsize=7)
        ax2.set_ylabel("Cumulative P&L ($)", color=_MUTED, fontsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor(_GRID)
        ax2.legend(fontsize=7.5, loc="upper left",
                   facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)
    else:
        ax_tr.text(0.5, 0.5, "No trades recorded",
                   ha="center", va="center", color=_MUTED, fontsize=11,
                   transform=ax_tr.transAxes)

    # ── Panel 4: Trade return distribution ───────────────────────────────
    _ax(ax_hi, "Trade Return Distribution", xlabel="Return (%)", ylabel="Count")
    if not tr.empty and "pnl_pct" in tr.columns:
        rets = tr["pnl_pct"].values * 100.0
        bins = min(30, max(8, len(rets) // 3))
        pos, neg = rets[rets > 0], rets[rets <= 0]
        if len(pos):
            ax_hi.hist(pos, bins=bins, color=_GREEN, alpha=0.70,
                       label=f"Wins ({len(pos)})")
        if len(neg):
            ax_hi.hist(neg, bins=bins, color=_RED,   alpha=0.70,
                       label=f"Losses ({len(neg)})")
        ax_hi.axvline(0,          color=_GRID, lw=1.0)
        ax_hi.axvline(rets.mean(), color=_ORG, lw=1.5, ls="--",
                      label=f"Mean {rets.mean():+.2f}%")
        ax_hi.legend(fontsize=7.5, facecolor=_PANEL,
                     edgecolor=_GRID, labelcolor=_TEXT)
    else:
        ax_hi.text(0.5, 0.5, "No trades", ha="center", va="center",
                   color=_MUTED, fontsize=11, transform=ax_hi.transAxes)

    # ── Panel 5: Monthly P&L ──────────────────────────────────────────────
    _ax(ax_mo, "Monthly Net P&L", xlabel="Month", ylabel="Net P&L ($)")
    if not tr.empty and "net_pnl" in tr.columns and "edt" in tr.columns:
        tmp = tr.dropna(subset=["edt"]).copy()
        if not tmp.empty:
            tmp["ym"]  = tmp["edt"].dt.tz_convert(None).dt.to_period("M")
            monthly    = tmp.groupby("ym")["net_pnl"].sum().reset_index()
            labels     = [str(p) for p in monthly["ym"]]
            vals       = monthly["net_pnl"].values
            mc         = [_GREEN if v >= 0 else _RED for v in vals]
            x          = np.arange(len(vals))
            ax_mo.bar(x, vals, color=mc, alpha=0.85, width=0.72)
            ax_mo.axhline(0, color=_GRID, lw=0.8)
            ax_mo.set_xticks(x)
            ax_mo.set_xticklabels(labels, rotation=40, ha="right",
                                  fontsize=7.5, color=_MUTED)
            for xi, vi in zip(x, vals):
                offset = abs(vi) * 0.07 + 0.5
                ax_mo.text(xi, vi + offset * (1 if vi >= 0 else -1),
                           f"${vi:+.0f}", ha="center",
                           va="bottom" if vi >= 0 else "top",
                           fontsize=6.5,
                           color=_GREEN if vi >= 0 else _RED)
        else:
            ax_mo.text(0.5, 0.5, "No trade timestamps", ha="center",
                       va="center", color=_MUTED, fontsize=11,
                       transform=ax_mo.transAxes)
    else:
        ax_mo.text(0.5, 0.5, "No trades", ha="center", va="center",
                   color=_MUTED, fontsize=11, transform=ax_mo.transAxes)

    # ── Panel 6: Metrics scorecard ────────────────────────────────────────
    ax_sc.set_facecolor(_PANEL)
    for spine in ax_sc.spines.values():
        spine.set_edgecolor(_GRID)
    ax_sc.set_xticks([])
    ax_sc.set_yticks([])
    _ax(ax_sc, "Performance Metrics")

    init_c = metrics.get("initial_capital", 10_000)
    final  = metrics.get("final_value", init_c)
    n_wins = metrics.get("n_wins", 0)
    n_loss = metrics.get("n_losses", 0)
    n_tr   = metrics.get("total_round_trips", n_wins + n_loss)

    rows = [
        ("Total Return",  f"{metrics.get('total_return', 0):+.2%}",  metrics.get("total_return", 0) >= 0),
        ("Final Value",   f"${final:,.0f}",                          final >= init_c),
        ("Sharpe Ratio",  f"{metrics.get('sharpe_ratio', 0):.3f}",   metrics.get("sharpe_ratio", 0) >= 0.5),
        ("Max Drawdown",  f"{metrics.get('max_drawdown', 0):.2%}",   metrics.get("max_drawdown", 0) < 0.15),
        ("Win Rate",      f"{metrics.get('win_rate', 0):.1%}",       metrics.get("win_rate", 0) >= 0.50),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}",  metrics.get("profit_factor", 0) >= 1.0),
        ("Expectancy",    f"${metrics.get('expectancy', 0):.2f}",    metrics.get("expectancy", 0) >= 0),
        ("Total Trades",  f"{n_tr}",                                  True),
        ("Wins / Losses", f"{n_wins} / {n_loss}",                    True),
    ]

    y = 0.93
    dy = 0.098
    for label, value, is_good in rows:
        vc = _GREEN if is_good else _RED
        ax_sc.text(0.07, y, label, transform=ax_sc.transAxes,
                   fontsize=8.5, color=_MUTED, va="top")
        ax_sc.text(0.93, y, value, transform=ax_sc.transAxes,
                   fontsize=8.5, color=vc, va="top", ha="right", fontweight="bold")
        y -= dy
        sep_y = y + dy * 0.15
        ax_sc.plot([0.04, 0.96], [sep_y, sep_y], color=_GRID, lw=0.3,
                   transform=ax_sc.transAxes)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"\n  Backtest dashboard saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_validation_dashboard(
    results: Dict,
    save_path: str = "./validation_dashboard.png",
) -> str:
    """Generate a 6-panel validation dashboard including Markov-chain MC.

    Args:
        results:   Dict returned by run_validation()  (must contain "markov_mc").
        save_path: Output PNG path.

    Returns:
        save_path
    """
    wf      = results.get("walk_forward", [])
    oos     = results.get("out_of_sample", {})
    mc_std  = results.get("monte_carlo", {})
    mc_mkv  = results.get("markov_mc", {})
    score   = results.get("score", 0)
    checks  = results.get("checks", [])

    fig = plt.figure(figsize=(22, 16), facecolor=_BG)
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.6, 2.2, 1.6],
        hspace=0.42, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.05,
    )
    ax_wf  = fig.add_subplot(gs[0, 0])   # walk-forward grouped bars
    ax_oos = fig.add_subplot(gs[0, 1])   # OOS cumulative equity
    ax_mcs = fig.add_subplot(gs[1, 0])   # standard MC histogram
    ax_mkv = fig.add_subplot(gs[1, 1])   # Markov MC fan chart
    ax_reg = fig.add_subplot(gs[2, 0])   # regime analysis
    ax_vrd = fig.add_subplot(gs[2, 1])   # verdict scorecard

    fig.suptitle(
        "Strategy Validation Report  ·  Markov-Chain Monte Carlo",
        fontsize=13, fontweight="bold", color=_TEXT, y=0.97,
    )

    # ── Panel 1: Walk-Forward IS vs OOS bars ─────────────────────────────
    if wf:
        labels   = [w["label"].replace("Window ", "W") for w in wf]
        is_rets  = [w["train"]["metrics"].get("total_return", 0) * 100 for w in wf]
        oos_rets = [w["test"]["metrics"].get("total_return",  0) * 100 for w in wf]
        is_sh    = [w["train"]["metrics"].get("sharpe_ratio", 0) for w in wf]
        oos_sh   = [w["test"]["metrics"].get("sharpe_ratio",  0) for w in wf]
        x  = np.arange(len(labels))
        bw = 0.35
        n_pos = sum(1 for r in oos_rets if r > 0)
        _ax(ax_wf,
            f"Walk-Forward  —  {n_pos}/{len(oos_rets)} OOS windows profitable",
            xlabel="Window", ylabel="Return (%)")
        ax_wf.bar(x - bw/2, is_rets,  bw, color=_BLUE,  alpha=0.8, label="In-Sample")
        ax_wf.bar(x + bw/2, oos_rets, bw,
                  color=[_GREEN if r >= 0 else _RED for r in oos_rets],
                  alpha=0.85, label="Out-of-Sample")
        ax_wf.axhline(0, color=_GRID, lw=0.9)
        ax_wf.set_xticks(x)
        ax_wf.set_xticklabels(labels, fontsize=8, color=_MUTED)
        ax_wf.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        # Annotate Sharpe on OOS bars
        for xi, r, sh in zip(x + bw/2, oos_rets, oos_sh):
            offset = 0.4 if r >= 0 else -0.4
            ax_wf.text(xi, r + offset, f"S={sh:.2f}",
                       ha="center", va="bottom" if r >= 0 else "top",
                       fontsize=6.5, color=_TEXT)
        ax_wf.legend(fontsize=8, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)
    else:
        _ax(ax_wf, "Walk-Forward")
        ax_wf.text(0.5, 0.5, "No WF results", ha="center", va="center",
                   color=_MUTED, transform=ax_wf.transAxes)

    # ── Panel 2: OOS cumulative equity ────────────────────────────────────
    _ax(ax_oos, "Out-of-Sample Cumulative P&L",
        xlabel="Trade #", ylabel="Cumulative P&L ($)")
    oos_paired = oos.get("paired_trades")
    oos_m      = oos.get("metrics", {})
    _oos_has_trades = False
    if oos_paired is not None:
        oos_df = oos_paired if isinstance(oos_paired, pd.DataFrame) \
                 else pd.DataFrame(oos_paired)
        if not oos_df.empty and "net_pnl" in oos_df.columns:
            _oos_has_trades = True
            pnls  = oos_df["net_pnl"].values
            cum   = np.cumsum(pnls)
            c     = _GREEN if cum[-1] >= 0 else _RED
            x_ax  = np.arange(len(cum))
            ax_oos.plot(x_ax, cum, color=c, lw=2.0, zorder=3)
            ax_oos.fill_between(x_ax, 0, cum, alpha=0.20, color=c)
            ax_oos.axhline(0, color=_MUTED, lw=0.8, ls="--")
            # Per-trade dots
            ax_oos.scatter(x_ax,
                           [p for p in pnls],
                           c=[_GREEN if p >= 0 else _RED for p in pnls],
                           s=18, zorder=4, alpha=0.7)
    if not _oos_has_trades:
        oos_ret = oos_m.get("total_return", 0)
        oos_sh  = oos_m.get("sharpe_ratio", 0)
        ax_oos.text(0.5, 0.6, f"OOS Return: {oos_ret:+.2%}",
                    ha="center", va="center", color=_TEXT, fontsize=14,
                    fontweight="bold", transform=ax_oos.transAxes)
        ax_oos.text(0.5, 0.42, f"Sharpe: {oos_sh:.3f}",
                    ha="center", va="center", color=_MUTED, fontsize=11,
                    transform=ax_oos.transAxes)
    else:
        ax_oos.set_title(
            f"OOS: {oos_m.get('total_return',0):+.2%} return  ·  "
            f"Sharpe {oos_m.get('sharpe_ratio',0):.2f}  ·  "
            f"{oos.get('n_trades', len(pnls))} trades",
            color=_TEXT, fontsize=9, fontweight="bold", pad=5,
        )

    # ── Panel 3: Standard MC return distribution ──────────────────────────
    _ax(ax_mcs,
        f"Standard Bootstrap MC  —  {mc_std.get('n_simulations', 0):,} paths",
        xlabel="Final Return (%)", ylabel="Frequency")
    if mc_std and len(mc_std.get("all_returns", [])) > 0:
        rets = np.asarray(mc_std["all_returns"]) * 100.0
        bins = min(60, max(20, len(rets) // 20))
        ax_mcs.hist(rets[rets >= 0], bins=bins, color=_GREEN, alpha=0.65,
                    label="Profitable")
        ax_mcs.hist(rets[rets <  0], bins=bins, color=_RED,   alpha=0.65,
                    label="Loss")
        ax_mcs.axvline(0, color=_GRID, lw=1.2)
        ax_mcs.axvline(mc_std["median_return"] * 100, color=_BLUE, lw=1.8, ls="--",
                       label=f"Median {mc_std['median_return']:+.2%}")
        ax_mcs.axvline(mc_std["p5_return"] * 100, color=_ORG, lw=1.2, ls=":",
                       label=f"P5 {mc_std['p5_return']:+.2%}")
        ax_mcs.text(0.97, 0.97,
                    f"P(profit): {mc_std.get('prob_profitable', 0):.0%}\n"
                    f"P95 DD:    {mc_std.get('p95_drawdown', 0):.1%}",
                    ha="right", va="top", transform=ax_mcs.transAxes,
                    fontsize=8, color=_TEXT,
                    bbox=dict(facecolor=_PANEL, edgecolor=_GRID, alpha=0.9))
        ax_mcs.legend(fontsize=7.5, facecolor=_PANEL,
                      edgecolor=_GRID, labelcolor=_TEXT)
    else:
        ax_mcs.text(0.5, 0.5, "No MC data", ha="center", va="center",
                    color=_MUTED, transform=ax_mcs.transAxes)

    # ── Panel 4: Markov-Chain MC fan chart ────────────────────────────────
    _ax(ax_mkv,
        "Markov-Chain MC  —  Regime-Switching Paths",
        xlabel="Trade #", ylabel="Equity ($)")
    paths = mc_mkv.get("path_equities", []) if mc_mkv else []
    if paths:
        init_c   = mc_mkv.get("initial_capital", 10_000)
        max_len  = max(len(p) for p in paths)
        # Pad shorter paths (equity hit 0 early)
        mat = np.array([
            np.array(p + [p[-1]] * (max_len - len(p)), dtype=float)
            for p in paths
        ])
        x   = np.arange(max_len)
        p5  = np.percentile(mat, 5,  axis=0)
        med = np.percentile(mat, 50, axis=0)
        p95 = np.percentile(mat, 95, axis=0)

        # 50 thin background paths
        for row in mat[:50]:
            ax_mkv.plot(x, row, lw=0.35, alpha=0.10, color=_BLUE)

        ax_mkv.fill_between(x, p5, p95, alpha=0.22, color=_BLUE,
                            label="5th–95th pct")
        ax_mkv.plot(x, med, lw=2.2, color=_GREEN,    label="Median path",  zorder=4)
        ax_mkv.plot(x, p5,  lw=1.0, color=_ORG, ls="--", alpha=0.85, zorder=3)
        ax_mkv.plot(x, p95, lw=1.0, color=_ORG, ls="--", alpha=0.85, zorder=3)
        ax_mkv.axhline(init_c, color=_MUTED, lw=0.7, ls="--", alpha=0.6)

        # Regime annotation box
        reg_names = mc_mkv.get("regime_names", ["Calm", "Stressed"])
        cur_reg   = mc_mkv.get("current_regime", 0)
        mkv_p95   = mc_mkv.get("p95_drawdown", 0)
        std_p95   = mc_std.get("p95_drawdown", 0) if mc_std else 0
        regime_color = _GREEN if cur_reg == 0 else _RED
        ax_mkv.text(
            0.97, 0.97,
            f"Current regime: {reg_names[cur_reg]}\n"
            f"Markov  P95 DD: {mkv_p95:.1%}\n"
            f"Standard P95 DD: {std_p95:.1%}",
            ha="right", va="top", transform=ax_mkv.transAxes,
            fontsize=8, color=regime_color,
            bbox=dict(facecolor=_PANEL, edgecolor=_GRID, alpha=0.92),
        )
        if T_val := mc_mkv.get("transition_matrix"):
            T = np.array(T_val)
            if T[1][1] > 0.70:
                ax_mkv.text(0.03, 0.04,
                            f"⚠ Stressed regime persists  T={T[1][1]:.2f}",
                            ha="left", va="bottom", transform=ax_mkv.transAxes,
                            fontsize=7.5, color=_RED)
        ax_mkv.legend(fontsize=7.5, facecolor=_PANEL,
                      edgecolor=_GRID, labelcolor=_TEXT)
    elif mc_mkv and len(mc_mkv.get("all_returns", [])) > 0:
        # Fall back to histogram if paths weren't stored
        _ax(ax_mkv, "Markov-Chain MC — Return Distribution",
            xlabel="Final Return (%)", ylabel="Frequency")
        rets = np.asarray(mc_mkv["all_returns"]) * 100.0
        bins = min(60, max(20, len(rets) // 20))
        ax_mkv.hist(rets, bins=bins, color=_PURP, alpha=0.70)
        ax_mkv.axvline(mc_mkv.get("median_return", 0) * 100,
                       color=_BLUE, lw=1.8, ls="--")
    else:
        ax_mkv.text(0.5, 0.5,
                    "Markov MC unavailable\n(need ≥ 6 trades)",
                    ha="center", va="center", color=_MUTED, fontsize=11,
                    transform=ax_mkv.transAxes)

    # ── Panel 5: Regime analysis ──────────────────────────────────────────
    ax_reg.set_facecolor(_PANEL)
    for spine in ax_reg.spines.values():
        spine.set_edgecolor(_GRID)
    ax_reg.set_xticks([])
    ax_reg.set_yticks([])
    ax_reg.set_title("Regime Analysis", color=_TEXT, fontsize=9,
                     fontweight="bold", pad=5)

    if mc_mkv and mc_mkv.get("transition_matrix") is not None:
        T         = np.array(mc_mkv["transition_matrix"])
        rnames    = mc_mkv.get("regime_names", ["Calm", "Stressed"])
        rlabs     = mc_mkv.get("regime_labels", [])
        cur       = mc_mkv.get("current_regime", 0)
        rmean     = mc_mkv.get("regime_mean",  [0.0, 0.0])
        rstd      = mc_mkv.get("regime_std",   [0.01, 0.01])
        n_c       = sum(1 for r in rlabs if r == 0)
        n_s       = sum(1 for r in rlabs if r == 1)
        total_r   = max(n_c + n_s, 1)

        # Left: 2×2 transition matrix heatmap
        ax_heat = ax_reg.inset_axes([0.04, 0.12, 0.44, 0.78])
        ax_heat.set_facecolor(_PANEL)
        im = ax_heat.imshow(T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax_heat.set_xticks([0, 1])
        ax_heat.set_yticks([0, 1])
        ax_heat.set_xticklabels([f"→ {r}" for r in rnames],
                                fontsize=8.5, color=_MUTED)
        ax_heat.set_yticklabels([f"{r} →" for r in rnames],
                                fontsize=8.5, color=_MUTED)
        for spine in ax_heat.spines.values():
            spine.set_edgecolor(_GRID)
        for i in range(2):
            for j in range(2):
                txt_c = "black" if T[i, j] > 0.55 else _TEXT
                ax_heat.text(j, i, f"{T[i, j]:.2f}",
                             ha="center", va="center",
                             fontsize=12, color=txt_c, fontweight="bold")
        ax_heat.set_title("Transition Matrix", color=_TEXT, fontsize=8, pad=4)

        # Right: regime stats text
        mkv_p95 = mc_mkv.get("p95_drawdown", 0)
        std_p95 = mc_std.get("p95_drawdown", 0) if mc_std else 0
        lines = [
            (_ORG,   "Regime Distribution", ""),
            (_MUTED, f"  Calm  ({n_c} trades)",
             f"{n_c/total_r:.0%}"),
            (_MUTED, f"  Stressed  ({n_s} trades)",
             f"{n_s/total_r:.0%}"),
            ("",     "", ""),
            (_ORG,   "Per-Regime Expected Return", ""),
            (_MUTED, "  Calm  (avg/trade)",     f"{rmean[0]:+.4f}"),
            (_MUTED, "  Stressed  (avg/trade)", f"{rmean[1]:+.4f}"),
            ("",     "", ""),
            (_ORG,   "Risk Comparison", ""),
            (_MUTED, "  Markov P95 DD",   f"{mkv_p95:.2%}"),
            (_MUTED, "  Standard P95 DD", f"{std_p95:.2%}"),
            ("",     "", ""),
            (regime_color, f"Current Regime",
             f"{rnames[cur]}"),
        ]
        y  = 0.95
        dy = 0.070
        for lc, label, value in lines:
            if not label:
                y -= dy * 0.35
                continue
            if not value:   # section header
                ax_reg.text(0.52, y, label, transform=ax_reg.transAxes,
                            fontsize=8, color=lc, va="top", fontweight="bold")
            else:
                ax_reg.text(0.52, y, label, transform=ax_reg.transAxes,
                            fontsize=7.5, color=_MUTED, va="top")
                vc = _GREEN if "Calm" in label else (_RED if "Stressed" in label else _TEXT)
                ax_reg.text(0.97, y, value, transform=ax_reg.transAxes,
                            fontsize=7.5, color=vc, va="top",
                            ha="right", fontweight="bold")
            y -= dy
    else:
        ax_reg.text(0.5, 0.5, "Regime data not available\n(need ≥ 6 trades)",
                    ha="center", va="center", color=_MUTED, fontsize=10,
                    transform=ax_reg.transAxes)

    # ── Panel 6: Verdict scorecard ────────────────────────────────────────
    ax_vrd.set_facecolor(_PANEL)
    for spine in ax_vrd.spines.values():
        spine.set_edgecolor(_GRID)
    ax_vrd.set_xticks([])
    ax_vrd.set_yticks([])
    ax_vrd.set_title("Validation Verdict", color=_TEXT, fontsize=9,
                     fontweight="bold", pad=5)

    score_color   = _GREEN if score >= 4 else (_ORG if score >= 3 else _RED)
    verdict_label = {5: "PASS", 4: "PASS", 3: "MARGINAL",
                     2: "WEAK", 1: "FAIL", 0: "FAIL"}.get(score, "—")

    ax_vrd.text(0.5, 0.91, f"{score} / 5",
                ha="center", va="top", transform=ax_vrd.transAxes,
                fontsize=22, fontweight="bold", color=score_color)
    ax_vrd.text(0.5, 0.80, verdict_label,
                ha="center", va="top", transform=ax_vrd.transAxes,
                fontsize=13, fontweight="bold", color=score_color)

    ax_vrd.plot([0.04, 0.96], [0.74, 0.74], color=_GRID, lw=0.6,
                transform=ax_vrd.transAxes)

    y  = 0.70
    dy = 0.115
    for check in checks:
        is_pass = check.startswith("✓")
        is_warn = check.startswith("~")
        c = _GREEN if is_pass else (_ORG if is_warn else _RED)
        short = check[:58] + ("…" if len(check) > 58 else "")
        ax_vrd.text(0.05, y, short, transform=ax_vrd.transAxes,
                    fontsize=7.5, color=c, va="top")
        y -= dy
        if y < 0.03:
            break

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"\n  Validation dashboard saved: {save_path}")
    return save_path
