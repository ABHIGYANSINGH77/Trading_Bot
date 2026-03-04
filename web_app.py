"""QuantBot Web Dashboard — Plotly Dash interactive portfolio viewer.

Run:
    python web_app.py
    # then open http://localhost:8050

Tabs:
    Overview   — headline metrics, equity curve, strategy breakdown
    Backtest   — full interactive backtest analysis (load any report JSON)
    Validation — walk-forward, Monte Carlo, Markov-chain, Sharpe CI, verdict

Auto-refreshes every 30 seconds to track live paper trading output.
Requires: dash, dash-bootstrap-components  (both in requirements.txt)
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colours matching the matplotlib dashboard palette ──────────────────────────
BG          = "#0d1117"
PANEL_BG    = "#161b22"
BORDER      = "#30363d"
TEXT        = "#e6edf3"
TEXT_MUTED  = "#8b949e"
GREEN       = "#3fb950"
RED         = "#f85149"
ORANGE      = "#f0883e"
BLUE        = "#58a6ff"
PURPLE      = "#d2a8ff"
YELLOW      = "#e3b341"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="monospace"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=11)),
)

ROOT = Path(__file__).parent

# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="QuantBot Dashboard",
    suppress_callback_exceptions=True,
)

# ── Helper: load JSON safely ───────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _file_mtime(path: str) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


def _list_backtest_files() -> list:
    files = sorted(
        ROOT.glob("backtest*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [str(f) for f in files]


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _metric_card(title: str, value: str, color: str = TEXT) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H4(value, style={"color": color, "fontFamily": "monospace", "fontWeight": "700"}),
        ]),
        style={"background": PANEL_BG, "border": f"1px solid {BORDER}", "borderRadius": "8px"},
        className="text-center",
    )


def _empty_fig(msg: str = "No data loaded") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       font=dict(color=TEXT_MUTED, size=14), showarrow=False)
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    return fig


# ── Layout ─────────────────────────────────────────────────────────────────────

def _make_layout() -> html.Div:
    return html.Div([
        # Data stores (hold loaded JSON + file mtimes)
        dcc.Store(id="bt-store"),
        dcc.Store(id="val-store"),
        dcc.Store(id="bt-mtime", data=0.0),
        dcc.Store(id="val-mtime", data=0.0),

        # Refresh timer — every 30s
        dcc.Interval(id="refresh-timer", interval=30_000, n_intervals=0),

        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("QuantBot Dashboard", style={"fontFamily": "monospace",
                                                              "fontWeight": "700",
                                                              "fontSize": "1.2rem"}),
                html.Span(id="refresh-status",
                          style={"color": TEXT_MUTED, "fontSize": "0.8rem", "fontFamily": "monospace"}),
            ], fluid=True),
            color="dark", dark=True, className="mb-3",
        ),

        dbc.Container([
            dbc.Tabs([
                dbc.Tab(label="Overview",   tab_id="overview"),
                dbc.Tab(label="Backtest",   tab_id="backtest"),
                dbc.Tab(label="Validation", tab_id="validation"),
            ], id="tabs", active_tab="overview", className="mb-3"),

            html.Div(id="tab-content"),
        ], fluid=True),
    ], style={"background": BG, "minHeight": "100vh"})


app.layout = _make_layout()


# ── Refresh — load data when files change ──────────────────────────────────────

@app.callback(
    Output("bt-store", "data"),
    Output("val-store", "data"),
    Output("bt-mtime", "data"),
    Output("val-mtime", "data"),
    Output("refresh-status", "children"),
    Input("refresh-timer", "n_intervals"),
    State("bt-mtime", "data"),
    State("val-mtime", "data"),
)
def _refresh_data(n_intervals, prev_bt_mtime, prev_val_mtime):
    """Reload JSON files only when they change on disk."""
    bt_files = _list_backtest_files()
    bt_path = bt_files[0] if bt_files else str(ROOT / "backtest_report.json")
    val_path = str(ROOT / "validation_results.json")

    new_bt_mtime = _file_mtime(bt_path)
    new_val_mtime = _file_mtime(val_path)

    bt_data = _load_json(bt_path) if new_bt_mtime != prev_bt_mtime or n_intervals == 0 else dash.no_update
    val_data = _load_json(val_path) if new_val_mtime != prev_val_mtime or n_intervals == 0 else dash.no_update

    ts = datetime.now().strftime("%H:%M:%S")
    status = f"Last refreshed: {ts}"
    return bt_data, val_data, new_bt_mtime, new_val_mtime, status


# ── Tab routing ────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("bt-store", "data"),
    State("val-store", "data"),
)
def _render_tab(active_tab, bt_data, val_data):
    bt_data = bt_data or {}
    val_data = val_data or {}
    if active_tab == "overview":
        return _tab_overview(bt_data)
    if active_tab == "backtest":
        return _tab_backtest(bt_data)
    if active_tab == "validation":
        return _tab_validation(val_data)
    return html.Div("Unknown tab")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _tab_overview(bt: dict) -> html.Div:
    metrics = bt.get("metrics", {})
    meta    = bt.get("meta", {})

    total_ret  = metrics.get("total_return", 0)
    sharpe     = metrics.get("sharpe_ratio", 0)
    max_dd     = metrics.get("max_drawdown", 0)
    win_rate   = metrics.get("win_rate", 0)
    n_trades   = metrics.get("total_trades", 0)

    ret_color   = GREEN if total_ret >= 0 else RED
    dd_color    = RED
    sharpe_color = GREEN if sharpe >= 1.0 else (ORANGE if sharpe >= 0 else RED)

    cards = dbc.Row([
        dbc.Col(_metric_card("Total Return",  f"{total_ret:+.2%}", ret_color),   md=2),
        dbc.Col(_metric_card("Sharpe Ratio",  f"{sharpe:.2f}",    sharpe_color), md=2),
        dbc.Col(_metric_card("Max Drawdown",  f"{max_dd:.1%}",    dd_color),     md=2),
        dbc.Col(_metric_card("Win Rate",      f"{win_rate:.1%}",  TEXT),         md=2),
        dbc.Col(_metric_card("Total Trades",  str(int(n_trades)),  TEXT),         md=2),
        dbc.Col(_metric_card("Symbols",
                             ", ".join(meta.get("symbols", []) or ["—"]),
                             TEXT), md=2),
    ], className="mb-3 g-2")

    # Equity curve
    equity_fig = _build_equity_fig(bt, show_drawdown_shade=True)

    # Strategy breakdown
    strat_fig = _build_strategy_bar(bt)

    # Return histogram
    hist_fig = _build_return_hist(bt)

    return html.Div([
        cards,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=equity_fig, config={"displaylogo": False}), md=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=strat_fig, config={"displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(figure=hist_fig,  config={"displaylogo": False}), md=6),
        ]),
    ])


def _build_equity_fig(bt: dict, show_drawdown_shade: bool = False) -> go.Figure:
    ec = bt.get("equity_curve", [])
    if not ec:
        return _empty_fig("No equity curve data")

    df = pd.DataFrame(ec)
    if "timestamp" not in df.columns or "value" not in df.columns:
        return _empty_fig("Equity curve missing columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # Equity
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["value"],
        mode="lines", name="Portfolio Value",
        line=dict(color=BLUE, width=1.5),
        fill="tozeroy" if not show_drawdown_shade else None,
        fillcolor="rgba(88,166,255,0.05)",
    ), row=1, col=1)

    # Drawdown
    if "drawdown" in df.columns:
        dd = df["drawdown"].abs() * 100
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=-dd,
            mode="lines", name="Drawdown",
            line=dict(color=RED, width=1.0),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
        ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=dict(text="Portfolio Equity Curve", font=dict(color=TEXT)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Value ($)", row=1, col=1, gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(title_text="DD %",      row=2, col=1, gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def _build_strategy_bar(bt: dict) -> go.Figure:
    paired = bt.get("paired_trades", [])
    if not paired:
        return _empty_fig("No paired trades")

    df = pd.DataFrame(paired)
    if "strategy" not in df.columns or "net_pnl" not in df.columns:
        return _empty_fig("Paired trades missing columns")

    summary = df.groupby("strategy")["net_pnl"].agg(["sum", "count"]).reset_index()
    summary.columns = ["strategy", "total_pnl", "trades"]
    summary["color"] = summary["total_pnl"].apply(lambda x: GREEN if x >= 0 else RED)

    fig = go.Figure(go.Bar(
        x=summary["strategy"],
        y=summary["total_pnl"],
        marker_color=summary["color"],
        text=summary["trades"].apply(lambda n: f"{n} trades"),
        textposition="auto",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      title=dict(text="P&L by Strategy", font=dict(color=TEXT)),
                      yaxis_title="Net P&L ($)")
    return fig


def _build_return_hist(bt: dict) -> go.Figure:
    paired = bt.get("paired_trades", [])
    if not paired:
        return _empty_fig("No trade data")

    df = pd.DataFrame(paired)
    if "pnl_pct" not in df.columns:
        return _empty_fig("Missing pnl_pct column")

    returns = df["pnl_pct"].dropna()
    wins  = returns[returns >= 0]
    losses = returns[returns < 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins,   name="Wins",   marker_color=GREEN,
                               opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=losses, name="Losses", marker_color=RED,
                               opacity=0.7, nbinsx=20))
    fig.add_vline(x=0, line_color=TEXT_MUTED, line_dash="dash")

    fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode="overlay",
                      title=dict(text="Trade Return Distribution", font=dict(color=TEXT)),
                      xaxis_title="Return (%)", yaxis_title="Count")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def _tab_backtest(bt: dict) -> html.Div:
    # File selector
    files = _list_backtest_files()
    options = [{"label": Path(f).name, "value": f} for f in files]
    default = files[0] if files else None

    selector = dbc.Row([
        dbc.Col([
            html.Label("Report file:", style={"color": TEXT_MUTED, "fontSize": "0.8rem"}),
            dcc.Dropdown(
                id="bt-file-select",
                options=options,
                value=default,
                clearable=False,
                style={"background": PANEL_BG, "color": TEXT, "border": f"1px solid {BORDER}"},
            ),
        ], md=6),
    ], className="mb-3")

    # Panels (built from current bt dict in store)
    panels = _build_backtest_panels(bt)

    return html.Div([
        selector,
        html.Div(id="bt-panels-container", children=panels),
    ])


@app.callback(
    Output("bt-panels-container", "children"),
    Input("bt-file-select", "value"),
    prevent_initial_call=True,
)
def _on_bt_file_change(path):
    if not path:
        return _empty_fig("Select a backtest file")
    data = _load_json(path)
    return _build_backtest_panels(data)


def _build_backtest_panels(bt: dict) -> html.Div:
    if not bt:
        return html.Div(
            html.P("No backtest report loaded. Run: python main.py backtest -s event_driven ...",
                   style={"color": TEXT_MUTED, "fontFamily": "monospace", "padding": "2rem"}),
        )

    equity_fig  = _build_equity_fig(bt)
    dd_fig      = _build_drawdown_fig(bt)
    pnl_fig     = _build_pnl_bars(bt)
    monthly_fig = _build_monthly_pnl(bt)
    hist_fig    = _build_return_hist(bt)
    table       = _build_metrics_table(bt)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=equity_fig, config={"displaylogo": False}), md=8),
            dbc.Col(table, md=4),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=dd_fig,  config={"displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(figure=pnl_fig, config={"displaylogo": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=monthly_fig, config={"displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(figure=hist_fig,    config={"displaylogo": False}), md=6),
        ]),
    ])


def _build_drawdown_fig(bt: dict) -> go.Figure:
    ec = bt.get("equity_curve", [])
    if not ec:
        return _empty_fig("No equity curve data")

    df = pd.DataFrame(ec)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if "drawdown" not in df.columns:
        return _empty_fig("No drawdown column")

    dd = df["drawdown"].abs() * 100

    fig = go.Figure(go.Scatter(
        x=df["timestamp"], y=-dd,
        mode="lines", name="Drawdown",
        line=dict(color=RED, width=1.0),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=250,
                      title=dict(text="Drawdown", font=dict(color=TEXT)),
                      yaxis_title="Drawdown %", hovermode="x unified")
    return fig


def _build_pnl_bars(bt: dict) -> go.Figure:
    paired = bt.get("paired_trades", [])
    if not paired:
        return _empty_fig("No trade data")

    df = pd.DataFrame(paired)
    if "net_pnl" not in df.columns:
        return _empty_fig("Missing net_pnl column")

    df = df.reset_index()
    colors = [GREEN if v >= 0 else RED for v in df["net_pnl"]]

    # Cumulative overlay
    cum = df["net_pnl"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["net_pnl"], name="Trade P&L",
                         marker_color=colors, opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=cum, name="Cumulative",
                             line=dict(color=BLUE, width=1.5)), secondary_y=True)

    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      title=dict(text="Per-Trade P&L", font=dict(color=TEXT)),
                      hovermode="x unified")
    fig.update_yaxes(title_text="P&L ($)", secondary_y=False, gridcolor=BORDER)
    fig.update_yaxes(title_text="Cumulative ($)", secondary_y=True, gridcolor=BORDER)
    return fig


def _build_monthly_pnl(bt: dict) -> go.Figure:
    paired = bt.get("paired_trades", [])
    if not paired:
        return _empty_fig("No trade data")

    df = pd.DataFrame(paired)
    time_col = "exit_time" if "exit_time" in df.columns else "entry_time"
    if time_col not in df.columns or "net_pnl" not in df.columns:
        return _empty_fig("Missing time/pnl columns")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df["month"] = df[time_col].dt.tz_convert(None).dt.to_period("M").astype(str)

    monthly = df.groupby("month")["net_pnl"].sum().reset_index()
    colors = [GREEN if v >= 0 else RED for v in monthly["net_pnl"]]

    fig = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["net_pnl"],
        marker_color=colors, name="Monthly P&L",
    ))
    fig.add_hline(y=0, line_color=TEXT_MUTED, line_dash="dash")
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      title=dict(text="Monthly P&L", font=dict(color=TEXT)),
                      yaxis_title="Net P&L ($)", xaxis_tickangle=-45)
    return fig


def _build_metrics_table(bt: dict) -> dbc.Card:
    metrics = bt.get("metrics", {})
    meta    = bt.get("meta", {})

    rows_data = [
        ("Strategy",    ", ".join(meta.get("strategies", []) or ["—"])),
        ("Period",      f"{meta.get('start','?')} → {meta.get('end','?')}"),
        ("Interval",    meta.get("interval", "—")),
        ("Total Return",f"{metrics.get('total_return', 0):+.2%}"),
        ("Sharpe",      f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ("Max DD",      f"{metrics.get('max_drawdown', 0):.2%}"),
        ("Win Rate",    f"{metrics.get('win_rate', 0):.1%}"),
        ("Trades",      str(int(metrics.get("total_trades", 0)))),
        ("Commission",  f"${metrics.get('total_commission', 0):,.2f}"),
        ("Init Cap",    f"${metrics.get('initial_capital', 0):,.0f}"),
        ("Final Val",   f"${metrics.get('final_value', 0):,.2f}"),
    ]

    table_rows = []
    for label, val in rows_data:
        color = TEXT
        if label == "Total Return":
            v = metrics.get("total_return", 0)
            color = GREEN if v >= 0 else RED
        elif label == "Sharpe":
            v = metrics.get("sharpe_ratio", 0)
            color = GREEN if v >= 1 else (ORANGE if v >= 0 else RED)
        table_rows.append(html.Tr([
            html.Td(label, style={"color": TEXT_MUTED, "fontSize": "0.8rem",
                                  "padding": "4px 8px"}),
            html.Td(val,   style={"color": color,      "fontFamily": "monospace",
                                  "fontSize": "0.85rem", "padding": "4px 8px",
                                  "fontWeight": "600"}),
        ]))

    return dbc.Card(
        dbc.CardBody([
            html.H6("Metrics", style={"color": TEXT, "marginBottom": "0.5rem"}),
            html.Table(table_rows, style={"width": "100%", "borderCollapse": "collapse"}),
        ]),
        style={"background": PANEL_BG, "border": f"1px solid {BORDER}", "height": "100%"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _tab_validation(val: dict) -> html.Div:
    if not val:
        return html.Div(
            html.P(
                "No validation results. Run: python main.py validate -s event_driven "
                "-i 15m -d ibkr --start 2024-01-01 --end 2025-12-31 --save-report",
                style={"color": TEXT_MUTED, "fontFamily": "monospace", "padding": "2rem"},
            )
        )

    wf_fig     = _build_wf_bars(val)
    oos_fig    = _build_oos_equity(val)
    mc_fig     = _build_mc_hist(val)
    markov_fig = _build_markov_fan(val)
    heatmap    = _build_regime_heatmap(val)
    sharpe_fig = _build_sharpe_ci(val)
    verdict    = _build_verdict_card(val)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=wf_fig,  config={"displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(figure=oos_fig, config={"displaylogo": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=mc_fig,     config={"displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(figure=markov_fig, config={"displaylogo": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=heatmap,   config={"displaylogo": False}), md=4),
            dbc.Col(dcc.Graph(figure=sharpe_fig, config={"displaylogo": False}), md=4),
            dbc.Col(verdict, md=4),
        ]),
    ])


def _build_wf_bars(val: dict) -> go.Figure:
    wf = val.get("walk_forward", [])
    if not wf:
        return _empty_fig("No walk-forward data")

    labels = []
    train_rets = []
    test_rets  = []
    for w in wf:
        labels.append(w.get("label", ""))
        train_rets.append(w.get("train", {}).get("metrics", {}).get("total_return", 0) * 100)
        test_rets.append( w.get("test",  {}).get("metrics", {}).get("total_return", 0) * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="In-Sample",     x=labels, y=train_rets,
                         marker_color=BLUE,   opacity=0.8))
    fig.add_trace(go.Bar(name="Out-of-Sample", x=labels, y=test_rets,
                         marker_color=ORANGE, opacity=0.8))
    fig.add_hline(y=0, line_color=TEXT_MUTED, line_dash="dash")

    fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group",
                      title=dict(text="Walk-Forward: IS vs OOS Returns", font=dict(color=TEXT)),
                      yaxis_title="Return (%)", xaxis_tickangle=-20)
    return fig


def _build_oos_equity(val: dict) -> go.Figure:
    oos = val.get("out_of_sample", {})
    paired = oos.get("paired_trades", [])

    if isinstance(paired, list) and paired:
        df = pd.DataFrame(paired)
    elif isinstance(paired, dict):
        df = pd.DataFrame(paired)
    else:
        return _empty_fig("No OOS trade data")

    if df.empty or "net_pnl" not in df.columns:
        return _empty_fig("No OOS P&L data")

    cum = df["net_pnl"].cumsum()

    fig = go.Figure(go.Scatter(
        y=cum, mode="lines+markers", name="OOS Cumulative P&L",
        line=dict(color=GREEN if cum.iloc[-1] >= 0 else RED, width=1.5),
        marker=dict(size=4),
    ))
    fig.add_hline(y=0, line_color=TEXT_MUTED, line_dash="dash")
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text="OOS Cumulative P&L", font=dict(color=TEXT)),
                      yaxis_title="Cumulative P&L ($)", xaxis_title="Trade #")
    return fig


def _build_mc_hist(val: dict) -> go.Figure:
    mc = val.get("monte_carlo", {})
    all_returns = mc.get("all_returns", [])
    if not all_returns:
        return _empty_fig("No Monte Carlo data")

    returns = np.array(all_returns) * 100
    p5  = np.percentile(returns, 5)
    p50 = np.percentile(returns, 50)
    p95 = np.percentile(returns, 95)

    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=50,
        marker_color=BLUE, opacity=0.75, name="MC Returns",
    ))
    for val_line, name, color in [(p5, "P5", RED), (p50, "Median", ORANGE), (p95, "P95", GREEN)]:
        fig.add_vline(x=val_line, line_color=color, line_dash="dash",
                      annotation_text=name, annotation_font_color=color,
                      annotation_position="top right")

    prob = mc.get("prob_profitable", 0)
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text=f"Standard MC Return Dist — {prob:.0%} profitable",
                                 font=dict(color=TEXT)),
                      xaxis_title="Total Return (%)", yaxis_title="Count")
    return fig


def _build_markov_fan(val: dict) -> go.Figure:
    mkv = val.get("markov_mc", {})
    paths = mkv.get("path_equities", [])
    init_cap = mkv.get("initial_capital", 10_000)

    if not paths:
        return _empty_fig("No Markov-chain MC data")

    # Pad paths to same length
    max_len = max(len(p) for p in paths)
    paths_arr = np.array([p + [p[-1]] * (max_len - len(p)) for p in paths], dtype=float)

    # Normalise to % return
    paths_pct = (paths_arr - init_cap) / init_cap * 100

    median_path = np.median(paths_pct, axis=0)
    p5_path     = np.percentile(paths_pct, 5,  axis=0)
    p95_path    = np.percentile(paths_pct, 95, axis=0)
    x = list(range(max_len))

    fig = go.Figure()

    # Fan — thin traces (show max 50)
    for path in paths_pct[:50]:
        fig.add_trace(go.Scatter(
            x=x, y=path.tolist(), mode="lines",
            line=dict(color="rgba(88,166,255,0.06)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # 5th/95th band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=p95_path.tolist() + p5_path.tolist()[::-1],
        fill="toself", fillcolor="rgba(240,136,62,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th pct",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=x, y=median_path.tolist(), mode="lines",
        line=dict(color=GREEN, width=2.0), name="Median path",
    ))
    fig.add_hline(y=0, line_color=TEXT_MUTED, line_dash="dash")

    cur_regime = mkv.get("current_regime", 0)
    regime_name = ["Calm", "Stressed"][cur_regime]
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text=f"Markov-Chain MC — regime: {regime_name}",
                                 font=dict(color=TEXT)),
                      yaxis_title="Return (%)", xaxis_title="Trade #")
    return fig


def _build_regime_heatmap(val: dict) -> go.Figure:
    mkv = val.get("markov_mc", {})
    T = mkv.get("transition_matrix")
    if T is None:
        return _empty_fig("No regime data")

    T = np.array(T)
    labels = mkv.get("regime_names", ["Calm", "Stressed"])

    fig = go.Figure(go.Heatmap(
        z=T,
        x=[f"→{l}" for l in labels],
        y=[f"{l}→" for l in labels],
        colorscale=[[0, "#1a1f2e"], [0.5, ORANGE], [1, GREEN]],
        zmin=0, zmax=1,
        text=[[f"{T[i][j]:.2f}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9, color=TEXT)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text="Regime Transition Matrix", font=dict(color=TEXT)))
    return fig


def _build_sharpe_ci(val: dict) -> go.Figure:
    sig = val.get("sharpe_significance", {})
    if not sig:
        return _empty_fig("No Sharpe CI data\n(run validate --save-report)")

    sharpe   = sig.get("sharpe", 0)
    ci_lo    = sig.get("ci_low_5pct", 0)
    ci_hi    = sig.get("ci_high_95pct", 0)
    p_value  = sig.get("p_value", 1)
    is_sig   = sig.get("is_significant", False)
    n        = sig.get("n_trades", 0)

    # Bootstrap distribution histogram
    boot = np.array(sig.get("boot_sharpes", []))

    fig = go.Figure()

    if len(boot) > 0:
        fig.add_trace(go.Histogram(
            x=boot, nbinsx=40,
            marker_color=BLUE, opacity=0.6, name="Bootstrap Sharpes",
        ))

    # CI band as vrect
    fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor="rgba(240,136,62,0.15)",
                  line=dict(color=ORANGE, width=1, dash="dash"),
                  annotation_text="90% CI", annotation_position="top left",
                  annotation_font_color=ORANGE)

    # Observed Sharpe line
    fig.add_vline(x=sharpe, line_color=GREEN if is_sig else ORANGE, line_width=2,
                  annotation_text=f"Sharpe={sharpe:.2f}", annotation_font_color=GREEN if is_sig else ORANGE)

    # Zero line
    fig.add_vline(x=0, line_color=RED, line_dash="dash", line_width=1)

    sig_txt = f"p={p_value:.3f} {'SIGNIFICANT' if is_sig else 'NOT sig.'}"
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text=f"Sharpe CI — n={n} trades — {sig_txt}",
                                 font=dict(color=GREEN if is_sig else ORANGE)),
                      xaxis_title="Annualised Sharpe", yaxis_title="Count")
    return fig


def _build_verdict_card(val: dict) -> dbc.Card:
    score   = val.get("score", 0)
    checks  = val.get("checks", [])

    if score >= 4:
        verdict_text  = "PASS"
        verdict_color = GREEN
    elif score >= 3:
        verdict_text  = "MARGINAL"
        verdict_color = ORANGE
    else:
        verdict_text  = "FAIL"
        verdict_color = RED

    check_items = []
    for c in checks:
        is_pass = c.startswith("✓") or c.startswith("✓")
        color = GREEN if "✓" in c else (ORANGE if "~" in c else RED)
        check_items.append(html.P(c, style={"color": color, "fontSize": "0.8rem",
                                            "fontFamily": "monospace", "margin": "2px 0"}))

    # Sharpe significance
    sig = val.get("sharpe_significance", {})
    if sig:
        is_sig = sig.get("is_significant", False)
        sig_color = GREEN if is_sig else RED
        sig_txt = (
            f"✓ Sharpe p={sig.get('p_value', 1):.3f} — significant"
            if is_sig else
            f"✗ Sharpe p={sig.get('p_value', 1):.3f} — NOT significant"
        )
        check_items.append(html.P(sig_txt, style={"color": sig_color, "fontSize": "0.8rem",
                                                    "fontFamily": "monospace", "margin": "2px 0"}))

    return dbc.Card(
        dbc.CardBody([
            html.H5("Validation Verdict", style={"color": TEXT, "marginBottom": "0.5rem"}),
            html.H2(
                f"{verdict_text}  {score}/5",
                style={"color": verdict_color, "fontFamily": "monospace", "fontWeight": "700"},
            ),
            html.Hr(style={"borderColor": BORDER}),
            *check_items,
        ]),
        style={"background": PANEL_BG, "border": f"2px solid {verdict_color}",
               "borderRadius": "8px", "height": "100%"},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  QuantBot Web Dashboard")
    print("  ─────────────────────────────────────────────────────")
    print("  http://localhost:8050")
    print()
    print("  Loads: backtest_report.json  (most recent)")
    print("         validation_results.json  (run validate --save-report)")
    print("  Auto-refreshes every 30 seconds.")
    print("  ─────────────────────────────────────────────────────\n")

    app.run(debug=False, host="127.0.0.1", port=8050)
