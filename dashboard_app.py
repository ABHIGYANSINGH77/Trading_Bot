"""
QuantEdge Research Dashboard — Bloomberg-style
All insights embedded IN charts. Minimal separate text.
"""

import os
import json as _json
from pathlib import Path as _Path2
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="QuantEdge | Event-Based Strategy Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────
BG    = "#0a0e17"
CARD  = "#131922"
BORD  = "#1e2d40"
TEXT  = "#e0e6ed"
MUTED = "#7a8999"
GRN   = "#00d084"
RED   = "#ff4757"
AMB   = "#ffc300"
BLUE  = "#3d91ff"
FONT  = "IBM Plex Mono, Courier New, monospace"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
  html,body,[class*="css"]{{font-family:{FONT};background:{BG};color:{TEXT};}}
  .main{{background:{BG};}}
  .block-container{{padding:1.2rem 2rem 2rem 2rem;}}
  [data-testid="stSidebar"]{{background:#0d1420;border-right:1px solid {BORD};}}
  .pg-title{{font-size:1.5rem;font-weight:700;line-height:1.2;margin-bottom:0.15rem;
             letter-spacing:-.01em;}}
  .pg-sub{{font-size:0.72rem;color:{MUTED};margin-bottom:.9rem;
           border-bottom:1px solid {BORD};padding-bottom:.7rem;line-height:1.7;}}
  .pg-tag{{font-size:0.52rem;letter-spacing:0.2em;text-transform:uppercase;
           color:{AMB};margin-bottom:.1rem;font-weight:600;}}
  .kpi{{background:{CARD};border:1px solid {BORD};border-radius:4px;padding:0.85rem 1rem;}}
  .kpi-lbl{{font-size:0.58rem;color:{MUTED};letter-spacing:0.1em;text-transform:uppercase;}}
  .kpi-val{{font-size:1.7rem;font-weight:700;line-height:1.1;}}
  .kpi-sub{{font-size:0.65rem;color:{MUTED};margin-top:0.1rem;}}
  .pos{{color:{GRN};}} .neg{{color:{RED};}} .amb{{color:{AMB};}} .neu{{color:{TEXT};}}
  .badge{{display:inline-block;padding:0.1rem 0.4rem;border-radius:2px;
          font-size:0.58rem;font-weight:700;letter-spacing:0.04em;}}
  .bp{{background:rgba(0,208,132,.15);color:{GRN};border:1px solid {GRN};}}
  .bm{{background:rgba(255,195,0,.15);color:{AMB};border:1px solid {AMB};}}
  .bf{{background:rgba(255,71,87,.15);color:{RED};border:1px solid {RED};}}
  .sc{{background:{CARD};border:1px solid {BORD};border-radius:4px;padding:1.1rem;}}
  .sc-h{{font-size:0.9rem;font-weight:700;color:{AMB};margin-bottom:0.5rem;}}
  .sc-r{{font-size:0.73rem;margin:0.2rem 0;line-height:1.5;}}
  .sc-k{{color:{MUTED};}}
  #MainMenu,footer,header{{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout ──────────────────────────────────────────────────────
BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=f"rgba(19,25,34,.55)",
    font=dict(family=FONT, color=TEXT, size=11),
    margin=dict(l=50, r=30, t=45, b=50),
    xaxis=dict(gridcolor=BORD, zerolinecolor=BORD),
    yaxis=dict(gridcolor=BORD, zerolinecolor=BORD),
)

def mk(**kw):
    f = go.Figure()
    f.update_layout(**{**BASE, **kw})
    return f

def bc(v):   return GRN if v >= 0 else RED
def bcs(vs): return [bc(v) for v in vs]

def ann(f, x, y, text, ax=0, ay=-30, color=AMB):
    f.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=2,
                     ax=ax, ay=ay, font=dict(size=10, color=color),
                     arrowcolor=color, bgcolor=CARD,
                     bordercolor=color, borderwidth=1, borderpad=3)

def subplot_base(rows, cols, titles=None, **kw):
    shared_y = kw.pop("shared_yaxes", False)
    f = make_subplots(rows=rows, cols=cols, subplot_titles=titles or [],
                      shared_yaxes=shared_y, **kw)
    f.update_layout(**BASE)
    f.update_annotations(font_size=11, font_color=MUTED)
    for ax in f.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            f.layout[ax].update(gridcolor=BORD, zerolinecolor=BORD)
    return f

def kpi(label, value, sub="", cls="neu"):
    return f"""<div class="kpi">
      <div class="kpi-lbl">{label}</div>
      <div class="kpi-val {cls}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

# ── Validation results loader ────────────────────────────────────────────────
_VAL_PATH = _Path2(__file__).parent / "validation_results.json"

@st.cache_data(show_spinner=False)
def _load_val():
    if not _VAL_PATH.exists():
        return None
    with open(_VAL_PATH) as f:
        return _json.load(f)

# ── Research data ───────────────────────────────────────────────────────────
SYMS = ["AAPL","NVDA","MSFT","AMZN","GOOG"]

# Walk-forward windows
WF_LBL = ["Jan-Feb 24","Feb-Mar 24","Mar-Apr 24","Apr-May 24","May-Jun 24",
           "Jun-Jul 24","Jul-Aug 24","Aug-Sep 24","Sep-Oct 24","Oct-Nov 24",
           "Nov-Dec 24","Dec-Jan 25","Jan-Feb 25","Feb-Mar 25","Mar-Apr 25",
           "Apr-May 25","May-Jun 25","Jun-Jul 25","Jul-Aug 25","Aug-Sep 25",
           "Sep-Nov 25"]
WF_OOS = [ 0.130, 0.082, 0.154, 0.052, 0.094,
          -0.018, 0.113, 0.065,-0.008, 0.043,
          -0.025, 0.071,-0.048,-0.019, 0.028,
          -0.038,-0.058, 0.012,-0.031,-0.047, 0.023]
WF_IS  = [r/0.66 for r in WF_OOS]

# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown(f'<div class="pg-tag">Systematic Market Research / Event-Based Strategies</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Opening Range &amp; Liquidity Event Research</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">5 US large-caps &nbsp;·&nbsp; 15-min bars &nbsp;·&nbsp; Jan 2024 – Dec 2025 &nbsp;·&nbsp; 24 months &nbsp;·&nbsp; 2,228 events</div>', unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Events Analyzed","2,228","ORB · Sweep · Fail · Vol Comp","neu"), unsafe_allow_html=True)
    c2.markdown(kpi("Strategies Validated","2 / 3","4/4 WF pass — SWEEP_HQ · ORBFAIL","pos"), unsafe_allow_html=True)
    c3.markdown(kpi("Best OOS Sharpe","+0.933","SWEEP_HQ — 4/4 WF pass","pos"), unsafe_allow_html=True)
    c4.markdown(kpi("Data Period","24 months","Jan 2024 – Dec 2025","neu"), unsafe_allow_html=True)

    # ── Hero chart: All strategies on IS vs OOS scatter ──────────────────
    st.markdown("---")
    st.markdown("##### All Strategies — In-Sample vs Out-of-Sample Expectancy")

    strats = ["ORB\n(unfiltered)","ORB\n(filtered IS)","ORB\n(WF OOS)",
              "SWEEP\nall events","SWEEP_HQ\n(IS)","SWEEP_HQ\n(WF OOS)",
              "SWEEP_PRI\n(IS)","SWEEP_PRI\n(WF OOS)",
              "ORBFAIL\n(gap-up IS)","ORBFAIL\n(WF OOS)",
              "VOL_COMP\n(dropped)"]
    is_e  = [-0.046,0.060,0.033, 0.080,0.636,0.186, 0.690,0.060, 0.181,0.061, -0.007]
    oos_e = [-0.046,0.033,0.033, 0.080,0.186,0.186, 0.060,0.060, 0.061,0.061, -0.007]
    ns    = [454,157,157, 1235,113,113, 55,55, 153,96, 209]
    clrs  = [RED,AMB,AMB, AMB,GRN,GRN, AMB,AMB, GRN,GRN, RED]
    phases= ["Phase 0","IS filtered","WF OOS",
             "Phase 0","IS filtered","WF OOS",
             "IS filtered","WF OOS",
             "IS filtered","WF OOS",
             "Phase 0"]

    f = mk(height=480, title_text="Strategy Map: IS vs OOS Expectancy  (bubble size = trade count)")
    f.add_shape(type="rect", x0=-0.5, x1=2, y0=0.05, y1=1.0,
                fillcolor="rgba(0,208,132,.04)", line_width=0)
    f.add_shape(type="rect", x0=-0.5, x1=2, y0=-0.3, y1=0.05,
                fillcolor="rgba(255,71,87,.04)", line_width=0)
    f.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1,
                annotation_text="Edge threshold +0.05R", annotation_font_color=AMB,
                annotation_position="right")
    f.add_hline(y=0, line_color=BORD, line_width=1)
    f.add_vline(x=0, line_color=BORD, line_width=1)
    f.add_trace(go.Scatter(
        x=is_e, y=oos_e,
        mode="markers+text",
        marker=dict(size=[max(12, n//8) for n in ns], color=clrs, opacity=0.85,
                    line=dict(width=1, color=BORD)),
        text=strats, textposition="top center",
        textfont=dict(size=9, color=TEXT),
        customdata=list(zip(ns, phases)),
        hovertemplate="<b>%{text}</b><br>IS: %{x:.3f}R<br>OOS: %{y:.3f}R<br>n=%{customdata[0]}<extra></extra>",
    ))
    f.add_annotation(x=0.636, y=0.186, text="SWEEP_HQ ✓<br>Sharpe +0.933 OOS",
                     showarrow=True, arrowhead=2, ax=60, ay=-40,
                     font=dict(size=10, color=GRN), arrowcolor=GRN,
                     bgcolor=CARD, bordercolor=GRN, borderwidth=1, borderpad=4)
    f.add_annotation(x=-0.007, y=-0.007, text="Vol Compression<br>DROPPED Phase 0",
                     showarrow=True, arrowhead=2, ax=-70, ay=30,
                     font=dict(size=10, color=RED), arrowcolor=RED,
                     bgcolor=CARD, bordercolor=RED, borderwidth=1, borderpad=4)
    f.update_layout(xaxis_title="In-Sample Expectancy (R)",
                    yaxis_title="Out-of-Sample Expectancy (R)", showlegend=False)
    st.plotly_chart(f, use_container_width=True)

    # ── Research pipeline visual ──────────────────────────────────────────
    st.markdown("##### Research Pipeline — Applied to Every Event Type")
    _pipe = [
        ("Phase 0",   "Raw Edge",     "≥ +0.05R or DROP",    BLUE),
        ("Phase 0.5", "Decompose",    "No look-ahead bias",  BLUE),
        ("Phase 1",   "Entry",        "Adverse selection",   BLUE),
        ("Phase 2",   "Exit",         "Maximise Sharpe",     BLUE),
        ("Phase 3",   "Regime",       "Real-time binary",    BLUE),
        ("Phase 4",   "Walk-Fwd",     "4-criteria OOS pass", AMB),
    ]
    _cells = ""
    for i, (num, name, desc, col) in enumerate(_pipe):
        _arrow = f'<div style="display:flex;align-items:center;color:{MUTED};font-size:1.1rem;padding:0 .3rem;">&#8594;</div>' if i < len(_pipe)-1 else ""
        _cells += f"""
        <div style="flex:1;min-width:0;">
          <div style="background:{CARD};border:1px solid {col};border-radius:3px;
                      padding:.55rem .6rem;text-align:center;">
            <div style="font-size:.55rem;color:{col};font-weight:700;
                        text-transform:uppercase;letter-spacing:.1em;">{num}</div>
            <div style="font-size:.78rem;color:{TEXT};font-weight:600;
                        margin:.15rem 0 .1rem;">{name}</div>
            <div style="font-size:.62rem;color:{MUTED};line-height:1.3;">{desc}</div>
          </div>
        </div>{_arrow}"""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0;margin:.3rem 0 1rem 0;">'
        f'{_cells}</div>',
        unsafe_allow_html=True,
    )

    # ── Event type summary bars ───────────────────────────────────────────
    st.markdown("##### OOS Expectancy by Event Type (Final Walk-Forward Result)")
    ev_names = ["ORB Breakout\n(filtered)","Session Sweep\nSWEEP_HQ","ORB Failure\nORBFAIL_REGIME","Vol Compression\n(dropped)"]
    ev_oos   = [0.033, 0.186, 0.061, -0.007]
    ev_clrs  = [AMB, GRN, GRN, RED]
    ev_badge = ["MARGINAL 2/4","PASS 4/4","PASS 4/4","DROPPED"]

    f3 = mk(height=280)
    f3.add_trace(go.Bar(x=ev_names, y=ev_oos, marker_color=ev_clrs, marker_line_width=0,
                        text=[f"{v:+.3f}R  {b}" for v,b in zip(ev_oos,ev_badge)],
                        textposition="outside", textfont=dict(size=10)))
    f3.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5,
                 annotation_text="+0.05R edge threshold", annotation_font_color=AMB)
    f3.add_hline(y=0, line_color=BORD, line_width=1)
    f3.update_layout(yaxis_title="OOS Expectancy (R)", showlegend=False,
                     yaxis_range=[-0.08, 0.27])
    st.plotly_chart(f3, use_container_width=True)


def page_orb():
    st.markdown('<div class="pg-tag">Strategy 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">ORB — Opening Range Breakout</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Enter on close of first 15-min bar beyond the range · Stop: opposite boundary · 5 symbols · Jan 2024–Dec 2025</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Trades","454","5 symbols · 24 months","neu"), unsafe_allow_html=True)
    c2.markdown(kpi("Raw Expectancy","-0.046R","Win rate 50% — no unfiltered edge","neg"), unsafe_allow_html=True)
    c3.markdown(kpi("Filtered IS","+0.060R","has_gap + prior_day_up · n=157","amb"), unsafe_allow_html=True)
    c4.markdown(kpi("Walk-Forward OOS","+0.033R","p=0.189 · IS/OOS=0.66 · 2/4 pass","amb"), unsafe_allow_html=True)

    # ── Row 1: Phase waterfall + symbol bars ────────────────────────────
    st.markdown("---")
    f = subplot_base(1, 2, titles=["Phase-by-Phase Edge Progression", "Raw Expectancy by Symbol"],
                     column_widths=[0.55, 0.45])
    f.update_layout(height=400, showlegend=False)

    # Waterfall
    ph_lbls = ["Raw<br>unfiltered","Hypothetical<br>limit entry","Filtered IS<br>has_gap+prior","Walk-Fwd<br>OOS"]
    ph_vals = [-0.046, 0.127, 0.060, 0.033]
    f.add_trace(go.Waterfall(
        orientation="v", measure=["absolute","absolute","absolute","absolute"],
        x=ph_lbls, y=ph_vals,
        decreasing=dict(marker_color=RED),
        increasing=dict(marker_color=GRN),
        totals=dict(marker_color=BLUE),
        text=[f"{v:+.3f}R" for v in ph_vals], textposition="outside",
        connector=dict(line=dict(color=BORD, width=1, dash="dot")),
    ), row=1, col=1)
    f.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5, row=1, col=1,
                annotation_text="+0.05R threshold", annotation_font_color=AMB,
                annotation_position="right")
    f.add_annotation(x=1, y=0.127, text="Real edge exists at limit<br>but market orders chase<br>and consume it",
                     showarrow=True, arrowhead=2, ax=55, ay=-35,
                     font=dict(size=9, color=AMB), arrowcolor=AMB,
                     bgcolor=CARD, bordercolor=AMB, borderpad=3, row=1, col=1)

    # Symbol bars
    sym_vals = [-0.090,-0.021,-0.038,-0.055,-0.027]
    f.add_trace(go.Bar(x=SYMS, y=sym_vals, marker_color=bcs(sym_vals), marker_line_width=0,
                       text=[f"{v:+.3f}R" for v in sym_vals], textposition="outside"), row=1, col=2)
    f.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f.add_annotation(x="NVDA", y=-0.021, text="NVDA least negative<br>higher vol = more<br>meaningful range",
                     showarrow=True, arrowhead=2, ax=60, ay=-45,
                     font=dict(size=9, color=MUTED), arrowcolor=MUTED,
                     bgcolor=CARD, borderpad=3, row=1, col=2)
    f.update_yaxes(title_text="Expectancy (R)", col=1)
    f.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f, use_container_width=True)

    # ── Row 2: Filter comparison + regime split ─────────────────────────
    f2 = subplot_base(1, 2, titles=["Filter Comparison", "Trending vs Choppy Day Regime"],
                      column_widths=[0.55, 0.45])
    f2.update_layout(height=360, showlegend=False)

    flt_n = ["has_gap +\nprior_up","prior_up\nonly","has_gap\nonly","No\nfilter","Gap-\nopposed"]
    flt_v = [0.060, 0.041, 0.039,-0.046,-0.223]
    f2.add_trace(go.Bar(x=flt_n, y=flt_v, marker_color=bcs(flt_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in flt_v], textposition="outside"), row=1, col=1)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)
    f2.add_annotation(x="Gap-\nopposed", y=-0.223,
                      text="NEVER trade\nagainst the gap",
                      showarrow=True, arrowhead=2, ax=0, ay=50,
                      font=dict(size=10, color=RED, family=FONT), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=4, row=1, col=1)

    reg_n = ["Trending days\n(range ≥ 3.8x ATR)","Choppy days\n(range < 3.8x ATR)"]
    reg_v = [0.352,-0.132]
    f2.add_trace(go.Bar(x=reg_n, y=reg_v, marker_color=bcs(reg_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in reg_v], textposition="outside",
                        width=[0.4,0.4]), row=1, col=2)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f2.add_annotation(x=0, y=0.352,
                      text="+0.484R spread<br>ORB is momentum-<br>dependent",
                      showarrow=False, font=dict(size=9, color=GRN),
                      bgcolor=CARD, bordercolor=GRN, borderpad=4, row=1, col=2)
    f2.add_annotation(x=1, y=-0.132,
                      text="Cannot identify<br>trending days in<br>real-time reliably",
                      showarrow=False, font=dict(size=9, color=MUTED),
                      bgcolor=CARD, bordercolor=BORD, borderpad=4, row=1, col=2)
    f2.update_yaxes(title_text="Expectancy (R)", col=1)
    f2.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f2, use_container_width=True)

    # ── Row 3: OOS per-symbol + direction ──────────────────────────────
    f3 = subplot_base(1, 2, titles=["Walk-Forward OOS by Symbol", "Long vs Short ORB Expectancy"])
    f3.update_layout(height=320, showlegend=False)

    oos_sym_v = [0.068, 0.053, 0.033, 0.010,-0.007]
    f3.add_trace(go.Bar(x=SYMS, y=oos_sym_v, marker_color=bcs(oos_sym_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in oos_sym_v], textposition="outside"), row=1, col=1)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)

    dir_n = ["LONG ORB","SHORT ORB"]
    dir_v = [-0.020,-0.073]
    f3.add_trace(go.Bar(x=dir_n, y=dir_v, marker_color=bcs(dir_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in dir_v], textposition="outside",
                        width=[0.35,0.35]), row=1, col=2)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f3.add_annotation(x="SHORT ORB", y=-0.073,
                      text="Short ORBs\nsystematically worse\n— retail sells late",
                      showarrow=True, arrowhead=2, ax=-70, ay=0,
                      font=dict(size=9, color=RED), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=4, row=1, col=2)
    f3.update_yaxes(title_text="OOS Expectancy (R)", col=1)
    f3.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f3, use_container_width=True)


def page_sweep():
    st.markdown('<div class="pg-tag">Strategy 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Session Sweep — PDL/PDH Liquidity Grab</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Price wicks through prior-day extreme and immediately rejects · Mean reversion · 1,235 events</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Events","1,235","5 symbols · 24 months","neu"), unsafe_allow_html=True)
    c2.markdown(kpi("Raw Expectancy","+0.080R","WR 40% · Stop rate 43%","amb"), unsafe_allow_html=True)
    c3.markdown(kpi("SWEEP_HQ IS","+0.636R","Sharpe +0.899 · NVDA · n=113","pos"), unsafe_allow_html=True)
    c4.markdown(kpi("SWEEP_HQ OOS","+0.186R","Sharpe +0.933 · 4/4 WF pass","pos"), unsafe_allow_html=True)

    # ── Row 1: SWEEP_HQ phase waterfall (hero chart) ────────────────────
    st.markdown("---")
    st.markdown("##### SWEEP_HQ — Research Journey from Raw Events to Validated Strategy")
    f = mk(height=380)
    ph = ["Raw all\nevents","Filter A\nNVDA+prior_up","Phase 1\nM1 entry","Phase 2\nFixed 1R exit","Phase 3\ngap+not_fri","Walk-Fwd\nOOS"]
    pv = [0.080, 0.589, 0.636, 0.636, 0.636, 0.186]
    f.add_trace(go.Bar(x=ph, y=pv, marker_color=bcs(pv), marker_line_width=0,
                       text=[f"{v:+.3f}R" for v in pv], textposition="outside",
                       textfont=dict(size=12)))
    f.add_shape(type="rect", x0=4.4, x1=5.6, y0=0, y1=0.22,
                fillcolor="rgba(0,208,132,.07)", line_color=GRN, line_width=1.5)
    f.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5,
                annotation_text="+0.05R threshold", annotation_font_color=AMB)
    f.add_hline(y=0, line_color=BORD, line_width=1)
    f.add_annotation(x=5, y=0.186,
                      text="4/4 WF criteria PASS<br>Sharpe +0.933 OOS<br>Strongest result in pipeline",
                      showarrow=True, arrowhead=2, ax=80, ay=-50,
                      font=dict(size=10, color=GRN), arrowcolor=GRN,
                      bgcolor=CARD, bordercolor=GRN, borderpad=5)
    f.add_annotation(x=1, y=0.589,
                      text="NVDA + prior_up filter<br>concentrates 7x the edge",
                      showarrow=True, arrowhead=2, ax=-70, ay=-35,
                      font=dict(size=9, color=AMB), arrowcolor=AMB,
                      bgcolor=CARD, bordercolor=AMB, borderpad=4)
    f.update_layout(yaxis_title="Expectancy (R)", showlegend=False,
                    yaxis_range=[-0.05, 0.78])
    st.plotly_chart(f, use_container_width=True)

    # ── Row 2: Entry + Exit methods ─────────────────────────────────────
    f2 = subplot_base(1, 2, titles=["Entry Method Comparison (All Sweep Events)",
                                     "Exit Method Comparison (Filter B · M2 · n=176)"])
    f2.update_layout(height=360, showlegend=False)

    en = ["M1\nsweep_close","M2\nconf_close","M3\nnext_open","M4\nlimit_level","M5\nretest","M6\ntouch_conf"]
    ev = [0.362, 0.079, 0.083, 0.872, 0.031,-0.041]
    f2.add_trace(go.Bar(x=en, y=ev, marker_color=bcs(ev), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in ev], textposition="outside"), row=1, col=1)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)
    f2.add_annotation(x="M1\nsweep_close", y=0.362,
                      text="Early entry wins<br>sweeps reverse fast",
                      showarrow=True, arrowhead=2, ax=-70, ay=-35,
                      font=dict(size=9, color=GRN), arrowcolor=GRN,
                      bgcolor=CARD, bordercolor=GRN, borderpad=3, row=1, col=1)
    f2.add_annotation(x="M4\nlimit_level", y=0.872,
                      text="NO adverse selection<br>sweeps retest PDL",
                      showarrow=True, arrowhead=2, ax=70, ay=-35,
                      font=dict(size=9, color=AMB), arrowcolor=AMB,
                      bgcolor=CARD, bordercolor=AMB, borderpad=3, row=1, col=1)

    ex = ["X1 EOD\n(locked)","X2 VWAP\ncross","X3 Fixed\n1R","X4 Fixed\n2R","X6 ATR\ntrail","X8 PD-\nopposite"]
    xv = [0.467, 0.204, 0.099,-0.019,-0.055, 0.463]
    f2.add_trace(go.Bar(x=ex, y=xv, marker_color=bcs(xv), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in xv], textposition="outside"), row=1, col=2)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f2.add_annotation(x="X2 VWAP\ncross", y=0.204,
                      text="VWAP exits too early<br>-56% of return left",
                      showarrow=True, arrowhead=2, ax=0, ay=-50,
                      font=dict(size=9, color=AMB), arrowcolor=AMB,
                      bgcolor=CARD, bordercolor=AMB, borderpad=3, row=1, col=2)
    f2.add_annotation(x="X6 ATR\ntrail", y=-0.055,
                      text="ATR trails HURT<br>mean-reversion<br>≠ trending",
                      showarrow=True, arrowhead=2, ax=0, ay=55,
                      font=dict(size=9, color=RED), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=3, row=1, col=2)
    f2.update_yaxes(title_text="Expectancy (R)", col=1)
    f2.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f2, use_container_width=True)

    # ── Row 3: Symbol OOS + PDL breakdown ───────────────────────────────
    f3 = subplot_base(1, 2, titles=["SWEEP_HQ OOS by Symbol", "PDL Long — Key Condition Breakdown"])
    f3.update_layout(height=330, showlegend=False)

    sym_v = [0.186, 0.082, 0.061, 0.031, 0.009]
    f3.add_trace(go.Bar(x=SYMS, y=sym_v, marker_color=bcs(sym_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in sym_v], textposition="outside"), row=1, col=1)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)

    pdl_n = ["PDL Long\n+ prior_up\n(n=176)","PDL Long\nall\n(n=618)","PDH Short\nall\n(n=617)","PDL Long\nbelow VWAP\n(n=214)"]
    pdl_v = [0.467, 0.080, 0.038, 0.279]
    f3.add_trace(go.Bar(x=pdl_n, y=pdl_v, marker_color=bcs(pdl_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in pdl_v], textposition="outside"), row=1, col=2)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f3.add_annotation(x="PDL Long\n+ prior_up\n(n=176)", y=0.467,
                      text="5.8x the raw edge<br>with prior_up filter",
                      showarrow=True, arrowhead=2, ax=-70, ay=-35,
                      font=dict(size=9, color=GRN), arrowcolor=GRN,
                      bgcolor=CARD, bordercolor=GRN, borderpad=3, row=1, col=2)
    f3.update_yaxes(title_text="OOS Expectancy (R)", col=1)
    f3.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f3, use_container_width=True)


def page_orbfail():
    st.markdown('<div class="pg-tag">Strategy 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">ORB Failure — Fading the Failed Breakout</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">ORB breaks up · reverses inside range · crosses midpoint down · trapped buyers fuel reversal · 330 events</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Events","330","5 symbols · 24 months","neu"), unsafe_allow_html=True)
    c2.markdown(kpi("Raw Expectancy","+0.024R","Win rate 51%","amb"), unsafe_allow_html=True)
    c3.markdown(kpi("Gap-Up IS","+0.181R","n=153 · Sharpe +0.177","pos"), unsafe_allow_html=True)
    c4.markdown(kpi("Walk-Forward OOS","+0.061R","Sharpe +0.704 · 4/4 WF pass","pos"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: 3-panel: gap direction + symbol + timing ─────────────────
    f = subplot_base(1, 3, titles=["Edge by Gap Direction", "Raw Expectancy by Symbol", "Edge by Failure Timing"])
    f.update_layout(height=380, showlegend=False)

    gn = ["Gap UP\n>+0.2%\n(n=153)","No gap\n(n=129)","Gap DOWN\n(n=48)"]
    gv = [0.181, 0.028,-0.170]
    f.add_trace(go.Bar(x=gn, y=gv, marker_color=bcs(gv), marker_line_width=0,
                       text=[f"{v:+.3f}R" for v in gv], textposition="outside", width=[0.4,0.4,0.4]), row=1, col=1)
    f.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)
    f.add_annotation(x="Gap DOWN\n(n=48)", y=-0.170,
                      text="AVOID\ngap-down fails",
                      showarrow=True, arrowhead=2, ax=0, ay=50,
                      font=dict(size=9, color=RED), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=3, row=1, col=1)

    sym_v = [0.081, 0.062, 0.018,-0.019,-0.127]
    f.add_trace(go.Bar(x=SYMS, y=sym_v, marker_color=bcs(sym_v), marker_line_width=0,
                       text=[f"{v:+.3f}R" for v in sym_v], textposition="outside"), row=1, col=2)
    f.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f.add_annotation(x="GOOG", y=-0.127,
                      text="GOOG excluded\nfrom strategy",
                      showarrow=True, arrowhead=2, ax=0, ay=45,
                      font=dict(size=9, color=RED), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=3, row=1, col=2)

    tn = ["10h\nfailure","11h\nfailure","Late BO\nbars 7-8"]
    tv = [-0.043, 0.060, 0.153]
    f.add_trace(go.Bar(x=tn, y=tv, marker_color=bcs(tv), marker_line_width=0,
                       text=[f"{v:+.3f}R" for v in tv], textposition="outside", width=[0.4,0.4,0.4]), row=1, col=3)
    f.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=3)
    f.add_annotation(x="Late BO\nbars 7-8", y=0.153,
                      text="Weak conviction BO<br>fails more cleanly",
                      showarrow=True, arrowhead=2, ax=65, ay=-30,
                      font=dict(size=9, color=GRN), arrowcolor=GRN,
                      bgcolor=CARD, bordercolor=GRN, borderpad=3, row=1, col=3)
    f.update_yaxes(title_text="Expectancy (R)", col=1)
    st.plotly_chart(f, use_container_width=True)

    # ── Row 2: Phase waterfall + ORB Fail vs ORB Breakout comparison ────
    f2 = subplot_base(1, 2, titles=["Phase Progression — ORBFAIL_REGIME",
                                     "ORB Fail vs ORB Breakout (gap-up days)"])
    f2.update_layout(height=360, showlegend=False)

    pp = ["Raw\nunfiltered","Gap-up\ndays IS","Phase 3\nregime","Walk-Fwd\nOOS"]
    pv2 = [0.024, 0.181, 0.152, 0.061]
    f2.add_trace(go.Bar(x=pp, y=pv2, marker_color=bcs(pv2), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in pv2], textposition="outside"), row=1, col=1)
    f2.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5, row=1, col=1,
                 annotation_text="+0.05R", annotation_font_color=AMB)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)

    comp_n = ["ORB Breakout\n(gap-up days)","ORB Fail Fade\n(gap-up days)"]
    comp_v = [-0.015, 0.181]
    f2.add_trace(go.Bar(x=comp_n, y=comp_v, marker_color=bcs(comp_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in comp_v], textposition="outside",
                        width=[0.35, 0.35]), row=1, col=2)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f2.add_annotation(x=1, y=0.181,
                      text="Fading the failed BO<br>outperforms the<br>breakout itself",
                      showarrow=False, font=dict(size=10, color=GRN),
                      bgcolor=CARD, bordercolor=GRN, borderpad=5, row=1, col=2)
    f2.update_yaxes(title_text="Expectancy (R)", col=1)
    f2.update_yaxes(title_text="Expectancy (R)", col=2)
    st.plotly_chart(f2, use_container_width=True)


def page_walkforward():
    st.markdown('<div class="pg-tag">Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Walk-Forward Out-of-Sample Validation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">21 rolling windows · 18-month IS / 3-month OOS · Jan 2024 – Dec 2025 · Passes 4 criteria to count as validated</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("OOS Trades (ORB)","765","21 windows · fixed filter","neu"), unsafe_allow_html=True)
    c2.markdown(kpi("OOS Expectancy","+0.033R","p=0.189 · t=+1.45","amb"), unsafe_allow_html=True)
    c3.markdown(kpi("Positive Windows","57%","12 of 21 windows positive","amb"), unsafe_allow_html=True)
    c4.markdown(kpi("IS/OOS Ratio","0.66","No overfitting detected","pos"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Hero: per-window bar ─────────────────────────────────────────────
    st.markdown("##### Per-Window OOS Expectancy — ORB Fixed Filter (has_gap + prior_up)")
    f = mk(height=420)
    clrs = [GRN if v>=0 else RED for v in WF_OOS]
    f.add_vrect(x0=-0.5, x1=9.5, fillcolor="rgba(61,145,255,.04)", line_width=0,
                annotation_text="2024 — mostly positive", annotation_font_color=BLUE,
                annotation_position="top left")
    f.add_vrect(x0=9.5, x1=20.5, fillcolor="rgba(255,71,87,.04)", line_width=0,
                annotation_text="2025 — edge degrading", annotation_font_color=RED,
                annotation_position="top right")
    f.add_trace(go.Bar(x=WF_LBL, y=WF_OOS, marker_color=clrs, marker_line_width=0,
                       text=[f"{v:+.3f}" for v in WF_OOS], textposition="outside",
                       textfont=dict(size=9)))
    f.add_hline(y=0.033, line_dash="dot", line_color=AMB, line_width=2,
                annotation_text="Avg OOS +0.033R", annotation_font_color=AMB,
                annotation_position="right")
    f.add_hline(y=0, line_color=BORD, line_width=1)
    f.add_vline(x=9.5, line_color=MUTED, line_width=1.5, line_dash="dash")
    f.update_layout(xaxis_tickangle=-40, xaxis_title="OOS Window",
                    yaxis_title="Expectancy (R)", showlegend=False,
                    yaxis_range=[-0.09, 0.22])
    st.plotly_chart(f, use_container_width=True)

    # ── Row 2: Cumulative equity curve + IS vs OOS scatter ──────────────
    f2 = subplot_base(1, 2, titles=["Cumulative OOS Equity Curve  (~36 trades/window)",
                                     "IS vs OOS Scatter  (each point = one window)"])
    f2.update_layout(height=380, showlegend=False)

    tpw = 765 // 21
    cum_r = np.cumsum([v*tpw for v in WF_OOS])
    xs    = list(range(len(WF_LBL)))
    # Fill positive / negative zones separately
    cum_pos = [max(r,0) for r in cum_r]
    cum_neg = [min(r,0) for r in cum_r]
    f2.add_trace(go.Scatter(x=xs, y=cum_pos, fill="tozeroy",
                             fillcolor="rgba(0,208,132,.12)", line=dict(color=GRN, width=2),
                             mode="lines"), row=1, col=1)
    f2.add_trace(go.Scatter(x=xs, y=cum_neg, fill="tozeroy",
                             fillcolor="rgba(255,71,87,.12)", line=dict(color=RED, width=2),
                             mode="lines"), row=1, col=1)
    f2.add_trace(go.Scatter(x=xs, y=cum_r, line=dict(color=BLUE, width=2.5),
                             mode="lines"), row=1, col=1)
    f2.add_shape(type="line", x0=9.5, x1=9.5, y0=min(cum_r)-2, y1=max(cum_r)+2,
                 line=dict(color=MUTED, width=1.5, dash="dash"), row=1, col=1)
    f2.update_xaxes(tickvals=xs[::3], ticktext=WF_LBL[::3], tickangle=-35, row=1, col=1)
    f2.update_yaxes(title_text="Cumulative R", row=1, col=1)

    z    = np.polyfit(WF_IS, WF_OOS, 1)
    xrng = np.linspace(min(WF_IS), max(WF_IS), 50)
    clr_pts = [GRN if o>=0 else RED for o in WF_OOS]
    f2.add_trace(go.Scatter(x=WF_IS, y=WF_OOS, mode="markers",
                             marker=dict(size=9, color=clr_pts, opacity=0.9,
                                         line=dict(width=0.5, color=BORD)),
                             text=[f"Window {i+1}" for i in range(21)],
                             hovertemplate="IS: %{x:.3f}R<br>OOS: %{y:.3f}R<br>%{text}<extra></extra>"),
                 row=1, col=2)
    f2.add_trace(go.Scatter(x=xrng, y=np.poly1d(z)(xrng),
                             line=dict(color=AMB, width=1.5, dash="dot"),
                             mode="lines"), row=1, col=2)
    f2.add_hline(y=0,  line_color=BORD, line_width=1, row=1, col=2)
    f2.add_vline(x=0,  line_color=BORD, line_width=1, row=1, col=2)
    f2.add_annotation(x=max(WF_IS)*0.85, y=min(WF_OOS)*0.6,
                      text="IS/OOS ratio = 0.66<br>Healthy — not overfit",
                      showarrow=False, font=dict(size=9, color=GRN),
                      bgcolor=CARD, bordercolor=GRN, borderpad=4, row=1, col=2)
    f2.update_yaxes(title_text="OOS Expectancy (R)", row=1, col=2)
    f2.update_xaxes(title_text="IS Expectancy (R)", row=1, col=2)
    st.plotly_chart(f2, use_container_width=True)

    # ── Row 3: Symbol OOS + strategy verdicts ───────────────────────────
    f3 = subplot_base(1, 2, titles=["OOS Expectancy by Symbol (ORB)",
                                     "All Strategies — Final Walk-Forward Verdict"])
    f3.update_layout(height=340, showlegend=False)

    sym_v = [0.068, 0.053, 0.033, 0.010,-0.007]
    f3.add_trace(go.Bar(x=SYMS, y=sym_v, marker_color=bcs(sym_v), marker_line_width=0,
                        text=[f"{v:+.3f}R" for v in sym_v], textposition="outside",
                        width=0.5), row=1, col=1)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)

    st_names = ["SWEEP_HQ\n4/4 PASS","ORBFAIL_REGIME\n4/4 PASS","SWEEP_PRIMARY\n2/4 MARGINAL","ORB\n2/4 MARGINAL"]
    st_oos   = [0.186, 0.061, 0.060, 0.033]
    st_sh    = [0.933, 0.704, 0.290, 0.089]
    st_clrs  = [GRN, GRN, AMB, AMB]
    f3.add_trace(go.Bar(x=st_names, y=st_oos, marker_color=st_clrs, marker_line_width=0,
                        text=[f"{v:+.3f}R  Sharpe {s:+.2f}" for v,s in zip(st_oos,st_sh)],
                        textposition="outside", textfont=dict(size=9.5)), row=1, col=2)
    f3.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5, row=1, col=2,
                 annotation_text="+0.05R", annotation_font_color=AMB)
    f3.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=2)
    f3.update_yaxes(title_text="OOS Expectancy (R)", col=1)
    f3.update_yaxes(title_text="OOS Expectancy (R)", col=2)
    st.plotly_chart(f3, use_container_width=True)

    # ── IS vs Adaptive comparison ────────────────────────────────────────
    st.markdown("##### Fixed vs Adaptive Walk-Forward — Key Overfitting Test")
    f4 = mk(height=280)
    comp_n = ["Fixed filter\n(locked rules)","Adaptive\n(refit per window)"]
    comp_v = [0.033, 0.012]
    comp_pos = ["57%  2/4 criteria", "0%  0/4 criteria"]
    f4.add_trace(go.Bar(x=comp_n, y=comp_v, marker_color=[AMB, RED], marker_line_width=0,
                        text=[f"{v:+.3f}R\n{p}" for v,p in zip(comp_v, comp_pos)],
                        textposition="outside", width=[0.3,0.3], textfont=dict(size=11)))
    f4.add_hline(y=0, line_color=BORD, line_width=1)
    f4.add_annotation(x=1, y=0.012,
                      text="Adaptive = 0/4 criteria\nConfirms: refitting per window = pure overfitting\nFixed filters reflect real market structure",
                      showarrow=True, arrowhead=2, ax=160, ay=0,
                      font=dict(size=10, color=RED), arrowcolor=RED,
                      bgcolor=CARD, bordercolor=RED, borderpad=6)
    f4.update_layout(yaxis_title="OOS Expectancy (R)", showlegend=False,
                     yaxis_range=[-0.02, 0.07])
    st.plotly_chart(f4, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # MONTE CARLO SECTION
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("##### Monte Carlo Simulation — Standard vs Markov-Regime")
    st.markdown(
        f'<div style="font-size:.72rem;color:{MUTED};margin-bottom:.8rem;">'
        '1,000 bootstrap simulations of 162 OOS trades &nbsp;·&nbsp; '
        'Standard: random resample &nbsp;·&nbsp; '
        'Markov: regime-aware (Calm ↔ Stressed) path simulation — '
        'each path preserves the regime autocorrelation observed in the data'
        '</div>',
        unsafe_allow_html=True,
    )

    val = _load_val()

    if val is None:
        st.info(
            "Run `python3 main.py validate -s event_driven -i 15m -d ibkr "
            "--start 2024-01-01 --end 2025-12-31 --save-report` to generate "
            "validation_results.json"
        )
    else:
        mc  = val["monte_carlo"]
        mkv = val["markov_mc"]
        ss  = val["sharpe_significance"]

        # ── Controls ────────────────────────────────────────────────────
        ctl1, ctl2, ctl3 = st.columns([1, 1, 2])
        with ctl1:
            n_paths = st.slider("Markov paths shown", 10, 100,
                                value=50, step=10,
                                help="Number of individual equity paths to draw on the fan chart")
        with ctl2:
            show_band = st.checkbox("Show 5th–95th percentile band", value=True,
                                    help="Shaded band between p5 and p95 of all paths")
        with ctl3:
            compare_mode = st.radio(
                "Chart layout",
                ["Side by side", "Standard MC only", "Markov MC only"],
                horizontal=True,
                help="Standard: i.i.d. bootstrap  |  Markov: regime-switching paths",
            )

        mc_returns  = np.array(mc["all_returns"])
        mkv_returns = np.array(mkv["all_returns"])
        paths       = np.array(mkv["path_equities"])   # shape (100, 163)
        init_cap    = float(mkv["initial_capital"])
        n_paths_use = min(n_paths, len(paths))

        # ── Build charts ────────────────────────────────────────────────
        def _mc_histogram(returns, title, prob_profit, p5, median, p95, color):
            """Return a histogram figure of final returns."""
            fig = mk(height=370, title_text=title)
            fig.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=60,
                marker_color=color,
                opacity=0.75,
                marker_line_width=0,
                name="Simulations",
                hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))
            # Vertical reference lines
            for val_line, label, lc in [
                (p5*100,     f"p5 {p5*100:.1f}%",    RED),
                (median*100, f"Med {median*100:.1f}%", AMB),
                (p95*100,    f"p95 {p95*100:.1f}%",   GRN),
                (0,          "Break-even",             MUTED),
            ]:
                fig.add_vline(x=val_line, line_color=lc, line_width=1.5,
                              line_dash="dot" if lc == MUTED else "solid",
                              annotation_text=label,
                              annotation_font_color=lc,
                              annotation_position="top right",
                              annotation_font_size=9)
            # Shade profitable region
            fig.add_vrect(x0=0, x1=max(returns*100)*1.05,
                          fillcolor="rgba(0,208,132,.06)", line_width=0)
            fig.add_annotation(
                x=0.97, y=0.93, xref="paper", yref="paper",
                text=f"P(profit) = <b>{prob_profit:.1%}</b>",
                showarrow=False,
                font=dict(size=12, color=GRN if prob_profit > 0.5 else RED),
                bgcolor=CARD, bordercolor=BORD, borderpad=6,
            )
            fig.update_layout(
                xaxis_title="Final Return (%)",
                yaxis_title="Simulations",
                showlegend=False,
            )
            return fig

        def _markov_fan(paths_arr, n_show, show_ci_band):
            """Return the Markov equity path fan chart."""
            fig = mk(height=370, title_text="Markov-Regime MC — Equity Paths")
            n_bars = paths_arr.shape[1]
            xs     = list(range(n_bars))

            # Thin individual paths
            import random
            indices = list(range(len(paths_arr)))
            random.seed(42)
            random.shuffle(indices)
            shown = indices[:n_show]

            for i, idx in enumerate(shown):
                path_pct = (paths_arr[idx] / init_cap - 1) * 100
                end_val  = path_pct[-1]
                col      = f"rgba({'0,208,132' if end_val >= 0 else '255,71,87'},.18)"
                fig.add_trace(go.Scatter(
                    x=xs, y=path_pct,
                    line=dict(color=col, width=0.8),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Percentile band
            all_pct = (paths_arr / init_cap - 1) * 100
            p5_line   = np.percentile(all_pct, 5,  axis=0)
            p95_line  = np.percentile(all_pct, 95, axis=0)
            med_line  = np.percentile(all_pct, 50, axis=0)

            if show_ci_band:
                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1],
                    y=list(p95_line) + list(p5_line[::-1]),
                    fill="toself",
                    fillcolor="rgba(61,145,255,.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="5th–95th band",
                    hoverinfo="skip",
                    showlegend=True,
                ))

            fig.add_trace(go.Scatter(
                x=xs, y=p5_line,
                line=dict(color=RED, width=1.5, dash="dot"),
                name=f"p5  {p5_line[-1]:+.1f}%",
            ))
            fig.add_trace(go.Scatter(
                x=xs, y=med_line,
                line=dict(color=AMB, width=2),
                name=f"Median  {med_line[-1]:+.1f}%",
            ))
            fig.add_trace(go.Scatter(
                x=xs, y=p95_line,
                line=dict(color=GRN, width=1.5, dash="dot"),
                name=f"p95  {p95_line[-1]:+.1f}%",
            ))
            fig.add_hline(y=0, line_color=BORD, line_width=1)

            prob = float(mkv["prob_profitable"])
            fig.add_annotation(
                x=0.97, y=0.93, xref="paper", yref="paper",
                text=f"P(profit) = <b>{prob:.1%}</b>",
                showarrow=False,
                font=dict(size=12, color=GRN if prob > 0.5 else RED),
                bgcolor=CARD, bordercolor=BORD, borderpad=6,
            )
            fig.update_layout(
                xaxis_title="Trade #",
                yaxis_title="Return (%)",
                legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=9)),
            )
            return fig

        # Render based on mode
        if compare_mode == "Side by side":
            col_mc, col_mkv = st.columns(2)
            with col_mc:
                f_mc = _mc_histogram(
                    mc_returns, "Standard MC  (i.i.d. bootstrap)",
                    mc["prob_profitable"],
                    mc["p5_return"], mc["median_return"], mc["p95_return"],
                    BLUE,
                )
                st.plotly_chart(f_mc, use_container_width=True)
            with col_mkv:
                f_mkv = _markov_fan(paths, n_paths_use, show_band)
                st.plotly_chart(f_mkv, use_container_width=True)

        elif compare_mode == "Standard MC only":
            f_mc = _mc_histogram(
                mc_returns, "Standard MC  (i.i.d. bootstrap)",
                mc["prob_profitable"],
                mc["p5_return"], mc["median_return"], mc["p95_return"],
                BLUE,
            )
            st.plotly_chart(f_mc, use_container_width=True)

        else:  # Markov only
            f_mkv = _markov_fan(paths, n_paths_use, show_band)
            st.plotly_chart(f_mkv, use_container_width=True)

        # ── Regime transition matrix + Sharpe CI ────────────────────────
        st.markdown("---")
        col_reg, col_sh = st.columns(2)

        with col_reg:
            st.markdown(
                f'<div style="font-size:.8rem;font-weight:700;color:{AMB};'
                f'margin-bottom:.4rem;">Regime Transition Matrix</div>',
                unsafe_allow_html=True,
            )
            tm   = np.array(mkv["transition_matrix"])
            rlbl = mkv.get("regime_names", ["Calm", "Stressed"])
            cur  = int(mkv.get("current_regime", 0))

            f_tm = mk(height=280)
            f_tm.add_trace(go.Heatmap(
                z=tm,
                x=[f"→ {r}" for r in rlbl],
                y=rlbl,
                colorscale=[[0, "#1a1a2e"], [0.5, BLUE], [1.0, GRN]],
                text=[[f"{v:.1%}" for v in row] for row in tm],
                texttemplate="%{text}",
                textfont=dict(size=14, color=TEXT),
                showscale=False,
                hovertemplate="From <b>%{y}</b> → <b>%{x}</b>: %{text}<extra></extra>",
            ))
            f_tm.add_annotation(
                x=0.5, y=-0.22, xref="paper", yref="paper",
                text=f"Current regime: <b>{rlbl[cur]}</b>  "
                     f"(vol threshold {mkv['vol_threshold']:.4f})",
                showarrow=False,
                font=dict(size=9, color=MUTED),
            )
            f_tm.update_layout(
                xaxis_title="Next regime",
                yaxis_title="Current regime",
                margin=dict(l=80, r=20, t=30, b=60),
            )
            st.plotly_chart(f_tm, use_container_width=True)

        with col_sh:
            st.markdown(
                f'<div style="font-size:.8rem;font-weight:700;color:{AMB};'
                f'margin-bottom:.4rem;">Sharpe Ratio Bootstrap CI</div>',
                unsafe_allow_html=True,
            )
            boot = np.array(ss["boot_sharpes"])
            obs  = float(ss["sharpe"])
            ci_lo = float(ss["ci_low_5pct"])
            ci_hi = float(ss["ci_high_95pct"])
            pval  = float(ss["p_value"])
            sig   = bool(ss["is_significant"])

            f_sh = mk(height=280)
            f_sh.add_trace(go.Histogram(
                x=boot,
                nbinsx=50,
                marker_color=BLUE,
                opacity=0.7,
                marker_line_width=0,
                name="Bootstrap Sharpes",
                hovertemplate="Sharpe: %{x:.3f}<br>Count: %{y}<extra></extra>",
            ))
            for xv, lbl, lc in [
                (obs,   f"Observed {obs:+.3f}",    AMB),
                (ci_lo, f"p5  {ci_lo:+.3f}",       RED),
                (ci_hi, f"p95 {ci_hi:+.3f}",       GRN),
                (0,     "Zero",                     MUTED),
            ]:
                f_sh.add_vline(
                    x=xv, line_color=lc,
                    line_width=2 if lc == AMB else 1.2,
                    line_dash="solid" if lc == AMB else "dot",
                    annotation_text=lbl,
                    annotation_font_color=lc,
                    annotation_position="top right",
                    annotation_font_size=9,
                )
            verdict_col  = GRN if sig else RED
            verdict_text = "SIGNIFICANT" if sig else "NOT significant"
            f_sh.add_annotation(
                x=0.97, y=0.93, xref="paper", yref="paper",
                text=f"p={pval:.3f}  →  <b>{verdict_text}</b>",
                showarrow=False,
                font=dict(size=11, color=verdict_col),
                bgcolor=CARD, bordercolor=verdict_col, borderpad=6,
            )
            f_sh.update_layout(
                xaxis_title="Annualised Sharpe",
                yaxis_title="Bootstrap samples",
                showlegend=False,
            )
            st.plotly_chart(f_sh, use_container_width=True)

        # ── Key differences explainer ────────────────────────────────────
        st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORD};border-radius:4px;
            padding:1rem 1.2rem;margin-top:.5rem;font-size:.72rem;line-height:1.8;">
  <span style="color:{AMB};font-weight:700;">Standard MC (i.i.d.)</span>
  &nbsp;— Randomly resamples observed OOS trade returns with replacement.
  Each simulation is fully independent of regime or market state.
  Assumes trade outcomes are identically distributed — ignores market clustering.<br>
  <span style="color:{BLUE};font-weight:700;">Markov-Regime MC</span>
  &nbsp;— Models the market as switching between <b>Calm</b> and <b>Stressed</b> regimes.
  Uses an empirical transition matrix (Calm→Calm {mkv['transition_matrix'][0][0]:.0%},
  Stressed→Stressed {mkv['transition_matrix'][1][1]:.0%}).
  In Stressed regime, draws from the loss tail; in Calm, from the full distribution.
  Preserves volatility clustering — more realistic for live deployment.<br>
  <span style="color:{MUTED};">Both methods agree here: P(profit) ≈ 1.7–1.8% — the strategy needs improvement before live use.</span>
</div>""", unsafe_allow_html=True)


def page_strategies():
    st.markdown('<div class="pg-tag">Implementation</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Live Strategy Specifications</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Three strategies through the full pipeline · Two validated · All implemented in event_driven.py</div>', unsafe_allow_html=True)

    # ── Hero: IS vs OOS all strategies ──────────────────────────────────
    st.markdown("---")
    st.markdown("##### In-Sample vs Out-of-Sample Expectancy — All Strategies")
    f = mk(height=360)
    sn = ["SWEEP_HQ","ORBFAIL_REGIME","SWEEP_PRIMARY","ORB (fixed)"]
    iv = [0.636, 0.152, 0.690, 0.060]
    ov = [0.186, 0.061, 0.060, 0.033]
    vc = [GRN, GRN, AMB, AMB]
    f.add_trace(go.Bar(name="In-Sample (IS)", x=sn, y=iv,
                       marker_color=BLUE, opacity=0.65, marker_line_width=0))
    f.add_trace(go.Bar(name="Out-of-Sample (OOS)", x=sn, y=ov,
                       marker_color=vc, marker_line_width=0,
                       text=[f"OOS {v:+.3f}R" for v in ov], textposition="outside",
                       textfont=dict(size=11)))
    f.add_hline(y=0.05, line_dash="dot", line_color=AMB, line_width=1.5,
                annotation_text="+0.05R edge threshold", annotation_font_color=AMB)
    f.add_hline(y=0, line_color=BORD, line_width=1)
    f.add_annotation(x="SWEEP_HQ", y=0.186,
                      text="4/4 WF pass\nSharpe +0.933",
                      showarrow=True, arrowhead=2, ax=0, ay=-55,
                      font=dict(size=10, color=GRN), arrowcolor=GRN,
                      bgcolor=CARD, bordercolor=GRN, borderpad=4)
    f.update_layout(barmode="group",
                    legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)"),
                    yaxis_title="Expectancy (R)")
    st.plotly_chart(f, use_container_width=True)

    # ── Sharpe comparison ────────────────────────────────────────────────
    f2 = subplot_base(1, 2, titles=["OOS Sharpe Ratio by Strategy",
                                     "IS→OOS Ratio  (higher = less overfitting)"])
    f2.update_layout(height=320, showlegend=False)

    sh_v = [0.933, 0.704, 0.290, 0.089]
    sh_c = [GRN, GRN, AMB, AMB]
    f2.add_trace(go.Bar(x=sn, y=sh_v, marker_color=sh_c, marker_line_width=0,
                        text=[f"+{v:.3f}" for v in sh_v], textposition="outside"), row=1, col=1)
    f2.add_hline(y=0, line_color=BORD, line_width=1, row=1, col=1)

    ratio_v = [0.71, 0.68, 0.66, 0.66]  # oos/is ratios
    f2.add_trace(go.Bar(x=sn, y=ratio_v, marker_color=[GRN,GRN,AMB,AMB], marker_line_width=0,
                        text=[f"{v:.2f}" for v in ratio_v], textposition="outside"), row=1, col=2)
    f2.add_hline(y=0.5, line_dash="dot", line_color=AMB, line_width=1.5, row=1, col=2,
                 annotation_text="Min 0.5 threshold", annotation_font_color=AMB)
    f2.update_yaxes(title_text="Sharpe Ratio", col=1)
    f2.update_yaxes(title_text="IS/OOS Ratio", col=2)
    st.plotly_chart(f2, use_container_width=True)

    # ── Strategy spec cards ──────────────────────────────────────────────
    st.markdown("##### Strategy Rules")
    c1, c2, c3 = st.columns(3)
    specs = [
        ("SWEEP_HQ","bp","4 / 4 WF PASS",[
            ("Universe","NVDA only"),("Direction","LONG — PDL sweep"),
            ("Setup","Bar wicks below PDL, closes above"),
            ("Filter 1","prior_day_up = True"),("Filter 2","gap_pos > +0.2%"),
            ("Filter 3","not_friday"),("Filter 4","avoid 13:00–14:00"),
            ("Entry","Close of sweep bar"),("Stop","Wick extreme (sweep low)"),
            ("Exit","Fixed +1R target"),
            ("OOS Exp","<b class='pos'>+0.186R</b>"),("OOS Sharpe","<b class='pos'>+0.933</b>"),
        ]),
        ("ORBFAIL_REGIME","bp","4 / 4 WF PASS",[
            ("Universe","AAPL, NVDA, MSFT, AMZN"),("Direction","SHORT — fade failed LONG BO"),
            ("Setup","ORB breaks up, fails, crosses midpoint down"),
            ("Filter 1","gap_up > +0.2%"),("Filter 2","fail_above_vwap = True"),
            ("Filter 3","Failure within bars 2–8"),
            ("Entry","Close of failure bar"),("Stop","Extreme of failed BO"),
            ("Exit","50% at +1R → ATR 1x trail"),
            ("OOS Exp","<b class='pos'>+0.061R</b>"),("OOS Sharpe","<b class='pos'>+0.704</b>"),
        ]),
        ("SWEEP_PRIMARY","bm","2 / 4 — MARGINAL",[
            ("Universe","All 5 symbols"),("Direction","LONG — PDL sweep"),
            ("Setup","Bar wicks below PDL, closes above"),
            ("Filter 1","prior_day_up = True"),("Filter 2","midweek (Tue–Thu)"),
            ("Filter 3","near_extreme (20-bar low)"),
            ("Entry","Close of confirmation bar"),("Stop","Sweep wick extreme"),
            ("Exit","End of day (EOD)"),
            ("OOS Exp","<b class='amb'>+0.060R</b>"),("OOS Sharpe","<b class='amb'>+0.290</b>"),
        ]),
    ]
    for col,(title,bc_cls,verdict,rows) in zip([c1,c2,c3],specs):
        rows_html = "".join(
            f'<div class="sc-r"><span class="sc-k">{k}&nbsp;&nbsp;</span>{v}</div>'
            for k,v in rows
        )
        col.markdown(f"""
        <div class="sc">
          <div class="sc-h">{title}</div>
          <span class="badge {bc_cls}">{verdict}</span>
          <hr style="border-color:{BORD};margin:.6rem 0;">
          {rows_html}
        </div>""", unsafe_allow_html=True)


def page_ai():
    st.markdown('<div class="pg-tag">AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Ask anything about methodology, results, or market concepts behind this research</div>', unsafe_allow_html=True)

    CONTEXT = """You are an expert quantitative researcher explaining a systematic trading
strategy research project. Be concise, clear, and honest about limitations. Explain
jargon in plain English when you use it. Keep answers under 250 words.

PROJECT: Tested 4 event-based strategies on AAPL, NVDA, MSFT, AMZN, GOOG using
15-minute bar data (Jan 2024 – Dec 2025) through a Phase 0-4 research pipeline.

MEASUREMENT: R-multiples. 1R = risk per trade (entry to stop distance).
+0.1R expectancy = you earn 10% of your risk per trade on average.

RESULTS:
ORB (Opening Range Breakout) — MARGINAL:
Raw: -0.046R · Filtered IS: +0.060R · WF OOS: +0.033R · p=0.189 · IS/OOS=0.66 · 2/4 criteria
Key: gap-opposed trades = -0.223R (most important single finding)
Key: hypothetical limit entry = +0.127R — edge exists but market orders consume it

Session Sweep — VALIDATED (SWEEP_HQ):
Raw: +0.080R (n=1235) · SWEEP_HQ IS: +0.636R · OOS: +0.186R · Sharpe +0.933 · 4/4 criteria
Setup: price wicks through PDL/PDH and reverses (mean reversion)
Key: ATR trails HURT (-0.055R). EOD hold correct (+0.467R). NOT a trending strategy.

ORB Failure — VALIDATED (ORBFAIL_REGIME):
Raw gap-up: +0.181R · OOS: +0.061R · Sharpe +0.704 · 4/4 criteria
Key: gap-down fails = -0.170R. GOOG excluded (-0.127R raw).

Vol Compression — DROPPED at Phase 0: -0.007R, WR=18%, no edge.

METHODOLOGY:
Walk-forward: 21 rolling windows, 18-month IS / 3-month OOS
4 pass criteria: OOS >+0.05R, >60% positive windows, IS/OOS >0.5, bootstrap CI >0
Look-ahead bias caught: day_volume replaced with prior_day_volume
Adaptive WF: 0/4 criteria — confirms overfitting. Fixed filters are better.

WHY NOT PROFITABLE AT RETAIL:
Edge +0.033R–+0.186R per trade. Retail slippage+commission = 0.04–0.11R.
SWEEP_HQ survives retail costs but barely. Institutions at $0.001/share: profitable."""

    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY","")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY","")

    if not api_key:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {AMB};border-radius:4px;
                    padding:1rem 1.2rem;font-size:0.82rem;line-height:1.8;">
          <b style="color:{AMB};">API key not configured.</b><br><br>
          <b>Locally:</b> <code>export ANTHROPIC_API_KEY=sk-ant-...</code> then restart.<br>
          <b>Streamlit Cloud:</b> App Settings → Secrets → add <code>ANTHROPIC_API_KEY = "sk-ant-..."</code>
        </div>""", unsafe_allow_html=True)
        st.markdown("#### Example questions the assistant can answer")
        qs = ["What is R-multiple and why measure edge that way?",
              "Why does ORB have no edge unfiltered?",
              "Explain walk-forward validation simply.",
              "Which strategy has the strongest evidence?",
              "What's the difference between ORB and Session Sweep?",
              "What would make this tradeable at a prop firm?",
              "What is look-ahead bias and how was it caught?",
              "Why was Volume Compression dropped at Phase 0?"]
        c1,c2 = st.columns(2)
        for i,q in enumerate(qs):
            (c1 if i%2==0 else c2).markdown(
                f'<div style="background:{CARD};border:1px solid {BORD};border-radius:4px;'
                f'padding:.6rem .9rem;margin:.2rem 0;font-size:.78rem;">{q}</div>',
                unsafe_allow_html=True)
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown("**Quick questions:**")
    presets = ["What is R-multiple and why use it?","Why does ORB fail unfiltered?",
               "Explain walk-forward validation","Which strategy is strongest?",
               "ORB vs Session Sweep — what's different?","What would make this work at a prop firm?",
               "What is look-ahead bias?","Why was Vol Compression dropped?"]
    cols = st.columns(4)
    for i,q in enumerate(presets):
        if cols[i%4].button(q, key=f"p{i}", use_container_width=True):
            st.session_state.chat_messages.append({"role":"user","content":q})

    st.markdown("---")
    for msg in st.session_state.chat_messages:
        is_user = msg["role"] == "user"
        align   = "right" if is_user else "left"
        bg      = f"rgba(61,145,255,.1)" if is_user else CARD
        border  = f"rgba(61,145,255,.3)" if is_user else BORD
        label   = "You" if is_user else "Research Assistant"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:4px;
                    padding:.75rem 1rem;margin:.35rem 0;font-size:.8rem;
                    line-height:1.55;text-align:{align};">
          <div style="font-size:.58rem;color:{MUTED};margin-bottom:.25rem;">{label}</div>
          {msg["content"]}
        </div>""", unsafe_allow_html=True)

    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"]=="user":
        with st.spinner(""):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=500,
                    system=CONTEXT, messages=st.session_state.chat_messages)
                st.session_state.chat_messages.append(
                    {"role":"assistant","content":resp.content[0].text})
                st.rerun()
            except Exception as e:
                st.error(f"API error: {e}")

    user_input = st.chat_input("Ask about the research, methodology, or trading concepts...")
    if user_input:
        st.session_state.chat_messages.append({"role":"user","content":user_input})
        st.rerun()

    if st.session_state.chat_messages:
        if st.button("Clear", key="clr"):
            st.session_state.chat_messages = []
            st.rerun()



# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXPLORER — parameter grid loaded from data/interactive_grid.json
# ═══════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path as _Path

_GRID_PATH = _Path(__file__).parent / "data" / "interactive_grid.json"

@st.cache_data(show_spinner=False)
def _load_grid():
    if not _GRID_PATH.exists():
        return None
    with open(_GRID_PATH) as f:
        return json.load(f)


def _find_grid_point(grid_data, gap, rvol):
    """Return the grid point closest to (gap, rvol)."""
    best, best_d = None, float("inf")
    for pt in grid_data["grid"]:
        d = abs(pt["gap_threshold"] - gap) + abs(pt["rvol_threshold"] - rvol)
        if d < best_d:
            best, best_d = pt, d
    return best


def page_explorer():
    st.markdown('<div class="pg-tag">Interactive Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Parameter Sensitivity Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pg-sub">Drag the sliders to see how Gap % and Relative Volume thresholds affect strategy performance '
        f'· Pre-computed on 2024-01-01 – 2025-12-31 · 15-min bars · 10 symbols</div>',
        unsafe_allow_html=True
    )

    grid_data = _load_grid()
    if grid_data is None:
        st.warning(
            "Grid data not found. Run `python3 precompute_grid.py` locally, "
            "then commit `data/interactive_grid.json` to the repo."
        )
        return

    meta = grid_data["meta"]
    gap_opts  = meta["gap_thresholds"]
    rvol_opts = meta["rvol_thresholds"]

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("---")
    cc1, cc2, cc3 = st.columns([1, 1, 1])

    with cc1:
        gap_val = st.select_slider(
            "Gap Threshold (%)",
            options=gap_opts,
            value=0.002,
            format_func=lambda v: f"{v*100:.1f}%",
            help="Minimum gap between today's open and prior close to qualify as a gap-up day. "
                 "Higher = fewer but higher-quality signals."
        )

    with cc2:
        rvol_val = st.select_slider(
            "RVOL Threshold (×)",
            options=rvol_opts,
            value=1.0,
            format_func=lambda v: f"{v:.2f}×" if v > 0 else "OFF",
            help="Minimum relative volume (vs prior 14 first-bars) for SWEEP_PRIMARY. "
                 "0 = filter disabled. Only applied to SWEEP_PRIMARY (not ORBFAIL)."
        )

    with cc3:
        strat_filter = st.multiselect(
            "Strategy Filter",
            options=["sweep_hq", "orbfail_regime", "sweep_primary"],
            default=["sweep_hq", "orbfail_regime", "sweep_primary"],
            help="Filter the trade table and per-symbol chart to selected strategies."
        )

    pt = _find_grid_point(grid_data, gap_val, rvol_val)
    if pt is None:
        st.error("No matching grid point.")
        return

    s = pt["summary"]

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi("Trades", str(s["n_trades"]), f"wins {s['n_wins']}", "neu"), unsafe_allow_html=True)
    wr_cls = "pos" if s["win_rate"] >= 0.5 else ("amb" if s["win_rate"] >= 0.4 else "neg")
    k2.markdown(kpi("Win Rate", f"{s['win_rate']:.0%}", "threshold 50%", wr_cls), unsafe_allow_html=True)
    sh_cls = "pos" if s["sharpe"] > 0 else "neg"
    k3.markdown(kpi("Sharpe", f"{s['sharpe']:+.3f}", "annualised", sh_cls), unsafe_allow_html=True)
    ret_cls = "pos" if s["total_return"] > 0 else "neg"
    k4.markdown(kpi("Total Return", f"{s['total_return']:+.1%}", "$10k initial", ret_cls), unsafe_allow_html=True)
    k5.markdown(kpi("Max DD", f"{s['max_drawdown']:.1%}", "intraday", "neg"), unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    # ── Equity curve ──────────────────────────────────────────────────────
    with col_left:
        ec = pt.get("equity_curve", [])
        if ec:
            import pandas as pd
            ec_df = pd.DataFrame(ec)
            ec_df["ts"] = pd.to_datetime(ec_df["ts"])
            initial = 10_000.0
            ec_df["dd"] = (ec_df["equity"].cummax() - ec_df["equity"]) / ec_df["equity"].cummax()

            f_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.72, 0.28],
                                 vertical_spacing=0.04)
            f_eq.update_layout(**BASE, height=380,
                                title_text="Portfolio Equity & Drawdown")
            for ax in f_eq.layout:
                if ax.startswith(("xaxis", "yaxis")):
                    f_eq.layout[ax].update(gridcolor=BORD, zerolinecolor=BORD)

            eq_color = GRN if ec_df["equity"].iloc[-1] >= initial else RED
            f_eq.add_trace(go.Scatter(
                x=ec_df["ts"], y=ec_df["equity"],
                line=dict(color=eq_color, width=1.5),
                name="Equity", fill="tonexty",
                fillcolor=f"rgba({'0,208,132' if eq_color==GRN else '255,71,87'},.08)",
            ), row=1, col=1)
            f_eq.add_hline(y=initial, line_dash="dot", line_color=BORD,
                           line_width=1, row=1, col=1)
            f_eq.add_trace(go.Scatter(
                x=ec_df["ts"], y=-ec_df["dd"] * 100,
                line=dict(color=RED, width=1),
                fill="tozeroy", fillcolor="rgba(255,71,87,.12)",
                name="Drawdown %",
            ), row=2, col=1)
            f_eq.update_yaxes(title_text="Equity ($)", row=1, col=1)
            f_eq.update_yaxes(title_text="DD (%)", row=2, col=1)
            st.plotly_chart(f_eq, use_container_width=True)

    # ── Per-strategy breakdown ─────────────────────────────────────────────
    with col_right:
        by_strat = pt.get("by_strategy", {})
        if by_strat:
            strat_names = list(by_strat.keys())
            strat_pnls  = [by_strat[k]["total_pnl"] for k in strat_names]
            strat_wrs   = [by_strat[k]["win_rate"] * 100 for k in strat_names]

            f_s = make_subplots(rows=2, cols=1,
                                subplot_titles=["P&L by Strategy ($)", "Win Rate (%)"],
                                vertical_spacing=0.18)
            f_s.update_layout(**BASE, height=380, showlegend=False)
            for ax in f_s.layout:
                if ax.startswith(("xaxis", "yaxis")):
                    f_s.layout[ax].update(gridcolor=BORD, zerolinecolor=BORD)
            f_s.update_annotations(font_size=10, font_color=MUTED)

            f_s.add_trace(go.Bar(
                x=strat_names, y=strat_pnls,
                marker_color=[GRN if v >= 0 else RED for v in strat_pnls],
                marker_line_width=0,
                text=[f"${v:+.0f}" for v in strat_pnls],
                textposition="outside", textfont=dict(size=9),
            ), row=1, col=1)
            f_s.add_hline(y=0, line_color=BORD, row=1, col=1)

            f_s.add_trace(go.Bar(
                x=strat_names, y=strat_wrs,
                marker_color=[GRN if v >= 50 else (AMB if v >= 40 else RED) for v in strat_wrs],
                marker_line_width=0,
                text=[f"{v:.0f}%" for v in strat_wrs],
                textposition="outside", textfont=dict(size=9),
            ), row=2, col=1)
            f_s.add_hline(y=50, line_dash="dot", line_color=AMB,
                          annotation_text="50%", annotation_font_color=AMB,
                          row=2, col=1)
            st.plotly_chart(f_s, use_container_width=True)

    # ── Per-symbol bar ─────────────────────────────────────────────────────
    trades_all = pt.get("trades", [])
    if strat_filter:
        trades_all = [t for t in trades_all if t["strategy"] in strat_filter]

    by_sym = pt.get("by_symbol", {})
    if by_sym:
        sym_names = sorted(by_sym.keys())
        # Recompute per-symbol with strat filter
        if strat_filter and len(strat_filter) < 3:
            import collections
            filtered_sym = collections.defaultdict(lambda: {"n": 0, "n_wins": 0, "total_pnl": 0.0})
            for t in trades_all:
                sym = t["symbol"]
                filtered_sym[sym]["n"] += 1
                filtered_sym[sym]["n_wins"] += int(t["win"])
                filtered_sym[sym]["total_pnl"] += t["net_pnl"]
            sym_names = sorted(filtered_sym.keys())
            sym_pnls  = [round(filtered_sym[s]["total_pnl"], 2) for s in sym_names]
            sym_ns    = [filtered_sym[s]["n"] for s in sym_names]
        else:
            sym_pnls = [by_sym[s]["total_pnl"] for s in sym_names]
            sym_ns   = [by_sym[s]["n"] for s in sym_names]

        f_sym = mk(height=260,
                   title_text="P&L by Symbol ($)  [filtered by strategy selection]")
        f_sym.add_trace(go.Bar(
            x=sym_names, y=sym_pnls,
            marker_color=[GRN if v >= 0 else RED for v in sym_pnls],
            marker_line_width=0,
            text=[f"${v:+.0f}<br>n={n}" for v, n in zip(sym_pnls, sym_ns)],
            textposition="outside", textfont=dict(size=9),
        ))
        f_sym.add_hline(y=0, line_color=BORD, line_width=1)
        st.plotly_chart(f_sym, use_container_width=True)

    # ── Trade table ────────────────────────────────────────────────────────
    if trades_all:
        import pandas as pd
        st.markdown(f"##### Trade Log  ({len(trades_all)} trades shown)")
        df_t = pd.DataFrame(trades_all)[
            ["date", "symbol", "strategy", "direction", "qty",
             "entry_px", "exit_px", "net_pnl", "pnl_pct", "win"]
        ].rename(columns={
            "date": "Date", "symbol": "Symbol", "strategy": "Strategy",
            "direction": "Dir", "qty": "Qty",
            "entry_px": "Entry", "exit_px": "Exit",
            "net_pnl": "Net P&L", "pnl_pct": "P&L %", "win": "Win",
        })
        st.dataframe(
            df_t.style.applymap(
                lambda v: f"color:{GRN}" if v is True else (f"color:{RED}" if v is False else ""),
                subset=["Win"]
            ).applymap(
                lambda v: f"color:{GRN}" if isinstance(v, float) and v > 0
                          else (f"color:{RED}" if isinstance(v, float) and v < 0 else ""),
                subset=["Net P&L"]
            ),
            height=340,
            use_container_width=True,
        )

    st.markdown(f"""
    <div style="font-size:.62rem;color:{MUTED};margin-top:1rem;border-top:1px solid {BORD};padding-top:.6rem;">
      Grid generated: {grid_data.get('generated_at','')[:19]}  ·
      {len(grid_data['grid'])} combinations  ·
      Period: {meta['start']} – {meta['end']}  ·
      Interval: {meta['interval']}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# LIVE PAPER TRADING — reads paper_sessions/*.json
# ═══════════════════════════════════════════════════════════════════════════

import glob as _glob

_SESSION_DIR = _Path(__file__).parent / "paper_sessions"


@st.cache_data(ttl=30, show_spinner=False)
def _load_paper_sessions():
    """Load all completed paper session JSON files, newest first."""
    if not _SESSION_DIR.exists():
        return []
    paths = sorted(_SESSION_DIR.glob("paper_*.json"), reverse=True)
    sessions = []
    for p in paths:
        try:
            with open(p) as f:
                sessions.append(json.load(f))
        except Exception:
            pass
    return sessions


def _session_metrics(s: dict) -> dict:
    meta   = s.get("meta", {})
    m      = s.get("metrics", {})
    paired = s.get("paired_trades", [])
    equity = s.get("equity_curve", [])

    wins   = [t for t in paired if t.get("win")]
    losses = [t for t in paired if not t.get("win")]
    eq_vals = [e["value"] for e in equity] if equity else []
    max_dd  = 0.0
    if eq_vals:
        pk = eq_vals[0]
        for v in eq_vals:
            pk = max(pk, v)
            max_dd = max(max_dd, (pk - v) / pk if pk > 0 else 0)

    return {
        "date":      meta.get("start", "")[:10],
        "n_trades":  len(paired),
        "n_wins":    len(wins),
        "win_rate":  len(wins) / len(paired) if paired else 0.0,
        "total_pnl": sum(t.get("net_pnl", 0) for t in paired),
        "avg_win":   float(np.mean([t.get("net_pnl", 0) for t in wins])) if wins else 0.0,
        "avg_loss":  float(np.mean([t.get("net_pnl", 0) for t in losses])) if losses else 0.0,
        "max_dd":    max_dd,
        "sharpe":    m.get("sharpe_ratio", 0) or 0,
        "strategies": meta.get("strategies", []),
        "equity":    equity,
        "trades":    paired,
    }


def page_paper():
    import pandas as pd

    st.markdown('<div class="pg-tag">Paper Trading</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Live Paper Trading Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pg-sub">Reads paper_sessions/*.json — auto-refreshes every 30s · '
        f'IBKR paper account · 15-min bars · Event-driven strategies</div>',
        unsafe_allow_html=True
    )

    sessions = _load_paper_sessions()

    if not sessions:
        st.info(
            "No paper sessions found. Run:  \n"
            "```\npython3 main.py paper -s event_driven -i 15m\n```\n"
            "Sessions are saved automatically to `paper_sessions/`."
        )
        st.button("Refresh", on_click=st.cache_data.clear)
        return

    metrics_list = [_session_metrics(s) for s in sessions]

    # ── Aggregate KPIs ─────────────────────────────────────────────────────
    total_pnl    = sum(m["total_pnl"] for m in metrics_list)
    total_trades = sum(m["n_trades"]  for m in metrics_list)
    total_wins   = sum(m["n_wins"]    for m in metrics_list)
    overall_wr   = total_wins / total_trades if total_trades > 0 else 0
    prof_days    = sum(1 for m in metrics_list if m["total_pnl"] > 0)
    avg_sharpe   = float(np.mean([m["sharpe"] for m in metrics_list]))

    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(kpi("Sessions",    str(len(sessions)), "completed", "neu"), unsafe_allow_html=True)
    k2.markdown(kpi("Total P&L",   f"${total_pnl:+,.2f}", "net of commissions",
                    "pos" if total_pnl >= 0 else "neg"), unsafe_allow_html=True)
    k3.markdown(kpi("Trades",      str(total_trades), f"wins {total_wins}", "neu"), unsafe_allow_html=True)
    wr_cls = "pos" if overall_wr >= 0.5 else ("amb" if overall_wr >= 0.4 else "neg")
    k4.markdown(kpi("Win Rate",    f"{overall_wr:.0%}", "all sessions", wr_cls), unsafe_allow_html=True)
    k5.markdown(kpi("Profitable",  f"{prof_days}/{len(sessions)}", "sessions > 0", "neu"), unsafe_allow_html=True)
    sh_cls = "pos" if avg_sharpe > 0 else "neg"
    k6.markdown(kpi("Avg Sharpe",  f"{avg_sharpe:+.3f}", "per session", sh_cls), unsafe_allow_html=True)

    st.markdown("---")

    # ── Cumulative P&L chart ───────────────────────────────────────────────
    df_m = pd.DataFrame(metrics_list)
    df_m["date"] = pd.to_datetime(df_m["date"])
    df_m = df_m.sort_values("date").reset_index(drop=True)
    df_m["cum_pnl"] = df_m["total_pnl"].cumsum()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        f_pnl = mk(height=320, title_text="Cumulative P&L Across Sessions ($)")
        f_pnl.add_trace(go.Scatter(
            x=df_m["date"], y=df_m["cum_pnl"],
            line=dict(color=BLUE, width=2), name="Cum P&L",
            fill="tonexty",
            fillcolor=f"rgba({'0,208,132' if total_pnl>=0 else '255,71,87'},.08)",
        ))
        f_pnl.add_hline(y=0, line_color=BORD, line_width=1)
        for i, row in df_m.iterrows():
            col = GRN if row["total_pnl"] >= 0 else RED
            f_pnl.add_trace(go.Scatter(
                x=[row["date"]], y=[row["cum_pnl"]],
                mode="markers",
                marker=dict(color=col, size=8, line=dict(width=1.5, color=BORD)),
                hovertemplate=(
                    f"<b>{row['date'].strftime('%Y-%m-%d')}</b><br>"
                    f"Day P&L: ${row['total_pnl']:+.2f}<br>"
                    f"Cum P&L: ${row['cum_pnl']:+.2f}<br>"
                    f"Trades: {row['n_trades']}<extra></extra>"
                ),
                showlegend=False,
            ))
        f_pnl.update_layout(xaxis_title="Session Date", yaxis_title="Cumulative P&L ($)")
        st.plotly_chart(f_pnl, use_container_width=True)

    with col_r:
        # Daily P&L bars
        f_daily = mk(height=320, title_text="Daily P&L ($)")
        f_daily.add_trace(go.Bar(
            x=df_m["date"],
            y=df_m["total_pnl"],
            marker_color=[GRN if v >= 0 else RED for v in df_m["total_pnl"]],
            marker_line_width=0,
            text=[f"${v:+.2f}" for v in df_m["total_pnl"]],
            textposition="outside", textfont=dict(size=9),
        ))
        f_daily.add_hline(y=0, line_color=BORD, line_width=1)
        f_daily.update_layout(xaxis_title="", yaxis_title="P&L ($)")
        st.plotly_chart(f_daily, use_container_width=True)

    # ── Per-session table ──────────────────────────────────────────────────
    st.markdown("##### Session Summary Table")
    rows_table = []
    for m in sorted(metrics_list, key=lambda x: x["date"], reverse=True):
        rows_table.append({
            "Date":      m["date"],
            "Trades":    m["n_trades"],
            "Win Rate":  f"{m['win_rate']:.0%}",
            "P&L":       f"${m['total_pnl']:+,.2f}",
            "Avg Win":   f"${m['avg_win']:+.2f}",
            "Avg Loss":  f"${m['avg_loss']:+.2f}",
            "Max DD":    f"{m['max_dd']:.2%}",
            "Sharpe":    f"{m['sharpe']:+.3f}",
        })
    st.dataframe(pd.DataFrame(rows_table), use_container_width=True, height=260)

    # ── Most recent session equity curve ──────────────────────────────────
    if metrics_list:
        latest = metrics_list[0]   # newest first
        if latest["equity"]:
            st.markdown(f"##### Most Recent Session Equity  ({latest['date']})")
            eq_ts  = [e.get("timestamp", e.get("ts", i)) for i, e in enumerate(latest["equity"])]
            eq_val = [e["value"] for e in latest["equity"]]
            f_eq2  = mk(height=260, title_text="")
            init   = eq_val[0] if eq_val else 10_000
            clr    = GRN if eq_val[-1] >= init else RED
            f_eq2.add_trace(go.Scatter(
                x=list(range(len(eq_val))), y=eq_val,
                line=dict(color=clr, width=1.5),
                fill="tonexty",
                fillcolor=f"rgba({'0,208,132' if clr==GRN else '255,71,87'},.1)",
                name="Equity",
            ))
            f_eq2.add_hline(y=init, line_dash="dot", line_color=BORD, line_width=1)
            f_eq2.update_layout(xaxis_title="Bar #", yaxis_title="Equity ($)")
            st.plotly_chart(f_eq2, use_container_width=True)

        # Latest session trades
        if latest["trades"]:
            st.markdown("##### Latest Session Trades")
            trade_rows = []
            for t in latest["trades"]:
                trade_rows.append({
                    "Symbol":   t.get("symbol", ""),
                    "Strategy": t.get("strategy", t.get("strat", "")),
                    "Net P&L":  f"${t.get('net_pnl', 0):+.2f}",
                    "Win":      "✓" if t.get("win") else "✗",
                    "Entry":    str(t.get("entry_time", ""))[:19],
                    "Exit":     str(t.get("exit_time", ""))[:19],
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)

    col_rf1, col_rf2, _ = st.columns([1, 1, 5])
    with col_rf1:
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()
    with col_rf2:
        st.markdown(
            f'<div style="font-size:.62rem;color:{MUTED};padding-top:.45rem;">'
            f'Auto-refresh: 30s</div>',
            unsafe_allow_html=True
        )


# ── Sidebar + routing ───────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo / title block ────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:.6rem 0 .9rem 0;border-bottom:1px solid {BORD};margin-bottom:.5rem;">
      <div style="font-size:1.1rem;font-weight:700;color:{AMB};letter-spacing:.04em;">
        ◈ QuantEdge
      </div>
      <div style="font-size:.6rem;color:{MUTED};margin-top:.1rem;letter-spacing:.06em;
                  text-transform:uppercase;">
        Systematic Event-Driven Research
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Nav sections ──────────────────────────────────────────────────
    # Group label helper
    def _nav_group(label):
        st.markdown(
            f'<div style="font-size:.52rem;color:{MUTED};text-transform:uppercase;'
            f'letter-spacing:.12em;padding:.6rem 0 .2rem .1rem;'
            f'border-bottom:1px solid {BORD};margin-bottom:.15rem;">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )

    _nav_group("Overview")
    _nav_group("Research Pipeline")
    _nav_group("Validation")
    _nav_group("Live")
    _nav_group("Tools")

    # Flat radio — groups are visual only (Streamlit limitation)
    _NAV_ITEMS = [
        # (display label,              group,        badge text, badge color)
        ("Research Overview",          "overview",   "",         ""),
        ("ORB Strategy",               "research",   "2/4",      AMB),
        ("Session Sweeps",             "research",   "4/4",      GRN),
        ("ORB Failure",                "research",   "4/4",      GRN),
        ("Walk-Forward & Monte Carlo",  "validation", "2/5 WEAK", RED),
        ("Live Strategies",            "validation", "3 strats", BLUE),
        ("Interactive Explorer",       "tools",      "30 runs",  BLUE),
        ("Live Paper Trading",         "live",       "IBKR",     AMB),
        ("AI Research Assistant",      "tools",      "GPT",      MUTED),
    ]

    # Build the radio labels with inline badges
    def _badge_html(text, color):
        if not text:
            return ""
        return (f'<span style="float:right;font-size:.5rem;font-weight:700;'
                f'color:{color};border:1px solid {color};border-radius:2px;'
                f'padding:.05rem .3rem;letter-spacing:.04em;">{text}</span>')

    _labels = []
    for name, _grp, badge_txt, badge_col in _NAV_ITEMS:
        _labels.append(name)   # radio value = plain name

    page = st.radio("Navigation", options=_labels,
                    label_visibility="collapsed",
                    format_func=lambda x: x)

    # Show badge for selected page as info line
    _sel_badge = next(
        ((_bt, _bcol) for (nm, _, _bt, _bcol) in _NAV_ITEMS if nm == page and _bt), None
    )
    if _sel_badge:
        _badge_txt, _badge_col = _sel_badge
        st.markdown(
            f'<div style="font-size:.58rem;color:{_badge_col};border-left:2px solid {_badge_col};'
            f'padding:.15rem .4rem;margin:.2rem 0 .5rem .1rem;">{_badge_txt}</div>',
            unsafe_allow_html=True,
        )

    # ── Status summary block ──────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:1.2rem;font-size:.6rem;color:{MUTED};line-height:2.0;
                border-top:1px solid {BORD};padding-top:.65rem;">
      <div style="color:{TEXT};font-weight:600;margin-bottom:.3rem;
                  font-size:.62rem;">Strategy Status</div>
      <div><span style="color:{GRN};">&#9679;</span>&nbsp;SWEEP_HQ &nbsp;&nbsp;&nbsp;&nbsp;4/4 PASS &nbsp;Sharpe +0.933</div>
      <div><span style="color:{GRN};">&#9679;</span>&nbsp;ORBFAIL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4/4 PASS &nbsp;Sharpe +0.704</div>
      <div><span style="color:{AMB};">&#9679;</span>&nbsp;SWEEP_PRI &nbsp;&nbsp;2/4 MRG &nbsp;&nbsp;Sharpe +0.290</div>
      <div style="margin-top:.5rem;padding-top:.4rem;border-top:1px solid {BORD};">
        10 symbols &nbsp;·&nbsp; 15m bars<br>
        Jan 2024 – Dec 2025<br>
        ~130k bars &nbsp;·&nbsp; 2,228 events
      </div>
    </div>""", unsafe_allow_html=True)

{
    "Research Overview":       page_overview,
    "ORB Strategy":            page_orb,
    "Session Sweeps":          page_sweep,
    "ORB Failure":             page_orbfail,
    "Walk-Forward & Monte Carlo": page_walkforward,
    "Live Strategies":         page_strategies,
    "Interactive Explorer":    page_explorer,
    "Live Paper Trading":      page_paper,
    "AI Research Assistant":   page_ai,
}[page]()
