# QuantBot - Event-Driven Strategy Research & Paper Trading Framework

A research-grade systematic trading framework built in Python.
Implements a full quantitative pipeline: **hypothesis → statistical edge testing → walk-forward validation → Alpaca live paper trading → interactive Streamlit dashboard**.

Built as a portfolio project targeting quantitative trader / quant researcher roles at proprietary trading firms (SIG, Optiver, Flow Traders).

**Live dashboard:** [QuantEdge](https://quantedge.streamlit.app//)

---

## Strategies

Three validated intraday event-driven strategies on US large-cap equities (15-minute bars):

| Strategy | Event | Entry | Sharpe | Walk-Forward | Universe |
|---|---|---|---|---|---|
| **SWEEP_HQ** | PDL liquidity sweep + reversal | Sweep bar close (M1) | +0.933 | 4/4 PASS | NVDA only |
| **ORBFAIL_REGIME** | Failed ORB breakout fade on gap-up days | Failure bar close (M1) | +0.704 | 4/4 PASS | All except GOOG |
| **SWEEP_PRIMARY** | PDL sweep + prior-day up + near extreme | Confirmation bar close (M2) | +0.333 | 2/4 marginal | All 10 symbols |

All three run together inside `strategies/event_driven.py`. Combined overall: **2/5 WEAK** — strategies are valid research prototypes, not ready for live capital.

**Universe (10 symbols):** AAPL, NVDA, MSFT, AMZN, GOOG, TSLA, META, AMD, AVGO, NFLX

---

## Architecture

```
trading_bot/
----
main.py                    # CLI: backtest / validate / paper / alpaca / simulate
dashboard_app.py           # Streamlit dashboard (9 pages, hosted on Streamlit Cloud)
precompute_grid.py         # Pre-compute 30 param-grid backtests for Interactive Explorer
fetch_alpaca_data.py       # Download 15-min bars from Alpaca Markets free API
web_app.py                 # (alt) Plotly Dash app - run locally at localhost:8050
----
config/
  settings_template.yaml  # Copy to settings.yaml and fill in your values
  settings.yaml            # Gitignored - your local config (broker, universe, risk)
----
strategies/
  event_driven.py          # SWEEP_HQ + ORBFAIL_REGIME + SWEEP_PRIMARY (v10)
----
backtest/
  __init__.py              # BacktestEngine: event-driven, realistic slippage + fills
  validate.py              # Walk-forward + OOS + MC bootstrap + Markov MC + Sharpe CI
  dashboard.py             # Matplotlib static dashboard PNG helper
----
core/
  events.py                # EventBus: MarketData -> Signal -> Order -> Fill
  portfolio.py             # Portfolio: P&L, equity curve, drawdown, metrics
----
risk/
  __init__.py              # RiskManager: position sizing, daily DD limit, partial exits
----
execution/
  simulated.py             # SimulatedExecution: slippage + commission model
  ibkr.py                  # IBKRExecution: live order routing via ib_insync
----
data/
  __init__.py              # DataManager: CSV cache loader + IBKR historical data
  interactive_grid.json    # Pre-computed 30-backtest grid for dashboard Explorer tab
----
data/cache/                # Cached CSVs (gitignored - generate with fetch_alpaca_data.py)
  {SYMBOL}_{START}_{END}_15_mins.csv
----
tests/                     # Unit + integration tests (pytest)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ABHIGYANSINGH77/Trading_Bot.git
cd Trading_Bot

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Config

```bash
cp config/settings_template.yaml config/settings.yaml
# Edit settings.yaml - set initial_capital, commission, universe
```

### 3. Download historical data

Free paper account at [alpaca.markets](https://alpaca.markets) - no credit card needed.

```bash
export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxx"
export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

python3 fetch_alpaca_data.py
# Downloads ~2 years of 15-min bars for all 10 symbols to data/cache/
```

The script skips files already on disk - safe to re-run.

---

## Step-by-Step Workflow

### Backtest

```bash
python3 main.py backtest -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31
```

Outputs: `backtest_report.json`, `backtest_dashboard.png`, `backtest_trades.csv`

### Validation

Runs 5 rigorous checks: walk-forward consistency, OOS profitability, Monte Carlo bootstrap (1000 paths), Markov-chain regime MC, and Sharpe bootstrap CI + t-test.

```bash
python3 main.py validate -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31 --save-report
```

`--save-report` writes `validation_results.json` which the dashboard reads automatically.

### Alpaca Live Paper Trading (recommended - no IBKR needed)

Polls Alpaca every 60 seconds for new completed 15-min bars during market hours (09:30-16:00 ET), runs the full strategy pipeline, logs signals to `paper_sessions/`.

```bash
python3 main.py alpaca
```

The Streamlit dashboard auto-refreshes every 30 seconds and shows live session stats.

### IBKR Paper Trading

Requires Interactive Brokers TWS or IB Gateway running locally in paper mode.

```bash
python3 main.py paper -s event_driven
```

---

## Dashboard

### Streamlit (hosted - recommended for viewers)

Live at: **[QuantEdge](https://quantedge.streamlit.app/)**

No setup needed. All research results are pre-loaded from committed JSON files.

### Run locally

```bash
streamlit run dashboard_app.py
# Opens at http://localhost:8501
```

**9 pages:**

| Page | Description |
|---|---|
| Research Overview | Pipeline flowchart, research decisions, overall verdict |
| ORB Strategy | Phase 0-4 ORB research: raw edge, entry/exit/regime optimization |
| Session Sweeps | SWEEP_HQ research: PDH/PDL sweep decomposition, filters, entry/exit |
| ORB Failure | ORBFAIL_REGIME research: gap-up day fade, regime filter results |
| Walk-Forward & Monte Carlo | WF IS/OOS bars, OOS equity, i.i.d. MC histogram, Markov fan chart, Sharpe CI |
| Live Strategies | Current strategy parameters, logic summary, per-strategy WF verdict |
| Interactive Explorer | Slider-based parameter explorer over 30 pre-computed backtests (gap × RVOL grid) |
| Live Paper Trading | Alpaca session monitor: P&L, trade log, equity curve (auto-refreshes every 30s) |
| AI Research Assistant | Ask questions about the strategies using Claude API |

### Pre-compute parameter grid (for Interactive Explorer)

The 30-combination grid is already committed (`data/interactive_grid.json`). To regenerate:

```bash
python3 precompute_grid.py           # full 30-combination grid (~25 min)
python3 precompute_grid.py --dry-run # 4-combination test (~3 min)
```

---

## Validation Results (10 symbols, Jan 2024 - Dec 2025)

| Check | Result |
|---|---|
| Walk-forward windows profitable | 0/3 |
| OOS return | -0.86% |
| OOS Sharpe | -0.74 |
| Monte Carlo P(profit) | 1.7% |
| Sharpe CI (95%) | -1.202 [-2.19 to -0.32] |
| **Overall score** | **2/5 WEAK** |

SWEEP_HQ and ORBFAIL_REGIME individually pass 4/4 walk-forward windows at the strategy level (Sharpe +0.933 / +0.704). The combined portfolio underperforms due to commission drag at $10k capital and slippage asymmetry tightening effective R:R on SWEEP_PRIMARY.

---

## Research Pipeline

Each event type goes through the same 5-phase pipeline. Drop on no Phase 0 edge.

```
Phase 0   - Raw edge test (does the event have any edge at all?)
Phase 0.5 - Edge decomposition (which sub-conditions drive the edge?)
Phase 1   - Entry method comparison (sweep_close vs conf_close vs limit vs retest)
Phase 2   - Exit method comparison (EOD vs VWAP vs fixed 1R vs ATR trail)
Phase 3   - Regime filter (gap, prior-day direction, volatility, time-of-day)
Phase 4   - Walk-forward validation (rolling windows, IS/OOS split)
```

**Events tested:**
- ORB (Opening Range Breakout) - raw edge -0.046R unfiltered; best filtered +0.060R
- Session Sweep + Rejection (PDH/PDL) - SWEEP_HQ: +0.933 Sharpe, validated
- ORB Failure (Fade) - ORBFAIL_REGIME: +0.704 Sharpe, validated
- Volatility Compression - DROPPED at Phase 0 (no edge, -0.007R, n=209)

---

## Key Design Decisions

**Bar-close entry over limit orders**
Limit entries suffer adverse selection on ORB trades - fills concentrate on losers. Bar-close (market order at next open) avoids this. Sweep strategies show the opposite: limits work because sweeps retest the level.

**No look-ahead bias**
All filters use only information available at bar close:
- VWAP computed from session-open bars seen so far
- Prior-day stats from yesterday's close only
- RVOL computed from prior 14 sessions' first-bar volumes (no current-day vol)
- Rolling regime labels updated from closed bars only

**Event-driven engine over vectorized**
Vectorized backtesting cannot correctly model:
- Partial exits (50% close at 1R, then ATR trail on remainder)
- Intraday fills across symbols at different timestamps
- Risk manager blocking trades based on live portfolio state

**Split-aware data handling**
NVDA (10:1 split Jun 2024) and AVGO (10:1 split Jul 2024) have price discontinuities in the CSV data. The engine uses a global 80% median filter to detect and skip pre-split bars automatically.

---

## Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Download data
export ALPACA_API_KEY="..."
export ALPACA_API_SECRET="..."
python3 fetch_alpaca_data.py

# Backtest
python3 main.py backtest -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31

# Validate + save results for dashboard
python3 main.py validate -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31 --save-report

# Alpaca live paper trading (no IBKR needed)
python3 main.py alpaca

# IBKR paper trading (requires TWS running)
python3 main.py paper -s event_driven

# Streamlit dashboard
streamlit run dashboard_app.py

# Plotly Dash dashboard (alternative)
python3 web_app.py

# Pre-compute parameter grid
python3 precompute_grid.py --dry-run   # quick 4-combo test
python3 precompute_grid.py             # full 30-combo grid

# Tests
python3 -m pytest tests/ -v
```

---

## Dependencies

```
Core:   numpy, pandas, scipy, statsmodels
Data:   requests (Alpaca REST API), ib_insync (IBKR)
Viz:    matplotlib, plotly, dash, streamlit
ML:     scikit-learn (regime detection)
CLI:    click, pyyaml, loguru
```

For the hosted dashboard only (no IBKR/ML needed):
```bash
pip install -r requirements_dashboard.txt
```

---

## Disclaimer

This project is for educational and research purposes only. Past backtest performance does not guarantee future results. The overall strategy scores 2/5 WEAK and is not recommended for live trading without further validation. Paper trading only.
