# QuantBot — Event-Driven Strategy Research & Paper Trading Framework

A rigorous, research-grade systematic trading framework built in Python.
Implements a full quantitative pipeline: **hypothesis → statistical edge testing → walk-forward validation → paper trading → interactive web dashboard**.

Designed as a portfolio project targeting roles at proprietary trading firms (SIG, Optiver, Flow Traders).

---

## Strategies

Three validated intraday strategies on US large-cap equities (15-minute bars):

| Strategy | Setup | Sharpe | WF Result | Symbols |
|---|---|---|---|---|
| **SWEEP_HQ** | PDL liquidity sweep → reversal (M1 entry, fixed +1R target) | +0.933 | 4/4 PASS | NVDA only |
| **ORBFAIL_REGIME** | Failed ORB breakout → fade (M1 entry, 50% @ 1R + ATR trail) | +0.704 | 4/4 PASS | All except GOOG |
| **SWEEP_PRIMARY** | PDL sweep → EOD hold (M2 conf-bar entry) | +0.333 | 2/4 marginal | All 10 symbols |

All three run together inside `strategies/event_driven.py`.

**Universe:** AAPL, NVDA, MSFT, AMZN, GOOG, TSLA, META, AMD, AVGO, NFLX

---

## Architecture

```
trading_bot/
├── main.py                    # CLI entry point (backtest / validate / paper / simulate)
├── web_app.py                 # Plotly Dash interactive web dashboard (localhost:8050)
├── fetch_alpaca_data.py       # Download 15-min bar data from Alpaca Markets API
│
├── config/
│   └── settings.yaml          # Broker, universe, risk, strategy config
│
├── strategies/
│   └── event_driven.py        # SWEEP_HQ + ORBFAIL_REGIME + SWEEP_PRIMARY (v10)
│
├── backtest/
│   ├── __init__.py            # BacktestEngine: event-driven, realistic fills
│   ├── validate.py            # Walk-forward + OOS + Monte Carlo + Markov MC + Sharpe CI
│   └── dashboard.py           # Matplotlib static dashboard PNGs
│
├── core/
│   ├── events.py              # EventBus: MarketData → Signal → Order → Fill
│   └── portfolio.py           # Portfolio: P&L, equity curve, drawdown, metrics
│
├── risk/
│   └── __init__.py            # RiskManager: position sizing, daily DD limit, partial exits
│
├── execution/
│   ├── simulated.py           # SimulatedExecution: slippage + commission model
│   └── ibkr.py                # IBKRExecution: live order routing via ib_insync
│
├── data/
│   └── __init__.py            # DataManager: CSV cache + IBKR historical data
│
├── data/cache/                # Cached CSVs (never committed to git)
│   └── {SYMBOL}_{START}_{END}_15_mins.csv
│
└── tests/                     # Unit + integration tests (pytest)
```

---

## Setup

### Prerequisites

- Python 3.10+
- Linux/macOS (tested on Ubuntu 24.04)
- Free [Alpaca Markets](https://alpaca.markets) paper account (for data download)
- Interactive Brokers TWS or IB Gateway (for paper trading only)

### Installation

```bash
git clone <your-repo-url>
cd trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/settings.yaml` with your IBKR paper account details:

```yaml
broker:
  host: "127.0.0.1"
  port: 7497          # TWS paper: 7497 | IB Gateway paper: 4002
  client_id: 1
  account: "DU1234567"   # your paper account ID

backtest:
  initial_capital: 10000.0
```

---

## Step 1 — Download Historical Data

Data is fetched from Alpaca Markets (free paper account, no credit card needed).

```bash
# Set your Alpaca API keys (get them from alpaca.markets → Paper Trading → API Keys)
export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxx"
export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Download 2 years of 15-min bars for all 10 symbols (~2024-01-01 to 2025-12-31)
python3 fetch_alpaca_data.py

# Custom date range
python3 fetch_alpaca_data.py 2024-06-01 2025-12-31
```

Files are saved to `data/cache/` as:
```
data/cache/NVDA_2024-01-01_2025-12-31_15_mins.csv
data/cache/AAPL_2024-01-01_2025-12-31_15_mins.csv
...
```

The script skips files already on disk — safe to re-run.

---

## Step 2 — Run Backtest

```bash
# Full 2-year backtest on all 10 symbols (uses local cache, IBKR not needed)
python3 main.py backtest -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31

# Shorter period
python3 main.py backtest -s event_driven -i 15m -d ibkr \
    --start 2025-01-01 --end 2025-12-31
```

**Outputs:**
| File | Description |
|---|---|
| `backtest_report.json` | Full report for web dashboard |
| `backtest_dashboard.png` | 6-panel static matplotlib chart |
| `backtest_chart.png` | Equity + drawdown chart |
| `backtest_trades.csv` | Per-trade log |

---

## Step 3 — Run Validation

Implements 5 rigorous validation checks:
1. Walk-forward consistency (3 rolling windows)
2. Out-of-sample profitability (final 20% holdout)
3. Monte Carlo bootstrap (1000 paths)
4. Markov-chain regime Monte Carlo
5. Sharpe bootstrap confidence interval + t-test

```bash
# Validate with full 2-year dataset + save results for web dashboard
python3 main.py validate -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31 \
    --save-report

# More walk-forward windows
python3 main.py validate -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31 \
    --windows 5 --mc 2000 --save-report
```

**Outputs:**
| File | Description |
|---|---|
| `validation_results.json` | Full results for web dashboard |
| `validation_dashboard.png` | 6-panel static validation chart |

---

## Step 4 — Launch Web Dashboard

```bash
python3 web_app.py
# Open: http://localhost:8050
```

The dashboard auto-refreshes every 30 seconds — useful during paper trading sessions.

**Tabs:**

**Overview** — Portfolio headline metrics, zoomable equity curve with drawdown shading, strategy P&L breakdown, return distribution histogram.

**Backtest** — Interactive 6-panel analysis with file selector (load any `backtest*.json`):
equity curve with trade markers, drawdown fill, per-trade P&L bars with cumulative overlay, monthly P&L, return histogram, metrics scorecard.

**Validation** — Walk-forward IS/OOS grouped bars, OOS cumulative P&L, standard MC return histogram, Markov-chain MC fan chart (50 paths + 5th/95th CI band), regime transition heatmap, Sharpe bootstrap CI chart, verdict scorecard (score/5).

---

## Step 5 — Paper Trading (IBKR)

Requires Interactive Brokers TWS or IB Gateway running in paper mode.

```bash
# Start TWS / IB Gateway in paper mode first, then:
python3 main.py paper -s event_driven

# Custom poll interval (seconds between market data snapshots)
python3 main.py paper -s event_driven -i 15

# Simulate without IBKR (Yahoo Finance, no real orders)
python3 main.py simulate -s event_driven -i 60
```

The paper session exports a timestamped JSON report on exit:
```
paper_report_20250301_093012.json
```
Load it in the Backtest tab of the web dashboard using the file selector dropdown.

---

## Validation Results (5-symbol, Jan 2024 – Dec 2025)

| Strategy | Walk-Forward | OOS Return | Sharpe | Verdict |
|---|---|---|---|---|
| SWEEP_HQ | 4/4 windows positive | — | +0.933 | **PASS** |
| ORBFAIL_REGIME | 4/4 windows positive | — | +0.704 | **PASS** |
| SWEEP_PRIMARY | 2/4 windows positive | — | +0.333 | marginal |

Statistical tests run on combined trade pool:
- Bootstrap Sharpe CI (1000 resamples): shows whether edge CI lower bound > 0
- t-test on per-trade returns: H₀ = zero mean
- Markov-chain MC: regime-conditional drawdown estimate

---

## Research Pipeline

The strategies were built through a rigorous multi-phase research pipeline:

```
Phase 0  — Raw edge test (does the event have any edge at all?)
Phase 0.5 — Edge decomposition (which sub-conditions drive the edge?)
Phase 1  — Entry method comparison (M1 sweep_close vs M2 conf_close vs limit vs retest)
Phase 2  — Exit method comparison (EOD vs VWAP vs fixed 1R vs ATR trail)
Phase 3  — Regime filter (gap, prior-day direction, volatility, time-of-day)
Phase 4  — Walk-forward validation (anchored windows, IS/OOS split)
```

**Events tested:**
- ORB (Opening Range Breakout) — raw edge is -0.046R unfiltered; marginal filtered
- Session Sweep + Rejection (PDH/PDL) — SWEEP_HQ: +0.933 Sharpe validated
- ORB Failure (Fade) — ORBFAIL_REGIME: +0.704 Sharpe validated
- Volatility Compression — DROPPED at Phase 0 (no edge, -0.007R)

Research scripts: `phase0_raw_edge.py`, `sweep_phase0_raw_edge.py`, `orbfail_phase0_raw_edge.py`, etc.

---

## Key Technical Decisions

**Why bar-close entry (not limit)?**
Limit entries suffer adverse selection on ORB trades — fills concentrate on losers.
Bar-close (market order at next bar) avoids this. Sweep strategies show the opposite: limits work because sweeps retest the level.

**Why no look-ahead bias?**
All filters use only information available at bar close:
- VWAP computed from session-open bars seen so far
- Prior-day stats from yesterday's close only
- Rolling volatility from past trades only

**Why event-driven (not vectorized)?**
Vectorized backtesting can't correctly model:
- Partial position exits (50% close at 1R, then trail remaining)
- Intraday order fills across symbols at different timestamps
- Risk manager blocking trades based on live portfolio state

---

## Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Download data (Alpaca keys required)
export ALPACA_API_KEY="..."
export ALPACA_API_SECRET="..."
python3 fetch_alpaca_data.py

# Backtest
python3 main.py backtest -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31

# Validate + save for dashboard
python3 main.py validate -s event_driven -i 15m -d ibkr \
    --start 2024-01-01 --end 2025-12-31 --save-report

# Web dashboard
python3 web_app.py

# Paper trading (requires IBKR TWS running)
python3 main.py paper -s event_driven

# Run tests
python3 -m pytest tests/ -v

# Run a specific phase research script
python3 sweep_phase0_raw_edge.py
python3 orbfail_phase0_raw_edge.py
```

---

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `dash`, `dash-bootstrap-components`
Data: `requests` (Alpaca API), `ib_insync` / `ib_async` (IBKR)
CLI: `click`, `pyyaml`

```bash
pip install -r requirements.txt
```

---

## Disclaimer

This project is for educational and research purposes. Past backtest performance does not guarantee future results. Paper trading only — do not use with real capital without independent review. This is project aims to stress test ideas and how its implemented and validated.
