# QuantBot - Algorithmic Trading Framework

A modular, extensible algorithmic trading framework built in Python for strategy development, backtesting, and live paper trading via Interactive Brokers.

## Architecture

```
trading_bot/
├── config/          # Configuration files & settings
├── core/            # Core engine: event system, portfolio, order management
├── data/            # Data ingestion: historical & realtime (IBKR, CSV, etc.)
├── strategies/      # Pluggable trading strategies
├── backtest/        # Backtesting engine (vectorized + event-driven)
├── execution/       # Order execution & broker connectivity (IBKR TWS)
├── risk/            # Risk management: position limits, drawdown, exposure
├── utils/           # Logging, metrics, helpers
├── tests/           # Unit & integration tests
├── logs/            # Runtime logs
└── notebooks/       # Jupyter notebooks for research & analysis
```

## Setup

### Prerequisites
- Python 3.10+
- Interactive Brokers TWS or IB Gateway (paper trading account)
- Ubuntu (tested on 24.04)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config/settings_template.yaml config/settings.yaml
# Edit settings.yaml with your IBKR connection details
```

### Running

```bash
# Run backtest
python -m backtest.engine --strategy pairs_trading --start 2023-01-01 --end 2024-01-01

# Run live paper trading
python main.py --mode paper --strategy pairs_trading

# Run dashboard
python -m utils.dashboard
```

## Strategies Included

1. **Mean Reversion Pairs Trading** - Cointegrated ETF pairs (SPY/IWM)
2. **Moving Average Crossover + Regime Filter** - Trend following with volatility regime detection

## Key Design Principles

- **Modularity**: Swap strategies, data sources, and brokers independently
- **Event-Driven**: Core engine uses an event bus for loose coupling
- **Realistic Simulation**: Transaction costs, slippage, and fill modeling
- **Risk-First**: Built-in position limits, drawdown stops, and exposure monitoring
