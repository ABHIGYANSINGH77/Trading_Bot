#!/usr/bin/env python3
"""Quick demo - run a pairs trading backtest with sample data.

This script generates synthetic data so you can test the full pipeline
without needing IBKR or yfinance connectivity.

Usage:
    python demo_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.events import EventBus, EventType, MarketDataEvent
from core.portfolio import Portfolio
from strategies import PairsTradingStrategy, MACrossoverStrategy
from risk import RiskManager
from execution import SimulatedExecution


def generate_cointegrated_pair(
    n_days: int = 500,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> tuple:
    """Generate synthetic cointegrated price series for testing.

    Creates two price series where:
    - Both have upward drift (like real stocks)
    - They share a common factor (cointegration)
    - The spread mean-reverts
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Common factor (market)
    market_returns = np.random.normal(0.0003, 0.01, n_days)
    market = 100 * np.exp(np.cumsum(market_returns))

    # Idiosyncratic components
    noise_a = np.random.normal(0, 0.005, n_days)
    noise_b = np.random.normal(0, 0.005, n_days)

    # Mean-reverting spread component
    spread = np.zeros(n_days)
    for i in range(1, n_days):
        spread[i] = 0.95 * spread[i - 1] + np.random.normal(0, 0.5)

    # Construct prices
    beta = 1.3  # Hedge ratio
    price_a = 450 + (market - 100) * 4.5 + spread + np.cumsum(noise_a) * 10
    price_b = 200 + (market - 100) * 4.5 / beta + np.cumsum(noise_b) * 5

    # Build DataFrames
    def make_ohlcv(prices, dates):
        df = pd.DataFrame(index=dates)
        df["close"] = prices
        df["open"] = prices * (1 + np.random.normal(0, 0.002, len(prices)))
        df["high"] = prices * (1 + abs(np.random.normal(0, 0.005, len(prices))))
        df["low"] = prices * (1 - abs(np.random.normal(0, 0.005, len(prices))))
        df["volume"] = np.random.randint(1_000_000, 10_000_000, len(prices))
        return df

    df_a = make_ohlcv(price_a, dates)
    df_b = make_ohlcv(price_b, dates)

    return df_a, df_b, dates


def run_pairs_demo():
    """Demo: Pairs Trading Backtest."""
    print("=" * 60)
    print("  PAIRS TRADING BACKTEST DEMO")
    print("  (Using synthetic cointegrated data)")
    print("=" * 60)

    # Generate data
    df_spy, df_iwm, dates = generate_cointegrated_pair(n_days=500)
    print(f"\nData: {len(dates)} trading days, {dates[0].date()} to {dates[-1].date()}")
    print(f"SPY range: ${df_spy['close'].min():.2f} - ${df_spy['close'].max():.2f}")
    print(f"IWM range: ${df_iwm['close'].min():.2f} - ${df_iwm['close'].max():.2f}")

    # Setup components
    event_bus = EventBus()
    portfolio = Portfolio(event_bus, initial_capital=100_000.0)
    risk_manager = RiskManager(event_bus, portfolio, {
        "max_position_pct": 0.30,
        "max_daily_drawdown_pct": 0.03,
        "max_total_drawdown_pct": 0.15,
    })
    execution = SimulatedExecution(event_bus, slippage_pct=0.001)

    strategy = PairsTradingStrategy(event_bus, {
        "pair": ["SPY", "IWM"],
        "lookback_period": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 3.5,
    })

    # Replay data
    print("\nRunning backtest...")
    for i in range(len(dates)):
        ts = dates[i]

        # Emit SPY bar
        event_spy = MarketDataEvent(
            symbol="SPY",
            open=df_spy.iloc[i]["open"],
            high=df_spy.iloc[i]["high"],
            low=df_spy.iloc[i]["low"],
            close=df_spy.iloc[i]["close"],
            volume=df_spy.iloc[i]["volume"],
            bar_timestamp=ts,
            timestamp=ts,
        )
        event_bus.publish(event_spy)
        event_bus.process_all()

        # Emit IWM bar
        event_iwm = MarketDataEvent(
            symbol="IWM",
            open=df_iwm.iloc[i]["open"],
            high=df_iwm.iloc[i]["high"],
            low=df_iwm.iloc[i]["low"],
            close=df_iwm.iloc[i]["close"],
            volume=df_iwm.iloc[i]["volume"],
            bar_timestamp=ts,
            timestamp=ts,
        )
        event_bus.publish(event_iwm)
        event_bus.process_all()

        # Snapshot
        portfolio.take_snapshot(ts)

    # Results
    metrics = portfolio.calculate_metrics()
    trades = portfolio.get_trade_log_df()
    diagnostics = strategy.get_diagnostics()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Initial Capital:        ${100_000:>14,.2f}")
    print(f"  Final Value:            ${portfolio.total_value:>14,.2f}")
    print(f"  Total Return:           {metrics.get('total_return', 0):>14.2%}")
    print(f"  Annualized Return:      {metrics.get('annualized_return', 0):>14.2%}")
    print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>14.3f}")
    print(f"  Sortino Ratio:          {metrics.get('sortino_ratio', 0):>14.3f}")
    print(f"  Max Drawdown:           {metrics.get('max_drawdown', 0):>14.2%}")
    print(f"  Total Trades:           {len(trades):>14d}")
    print(f"  Total Commission:       ${portfolio.total_commission:>14,.2f}")
    print(f"  Win Rate:               {metrics.get('win_rate', 0):>14.2%}")
    print("=" * 60)

    print(f"\nStrategy diagnostics: {diagnostics}")

    if not trades.empty:
        print(f"\nLast 5 trades:")
        print(trades.tail().to_string(index=False))

    # Risk log
    risk_log = risk_manager.risk_log
    if risk_log:
        approved = sum(1 for r in risk_log if r["action"] == "APPROVED")
        rejected = sum(1 for r in risk_log if r["action"] == "REJECTED")
        print(f"\nRisk: {approved} orders approved, {rejected} rejected")

    return portfolio, metrics


def run_ma_demo():
    """Demo: MA Crossover Backtest."""
    print("\n" + "=" * 60)
    print("  MA CROSSOVER BACKTEST DEMO")
    print("  (Using synthetic trending data)")
    print("=" * 60)

    np.random.seed(123)
    n_days = 500
    dates = pd.bdate_range(start="2022-01-01", periods=n_days)

    # Generate trending price with regime changes
    prices = [400.0]
    trend = 0.001  # Uptrend
    for i in range(1, n_days):
        if i == 150:
            trend = -0.002  # Downtrend
        elif i == 300:
            trend = 0.0015  # Uptrend again
        elif i == 420:
            trend = -0.001  # Mild downtrend

        ret = trend + np.random.normal(0, 0.012)
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # Setup
    event_bus = EventBus()
    portfolio = Portfolio(event_bus, initial_capital=100_000.0)
    risk_manager = RiskManager(event_bus, portfolio, {"max_position_pct": 0.95})
    execution = SimulatedExecution(event_bus)

    strategy = MACrossoverStrategy(event_bus, {
        "symbols": ["SPY"],
        "fast_period": 20,
        "slow_period": 50,
        "vol_lookback": 30,
        "vol_threshold": 0.20,
    })

    # Replay
    print("Running backtest...")
    for i in range(n_days):
        event = MarketDataEvent(
            symbol="SPY",
            open=prices[i] * 0.999,
            high=prices[i] * 1.005,
            low=prices[i] * 0.995,
            close=prices[i],
            volume=5_000_000,
            bar_timestamp=dates[i],
            timestamp=dates[i],
        )
        event_bus.publish(event)
        event_bus.process_all()
        portfolio.take_snapshot(dates[i])

    metrics = portfolio.calculate_metrics()
    trades = portfolio.get_trade_log_df()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Buy & Hold Return:      {(prices[-1] / prices[0] - 1):>14.2%}")
    print(f"  Strategy Return:        {metrics.get('total_return', 0):>14.2%}")
    print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>14.3f}")
    print(f"  Max Drawdown:           {metrics.get('max_drawdown', 0):>14.2%}")
    print(f"  Total Trades:           {len(trades):>14d}")
    print("=" * 60)

    return portfolio, metrics


if __name__ == "__main__":
    print("QuantBot Demo - Testing core pipeline with synthetic data\n")

    portfolio1, metrics1 = run_pairs_demo()
    portfolio2, metrics2 = run_ma_demo()

    print("\n" + "=" * 60)
    print("  ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Connect to IBKR TWS (paper account) and update config/settings.yaml")
    print("  3. Run with real data: python main.py backtest --strategy pairs_trading")
    print("  4. Go live on paper: python main.py paper --strategy pairs_trading")
    print("  5. Monitor via dashboard (coming soon)")
    print()
    print("  To add your own strategy:")
    print("  - Create a new class in strategies/ inheriting from BaseStrategy")
    print("  - Implement calculate_signal() method")
    print("  - Register it in STRATEGY_REGISTRY")
    print("  - Add params to config/settings_template.yaml")
