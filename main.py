#!/usr/bin/env python3
"""QuantBot - Main entry point.

Usage:
    # Run backtest on historical data (yfinance, no IBKR needed)
    python main.py backtest --strategy pairs_trading
    python main.py backtest --strategy pairs_trading --start 2023-01-01 --end 2024-06-01

    # Run live simulation using Yahoo Finance (no IBKR needed!)
    python main.py simulate --strategy pairs_trading
    python main.py simulate --strategy ma_crossover --interval 60

    # Run paper trading via IBKR (requires TWS + market data subscription)
    python main.py paper --strategy pairs_trading
"""

import sys
import os
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from core.events import EventBus, EventType, Event, MarketDataEvent
from core.portfolio import Portfolio
from strategies import create_strategy
from risk import RiskManager
from execution import SimulatedExecution, IBKRExecution
from data import DataManager
from backtest import BacktestEngine
from utils import setup_logging, logger


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: dict, strategy_name: str, strategy_params: dict = None):
        self.config = config
        self.strategy_name = strategy_name
        self._running = False

        setup_logging(config)
        logger.info(f"Initializing QuantBot with strategy: {strategy_name}")

        # Core components
        self.event_bus = EventBus()
        self.portfolio = Portfolio(
            self.event_bus,
            initial_capital=config.get("backtest", {}).get("initial_capital", 100_000.0),
        )
        self.risk_manager = RiskManager(
            self.event_bus,
            self.portfolio,
            config.get("risk", {}),
        )
        self.data_manager = DataManager(config)

        # Strategy
        strat_params = config.get("strategies", {}).get(strategy_name, {})
        if strategy_params:
            strat_params.update(strategy_params)
        self.strategy = create_strategy(strategy_name, self.event_bus, strat_params)

        # Execution (set based on mode)
        self.execution = None

        # System event handler
        self.event_bus.subscribe(EventType.RISK_BREACH, self._on_risk_breach)

    def _on_risk_breach(self, event: Event):
        logger.warning(f"RISK BREACH: {event.data.get('reason', 'Unknown')}")
        if event.data.get("action") == "HALT":
            logger.error("Trading HALTED due to risk breach!")

    def _get_strategy_symbols(self) -> set:
        """Extract symbols from strategy params."""
        symbols = set()
        if "pair" in self.strategy.params:
            symbols.update(self.strategy.params["pair"])
        if "symbols" in self.strategy.params:
            symbols.update(self.strategy.params["symbols"])
        if not symbols:
            symbols = set(self.config.get("trading", {}).get("universe", ["AAPL"]))
        return symbols

    # ================================================================
    #  MODE 1: SIMULATE - Live trading sim via Yahoo Finance
    #  No IBKR needed! Polls real stock prices and simulates trades.
    # ================================================================

    def run_simulate(self, poll_interval: int = 30, warmup_days: int = 90):
        """Run live trading simulation using Yahoo Finance data.

        This mode:
        1. Fetches historical data to warm up strategy indicators
        2. Polls Yahoo Finance for latest prices every poll_interval seconds
        3. Runs full strategy -> risk -> execution pipeline
        4. Simulates fills with realistic slippage and commissions

        Works with any stock: AAPL, NVDA, TSLA, MSFT, GOOG, etc.
        No IBKR or market data subscription required.
        """
        import yfinance as yf

        symbols = self._get_strategy_symbols()
        bt_config = self.config.get("backtest", {})

        # Set up simulated execution
        self.execution = SimulatedExecution(
            self.event_bus,
            commission_per_share=bt_config.get("commission_per_share", 0.005),
            min_commission=bt_config.get("min_commission", 1.0),
            slippage_pct=bt_config.get("slippage_pct", 0.001),
        )

        logger.info(f"Starting Yahoo Finance simulation for {symbols}")
        logger.info(f"Poll interval: {poll_interval}s | Warmup: {warmup_days} days")
        logger.info(f"Capital: ${self.portfolio.initial_capital:,.2f}")

        # --- Phase 1: Warm up with historical data ---
        logger.info("")
        logger.info("=" * 50)
        logger.info("  PHASE 1: Historical warmup")
        logger.info("=" * 50)

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=warmup_days + 30)).strftime("%Y-%m-%d")

        for symbol in sorted(symbols):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval="1d")

                if hist.empty:
                    logger.error(f"No data for {symbol}. Check ticker symbol.")
                    return

                logger.info(f"  {symbol}: {len(hist)} daily bars ({hist.index[0].date()} -> {hist.index[-1].date()}) "
                            f"| Last: ${hist['Close'].iloc[-1]:.2f}")

                # Feed historical bars to warm up indicators
                for timestamp, row in hist.iterrows():
                    event = MarketDataEvent(
                        symbol=symbol,
                        open=row["Open"],
                        high=row["High"],
                        low=row["Low"],
                        close=row["Close"],
                        volume=row.get("Volume", 0),
                        bar_timestamp=timestamp.to_pydatetime(),
                        timestamp=timestamp.to_pydatetime(),
                    )
                    self.event_bus.publish(event)
                    self.event_bus.process_all()

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                return

        self.portfolio.take_snapshot()
        diagnostics = self.strategy.get_diagnostics()
        logger.info(f"\n  Strategy state after warmup: {diagnostics}")

        # --- Phase 2: Live polling loop ---
        logger.info("")
        logger.info("=" * 50)
        logger.info("  PHASE 2: Live simulation")
        logger.info("=" * 50)
        logger.info(f"Polling {symbols} every {poll_interval}s. Ctrl+C to stop.\n")

        self._running = True
        poll_count = 0
        last_prices = {}

        while self._running:
            poll_count += 1
            now = datetime.now()

            for symbol in sorted(symbols):
                try:
                    ticker = yf.Ticker(symbol)
                    live = ticker.history(period="1d", interval="1m")

                    if live.empty:
                        continue

                    latest = live.iloc[-1]
                    price = float(latest["Close"])

                    # Skip if price unchanged
                    prev = last_prices.get(symbol)
                    if prev is not None and abs(price - prev) < 0.001:
                        continue

                    last_prices[symbol] = price

                    event = MarketDataEvent(
                        symbol=symbol,
                        open=float(latest["Open"]),
                        high=float(latest["High"]),
                        low=float(latest["Low"]),
                        close=price,
                        volume=float(latest.get("Volume", 0)),
                        bar_timestamp=now,
                        timestamp=now,
                    )
                    self.event_bus.publish(event)

                except Exception as e:
                    logger.warning(f"Fetch failed for {symbol}: {e}")

            # Process events
            self.event_bus.process_all()
            self.portfolio.take_snapshot(now)

            # --- Log status ---
            risk_status = self.risk_manager.get_status()

            price_parts = [f"{s}: ${p:.2f}" for s, p in sorted(last_prices.items())]
            price_str = " | ".join(price_parts)

            pos_parts = [
                f"{s}:{p.quantity}@${p.current_price:.2f}(PnL:${p.unrealized_pnl:+,.2f})"
                for s, p in self.portfolio.active_positions.items()
            ]
            pos_str = ", ".join(pos_parts) if pos_parts else "flat"

            logger.info(
                f"[#{poll_count:>4}] {price_str} | "
                f"Value: ${self.portfolio.total_value:,.2f} ({self.portfolio.total_return:+.2%}) | "
                f"Pos: [{pos_str}] | "
                f"DD: {risk_status['total_drawdown']:.2%}"
            )

            # Log strategy diagnostics (compact)
            diag = self.strategy.get_diagnostics()
            if isinstance(diag, dict):
                z = diag.get("z_score")
                if z is not None:
                    logger.info(f"       z={z:.3f} | hedge={diag.get('hedge_ratio', '?'):.4f} | "
                                f"side={diag.get('position_side', 'none')}")

            # Wait
            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                break

        self._print_session_summary()

    # ================================================================
    #  MODE 2: PAPER - Live trading via IBKR TWS (delayed data, free)
    # ================================================================

    def run_paper(self, poll_interval: int = 10, warmup_days: int = 90):
        """Run paper trading connected to IBKR using delayed data (free).

        Uses reqMarketDataType(3) for delayed data — no subscription needed.
        Fetches historical bars for warmup, then polls delayed snapshots.
        Places REAL orders on the paper account.
        """
        broker_config = self.config.get("broker", {})
        host = broker_config.get("host", "127.0.0.1")
        port = broker_config.get("port", 7497)
        client_id = broker_config.get("client_id", 1)
        asset_type = self.config.get("trading", {}).get("asset_type", "STK")

        try:
            from ib_async import IB, Stock, Forex, MarketOrder, LimitOrder, util
        except ImportError:
            try:
                from ib_insync import IB, Stock, Forex, MarketOrder, LimitOrder, util
            except ImportError:
                logger.error("No IBKR library. pip install ib_insync")
                return

        logger.info(f"Connecting to IBKR TWS at {host}:{port} (clientId={client_id})...")
        ib = IB()
        tickers = {}
        contracts = {}

        try:
            ib.connect(host, port, clientId=client_id)
            accounts = ib.managedAccounts()
            logger.info(f"Connected. Account: {accounts}")

            # USE DELAYED DATA (type 3) — free, no subscription required
            # Type 1 = live (paid), 3 = delayed (free), 4 = delayed-frozen
            ib.reqMarketDataType(3)
            logger.info("Using DELAYED market data (free, ~15 min delay)")

            # Wire execution — orders go to IBKR paper account
            self.execution = IBKRExecution(
                self.event_bus, host=host, port=port,
                client_id=client_id, asset_type=asset_type,
            )
            self.execution._ib = ib
            ib.orderStatusEvent += self.execution._on_order_status
            ib.execDetailsEvent += self.execution._on_exec_details

            symbols = self._get_strategy_symbols()

            # Build contracts
            contracts = {}
            for symbol in symbols:
                if asset_type == "FX":
                    c = Forex(symbol)
                else:
                    c = Stock(symbol, "SMART", "USD")
                ib.qualifyContracts(c)
                contracts[symbol] = c
                logger.info(f"Qualified contract: {symbol} -> {c}")

            # --- Phase 1: Warmup with IBKR historical data ---
            logger.info("")
            logger.info("=" * 50)
            logger.info("  PHASE 1: Historical warmup via IBKR")
            logger.info("=" * 50)

            for symbol in sorted(symbols):
                c = contracts[symbol]
                logger.info(f"  Fetching history for {symbol}...")

                try:
                    bars = ib.reqHistoricalData(
                        c,
                        endDateTime="",
                        durationStr=f"{warmup_days} D",
                        barSizeSetting="1 day",
                        whatToShow="MIDPOINT" if asset_type == "FX" else "TRADES",
                        useRTH=True,
                        formatDate=1,
                    )

                    if not bars:
                        logger.warning(f"  No historical data for {symbol}, trying ADJUSTED_LAST...")
                        bars = ib.reqHistoricalData(
                            c, endDateTime="", durationStr=f"{warmup_days} D",
                            barSizeSetting="1 day", whatToShow="ADJUSTED_LAST",
                            useRTH=True, formatDate=1,
                        )

                    if bars:
                        logger.info(f"  {symbol}: {len(bars)} bars loaded")
                        for bar in bars:
                            event = MarketDataEvent(
                                symbol=symbol,
                                open=bar.open, high=bar.high,
                                low=bar.low, close=bar.close,
                                volume=getattr(bar, 'volume', 0),
                                bar_timestamp=bar.date if hasattr(bar.date, 'timestamp') else datetime.now(),
                                timestamp=bar.date if hasattr(bar.date, 'timestamp') else datetime.now(),
                            )
                            self.event_bus.publish(event)
                            self.event_bus.process_all()
                    else:
                        logger.error(f"  No data at all for {symbol}. Check if market was open.")

                except Exception as e:
                    logger.error(f"  Historical fetch failed for {symbol}: {e}")

            self.portfolio.take_snapshot()
            logger.info(f"\n  Strategy state: {self.strategy.get_diagnostics()}")

            # --- Phase 2: Live delayed polling ---
            logger.info("")
            logger.info("=" * 50)
            logger.info("  PHASE 2: Live paper trading (delayed data)")
            logger.info("=" * 50)

            # Request market data snapshots for each symbol
            tickers = {}
            for symbol in symbols:
                c = contracts[symbol]
                ticker = ib.reqMktData(c, "", False, False)
                tickers[symbol] = ticker
                logger.info(f"  Requested delayed data for {symbol}")

            # Give IBKR a moment to start sending data
            ib.sleep(2)

            self._running = True
            logger.info(f"\nPaper trading live. Polling every {poll_interval}s. Ctrl+C to stop.\n")
            poll_count = 0
            last_prices = {}

            while self._running:
                poll_count += 1
                now = datetime.now()

                # Let IBKR process incoming data
                ib.sleep(0.5)

                for symbol in sorted(symbols):
                    ticker = tickers[symbol]

                    # Get best available price from delayed snapshot
                    price = None
                    if ticker.last and ticker.last > 0:
                        price = ticker.last
                    elif ticker.close and ticker.close > 0:
                        price = ticker.close
                    elif ticker.bid and ticker.bid > 0 and ticker.ask and ticker.ask > 0:
                        price = (ticker.bid + ticker.ask) / 2.0

                    if price is None or price <= 0:
                        continue

                    # Skip if unchanged
                    prev = last_prices.get(symbol)
                    if prev is not None and abs(price - prev) < 0.001:
                        continue

                    last_prices[symbol] = price

                    high = ticker.high if ticker.high and ticker.high > 0 else price
                    low = ticker.low if ticker.low and ticker.low > 0 else price
                    opn = ticker.open if ticker.open and ticker.open > 0 else price
                    vol = ticker.volume if ticker.volume else 0

                    event = MarketDataEvent(
                        symbol=symbol,
                        open=opn, high=high, low=low, close=price,
                        volume=vol, bar_timestamp=now, timestamp=now,
                    )
                    self.event_bus.publish(event)

                # Process all events
                self.event_bus.process_all()
                self.portfolio.take_snapshot(now)

                # Log status
                risk_status = self.risk_manager.get_status()
                price_parts = [f"{s}: ${p:.2f}" for s, p in sorted(last_prices.items())]
                pos_parts = [
                    f"{s}:{p.quantity}@${p.current_price:.2f}(${p.unrealized_pnl:+,.2f})"
                    for s, p in self.portfolio.active_positions.items()
                ]

                logger.info(
                    f"[#{poll_count:>4}] {' | '.join(price_parts)} | "
                    f"Value: ${self.portfolio.total_value:,.2f} ({self.portfolio.total_return:+.2%}) | "
                    f"Pos: [{', '.join(pos_parts) or 'flat'}]"
                )

                diag = self.strategy.get_diagnostics()
                if isinstance(diag, dict):
                    z = diag.get("z_score")
                    if z is not None:
                        logger.info(f"       z={z:.3f} | hedge={diag.get('hedge_ratio', '?'):.4f} | "
                                    f"side={diag.get('position_side', 'none')}")

                # Wait
                try:
                    remaining = poll_interval - 0.5  # account for ib.sleep above
                    if remaining > 0:
                        time.sleep(remaining)
                except KeyboardInterrupt:
                    break

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if ib.isConnected():
                # Cancel market data
                for symbol, ticker in tickers.items():
                    try:
                        ib.cancelMktData(contracts[symbol])
                    except Exception:
                        pass
                ib.disconnect()
                logger.info("Disconnected from IBKR.")
            self._print_session_summary()

    # ================================================================
    #  MODE 3: BACKTEST
    # ================================================================

    def run_backtest(self, start: str = None, end: str = None):
        """Run backtest on historical data."""
        engine = BacktestEngine(self.config)
        engine.add_strategy(self.strategy_name)
        results = engine.run(start=start, end=end)
        engine.print_results()
        chart_path = engine.plot_results()
        engine.export_trades()
        return results, chart_path

    # ================================================================
    #  Helpers
    # ================================================================

    def _log_status(self):
        risk_status = self.risk_manager.get_status()
        pos_parts = [
            f"{s}:{p.quantity}@${p.current_price:.2f}"
            for s, p in self.portfolio.active_positions.items()
        ]
        logger.info(
            f"Portfolio: ${self.portfolio.total_value:,.2f} ({self.portfolio.total_return:+.2%}) | "
            f"Pos: [{', '.join(pos_parts) or 'flat'}] | "
            f"DD: {risk_status['total_drawdown']:.2%}"
        )

    def _print_session_summary(self):
        metrics = self.portfolio.calculate_metrics()
        trades_df = self.portfolio.get_trade_log_df()

        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Initial Capital:  ${self.portfolio.initial_capital:>14,.2f}")
        print(f"  Final Value:      ${self.portfolio.total_value:>14,.2f}")
        print(f"  Total Return:      {self.portfolio.total_return:>14.2%}")
        print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):>14.2%}")
        print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>14.3f}")
        print(f"  Total Trades:      {len(self.portfolio.trade_log):>14d}")
        print(f"  Total Commission: ${self.portfolio.total_commission:>14,.2f}")

        if not trades_df.empty:
            print(f"\n  Recent trades:")
            for _, t in trades_df.tail(5).iterrows():
                print(f"    {t['side']:>4} {t['quantity']:>5} {t['symbol']:<6} "
                      f"@ ${t['price']:>10,.2f}  ({t['strategy']})")

        active = self.portfolio.active_positions
        if active:
            print(f"\n  Open positions:")
            for sym, pos in active.items():
                print(f"    {sym:<6} {pos.quantity:>6} shares | "
                      f"avg ${pos.avg_entry_price:.2f} | "
                      f"PnL: ${pos.total_pnl:+,.2f}")

        print("=" * 60)

    def stop(self):
        self._running = False


# ============================
#  CLI Commands
# ============================

@click.group()
def cli():
    """QuantBot - Algorithmic Trading Framework"""
    pass


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name (bos, ma_crossover, pairs_trading)")
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.option("--interval", "-i", default="1d", help="Bar interval: 5m, 15m, 1h, 1d (default: 1d)")
@click.option("--source", "-d", default="yfinance", type=click.Choice(["yfinance", "ibkr"]),
              help="Data source: yfinance (free, 60d intraday) or ibkr (years of intraday, needs TWS)")
@click.option("--config", "-c", default=None, help="Config file path")
def backtest(strategy, start, end, interval, source, config):
    """Run backtest on historical data.

    \b
    Examples:
        # Yahoo Finance (free, last 60 days for intraday)
        python main.py backtest -s bos -i 15m

        # IBKR (years of intraday, needs TWS/Gateway running)
        python main.py backtest -s bos -i 15m -d ibkr --start 2024-06-01 --end 2024-12-31
        python main.py backtest -s bos -i 5m -d ibkr --start 2025-01-01 --end 2025-06-01
        python main.py backtest -s bos -i 1h -d ibkr --start 2023-01-01 --end 2024-01-01

        # Daily bars (either source, any date range)
        python main.py backtest -s bos --start 2023-01-01 --end 2024-01-01
    """
    cfg = load_config(config)

    engine = BacktestEngine(cfg)
    engine.add_strategy(strategy)

    # If using IBKR, connect first
    if source == "ibkr":
        ibkr_cfg = cfg.get("ibkr", {})
        engine.data_manager.add_ibkr_source(
            host=ibkr_cfg.get("host", "127.0.0.1"),
            port=ibkr_cfg.get("port", 7497),
            client_id=ibkr_cfg.get("client_id", 10),  # Use different client ID for backtest
        )

    results = engine.run(start=start, end=end, interval=interval, data_source=source)

    if not results:
        print("Backtest failed. Check data source and parameters.")
        return

    engine.print_results()
    chart_path = engine.plot_results()
    engine.export_trades()
    report_path = engine.export_report()

    print(f"\n  To visualize this backtest:")
    print(f"  python visualizer.py --report {report_path}")
    print()


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name")
@click.option("--interval", "-i", default=30, help="Poll interval seconds (default: 30)")
@click.option("--warmup", "-w", default=90, help="Warmup days of history (default: 90)")
@click.option("--config", "-c", default=None, help="Config file path")
def simulate(strategy, interval, warmup, config):
    """Live simulation via Yahoo Finance — no IBKR needed.

    Polls real stock prices and trades with simulated execution.

    \b
    Examples:
        python main.py simulate -s pairs_trading
        python main.py simulate -s pairs_trading -i 60
        python main.py simulate -s ma_crossover -w 120
    """
    cfg = load_config(config)
    bot = TradingBot(cfg, strategy)

    def handle_sig(signum, frame):
        bot.stop()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    bot.run_simulate(poll_interval=interval, warmup_days=warmup)


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name")
@click.option("--interval", "-i", default=10, help="Poll interval seconds (default: 10)")
@click.option("--warmup", "-w", default=90, help="Warmup days of history (default: 90)")
@click.option("--config", "-c", default=None, help="Config file path")
def paper(strategy, interval, warmup, config):
    """Live paper trading via IBKR TWS (uses free delayed data)."""
    cfg = load_config(config)
    bot = TradingBot(cfg, strategy)

    def handle_sig(signum, frame):
        bot.stop()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    bot.run_paper(poll_interval=interval, warmup_days=warmup)


if __name__ == "__main__":
    cli()