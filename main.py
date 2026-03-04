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

        # Trade journal — logs every bar and trade for post-session review
        self._journal_bars: list = []      # All bars received
        self._journal_signals: list = []   # All signals/trades
        self._session_start = datetime.now()
        self._journal_active = False

        # Subscribe to fills to log trades
        self.event_bus.subscribe(EventType.FILL, self._on_fill_journal)

    def _on_risk_breach(self, event: Event):
        logger.warning(f"RISK BREACH: {event.data.get('reason', 'Unknown')}")
        if event.data.get("action") == "HALT":
            logger.error("Trading HALTED due to risk breach!")

    def _on_fill_journal(self, event):
        """Log every fill to the trade journal."""
        if not self._journal_active:
            return
        self._journal_signals.append({
            "timestamp": str(datetime.now()),
            "symbol": event.symbol,
            "type": "buy" if event.side.value == "BUY" else "sell",
            "price": float(event.fill_price),
            "quantity": int(event.quantity),
            "reason": event.strategy_name or "",
        })
        logger.info(f"  📝 TRADE LOGGED: {event.side.value} {event.quantity} {event.symbol} @ ${event.fill_price:.2f}")

    def _journal_bar(self, symbol: str, opn: float, high: float, low: float, close: float, volume: float):
        """Log a bar to the journal for post-session replay."""
        if not self._journal_active:
            return
        self._journal_bars.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "open": round(opn, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close, 4),
            "volume": int(volume),
        })

    def _log_bos_diagnostics(self):
        """Log BOS strategy diagnostics (structure, position, pending signals)."""
        diag = self.strategy.get_diagnostics()
        if not isinstance(diag, dict):
            return

        for symbol in self._get_strategy_symbols():
            sym_diag = diag.get(symbol, diag) if symbol in diag else diag
            if not isinstance(sym_diag, dict):
                continue

            trend = sym_diag.get("trend", "?")
            bos = sym_diag.get("bos", "none")
            hh = sym_diag.get("hh", 0)
            hl = sym_diag.get("hl", 0)
            lh = sym_diag.get("lh", 0)
            ll = sym_diag.get("ll", 0)
            pos = sym_diag.get("in_position", "none")
            pending = sym_diag.get("pending_entry", False)

            trend_icon = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "⚪"
            parts = [f"{trend_icon} {symbol}: {trend.upper()}"]
            parts.append(f"HH:{hh} HL:{hl} LH:{lh} LL:{ll}")

            if bos and bos != "none":
                parts.append(f"⚡ {bos.upper()}")
            if pos and pos != "none":
                parts.append(f"POS: {pos}")
            if pending:
                parts.append("⏳ WAITING PULLBACK")

            logger.info(f"       {' | '.join(parts)}")

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
        self._journal_active = True
        self._session_start = datetime.now()
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
                    opn = float(latest["Open"])
                    high = float(latest["High"])
                    low = float(latest["Low"])
                    vol = float(latest.get("Volume", 0))

                    event = MarketDataEvent(
                        symbol=symbol,
                        open=opn, high=high, low=low, close=price,
                        volume=vol, bar_timestamp=now, timestamp=now,
                    )
                    self.event_bus.publish(event)
                    self._journal_bar(symbol, opn, high, low, price, vol)

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

            # BOS diagnostics
            self._log_bos_diagnostics()

            # Wait
            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                break

        self._journal_active = False
        self._print_session_summary()
        self._export_session_report()

    # ================================================================
    #  MODE 2b: ALPACA — Live paper simulation via Alpaca IEX data
    #  No IBKR needed. Uses SimulatedExecution for fills.
    # ================================================================

    def run_alpaca_simulate(self, poll_interval: int = 60, warmup_days: int = 30):
        """Paper-trading simulation using Alpaca IEX 15m data + SimulatedExecution.

        Phase 1: Fetch last warmup_days of 15m bars from Alpaca to warm up
                 strategy state (ORB levels, RVOL deques, session history).
        Phase 2: Poll Alpaca every poll_interval seconds for newly completed
                 15m bars and run the full strategy → risk → simulated fill
                 pipeline.  Outputs to backtest_report.json for the dashboard.
        """
        import requests
        from zoneinfo import ZoneInfo
        import time as _time

        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        if not api_key or not api_secret:
            logger.error("Set ALPACA_API_KEY and ALPACA_API_SECRET env vars before running.")
            return

        BASE = "https://data.alpaca.markets/v2/stocks"
        HEADERS = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        ET = ZoneInfo("America/New_York")

        symbols = self._get_strategy_symbols()
        bt_cfg = self.config.get("backtest", {})

        self.execution = SimulatedExecution(
            self.event_bus,
            commission_per_share=bt_cfg.get("commission_per_share", 0.005),
            min_commission=bt_cfg.get("min_commission", 1.0),
            slippage_pct=bt_cfg.get("slippage_pct", 0.001),
        )

        def _fetch_bars(symbol: str, start_iso: str, end_iso: str, limit: int = 1000):
            """Return list of bar dicts from Alpaca REST API."""
            bars = []
            params = {
                "timeframe": "15Min",
                "start": start_iso,
                "end": end_iso,
                "limit": limit,
                "adjustment": "raw",
                "feed": "iex",
            }
            while True:
                resp = requests.get(f"{BASE}/{symbol}/bars", params=params,
                                    headers=HEADERS, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                bars.extend(data.get("bars", []))
                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token
            return bars

        # ── Phase 1: Historical warmup ──────────────────────────────
        logger.info("")
        logger.info("=" * 55)
        logger.info("  PHASE 1: Alpaca historical warmup (%d days)" % warmup_days)
        logger.info("=" * 55)

        now_utc = datetime.utcnow()
        warmup_start = (now_utc - timedelta(days=warmup_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        warmup_end   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        for symbol in sorted(symbols):
            try:
                raw_bars = _fetch_bars(symbol, warmup_start, warmup_end)
                logger.info(f"  {symbol}: {len(raw_bars)} warmup bars")
                for b in raw_bars:
                    ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")) \
                                 .astimezone(ET)
                    event = MarketDataEvent(
                        symbol=symbol,
                        open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                        volume=b["v"],
                        bar_timestamp=ts,
                        timestamp=ts,
                    )
                    self.event_bus.publish(event)
                    self.event_bus.process_all()
            except Exception as exc:
                logger.error(f"  Warmup failed for {symbol}: {exc}")

        self.portfolio.take_snapshot()
        logger.info(f"  Strategy: {self.strategy.get_diagnostics()}")

        # ── Phase 2: Live polling loop ──────────────────────────────
        logger.info("")
        logger.info("=" * 55)
        logger.info("  PHASE 2: Live Alpaca simulation")
        logger.info("=" * 55)
        logger.info(f"  Polling every {poll_interval}s — Ctrl+C to stop.\n")

        self._running = True
        self._journal_active = True
        self._session_start = datetime.now()

        # Per-symbol: last bar timestamp we've already processed.
        last_bar_ts: dict = {}

        while self._running:
            now_et = datetime.now(tz=ET)

            # Outside market hours: sleep without processing
            market_open  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

            if now_et.weekday() >= 5 or not (market_open <= now_et < market_close):
                next_check = 300 if now_et.weekday() >= 5 else 60
                logger.info(f"Outside market hours ({now_et.strftime('%H:%M ET')}). "
                            f"Sleeping {next_check}s...")
                _time.sleep(next_check)
                continue

            # Fetch last 90 min of bars to catch any we might have missed
            fetch_start = (datetime.utcnow() - timedelta(minutes=90)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            fetch_end   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            new_bars_count = 0
            for symbol in sorted(symbols):
                try:
                    raw_bars = _fetch_bars(symbol, fetch_start, fetch_end, limit=10)
                    for b in raw_bars:
                        ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")) \
                                     .astimezone(ET)
                        # Skip already-processed bars or bars that haven't closed yet
                        # A 15m bar with timestamp T is complete once T + 15min <= now
                        bar_close_time = ts.replace(tzinfo=None) + timedelta(minutes=15)
                        now_naive = now_et.replace(tzinfo=None)
                        if ts == last_bar_ts.get(symbol):
                            continue
                        if bar_close_time > now_naive:
                            continue   # bar not yet complete

                        event = MarketDataEvent(
                            symbol=symbol,
                            open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                            volume=b["v"],
                            bar_timestamp=ts,
                            timestamp=ts,
                        )
                        self.event_bus.publish(event)
                        self.event_bus.process_all()
                        last_bar_ts[symbol] = ts
                        new_bars_count += 1

                except Exception as exc:
                    logger.warning(f"  {symbol} fetch error: {exc}")

            if new_bars_count:
                logger.info(f"[{now_et.strftime('%H:%M:%S ET')}] "
                            f"Processed {new_bars_count} new bars  |  "
                            f"Portfolio: ${self.portfolio.total_value:,.2f}")
                # Export report every time we have new data (dashboard auto-refreshes)
                try:
                    self._export_session_report()
                except Exception:
                    pass

            try:
                _time.sleep(poll_interval)
            except KeyboardInterrupt:
                break

        self._journal_active = False
        self._print_session_summary()
        self._export_session_report()

    # ================================================================
    #  MODE 3: PAPER - Live trading via IBKR TWS (delayed data, free)
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

            # --- Phase 1: Warmup with IBKR 15m historical bars ---
            # event_driven needs 15m OHLCV bars (not daily) to build ORB, RVOL,
            # PDL levels, and VWAP correctly from the first live bar onward.
            logger.info("")
            logger.info("=" * 50)
            logger.info(f"  PHASE 1: 15m warmup via IBKR ({warmup_days} days)")
            logger.info("=" * 50)

            show = "MIDPOINT" if asset_type == "FX" else "TRADES"
            for symbol in sorted(symbols):
                c = contracts[symbol]
                logger.info(f"  Fetching 15m history for {symbol}...")
                try:
                    bars = ib.reqHistoricalData(
                        c,
                        endDateTime="",
                        durationStr=f"{warmup_days} D",
                        barSizeSetting="15 mins",
                        whatToShow=show,
                        useRTH=True,
                        formatDate=2,   # formatDate=2 → datetime objects
                    )
                    if bars:
                        logger.info(f"  {symbol}: {len(bars)} 15m bars loaded")
                        for bar in bars:
                            bar_ts = bar.date if isinstance(bar.date, datetime) \
                                     else datetime.strptime(str(bar.date), "%Y%m%d %H:%M:%S")
                            event = MarketDataEvent(
                                symbol=symbol,
                                open=bar.open, high=bar.high,
                                low=bar.low, close=bar.close,
                                volume=getattr(bar, "volume", 0),
                                bar_timestamp=bar_ts,
                                timestamp=bar_ts,
                            )
                            self.event_bus.publish(event)
                            self.event_bus.process_all()
                    else:
                        logger.warning(f"  No 15m data for {symbol} — strategy state may be cold")
                    # IBKR rate-limits historical requests; brief pause between symbols
                    ib.sleep(1)
                except Exception as e:
                    logger.error(f"  Warmup failed for {symbol}: {e}")

            self.portfolio.take_snapshot()
            logger.info(f"\n  Strategy state: {self.strategy.get_diagnostics()}")

            # --- Phase 2: Live 15m bar polling ---
            # Poll reqHistoricalData every poll_interval seconds for newly
            # completed 15m bars. IBKR delayed data (type 3) is ~5 min behind,
            # which is fine — our strategy fires on bar-close, not sub-minute.
            logger.info("")
            logger.info("=" * 50)
            logger.info("  PHASE 2: Live 15m paper trading (delayed IBKR)")
            logger.info("=" * 50)
            logger.info(f"  Polling every {poll_interval}s. Ctrl+C to stop.\n")

            self._running = True
            self._journal_active = True
            self._session_start = datetime.now()

            # ── Session log: persistent per-bar + per-trade record ──────
            import json as _json
            session_dir = Path("./paper_sessions")
            session_dir.mkdir(exist_ok=True)
            session_tag = self._session_start.strftime("%Y%m%d_%H%M%S")
            session_log_path = session_dir / f"session_{session_tag}.jsonl"
            logger.info(f"  Session log: {session_log_path}")

            def _log_to_session(record: dict):
                """Append one JSON line to the session log file."""
                try:
                    with open(session_log_path, "a") as _lf:
                        _lf.write(_json.dumps(record, default=str) + "\n")
                except Exception:
                    pass

            # Subscribe fills to session log
            def _on_fill_session(event):
                _log_to_session({
                    "type": "fill",
                    "ts": str(datetime.now()),
                    "symbol": event.symbol,
                    "side": event.side.value,
                    "qty": int(event.quantity),
                    "price": float(event.fill_price),
                    "strategy": event.strategy_name or "",
                })
            self.event_bus.subscribe(EventType.FILL, _on_fill_session)

            # Per-symbol: last processed bar timestamp (avoids re-processing)
            last_bar_ts: dict = {}
            poll_count = 0

            while self._running:
                poll_count += 1
                now = datetime.now()
                new_bars_this_cycle = 0

                for symbol in sorted(symbols):
                    try:
                        # Fetch last ~45 min of 15m bars to catch any missed bars
                        hist = ib.reqHistoricalData(
                            contracts[symbol],
                            endDateTime="",
                            durationStr="3600 S",   # last 60 min → 4 bars max
                            barSizeSetting="15 mins",
                            whatToShow=show,
                            useRTH=True,
                            formatDate=2,
                        )
                        ib.sleep(0.5)   # be kind to IBKR rate limits

                        for bar in (hist or []):
                            bar_ts = bar.date if isinstance(bar.date, datetime) \
                                     else datetime.strptime(str(bar.date), "%Y%m%d %H:%M:%S")

                            # Skip bars we've already processed
                            if bar_ts <= last_bar_ts.get(symbol, datetime.min):
                                continue
                            # Skip a bar that hasn't closed yet (its end time > now)
                            if bar_ts + timedelta(minutes=15) > now:
                                continue

                            event = MarketDataEvent(
                                symbol=symbol,
                                open=bar.open, high=bar.high,
                                low=bar.low, close=bar.close,
                                volume=getattr(bar, "volume", 0),
                                bar_timestamp=bar_ts,
                                timestamp=bar_ts,
                            )
                            self.event_bus.publish(event)
                            self.event_bus.process_all()
                            last_bar_ts[symbol] = bar_ts
                            new_bars_this_cycle += 1

                            # Add to in-memory journal (used by _export_session_report)
                            self._journal_bar(symbol, bar.open, bar.high,
                                              bar.low, bar.close,
                                              getattr(bar, "volume", 0))

                            # Log bar to session file
                            _log_to_session({
                                "type": "bar",
                                "ts": str(bar_ts),
                                "symbol": symbol,
                                "open": bar.open, "high": bar.high,
                                "low": bar.low, "close": bar.close,
                                "volume": getattr(bar, "volume", 0),
                            })

                    except Exception as e:
                        logger.warning(f"  {symbol} bar fetch error: {e}")

                if new_bars_this_cycle:
                    self.portfolio.take_snapshot(now)
                    pos_parts = [
                        f"{s}:{p.quantity}@${p.current_price:.2f}"
                        f"({p.unrealized_pnl:+,.0f})"
                        for s, p in self.portfolio.active_positions.items()
                    ]
                    logger.info(
                        f"[{now.strftime('%H:%M:%S')} #{poll_count}] "
                        f"{new_bars_this_cycle} new bar(s) | "
                        f"Portfolio: ${self.portfolio.total_value:,.2f} "
                        f"({self.portfolio.total_return:+.2%}) | "
                        f"Pos: [{', '.join(pos_parts) or 'flat'}]"
                    )
                    # Export dashboard report whenever we have new data
                    try:
                        self._export_session_report()
                    except Exception:
                        pass

                try:
                    time.sleep(poll_interval)
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
            self._journal_active = False
            self._print_session_summary()
            self._export_session_report()

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
        print(f"  Duration:         {datetime.now() - self._session_start}")
        print(f"  Initial Capital:  ${self.portfolio.initial_capital:>14,.2f}")
        print(f"  Final Value:      ${self.portfolio.total_value:>14,.2f}")
        print(f"  Total Return:      {self.portfolio.total_return:>14.2%}")
        print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0):>14.2%}")
        print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>14.3f}")
        print(f"  Total Trades:      {len(self.portfolio.trade_log):>14d}")
        print(f"  Total Commission: ${self.portfolio.total_commission:>14,.2f}")
        print(f"  Bars Received:     {len(self._journal_bars):>14d}")

        if not trades_df.empty:
            print(f"\n  Recent trades:")
            for _, t in trades_df.tail(10).iterrows():
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

    def _export_session_report(self):
        """Export the live/paper session as a JSON report for the visualizer.

        This lets you replay exactly what happened during paper trading,
        see where trades were executed, and review your strategy's decisions.
        """
        import json

        if not self._journal_bars:
            print("  No bars logged — nothing to export.")
            return

        symbols = list(self._get_strategy_symbols())

        # Build per-symbol bar arrays
        bars_by_symbol = {}
        for bar in self._journal_bars:
            sym = bar["symbol"]
            if sym not in bars_by_symbol:
                bars_by_symbol[sym] = []
            bars_by_symbol[sym].append({
                "date": bar["date"],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            })

        # Use first symbol as primary
        primary_sym = symbols[0] if symbols else list(bars_by_symbol.keys())[0]
        primary_bars = bars_by_symbol.get(primary_sym, [])

        # Map signals to bar indices
        signals_with_idx = []
        for sig in self._journal_signals:
            sym = sig.get("symbol", primary_sym)
            sym_bars = bars_by_symbol.get(sym, primary_bars)
            # Find closest bar by timestamp
            best_idx = len(sym_bars) - 1
            for i, b in enumerate(sym_bars):
                if b["date"] >= sig["timestamp"]:
                    best_idx = i
                    break
            signals_with_idx.append({
                "index": best_idx,
                "symbol": sym,
                "type": sig["type"],
                "price": sig["price"],
                "quantity": sig.get("quantity", 0),
                "reason": sig.get("reason", ""),
                "timestamp": sig["timestamp"],
            })

        # Pair trades for PnL
        paired = []
        open_pos = {}
        for sig in signals_with_idx:
            sym = sig["symbol"]
            if sym not in open_pos:
                open_pos[sym] = sig
            else:
                entry = open_pos.pop(sym)
                is_long = entry["type"] == "buy"
                pnl = (sig["price"] - entry["price"]) if is_long else (entry["price"] - sig["price"])
                pnl_pct = pnl / entry["price"] * 100
                paired.append({
                    "symbol": sym,
                    "direction": "LONG" if is_long else "SHORT",
                    "entry_time": entry["timestamp"],
                    "entry_price": entry["price"],
                    "exit_time": sig["timestamp"],
                    "exit_price": sig["price"],
                    "quantity": entry.get("quantity", 0),
                    "net_pnl": round(pnl * entry.get("quantity", 1), 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "duration": "",
                    "win": pnl > 0,
                })

        # Build equity curve
        equity_curve = []
        history = self.portfolio.get_history_df()
        if not history.empty:
            for ts, row in history.iterrows():
                equity_curve.append({
                    "timestamp": str(ts),
                    "value": round(float(row["total_value"]), 2),
                    "drawdown": round(float(row.get("drawdown", 0)), 4),
                })

        session_type = "paper" if self.execution.__class__.__name__ == "IBKRExecution" else "simulate"
        metrics = self.portfolio.calculate_metrics()
        report = {
            "meta": {
                "symbols": symbols,
                "interval": "15m",
                "start": str(self._session_start),
                "end": str(datetime.now()),
                "strategies": [self.strategy_name],
                "session_type": session_type,
                "generated_at": datetime.now().isoformat(),
            },
            "metrics": metrics,
            "bars": bars_by_symbol,
            "signals": signals_with_idx,
            "paired_trades": paired,
            "equity_curve": equity_curve,
        }

        # 1. Always overwrite backtest_report.json so the web dashboard picks it up
        with open("./backtest_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # 2. Save a timestamped copy to paper_sessions/ for permanent record
        session_dir = Path("./paper_sessions")
        session_dir.mkdir(exist_ok=True)
        dated_path = session_dir / f"{session_type}_{self._session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(dated_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  Session report: {dated_path}")
        print(f"  Dashboard copy: ./backtest_report.json  (open python web_app.py)")
        print()

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

    from backtest.dashboard import plot_backtest_dashboard
    dashboard_path = plot_backtest_dashboard(report_path, "./backtest_dashboard.png")

    print(f"\n  Outputs:")
    print(f"    Chart     : {chart_path}")
    print(f"    Dashboard : {dashboard_path}")
    print(f"    Report    : {report_path}")
    print(f"\n  To visualize this backtest:")
    print(f"  python visualizer.py --report {report_path}")
    print()


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name (bos, ma_crossover)")
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.option("--interval", "-i", default="15m", help="Bar interval: 5m, 15m, 30m, 1h, 1d")
@click.option("--source", "-d", default="yfinance", type=click.Choice(["yfinance", "ibkr"]),
              help="Data source")
@click.option("--windows", "-w", default=3, help="Walk-forward windows (default: 3)")
@click.option("--mc", default=1000, help="Monte Carlo simulations (default: 1000)")
@click.option("--save-report", is_flag=True, default=False,
              help="Save validation results as validation_results.json for web dashboard")
@click.option("--config", "-c", default=None, help="Config file path")
def validate(strategy, start, end, interval, source, windows, mc, save_report, config):
    """Validate strategy: walk-forward + out-of-sample + Monte Carlo.

    \b
    Examples:
        python main.py validate -s bos -i 15m -d ibkr --start 2025-01-01 --end 2025-12-31
        python main.py validate -s ma_crossover -i 1h -d ibkr --start 2025-06-01 --end 2026-01-01
        python main.py validate -s bos -i 15m --windows 5 --mc 2000
    """
    cfg = load_config(config)

    if source == "ibkr":
        ibkr_cfg = cfg.get("ibkr", {})

    from backtest.validate import run_validation
    val_results = run_validation(
        config=cfg,
        strategy_name=strategy,
        start=start or "2025-01-01",
        end=end or "2025-12-31",
        interval=interval,
        data_source=source,
        n_walk_forward=windows,
        mc_simulations=mc,
    )

    if val_results:
        from backtest.dashboard import plot_validation_dashboard
        plot_validation_dashboard(val_results, "./validation_dashboard.png")

        if save_report:
            import json as _json

            def _make_serializable(obj):
                """Recursively convert numpy types to Python native types."""
                import numpy as _np
                import pandas as _pd
                if isinstance(obj, dict):
                    return {k: _make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_make_serializable(v) for v in obj]
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, _pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, _pd.Series):
                    return obj.tolist()
                return obj

            report_out = _make_serializable(val_results)
            with open("./validation_results.json", "w") as _f:
                _json.dump(report_out, _f, indent=2, default=str)
            print(f"\n  Validation report saved: ./validation_results.json")
            print(f"  Launch dashboard: python web_app.py")


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


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name")
@click.option("--interval", "-i", default=60,
              help="Poll interval in seconds (default: 60)")
@click.option("--warmup", "-w", default=30,
              help="Warmup days of 15m Alpaca history (default: 30)")
@click.option("--config", "-c", default=None, help="Config file path")
def alpaca(strategy, interval, warmup, config):
    """Live paper simulation via Alpaca IEX 15m data — no IBKR needed.

    \b
    Warm up with 15m Alpaca historical bars, then poll live bars each interval.
    Uses SimulatedExecution (no real orders placed on the Alpaca account).
    Outputs backtest_report.json every cycle — open the web dashboard to watch live.

    \b
    Set env vars before running:
        export ALPACA_API_KEY=<your_key>
        export ALPACA_API_SECRET=<your_secret>
        python main.py alpaca -s event_driven

    \b
    Then in another terminal:
        python web_app.py          # http://localhost:8050 auto-refreshes every 30s
    """
    cfg = load_config(config)
    bot = TradingBot(cfg, strategy)

    def handle_sig(signum, frame):
        bot.stop()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    bot.run_alpaca_simulate(poll_interval=interval, warmup_days=warmup)


if __name__ == "__main__":
    cli()