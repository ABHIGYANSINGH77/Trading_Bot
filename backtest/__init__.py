"""Backtesting engine.

Event-driven backtester that replays historical data through the full
strategy -> risk -> execution -> portfolio pipeline.

Features:
- Supports intraday (5m, 15m, 1h) and daily bars
- Detailed trade log with entry/exit pairing and PnL per trade
- Exports JSON report that the visualizer can replay
- No lookahead bias — strategy only sees past data
- Realistic fills with slippage and commissions
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.events import EventBus, EventType, MarketDataEvent
from core.portfolio import Portfolio
from strategies import BaseStrategy, create_strategy
from risk import RiskManager
from execution import SimulatedExecution
from data import DataManager


class BacktestEngine:
    """Event-driven backtesting engine.

    Usage:
        engine = BacktestEngine(config)
        engine.add_strategy("bos")
        results = engine.run(start="2024-06-01", end="2024-12-31", interval="15m")
        engine.print_results()
        engine.export_report()  # JSON for visualizer
    """

    def __init__(self, config: dict):
        self.config = config
        bt_config = config.get("backtest", {})

        self.event_bus = EventBus()
        self.portfolio = Portfolio(
            self.event_bus,
            initial_capital=bt_config.get("initial_capital", 100_000.0),
        )
        self.risk_manager = RiskManager(
            self.event_bus, self.portfolio, config.get("risk", {}),
        )
        self.execution = SimulatedExecution(
            self.event_bus,
            commission_per_share=bt_config.get("commission_per_share", 0.005),
            min_commission=bt_config.get("min_commission", 1.0),
            slippage_pct=bt_config.get("slippage_pct", 0.001),
        )
        self.data_manager = DataManager(config)

        self.strategies: List[BaseStrategy] = []
        self._results: Optional[Dict] = None
        self._benchmark_data: Optional[pd.DataFrame] = None
        self._bar_data: Dict[str, pd.DataFrame] = {}  # Raw bars for export
        self._interval = "1d"

    def add_strategy(self, name: str, params: dict = None) -> BaseStrategy:
        """Add a strategy to the backtest."""
        strategy_params = self.config.get("strategies", {}).get(name, {})
        if params:
            strategy_params.update(params)
        strategy = create_strategy(name, self.event_bus, strategy_params)
        self.strategies.append(strategy)
        return strategy

    def run(
        self,
        start: str = None,
        end: str = None,
        symbols: List[str] = None,
        interval: str = None,
        data_source: str = "yfinance",
        benchmark: str = None,
    ) -> Dict:
        """Run the backtest.

        Args:
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD
            symbols: Override symbols (otherwise pulled from strategy params)
            interval: Bar interval — "5m", "15m", "1h", "1d" (default from config)
            data_source: "yfinance" or "csv" or "ibkr"
            benchmark: Benchmark symbol (default None for intraday, "SPY" for daily)
        """
        bt_config = self.config.get("backtest", {})
        start = start or bt_config.get("start_date", "2024-01-01")
        end = end or bt_config.get("end_date", "2024-12-31")
        interval = interval or bt_config.get("interval", "1d")
        self._interval = interval

        # Yahoo Finance intraday data limits:
        #   1m: last 7 days, 5m/15m: last 60 days, 1h: last 730 days
        # Auto-adjust dates if user requests a range that's too old
        is_intraday = interval in ("1m", "5m", "15m", "1h")
        if is_intraday and data_source == "yfinance":
            from datetime import datetime, timedelta
            max_days = {"1m": 6, "5m": 59, "15m": 59, "1h": 729}.get(interval, 59)
            now = datetime.now()
            earliest_allowed = now - timedelta(days=max_days)
            end_dt = datetime.strptime(end, "%Y-%m-%d") if end else now
            start_dt = datetime.strptime(start, "%Y-%m-%d") if start else earliest_allowed

            if start_dt < earliest_allowed:
                old_start = start
                start_dt = earliest_allowed
                start = start_dt.strftime("%Y-%m-%d")
                print(f"  ⚠ Yahoo {interval} data only available for last {max_days} days.")
                print(f"    Adjusted start: {old_start} → {start}")

            if end_dt > now:
                end = now.strftime("%Y-%m-%d")

            # Also clamp the range
            if (end_dt - start_dt).days > max_days:
                start_dt = end_dt - timedelta(days=max_days)
                start = start_dt.strftime("%Y-%m-%d")
                print(f"    Clamped to {max_days}-day window: {start} → {end}")

        if not symbols:
            symbols = set()
            for strategy in self.strategies:
                params = strategy.params
                if "pair" in params:
                    symbols.update(params["pair"])
                if "symbols" in params:
                    symbols.update(params["symbols"])
            symbols = list(symbols) if symbols else self.config.get(
                "trading", {}
            ).get("universe", ["AAPL"])

        # Map interval to yfinance/data manager format
        bar_size_map = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "1h": "1 hour", "1d": "1 day",
        }
        bar_size = bar_size_map.get(interval, "1 day")

        # For intraday, yfinance has date range limits
        # 5m: max 60 days, 15m: max 60 days, 1h: max 730 days
        is_intraday = interval in ("1m", "5m", "15m", "1h")

        # Handle benchmark
        if benchmark is None and not is_intraday:
            benchmark = "SPY"

        fetch_symbols = list(symbols)
        if benchmark and benchmark not in fetch_symbols:
            fetch_symbols.append(benchmark)

        print(f"\n{'='*60}")
        print(f"  BACKTEST: {', '.join(symbols)}")
        print(f"  Interval: {interval} | Period: {start} → {end}")
        print(f"  Strategies: {[s.name for s in self.strategies]}")
        print(f"{'='*60}\n")

        print(f"Fetching {interval} data for {fetch_symbols}...")
        all_data = self.data_manager.get_data(
            fetch_symbols, start, end, data_source, bar_size
        )

        if not all_data:
            print("ERROR: No data fetched.")
            return {}

        # Separate benchmark
        if benchmark and benchmark in all_data:
            self._benchmark_data = all_data[benchmark]

        strategy_data = {s: all_data[s] for s in symbols if s in all_data}
        self._bar_data = strategy_data

        for s, df in strategy_data.items():
            print(f"  {s}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

        print(f"\nRunning backtest...")
        start_time = time.time()

        # Build sorted event timeline
        events = []
        for symbol, df in strategy_data.items():
            for timestamp, row in df.iterrows():
                events.append((timestamp, symbol, row))
        events.sort(key=lambda x: x[0])

        # Process events chronologically
        bar_count = 0
        for timestamp, symbol, row in events:
            event = MarketDataEvent(
                symbol=symbol,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0),
                bar_timestamp=timestamp,
                timestamp=timestamp,
            )
            self.event_bus.publish(event)
            self.event_bus.process_all()
            self.portfolio.take_snapshot(timestamp)
            bar_count += 1

        elapsed = time.time() - start_time
        print(f"Done: {bar_count} bars in {elapsed:.2f}s")

        # Build results
        metrics = self.portfolio.calculate_metrics()
        metrics["elapsed_seconds"] = elapsed
        metrics["total_bars"] = bar_count
        metrics["interval"] = interval

        trade_log = self.portfolio.get_trade_log_df()
        paired_trades = self._pair_trades(trade_log)

        self._results = {
            "metrics": metrics,
            "portfolio_history": self.portfolio.get_history_df(),
            "trade_log": trade_log,
            "paired_trades": paired_trades,
            "risk_log": self.risk_manager.risk_log,
            "symbols": symbols,
            "start": start,
            "end": end,
            "interval": interval,
        }

        return self._results

    def _pair_trades(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """Pair entry/exit trades to compute per-trade PnL.

        This is the key data for understanding strategy performance.
        """
        if trade_log.empty:
            return pd.DataFrame()

        paired = []
        open_positions = {}  # symbol -> entry info

        for _, row in trade_log.iterrows():
            symbol = row["symbol"]
            side = row["side"]
            qty = row["quantity"]
            price = row["price"]
            ts = row["timestamp"]
            strategy = row.get("strategy", "")
            commission = row.get("commission", 0)

            if symbol not in open_positions:
                # Opening trade
                open_positions[symbol] = {
                    "entry_time": ts,
                    "entry_side": side,
                    "entry_price": price,
                    "quantity": qty,
                    "entry_commission": commission,
                    "strategy": strategy,
                }
            else:
                # Closing trade
                entry = open_positions.pop(symbol)
                if entry["entry_side"] == "BUY":
                    pnl = (price - entry["entry_price"]) * entry["quantity"]
                    direction = "LONG"
                else:
                    pnl = (entry["entry_price"] - price) * entry["quantity"]
                    direction = "SHORT"

                total_commission = entry["entry_commission"] + commission
                net_pnl = pnl - total_commission
                pnl_pct = (price / entry["entry_price"] - 1) * (1 if direction == "LONG" else -1)

                # Duration
                try:
                    duration = (pd.Timestamp(ts) - pd.Timestamp(entry["entry_time"]))
                    duration_str = str(duration)
                except Exception:
                    duration_str = "?"

                paired.append({
                    "symbol": symbol,
                    "direction": direction,
                    "entry_time": entry["entry_time"],
                    "entry_price": entry["entry_price"],
                    "exit_time": ts,
                    "exit_price": price,
                    "quantity": entry["quantity"],
                    "gross_pnl": round(pnl, 2),
                    "commission": round(total_commission, 2),
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "duration": duration_str,
                    "strategy": entry["strategy"],
                    "win": net_pnl > 0,
                })

        return pd.DataFrame(paired) if paired else pd.DataFrame()

    def print_results(self) -> None:
        """Pretty print backtest results."""
        if not self._results:
            print("No results. Run backtest first.")
            return

        m = self._results["metrics"]
        paired = self._results["paired_trades"]

        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"  Interval:              {self._results.get('interval', '?'):>14}")
        print(f"  Total Bars:            {m.get('total_bars', 0):>14,}")
        print(f"  Initial Capital:       ${self.portfolio.initial_capital:>13,.2f}")
        print(f"  Final Value:           ${m.get('final_value', 0):>13,.2f}")
        print(f"  Total Return:          {m.get('total_return', 0):>14.2%}")
        print(f"  Annualized Return:     {m.get('annualized_return', 0):>14.2%}")
        print(f"  Volatility (ann.):     {m.get('volatility', 0):>14.2%}")
        print(f"  Sharpe Ratio:          {m.get('sharpe_ratio', 0):>14.3f}")
        print(f"  Sortino Ratio:         {m.get('sortino_ratio', 0):>14.3f}")
        print(f"  Max Drawdown:          {m.get('max_drawdown', 0):>14.2%}")
        print(f"  Total Commission:      ${m.get('total_commission', 0):>13,.2f}")
        print(f"{'='*60}")

        if not paired.empty:
            wins = paired[paired["win"]]
            losses = paired[~paired["win"]]
            total_trades = len(paired)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = wins["net_pnl"].mean() if not wins.empty else 0
            avg_loss = losses["net_pnl"].mean() if not losses.empty else 0
            profit_factor = (
                abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
                if not losses.empty and losses["net_pnl"].sum() != 0 else float("inf")
            )
            expectancy = paired["net_pnl"].mean()
            max_win = paired["net_pnl"].max()
            max_loss = paired["net_pnl"].min()
            avg_duration = paired["duration"].iloc[0] if len(paired) == 1 else "varies"

            print(f"\n  TRADE ANALYSIS ({total_trades} round-trip trades)")
            print(f"  {'-'*56}")
            print(f"  Win Rate:              {win_rate:>14.1%}")
            print(f"  Avg Win:               ${avg_win:>13,.2f}")
            print(f"  Avg Loss:              ${avg_loss:>13,.2f}")
            print(f"  Profit Factor:         {profit_factor:>14.2f}")
            print(f"  Expectancy:            ${expectancy:>13,.2f}")
            print(f"  Best Trade:            ${max_win:>13,.2f}")
            print(f"  Worst Trade:           ${max_loss:>13,.2f}")
            print(f"  Total P&L:             ${paired['net_pnl'].sum():>13,.2f}")

            print(f"\n  {'Symbol':<6} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'P&L%':>7} {'Duration'}")
            print(f"  {'-'*70}")
            for _, t in paired.iterrows():
                color = "+" if t["win"] else "-"
                print(
                    f"  {t['symbol']:<6} {t['direction']:<5} "
                    f"${t['entry_price']:>9,.2f} ${t['exit_price']:>9,.2f} "
                    f"${t['net_pnl']:>9,.2f} {t['pnl_pct']:>6.1f}% "
                    f"{t['duration']}"
                )

        print(f"{'='*60}")

        # Risk report
        risk_log = self._results.get("risk_log", [])
        approved = sum(1 for r in risk_log if r.get("action") == "APPROVED")
        rejected = sum(1 for r in risk_log if r.get("action") == "REJECTED")
        if risk_log:
            print(f"\n  Risk: {approved} orders approved, {rejected} rejected")
            for r in risk_log:
                if r.get("action") == "REJECTED":
                    print(f"    REJECTED: {r.get('detail', '?')} | {r.get('symbol', '?')}")

    def plot_results(self, save_path: str = None) -> str:
        """Generate performance charts."""
        if not self._results:
            print("No results to plot.")
            return ""

        history = self._results["portfolio_history"]
        if history.empty:
            print("No portfolio history.")
            return ""

        paired = self._results["paired_trades"]

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 12),
            gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        fig.suptitle(
            f"Backtest: {', '.join(self._results['symbols'])} | "
            f"{self._results['interval']} | "
            f"{self._results['start']} → {self._results['end']}",
            fontsize=14, fontweight="bold"
        )

        # Panel 1: Equity curve
        ax1 = axes[0]
        ax1.plot(history.index, history["total_value"], linewidth=2, color="#2196F3", label="Strategy")

        if self._benchmark_data is not None and not self._benchmark_data.empty:
            bench = self._benchmark_data["close"]
            common = history.index.intersection(bench.index)
            if len(common) > 0:
                start_val = self.portfolio.initial_capital
                bench_aligned = bench.loc[common]
                bench_norm = bench_aligned / bench_aligned.iloc[0] * start_val
                ax1.plot(bench_norm.index, bench_norm.values, linewidth=1.5,
                         color="#999", linestyle="--", label="Benchmark")

        # Mark trades
        if not paired.empty:
            for _, t in paired.iterrows():
                color = "#4CAF50" if t["win"] else "#F44336"
                try:
                    entry_ts = pd.Timestamp(t["entry_time"])
                    exit_ts = pd.Timestamp(t["exit_time"])
                    ax1.axvline(entry_ts, color=color, alpha=0.3, linewidth=1)
                    ax1.axvline(exit_ts, color=color, alpha=0.3, linewidth=1, linestyle="--")
                except Exception:
                    pass

        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Drawdown
        ax2 = axes[1]
        ax2.fill_between(history.index, -history["drawdown"] * 100, 0,
                         color="#F44336", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Per-trade PnL
        ax3 = axes[2]
        if not paired.empty:
            colors = ["#4CAF50" if w else "#F44336" for w in paired["win"]]
            x = range(len(paired))
            ax3.bar(x, paired["net_pnl"], color=colors, alpha=0.7)
            ax3.axhline(0, color="white", linewidth=0.5)
            ax3.set_ylabel("Trade P&L ($)")
            ax3.set_xlabel("Trade #")
        else:
            ax3.text(0.5, 0.5, "No trades", transform=ax3.transAxes,
                     ha="center", va="center", color="#999")

        plt.tight_layout()
        save_path = save_path or "./backtest_results.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nChart saved: {save_path}")
        return save_path

    def export_trades(self, path: str = None) -> str:
        """Export trade log to CSV."""
        if not self._results:
            return ""

        path = path or "./trade_log.csv"
        paired = self._results["paired_trades"]
        if not paired.empty:
            paired.to_csv(path, index=False)
            print(f"Trade log: {path}")
        return path

    def export_report(self, path: str = None) -> str:
        """Export full backtest report as JSON for the visualizer.

        This is the bridge between backtest and visualizer:
        the visualizer loads this JSON and replays the exact same trades.
        """
        if not self._results:
            return ""

        path = path or "./backtest_report.json"

        # Build bar data
        all_bars = {}
        for symbol, df in self._bar_data.items():
            bars = []
            for ts, row in df.iterrows():
                date_str = (
                    ts.strftime("%Y-%m-%d %H:%M")
                    if self._interval not in ("1d", "1wk")
                    else ts.strftime("%Y-%m-%d")
                )
                bars.append({
                    "date": date_str,
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                    "volume": int(row.get("volume", 0)),
                })
            all_bars[symbol] = bars

        # Build signals from trade log
        signals = []
        trade_log = self._results["trade_log"]
        if not trade_log.empty:
            for _, row in trade_log.iterrows():
                symbol = row["symbol"]
                ts = pd.Timestamp(row["timestamp"])
                if symbol in self._bar_data:
                    df = self._bar_data[symbol]
                    # Handle timezone mismatch: strip tz from both sides
                    df_idx = df.index.tz_localize(None) if df.index.tz else df.index
                    ts_naive = ts.tz_localize(None) if ts.tzinfo else ts
                    idx = df_idx.get_indexer([ts_naive], method="nearest")[0]
                else:
                    idx = 0

                signals.append({
                    "index": int(idx),
                    "symbol": symbol,
                    "type": "buy" if row["side"] == "BUY" else "sell",
                    "price": round(float(row["price"]), 4),
                    "quantity": int(row["quantity"]),
                    "strategy": row.get("strategy", ""),
                    "timestamp": str(row["timestamp"]),
                })

        # Paired trades
        paired_list = []
        paired = self._results["paired_trades"]
        if not paired.empty:
            for _, t in paired.iterrows():
                paired_list.append({
                    "symbol": t["symbol"],
                    "direction": t["direction"],
                    "entry_time": str(t["entry_time"]),
                    "entry_price": t["entry_price"],
                    "exit_time": str(t["exit_time"]),
                    "exit_price": t["exit_price"],
                    "quantity": int(t["quantity"]),
                    "net_pnl": t["net_pnl"],
                    "pnl_pct": t["pnl_pct"],
                    "duration": t["duration"],
                    "win": bool(t["win"]),
                })

        # Equity curve
        history = self._results["portfolio_history"]
        equity_curve = []
        if not history.empty:
            for ts, row in history.iterrows():
                equity_curve.append({
                    "timestamp": str(ts),
                    "value": round(float(row["total_value"]), 2),
                    "drawdown": round(float(row.get("drawdown", 0)), 4),
                })

        report = {
            "meta": {
                "symbols": self._results["symbols"],
                "interval": self._results["interval"],
                "start": self._results["start"],
                "end": self._results["end"],
                "strategies": [s.name for s in self.strategies],
                "generated_at": datetime.now().isoformat(),
            },
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self._results["metrics"].items()
            },
            "bars": all_bars,
            "signals": signals,
            "paired_trades": paired_list,
            "equity_curve": equity_curve,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report: {path}")
        return path


def run_backtest(config: dict, strategy_name: str, strategy_params: dict = None) -> Dict:
    """Convenience function to run a quick backtest."""
    engine = BacktestEngine(config)
    engine.add_strategy(strategy_name, strategy_params)
    results = engine.run()
    engine.print_results()
    return results