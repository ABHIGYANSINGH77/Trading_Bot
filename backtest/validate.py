"""Strategy Validation Module — Adding more robust testing .

Implements:
  1. Walk-Forward Analysis    — train/test splits across rolling windows
  2. Out-of-Sample Testing    — final holdout period the strategy never saw
  3. Monte Carlo Simulation   — reshuffle trades 1000x to find confidence intervals

Usage:
    python main.py validate -s bos -i 15m -d ibkr --start 2025-01-01 --end 2025-12-31

This will:
  - Split data 80/20 into in-sample and out-of-sample
  - Run walk-forward with 3 rolling windows on the in-sample portion
  - Run a single backtest on the out-of-sample portion
  - Run 1000 Monte Carlo reshuffles of the combined trade results
  - Print a comprehensive validation report
"""

import time
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.events import EventBus, EventType, MarketDataEvent
from core.portfolio import Portfolio
from strategies import create_strategy
from risk import RiskManager
from execution import SimulatedExecution
from data import DataManager


def _run_single_backtest(
    config: dict,
    strategy_name: str,
    bar_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    interval: str,
    label: str = "",
) -> Dict:
    """Run one backtest on pre-fetched bar data. Returns metrics + paired trades."""
    bt_config = config.get("backtest", {})

    event_bus = EventBus()
    portfolio = Portfolio(event_bus, initial_capital=bt_config.get("initial_capital", 10_000))
    risk_mgr = RiskManager(event_bus, portfolio, config.get("risk", {}))
    execution = SimulatedExecution(
        event_bus,
        commission_per_share=bt_config.get("commission_per_share", 0.005),
        min_commission=bt_config.get("min_commission", 1.0),
        slippage_pct=bt_config.get("slippage_pct", 0.001),
    )

    # Create strategy
    strat_params = config.get("strategies", {}).get(strategy_name, {})
    strat_params["symbols"] = symbols
    strat_params["interval"] = interval
    strategy = create_strategy(strategy_name, event_bus, strat_params)

    risk_mgr.set_symbols(symbols)

    # Build sorted event timeline
    events = []
    for symbol, df in bar_data.items():
        if symbol not in symbols:
            continue
        for timestamp, row in df.iterrows():
            events.append((timestamp, symbol, row))
    events.sort(key=lambda x: x[0])

    if not events:
        return {"metrics": {}, "paired_trades": pd.DataFrame(), "n_bars": 0, "label": label}

    # Replay bars
    bar_count = 0
    for timestamp, symbol, row in events:
        event = MarketDataEvent(
            symbol=symbol,
            open=row["open"], high=row["high"],
            low=row["low"], close=row["close"],
            volume=row.get("volume", 0),
            bar_timestamp=timestamp, timestamp=timestamp,
        )
        event_bus.publish(event)
        event_bus.process_all()
        portfolio.take_snapshot(timestamp)
        bar_count += 1

    # Calculate metrics
    try:
        metrics = portfolio.calculate_metrics(interval=interval)
    except TypeError:
        metrics = portfolio.calculate_metrics()
        metrics["interval"] = interval

    metrics["total_bars"] = bar_count
    metrics["initial_capital"] = portfolio.initial_capital
    metrics["final_value"] = portfolio.total_value

    # Pair trades — need BacktestEngine's _pair_trades method
    # Import it lazily to avoid circular import
    from backtest import BacktestEngine
    dummy = BacktestEngine.__new__(BacktestEngine)
    trade_log = portfolio.get_trade_log_df()
    paired = dummy._pair_trades(trade_log)

    # Risk stats
    risk_log = risk_mgr.risk_log
    approved = sum(1 for r in risk_log if r.get("action") == "APPROVED")
    blocked = sum(1 for r in risk_log if r.get("action") == "BLOCKED")

    return {
        "metrics": metrics,
        "paired_trades": paired,
        "n_bars": bar_count,
        "n_trades": len(paired) if not paired.empty else 0,
        "label": label,
        "start": str(events[0][0].date()) if events else "?",
        "end": str(events[-1][0].date()) if events else "?",
        "risk_approved": approved,
        "risk_blocked": blocked,
    }


def _split_data(
    data: Dict[str, pd.DataFrame],
    split_pct: float = 0.80,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split bar data into train and test sets by time."""
    train = {}
    test = {}
    for symbol, df in data.items():
        n = len(df)
        split_idx = int(n * split_pct)
        train[symbol] = df.iloc[:split_idx].copy()
        test[symbol] = df.iloc[split_idx:].copy()
    return train, test


def _walk_forward_splits(
    data: Dict[str, pd.DataFrame],
    n_windows: int = 3,
    train_pct: float = 0.70,
) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], str]]:
    """Create rolling walk-forward train/test splits.

    Returns list of (train_data, test_data, label) tuples.
    Each window advances through time so test periods don't overlap.
    """
    # Get total bar count from any symbol
    any_symbol = list(data.keys())[0]
    total_bars = len(data[any_symbol])

    window_size = total_bars // n_windows
    train_size = int(window_size * train_pct)
    test_size = window_size - train_size

    splits = []
    for i in range(n_windows):
        start = i * window_size
        train_end = start + train_size
        test_end = min(start + window_size, total_bars)

        train_data = {}
        test_data = {}
        for symbol, df in data.items():
            train_data[symbol] = df.iloc[start:train_end].copy()
            test_data[symbol] = df.iloc[train_end:test_end].copy()

        label = f"Window {i+1}/{n_windows}"
        splits.append((train_data, test_data, label))

    return splits


def monte_carlo(
    trade_pnls: np.ndarray,
    n_simulations: int = 1000,
    initial_capital: float = 10_000,
) -> Dict:
    """Run Monte Carlo by reshuffling trade order.

    Same trades, different sequence. Shows what range of outcomes
    was possible with the same edge (or lack of it).
    """
    if len(trade_pnls) == 0:
        return {
            "median_return": 0, "mean_return": 0,
            "p5_return": 0, "p95_return": 0,
            "p5_drawdown": 0, "median_drawdown": 0, "p95_drawdown": 0,
            "prob_profitable": 0, "prob_ruin": 0,
            "n_simulations": n_simulations,
            "n_trades": 0,
            "all_returns": np.array([]),
            "all_drawdowns": np.array([]),
        }

    final_returns = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)

    rng = np.random.default_rng(seed=42)

    for i in range(n_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(trade_pnls)

        # Build equity curve
        equity = initial_capital
        peak = initial_capital
        worst_dd = 0.0

        for pnl in shuffled:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > worst_dd:
                worst_dd = dd

        final_returns[i] = (equity - initial_capital) / initial_capital
        max_drawdowns[i] = worst_dd

    return {
        "median_return": float(np.median(final_returns)),
        "mean_return": float(np.mean(final_returns)),
        "p5_return": float(np.percentile(final_returns, 5)),
        "p25_return": float(np.percentile(final_returns, 25)),
        "p75_return": float(np.percentile(final_returns, 75)),
        "p95_return": float(np.percentile(final_returns, 95)),
        "p5_drawdown": float(np.percentile(max_drawdowns, 5)),
        "median_drawdown": float(np.median(max_drawdowns)),
        "p95_drawdown": float(np.percentile(max_drawdowns, 95)),
        "prob_profitable": float((final_returns > 0).mean()),
        "prob_ruin": float((final_returns < -0.50).mean()),  # >50% loss
        "n_simulations": n_simulations,
        "n_trades": len(trade_pnls),
        "all_returns": final_returns,
        "all_drawdowns": max_drawdowns,
    }


def run_validation(
    config: dict,
    strategy_name: str,
    start: str,
    end: str,
    interval: str,
    data_source: str = "yfinance",
    n_walk_forward: int = 3,
    oos_pct: float = 0.20,
    mc_simulations: int = 1000,
) -> Dict:
    """Full validation pipeline: walk-forward + OOS + Monte Carlo.

    Args:
        config: Trading bot config dict
        strategy_name: Name of strategy to validate
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        interval: Bar interval (15m, 1h, etc.)
        data_source: "yfinance" or "ibkr"
        n_walk_forward: Number of walk-forward windows (default 3)
        oos_pct: Fraction of data held out for out-of-sample (default 20%)
        mc_simulations: Number of Monte Carlo reshuffles (default 1000)
    """
    print(f"\n{'='*65}")
    print(f"  STRATEGY VALIDATION — {strategy_name}")
    print(f"  {interval} | {start} → {end} | {n_walk_forward} walk-forward windows")
    print(f"{'='*65}")

    # --- Fetch data once ---
    dm = DataManager(config)

    # Get symbols from strategy config
    strat_params = config.get("strategies", {}).get(strategy_name, {})
    symbols = strat_params.get("symbols", config.get("trading", {}).get("universe", ["AAPL"]))

    # Map interval to IBKR/data format
    bar_size_map = {
        "1m": "1 min", "2m": "2 mins", "3m": "3 mins",
        "5m": "5 mins", "10m": "10 mins", "15m": "15 mins",
        "20m": "20 mins", "30m": "30 mins",
        "1h": "1 hour", "2h": "2 hours", "3h": "3 hours",
        "4h": "4 hours", "8h": "8 hours",
        "1d": "1 day", "1wk": "1 week",
    }
    bar_size = bar_size_map.get(interval, "1 day")

    if data_source == "ibkr":
        ibkr_cfg = config.get("ibkr", {})
        dm.add_ibkr_source(
            host=ibkr_cfg.get("host", "127.0.0.1"),
            port=ibkr_cfg.get("port", 7497),
            client_id=ibkr_cfg.get("client_id", 10),
        )

    print(f"\n  Fetching {interval} data for {symbols}...")
    all_data = dm.get_data(symbols, start, end, data_source, bar_size)

    if not all_data:
        print("  ERROR: No data fetched.")
        return {}

    for s, df in all_data.items():
        print(f"  {s}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    total_bars = len(list(all_data.values())[0])

    # ============================================================
    #  STEP 1: Split into In-Sample (80%) and Out-of-Sample (20%)
    # ============================================================
    in_sample, out_of_sample = _split_data(all_data, split_pct=1.0 - oos_pct)

    any_sym = list(all_data.keys())[0]
    is_bars = len(in_sample[any_sym])
    oos_bars = len(out_of_sample[any_sym])

    print(f"\n  Data split: {is_bars} in-sample ({100-oos_pct*100:.0f}%) "
          f"+ {oos_bars} out-of-sample ({oos_pct*100:.0f}%)")

    # ============================================================
    #  STEP 2: Walk-Forward Analysis on In-Sample
    # ============================================================
    print(f"\n{'─'*65}")
    print(f"  WALK-FORWARD ANALYSIS ({n_walk_forward} windows)")
    print(f"{'─'*65}")

    wf_splits = _walk_forward_splits(in_sample, n_windows=n_walk_forward)
    wf_results = []

    for train_data, test_data, label in wf_splits:
        train_bars = len(list(train_data.values())[0])
        test_bars = len(list(test_data.values())[0])

        # Run on train
        train_result = _run_single_backtest(
            config, strategy_name, train_data, symbols, interval,
            label=f"{label} [TRAIN]"
        )

        # Run on test
        test_result = _run_single_backtest(
            config, strategy_name, test_data, symbols, interval,
            label=f"{label} [TEST]"
        )

        wf_results.append({
            "label": label,
            "train": train_result,
            "test": test_result,
        })

        train_m = train_result["metrics"]
        test_m = test_result["metrics"]
        train_ret = train_m.get("total_return", 0)
        test_ret = test_m.get("total_return", 0)
        train_trades = train_result["n_trades"]
        test_trades = test_result["n_trades"]

        print(f"\n  {label}:")
        print(f"    Train: {train_bars:>5} bars | {train_trades:>3} trades | "
              f"return {train_ret:>+7.2%} | "
              f"Sharpe {train_m.get('sharpe_ratio', 0):>6.2f}")
        print(f"    Test:  {test_bars:>5} bars | {test_trades:>3} trades | "
              f"return {test_ret:>+7.2%} | "
              f"Sharpe {test_m.get('sharpe_ratio', 0):>6.2f}")

        # Flag degradation
        if train_ret > 0 and test_ret < train_ret * 0.3:
            print(f"    ⚠ Significant degradation: test return << train return")

    # Walk-forward summary
    wf_test_returns = [w["test"]["metrics"].get("total_return", 0) for w in wf_results]
    wf_train_returns = [w["train"]["metrics"].get("total_return", 0) for w in wf_results]
    wf_consistency = sum(1 for r in wf_test_returns if r > 0) / len(wf_test_returns) if wf_test_returns else 0

    print(f"\n  Walk-Forward Summary:")
    print(f"    Avg train return:  {np.mean(wf_train_returns):>+7.2%}")
    print(f"    Avg test return:   {np.mean(wf_test_returns):>+7.2%}")
    print(f"    Test consistency:  {wf_consistency:.0%} of windows profitable")

    if wf_consistency >= 0.67:
        print(f"    ✓ Strategy shows consistency across windows")
    elif wf_consistency >= 0.33:
        print(f"    ~ Strategy shows mixed results — possible overfitting")
    else:
        print(f"    ✗ Strategy fails walk-forward — likely overfit")

    # ============================================================
    #  STEP 3: Out-of-Sample Test
    # ============================================================
    print(f"\n{'─'*65}")
    print(f"  OUT-OF-SAMPLE TEST (final {oos_pct*100:.0f}% of data)")
    print(f"{'─'*65}")

    oos_result = _run_single_backtest(
        config, strategy_name, out_of_sample, symbols, interval,
        label="Out-of-Sample"
    )

    oos_m = oos_result["metrics"]
    oos_ret = oos_m.get("total_return", 0)
    oos_sharpe = oos_m.get("sharpe_ratio", 0)
    oos_dd = oos_m.get("max_drawdown", 0)
    oos_trades = oos_result["n_trades"]

    print(f"\n  Period:     {oos_result['start']} → {oos_result['end']}")
    print(f"  Bars:       {oos_result['n_bars']}")
    print(f"  Trades:     {oos_trades}")
    print(f"  Return:     {oos_ret:>+7.2%}")
    print(f"  Sharpe:     {oos_sharpe:>7.2f}")
    print(f"  Max DD:     {oos_dd:>7.2%}")

    # Compare to in-sample average
    avg_is_ret = np.mean(wf_train_returns) if wf_train_returns else 0
    if avg_is_ret > 0 and oos_ret > 0:
        print(f"  ✓ Profitable in both in-sample and out-of-sample")
    elif avg_is_ret > 0 and oos_ret <= 0:
        print(f"  ✗ Profitable in-sample but LOSES money out-of-sample → overfitting")
    elif avg_is_ret <= 0 and oos_ret <= 0:
        print(f"  ✗ Loses money in both — strategy needs fundamental rework")
    else:
        print(f"  ~ Loses in-sample but profits out-of-sample — unusual, investigate")

    # ============================================================
    #  STEP 4: Monte Carlo Simulation
    # ============================================================
    print(f"\n{'─'*65}")
    print(f"  MONTE CARLO SIMULATION ({mc_simulations:,} reshuffles)")
    print(f"{'─'*65}")

    # Collect all trade P&Ls from all backtests
    all_pnls = []
    for w in wf_results:
        for phase in ("train", "test"):
            paired = w[phase]["paired_trades"]
            if not paired.empty and "net_pnl" in paired.columns:
                all_pnls.extend(paired["net_pnl"].tolist())

    oos_paired = oos_result["paired_trades"]
    if not oos_paired.empty and "net_pnl" in oos_paired.columns:
        all_pnls.extend(oos_paired["net_pnl"].tolist())

    if not all_pnls:
        print("\n  No trades to simulate. Strategy generated 0 round-trips.")
        mc_result = monte_carlo(np.array([]), mc_simulations)
    else:
        trade_pnls = np.array(all_pnls)
        initial_cap = config.get("backtest", {}).get("initial_capital", 10_000)
        mc_result = monte_carlo(trade_pnls, mc_simulations, initial_cap)

        n_trades = mc_result["n_trades"]
        print(f"\n  Trades reshuffled:  {n_trades}")
        print(f"  Avg trade P&L:      ${trade_pnls.mean():>+8.2f}")
        print(f"  Win rate:           {(trade_pnls > 0).mean():.1%}")

        print(f"\n  Return Distribution ({mc_simulations:,} paths):")
        print(f"    5th percentile:   {mc_result['p5_return']:>+7.2%}  (worst case)")
        print(f"    25th percentile:  {mc_result['p25_return']:>+7.2%}")
        print(f"    Median:           {mc_result['median_return']:>+7.2%}")
        print(f"    75th percentile:  {mc_result['p75_return']:>+7.2%}")
        print(f"    95th percentile:  {mc_result['p95_return']:>+7.2%}  (best case)")

        print(f"\n  Max Drawdown Distribution:")
        print(f"    5th percentile:   {mc_result['p5_drawdown']:>7.2%}  (lucky)")
        print(f"    Median:           {mc_result['median_drawdown']:>7.2%}")
        print(f"    95th percentile:  {mc_result['p95_drawdown']:>7.2%}  (unlucky)")

        print(f"\n  Probability of profit:   {mc_result['prob_profitable']:>6.1%}")
        print(f"  Probability of ruin:     {mc_result['prob_ruin']:>6.1%}  (>50% loss)")

        if mc_result["prob_profitable"] >= 0.70:
            print(f"  ✓ Strategy has a robust edge — profitable in {mc_result['prob_profitable']:.0%} of paths")
        elif mc_result["prob_profitable"] >= 0.50:
            print(f"  ~ Strategy is marginally profitable — coin flip")
        else:
            print(f"  ✗ Strategy loses money in majority of scenarios")

    # ============================================================
    #  FINAL VERDICT
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  VALIDATION VERDICT — {strategy_name}")
    print(f"{'='*65}")

    score = 0
    checks = []

    # Check 1: Walk-forward consistency
    if wf_consistency >= 0.67:
        score += 1
        checks.append("✓ Walk-forward: consistent across windows")
    else:
        checks.append("✗ Walk-forward: inconsistent — possible overfit")

    # Check 2: OOS profitable
    if oos_ret > 0:
        score += 1
        checks.append(f"✓ Out-of-sample: profitable ({oos_ret:+.2%})")
    else:
        checks.append(f"✗ Out-of-sample: unprofitable ({oos_ret:+.2%})")

    # Check 3: Monte Carlo
    if all_pnls and mc_result["prob_profitable"] >= 0.60:
        score += 1
        checks.append(f"✓ Monte Carlo: {mc_result['prob_profitable']:.0%} probability of profit")
    elif all_pnls:
        checks.append(f"✗ Monte Carlo: only {mc_result['prob_profitable']:.0%} probability of profit")
    else:
        checks.append("✗ Monte Carlo: no trades to simulate")

    # Check 4: Not too few trades
    total_trades = sum(w["test"]["n_trades"] for w in wf_results) + oos_trades
    if total_trades >= 20:
        score += 1
        checks.append(f"✓ Sample size: {total_trades} trades (sufficient)")
    else:
        checks.append(f"✗ Sample size: {total_trades} trades (need 20+)")

    # Check 5: Max drawdown acceptable
    if oos_dd < 0.25:
        score += 1
        checks.append(f"✓ Drawdown: {oos_dd:.1%} (acceptable)")
    else:
        checks.append(f"✗ Drawdown: {oos_dd:.1%} (too high)")

    for c in checks:
        print(f"  {c}")

    print(f"\n  Score: {score}/5")
    if score >= 4:
        print(f"  PASS — Strategy is validated for paper trading")
    elif score >= 3:
        print(f"  MARGINAL — Strategy shows promise but needs tuning")
    elif score >= 2:
        print(f"  WEAK — Strategy has issues, do NOT trade live")
    else:
        print(f"  FAIL — Strategy is not viable in current form")

    print(f"{'='*65}\n")

    return {
        "walk_forward": wf_results,
        "out_of_sample": oos_result,
        "monte_carlo": mc_result,
        "score": score,
        "checks": checks,
    }