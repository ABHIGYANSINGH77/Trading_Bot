"""Portfolio management - tracks positions, cash, PnL, and portfolio metrics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.events import (
    EventBus, EventType, FillEvent, OrderSide,
    Event, MarketDataEvent
)


# ================================================================
#  Annualization helpers
# ================================================================

# Approximate trading bars per day for each interval
# US equity market: 6.5 hours/day = 390 minutes
_BARS_PER_DAY = {
    "1m": 390, "2m": 195, "3m": 130,
    "5m": 78, "10m": 39, "15m": 26,
    "20m": 20, "30m": 13,
    "1h": 7, "2h": 3, "3h": 2, "4h": 2, "8h": 1,
    "1d": 1, "1wk": 0.2,
}

def bars_per_year(interval: str) -> float:
    """Return expected number of bars in a trading year for given interval."""
    bpd = _BARS_PER_DAY.get(interval, 1)
    return bpd * 252

def annualization_factor(interval: str) -> float:
    """Return sqrt(bars_per_year) for annualizing standard deviation."""
    return np.sqrt(bars_per_year(interval))


# ================================================================
#  Position
# ================================================================

@dataclass
class Position:
    """Represents a position in a single instrument."""
    symbol: str
    quantity: int = 0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.total_commission

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def update_price(self, price: float) -> None:
        self.current_price = price

    def apply_fill(self, quantity: int, price: float, commission: float) -> None:
        """Update position based on a fill.

        Handles position increase, decrease, and reversal correctly.
        """
        self.total_commission += commission

        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.avg_entry_price = price
            self.current_price = price
            return

        # Same direction - increase position (pyramid)
        if (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            total_cost = self.avg_entry_price * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            if self.quantity != 0:
                self.avg_entry_price = total_cost / abs(self.quantity)
        else:
            # Opposite direction - reduce or reverse
            close_qty = min(abs(quantity), abs(self.quantity))
            # Realize PnL on closed portion
            if self.quantity > 0:
                self.realized_pnl += close_qty * (price - self.avg_entry_price)
            else:
                self.realized_pnl += close_qty * (self.avg_entry_price - price)

            remaining = abs(quantity) - close_qty
            self.quantity += quantity

            if remaining > 0 and self.quantity != 0:
                # Position reversed
                self.avg_entry_price = price

        self.current_price = price


# ================================================================
#  Snapshot
# ================================================================

@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state for tracking history."""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float
    positions: Dict[str, int]  # symbol -> quantity
    drawdown: float = 0.0


# ================================================================
#  Portfolio
# ================================================================

class Portfolio:
    """Manages all positions and portfolio-level metrics.

    Subscribes to FILL and MARKET_DATA events to keep state updated.
    """

    def __init__(self, event_bus: EventBus, initial_capital: float = 100_000.0):
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[PortfolioSnapshot] = []
        self.peak_value = initial_capital
        self.trade_log: List[Dict] = []

        # Subscribe to events
        self.event_bus.subscribe(EventType.FILL, self.on_fill)
        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)

    # --- Properties ---

    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def total_commission(self) -> float:
        return sum(p.total_commission for p in self.positions.values())

    @property
    def total_return(self) -> float:
        return (self.total_value - self.initial_capital) / self.initial_capital

    @property
    def drawdown(self) -> float:
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    @property
    def active_positions(self) -> Dict[str, Position]:
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    # --- Event Handlers ---

    def on_fill(self, event: FillEvent) -> None:
        """Handle order fill - update position and cash."""
        symbol = event.symbol
        quantity = event.quantity if event.side == OrderSide.BUY else -event.quantity
        price = event.fill_price
        commission = event.commission

        # Update or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        self.positions[symbol].apply_fill(quantity, price, commission)

        # Update cash
        self.cash -= quantity * price + commission

        # Log trade
        self.trade_log.append({
            "timestamp": event.timestamp,
            "symbol": symbol,
            "side": event.side.value,
            "quantity": event.quantity,
            "price": price,
            "commission": commission,
            "strategy": event.strategy_name,
            "order_id": event.order_id,
            "cash_after": self.cash,
            "portfolio_value": self.total_value,
        })

        # Publish portfolio update
        self.event_bus.publish(Event(
            event_type=EventType.PORTFOLIO_UPDATE,
            data={"total_value": self.total_value, "cash": self.cash}
        ))

    def on_market_data(self, event: MarketDataEvent) -> None:
        """Update position prices with latest market data."""
        if event.symbol in self.positions:
            self.positions[event.symbol].update_price(event.close)

    # --- Snapshot & Reporting ---

    def take_snapshot(self, timestamp: Optional[datetime] = None) -> PortfolioSnapshot:
        """Record current portfolio state."""
        ts = timestamp or datetime.utcnow()

        # Update peak
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value

        snapshot = PortfolioSnapshot(
            timestamp=ts,
            total_value=self.total_value,
            cash=self.cash,
            positions_value=self.positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_commission=self.total_commission,
            positions={s: p.quantity for s, p in self.active_positions.items()},
            drawdown=self.drawdown,
        )
        self.history.append(snapshot)
        return snapshot

    def get_history_df(self) -> pd.DataFrame:
        """Return portfolio history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "total_value": s.total_value,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "unrealized_pnl": s.unrealized_pnl,
                "realized_pnl": s.realized_pnl,
                "commission": s.total_commission,
                "drawdown": s.drawdown,
            }
            for s in self.history
        ]).set_index("timestamp")

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def get_position_summary(self) -> pd.DataFrame:
        """Current position summary."""
        rows = []
        for symbol, pos in self.active_positions.items():
            rows.append({
                "symbol": symbol,
                "quantity": pos.quantity,
                "avg_entry": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "total_pnl": pos.total_pnl,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def calculate_metrics(self, interval: str = "1d") -> Dict:
        """Calculate portfolio-level performance metrics.

        Args:
            interval: Bar interval for correct annualization.
                      "1d", "15m", "5m", "1h", etc.

        NOTE: This computes portfolio-level stats only (return, vol, Sharpe).
              Trade-level stats (win rate, profit factor) are computed in
              BacktestEngine.print_results() from paired_trades, because
              the portfolio doesn't know about round-trip trades.
        """
        df = self.get_history_df()
        if df.empty or len(df) < 2:
            return {
                "initial_capital": self.initial_capital,
                "final_value": self.total_value,
                "total_return": 0.0,
            }

        # --- Total return: always from initial_capital ---
        total_return = (self.total_value - self.initial_capital) / self.initial_capital

        # --- Bar-level returns for risk metrics ---
        returns = df["total_value"].pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        n_bars = len(returns)

        # --- Annualization: interval-aware ---
        bpy = bars_per_year(interval)
        ann = annualization_factor(interval)  # sqrt(bars_per_year)

        # Annualized return
        if n_bars > 0 and bpy > 0:
            ann_return = (1 + total_return) ** (bpy / n_bars) - 1
        else:
            ann_return = 0.0

        # Volatility (annualized)
        vol = returns.std() * ann if returns.std() > 0 else 0.0

        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * ann
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * ann
        else:
            sortino = 0.0

        # Max drawdown
        max_dd = df["drawdown"].max()

        metrics = {
            "initial_capital": self.initial_capital,
            "final_value": round(self.total_value, 2),
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "total_commission": self.total_commission,
            "total_fills": len(self.trade_log),
            "n_bars": n_bars,
            "interval": interval,
            "bars_per_year": bpy,
        }

        # Calmar ratio
        if max_dd > 0:
            metrics["calmar_ratio"] = ann_return / max_dd
        else:
            metrics["calmar_ratio"] = 0.0

        return metrics

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.history.clear()
        self.peak_value = self.initial_capital
        self.trade_log.clear()