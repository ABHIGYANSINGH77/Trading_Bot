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

        # Same direction - increase position
        if (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            total_cost = self.avg_entry_price * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
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

    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        df = self.get_history_df()
        if df.empty or len(df) < 2:
            return {}

        returns = df["total_value"].pct_change().dropna()
        total_return = (df["total_value"].iloc[-1] / df["total_value"].iloc[0]) - 1
        trading_days = len(returns)
        ann_factor = np.sqrt(252)

        metrics = {
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (252 / max(trading_days, 1)) - 1,
            "volatility": returns.std() * ann_factor,
            "sharpe_ratio": (returns.mean() / returns.std() * ann_factor) if returns.std() > 0 else 0,
            "max_drawdown": df["drawdown"].max(),
            "win_rate": (returns > 0).mean(),
            "total_trades": len(self.trade_log),
            "total_commission": self.total_commission,
            "final_value": self.total_value,
            "trading_days": trading_days,
        }

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics["sortino_ratio"] = returns.mean() / downside.std() * ann_factor
        else:
            metrics["sortino_ratio"] = 0.0

        # Calmar ratio
        if metrics["max_drawdown"] > 0:
            metrics["calmar_ratio"] = metrics["annualized_return"] / metrics["max_drawdown"]
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
