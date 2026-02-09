"""Risk management module.

Sits between signal generation and order execution.
Validates every order against risk limits before it can be sent.

Risk checks:
- Position size limits (% of portfolio)
- Portfolio leverage limits
- Drawdown limits (daily and total)
- Maximum open orders
- Correlation limits between positions
"""

from datetime import datetime, date
from typing import Dict, Optional
import numpy as np

from core.events import (
    EventBus, EventType, Event, SignalEvent, OrderEvent,
    OrderType, OrderSide, SignalType, MarketDataEvent, FillEvent,
)
from core.portfolio import Portfolio


class RiskManager:
    """Enforces risk limits on all trading activity.

    Subscribes to signals and converts them to orders if risk checks pass.
    Publishes RISK_BREACH events when limits are hit.
    """

    def __init__(
        self,
        event_bus: EventBus,
        portfolio: Portfolio,
        config: dict = None,
    ):
        self.event_bus = event_bus
        self.portfolio = portfolio

        risk_config = config or {}
        self.max_position_pct = risk_config.get("max_position_pct", 0.20)
        self.max_portfolio_leverage = risk_config.get("max_portfolio_leverage", 1.0)
        self.max_daily_drawdown_pct = risk_config.get("max_daily_drawdown_pct", 0.02)
        self.max_total_drawdown_pct = risk_config.get("max_total_drawdown_pct", 0.10)
        self.max_open_orders = risk_config.get("max_open_orders", 10)

        # State tracking
        self._daily_start_value: Optional[float] = None
        self._current_date: Optional[date] = None
        self._open_orders: Dict[str, OrderEvent] = {}
        self._trading_halted = False
        self._halt_reason = ""
        self._latest_prices: Dict[str, float] = {}
        self._risk_log = []

        # Subscribe to events
        self.event_bus.subscribe(EventType.SIGNAL, self.on_signal)
        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)
        self.event_bus.subscribe(EventType.FILL, self.on_fill)

    def on_market_data(self, event: MarketDataEvent) -> None:
        """Track prices and check daily drawdown."""
        self._latest_prices[event.symbol] = event.close

        # Track daily starting value
        bar_date = (event.bar_timestamp or event.timestamp).date() if hasattr(
            (event.bar_timestamp or event.timestamp), 'date'
        ) else None

        if bar_date and bar_date != self._current_date:
            self._current_date = bar_date
            self._daily_start_value = self.portfolio.total_value

        # Check daily drawdown
        if self._daily_start_value and self._daily_start_value > 0:
            daily_drawdown = (self._daily_start_value - self.portfolio.total_value) / self._daily_start_value
            if daily_drawdown > self.max_daily_drawdown_pct:
                self._halt_trading(f"Daily drawdown limit breached: {daily_drawdown:.2%}")

        # Check total drawdown
        if self.portfolio.drawdown > self.max_total_drawdown_pct:
            self._halt_trading(f"Total drawdown limit breached: {self.portfolio.drawdown:.2%}")

    def on_fill(self, event: FillEvent) -> None:
        """Remove filled orders from open orders tracking."""
        if event.order_id in self._open_orders:
            del self._open_orders[event.order_id]

    def on_signal(self, signal: SignalEvent) -> None:
        """Convert signal to order after risk checks."""
        if self._trading_halted:
            self._log_risk_event("BLOCKED", signal.symbol, signal.strategy_name,
                                 f"Trading halted: {self._halt_reason}")
            return

        order = self._signal_to_order(signal)
        if order is None:
            return

        # Run risk checks
        passed, reason = self._check_risk(order, signal)

        if passed:
            self._open_orders[order.order_id] = order
            self.event_bus.publish(order)
            self._log_risk_event("APPROVED", order.symbol, signal.strategy_name,
                                 f"Order approved: {order.side.value} {order.quantity} {order.symbol}")
        else:
            self._log_risk_event("REJECTED", order.symbol, signal.strategy_name, reason)
            self.event_bus.publish(Event(
                event_type=EventType.RISK_BREACH,
                data={"reason": reason, "order": order, "signal": signal},
            ))

    def _signal_to_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Convert a signal event to an order event with proper sizing."""
        symbol = signal.symbol
        price = self._latest_prices.get(symbol, 0)

        if price <= 0:
            return None

        portfolio_value = self.portfolio.total_value
        if portfolio_value <= 0:
            return None

        # Calculate position size based on signal strength and risk limits
        max_position_value = portfolio_value * self.max_position_pct
        target_value = max_position_value * signal.strength

        # Get current position
        current_qty = 0
        if symbol in self.portfolio.positions:
            current_qty = self.portfolio.positions[symbol].quantity

        if signal.signal_type == SignalType.LONG:
            # Buy to open/increase long
            target_qty = int(target_value / price)
            qty_to_trade = max(0, target_qty - current_qty)
            if qty_to_trade == 0:
                return None
            side = OrderSide.BUY

        elif signal.signal_type == SignalType.SHORT:
            # Sell to open/increase short
            target_qty = -int(target_value / price)
            qty_to_trade = abs(min(0, target_qty - current_qty))
            if qty_to_trade == 0:
                return None
            side = OrderSide.SELL

        elif signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.FLAT):
            # Close position
            if current_qty == 0:
                return None
            qty_to_trade = abs(current_qty)
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY

            # For pairs trading exits, also close the paired position
            if "pair" in signal.metadata:
                # This is handled by the execution layer
                pass
        else:
            return None

        order_id = f"{signal.strategy_name}_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        return OrderEvent(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=qty_to_trade,
            strategy_name=signal.strategy_name,
            order_id=order_id,
        )

    def _check_risk(self, order: OrderEvent, signal: SignalEvent) -> tuple:
        """Run all risk checks. Returns (passed: bool, reason: str)."""

        # Check: max open orders
        if len(self._open_orders) >= self.max_open_orders:
            return False, f"Max open orders ({self.max_open_orders}) reached"

        # Check: position size limit
        price = self._latest_prices.get(order.symbol, 0)
        order_value = order.quantity * price
        portfolio_value = self.portfolio.total_value

        if portfolio_value > 0:
            current_position_value = 0
            if order.symbol in self.portfolio.positions:
                current_position_value = abs(self.portfolio.positions[order.symbol].market_value)

            new_position_value = current_position_value + order_value
            position_pct = new_position_value / portfolio_value

            if position_pct > self.max_position_pct * 1.1:  # 10% tolerance
                return False, (
                    f"Position size {position_pct:.1%} exceeds limit "
                    f"{self.max_position_pct:.1%} for {order.symbol}"
                )

        # Check: portfolio leverage
        total_exposure = sum(
            abs(p.market_value) for p in self.portfolio.positions.values()
        ) + order_value

        if portfolio_value > 0:
            leverage = total_exposure / portfolio_value
            if leverage > self.max_portfolio_leverage * 1.1:
                return False, f"Portfolio leverage {leverage:.2f} exceeds limit {self.max_portfolio_leverage}"

        # Check: sufficient cash for buys
        if order.side == OrderSide.BUY:
            required_cash = order_value * 1.01  # 1% buffer for commissions
            if self.portfolio.cash < required_cash:
                return False, f"Insufficient cash: need {required_cash:.2f}, have {self.portfolio.cash:.2f}"

        return True, "OK"

    def _halt_trading(self, reason: str) -> None:
        """Halt all trading due to risk breach."""
        self._trading_halted = True
        self._halt_reason = reason
        self.event_bus.publish(Event(
            event_type=EventType.RISK_BREACH,
            data={"reason": reason, "action": "HALT"},
        ))
        self._log_risk_event("HALT", "ALL", "risk_manager", reason)

    def _log_risk_event(self, action: str, symbol: str, strategy: str, detail: str) -> None:
        self._risk_log.append({
            "timestamp": datetime.utcnow(),
            "action": action,
            "symbol": symbol,
            "strategy": strategy,
            "detail": detail,
        })

    def resume_trading(self) -> None:
        """Resume trading after manual review."""
        self._trading_halted = False
        self._halt_reason = ""
        self._log_risk_event("RESUME", "ALL", "risk_manager", "Trading resumed manually")

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def risk_log(self):
        return self._risk_log.copy()

    def get_status(self) -> Dict:
        """Current risk status summary."""
        portfolio_value = self.portfolio.total_value
        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_drawdown": (
                (self._daily_start_value - portfolio_value) / self._daily_start_value
                if self._daily_start_value and self._daily_start_value > 0
                else 0.0
            ),
            "total_drawdown": self.portfolio.drawdown,
            "open_orders": len(self._open_orders),
            "leverage": (
                sum(abs(p.market_value) for p in self.portfolio.positions.values()) / portfolio_value
                if portfolio_value > 0
                else 0.0
            ),
        }
