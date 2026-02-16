"""Risk management module.

Sits between signal generation and order execution.
Validates every order against risk limits before it can be sent.

Risk checks:
- Position size limits (% of portfolio)
- Portfolio leverage limits
- Drawdown limits (daily and total)
- Maximum open orders

CRITICAL: Exit orders (closing positions) are ALWAYS allowed through
position size checks. Only entry orders that INCREASE exposure are
checked against limits. Blocking exits is the #1 risk manager bug
in quant systems — it traps you in losing positions.
"""

from datetime import datetime, date
from typing import Dict, List, Optional
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
        self.max_daily_drawdown_pct = risk_config.get("max_daily_drawdown_pct", 0.05)  # 5%
        self.max_total_drawdown_pct = risk_config.get("max_total_drawdown_pct", 0.25)  # 25%
        self.max_open_orders = risk_config.get("max_open_orders", 10)

        # --- Position sizing controls ---
        # Minimum order value — trades below this aren't worth the commission
        # At $1 min commission, a $200 trade costs 0.5% in commission alone
        self.min_order_value = risk_config.get("min_order_value", 500.0)

        # Risk per trade — max % of portfolio risked on any single trade
        # This is the prop firm standard: risk 2-5% of capital per trade
        # If stop loss is 2% away, position size = (portfolio * risk_pct) / stop_distance
        self.risk_per_trade_pct = risk_config.get("risk_per_trade_pct", 0.03)  # 3%

        # State tracking
        self._daily_start_value: Optional[float] = None
        self._current_date: Optional[date] = None
        self._open_orders: Dict[str, OrderEvent] = {}
        self._trading_halted = False
        self._halt_reason = ""
        self._halt_type = ""  # "daily" or "total" — daily resets each morning
        self._latest_prices: Dict[str, float] = {}
        self._latest_bar_time: Optional[datetime] = None  # Simulated time, not wall clock
        self._risk_log = []
        self._symbols: List[str] = []

        # Track signal metadata (stop/target) for visualizer
        # Maps order_id → signal metadata dict
        self._signal_metadata: Dict[str, Dict] = {}

        # Subscribe to events
        self.event_bus.subscribe(EventType.SIGNAL, self.on_signal)
        self.event_bus.subscribe(EventType.MARKET_DATA, self.on_market_data)
        self.event_bus.subscribe(EventType.FILL, self.on_fill)

    def set_symbols(self, symbols: list):
        """Auto-scale position limits based on number of symbols."""
        self._symbols = symbols
        n = len(symbols)
        if n <= 1:
            self.max_position_pct = 0.90  # Single stock: use up to 90%
        elif n == 2:
            self.max_position_pct = 0.50
        elif n <= 5:
            self.max_position_pct = 0.30
        else:
            self.max_position_pct = 0.20  # 6+: standard diversification

        # For small accounts, reduce min order value proportionally
        portfolio_value = self.portfolio.initial_capital if hasattr(self.portfolio, 'initial_capital') else 10000
        if hasattr(self.portfolio, 'initial_capital'):
            portfolio_value = self.portfolio.initial_capital
        # Min order = at least 5% of portfolio, so you don't have too many micro-positions
        # But never below $200 (commission floor)
        self.min_order_value = max(200, portfolio_value * 0.05)

    def on_market_data(self, event: MarketDataEvent) -> None:
        """Track prices, bar timestamps, and check daily drawdown."""
        self._latest_prices[event.symbol] = event.close
        self._latest_bar_time = event.bar_timestamp or event.timestamp

        # Track daily starting value — reset daily halt each new trading day
        bar_date = None
        ts = event.bar_timestamp or event.timestamp
        if hasattr(ts, 'date'):
            bar_date = ts.date()

        if bar_date and bar_date != self._current_date:
            self._current_date = bar_date
            self._daily_start_value = self.portfolio.total_value
            # Auto-resume if halted by daily drawdown (total drawdown stays halted)
            if self._trading_halted and self._halt_type == "daily":
                self._trading_halted = False
                self._halt_reason = ""
                self._halt_type = ""
                self._log_risk_event("RESUME", "ALL", "risk_manager",
                                     f"New trading day {bar_date} — daily halt reset")

        # Check daily drawdown
        if self._daily_start_value and self._daily_start_value > 0:
            daily_drawdown = (self._daily_start_value - self.portfolio.total_value) / self._daily_start_value
            if daily_drawdown > self.max_daily_drawdown_pct:
                if not self._trading_halted:
                    self._halt_trading(f"Daily drawdown limit breached: {daily_drawdown:.2%}", halt_type="daily")

        # Check total drawdown (permanent halt — needs manual resume)
        if self.portfolio.drawdown > self.max_total_drawdown_pct:
            if not self._trading_halted:
                self._halt_trading(f"Total drawdown limit breached: {self.portfolio.drawdown:.2%}", halt_type="total")

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
            # Log why — don't silently drop
            price = self._latest_prices.get(signal.symbol, 0)
            current_qty = 0
            if signal.symbol in self.portfolio.positions:
                current_qty = self.portfolio.positions[signal.symbol].quantity
            self._log_risk_event("DROPPED", signal.symbol, signal.strategy_name,
                                 f"{signal.signal_type.value} dropped: "
                                 f"price=${price:.2f}, pos_qty={current_qty}, "
                                 f"cash=${self.portfolio.cash:.0f}")
            return

        # Determine if this is an exit (position-reducing) order
        is_exit = signal.signal_type in (
            SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.FLAT
        )

        # Run risk checks
        passed, reason = self._check_risk(order, signal, is_exit=is_exit)

        if passed:
            self._open_orders[order.order_id] = order
            # Store signal metadata (stop, target, etc.) for visualizer
            self._signal_metadata[order.order_id] = signal.metadata.copy()
            self.event_bus.publish(order)
            self._log_risk_event("APPROVED", order.symbol, signal.strategy_name,
                                 f"{order.side.value} {order.quantity} {order.symbol} "
                                 f"@ ~${self._latest_prices.get(order.symbol, 0):.2f}"
                                 f"{' [EXIT]' if is_exit else ''}")
        else:
            self._log_risk_event("REJECTED", order.symbol, signal.strategy_name, reason)

    def _signal_to_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """Convert a signal event to an order event with proper sizing.

        Position sizing uses TWO methods and takes the smaller:

        1. POSITION LIMIT: max_position_pct of portfolio (e.g., 90% for single stock)
        2. RISK-PER-TRADE: if signal has stop_loss in metadata, size so that
           hitting the stop loses at most risk_per_trade_pct of portfolio (e.g., 3%)

        This mimics how prop firms size: "I'm willing to lose 3% of my capital
        on this trade, and my stop is 1.5% away, so I can buy X shares."

        Also enforces minimum order value to avoid commission-eaten tiny trades.
        """
        symbol = signal.symbol
        price = self._latest_prices.get(symbol, 0)

        if price <= 0:
            return None

        portfolio_value = self.portfolio.total_value
        if portfolio_value <= 0:
            return None

        # Get current position
        current_qty = 0
        current_position_value = 0
        if symbol in self.portfolio.positions:
            current_qty = self.portfolio.positions[symbol].quantity
            current_position_value = abs(current_qty * price)

        if signal.signal_type == SignalType.LONG:
            # Method 1: Position limit
            max_allowed_value = portfolio_value * self.max_position_pct
            remaining_value = max(0, max_allowed_value - current_position_value)
            target_value = remaining_value * max(0.1, min(signal.strength, 1.0))

            # Method 2: Risk-per-trade (if stop loss provided in signal metadata)
            stop_price = signal.metadata.get("stop")
            if stop_price and stop_price > 0 and stop_price < price:
                stop_distance_pct = (price - stop_price) / price
                if stop_distance_pct > 0.001:  # Avoid division by tiny stop
                    risk_budget = portfolio_value * self.risk_per_trade_pct
                    risk_based_value = risk_budget / stop_distance_pct
                    target_value = min(target_value, risk_based_value)

            # Never exceed available cash (keep 5% buffer)
            available_cash = self.portfolio.cash * 0.95
            target_value = min(target_value, available_cash)

            # Enforce minimum order value
            if target_value < self.min_order_value:
                return None

            qty_to_trade = int(target_value / price)
            if qty_to_trade <= 0:
                return None
            side = OrderSide.BUY

        elif signal.signal_type == SignalType.SHORT:
            max_allowed_value = portfolio_value * self.max_position_pct
            remaining_value = max(0, max_allowed_value - current_position_value)
            target_value = remaining_value * max(0.1, min(signal.strength, 1.0))

            # Risk-per-trade for shorts
            stop_price = signal.metadata.get("stop")
            if stop_price and stop_price > 0 and stop_price > price:
                stop_distance_pct = (stop_price - price) / price
                if stop_distance_pct > 0.001:
                    risk_budget = portfolio_value * self.risk_per_trade_pct
                    risk_based_value = risk_budget / stop_distance_pct
                    target_value = min(target_value, risk_based_value)

            # Enforce minimum order value
            if target_value < self.min_order_value:
                return None

            qty_to_trade = int(target_value / price)
            if qty_to_trade <= 0:
                return None
            side = OrderSide.SELL

        elif signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.FLAT):
            # Close position — no sizing logic, just close what we have
            if current_qty == 0:
                return None
            qty_to_trade = abs(current_qty)
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
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
            timestamp=self._latest_bar_time or datetime.utcnow(),
        )

    def _check_risk(self, order: OrderEvent, signal: SignalEvent, is_exit: bool = False) -> tuple:
        """Run all risk checks. Returns (passed: bool, reason: str).

        CRITICAL: Exit orders (is_exit=True) ALWAYS pass position size
        and leverage checks. You must never block a trader from closing
        a position — that's the opposite of risk management.
        """

        # Check: max open orders
        if len(self._open_orders) >= self.max_open_orders:
            return False, f"Max open orders ({self.max_open_orders}) reached"

        price = self._latest_prices.get(order.symbol, 0)
        order_value = order.quantity * price
        portfolio_value = self.portfolio.total_value

        # --- Position size & leverage: SKIP for exits ---
        if not is_exit and portfolio_value > 0:
            # For entries: compute what the position WILL be after this order
            current_qty = 0
            if order.symbol in self.portfolio.positions:
                current_qty = self.portfolio.positions[order.symbol].quantity

            # Compute new position after fill
            if order.side == OrderSide.BUY:
                new_qty = current_qty + order.quantity
            else:
                new_qty = current_qty - order.quantity

            new_position_value = abs(new_qty * price)
            position_pct = new_position_value / portfolio_value

            if position_pct > self.max_position_pct * 1.05:  # 5% tolerance
                return False, (
                    f"Position size {position_pct:.1%} would exceed limit "
                    f"{self.max_position_pct:.1%} for {order.symbol}"
                )

            # Check: portfolio leverage
            # Sum all other positions + new position value for this symbol
            total_exposure = sum(
                abs(p.market_value) for sym, p in self.portfolio.positions.items()
                if sym != order.symbol
            ) + new_position_value

            leverage = total_exposure / portfolio_value
            if leverage > self.max_portfolio_leverage * 1.05:
                return False, f"Portfolio leverage {leverage:.2f} would exceed limit {self.max_portfolio_leverage}"

        # Check: sufficient cash for buys (applies to both entries and exits of shorts)
        if order.side == OrderSide.BUY:
            required_cash = order_value * 1.01  # 1% buffer for commissions
            if self.portfolio.cash < required_cash:
                return False, f"Insufficient cash: need ${required_cash:.2f}, have ${self.portfolio.cash:.2f}"

        return True, "OK"

    def _halt_trading(self, reason: str, halt_type: str = "total") -> None:
        """Halt all trading due to risk breach.
        
        halt_type: "daily" resets next morning, "total" is permanent.
        """
        self._trading_halted = True
        self._halt_reason = reason
        self._halt_type = halt_type
        self.event_bus.publish(Event(
            event_type=EventType.RISK_BREACH,
            data={"reason": reason, "action": "HALT", "type": halt_type},
        ))
        self._log_risk_event("HALT", "ALL", "risk_manager", f"[{halt_type}] {reason}")

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

    @property
    def signal_metadata(self) -> Dict[str, Dict]:
        """Signal metadata (stop/target) keyed by order_id."""
        return self._signal_metadata

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