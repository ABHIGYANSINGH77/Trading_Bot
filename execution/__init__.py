# """Execution module - handles order routing and fills.

# Two execution handlers:
# 1. SimulatedExecution - For backtesting with configurable slippage/commission
# 2. IBKRExecution - For live/paper trading via Interactive Brokers
# """

# from datetime import datetime
# from typing import Dict, Optional
# import uuid

# from core.events import (
#     EventBus, EventType, OrderEvent, FillEvent,
#     OrderSide, OrderStatus, MarketDataEvent,
# )


# class SimulatedExecution:
#     """Simulated order execution for backtesting.

#     Models realistic fills with:
#     - Commission per share (IBKR-like)
#     - Minimum commission
#     - Slippage (percentage of price)
#     """

#     def __init__(
#         self,
#         event_bus: EventBus,
#         commission_per_share: float = 0.005,
#         min_commission: float = 1.0,
#         slippage_pct: float = 0.001,
#     ):
#         self.event_bus = event_bus
#         self.commission_per_share = commission_per_share
#         self.min_commission = min_commission
#         self.slippage_pct = slippage_pct
#         self._latest_prices: Dict[str, float] = {}
#         self._fill_count = 0

#         # Subscribe to events
#         self.event_bus.subscribe(EventType.ORDER, self.on_order)
#         self.event_bus.subscribe(EventType.MARKET_DATA, self._track_prices)

#     def _track_prices(self, event: MarketDataEvent) -> None:
#         self._latest_prices[event.symbol] = event.close

#     def on_order(self, order: OrderEvent) -> None:
#         """Simulate order fill."""
#         price = self._latest_prices.get(order.symbol)
#         if price is None or price <= 0:
#             return  # Can't fill without a price

#         # Apply slippage
#         if order.side == OrderSide.BUY:
#             fill_price = price * (1 + self.slippage_pct)
#         else:
#             fill_price = price * (1 - self.slippage_pct)

#         # Calculate commission
#         commission = max(
#             order.quantity * self.commission_per_share,
#             self.min_commission,
#         )

#         self._fill_count += 1

#         fill = FillEvent(
#             symbol=order.symbol,
#             side=order.side,
#             quantity=order.quantity,
#             fill_price=fill_price,
#             commission=commission,
#             order_id=order.order_id,
#             strategy_name=order.strategy_name,
#             timestamp=order.timestamp,
#         )

#         self.event_bus.publish(fill)

#     @property
#     def fill_count(self) -> int:
#         return self._fill_count


# class IBKRExecution:
#     """Live order execution via Interactive Brokers.

#     Sends orders to TWS/Gateway and listens for fills.
#     Uses ib_insync for API communication.
#     """

#     def __init__(
#         self,
#         event_bus: EventBus,
#         host: str = "127.0.0.1",
#         port: int = 7497,
#         client_id: int = 1,
#     ):
#         self.event_bus = event_bus
#         self.host = host
#         self.port = port
#         self.client_id = client_id
#         self._ib = None
#         self._pending_orders: Dict[int, OrderEvent] = {}

#         self.event_bus.subscribe(EventType.ORDER, self.on_order)

#     def connect(self):
#         """Connect to TWS/Gateway."""
#         from ib_insync import IB
#         self._ib = IB()
#         self._ib.connect(self.host, self.port, clientId=self.client_id)

#         # Set up fill callback
#         self._ib.orderStatusEvent += self._on_order_status
#         self._ib.execDetailsEvent += self._on_exec_details

#         return self._ib

#     def disconnect(self):
#         if self._ib and self._ib.isConnected():
#             self._ib.disconnect()

#     def _ensure_connected(self):
#         if self._ib is None or not self._ib.isConnected():
#             self.connect()

#     def on_order(self, order_event: OrderEvent) -> None:
#         """Send order to IBKR."""
#         from ib_insync import Stock, MarketOrder, LimitOrder

#         self._ensure_connected()

#         contract = Stock(order_event.symbol, "SMART", "USD")
#         self._ib.qualifyContracts(contract)

#         action = "BUY" if order_event.side == OrderSide.BUY else "SELL"

#         # Create IB order
#         if order_event.limit_price:
#             ib_order = LimitOrder(action, order_event.quantity, order_event.limit_price)
#         else:
#             ib_order = MarketOrder(action, order_event.quantity)

#         # Place order
#         trade = self._ib.placeOrder(contract, ib_order)

#         # Track for fill matching
#         self._pending_orders[trade.order.orderId] = order_event

#     def _on_order_status(self, trade):
#         """Handle order status updates from IBKR."""
#         pass  # Logging can go here

#     def _on_exec_details(self, trade, fill):
#         """Handle execution details (fills) from IBKR."""
#         ib_order_id = trade.order.orderId

#         if ib_order_id not in self._pending_orders:
#             return

#         original_order = self._pending_orders[ib_order_id]

#         fill_event = FillEvent(
#             symbol=original_order.symbol,
#             side=original_order.side,
#             quantity=int(fill.execution.shares),
#             fill_price=fill.execution.price,
#             commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
#             order_id=original_order.order_id,
#             strategy_name=original_order.strategy_name,
#         )

#         self.event_bus.publish(fill_event)

#         # Clean up if fully filled
#         if trade.orderStatus.remaining == 0:
#             del self._pending_orders[ib_order_id]

#     def run_event_loop(self, timeout: float = 0.1):
#         """Process IBKR events. Call this in main loop."""
#         if self._ib and self._ib.isConnected():
#             self._ib.sleep(timeout)

"""Execution module - handles order routing and fills.

Two execution handlers:
1. SimulatedExecution - For backtesting with configurable slippage/commission
2. IBKRExecution - For live/paper trading via Interactive Brokers

IBKR Library:
  The original `ib_insync` was renamed to `ib_async` after the author's passing.
  Install the actively maintained version:
      pip install ib_async
  Legacy `pip install ib_insync` still works but is frozen at v0.9.86 (July 2023).
"""

from datetime import datetime
from typing import Dict, Optional
import uuid

from core.events import (
    EventBus, EventType, OrderEvent, FillEvent,
    OrderSide, OrderStatus, MarketDataEvent,
)

# Support both ib_async (new) and ib_insync (legacy)
try:
    from ib_async import IB, Stock, MarketOrder, LimitOrder, util
    IBKR_LIB = "ib_async"
except ImportError:
    try:
        from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
        IBKR_LIB = "ib_insync"
    except ImportError:
        IB = None
        IBKR_LIB = None


class SimulatedExecution:
    """Simulated order execution for backtesting.

    Models realistic fills with:
    - Commission per share (IBKR-like)
    - Minimum commission
    - Slippage (percentage of price)
    """

    def __init__(
        self,
        event_bus: EventBus,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_pct: float = 0.001,
    ):
        self.event_bus = event_bus
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_pct = slippage_pct
        self._latest_prices: Dict[str, float] = {}
        self._fill_count = 0

        # Subscribe to events
        self.event_bus.subscribe(EventType.ORDER, self.on_order)
        self.event_bus.subscribe(EventType.MARKET_DATA, self._track_prices)

    def _track_prices(self, event: MarketDataEvent) -> None:
        self._latest_prices[event.symbol] = event.close

    def on_order(self, order: OrderEvent) -> None:
        """Simulate order fill."""
        price = self._latest_prices.get(order.symbol)
        if price is None or price <= 0:
            return  # Can't fill without a price

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        # Calculate commission
        commission = max(
            order.quantity * self.commission_per_share,
            self.min_commission,
        )

        self._fill_count += 1

        fill = FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order.order_id,
            strategy_name=order.strategy_name,
            timestamp=order.timestamp,
        )

        self.event_bus.publish(fill)

    @property
    def fill_count(self) -> int:
        return self._fill_count


class IBKRExecution:
    """Live order execution via Interactive Brokers.

    Sends orders to TWS/Gateway and listens for fills.
    Uses ib_insync for API communication.
    """

    def __init__(
        self,
        event_bus: EventBus,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        asset_type: str = "STK",
    ):
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.client_id = client_id
        self.asset_type = asset_type
        self._ib = None
        self._pending_orders: Dict[int, OrderEvent] = {}

        self.event_bus.subscribe(EventType.ORDER, self.on_order)

    def connect(self):
        """Connect to TWS/Gateway."""
        if IB is None:
            raise ImportError(
                "No IBKR library found. Install one of:\n"
                "  pip install ib_async      (recommended, actively maintained)\n"
                "  pip install ib_insync     (legacy, frozen at v0.9.86)\n"
            )
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)

        # Set up fill callback
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details

        return self._ib

    def disconnect(self):
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def _ensure_connected(self):
        if self._ib is None or not self._ib.isConnected():
            self.connect()

    def on_order(self, order_event: OrderEvent) -> None:
        """Send order to IBKR."""
        self._ensure_connected()

        if self.asset_type == "FX":
            try:
                from ib_async import Forex
            except ImportError:
                from ib_insync import Forex
            contract = Forex(order_event.symbol)
        else:
            contract = Stock(order_event.symbol, "SMART", "USD")

        self._ib.qualifyContracts(contract)

        action = "BUY" if order_event.side == OrderSide.BUY else "SELL"

        # Create IB order
        if order_event.limit_price:
            ib_order = LimitOrder(action, order_event.quantity, order_event.limit_price)
        else:
            ib_order = MarketOrder(action, order_event.quantity)

        # Place order
        trade = self._ib.placeOrder(contract, ib_order)

        # Track for fill matching
        self._pending_orders[trade.order.orderId] = order_event

    def _on_order_status(self, trade):
        """Handle order status updates from IBKR."""
        pass  # Logging can go here

    def _on_exec_details(self, trade, fill):
        """Handle execution details (fills) from IBKR."""
        ib_order_id = trade.order.orderId

        if ib_order_id not in self._pending_orders:
            return

        original_order = self._pending_orders[ib_order_id]

        fill_event = FillEvent(
            symbol=original_order.symbol,
            side=original_order.side,
            quantity=int(fill.execution.shares),
            fill_price=fill.execution.price,
            commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
            order_id=original_order.order_id,
            strategy_name=original_order.strategy_name,
        )

        self.event_bus.publish(fill_event)

        # Clean up if fully filled
        if trade.orderStatus.remaining == 0:
            del self._pending_orders[ib_order_id]

    def run_event_loop(self, timeout: float = 0.1):
        """Process IBKR events. Call this in main loop."""
        if self._ib and self._ib.isConnected():
            self._ib.sleep(timeout)