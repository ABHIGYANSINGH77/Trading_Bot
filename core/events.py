"""Event system for the trading engine.

All components communicate through events, enabling loose coupling
and easy testing. This is how professional quant systems are built.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import queue


class EventType(Enum):
    """All event types in the system."""
    MARKET_DATA = auto()      # New bar / tick received
    SIGNAL = auto()           # Strategy generated a signal
    ORDER = auto()            # Order to be sent to broker
    FILL = auto()             # Order was filled
    POSITION_UPDATE = auto()  # Position changed
    PORTFOLIO_UPDATE = auto() # Portfolio value changed
    RISK_BREACH = auto()      # Risk limit breached
    SYSTEM = auto()           # System events (start, stop, error)


class SignalType(Enum):
    """Trading signal directions."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    FLAT = "FLAT"             # Close all positions


class OrderType(Enum):
    """Order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Event:
    """Base event."""
    event_type: EventType = EventType.SYSTEM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataEvent(Event):
    """New market data received."""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    bar_timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA


@dataclass
class SignalEvent(Event):
    """Strategy signal."""
    symbol: str = ""
    signal_type: SignalType = SignalType.FLAT
    strength: float = 0.0     # Signal strength / confidence [0, 1]
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Order to be executed."""
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_name: str = ""
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Order fill confirmation."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    strategy_name: str = ""

    def __post_init__(self):
        self.event_type = EventType.FILL


class EventBus:
    """Central event dispatcher.

    Components register handlers for event types.
    Events are processed in FIFO order.
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = {et: [] for et in EventType}
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._event_log: List[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Register a handler for an event type."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Remove a handler."""
        self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]

    def publish(self, event: Event) -> None:
        """Add event to the queue."""
        self._queue.put(event)

    def process_next(self) -> bool:
        """Process the next event in the queue.

        Returns True if an event was processed, False if queue is empty.
        """
        try:
            event = self._queue.get_nowait()
        except queue.Empty:
            return False

        self._event_log.append(event)

        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                # Log but don't crash - resilience is key
                error_event = Event(
                    event_type=EventType.SYSTEM,
                    data={"error": str(e), "source_event": event, "handler": handler.__name__}
                )
                self._event_log.append(error_event)
                raise  # Re-raise in development; in production you might just log

        return True

    def process_all(self) -> int:
        """Process all pending events. Returns count processed."""
        count = 0
        while self.process_next():
            count += 1
        return count

    def clear(self) -> None:
        """Clear the event queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def event_log(self) -> List[Event]:
        return self._event_log.copy()
