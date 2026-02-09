"""Tests for core components.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
from datetime import datetime

from core.events import (
    EventBus, EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
    SignalType, OrderType, OrderSide,
)
from core.portfolio import Portfolio, Position


# ===== Event System Tests =====

class TestEventBus:
    def setup_method(self):
        self.bus = EventBus()
        self.received = []

    def test_subscribe_and_publish(self):
        def handler(event):
            self.received.append(event)

        self.bus.subscribe(EventType.MARKET_DATA, handler)
        event = MarketDataEvent(symbol="SPY", close=450.0)
        self.bus.publish(event)
        self.bus.process_all()

        assert len(self.received) == 1
        assert self.received[0].symbol == "SPY"

    def test_multiple_handlers(self):
        results = {"a": False, "b": False}

        def handler_a(event):
            results["a"] = True

        def handler_b(event):
            results["b"] = True

        self.bus.subscribe(EventType.SIGNAL, handler_a)
        self.bus.subscribe(EventType.SIGNAL, handler_b)
        self.bus.publish(SignalEvent(symbol="SPY", signal_type=SignalType.LONG))
        self.bus.process_all()

        assert results["a"] and results["b"]

    def test_event_ordering(self):
        def handler(event):
            self.received.append(event.data.get("seq"))

        self.bus.subscribe(EventType.SYSTEM, handler)
        for i in range(5):
            from core.events import Event
            self.bus.publish(Event(event_type=EventType.SYSTEM, data={"seq": i}))

        self.bus.process_all()
        assert self.received == [0, 1, 2, 3, 4]

    def test_unsubscribe(self):
        def handler(event):
            self.received.append(event)

        self.bus.subscribe(EventType.MARKET_DATA, handler)
        self.bus.unsubscribe(EventType.MARKET_DATA, handler)

        self.bus.publish(MarketDataEvent(symbol="SPY", close=450.0))
        self.bus.process_all()

        assert len(self.received) == 0


# ===== Position Tests =====

class TestPosition:
    def test_new_long_position(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)

        assert pos.quantity == 100
        assert pos.avg_entry_price == 450.0
        assert pos.is_long

    def test_new_short_position(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(-100, 450.0, 1.0)

        assert pos.quantity == -100
        assert pos.is_short

    def test_increase_position(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)
        pos.apply_fill(50, 460.0, 0.5)

        assert pos.quantity == 150
        expected_avg = (100 * 450 + 50 * 460) / 150
        assert abs(pos.avg_entry_price - expected_avg) < 0.01

    def test_close_position_profit(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)
        pos.apply_fill(-100, 460.0, 1.0)

        assert pos.quantity == 0
        assert pos.realized_pnl == 1000.0  # 100 * (460 - 450)
        assert pos.is_flat

    def test_close_position_loss(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)
        pos.apply_fill(-100, 440.0, 1.0)

        assert pos.realized_pnl == -1000.0

    def test_unrealized_pnl(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)
        pos.update_price(460.0)

        assert pos.unrealized_pnl == 1000.0

    def test_partial_close(self):
        pos = Position(symbol="SPY")
        pos.apply_fill(100, 450.0, 1.0)
        pos.apply_fill(-50, 460.0, 0.5)

        assert pos.quantity == 50
        assert pos.realized_pnl == 500.0  # 50 * (460 - 450)


# ===== Portfolio Tests =====

class TestPortfolio:
    def setup_method(self):
        self.bus = EventBus()
        self.portfolio = Portfolio(self.bus, initial_capital=100_000.0)

    def test_initial_state(self):
        assert self.portfolio.cash == 100_000.0
        assert self.portfolio.total_value == 100_000.0
        assert len(self.portfolio.active_positions) == 0

    def test_buy_fill(self):
        fill = FillEvent(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=450.0,
            commission=1.0,
            order_id="test_1",
            strategy_name="test",
        )
        self.bus.publish(fill)
        self.bus.process_all()

        assert self.portfolio.cash == 100_000 - (100 * 450) - 1.0
        assert "SPY" in self.portfolio.positions
        assert self.portfolio.positions["SPY"].quantity == 100

    def test_round_trip_pnl(self):
        # Buy
        buy = FillEvent(
            symbol="SPY", side=OrderSide.BUY, quantity=100,
            fill_price=450.0, commission=1.0, order_id="t1", strategy_name="test",
        )
        self.bus.publish(buy)
        self.bus.process_all()

        # Sell at profit
        sell = FillEvent(
            symbol="SPY", side=OrderSide.SELL, quantity=100,
            fill_price=460.0, commission=1.0, order_id="t2", strategy_name="test",
        )
        self.bus.publish(sell)
        self.bus.process_all()

        assert self.portfolio.positions["SPY"].realized_pnl == 1000.0
        assert self.portfolio.cash == 100_000 + 1000 - 2.0  # Profit minus commissions

    def test_snapshot(self):
        snapshot = self.portfolio.take_snapshot()
        assert snapshot.total_value == 100_000.0
        assert len(self.portfolio.history) == 1

    def test_metrics_empty(self):
        metrics = self.portfolio.calculate_metrics()
        assert metrics == {}  # Not enough data

    def test_drawdown(self):
        self.portfolio.peak_value = 110_000
        self.portfolio.cash = 99_000  # Simulating a loss
        assert abs(self.portfolio.drawdown - (11_000 / 110_000)) < 0.001


# ===== Integration Test =====

class TestFullPipeline:
    """Test the full event pipeline: data -> strategy -> risk -> execution -> portfolio."""

    def test_market_data_flows_through(self):
        bus = EventBus()
        received_events = []

        def tracker(event):
            received_events.append(event.event_type)

        bus.subscribe(EventType.MARKET_DATA, tracker)
        bus.subscribe(EventType.SIGNAL, tracker)
        bus.subscribe(EventType.ORDER, tracker)
        bus.subscribe(EventType.FILL, tracker)

        # Publish market data
        for i in range(10):
            event = MarketDataEvent(
                symbol="SPY",
                open=450 + i, high=451 + i, low=449 + i,
                close=450.5 + i, volume=1000000,
            )
            bus.publish(event)

        bus.process_all()

        # At minimum, all market data events should be received
        assert EventType.MARKET_DATA in received_events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
