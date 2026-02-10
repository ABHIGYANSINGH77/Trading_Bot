"""Data ingestion module.

Supports:
- CSV files (for backtesting)
- yfinance (free historical data)
- Interactive Brokers TWS API (historical + realtime)

All data sources produce standardized OHLCV DataFrames.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.events import EventBus, MarketDataEvent


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_historical(
        self, symbol: str, start: str, end: str, bar_size: str = "1 day"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex
        """
        pass

    @abstractmethod
    def get_multiple(
        self, symbols: List[str], start: str, end: str, bar_size: str = "1 day"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols."""
        pass


class CSVDataSource(DataSource):
    """Load data from CSV files. Great for reproducible backtests."""

    def __init__(self, data_dir: str = "./data/cache"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_historical(
        self, symbol: str, start: str, end: str, bar_size: str = "1 day"
    ) -> pd.DataFrame:
        filepath = self.data_dir / f"{symbol}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"No CSV data for {symbol} at {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]

        # Standardize column names
        col_map = {}
        for col in df.columns:
            if "open" in col:
                col_map[col] = "open"
            elif "high" in col:
                col_map[col] = "high"
            elif "low" in col:
                col_map[col] = "low"
            elif "close" in col and "adj" not in col:
                col_map[col] = "close"
            elif "volume" in col:
                col_map[col] = "volume"
            elif "adj" in col and "close" in col:
                col_map[col] = "adj_close"
        df = df.rename(columns=col_map)

        # Filter date range
        df = df.loc[start:end]
        return df[["open", "high", "low", "close", "volume"]].dropna()

    def get_multiple(
        self, symbols: List[str], start: str, end: str, bar_size: str = "1 day"
    ) -> Dict[str, pd.DataFrame]:
        return {s: self.get_historical(s, start, end, bar_size) for s in symbols}

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to CSV for caching."""
        filepath = self.data_dir / f"{symbol}.csv"
        df.to_csv(filepath)


class YFinanceDataSource(DataSource):
    """Fetch data from Yahoo Finance. Free but rate-limited."""

    def __init__(self, cache_dir: Optional[str] = "./data/cache"):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._csv_source = CSVDataSource(cache_dir) if cache_dir else None

    def get_historical(
        self, symbol: str, start: str, end: str, bar_size: str = "1 day"
    ) -> pd.DataFrame:
        import yfinance as yf

        # Map bar_size to yfinance interval
        interval_map = {
            "1 day": "1d", "1 hour": "1h", "5 mins": "5m",
            "15 mins": "15m", "1 min": "1m",
        }
        interval = interval_map.get(bar_size, "1d")
        is_intraday = interval in ("1m", "5m", "15m", "1h")

        # For daily data, try cache first
        if not is_intraday and self._csv_source:
            try:
                return self._csv_source.get_historical(symbol, start, end, bar_size)
            except FileNotFoundError:
                pass

        ticker = yf.Ticker(symbol)

        if is_intraday:
            # For intraday: compute a period string that covers start→end
            from datetime import datetime, timedelta
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end) if end else pd.Timestamp.now()
            days = (pd.Timestamp.now() - start_dt).days + 1

            # Map to yfinance period
            if days <= 5:
                period = "5d"
            elif days <= 30:
                period = "1mo"
            else:
                period = "60d"  # max for 5m/15m

            df = ticker.history(period=period, interval=interval)
        else:
            df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Keep only OHLCV
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols]

        # Filter to requested date range
        if is_intraday:
            # For intraday, filter by date part only
            start_date = pd.Timestamp(start).date()
            end_date = pd.Timestamp(end).date() if end else pd.Timestamp.now().date()
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
        else:
            df = df.loc[start:end]
            # Cache daily data
            if self._csv_source and not df.empty:
                self._csv_source.save(symbol, df)

        return df.dropna()

    def get_multiple(
        self, symbols: List[str], start: str, end: str, bar_size: str = "1 day"
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_historical(symbol, start, end, bar_size)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        return result


class IBKRDataSource(DataSource):
    """Fetch data from Interactive Brokers TWS/Gateway.

    Requires TWS or IB Gateway to be running.
    Supports both ib_async (new) and ib_insync (legacy).

    IBKR Historical Data Limits:
    - 1 min bars: max 1 day per request, up to 6 months total
    - 5 min bars: max 1 week per request, up to 1 year total
    - 15 min bars: max 2 weeks per request, up to 2 years total
    - 1 hour bars: max 1 month per request, up to 5 years total
    - 1 day bars: max 1 year per request, up to 15+ years total

    For long date ranges, this class automatically splits into multiple
    requests and concatenates the results.
    """

    # Max duration per single request for each bar size
    # IBKR pacing: max 60 historical data requests per 10 minutes
    _MAX_DURATION = {
        "1 min": 1,       # 1 day per request
        "2 mins": 2,
        "3 mins": 3,
        "5 mins": 7,      # 1 week
        "10 mins": 14,    # 2 weeks
        "15 mins": 14,    # 2 weeks
        "20 mins": 14,
        "30 mins": 30,    # 1 month
        "1 hour": 30,     # 1 month
        "2 hours": 30,
        "3 hours": 30,
        "4 hours": 30,
        "8 hours": 60,
        "1 day": 365,     # 1 year
        "1 week": 730,
        "1 month": 1825,
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def _get_ibkr_lib(self):
        """Import the IBKR library (ib_async preferred, ib_insync fallback)."""
        try:
            import ib_async as lib
            return lib
        except ImportError:
            try:
                import ib_insync as lib
                return lib
            except ImportError:
                raise ImportError(
                    "No IBKR library found. Install one of:\n"
                    "  pip install ib_async      (recommended, actively maintained)\n"
                    "  pip install ib_insync     (legacy, frozen at v0.9.86)\n"
                )

    def connect(self):
        """Connect to TWS/Gateway."""
        lib = self._get_ibkr_lib()
        self._ib = lib.IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)
        print(f"  IBKR connected: {self.host}:{self.port} (client {self.client_id})")
        return self._ib

    def disconnect(self):
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def _ensure_connected(self):
        if self._ib is None or not self._ib.isConnected():
            self.connect()

    def _fetch_chunk(self, contract, end_dt: str, duration: str, bar_size: str, lib) -> pd.DataFrame:
        """Fetch a single chunk of historical data."""
        import time as _time

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        _time.sleep(0.5)  # Respect IBKR rate limits (max 60 req/10min)

        if not bars:
            return pd.DataFrame()

        df = lib.util.df(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]]

    def get_historical(
        self, symbol: str, start: str, end: str, bar_size: str = "1 day"
    ) -> pd.DataFrame:
        lib = self._get_ibkr_lib()
        self._ensure_connected()

        contract = lib.Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        total_days = (end_dt - start_dt).days

        max_days = self._MAX_DURATION.get(bar_size, 365)

        if total_days <= max_days:
            # Single request
            duration = f"{max(1, total_days)} D" if total_days <= 365 else f"{total_days // 365 + 1} Y"
            print(f"  IBKR: {symbol} {bar_size} | {duration} ending {end}")
            df = self._fetch_chunk(contract, end, duration, bar_size, lib)
        else:
            # Split into multiple requests
            print(f"  IBKR: {symbol} {bar_size} | {total_days} days → splitting into {max_days}-day chunks")
            chunks = []
            chunk_end = end_dt

            while chunk_end > start_dt:
                chunk_start = max(start_dt, chunk_end - pd.Timedelta(days=max_days))
                chunk_days = (chunk_end - chunk_start).days
                if chunk_days <= 0:
                    break

                duration = f"{chunk_days} D"
                end_str = chunk_end.strftime("%Y%m%d %H:%M:%S")
                print(f"    Chunk: {chunk_start.date()} → {chunk_end.date()} ({duration})")
                df_chunk = self._fetch_chunk(contract, end_str, duration, bar_size, lib)

                if not df_chunk.empty:
                    chunks.append(df_chunk)

                chunk_end = chunk_start

            if not chunks:
                raise ValueError(f"No data returned from IBKR for {symbol}")

            df = pd.concat(chunks).sort_index()
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicates

        if df.empty:
            raise ValueError(f"No data returned from IBKR for {symbol}")

        # Filter to exact date range
        df = df.loc[start:end]
        print(f"  IBKR: {symbol} → {len(df)} bars")
        return df

    def get_multiple(
        self, symbols: List[str], start: str, end: str, bar_size: str = "1 day"
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_historical(symbol, start, end, bar_size)
            except Exception as e:
                print(f"  Warning: IBKR fetch failed for {symbol}: {e}")
        return result

    def subscribe_realtime(self, symbol: str, event_bus: EventBus) -> None:
        """Subscribe to realtime 5-second bars for a symbol."""
        lib = self._get_ibkr_lib()
        self._ensure_connected()

        contract = lib.Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        def on_bar_update(bars, has_new_bar):
            if has_new_bar and len(bars) > 0:
                bar = bars[-1]
                event = MarketDataEvent(
                    symbol=symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    bar_timestamp=bar.time,
                )
                event_bus.publish(event)

        self._ib.reqRealTimeBars(contract, 5, "TRADES", False)
        # The callback is set via ib.barUpdateEvent
        self._ib.barUpdateEvent += on_bar_update


class DataManager:
    """High-level data manager that coordinates data sources.

    Provides a clean interface for the rest of the system.
    """

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = config.get("data", {}).get("cache_dir", "./data/cache")
        self._sources: Dict[str, DataSource] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Always have CSV and yfinance available
        self._sources["csv"] = CSVDataSource(self.cache_dir)
        self._sources["yfinance"] = YFinanceDataSource(self.cache_dir)

    def add_ibkr_source(self, host: str, port: int, client_id: int) -> None:
        """Add IBKR as a data source."""
        self._sources["ibkr"] = IBKRDataSource(host, port, client_id)

    def get_data(
        self,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "yfinance",
        bar_size: str = "1 day",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from specified source with caching."""
        if source not in self._sources:
            raise ValueError(f"Unknown data source: {source}. Available: {list(self._sources.keys())}")

        data = self._sources[source].get_multiple(symbols, start, end, bar_size)

        # Cache in memory
        for symbol, df in data.items():
            cache_key = f"{symbol}_{start}_{end}_{bar_size}"
            self._data_cache[cache_key] = df

        return data

    def get_close_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Get aligned close prices for multiple symbols. Useful for pairs trading."""
        data = self.get_data(symbols, start, end, source)
        closes = pd.DataFrame({s: df["close"] for s, df in data.items()})
        return closes.dropna()

    def replay_data(
        self,
        data: Dict[str, pd.DataFrame],
        event_bus: EventBus,
    ) -> None:
        """Replay historical data as events for backtesting.

        Merges all symbols into a single timeline and emits events in order.
        """
        # Build combined timeline
        all_events = []
        for symbol, df in data.items():
            for timestamp, row in df.iterrows():
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
                all_events.append((timestamp, event))

        # Sort by timestamp to maintain proper ordering
        all_events.sort(key=lambda x: x[0])

        # Publish all events
        for _, event in all_events:
            event_bus.publish(event)