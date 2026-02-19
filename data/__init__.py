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

    Includes a multi-layer bad-tick filtering pipeline:
      Layer 1: Intra-chunk bar-level outlier detection (rolling median + spike)
      Layer 2: Cross-chunk consistency validation (median comparison)
      Layer 3: Post-stitch boundary and isolated spike detection
      Layer 4: Final sanity pass (OHLC relationship, duplicates, non-positive)
    """

    _MAX_DURATION = {
        "1 min": 1,
        "2 mins": 2,
        "3 mins": 3,
        "5 mins": 7,
        "10 mins": 14,
        "15 mins": 14,
        "20 mins": 14,
        "30 mins": 30,
        "1 hour": 30,
        "2 hours": 30,
        "3 hours": 30,
        "4 hours": 30,
        "8 hours": 60,
        "1 day": 365,
        "1 week": 730,
        "1 month": 1825,
    }

    # Configurable thresholds
    TICK_FILTER_CONFIG = {
        "rolling_window": 10,
        "rolling_threshold": 0.15,          # 15% deviation from rolling median
        "chunk_deviation_threshold": 0.20,  # 20% chunk median deviation
        "boundary_jump_threshold": 0.10,    # 10% jump at chunk boundary
        "min_price": 0.01,
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, filter_config: Optional[Dict] = None):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

        self.filter_config = {**self.TICK_FILTER_CONFIG}
        if filter_config:
            self.filter_config.update(filter_config)

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
                    "  pip install ib_async      (recommended)\n"
                    "  pip install ib_insync     (legacy)\n"
                )

    def connect(self):
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

    def _fetch_chunk(self, contract, end_dt: str, duration: str,
                     bar_size: str, lib) -> pd.DataFrame:
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
        _time.sleep(0.6)

        if not bars:
            return pd.DataFrame()

        df = lib.util.df(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------ #
    #                    LAYER 1: Intra-chunk filtering                   #
    # ------------------------------------------------------------------ #
    def _filter_bars_in_chunk(self, df: pd.DataFrame, symbol: str,
                              chunk_label: str = "") -> pd.DataFrame:
        """Remove bad bars and repair bad HIGH/LOW within a single chunk.

        KEY INSIGHT: IBKR aggregates all exchange prints into 15m bars.
        An erroneous print (odd lot, late report, bust) becomes the bar's
        HIGH or LOW, while CLOSE (last trade) remains normal.
        
        Example: O=181.20 H=339.49 L=180.90 C=181.50
        Every previous filter checked only CLOSE (181.50 = normal) and
        missed the HIGH (339.49 = erroneous print, 87% above close).
        
        Strategy then sees H=339.49, enters at that price → impossible trade.
        
        Fix: Check ALL OHLC columns. Clamp bad HIGH/LOW to reasonable
        values (preserves valid close data). Drop bars only when CLOSE is bad.
        """
        if len(df) < 3:
            return df

        cfg = self.filter_config
        window = cfg["rolling_window"]
        threshold = cfg["rolling_threshold"]

        closes = df["close"].copy()

        # --- Compute rolling median of CLOSE as the price reference ---
        rolling_med = closes.rolling(window=window, center=True, min_periods=3).median()
        expanding_med = closes.expanding(min_periods=1).median()
        rolling_med = rolling_med.fillna(expanding_med)

        # ============================================================
        # REPAIR 1: Clamp bad HIGH/LOW values (close is valid)
        # ============================================================
        # A bad HIGH doesn't mean the bar is bad — only that one erroneous
        # print inflated the high. We clamp it to a reasonable value.
        n_clamped = 0
        
        # Check HIGH: should not be more than threshold above close median
        high_dev = ((df["high"] - rolling_med) / rolling_med).abs()
        bad_highs = high_dev > threshold
        if bad_highs.any():
            for ts in df.index[bad_highs]:
                med = rolling_med.loc[ts]
                old_high = df.loc[ts, "high"]
                close_val = df.loc[ts, "close"]
                open_val = df.loc[ts, "open"]
                # Clamp HIGH to max of (close, open) + small buffer
                # This preserves the bar direction while removing the spike
                reasonable_high = max(close_val, open_val) * 1.005  # 0.5% buffer
                if old_high > reasonable_high * 1.05:  # only clamp if truly excessive
                    df.loc[ts, "high"] = reasonable_high
                    n_clamped += 1
                    print(f"  ⚠ Clamped HIGH [{symbol} {chunk_label}]: {ts} "
                          f"H={old_high:.2f} → {reasonable_high:.2f} "
                          f"(C={close_val:.2f}, median={med:.2f})")

        # Check LOW: should not be more than threshold below close median
        low_dev = ((rolling_med - df["low"]) / rolling_med).abs()
        bad_lows = low_dev > threshold
        if bad_lows.any():
            for ts in df.index[bad_lows]:
                med = rolling_med.loc[ts]
                old_low = df.loc[ts, "low"]
                close_val = df.loc[ts, "close"]
                open_val = df.loc[ts, "open"]
                reasonable_low = min(close_val, open_val) * 0.995  # 0.5% buffer
                if old_low < reasonable_low * 0.95:  # only clamp if truly excessive
                    df.loc[ts, "low"] = reasonable_low
                    n_clamped += 1
                    print(f"  ⚠ Clamped LOW [{symbol} {chunk_label}]: {ts} "
                          f"L={old_low:.2f} → {reasonable_low:.2f} "
                          f"(C={close_val:.2f}, median={med:.2f})")

        # Check OPEN: similar to high/low
        open_dev = ((df["open"] - rolling_med) / rolling_med).abs()
        bad_opens = open_dev > threshold
        if bad_opens.any():
            for ts in df.index[bad_opens]:
                med = rolling_med.loc[ts]
                old_open = df.loc[ts, "open"]
                close_val = df.loc[ts, "close"]
                # Clamp open toward close
                df.loc[ts, "open"] = close_val
                n_clamped += 1
                print(f"  ⚠ Clamped OPEN [{symbol} {chunk_label}]: {ts} "
                      f"O={old_open:.2f} → {close_val:.2f} "
                      f"(median={med:.2f})")

        # ============================================================
        # REPAIR 2: Intrabar range check — catch any remaining bad wicks
        # ============================================================
        # For 15m bars on large-cap stocks, (H-L)/C > 5% is suspicious
        # Normal range: 0.1-1%. Even on volatile days, 3% is extreme.
        max_intrabar_range = 0.05  # 5% — very generous
        intrabar = (df["high"] - df["low"]) / df["close"]
        wide_bars = intrabar > max_intrabar_range
        if wide_bars.any():
            for ts in df.index[wide_bars]:
                close_val = df.loc[ts, "close"]
                open_val = df.loc[ts, "open"]
                old_high = df.loc[ts, "high"]
                old_low = df.loc[ts, "low"]
                rng = (old_high - old_low) / close_val * 100
                
                # Determine which side is bad
                high_from_close = (old_high - close_val) / close_val
                low_from_close = (close_val - old_low) / close_val
                
                repaired = False
                if high_from_close > 0.03:  # HIGH more than 3% above close
                    new_high = max(close_val, open_val) * 1.005
                    if abs(old_high - new_high) > 0.01:
                        df.loc[ts, "high"] = new_high
                        repaired = True
                if low_from_close > 0.03:  # LOW more than 3% below close
                    new_low = min(close_val, open_val) * 0.995
                    if abs(old_low - new_low) > 0.01:
                        df.loc[ts, "low"] = new_low
                        repaired = True
                
                if repaired:
                    n_clamped += 1
                    print(f"  ⚠ Range repair [{symbol} {chunk_label}]: {ts} "
                          f"range was {rng:.1f}% "
                          f"H={old_high:.2f}→{df.loc[ts,'high']:.2f} "
                          f"L={old_low:.2f}→{df.loc[ts,'low']:.2f}")

        if n_clamped > 0:
            print(f"  ✓ Repaired {n_clamped} bad OHLC value(s) in {chunk_label}")

        # ============================================================
        # DROP: Only bars where CLOSE itself is bad (entire bar is garbage)
        # ============================================================
        close_dev = ((closes - rolling_med) / rolling_med).abs()
        close_bad = close_dev > threshold

        # Spike detection on closes
        if len(closes) >= 3:
            vals = closes.values
            for i in range(1, len(vals) - 1):
                prev_val, curr_val, next_val = vals[i-1], vals[i], vals[i+1]
                if prev_val > 0 and next_val > 0:
                    jump_in = abs(curr_val - prev_val) / prev_val
                    jump_out = abs(next_val - curr_val) / max(curr_val, 0.01)
                    revert = abs(next_val - prev_val) / prev_val
                    if jump_in > threshold and jump_out > threshold and revert < threshold:
                        close_bad.iloc[i] = True

        # OHLC internal consistency (after repairs)
        ohlc_bad = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["close"] <= cfg["min_price"]) |
            (df["open"] <= cfg["min_price"])
        )

        drop_mask = close_bad | ohlc_bad
        n_drop = drop_mask.sum()

        if n_drop > 0:
            for ts, row in df[drop_mask].iterrows():
                med_val = rolling_med.loc[ts] if ts in rolling_med.index else 0
                print(f"  ✗ Dropping [{symbol} {chunk_label}]: {ts} "
                      f"O={row['open']:.2f} H={row['high']:.2f} "
                      f"L={row['low']:.2f} C={row['close']:.2f} "
                      f"(median={med_val:.2f})" if isinstance(med_val, (int, float))
                      else f"  ✗ Dropping [{symbol} {chunk_label}]: {ts}")

            if n_drop <= len(df) * 0.05:
                for col in ["open", "high", "low", "close"]:
                    df.loc[drop_mask, col] = np.nan
                df[["open", "high", "low", "close"]] = (
                    df[["open", "high", "low", "close"]]
                    .interpolate(method="linear").ffill().bfill()
                )
                df.loc[drop_mask, "volume"] = 0
                print(f"  ✓ Interpolated {n_drop} fully bad bar(s) in {chunk_label}")
            else:
                df = df[~drop_mask]
                print(f"  ✗ Dropped {n_drop} fully bad bar(s) in {chunk_label}")

        return df

    # ------------------------------------------------------------------ #
    #                  LAYER 2: Cross-chunk validation                    #
    # ------------------------------------------------------------------ #
    def _validate_chunks(self, chunks: List[pd.DataFrame],
                         symbol: str) -> List[pd.DataFrame]:
        """Remove entire chunks inconsistent with neighbors.

        Catches the case where IBKR returns a chunk with a different
        adjustment factor (e.g., all prices are 2x due to a split).
        """
        if len(chunks) < 3:
            return chunks

        threshold = self.filter_config["chunk_deviation_threshold"]
        chunks.sort(key=lambda c: c.index[0])
        chunk_medians = [c["close"].median() for c in chunks]

        bad_indices = set()
        for i in range(len(chunks)):
            neighbors = []
            for j in range(max(0, i - 2), min(len(chunks), i + 3)):
                if j != i:
                    neighbors.append(chunk_medians[j])

            if not neighbors:
                continue

            neighbor_med = np.median(neighbors)
            if neighbor_med > 0:
                dev = abs(chunk_medians[i] - neighbor_med) / neighbor_med
                if dev > threshold:
                    cs = chunks[i].index[0]
                    ce = chunks[i].index[-1]
                    print(f"  ⚠ BAD CHUNK [{symbol}]: {cs} -> {ce}")
                    print(f"    Chunk median: ${chunk_medians[i]:.2f}, "
                          f"Neighbor median: ${neighbor_med:.2f}, "
                          f"Deviation: {dev:.1%}")
                    bad_indices.add(i)

        if bad_indices:
            chunks = [c for i, c in enumerate(chunks) if i not in bad_indices]
            print(f"  ⚠ Removed {len(bad_indices)} bad chunk(s) for {symbol}")

        return chunks

    # ------------------------------------------------------------------ #
    #                  LAYER 3: Post-stitch boundary fix                  #
    # ------------------------------------------------------------------ #
    def _fix_stitch_boundaries(self, df: pd.DataFrame,
                               symbol: str) -> pd.DataFrame:
        """Fix price jumps at chunk boundaries after stitching.

        Interpolates isolated spikes instead of just dropping them,
        preserving bar count and timestamp continuity.
        """
        if len(df) < 10:
            return df

        threshold = self.filter_config["boundary_jump_threshold"]
        closes = df["close"].values
        bad_mask = np.zeros(len(df), dtype=bool)

        for i in range(1, len(df)):
            prev_val = closes[i - 1]
            curr_val = closes[i]
            if prev_val <= 0:
                continue

            jump = abs(curr_val - prev_val) / prev_val
            if jump <= threshold:
                continue

            window = 5
            before = closes[max(0, i - window):i]
            after = closes[i + 1:min(len(df), i + 1 + window)]

            if len(before) == 0 or len(after) == 0:
                continue

            before_med = np.median(before)
            after_med = np.median(after)

            # Current bar is far from BOTH before and after = isolated spike
            far_from_before = (abs(curr_val - before_med) / before_med > threshold
                               if before_med > 0 else False)
            far_from_after = (abs(curr_val - after_med) / after_med > threshold
                              if after_med > 0 else False)

            if far_from_before and far_from_after:
                bad_mask[i] = True
                print(f"  ⚠ Isolated spike [{symbol}]: {df.index[i]} "
                      f"C=${curr_val:.2f} (before={before_med:.2f}, "
                      f"after={after_med:.2f})")

        n_bad = bad_mask.sum()
        if n_bad > 0:
            for col in ["open", "high", "low", "close"]:
                df.loc[df.index[bad_mask], col] = np.nan
            df[["open", "high", "low", "close"]] = (
                df[["open", "high", "low", "close"]]
                .interpolate(method="linear")
                .ffill()
                .bfill()
            )
            df.loc[df.index[bad_mask], "volume"] = 0
            print(f"  ✓ Fixed {n_bad} boundary spike(s) for {symbol}")

        return df

    # ------------------------------------------------------------------ #
    #               LAYER 4: Final sanity pass                            #
    # ------------------------------------------------------------------ #
    def _final_sanity_check(self, df: pd.DataFrame,
                            symbol: str) -> pd.DataFrame:
        """Last-resort validation on the final assembled DataFrame."""
        if df.empty:
            return df

        price_cols = ["open", "high", "low", "close"]
        valid = (df[price_cols] > 0).all(axis=1)
        if not valid.all():
            n_invalid = (~valid).sum()
            print(f"  ⚠ Final pass: removing {n_invalid} rows with "
                  f"non-positive prices for {symbol}")
            df = df[valid]

        if df.index.duplicated().any():
            n_dup = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep='last')]
            print(f"  ⚠ Removed {n_dup} duplicate timestamps for {symbol}")

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------ #
    #                      Main fetch method                              #
    # ------------------------------------------------------------------ #
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
            duration = (f"{max(1, total_days)} D" if total_days <= 365
                        else f"{total_days // 365 + 1} Y")
            print(f"  IBKR: {symbol} {bar_size} | {duration} ending {end}")
            df = self._fetch_chunk(contract, end, duration, bar_size, lib)

            # Layer 1: Filter bad bars even in single chunk
            if not df.empty:
                df = self._filter_bars_in_chunk(df, symbol, "single-chunk")
        else:
            print(f"  IBKR: {symbol} {bar_size} | {total_days} days -> "
                  f"splitting into {max_days}-day chunks")
            chunks = []
            chunk_end = end_dt
            chunk_idx = 0

            while chunk_end > start_dt:
                chunk_start = max(start_dt,
                                  chunk_end - pd.Timedelta(days=max_days))
                chunk_days = (chunk_end - chunk_start).days
                if chunk_days <= 0:
                    break

                duration = f"{chunk_days} D"
                end_str = chunk_end.strftime("%Y%m%d %H:%M:%S")
                print(f"    Chunk: {chunk_start.date()} -> {chunk_end.date()} ({duration})")

                df_chunk = self._fetch_chunk(
                    contract, end_str, duration, bar_size, lib)

                if not df_chunk.empty:
                    # Layer 1: Filter each chunk individually
                    df_chunk = self._filter_bars_in_chunk(
                        df_chunk, symbol, f"chunk-{chunk_idx}")
                    if not df_chunk.empty:
                        chunks.append(df_chunk)

                chunk_end = chunk_start
                chunk_idx += 1

            if not chunks:
                raise ValueError(f"No data returned from IBKR for {symbol}")

            # Layer 2: Cross-chunk validation
            chunks = self._validate_chunks(chunks, symbol)

            if not chunks:
                raise ValueError(f"All chunks were bad for {symbol}")

            # Stitch
            df = pd.concat(chunks).sort_index()
            df = df[~df.index.duplicated(keep='last')]

            # Layer 3: Fix boundary artifacts
            df = self._fix_stitch_boundaries(df, symbol)

        if df.empty:
            raise ValueError(f"No data returned from IBKR for {symbol}")

        # Layer 4: Final sanity
        df = self._final_sanity_check(df, symbol)

        # Filter to exact date range
        df = df.loc[start:end]
        print(f"  IBKR: {symbol} -> {len(df)} clean bars")
        return df

    def get_multiple(
        self, symbols: List[str], start: str, end: str,
        bar_size: str = "1 day"
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_historical(
                    symbol, start, end, bar_size)
            except Exception as e:
                print(f"  Warning: IBKR fetch failed for {symbol}: {e}")
        return result

    def subscribe_realtime(self, symbol: str, event_bus: EventBus) -> None:
        """Subscribe to realtime 5-second bars with tick filtering."""
        lib = self._get_ibkr_lib()
        self._ensure_connected()

        contract = lib.Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        price_buffer: List[float] = []
        buffer_size = 20

        def on_bar_update(bars, has_new_bar):
            if not has_new_bar or len(bars) == 0:
                return

            bar = bars[-1]
            close = bar.close

            # Realtime tick filter: compare against recent buffer
            if len(price_buffer) >= 3:
                median_price = np.median(price_buffer[-buffer_size:])
                if median_price > 0:
                    deviation = abs(close - median_price) / median_price
                    threshold = self.filter_config["rolling_threshold"]
                    if deviation > threshold:
                        print(f"  ⚠ RT bad tick [{symbol}]: "
                              f"close=${close:.2f}, "
                              f"median=${median_price:.2f}, "
                              f"dev={deviation:.1%} — SKIPPED")
                        return  # Don't publish this bar

            price_buffer.append(close)
            if len(price_buffer) > buffer_size * 2:
                del price_buffer[:buffer_size]

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
        self._ib.barUpdateEvent += on_bar_update


class DataManager:
    """High-level data manager that coordinates data sources.

    Provides a clean interface for the rest of the system.
    Includes disk caching and cache validation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = config.get("data", {}).get("cache_dir", "./data/cache")
        self._sources: Dict[str, DataSource] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Always have CSV and yfinance available
        self._sources["csv"] = CSVDataSource(self.cache_dir)
        self._sources["yfinance"] = YFinanceDataSource(self.cache_dir)

    def add_ibkr_source(self, host: str, port: int, client_id: int,
                        filter_config: Optional[Dict] = None) -> None:
        """Add IBKR as a data source with optional filter configuration."""
        self._sources["ibkr"] = IBKRDataSource(
            host, port, client_id, filter_config=filter_config)

    def _validate_cached_data(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Quick validation on cached data — checks ALL OHLC columns.
        
        Previous versions only checked CLOSE and missed bad HIGH/LOW values.
        """
        if df.empty or len(df) < 10:
            return df

        closes = df["close"].values
        rolling_med = pd.Series(closes).rolling(10, center=True, min_periods=3).median()
        rolling_med = rolling_med.fillna(pd.Series(closes).expanding(min_periods=1).median())

        # Check ALL columns against close median, not just close
        for col in ["open", "high", "low", "close"]:
            deviation = (pd.Series(df[col].values) - rolling_med).abs() / rolling_med
            bad = deviation > 0.15
            if bad.any():
                n_bad = bad.sum()
                print(f"  ⚠ Cache for {symbol} has {n_bad} bad {col.upper()} values — re-fetching")
                return None  # Signal to caller to delete cache and re-fetch

        # Also check intrabar range
        intrabar = (df["high"] - df["low"]) / df["close"]
        if (intrabar > 0.05).any():
            n_wide = (intrabar > 0.05).sum()
            print(f"  ⚠ Cache for {symbol} has {n_wide} bars with >5% range — re-fetching")
            return None

        return df

    def get_data(
        self,
        symbols: List[str],
        start: str,
        end: str,
        source: str = "yfinance",
        bar_size: str = "1 day",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from specified source with disk caching.

        On first fetch from IBKR, saves validated data to CSV in cache_dir.
        On subsequent runs, loads from cache (fast, no IBKR needed).
        Delete cache files to force re-fetch: rm -rf data/cache/
        """
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

        bar_label = bar_size.replace(" ", "_")
        data = {}
        uncached = []

        for symbol in symbols:
            cache_file = os.path.join(self.cache_dir,
                f"{symbol}_{start}_{end}_{bar_label}.csv")
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Validate cached data
                validated = self._validate_cached_data(df, symbol)
                if validated is not None:
                    data[symbol] = validated
                    print(f"  Cache hit: {symbol} -> {len(validated)} bars")
                else:
                    # Cache has bad ticks — delete and re-fetch
                    os.remove(cache_file)
                    print(f"  Cache invalidated for {symbol}, will re-fetch")
                    uncached.append(symbol)
            else:
                uncached.append(symbol)

        # Fetch uncached symbols from source
        if uncached:
            if source not in self._sources:
                raise ValueError(f"Unknown data source: {source}. "
                                 f"Available: {list(self._sources.keys())}")

            fetched = self._sources[source].get_multiple(
                uncached, start, end, bar_size)

            # Save to disk cache
            for symbol, df in fetched.items():
                cache_file = os.path.join(self.cache_dir,
                    f"{symbol}_{start}_{end}_{bar_label}.csv")
                df.to_csv(cache_file)
                print(f"  Cached: {symbol} -> {cache_file}")
                data[symbol] = df

        # Cache in memory too
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
        """Get aligned close prices for multiple symbols."""
        data = self.get_data(symbols, start, end, source)
        closes = pd.DataFrame({s: df["close"] for s, df in data.items()})
        return closes.dropna()

    def replay_data(
        self,
        data: Dict[str, pd.DataFrame],
        event_bus: EventBus,
    ) -> None:
        """Replay historical data as events for backtesting."""
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

        all_events.sort(key=lambda x: x[0])

        for _, event in all_events:
            event_bus.publish(event)