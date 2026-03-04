"""Fetch historical 15-minute bar data from Alpaca Markets Data API v2.

Saves CSVs to data/cache/ in the same format as existing files:
  {SYMBOL}_{START}_{END}_15_mins.csv
  columns: date, open, high, low, close, volume
  timestamps in US/Eastern with UTC offset (e.g., 2024-01-02 09:30:00-05:00)


Usage:
  python3 fetch_alpaca_data.py                          # default: 2024-01-01 to 2025-12-31
  python3 fetch_alpaca_data.py 2024-01-01 2025-12-31    # explicit date range

  The script skips files already present in data/cache/.
  Alpaca free tier (IEX feed) covers up to 2 years of 15-min bars for US equities.
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS   = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "TSLA", "META", "AMD", "AVGO", "NFLX"]
TIMEFRAME = "15Min"
CACHE_DIR = Path(__file__).parent / "data" / "cache"

BASE_URL  = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

DEFAULT_START = "2024-01-01"
DEFAULT_END   = "2025-12-31"

EASTERN = ZoneInfo("US/Eastern")
UTC     = ZoneInfo("UTC")

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_bars(symbol: str, api_key: str, api_secret: str,
               start: str, end: str) -> list:
    """Fetch all 15-min bars for a symbol, handling pagination."""
    headers = {
        "APCA-API-KEY-ID":     api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params: dict = {
        "timeframe":  TIMEFRAME,
        "start":      start,
        "end":        end,
        "limit":      10000,
        "adjustment": "raw",
        "feed":       "iex",   # free tier feed; switch to "sip" with paid plan
    }

    url      = BASE_URL.format(symbol=symbol)
    all_bars = []
    page     = 0

    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 60))
            print(f"\n    [rate-limited] waiting {wait}s...", end="", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code == 403:
            print(f"\n  ERROR 403 Forbidden — check API keys and that they are for a paper account.")
            return []

        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", [])
        all_bars.extend(bars)
        page += 1

        next_token = data.get("next_page_token")
        if not next_token:
            break

        params["page_token"] = next_token
        # Polite rate-limit pause (Alpaca free tier: 200 req/min)
        time.sleep(0.35)

    return all_bars


def bars_to_csv_row(bar: dict) -> dict:
    """Convert one Alpaca bar dict to CSV row dict.

    Alpaca returns UTC ISO timestamps like '2024-01-02T14:30:00Z'.
    We convert to US/Eastern with UTC offset to match existing CSV format.
    """
    ts_utc  = pd.Timestamp(bar["t"]).tz_convert("US/Eastern")
    # Format: "2024-01-02 09:30:00-05:00"
    ts_str  = ts_utc.strftime("%Y-%m-%d %H:%M:%S%z")
    # Insert colon in offset: -0500 → -05:00
    ts_str  = ts_str[:-2] + ":" + ts_str[-2:]
    return {
        "date":   ts_str,
        "open":   bar["o"],
        "high":   bar["h"],
        "low":    bar["l"],
        "close":  bar["c"],
        "volume": bar["v"],
    }


def save_symbol(symbol: str, bars: list, start: str, end: str) -> Path:
    """Save bars to CSV, return the output path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Match existing naming: AAPL_2024-01-01_2025-12-31_15_mins.csv
    out_path = CACHE_DIR / f"{symbol}_{start}_{end}_15_mins.csv"

    rows = [bars_to_csv_row(b) for b in bars]
    df   = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df.to_csv(out_path, index=False)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: Alpaca API keys not set.\n")
        print("  Steps:")
        print("  1. Create a free paper account at https://alpaca.markets")
        print("  2. Paper Trading → API Keys → Generate New Key")
        print("  3. Run:")
        print('       export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxx"')
        print('       export ALPACA_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
        print("  4. Re-run this script.\n")
        sys.exit(1)

    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_END

    print("=" * 70)
    print("  ALPACA DATA FETCHER")
    print("=" * 70)
    print(f"  Symbols:    {', '.join(SYMBOLS)}")
    print(f"  Date range: {start} → {end}")
    print(f"  Timeframe:  {TIMEFRAME}")
    print(f"  Feed:       IEX (free tier)")
    print(f"  Output:     {CACHE_DIR}/")
    print()

    for symbol in SYMBOLS:
        out_path = CACHE_DIR / f"{symbol}_{start}_{end}_15_mins.csv"

        if out_path.exists():
            df_existing = pd.read_csv(out_path)
            print(f"  {symbol:6s}  already cached  ({len(df_existing):,} bars)  →  {out_path.name}")
            continue

        print(f"  {symbol:6s}  fetching ...", end="", flush=True)
        try:
            bars = fetch_bars(symbol, api_key, api_secret, start, end)
        except requests.HTTPError as e:
            print(f"  HTTP error: {e}")
            continue
        except Exception as e:
            print(f"  error: {e}")
            continue

        if not bars:
            print("  no data returned (check dates or API permissions)")
            continue

        saved = save_symbol(symbol, bars, start, end)
        print(f"  {len(bars):,} bars  →  {saved.name}")
        time.sleep(0.5)  # brief pause between symbols

    print()
    print("  Done. Run phase4_walkforward.py with the new files:")
    print(f"  python3 phase4_walkforward.py {CACHE_DIR}/*.csv")
    print()


if __name__ == "__main__":
    main()
