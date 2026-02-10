#!/usr/bin/env python3
"""QuantBot Visual Backtester — Standalone Web App.

Two modes:
1. LIVE MODE: Fetches real data from Yahoo Finance, computes features
   python visualizer.py -s AAPL -i 15m -p 5d

2. REPORT MODE: Replays a backtest report (exact same trades)
   python main.py backtest -s bos -i 15m --start 2024-10-01  # generates report
   python visualizer.py --report backtest_report.json          # replays it

Open http://localhost:5050 in your browser.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

REPORT_DATA = None  # Loaded backtest report (if --report mode)


def fetch_live_data(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """Fetch OHLCV from Yahoo Finance, compute features + signals."""
    import yfinance as yf
    from features import (
        find_swing_points, detect_market_structure, find_support_resistance,
        atr, rsi, relative_volume, vwap, bollinger_bands,
        realized_volatility, volatility_regime, ema, rate_of_change,
        StructureBreak,
    )

    print(f"  Fetching {symbol} | period={period} | interval={interval}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return {"error": f"No data for {symbol}"}

    try:
        name = ticker.info.get("shortName", symbol)
    except Exception:
        name = symbol

    print(f"  {name}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    h, l, c, o = df["High"].values, df["Low"].values, df["Close"].values, df["Open"].values
    v = df["Volume"].values.astype(float)

    # Features
    atr_14 = atr(h, l, c, 14)
    rsi_14 = rsi(c, 14)
    rvol_20 = relative_volume(v, 20)
    vwap_v = vwap(h, l, c, v)
    ema_9, ema_21, ema_50 = ema(c, 9), ema(c, 21), ema(c, 50)
    bb_u, bb_m, bb_l = bollinger_bands(c, 20, 2.0)

    swing_lb = max(3, min(7, len(c) // 50))
    swings = find_swing_points(h, l, swing_lb, swing_lb)
    sr_levels = find_support_resistance(h, l, c, swings)

    # Structure at each bar
    structures = []
    for i in range(len(c)):
        if i < 30:
            structures.append({"trend": "ranging", "bos": "none", "hh": 0, "hl": 0, "lh": 0, "ll": 0, "last_sh": None, "last_sl": None})
            continue
        s = detect_market_structure(h[:i+1], l[:i+1], c[:i+1],
                                     [sw for sw in swings if sw.index <= i], swing_lb, 2)
        structures.append({
            "trend": s["trend"],
            "bos": s["structure_break"].value if hasattr(s["structure_break"], "value") else "none",
            "hh": s["higher_highs"], "hl": s["higher_lows"],
            "lh": s["lower_highs"], "ll": s["lower_lows"],
            "last_sh": float(s["last_swing_high"].price) if s["last_swing_high"] else None,
            "last_sl": float(s["last_swing_low"].price) if s["last_swing_low"] else None,
        })

    # BOS signals
    signals = []
    in_pos = None
    for i in range(30, len(c)):
        st = structures[i]
        rv = float(rvol_20[i]) if not np.isnan(rvol_20[i]) else 1.0
        rs = float(rsi_14[i]) if not np.isnan(rsi_14[i]) else 50.0
        at = float(atr_14[i]) if not np.isnan(atr_14[i]) else 0

        if not in_pos:
            if "bullish" in st["bos"] and rv > 1.0 and rs < 72:
                signals.append({"index": i, "type": "buy", "reason": st["bos"], "price": float(c[i])})
                in_pos = {"type": "long", "entry": float(c[i]), "stop": st["last_sl"] or float(c[i]) - 2*at}
            elif "bearish" in st["bos"] and rv > 1.0 and rs > 28:
                signals.append({"index": i, "type": "sell", "reason": st["bos"], "price": float(c[i])})
                in_pos = {"type": "short", "entry": float(c[i]), "stop": st["last_sh"] or float(c[i]) + 2*at}
        else:
            if in_pos["type"] == "long":
                pnl = (float(c[i]) - in_pos["entry"]) / in_pos["entry"]
                if float(c[i]) <= in_pos["stop"] or pnl >= 0.05 or "bearish" in st["bos"]:
                    reason = "stop" if float(c[i]) <= in_pos["stop"] else "target" if pnl >= 0.05 else "reversal"
                    signals.append({"index": i, "type": "sell", "reason": reason, "price": float(c[i])})
                    in_pos = None
            else:
                pnl = (in_pos["entry"] - float(c[i])) / in_pos["entry"]
                if float(c[i]) >= in_pos["stop"] or pnl >= 0.05 or "bullish" in st["bos"]:
                    reason = "stop" if float(c[i]) >= in_pos["stop"] else "target" if pnl >= 0.05 else "reversal"
                    signals.append({"index": i, "type": "buy", "reason": reason, "price": float(c[i])})
                    in_pos = None

    sf = lambda x: None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 4)
    is_intra = interval not in ("1d", "1wk", "1mo")

    return {
        "symbol": symbol, "name": name, "interval": interval, "period": period,
        "bars": [{"date": ts.strftime("%Y-%m-%d %H:%M") if is_intra else ts.strftime("%Y-%m-%d"),
                  "open": round(float(o[i]),2), "high": round(float(h[i]),2),
                  "low": round(float(l[i]),2), "close": round(float(c[i]),2),
                  "volume": int(v[i])} for i, ts in enumerate(df.index)],
        "features": {
            "atr_14": [sf(x) for x in atr_14], "rsi_14": [sf(x) for x in rsi_14],
            "rvol_20": [sf(x) for x in rvol_20], "vwap": [sf(x) for x in vwap_v],
            "ema_9": [sf(x) for x in ema_9], "ema_21": [sf(x) for x in ema_21],
            "ema_50": [sf(x) for x in ema_50],
            "bb_upper": [sf(x) for x in bb_u], "bb_mid": [sf(x) for x in bb_m],
            "bb_lower": [sf(x) for x in bb_l],
        },
        "swings": [{"index": s.index, "price": round(s.price, 2), "type": s.swing_type.value} for s in swings],
        "structures": structures,
        "sr_levels": [{"price": round(lv.price,2), "type": lv.level_type, "touches": lv.touches, "strength": round(lv.strength,2)} for lv in sr_levels[:10]],
        "signals": signals,
        "paired_trades": [],
        "equity_curve": [],
    }


def load_report(path: str) -> dict:
    """Load a backtest report JSON for replay."""
    with open(path) as f:
        report = json.load(f)

    # Restructure for the visualizer
    meta = report.get("meta", {})
    symbols = meta.get("symbols", [])
    first_sym = symbols[0] if symbols else "?"
    bars_dict = report.get("bars", {})
    bars = bars_dict.get(first_sym, [])

    # Compute features from the bar data
    if bars:
        from features import (
            find_swing_points, detect_market_structure, find_support_resistance,
            atr, rsi, relative_volume, vwap, bollinger_bands, ema, StructureBreak,
        )

        h = np.array([b["high"] for b in bars])
        l = np.array([b["low"] for b in bars])
        c = np.array([b["close"] for b in bars])
        o = np.array([b["open"] for b in bars])
        v = np.array([b["volume"] for b in bars], dtype=float)

        atr_14 = atr(h, l, c, 14)
        rsi_14 = rsi(c, 14)
        rvol_20 = relative_volume(v, 20)
        vwap_v = vwap(h, l, c, v)
        ema_9, ema_21, ema_50 = ema(c, 9), ema(c, 21), ema(c, 50)
        bb_u, bb_m, bb_l = bollinger_bands(c, 20, 2.0)

        swing_lb = max(3, min(7, len(c) // 50))
        swings = find_swing_points(h, l, swing_lb, swing_lb)
        sr_levels = find_support_resistance(h, l, c, swings)

        structures = []
        for i in range(len(c)):
            if i < 30:
                structures.append({"trend":"ranging","bos":"none","hh":0,"hl":0,"lh":0,"ll":0,"last_sh":None,"last_sl":None})
                continue
            s = detect_market_structure(h[:i+1], l[:i+1], c[:i+1],
                                         [sw for sw in swings if sw.index <= i], swing_lb, 2)
            structures.append({
                "trend": s["trend"],
                "bos": s["structure_break"].value if hasattr(s["structure_break"], "value") else "none",
                "hh": s["higher_highs"], "hl": s["higher_lows"],
                "lh": s["lower_highs"], "ll": s["lower_lows"],
                "last_sh": float(s["last_swing_high"].price) if s["last_swing_high"] else None,
                "last_sl": float(s["last_swing_low"].price) if s["last_swing_low"] else None,
            })

        sf = lambda x: None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 4)

        features = {
            "atr_14": [sf(x) for x in atr_14], "rsi_14": [sf(x) for x in rsi_14],
            "rvol_20": [sf(x) for x in rvol_20], "vwap": [sf(x) for x in vwap_v],
            "ema_9": [sf(x) for x in ema_9], "ema_21": [sf(x) for x in ema_21],
            "ema_50": [sf(x) for x in ema_50],
            "bb_upper": [sf(x) for x in bb_u], "bb_mid": [sf(x) for x in bb_m],
            "bb_lower": [sf(x) for x in bb_l],
        }
    else:
        features = {}
        swings = []
        structures = []
        sr_levels = []

    return {
        "symbol": first_sym,
        "name": f"{first_sym} — {', '.join(meta.get('strategies', ['unknown']))}",
        "interval": meta.get("interval", "1d"),
        "period": f"{meta.get('start', '?')} → {meta.get('end', '?')}",
        "bars": bars,
        "features": features,
        "swings": [{"index": s.index, "price": round(s.price, 2), "type": s.swing_type.value} for s in swings],
        "structures": structures,
        "sr_levels": [{"price": round(lv.price,2), "type": lv.level_type, "touches": lv.touches, "strength": round(lv.strength,2)} for lv in sr_levels[:10]],
        "signals": report.get("signals", []),
        "paired_trades": report.get("paired_trades", []),
        "equity_curve": report.get("equity_curve", []),
        "metrics": report.get("metrics", {}),
    }


# ============================================================
#  HTML — loaded from dashboard.html
# ============================================================
def get_html():
    """Load the professional dashboard HTML."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return html_path.read_text()
    else:
        return "<html><body><h1>Error: dashboard.html not found</h1><p>Expected at: " + str(html_path) + "</p></body></html>"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/data":
            q = urllib.parse.parse_qs(parsed.query)
            sym = q.get("symbol", ["AAPL"])[0].upper()
            per = q.get("period", ["1y"])[0]
            itv = q.get("interval", ["1d"])[0]
            data = fetch_live_data(sym, per, itv)
            self._json(data)
        elif parsed.path == "/api/report":
            if REPORT_DATA:
                self._json(REPORT_DATA)
            else:
                self._json({"error": "No report loaded"})
        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(get_html().encode())
        else:
            self.send_error(404)

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, *a): pass


def main():
    global REPORT_DATA

    parser = argparse.ArgumentParser(description="QuantBot Visual Backtester")
    parser.add_argument("--symbol", "-s", default="AAPL", help="Ticker symbol")
    parser.add_argument("--period", "-p", default="1y", help="Period: 5d, 1mo, 3mo, 6mo, 1y")
    parser.add_argument("--interval", "-i", default="1d", help="Interval: 5m, 15m, 1h, 1d")
    parser.add_argument("--report", "-r", default=None, help="Load backtest report JSON")
    parser.add_argument("--port", type=int, default=5050, help="Server port")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  ◆ QUANTBOT — INSTITUTIONAL BACKTESTER")
    print(f"{'='*55}")

    if args.report:
        print(f"  Mode: REPORT REPLAY")
        print(f"  File: {args.report}")
        REPORT_DATA = load_report(args.report)
        print(f"  Loaded: {REPORT_DATA['symbol']} | {len(REPORT_DATA['bars'])} bars | {len(REPORT_DATA['signals'])} signals")
        url = f"http://localhost:{args.port}?mode=report"
    else:
        print(f"  Mode: LIVE DATA")
        print(f"  Symbol: {args.symbol} | Period: {args.period} | Interval: {args.interval}")
        fetch_live_data(args.symbol, args.period, args.interval)  # verify
        url = f"http://localhost:{args.port}?symbol={args.symbol}&period={args.period}&interval={args.interval}"

    print(f"\n  ✓ Open: {url}")
    print(f"  ✓ Ctrl+C to stop")
    print(f"{'='*55}\n")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()