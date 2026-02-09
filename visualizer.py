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
        "name": f"{first_sym} (backtest: {meta.get('strategies', ['?'])})",
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
#  HTML
# ============================================================
def get_html():
    return r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>QuantBot Visual Backtester</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#e2e8f0;font-family:'JetBrains Mono',monospace;overflow:hidden}
#app{display:flex;flex-direction:column;height:100vh}
.hdr{padding:10px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b;background:#0d1220}
.logo{font-size:18px;font-weight:700;color:#22d3ee;letter-spacing:2px}
.sub{font-size:11px;color:#64748b;letter-spacing:1px;margin-left:12px}
.tb{padding:8px 20px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #1e293b;background:#0d1220}
.btn{background:#334155;color:#e2e8f0;border:none;border-radius:4px;padding:6px 12px;cursor:pointer;font:600 12px inherit;transition:all .15s}
.btn:hover{filter:brightness(1.2)}.btn.on{background:#1e40af}.btn.play{background:#22c55e}.btn.pause{background:#dc2626}
.btn-sm{font-size:10px;padding:4px 8px}
.inp{background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:5px 10px;font:12px inherit;width:90px;text-transform:uppercase}
.inp:focus{outline:none;border-color:#22d3ee}
select.sel{background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:5px 8px;font:11px inherit}
.main{display:flex;flex:1;min-height:0}
.chart{flex:1;position:relative}
.chart canvas{display:block}
.sb{width:270px;border-left:1px solid #1e293b;overflow-y:auto;background:#0d1220;font-size:11px}
.pn{padding:10px 12px;border-bottom:1px solid #1e293b}
.ph{font-size:10px;color:#64748b;letter-spacing:1px;margin-bottom:6px;font-weight:600}
.big{font-size:26px;font-weight:700}.bull{color:#22c55e}.bear{color:#ef4444}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:3px 10px}.lb{color:#64748b}.vl{color:#94a3b8;text-align:right}
.tag{display:inline-block;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:.5px}
.tag.bullish{background:#166534;color:#4ade80}.tag.bearish{background:#7f1d1d;color:#fca5a5}.tag.ranging{background:#374151;color:#9ca3af}
.tag.bos-b{background:#065f46;color:#6ee7b7}.tag.bos-r{background:#7c2d12;color:#fdba74}
.sr-s{color:#06b6d4}.sr-r{color:#f97316}
.ld{display:flex;align-items:center;justify-content:center;height:100vh;font-size:16px;color:#64748b}
.sp{width:30px;height:30px;border:3px solid #1e293b;border-top-color:#22d3ee;border-radius:50%;animation:spin .8s linear infinite;margin-right:12px}
@keyframes spin{to{transform:rotate(360deg)}}
.rbar{height:6px;background:#1e293b;border-radius:3px;position:relative;margin-top:2px}
.rdot{position:absolute;top:-2px;width:10px;height:10px;border-radius:50%;transform:translateX(-50%)}
</style></head><body>
<div id="app"><div class="ld" id="ld"><div class="sp"></div><span>Loading...</span></div></div>
<script>
let D=null,ci=60,playing=false,pI=null,spd=150;
// roundRect polyfill
if(!CanvasRenderingContext2D.prototype.roundRect){CanvasRenderingContext2D.prototype.roundRect=function(x,y,w,h,r){r=Math.min(r,w/2,h/2);this.moveTo(x+r,y);this.lineTo(x+w-r,y);this.quadraticCurveTo(x+w,y,x+w,y+r);this.lineTo(x+w,y+h-r);this.quadraticCurveTo(x+w,y+h,x+w-r,y+h);this.lineTo(x+r,y+h);this.quadraticCurveTo(x,y+h,x,y+h-r);this.lineTo(x,y+r);this.quadraticCurveTo(x,y,x+r,y);}}
const tog={ema:true,bb:false,vwap:true,swings:true,sr:true,vol:true,trades:true};

async function loadData(sym,per,itv){
  document.getElementById("ld").style.display="flex";
  try{
    const r=await fetch(`/api/data?symbol=${sym}&period=${per}&interval=${itv}`);
    D=await r.json();if(D.error){alert(D.error);return}
    ci=Math.min(60,D.bars.length-1);
    document.getElementById("ld").style.display="none";
    buildUI();render();
  }catch(e){alert("Load failed: "+e.message)}
}

function drawChart(){
  const cv=document.getElementById("cv");if(!cv||!D)return;
  const ct=cv.parentElement;cv.width=ct.clientWidth;cv.height=ct.clientHeight;
  const x=cv.getContext("2d"),W=cv.width,H=cv.height;
  x.fillStyle="#0a0e17";x.fillRect(0,0,W,H);
  const sV=tog.vol,cT=30,cB=sV?H-100:H-40,cH=cB-cT,vT=cB+10,vB=H-10,vH=vB-vT;
  const lb=Math.min(ci+1,120),si=Math.max(0,ci-lb+1),sl=D.bars.slice(si,ci+1);
  if(!sl.length)return;
  let mn=1e15,mx=-1e15,mV=0;
  sl.forEach(b=>{mn=Math.min(mn,b.low);mx=Math.max(mx,b.high);mV=Math.max(mV,b.volume)});
  const pd=(mx-mn)*.08;mn-=pd;mx+=pd;
  const bW=(W-80)/sl.length;
  const px=i=>50+(i-si)*bW+bW/2, py=p=>cT+cH*(1-(p-mn)/(mx-mn)), vy=v=>vB-(v/mV)*vH;
  // grid
  x.strokeStyle="#1a2035";x.lineWidth=.5;
  for(let i=0;i<=5;i++){const p=mn+(mx-mn)*i/5,y=py(p);x.beginPath();x.moveTo(50,y);x.lineTo(W-30,y);x.stroke();
    x.fillStyle="#4a5568";x.font="11px 'JetBrains Mono'";x.textAlign="right";x.fillText("$"+p.toFixed(2),46,y+4)}
  // BB
  if(tog.bb&&D.features.bb_upper){
    x.fillStyle="rgba(99,102,241,.06)";x.beginPath();let s=0;
    for(let i=si;i<=ci;i++){const u=D.features.bb_upper[i];if(u==null)continue;if(!s){x.moveTo(px(i),py(u));s=1}else x.lineTo(px(i),py(u))}
    for(let i=ci;i>=si;i--){const l=D.features.bb_lower[i];if(l==null)continue;x.lineTo(px(i),py(l))}
    x.closePath();x.fill();
    x.strokeStyle="rgba(99,102,241,.3)";x.lineWidth=1;x.setLineDash([3,3]);
    ["bb_upper","bb_lower"].forEach(k=>{x.beginPath();s=0;for(let i=si;i<=ci;i++){const v=D.features[k][i];if(v==null)continue;if(!s){x.moveTo(px(i),py(v));s=1}else x.lineTo(px(i),py(v))}x.stroke()});
    x.setLineDash([])}
  // VWAP
  if(tog.vwap&&D.features.vwap){x.strokeStyle="#f59e0b";x.lineWidth=1.5;x.setLineDash([5,3]);x.beginPath();let s=0;
    for(let i=si;i<=ci;i++){const v=D.features.vwap[i];if(v==null)continue;if(!s){x.moveTo(px(i),py(v));s=1}else x.lineTo(px(i),py(v))}
    x.stroke();x.setLineDash([])}
  // EMA
  if(tog.ema){[{k:"ema_9",c:"#22d3ee"},{k:"ema_21",c:"#f472b6"},{k:"ema_50",c:"#a78bfa"}].forEach(({k,c})=>{
    if(!D.features[k])return;x.strokeStyle=c;x.lineWidth=1.2;x.beginPath();let s=0;
    for(let i=si;i<=ci;i++){const v=D.features[k][i];if(v==null)continue;if(!s){x.moveTo(px(i),py(v));s=1}else x.lineTo(px(i),py(v))}x.stroke()})}
  // candles
  sl.forEach((b,i)=>{const idx=si+i,xp=px(idx),bl=b.close>=b.open;
    const bt=py(Math.max(b.open,b.close)),bb=py(Math.min(b.open,b.close)),bh=Math.max(bb-bt,1);
    x.strokeStyle=bl?"#22c55e":"#ef4444";x.lineWidth=1;x.beginPath();x.moveTo(xp,py(b.high));x.lineTo(xp,py(b.low));x.stroke();
    x.fillStyle=bl?"#22c55e":"#ef4444";x.fillRect(xp-bW*.35,bt,bW*.7,bh);
    if(idx===ci){x.strokeStyle="#fff";x.lineWidth=2;x.strokeRect(xp-bW*.45,bt-2,bW*.9,bh+4)}});
  // swings
  if(tog.swings){D.swings.forEach(s=>{if(s.index<si||s.index>ci)return;const xp=px(s.index),y=py(s.price),isH=s.type==="swing_high";
    x.fillStyle=isH?"#f97316":"#06b6d4";x.beginPath();
    if(isH){x.moveTo(xp,y-12);x.lineTo(xp-5,y-4);x.lineTo(xp+5,y-4)}else{x.moveTo(xp,y+12);x.lineTo(xp-5,y+4);x.lineTo(xp+5,y+4)}
    x.closePath();x.fill()})}
  // SR
  if(tog.sr){const act=D.swings.filter(s=>s.index<=ci&&s.index>=si-20);
    [...act.filter(s=>s.type==="swing_high").slice(-3),...act.filter(s=>s.type==="swing_low").slice(-3)].forEach(s=>{
    x.strokeStyle=s.type==="swing_high"?"rgba(249,115,22,.25)":"rgba(6,182,212,.25)";x.lineWidth=1;x.setLineDash([4,4]);
    x.beginPath();x.moveTo(px(s.index),py(s.price));x.lineTo(W-30,py(s.price));x.stroke();x.setLineDash([])})}
  // ===== TRADE SIGNALS & TRADE ZONES =====
  if(tog.trades){
  // First: draw trade zones (shaded area between entry and exit)
  const visibleSigs = D.signals.filter(sg=>sg.index>=si&&sg.index<=ci);
  // Pair signals into trades for zone drawing
  for(let t=0;t<visibleSigs.length-1;t++){
    const entry=visibleSigs[t], exit=visibleSigs[t+1];
    // entry=buy+exit=sell or entry=sell+exit=buy
    const isLong=(entry.type==="buy"&&exit.type==="sell");
    const isShort=(entry.type==="sell"&&exit.type==="buy");
    if(!isLong&&!isShort) continue;
    const ex1=px(entry.index),ex2=px(exit.index);
    const ey1=py(entry.price),ey2=py(exit.price);
    const win=isLong?(exit.price>entry.price):(entry.price>exit.price);
    // Shaded zone between entry and exit
    x.fillStyle=win?"rgba(34,197,94,0.07)":"rgba(239,68,68,0.07)";
    const zTop=Math.min(ey1,ey2)-5, zBot=Math.max(ey1,ey2)+5;
    x.fillRect(ex1,zTop,ex2-ex1,zBot-zTop);
    // Dashed connecting line
    x.strokeStyle=win?"rgba(34,197,94,0.5)":"rgba(239,68,68,0.5)";
    x.lineWidth=1.5;x.setLineDash([4,3]);
    x.beginPath();x.moveTo(ex1,ey1);x.lineTo(ex2,ey2);x.stroke();x.setLineDash([]);
    // PnL label at midpoint
    const mx=(ex1+ex2)/2, my=Math.min(ey1,ey2)-12;
    const pnl=isLong?((exit.price-entry.price)/entry.price*100):((entry.price-exit.price)/entry.price*100);
    x.font="bold 11px 'JetBrains Mono'";x.textAlign="center";
    x.fillStyle=win?"#22c55e":"#ef4444";
    x.fillText((pnl>=0?"+":"")+pnl.toFixed(1)+"%",mx,my);
    t++; // skip exit signal in next iteration
  }
  // Now draw the actual signal markers (big, impossible to miss)
  D.signals.forEach(sg=>{
    if(sg.index<si||sg.index>ci) return;
    const xp=px(sg.index), b=D.bars[sg.index];
    const isBuy=sg.type==="buy";
    const price=sg.price||b.close;
    const yPrice=py(price);
    const yArrow=isBuy?py(b.low)+8:py(b.high)-8;
    // Vertical highlight line
    x.strokeStyle=isBuy?"rgba(34,197,94,0.25)":"rgba(239,68,68,0.25)";
    x.lineWidth=1;x.setLineDash([2,2]);
    x.beginPath();x.moveTo(xp,cT);x.lineTo(xp,cB);x.stroke();x.setLineDash([]);
    // Big arrow
    const aSize=10;
    x.fillStyle=isBuy?"#22c55e":"#ef4444";
    x.beginPath();
    if(isBuy){
      x.moveTo(xp,yArrow+aSize*2.5);
      x.lineTo(xp-aSize,yArrow+aSize*2.5+aSize*1.5);
      x.lineTo(xp-aSize*.35,yArrow+aSize*2.5+aSize*1.5);
      x.lineTo(xp-aSize*.35,yArrow+aSize*4);
      x.lineTo(xp+aSize*.35,yArrow+aSize*4);
      x.lineTo(xp+aSize*.35,yArrow+aSize*2.5+aSize*1.5);
      x.lineTo(xp+aSize,yArrow+aSize*2.5+aSize*1.5);
    } else {
      x.moveTo(xp,yArrow-aSize*2.5);
      x.lineTo(xp-aSize,yArrow-aSize*2.5-aSize*1.5);
      x.lineTo(xp-aSize*.35,yArrow-aSize*2.5-aSize*1.5);
      x.lineTo(xp-aSize*.35,yArrow-aSize*4);
      x.lineTo(xp+aSize*.35,yArrow-aSize*4);
      x.lineTo(xp+aSize*.35,yArrow-aSize*2.5-aSize*1.5);
      x.lineTo(xp+aSize,yArrow-aSize*2.5-aSize*1.5);
    }
    x.closePath();x.fill();
    // Glow effect
    x.shadowColor=isBuy?"#22c55e":"#ef4444";x.shadowBlur=12;x.fill();x.shadowBlur=0;
    // Label badge (BUY/SELL with price)
    const label=isBuy?"BUY":"SELL";
    const labelY=isBuy?yArrow+aSize*4.5+14:yArrow-aSize*4.5-8;
    // Badge background
    const txt=`${label} $${price.toFixed(2)}`;
    x.font="bold 10px 'JetBrains Mono'";
    const tw=x.measureText(txt).width;
    const bx=xp-tw/2-6, by=labelY-10, bw=tw+12, bht=16;
    x.fillStyle=isBuy?"rgba(34,197,94,0.9)":"rgba(239,68,68,0.9)";
    x.beginPath();x.roundRect(bx,by,bw,bht,4);x.fill();
    // Badge text
    x.fillStyle="#fff";x.textAlign="center";
    x.fillText(txt,xp,labelY);
    // Reason tag
    if(sg.reason){
      x.font="9px 'JetBrains Mono'";x.fillStyle=isBuy?"rgba(34,197,94,0.7)":"rgba(239,68,68,0.7)";
      x.fillText(sg.reason.replace(/_/g," "),xp,labelY+(isBuy?13:-15));
    }
  });
  } // end tog.trades
  // volume
  if(sV){sl.forEach((b,i)=>{const idx=si+i,xp=px(idx),rv=D.features.rvol_20?D.features.rvol_20[idx]:1,bl=b.close>=b.open;
    const a=(rv!=null&&rv>1.5)?.8:.35;x.fillStyle=bl?`rgba(34,197,94,${a})`:`rgba(239,68,68,${a})`;
    const h=(b.volume/mV)*vH;x.fillRect(xp-bW*.35,vB-h,bW*.7,h)});
    x.fillStyle="#4a5568";x.font="9px 'JetBrains Mono'";x.textAlign="left";x.fillText("VOLUME",52,vT+10)}
  x.font="bold 13px 'JetBrains Mono'";x.fillStyle="#334155";x.textAlign="left";
  x.fillText(`${D.symbol} · ${D.interval}`,55,20);
}

function buildUI(){
  const a=document.getElementById("app");
  a.innerHTML=`<div class="hdr"><div style="display:flex;align-items:center"><span class="logo">◈ QUANTBOT</span><span class="sub">VISUAL BACKTESTER</span></div><div id="info" style="font-size:12px;color:#94a3b8"></div></div>
  <div class="tb" id="tb"></div><div class="main"><div class="chart"><canvas id="cv"></canvas></div><div class="sb" id="sb"></div></div>`;
  buildTB();window.addEventListener("resize",render);
}

function buildTB(){
  const t=document.getElementById("tb");
  t.innerHTML=`<input class="inp" id="ti" value="${D.symbol}" placeholder="Symbol"/>
  <select class="sel" id="ps"><option value="5d">5 Days</option><option value="1mo">1 Month</option><option value="3mo">3 Months</option><option value="6mo">6 Months</option><option value="1y">1 Year</option><option value="2y">2 Years</option></select>
  <select class="sel" id="is"><option value="5m">5 min</option><option value="15m">15 min</option><option value="1h">1 hour</option><option value="1d">Daily</option><option value="1wk">Weekly</option></select>
  <button class="btn" id="bl" onclick="doLoad()">LOAD</button>
  <span style="color:#1e293b">|</span>
  <button class="btn" onclick="go(0)">⏮</button><button class="btn" onclick="step(-1)">◀</button>
  <button class="btn" id="bp" onclick="togPlay()">▶ PLAY</button>
  <button class="btn" onclick="step(1)">▶</button><button class="btn" onclick="go(D.bars.length-1)">⏭</button>
  <span style="color:#64748b;font-size:11px;margin-left:4px">Speed:</span>
  <input type="range" min="20" max="500" value="350" style="width:80px;accent-color:#22d3ee" oninput="spd=500-+this.value"/>
  <div style="margin-left:auto;display:flex;gap:4px" id="tg"></div>`;
  // set current values
  const ps=document.getElementById("ps"),is2=document.getElementById("is");
  ps.value=D.period||"1y";is2.value=D.interval||"1d";
  // enter key on ticker input
  document.getElementById("ti").addEventListener("keydown",e=>{if(e.key==="Enter")doLoad()});
  // toggles
  const c=document.getElementById("tg");
  [["EMA","ema"],["BB","bb"],["VWAP","vwap"],["Swings","swings"],["S/R","sr"],["Vol","vol"],["Trades","trades"]].forEach(([l,k])=>{
    const b=document.createElement("button");b.className=`btn btn-sm ${tog[k]?"on":""}`;b.textContent=l;
    b.onclick=()=>{tog[k]=!tog[k];b.className=`btn btn-sm ${tog[k]?"on":""}`;render()};c.appendChild(b)});
  document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
    if(e.key==="ArrowRight")step(1);else if(e.key==="ArrowLeft")step(-1);
    else if(e.key===" "){e.preventDefault();togPlay()}});
}

function doLoad(){
  const sym=document.getElementById("ti").value.trim().toUpperCase();
  const per=document.getElementById("ps").value;
  const itv=document.getElementById("is").value;
  if(sym)loadData(sym,per,itv);
}

function step(d){ci=Math.max(0,Math.min(D.bars.length-1,ci+d));render()}
function go(i){ci=Math.max(0,Math.min(D.bars.length-1,i));render()}
function togPlay(){playing=!playing;
  if(playing){pI=setInterval(()=>{if(ci>=D.bars.length-1){playing=false;clearInterval(pI);updPB();return}ci++;render()},spd)}
  else clearInterval(pI);updPB()}
function updPB(){const b=document.getElementById("bp");if(b){b.textContent=playing?"⏸ PAUSE":"▶ PLAY";b.className=`btn ${playing?"pause":"play"}`}}
function render(){drawChart();updSB();updHdr()}

function updHdr(){const b=D.bars[ci],st=D.structures?D.structures[ci]:{trend:"?",bos:"none"};
  document.getElementById("info").innerHTML=`Bar ${ci+1}/${D.bars.length} | ${b.date} | <span style="color:${st.trend==='bullish'?'#22c55e':st.trend==='bearish'?'#ef4444':'#94a3b8'};font-weight:600">${st.trend.toUpperCase()}</span>`}

function updSB(){
  const b=D.bars[ci],st=D.structures?D.structures[ci]:{trend:"ranging",bos:"none",hh:0,hl:0,lh:0,ll:0,last_sh:null,last_sl:null};
  const f=D.features||{},bl=b.close>=b.open;
  const rs=f.rsi_14?f.rsi_14[ci]:null,at=f.atr_14?f.atr_14[ci]:null,rv=f.rvol_20?f.rvol_20[ci]:null,vw=f.vwap?f.vwap[ci]:null;
  // pnl
  let eq=100000,tr=0,wins=0,pos=null;
  const pt=D.paired_trades||[];
  if(pt.length>0){
    // use actual backtest paired trades
    pt.forEach(t=>{const idx=D.signals.find(s=>s.timestamp===t.exit_time||s.price===t.exit_price);
      tr++;if(t.win)wins++;eq+=t.net_pnl*.2});
  }else{
    for(const sg of D.signals){if(sg.index>ci)break;
      if(sg.type==="buy"&&!pos){pos={e:sg.price,t:"long"}}
      else if(sg.type==="sell"&&!pos){pos={e:sg.price,t:"short"}}
      else if(sg.type==="sell"&&pos?.t==="long"){const p=(sg.price-pos.e)/pos.e;eq*=(1+p*.2);tr++;if(p>0)wins++;pos=null}
      else if(sg.type==="buy"&&pos?.t==="short"){const p=(pos.e-sg.price)/pos.e;eq*=(1+p*.2);tr++;if(p>0)wins++;pos=null}}}
  const pp=((eq/100000)-1)*100,wr=tr>0?(wins/tr*100):0;
  const sigs=D.signals.filter(s=>s.index<=ci).slice(-8).reverse();
  const sb=document.getElementById("sb");
  sb.innerHTML=`
<div class="pn"><div class="ph">${D.name||D.symbol}</div>
<div class="big ${bl?'bull':'bear'}">$${b.close.toFixed(2)}</div>
<div class="g2" style="margin-top:6px"><span class="lb">O</span><span class="vl">$${b.open.toFixed(2)}</span><span class="lb">H</span><span class="vl">$${b.high.toFixed(2)}</span><span class="lb">L</span><span class="vl">$${b.low.toFixed(2)}</span><span class="lb">Vol</span><span class="vl">${(b.volume/1e6).toFixed(1)}M</span></div></div>
<div class="pn"><div class="ph">STRUCTURE</div>
<div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap"><span class="tag ${st.trend}">${st.trend.toUpperCase()}</span>${st.bos!=="none"?`<span class="tag ${st.bos.includes("bullish")?"bos-b":"bos-r"}">${st.bos.replace(/_/g," ").toUpperCase()}</span>`:""}</div>
<div class="g2"><span style="color:#22c55e">HH:${st.hh}</span><span style="color:#22c55e">HL:${st.hl}</span><span style="color:#ef4444">LH:${st.lh}</span><span style="color:#ef4444">LL:${st.ll}</span></div>
${st.last_sh?`<div style="color:#f97316;margin-top:4px">SH: $${st.last_sh.toFixed(2)}</div>`:""}${st.last_sl?`<div style="color:#06b6d4">SL: $${st.last_sl.toFixed(2)}</div>`:""}</div>
<div class="pn"><div class="ph">INDICATORS</div><div class="g2">
<span class="lb">RSI</span><span style="color:${rs>70?"#ef4444":rs<30?"#22c55e":"#94a3b8"};text-align:right">${rs?.toFixed(1)??"—"}</span>
<span class="lb">ATR</span><span class="vl">${at?.toFixed(2)??"—"}</span>
<span class="lb">RVOL</span><span style="color:${rv>1.5?"#fbbf24":"#94a3b8"};text-align:right">${rv?.toFixed(2)??"—"}x</span>
<span class="lb">VWAP</span><span class="vl">$${vw?.toFixed(2)??"—"}</span></div>
${rs!=null?`<div style="margin-top:8px"><div style="display:flex;justify-content:space-between;font-size:9px;color:#4a5568"><span>30</span><span>50</span><span>70</span></div><div class="rbar"><div class="rdot" style="left:${((rs-10)/80)*100}%;background:${rs>70?"#ef4444":rs<30?"#22c55e":"#94a3b8"}"></div></div></div>`:""}</div>
<div class="pn"><div class="ph">PERFORMANCE</div>
<div style="font-size:18px;font-weight:700;color:${eq>=100000?"#22c55e":"#ef4444"}">$${eq.toFixed(0)}</div>
<div style="color:${pp>=0?"#22c55e":"#ef4444"};font-size:13px">${pp>=0?"+":""}${pp.toFixed(2)}%</div>
<div class="g2" style="margin-top:6px"><span class="lb">Trades</span><span class="vl">${tr}</span><span class="lb">Win Rate</span><span style="color:${wr>50?"#22c55e":"#ef4444"};text-align:right">${wr.toFixed(0)}%</span></div></div>
<div class="pn"><div class="ph">SIGNAL LOG</div><div style="max-height:160px;overflow-y:auto">
${sigs.map(s=>`<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b;${s.index===ci?"font-weight:700":"opacity:.6"}"><span style="color:${s.type==="buy"?"#22c55e":"#ef4444"}">${s.type.toUpperCase()}</span><span style="color:#64748b">$${s.price.toFixed(2)}</span><span style="color:#4a5568;font-size:9px">${s.reason||""}</span></div>`).join("")}
${sigs.length===0?'<div style="color:#4a5568">No signals yet</div>':""}
</div></div>
<div style="padding:8px 12px;border-top:1px solid #1e293b;color:#4a5568;font-size:9px"><span style="color:#64748b">KEYS:</span> ← → step | Space play | Enter load ticker</div>`;
}

// INIT
const u=new URLSearchParams(window.location.search);
const mode=u.get("mode")||"live";
if(mode==="report"){
  fetch("/api/report").then(r=>r.json()).then(d=>{D=d;ci=Math.min(60,D.bars.length-1);document.getElementById("ld").style.display="none";buildUI();render()});
}else{
  loadData(u.get("symbol")||"AAPL",u.get("period")||"1y",u.get("interval")||"1d");
}
</script></body></html>"""


# ============================================================
#  HTTP Server
# ============================================================
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
    print(f"  ◈ QUANTBOT VISUAL BACKTESTER")
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