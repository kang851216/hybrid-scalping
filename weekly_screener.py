#!/usr/bin/env python3
"""
Weekly Stock Screener for RSI(2) Scalping
==========================================
Run every weekend to select stocks for the coming week.
Uses LAST 60 DAYS of data (not future data) for all filters.
"""
import yfinance as yf, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

# Broader candidate pool — liquid US stocks
CANDIDATES = [
    # Tech
    'NVDA','AMD','AVGO','INTC','MU','TXN','QCOM','AMAT','ADI','LRCX',
    # Internet/Cloud
    'META','GOOGL','NFLX','AMZN','CRM','ADBE','NOW','SNOW','DDOG','NET',
    # Hardware/Consumer
    'AAPL','MSFT','TSLA','DELL','HPQ','LOGI','GRMN',
    # Finance
    'JPM','GS','MS','BAC','C','BLK','SCHW','V','MA',
    # Healthcare
    'PFE','JNJ','MRK','ABBV','LLY','UNH','CVS','ISRG','REGN','VRTX',
    # Consumer/Retail
    'KO','PEP','PG','WMT','COST','HD','MCD','SBUX','NKE','LULU',
    # Energy/Industrial
    'XOM','CVX','COP','CAT','DE','GE','BA','RTX','LMT','HON',
    # Telecom/Utils
    'VZ','T','TMUS','SO','DUK','NEE',
]

def screen_stocks(lookback_days=60):
    """Screen stocks using only PAST data (no forward-looking bias)."""
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=lookback_days)
    
    results = []
    
    for sym in CANDIDATES:
        try:
            df = yf.download(sym, start=start.strftime('%Y-%m-%d'), 
                           end=end.strftime('%Y-%m-%d'), interval='1h', 
                           progress=False, auto_adjust=True)
            if len(df) < 100: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            cols = {}
            for c in df.columns:
                cl = str(c).lower()
                if 'close' in cl: cols['close'] = c
                elif 'high' in cl: cols['high'] = c
                elif 'low' in cl: cols['low'] = c
                elif 'open' in cl: cols['open'] = c
                elif 'volume' in cl: cols['volume'] = c
            
            c = df[cols['close']]; h = df[cols['high']]; l = df[cols['low']]
            
            # ── FILTER 1: Price Range ──────────────────
            price = float(c.iloc[-1])
            if price < 20 or price > 600: continue
            
            # ── FILTER 2: ADX (range-bound check) ──────
            tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.ewm(span=14, adjust=False).mean()
            up = (h - h.shift(1)).clip(lower=0)
            dn = (l.shift(1) - l).clip(lower=0)
            pDM = up.where((up > dn) & (up > 0), 0)
            mDM = dn.where((dn > up) & (dn > 0), 0)
            pDI = (pDM.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10)) * 100
            mDI = (mDM.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10)) * 100
            dx = (abs(pDI - mDI) / (pDI + mDI).replace(0, 1)) * 100
            adx = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])
            if adx > 25: continue  # Too trendy
            
            # ── FILTER 3: ATR% (minimum volatility) ────
            atr_pct = float(atr.iloc[-1] / price * 100)
            if atr_pct < 0.15: continue  # Too quiet
            
            # ── FILTER 4: Gap Risk ─────────────────────
            gap = abs(c - c.shift(1)) / c.shift(1) * 100
            max_gap = float(gap.max())
            if max_gap > 12: continue  # Too gappy
            
            # ── FILTER 5: Trend direction ──────────────
            ret_30d = (c.iloc[-1] / c.iloc[-min(len(c),130)] - 1) * 100  # ~30 trading days
            if ret_30d < -15 and adx > 15: continue  # Persistent downtrend
            
            # ── SCORE: Higher is better ────────────────
            # Prefer: lower ADX, higher ATR%, moderate price, lower gap
            adx_score = max(0, 25 - adx) / 25 * 30      # 0-30 pts
            vol_score = min(atr_pct / 1.0, 1) * 25       # 0-25 pts
            gap_score = max(0, 12 - max_gap) / 12 * 15   # 0-15 pts
            price_score = (1 - abs(price-200)/400) * 10  # 0-10 pts (prefer $200)
            ret_score = max(0, (ret_30d + 15) / 30) * 10 # 0-10 pts
            liq_score = min(float(df[cols.get('volume',cols['close'])].iloc[-1]) / 5e6, 1) * 10  # 0-10 pts
            
            total_score = adx_score + vol_score + gap_score + price_score + ret_score + liq_score
            
            results.append({
                'symbol': sym,
                'price': price,
                'adx': round(adx, 1),
                'atr_pct': round(atr_pct, 3),
                'max_gap': round(max_gap, 2),
                'ret_30d': round(ret_30d, 1),
                'score': round(total_score, 1),
            })
        except:
            continue
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

if __name__ == '__main__':
    print("=" * 75)
    print("  WEEKLY STOCK SCREENER — RSI(2) Scalping Candidates")
    print(f"  Screened: {len(CANDIDATES)} stocks | Using last 60 days data")
    print("=" * 75)
    
    screened = screen_stocks()
    
    if not screened:
        print("\n  No stocks passed filters. Market may be too trendy.")
        print("  Consider: wait for range-bound conditions or reduce ADX threshold.")
    else:
        print(f"\n  Passed: {len(screened)}/{len(CANDIDATES)} stocks")
        print(f"\n  {'Rank':<5} {'Symbol':<8} {'Price':>8} {'ADX':>6} {'ATR%':>7} {'Gap%':>7} {'30dRet':>8} {'Score':>7}")
        print(f"  {'-'*60}")
        
        for i, s in enumerate(screened[:15]):
            mark = '⭐' if s['score'] > 70 else ('✅' if s['score'] > 50 else '  ')
            print(f"  {mark} {i+1:<3} {s['symbol']:<8} ${s['price']:>7,.2f} {s['adx']:>5.1f} {s['atr_pct']:>6.3f}% {s['max_gap']:>6.2f}% {s['ret_30d']:>+7.1f}% {s['score']:>6.1f}")
        
        print(f"\n  [Recommended Action]")
        top_n = min(5, len(screened))
        top = [s['symbol'] for s in screened[:top_n]]
        print(f"  Trade these {top_n} stocks this week: {', '.join(top)}")
        print(f"  Parameters: SL=0.2%, TP=2.5%, RSI<25/>75, Pos=90%")
        print(f"  Capital: $5,000 per stock = ${5000*top_n:,} total")
        print(f"  Re-screen next weekend")
