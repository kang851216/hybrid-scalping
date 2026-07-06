#!/usr/bin/env python3
"""
Cloud Scalping Engine — Google Cloud Deployable
=================================================
Run every 6 hours: screen stocks → trade simulation → save results.
"""
import yfinance as yf, pandas as pd, numpy as np, json, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
CONFIG = {
    "fee_per_side": 1.00,
    "slippage": 0.00005,
    "init_cap": 5000.0,
    # ATR-based dynamic SL/TP
    "sl_atr_mult": 0.5,
    "tp_sl_ratio": 10,
    "min_sl_pct": 0.001,
    "max_sl_pct": 0.008,
    "rsi_period": 2,
    "rsi_long": 25,
    "rsi_short": 75,
    # === NEW: Risk Management ===
    "kelly_fraction": 0.5,      # 50% of full Kelly (conservative)
    "max_pos_pct": 0.90,        # absolute max position
    "trail_be_pct": 0.5,        # move SL to BE after 50% of TP reached
    "vix_halt": 30,             # halt all trading if VIX > this
    "vix_reduce": 25,           # reduce position 50% if VIX > this
    "max_daily_loss_pct": 3.0,  # halt if daily loss exceeds 3%
    "max_weekly_loss_pct": 8.0, # halt for rest of week if exceeded
    "max_stocks": 5,
    "lookback_days": 60,
    "trade_window_hours": 6,
    "output_dir": "static",
}

CANDIDATES = [
    'NVDA','AMD','AVGO','INTC','MU','TXN','QCOM','AMAT','ADI','LRCX',
    'META','GOOGL','NFLX','AMZN','CRM','ADBE','NOW','SNOW','DDOG','NET',
    'AAPL','MSFT','TSLA','DELL','HPQ','LOGI','GRMN',
    'JPM','GS','MS','BAC','C','BLK','SCHW','V','MA',
    'PFE','JNJ','MRK','ABBV','LLY','UNH','CVS','ISRG','REGN','VRTX',
    'KO','PEP','PG','WMT','COST','HD','MCD','SBUX','NKE','LULU',
    'XOM','CVX','COP','CAT','DE','GE','BA','RTX','LMT','HON',
    'VZ','T','TMUS','SO','DUK','NEE',
]

# ═══════════════════════════════════════════════════
# STOCK SCREENER
# ═══════════════════════════════════════════════════
def screen_stocks():
    end = datetime.now()
    start = end - timedelta(days=CONFIG['lookback_days'])
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
            
            c = df[cols['close']]; h = df[cols['high']]; l = df[cols['low']]
            price = float(c.iloc[-1])
            if price < 20 or price > 600: continue
            
            # ADX
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
            if adx > 25: continue
            
            # ATR%
            atr_pct = float(atr.iloc[-1] / price * 100)
            if atr_pct < 0.40: continue  # too quiet for scalping
            
            # Gap risk
            gap = abs(c - c.shift(1)) / c.shift(1) * 100
            max_gap = float(gap.max())
            if max_gap > 12: continue
            
            # Trend
            ret_30d = (c.iloc[-1] / c.iloc[-min(len(c),130)] - 1) * 100
            if ret_30d < -15 and adx > 15: continue
            
            # Score
            adx_score = max(0, 25 - adx) / 25 * 30
            vol_score = min(atr_pct / 1.0, 1) * 25
            gap_score = max(0, 12 - max_gap) / 12 * 15
            price_score = (1 - abs(price-200)/400) * 10
            ret_score = max(0, (ret_30d + 15) / 30) * 10
            total_score = adx_score + vol_score + gap_score + price_score + ret_score
            
            results.append({
                'symbol': sym, 'price': round(price, 2),
                'adx': round(adx, 1), 'atr_pct': round(atr_pct, 3),
                'max_gap': round(max_gap, 2), 'ret_30d': round(ret_30d, 1),
                'score': round(total_score, 1),
            })
        except:
            continue
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:CONFIG['max_stocks']]

# ═══════════════════════════════════════════════════
# BACKTEST ENGINE (Upgraded: Kelly + Multi-TF + Trailing + VIX + Time Filter + Loss Limit)
# ═══════════════════════════════════════════════════
def simulate_trading(symbol, window_hours=6):
    end = datetime.now()
    start = end - timedelta(hours=window_hours + 100)

    # Fetch 5-min data (primary timeframe)
    try:
        df_raw = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), interval='5m',
                       progress=False, auto_adjust=True)
        if len(df_raw) < 50: return None
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = [c[0].lower() for c in df_raw.columns]
        else:
            df_raw.columns = [c.lower() for c in df_raw.columns]

        # Find OHLC columns
        col_names = list(df_raw.columns)
        o_col = next((c for c in col_names if 'open' in str(c).lower()), col_names[0])
        h_col = next((c for c in col_names if 'high' in str(c).lower()), col_names[1])
        l_col = next((c for c in col_names if 'low' in str(c).lower()), col_names[2])
        c_col = next((c for c in col_names if 'close' in str(c).lower()), col_names[3])

        df = pd.DataFrame({
            'open': pd.to_numeric(df_raw[o_col], errors='coerce'),
            'high': pd.to_numeric(df_raw[h_col], errors='coerce'),
            'low':  pd.to_numeric(df_raw[l_col], errors='coerce'),
            'close':pd.to_numeric(df_raw[c_col], errors='coerce'),
        })
        df = df.dropna()
        if len(df) < 20: return None
    except:
        return None

    c = df['close'].values.ravel(); h = df['high'].values.ravel()
    l = df['low'].values.ravel(); o = df['open'].values.ravel()
    n = len(c)

    # ── IMPROVEMENT 2: Multi-TF RSI (15min + 1h) ──
    # RSI on 5-min data (used for 15-min equivalent = 3 bars lookback)
    d5 = np.diff(c, prepend=c[0])
    g5 = np.where(d5 > 0, d5, 0.); lo5 = np.where(d5 < 0, -d5, 0.)
    ag5 = np.zeros(n); al5 = np.zeros(n)
    if n > 2: ag5[2] = np.mean(g5[1:3]); al5[2] = np.mean(lo5[1:3])
    for i in range(3, n):
        ag5[i] = (ag5[i-1] + g5[i]) / 2
        al5[i] = (al5[i-1] + lo5[i]) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        rs5 = np.where(al5 != 0, ag5/al5, 100.)
        rsi_short = np.where(al5 != 0, 100. - 100./(1.+rs5), 100.)
    rsi_short = np.nan_to_num(rsi_short, nan=50.)

    # RSI on 15-min bars (resampled from 5-min, 3 bars = 15min)
    df_15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    c15 = df_15['close'].values
    if len(c15) > 3:
        d15 = np.diff(c15, prepend=c15[0])
        g15 = np.where(d15 > 0, d15, 0.); lo15 = np.where(d15 < 0, -d15, 0.)
        ag15 = np.zeros(len(c15)); al15 = np.zeros(len(c15))
        if len(c15) > 2: ag15[2] = np.mean(g15[1:3]); al15[2] = np.mean(lo15[1:3])
        for i in range(3, len(c15)):
            ag15[i] = (ag15[i-1] + g15[i]) / 2
            al15[i] = (al15[i-1] + lo15[i]) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            rs15 = np.where(al15 != 0, ag15/al15, 100.)
            rsi_long = np.where(al15 != 0, 100. - 100./(1.+rs15), 100.)
        rsi_long = np.nan_to_num(rsi_long, nan=50.)

        # Map 15-min RSI back to 5-min index
        rsi_15_mapped = np.full(n, 50.)
        df_15_idx = df_15.index
        for i in range(len(df)):
            ts = df.index[i]
            match = df_15_idx[df_15_idx <= ts]
            if len(match) > 0:
                pos = len(match) - 1
                if pos < len(rsi_long):
                    rsi_15_mapped[i] = rsi_long[pos]
    else:
        rsi_15_mapped = np.full(n, 50.)

    # ── IMPROVEMENT 4: VIX Filter ──
    vix_val = 20  # default
    try:
        vix_df = yf.download('^VIX', start=start.strftime('%Y-%m-%d'),
                            end=end.strftime('%Y-%m-%d'), interval='5m',
                            progress=False, auto_adjust=True)
        if len(vix_df) > 0:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0].lower() for c in vix_df.columns]
            else:
                vix_df.columns = [c.lower() for c in vix_df.columns]
            for vcol in vix_df.columns:
                if 'close' in str(vcol).lower():
                    vix_val = float(vix_df[vcol].iloc[-1])
                    break
    except:
        pass

    # VIX-based trading state
    vix_halted = vix_val > CONFIG['vix_halt']
    vix_reduced = vix_val > CONFIG['vix_reduce']
    base_pos = CONFIG['max_pos_pct'] * 0.5 if vix_reduced else CONFIG['max_pos_pct']

    # ── HYBRID: Regime Detection (5-Day Return > 0 = BH) ──
    df_daily = df['close'].resample('1D').last().dropna()
    if len(df_daily) >= 5:
        cd_d = df_daily.values
        # Map to bar-level: 0=scalp, 1=buy&hold
        regime = np.zeros(n, dtype=int)
        for i in range(n):
            bar_day = df.index[i].date()
            day_idx = None
            for j in range(len(df_daily.index)):
                if df_daily.index[j].date() == bar_day:
                    day_idx = j; break
            if day_idx is not None and day_idx >= 5:
                uptrend = cd_d[day_idx] > cd_d[max(0, day_idx - 5)]
                if uptrend:
                    regime[i] = 1
    else:
        regime = np.zeros(n, dtype=int)
    # ── END HYBRID ──

    # ── Signals with Multi-TF confirmation ──
    sig = np.zeros(n, dtype=int)
    for i in range(10, n):
        if vix_halted: continue
        # Check time-of-day (skip open/close 30 min)
        ts = df.index[i]
        hour_frac = ts.hour + ts.minute / 60
        if hour_frac < 10.0 or hour_frac >= 15.5: continue  # skip 9:30-10:00 and 15:30-16:00

        # Multi-TF: both 15-min and 5-min RSI agree
        short_oversold = rsi_short[i] < CONFIG['rsi_long']
        long_oversold = rsi_15_mapped[i] < CONFIG['rsi_long']
        short_overbought = rsi_short[i] > CONFIG['rsi_short']
        long_overbought = rsi_15_mapped[i] > CONFIG['rsi_short']

        if short_oversold and long_oversold:
            sig[i] = 1
        # SHORT signals disabled — LONG only strategy

    # ATR
    tr_arr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr_arr[0] = h[0] - l[0]
    atr = pd.Series(tr_arr).ewm(span=14, adjust=False).mean().values
    atr_pct_arr = atr / np.where(c > 0, c, 1)

    # ── Backtest with Kelly + Trailing + Daily Loss ──
    fee = CONFIG['fee_per_side']; slp = CONFIG['slippage']
    cap = float(CONFIG['init_cap'])
    trades = []; eq = [cap]
    ip, posd = False, None
    daily_pnl = 0.0; current_day = None
    weekly_pnl = 0.0; current_week = None
    halted = False
    # Hybrid state
    in_bh = False; bh_ep = None; bh_sh = None
    bh_entries = 0; bh_exits = 0

    for i in range(1, n):
        if halted or vix_halted: break

        # ── Daily/Weekly loss tracking ──
        bar_day = df.index[i].date()
        bar_week = df.index[i].isocalendar()[1]
        if current_day is None: current_day = bar_day
        if current_week is None: current_week = bar_week

        if bar_day != current_day:
            if abs(daily_pnl) > CONFIG['init_cap'] * CONFIG['max_daily_loss_pct'] / 100:
                halted = True; break
            daily_pnl = 0.0; current_day = bar_day

        if bar_week != current_week:
            if abs(weekly_pnl) > CONFIG['init_cap'] * CONFIG['max_weekly_loss_pct'] / 100:
                halted = True; break
            weekly_pnl = 0.0; current_week = bar_week

        # ── HYBRID: Regime switching with Rally Detection ──
        current_regime = regime[i] if i < len(regime) else 0

        # Enter BH when trend is confirmed (EMA20>EMA50 + ADX>18)
        enter_bh = (current_regime == 1 and not in_bh)

        if enter_bh:
            # Close any open scalp position
            if ip:
                p = posd
                xp = c[i-1]
                gp = p['sh'] * (xp - p['ep']) if p['d'] == 'LONG' else p['sh'] * (p['ep'] - xp)
                gp -= p['sh'] * p['ep'] * slp
                trade_pnl = gp - 2*fee
                cap += gp - fee
                daily_pnl += trade_pnl; weekly_pnl += trade_pnl
                trades.append({
                    'time': str(df.index[i]), 'dir': p['d'],
                    'result': 'SWITCH_TO_BH', 'entry': round(p['ep'], 2),
                    'exit': round(xp, 2), 'pnl': round(trade_pnl, 2),
                    'equity': round(cap, 2),
                })
                eq.append(cap)
                ip, posd = False, None
            # Enter Buy & Hold
            bh_ep = o[i]
            bh_sh = cap * 0.90 / bh_ep
            cap -= fee
            in_bh = True
            bh_entries += 1
            continue

        elif current_regime == 0 and in_bh:
            # Exit Buy & Hold
            xp = o[i]
            gp = bh_sh * (xp - bh_ep)
            cap += gp - fee
            trades.append({
                'time': str(df.index[i]), 'dir': 'BH_EXIT',
                'result': 'WIN' if gp > 0 else 'LOSS',
                'entry': round(bh_ep, 2), 'exit': round(xp, 2),
                'pnl': round(gp - 2*fee, 2), 'equity': round(cap, 2),
            })
            eq.append(cap)
            in_bh = False; bh_sh = None
            bh_exits += 1

        if in_bh: continue  # Hold position during uptrend

        # ── Scalping mode ──
        if not ip:
            s = sig[i-1]
            if s == 0: continue
            ep = o[i]
            dyn_sl = np.clip(atr_pct_arr[i-1] * CONFIG['sl_atr_mult'],
                           CONFIG['min_sl_pct'], CONFIG['max_sl_pct']) if i > 0 else CONFIG['min_sl_pct']
            dyn_tp = dyn_sl * CONFIG['tp_sl_ratio']

            # LONG only — no SHORT signals
            ee = ep * (1 + slp)
            sl_pr = ee * (1 - dyn_sl); tp_pr = ee * (1 + dyn_tp); d = 'LONG'

            # ── IMPROVEMENT 1: Kelly Position Sizing ──
            win_prob = 0.20  # base estimate
            win_loss_ratio = dyn_tp / dyn_sl
            kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_f = max(0.05, min(kelly_f, 0.5)) * CONFIG['kelly_fraction']
            pos_size = min(kelly_f, base_pos)

            pz = cap * pos_size; sh = pz / ee
            cap -= fee
            posd = {'d': d, 'ep': ee, 'ei': i, 'sl': sl_pr, 'tp': tp_pr,
                    'sh': sh, 'trailed': False, 'entry_cap': cap + fee}
            ip = True
        else:
            p = posd

            # ── IMPROVEMENT 3: Trailing Stop ──
            if not p['trailed'] and p['d'] == 'LONG':
                progress = (h[i] - p['ep']) / (p['tp'] - p['ep']) if p['tp'] != p['ep'] else 0
                if progress >= CONFIG['trail_be_pct']:
                    p['sl'] = p['ep']  # move SL to breakeven
                    p['trailed'] = True
            elif not p['trailed'] and p['d'] == 'SHORT':
                progress = (p['ep'] - l[i]) / (p['ep'] - p['tp']) if p['ep'] != p['tp'] else 0
                if progress >= CONFIG['trail_be_pct']:
                    p['sl'] = p['ep']  # move SL to breakeven
                    p['trailed'] = True

            # Standard exit check — LONG only
            if l[i] <= p['sl']: xp, xr = p['sl'], 'BE' if p['trailed'] else 'SL'
            elif h[i] >= p['tp']: xp, xr = p['tp'], 'TP'
            else: continue

            gp = p['sh'] * (xp - p['ep'])
            gp -= p['sh'] * p['ep'] * slp
            trade_pnl = gp - 2*fee
            cap += gp - fee
            daily_pnl += trade_pnl; weekly_pnl += trade_pnl

            trades.append({
                'time': str(df.index[i]),
                'dir': p['d'], 'result': 'WIN' if trade_pnl > 0 else ('BE' if xr == 'BE' else 'LOSS'),
                'entry': round(p['ep'], 2), 'exit': round(xp, 2),
                'pnl': round(trade_pnl, 2),
                'equity': round(cap, 2),
            })
            eq.append(cap)
            ip, posd = False, None

    # Close any open BH position
    if in_bh and bh_sh is not None:
        xp = c[-1] if n > 0 else bh_ep
        gp = bh_sh * (xp - bh_ep)
        cap += gp - fee
        trades.append({
            'time': str(df.index[-1]) if n > 0 else '',
            'dir': 'BH_CLOSE', 'result': 'WIN' if gp > 0 else 'LOSS',
            'entry': round(bh_ep, 2), 'exit': round(xp, 2),
            'pnl': round(gp - 2*fee, 2), 'equity': round(cap, 2),
        })

    # Metrics
    n_trades = len(trades)
    winners = [t for t in trades if t['pnl'] > 0]
    wr = len(winners) / n_trades * 100 if n_trades > 0 else 0
    net_ret = (cap / CONFIG['init_cap'] - 1) * 100

    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = (eq_arr - peak) / peak * 100
    dd = np.where(np.isfinite(dd), dd, 0)
    mdd = abs(np.min(dd))

    return {
        'symbol': symbol,
        'price': round(float(c[-1]), 2),
        'trades': n_trades,
        'winners': len(winners),
        'win_rate': round(wr, 1),
        'return': round(net_ret, 2),
        'mdd': round(mdd, 2),
        'final_equity': round(cap, 2),
        'equity_curve': [round(x, 2) for x in eq[::max(1,len(eq)//50)]],
        'trade_list': trades[-20:],
        'rsi': [round(x, 1) for x in rsi_short[-100:].tolist()],
        'price_history': [round(x, 2) for x in c[-100:].tolist()],
        'vix': round(vix_val, 1),
        'vix_halted': vix_halted,
        'vix_reduced': vix_reduced,
        'hybrid_bh_entries': bh_entries,
        'hybrid_bh_exits': bh_exits,
        'hybrid_active': in_bh,
    }

# ═══════════════════════════════════════════════════
# MAIN — runs every 6 hours
# ═══════════════════════════════════════════════════
def run_cycle():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Starting 6-hour cycle...")
    
    # 1. Screen stocks
    print("  Screening stocks...")
    picks = screen_stocks()
    print(f"  Selected: {[p['symbol'] for p in picks]}")
    
    # 2. Simulate trading
    print("  Simulating trading...")
    portfolio = []
    for pick in picks:
        result = simulate_trading(pick['symbol'])
        if result:
            result['screen_score'] = pick['score']
            portfolio.append(result)
            print(f"    {pick['symbol']}: {result['trades']} trades, {result['return']:+.2f}%")
        else:
            print(f"    {pick['symbol']}: no data")
    
    # 3. Portfolio summary
    total_trades = sum(p['trades'] for p in portfolio)
    avg_return = np.mean([p['return'] for p in portfolio]) if portfolio else 0
    total_wins = sum(p['winners'] for p in portfolio)
    total_t = sum(p['trades'] for p in portfolio)
    avg_wr = total_wins/total_t*100 if total_t > 0 else 0
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stocks': [p['symbol'] for p in picks],
        'scores': {p['symbol']: p['score'] for p in picks},
        'total_trades': total_trades,
        'avg_return': round(avg_return, 2),
        'avg_win_rate': round(avg_wr, 1),
        'avg_mdd': round(np.mean([p['mdd'] for p in portfolio]), 2) if portfolio else 0,
        'portfolio': portfolio,
    }
    
    # 4. Save to JSON
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    with open(f"{CONFIG['output_dir']}/results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 5. Update history
    history_file = f"{CONFIG['output_dir']}/history.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    
    history.append({
        'time': summary['timestamp'],
        'return': summary['avg_return'],
        'trades': summary['total_trades'],
        'stocks': summary['stocks'],
    })
    
    # Keep last 100 entries
    history = history[-100:]
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"  Done. Portfolio: {avg_return:+.2f}% | {total_trades} trades")
    return summary

if __name__ == '__main__':
    run_cycle()
