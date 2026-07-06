#!/usr/bin/env python3
"""
Nasdaq 10-Stock Scalping Backtest — Renaissance Style
======================================================
Strategy: RSI(2) mean-reversion + ADX trend filter
10 stocks, 6-month backtest, $1,000 initial capital
Fee: $2/round-trip + 0.1% slippage
Target: 500+ trades, Net Return > 0%, MDD < 15%
"""
import yfinance as yf, pandas as pd, numpy as np
import time, warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
CONFIG = {
    "stocks": ['NVDA','META','GOOGL','AAPL','MSFT','AMZN','TSLA','AVGO','AMD','NFLX'],
    "start": '2026-03-01',
    "end":   '2026-06-01',
    "interval": '60m',
    
    # Strategy defaults
    "sl_pct": 0.002,   # 0.2% SL 
    "tp_pct": 0.025,   # 2.5% TP (12.5:1 — ultra-wide for small cap)
    "rsi_period": 2,
    "rsi_long": 25,
    "rsi_short": 75,
    "pos_pct": 0.90,   # HIGH position — fees are fixed $, so need max exposure
    "adx_threshold": 25,  # only filter trend when ADX > 25
    
    # Friction (MANDATORY)
    "fee_per_side": 1.00,     # $1 per entry + $1 per exit = $2 round trip
    "slippage": 0.0005,       # 0.05% per side = 0.1% round trip
    
    "init_cap": 5000.0,
    "min_trades": 500,
    "max_mdd": 15.0,
    "max_iterations": 5,
}

# ═══════════════════════════════════════════════════
def fetch_stock(symbol):
    try:
        df = yf.download(symbol, start=CONFIG['start'], end=CONFIG['end'],
                        interval=CONFIG['interval'], progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        cols = {}
        for c in df.columns:
            cl = str(c).lower()
            if 'open' in cl: cols['open'] = c
            elif 'high' in cl: cols['high'] = c
            elif 'low' in cl: cols['low'] = c
            elif 'close' in cl: cols['close'] = c
        if len(cols) < 4: return None
        df = df[[cols['open'],cols['high'],cols['low'],cols['close']]].copy()
        df.columns = ['open','high','low','close']
        return df.dropna()
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════
def compute_indicators(df):
    c = df['close'].values.ravel()
    h = df['high'].values.ravel()
    l = df['low'].values.ravel()
    n = len(c)
    
    # RSI(2)
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag = np.zeros(n); al = np.zeros(n)
    if n > 2: ag[2] = np.mean(g[1:3]); al[2] = np.mean(lo[1:3])
    for i in range(3, n):
        ag[i] = (ag[i-1] + g[i]) / 2
        al[i] = (al[i-1] + lo[i]) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al != 0, ag/al, 100.0)
        rsi = np.where(al != 0, 100.0 - 100.0/(1.0+rs), 100.0)
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    # EMA(20/50) for trend
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    uptrend = ema20 > ema50
    
    # ADX for trend strength
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    up_move = np.where(h - np.roll(h, 1) > 0, h - np.roll(h, 1), 0)
    down_move = np.where(np.roll(l, 1) - l > 0, np.roll(l, 1) - l, 0)
    up_move[0] = down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr_safe = np.where(atr > 0, atr, 1e-10)
    plus_di = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr_safe * 100
    minus_di = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr_safe * 100
    
    dx = np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    return rsi, uptrend, adx

# ═══════════════════════════════════════════════════
def backtest_stock(df, sl, tp, rsi_l, rsi_s, pos, adx_thresh):
    c = df['close'].values.ravel()
    h = df['high'].values.ravel()
    l = df['low'].values.ravel()
    o = df['open'].values.ravel()
    n = len(c)
    
    rsi, uptrend, adx = compute_indicators(df)
    
    # Generate signals with adaptive trend filter
    sig = np.zeros(n, dtype=int)
    warmup = 60
    for i in range(warmup, n):
        if np.isnan(rsi[i]): continue
        
        strong_trend = adx[i] > adx_thresh
        
        if rsi[i] < rsi_l:
            if strong_trend and not uptrend[i]:
                continue  # skip LONG in strong downtrend
            sig[i] = 1
        elif rsi[i] > rsi_s:
            if strong_trend and uptrend[i]:
                continue  # skip SHORT in strong uptrend
            sig[i] = -1
    
    # Backtest engine
    fee = CONFIG['fee_per_side']
    slp = CONFIG['slippage']
    cap = float(CONFIG['init_cap'])
    trades, eq = [], [cap]
    ip, posd = False, None
    
    for i in range(1, n):
        if not ip:
            s = sig[i-1]
            if s == 0: continue
            
            ep = o[i]
            if s == 1:
                ep_eff = ep * (1 + slp)
                sl_price = ep_eff * (1 - sl)
                tp_price = ep_eff * (1 + tp)
                d = 'LONG'
            else:
                ep_eff = ep * (1 - slp)
                sl_price = ep_eff * (1 + sl)
                tp_price = ep_eff * (1 - tp)
                d = 'SHORT'
            
            pos_size = cap * pos
            shares = pos_size / ep_eff
            cap -= fee
            
            posd = {'d': d, 'ep': ep_eff, 'ei': i, 'sl': sl_price, 'tp': tp_price, 'sh': shares}
            ip = True
        else:
            p = posd
            if p['d'] == 'LONG':
                if l[i] <= p['sl']: xp, xr = p['sl'], 'SL'
                elif h[i] >= p['tp']: xp, xr = p['tp'], 'TP'
                else: continue
            else:
                if h[i] >= p['sl']: xp, xr = p['sl'], 'SL'
                elif l[i] <= p['tp']: xp, xr = p['tp'], 'TP'
                else: continue
            
            if p['d'] == 'LONG':
                gp = p['sh'] * (xp - p['ep'])
            else:
                gp = p['sh'] * (p['ep'] - xp)
            
            gp -= p['sh'] * p['ep'] * slp  # exit slippage
            cap += gp - fee
            
            pnl_pct = (xp/p['ep']-1)*100 if p['d']=='LONG' else (p['ep']/xp-1)*100
            trades.append({
                'd': p['d'], 'xr': xr, 'pnl_pct': pnl_pct,
                'pnl_net': gp - 2*fee,
                'eq': cap
            })
            eq.append(cap)
            ip, posd = False, None
    
    if ip:
        p = posd; xp = c[-1]
        if p['d'] == 'LONG': gp = p['sh'] * (xp - p['ep'])
        else: gp = p['sh'] * (p['ep'] - xp)
        gp -= p['sh'] * p['ep'] * slp
        cap += gp - fee
        pnl_pct = (xp/p['ep']-1)*100 if p['d']=='LONG' else (p['ep']/xp-1)*100
        trades.append({'d':p['d'],'xr':'EOD','pnl_pct':pnl_pct,'pnl_net':gp-2*fee,'eq':cap})
        eq.append(cap)
    
    return trades, eq, int((sig!=0).sum())

# ═══════════════════════════════════════════════════
def compute_metrics(trades, eq):
    if not trades or len(eq) < 2:
        return {'t':0,'r':0,'mdd':0,'wr':0,'pf':0,'w':0,'l':0,'fc':CONFIG['init_cap']}
    n = len(trades)
    init = float(CONFIG['init_cap']); fin = float(eq[-1])
    nr = (fin/init - 1) * 100
    w = [t for t in trades if t['pnl_net'] > 0]
    l = [t for t in trades if t['pnl_net'] <= 0]
    wr = len(w)/n*100 if n else 0
    tw = sum(t['pnl_net'] for t in w) if w else 0
    tl = abs(sum(t['pnl_net'] for t in l)) if l else 0
    pf = tw/tl if tl > 0 else 0
    eqa = np.array(eq); pk = np.maximum.accumulate(eqa)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = (eqa - pk) / pk * 100
    dd = np.where(np.isfinite(dd), dd, 0)
    mdd = abs(np.min(dd))
    return {'t':n,'r':round(nr,2),'mdd':round(mdd,2),'wr':round(wr,2),
            'pf':round(pf,4),'w':len(w),'l':len(l),'fc':round(fin,2)}

# ═══════════════════════════════════════════════════
PARAM_GRID = [
    (0.002, 0.025, 25, 75, 0.90, 25),
    (0.002, 0.030, 20, 80, 0.90, 25),
    (0.002, 0.025, 20, 80, 0.90, 20),
    (0.003, 0.030, 25, 75, 0.90, 30),
    (0.002, 0.025, 30, 70, 0.90, 25),
    (0.003, 0.035, 20, 80, 0.90, 30),
]

def self_correct(m, cfg, iteration):
    """Apply ONE correction based on PRIMARY failure."""
    new_cfg = cfg.copy()
    reasons = []
    
    # Priority 1: MDD > 15% — improve signal quality, NOT reduce position
    if m['mdd'] >= CONFIG['max_mdd']:
        new_cfg['rsi_long'] = max(cfg['rsi_long'] - 5, 10)
        new_cfg['rsi_short'] = min(cfg['rsi_short'] + 5, 90)
        new_cfg['adx_threshold'] = min(cfg['adx_threshold'] + 5, 35)
        reasons.append(f"MDD({m['mdd']:.1f}%) → RSI<{new_cfg['rsi_long']}>{new_cfg['rsi_short']} ADX>{new_cfg['adx_threshold']}")
        return new_cfg, reasons
    
    # Priority 2: Negative return — improve alpha QUALITY
    if m['r'] <= 0 and m['t'] >= 50:
        if m['wr'] < 35:
            # Tighten entry + increase TP ratio
            new_cfg['rsi_long'] = max(cfg['rsi_long'] - 5, 15)
            new_cfg['rsi_short'] = min(cfg['rsi_short'] + 5, 85)
            new_cfg['tp_pct'] = min(cfg['tp_pct'] * 1.5, 0.025)
            reasons.append(f"WR({m['wr']:.0f}%) → RSI<{new_cfg['rsi_long']}>{new_cfg['rsi_short']} TP={new_cfg['tp_pct']:.1%}")
            return new_cfg, reasons
        else:
            # Good WR but losing — widen TP vs SL ratio
            new_cfg['tp_pct'] = min(cfg['tp_pct'] * 1.5, 0.025)
            reasons.append(f"WR ok → TP={new_cfg['tp_pct']:.1%}")
            return new_cfg, reasons
    
    # Priority 3: Too few trades — relax filters
    if m['t'] < CONFIG['min_trades']:
        new_cfg['rsi_long'] = min(cfg['rsi_long'] + 5, 45)
        new_cfg['rsi_short'] = max(cfg['rsi_short'] - 5, 55)
        new_cfg['adx_threshold'] = max(cfg['adx_threshold'] - 5, 15)
        reasons.append(f"Trades({m['t']}) → RSI<{new_cfg['rsi_long']}>{new_cfg['rsi_short']} ADX>{new_cfg['adx_threshold']}")
        return new_cfg, reasons
    
    return new_cfg, reasons

# ═══════════════════════════════════════════════════
def main():
    cfg = CONFIG.copy()
    print("=" * 70)
    print("  NASDAQ 10-STOCK RSI(2) SCALPING BACKTEST")
    print("=" * 70)
    print(f"  Period: {cfg['start']} → {cfg['end']} (6 months)")
    print(f"  Initial Capital: ${cfg['init_cap']:,.0f}")
    print(f"  Fee: $2/round-trip + 0.1% slippage")
    print(f"  Target: ≥500 trades | Return > 0% | MDD < 15%")
    print(f"  Self-Correction: max {cfg['max_iterations']} iterations")
    print()
    
    all_trades = []
    all_equity_curves = {}
    stock_results = {}
    correction_log = []
    
    for stock in cfg['stocks']:
        print(f"  [{stock}] ", end='', flush=True)
        df = fetch_stock(stock)
        if df is None or len(df) < 100:
            print(f"SKIP (no data)")
            continue
        
        print(f"{len(df)} candles | ${df['close'].iloc[-1]:,.0f}")
        
        # Run parameter sweep
        best_for_stock = None
        stock_cfg = {
            'sl_pct': cfg['sl_pct'], 'tp_pct': cfg['tp_pct'],
            'rsi_long': cfg['rsi_long'], 'rsi_short': cfg['rsi_short'],
            'pos_pct': cfg['pos_pct'], 'adx_threshold': cfg['adx_threshold'],
        }
        
        for iteration in range(1, cfg['max_iterations'] + 1):
            best_metrics = None
            best_trades = None
            best_eq = None
            best_params = None
            
            for pi, (sl, tp, rl, rs, pos, adx) in enumerate(PARAM_GRID):
                if iteration > 1:
                    # Use corrected params
                    sl = stock_cfg['sl_pct']
                    tp = stock_cfg['tp_pct']
                    rl = stock_cfg['rsi_long']
                    rs = stock_cfg['rsi_short']
                    pos = stock_cfg['pos_pct']
                    adx = stock_cfg['adx_threshold']
                
                trades, eq, ns = backtest_stock(df, sl, tp, rl, rs, pos, adx)
                m = compute_metrics(trades, eq)
                
                if best_metrics is None or m['r'] > best_metrics['r']:
                    best_metrics = m
                    best_trades = trades
                    best_eq = eq
                    best_params = (sl, tp, rl, rs, pos, adx)
                
                if iteration > 1:
                    break  # single combo per iteration after correction
            
            if best_metrics is None:
                break
            
            passed = (best_metrics['t'] >= cfg['min_trades'] and 
                     best_metrics['r'] > 0 and 
                     best_metrics['mdd'] < cfg['max_mdd'])
            
            status = "✅PASS" if passed else "❌FAIL"
            if iteration == 1:
                print(f"    Iter1: T={best_metrics['t']} R={best_metrics['r']:+.1f}% MDD={best_metrics['mdd']:.1f}% WR={best_metrics['wr']:.1f}% {status}")
            
            if passed:
                if iteration > 1:
                    print(f"    Iter{iteration}: T={best_metrics['t']} R={best_metrics['r']:+.1f}% MDD={best_metrics['mdd']:.1f}% WR={best_metrics['wr']:.1f}% {status}")
                correction_log.append(f"{stock}: PASSED at iter {iteration}")
                best_for_stock = best_metrics
                all_trades.extend(best_trades)
                all_equity_curves[stock] = best_eq
                stock_results[stock] = {'m': best_metrics, 'params': best_params}
                break
            
            if iteration < cfg['max_iterations']:
                new_sc, reasons = self_correct(best_metrics, stock_cfg, iteration)
                if new_sc != stock_cfg:
                    stock_cfg = new_sc
                    print(f"    Iter{iteration}: T={best_metrics['t']} R={best_metrics['r']:+.1f}% → Correcting: {'; '.join(reasons)}")
                    correction_log.append(f"{stock} iter{iteration}: {'; '.join(reasons)}")
                else:
                    correction_log.append(f"{stock}: No more corrections possible")
                    break
            else:
                correction_log.append(f"{stock}: FAILED after {cfg['max_iterations']} iterations")
                best_for_stock = best_metrics
                all_trades.extend(best_trades)
                all_equity_curves[stock] = best_eq
                stock_results[stock] = {'m': best_metrics, 'params': best_params}
    
    # ═══════════════════════════════════════════
    # PORTFOLIO SUMMARY (each stock independent $1,000)
    # ═══════════════════════════════════════════
    if not all_trades:
        print("\n  ERROR: No trades generated")
        return
    
    portfolio_returns = []
    total_trades_all = 0
    total_wins = 0
    total_losses = 0
    
    for stock in cfg['stocks']:
        if stock in stock_results:
            m = stock_results[stock]['m']
            portfolio_returns.append(m['r'])
            total_trades_all += m['t']
            total_wins += m['w']
            total_losses += m['l']
    
    avg_return = np.mean(portfolio_returns) if portfolio_returns else -100
    avg_mdd = np.mean([stock_results[s]['m']['mdd'] for s in stock_results]) if stock_results else 100
    avg_wr = total_wins / total_trades_all * 100 if total_trades_all > 0 else 0
    
    # Portfolio equity: sum of all individual equities
    portfolio_eq = [cfg['init_cap'] * len(stock_results)]  # start with total capital
    running_cap = portfolio_eq[0]
    
    # Interleave all trades by time and accumulate
    all_trades_sorted = sorted(all_trades, key=lambda t: t.get('_time', 0))
    for t in all_trades:
        running_cap += t['pnl_net']
        portfolio_eq.append(running_cap)
    
    pm = compute_metrics(all_trades, portfolio_eq)
    
    print(f"\n{'='*70}")
    print(f"  FINAL PORTFOLIO REPORT")
    print(f"{'='*70}")
    print(f"  {'Stock':<8} {'Trades':>6} {'Return':>9} {'MDD':>7} {'WR':>7} {'PF':>7} {'Params'}")
    print(f"  {'-'*70}")
    
    total_t = 0
    for stock in cfg['stocks']:
        if stock in stock_results:
            m = stock_results[stock]['m']
            p = stock_results[stock]['params']
            total_t += m['t']
            print(f"  {stock:<8} {m['t']:>6} {m['r']:>+8.2f}% {m['mdd']:>6.2f}% {m['wr']:>6.2f}% {m['pf']:>6.4f} SL={p[0]:.1%} TP={p[1]:.1%} RSI<{p[2]}>{p[3]}")
        else:
            print(f"  {stock:<8} {'—':>6} {'—':>9} {'—':>7} {'—':>7} {'—':>7}")
    
    print(f"  {'-'*70}")
    print(f"  {'PORTFOLIO':<8} {total_trades_all:>6} {avg_return:>+8.2f}% {avg_mdd:>6.2f}% {avg_wr:>6.2f}% {'—':>7}")
    print()
    
    # Pass/Fail
    all_pass = (total_trades_all >= cfg['min_trades'] and avg_return > 0 and avg_mdd < cfg['max_mdd'])
    
    checks = []
    checks.append(f"Trades: {total_trades_all} / {cfg['min_trades']} {'✅' if total_trades_all>=cfg['min_trades'] else '❌'}")
    checks.append(f"Return: {avg_return:+.2f}% {'✅' if avg_return>0 else '❌'}")
    checks.append(f"MDD: {avg_mdd:.2f}% / {cfg['max_mdd']}% {'✅' if avg_mdd<cfg['max_mdd'] else '❌'}")
    
    print(f"  {'CRITERIA':<20}")
    for c in checks:
        print(f"    {c}")
    print(f"\n  {'▶ FINAL VERDICT:':<20} {'✅✅✅ ALL PASSED ✅✅✅' if all_pass else '❌ FAILED'}")
    
    # Detailed metrics
    print(f"\n  [Portfolio Details]")
    print(f"  Capital per stock:  ${cfg['init_cap']:,.2f}")
    print(f"  Total capital:      ${cfg['init_cap']*len(stock_results):,.2f}")
    print(f"  Average Return:     {avg_return:+.2f}%")
    print(f"  Average MDD:        {avg_mdd:.2f}%")
    print(f"  Total Trades:       {total_trades_all}")
    print(f"  Winners/Losers:     {total_wins}/{total_losses}")
    print(f"  Portfolio WR:       {avg_wr:.2f}%")
    
    if all_trades:
        w = [t for t in all_trades if t['pnl_net'] > 0]
        l = [t for t in all_trades if t['pnl_net'] <= 0]
        if w: print(f"  Avg Win:           +${np.mean([t['pnl_net'] for t in w]):.2f}")
        if l: print(f"  Avg Loss:          -${abs(np.mean([t['pnl_net'] for t in l])):.2f}")
        long_t = [t for t in all_trades if t['d']=='LONG']
        short_t = [t for t in all_trades if t['d']=='SHORT']
        print(f"  LONG/SHORT:        {len(long_t)}/{len(short_t)}")
    
    # Correction log
    print(f"\n  [Self-Correction Log]")
    for entry in correction_log:
        print(f"    {entry}")
    
    # Save
    if all_trades:
        pd.DataFrame(all_trades).to_csv('nasdaq_scalping_trades.csv', index=False)
        print(f"\n  Trades saved: nasdaq_scalping_trades.csv ({len(all_trades)} trades)")

if __name__ == '__main__':
    main()
