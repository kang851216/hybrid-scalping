#!/usr/bin/env python3
"""US Market Engine — 5D Hybrid Scalping"""
import yfinance as yf, pandas as pd, numpy as np, json, os, time, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

CONFIG = {
    "fee_per_side": 1.00,
    "slippage": 0.00005,
    "init_cap": 1000.0,
    "max_stocks": 3,
    "state_file": "us_state.json",
}

CANDIDATES = [
    'NVDA', 'AMD', 'AVGO', 'INTC', 'MU', 'TXN',
    'QCOM', 'AMAT', 'LRCX', 'KLAC', 'ADI', 'MRVL',
    'NXPI', 'MCHP', 'MPWR', 'ON', 'SWKS', 'QRVO',
    'TER', 'STM', 'ASML', 'SNPS', 'CDNS', 'ANET',
    'FSLR', 'ENPH', 'OLED', 'COHR', 'RMBS', 'SLAB',
    'META', 'GOOGL', 'NFLX', 'AMZN', 'CRM', 'ADBE',
    'AAPL', 'MSFT', 'ORCL', 'NOW', 'INTU', 'ADSK',
    'WDAY', 'TEAM', 'DDOG', 'SNOW', 'CRWD', 'ZS',
    'PANW', 'FTNT', 'NET', 'OKTA', 'MDB', 'SPLK',
    'PLTR', 'DELL', 'HPQ', 'IBM', 'CSCO', 'JNPR',
    'FFIV', 'NTAP', 'STX', 'WDC', 'LOGI', 'XRX',
    'PSTG', 'NTNX', 'SMCI', 'HPE', 'JPM', 'GS',
    'MS', 'BAC', 'C', 'WFC', 'USB', 'PNC',
    'TFC', 'COF', 'BK', 'STT', 'NTRS', 'AMP',
    'BLK', 'SCHW', 'BX', 'KKR', 'APO', 'ARES',
    'V', 'MA', 'AXP', 'DFS', 'SYF', 'BRK-B',
    'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'AFL',
    'LNC', 'PFG', 'UNM', 'PFE', 'JNJ', 'MRK',
    'ABBV', 'LLY', 'BMY', 'GILD', 'AMGN', 'BIIB',
    'REGN', 'VRTX', 'MRNA', 'ALNY', 'IONS', 'ILMN',
    'TMO', 'DHR', 'A', 'WAT', 'MTD', 'UNH',
    'CI', 'HUM', 'CVS', 'CNC', 'MOH', 'ELV',
    'HCA', 'UHS', 'THC', 'DGX', 'LH', 'MCK',
    'ABC', 'CAH', 'KO', 'PEP', 'PG', 'WMT',
    'COST', 'CL', 'CLX', 'KMB', 'MDLZ', 'K',
    'GIS', 'SJM', 'CAG', 'CPB', 'HSY', 'MCD',
    'SBUX', 'NKE', 'HD', 'LOW', 'TJX', 'ROST',
    'BBY', 'ULTA', 'AZO', 'ORLY', 'DHI', 'LEN',
    'PHM', 'TOL', 'RL', 'TPR', 'DECK', 'SKX',
    'CROX', 'XOM', 'CVX', 'COP', 'EOG', 'PXD',
    'DVN', 'HES', 'FANG', 'MRO', 'OXY', 'SLB',
    'HAL', 'BKR', 'NOV', 'WMB', 'CAT', 'DE',
    'GE', 'BA', 'HON', 'MMM', 'ITW', 'EMR',
    'ROK', 'ETN', 'PH', 'CMI', 'IR', 'DOV',
    'PNR', 'AME', 'ROP', 'LII', 'AOS', 'MAS',
    'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'HEI',
    'TDG', 'TXT', 'HII', 'CW', 'UNP', 'CSX',
    'NSC', 'FDX', 'UPS', 'DAL', 'UAL', 'LUV',
    'JBHT', 'ODFL', 'SO', 'DUK', 'NEE', 'D',
    'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ETR',
    'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'LUMN',
    'IRDM', 'FYBR', 'LIN', 'APD', 'SHW', 'ECL',
    'NEM', 'FCX', 'DOW', 'DD', 'PPG', 'IFF',
    'PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'O',
    'DLR', 'WELL', 'DIS', 'SPOT', 'LYV', 'WBD',
    'PARA', 'FOXA', 'RBLX', 'TTWO', 'TSLA', 'F',
    'GM', 'RIVN', 'LCID', 'LI', 'XPEV', 'NIO',
    'CMG', 'YUM', 'DPZ', 'QSR', 'TXRH', 'WING',
    'EBAY', 'ETSY', 'CHWY', 'CVNA', 'DASH',
]

def load_state():
    if os.path.exists(CONFIG['state_file']):
        with open(CONFIG['state_file']) as f: return json.load(f)
    return {'stocks': [], 'last_screen': None, 'last_rebalance_month': None}

def save_state(state):
    tmp = CONFIG['state_file'] + '.tmp'
    with open(tmp, 'w') as f: json.dump(state, f, indent=2, default=str)
    os.replace(tmp, CONFIG['state_file'])

def compute_rsi(close, period=2):
    n = len(close); d = np.diff(close, prepend=close[0])
    g = np.where(d > 0, d, 0.); lo = np.where(d < 0, -d, 0.)
    ag = np.zeros(n); al = np.zeros(n)
    if n > period: ag[period] = np.mean(g[1:period+1]); al[period] = np.mean(lo[1:period+1])
    for i in range(period+1, n):
        ag[i] = (ag[i-1]*(period-1) + g[i]) / period
        al[i] = (al[i-1]*(period-1) + lo[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al != 0, ag/al, 100.)
        return np.nan_to_num(100. - 100./(1.+np.where(al != 0, rs, 100.)), nan=50.)

def compute_regime(df):
    dfd = df['close'].resample('1D').last().dropna(); cd = dfd.values; n = len(df)
    regime = np.zeros(n, dtype=int)
    for i in range(n):
        bar_day = df.index[i].date(); day_idx = None
        for j in range(len(dfd.index)):
            if dfd.index[j].date() == bar_day: day_idx = j; break
        if day_idx is not None and day_idx >= 5:
            if cd[day_idx] > cd[max(0, day_idx - 5)]: regime[i] = 1
    return regime

def screen_market(candidates):
    end = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
    start = end - pd.Timedelta(days=30)
    results = []
    for sym in candidates:
        try:
            df = yf.download(sym, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                           interval='1h', progress=False, auto_adjust=True)
            if len(df) < 100: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
            else: df.columns = [c.lower() for c in df.columns]
            cn = list(df.columns)
            cc = next((c for c in cn if 'close' in str(c).lower()), cn[3])
            c_arr = pd.to_numeric(df[cc], errors='coerce')
            price = float(c_arr.iloc[-1])
            atr_pct = float((pd.concat([df.iloc[:,1]-df.iloc[:,2],
                (df.iloc[:,1]-c_arr.shift(1)).abs(),(df.iloc[:,2]-c_arr.shift(1)).abs()],axis=1)
                .max(axis=1).ewm(span=14,adjust=False).mean().iloc[-1]) / price * 100)
            if atr_pct < 0.40: continue
            up=(df.iloc[:,1]-df.iloc[:,1].shift(1)).clip(0);dn=(df.iloc[:,2].shift(1)-df.iloc[:,2]).clip(0)
            pDM=up.where((up>dn)&(up>0),0);mDM=dn.where((dn>up)&(dn>0),0)
            tr_s=pd.concat([df.iloc[:,1]-df.iloc[:,2],(df.iloc[:,1]-c_arr.shift(1)).abs(),(df.iloc[:,2]-c_arr.shift(1)).abs()],axis=1).max(axis=1)
            atr=tr_s.ewm(span=14,adjust=False).mean()
            pDI=(pDM.ewm(span=14,adjust=False).mean()/atr.replace(0,1e-10))*100;mDI=(mDM.ewm(span=14,adjust=False).mean()/atr.replace(0,1e-10))*100
            dx=(abs(pDI-mDI)/(pDI+mDI).replace(0,1))*100;adx=float(dx.ewm(span=14,adjust=False).mean().iloc[-1])
            if adx>25:continue
            ret_30d=(c_arr.iloc[-1]/c_arr.iloc[-min(len(c_arr),130)]-1)*100
            score=max(0,25-adx)/25*30+min(atr_pct/1.0,1)*25+max(0,(ret_30d+15)/30)*10
            results.append({'symbol':sym,'price':round(price,2),'score':round(score,1)})
        except:continue
    results.sort(key=lambda x:x['score'],reverse=True)
    return results[:CONFIG['max_stocks']]

def get_current_price(symbol):
    df=yf.download(symbol,period='1d',interval='1h',progress=False,auto_adjust=True)
    if len(df)>0:
        if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0].lower() for c in df.columns]
        else:df.columns=[c.lower() for c in df.columns]
        for c in df.columns:
            if 'close' in str(c).lower():return float(df[c].iloc[-1])
    return 0.0

def process(st):
    sym=st['symbol'];end=datetime.now();start=end-timedelta(days=14)
    try:
        df_raw=yf.download(sym,start=start.strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'),interval='1h',progress=False,auto_adjust=True)
        if len(df_raw)<20:return st
        if isinstance(df_raw.columns,pd.MultiIndex):df_raw.columns=[c[0].lower() for c in df_raw.columns]
        else:df_raw.columns=[c.lower() for c in df_raw.columns]
        cn=list(df_raw.columns)
        oc=next((c for c in cn if 'open' in str(c).lower()),cn[0]);hc_=next((c for c in cn if 'high' in str(c).lower()),cn[1]);lc_=next((c for c in cn if 'low' in str(c).lower()),cn[2]);cc_=next((c for c in cn if 'close' in str(c).lower()),cn[3])
        df=pd.DataFrame({'open':pd.to_numeric(df_raw[oc],errors='coerce'),'high':pd.to_numeric(df_raw[hc_],errors='coerce'),'low':pd.to_numeric(df_raw[lc_],errors='coerce'),'close':pd.to_numeric(df_raw[cc_],errors='coerce')}).dropna()
    except:return st
    ca=df['close'].values;ha=df['high'].values;la=df['low'].values;oa=df['open'].values;n=len(ca)
    slp=CONFIG['slippage'];fee=CONFIG['fee_per_side']/2
    rsi=compute_rsi(ca);regime=compute_regime(df)
    tr=np.maximum(ha-la,np.maximum(np.abs(ha-np.roll(ca,1)),np.abs(la-np.roll(ca,1))));tr[0]=ha[0]-la[0]
    atr=pd.Series(tr).ewm(span=14,adjust=False).mean().values;atr_pct=atr/np.where(ca>0,ca,1)
    i=n-1;cr=regime[i] if i<len(regime) else 0
    rsi_now=rsi[-1] if len(rsi)>0 else 50
    mode_before='BH' if st['in_bh'] else ('SCALP' if st['in_scalp'] else 'CASH')

    def log_detail(action, reason, trade_px=0, trade_sh=0, pnl_val=0):
        tv=st['cash']
        if st['in_bh']:tv+=st['bh_shares']*ca[i]
        if st['in_scalp']:tv+=st['scalp_shares']*ca[i]
        # save to us_trade_log.json
        log_file='us_trade_log.json';existing=[]
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:existing=json.load(f)
            except:pass
        existing.append({'time':str(df.index[i])[:16],'symbol':sym,'price':round(ca[i],2),'rsi':round(rsi_now,1),'regime':int(cr),'mode_before':mode_before,'action':action,'reason':reason,'trade_price':round(trade_px,2) if trade_px else '','trade_shares':round(trade_sh,4) if trade_sh else '','pnl':round(pnl_val,2) if pnl_val else '','cash':round(st['cash'],2),'equity':round(tv,2),'return_pct':round((tv/CONFIG['init_cap']-1)*100,2)})
        with open(log_file,'w') as f:json.dump(existing,f,indent=2,default=str)

    # BH ENTER
    if cr==1 and not st['in_bh']:
        if st['in_scalp']:
            s=st['scalp_shares'];xp=ca[i-1];pnl=s*(xp-st['scalp_entry'])-CONFIG['fee_per_side']
            st['cash']+=s*xp-fee;st['total_pnl']+=pnl;st['in_scalp']=False;st['scalp_shares']=0
            st['trades'].append({'type':'SCALP','entry':round(st['scalp_entry'],2),'exit':round(xp,2),'shares':round(s,4),'pnl':round(pnl,2)})
            log_detail('SCALP_EXIT','BH force close',round(xp,2),round(s,4),round(pnl,2))
        st['bh_shares']=st['cash']*0.90/oa[i];st['cash']-=st['bh_shares']*oa[i]+fee
        st['bh_entry']=oa[i];st['in_bh']=True
        st['trades'].append({'type':'BH_ENTER','price':round(oa[i],2),'shares':round(st['bh_shares'],4)})
        log_detail('BH_ENTER','5D Ret>0',round(oa[i],2),round(st['bh_shares'],4))
    # BH EXIT
    elif cr==0 and st['in_bh']:
        s=st['bh_shares'];pnl=s*(oa[i]-st['bh_entry'])-CONFIG['fee_per_side']
        st['cash']+=s*oa[i]-fee;st['total_pnl']+=pnl;st['in_bh']=False;st['bh_shares']=0
        st['trades'].append({'type':'BH_EXIT','entry':round(st['bh_entry'],2),'exit':round(oa[i],2),'shares':round(s,4),'pnl':round(pnl,2)})
        log_detail('BH_EXIT','5D Ret<=0',round(oa[i],2),round(s,4),round(pnl,2))
    if st['in_bh']:st['last_trade_time']=str(df.index[i]);return st

    # SCALP ENTER
    if not st['in_scalp'] and st['cash']>10 and len(rsi)>10 and cr==0 and rsi[-1]<25:
        dsl=np.clip(atr_pct[i]*0.5,0.001,0.008) if i>0 else 0.002;dtp=dsl*10;ee=oa[i]*(1+slp)
        pz=st['cash']*0.20;s=pz/ee;st['cash']-=pz+fee;st['scalp_shares']=s;st['scalp_entry']=ee
        st['scalp_sl']=ee*(1-dsl);st['scalp_tp']=ee*(1+dtp);st['in_scalp']=True
        st['trades'].append({'type':'SCALP_ENTER','price':round(ee,2),'shares':round(s,4)})
        log_detail('SCALP_ENTER',f'RSI={rsi_now:.1f}<25 SL={ee*(1-dsl):.2f} TP={ee*(1+dtp):.2f}',round(ee,2),round(s,4))
    # SCALP EXIT
    elif st['in_scalp']:
        sl=st['scalp_sl'];tp=st['scalp_tp'];entry=st['scalp_entry'];xp=None
        if la[i]<=sl:xp=sl
        elif ha[i]>=tp:xp=tp
        if xp:
            s=st['scalp_shares'];pnl=s*(xp-entry)-CONFIG['fee_per_side']
            st['cash']+=s*xp-fee;st['total_pnl']+=pnl;st['in_scalp']=False;st['scalp_shares']=0
            st['trades'].append({'type':'SCALP_EXIT','entry':round(entry,2),'exit':round(xp,2),'shares':round(s,4),'pnl':round(pnl,2)})
            log_detail('SCALP_EXIT',f'{"SL" if la[i]<=sl else "TP"} hit',round(xp,2),round(s,4),round(pnl,2))
    st['last_trade_time']=str(df.index[i])
    return st

def init_stock(symbol):
    return {'symbol':symbol,'cash':CONFIG['init_cap'],'shares':0,'in_bh':False,'bh_entry':None,'bh_shares':0,'in_scalp':False,'scalp_entry':None,'scalp_shares':0,'scalp_sl':None,'scalp_tp':None,'trades':[],'total_pnl':0,'last_trade_time':None}

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] US Engine")
    state=load_state();now=datetime.now()
    if not state['stocks'] or state.get('last_rebalance_month')!=now.month:
        picks=screen_market(CANDIDATES)
        old_val=0
        for s in state['stocks']:
            tv=s['cash']
            try:
                p=get_current_price(s['symbol'])
                if s['in_bh']:tv+=s['bh_shares']*p
                if s['in_scalp']:tv+=s['scalp_shares']*p
            except:pass
            old_val+=tv
        per_stock=old_val/len(picks) if picks and old_val>0 else CONFIG['init_cap']
        state['stocks']=[]
        for pick in picks:
            st=init_stock(pick['symbol']);st['cash']=per_stock;state['stocks'].append(st)
        state['last_screen']=now.strftime('%Y-%m-%d');state['last_rebalance_month']=now.month
        print(f"  Rebalanced: {[p['symbol'] for p in picks]}")
    else:print(f"  Processing {len(state['stocks'])} stocks")

    totals=[]
    log_entries=[]
    now_ts=datetime.now().strftime('%Y-%m-%d %H:%M')
    for st in state['stocks']:
        st=process(st)
        tv=st['cash'];cur_p=0
        try:
            cur_p=get_current_price(st['symbol'])
            if st['in_bh']:tv+=st['bh_shares']*cur_p
            if st['in_scalp']:tv+=st['scalp_shares']*cur_p
        except:pass
        ret=(tv/CONFIG['init_cap']-1)*100
        mode='BH' if st['in_bh'] else ('SCALP' if st['in_scalp'] else 'CASH')
        price_str=f'${cur_p:,.2f}' if cur_p else 'N/A'
        print(f"  {st['symbol']:<8} {price_str:<12} {mode:<6} ${tv:,.0f} ({ret:+.1f}%) | {len(st['trades'])} trades")
        totals.append(tv)
        # Build log entry
        log_entries.append({
            'time': now_ts,
            'symbol': st['symbol'],
            'price': cur_p,
            'mode': mode,
            'value': tv,
            'return_pct': round(ret,2),
            'cash': round(st['cash'],2),
            'bh_shares': round(st.get('bh_shares',0),4),
            'scalp_shares': round(st.get('scalp_shares',0),4),
            'total_trades': len(st['trades']),
        })
    total_port = sum(totals)
    print(f"  Total: ${total_port:,.0f} ({(total_port/(len(state['stocks'])*CONFIG['init_cap'])-1)*100:+.1f}%)")
    save_state(state)
    # Append to log file
    log_file = CONFIG['state_file'].replace('state.json','log.json')
    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f: existing = json.load(f)
        except: pass
    existing.extend(log_entries)
    with open(log_file, 'w') as f: json.dump(existing, f, indent=2, default=str)
    print(f"  State saved | Log: {log_file}")

if __name__=='__main__':
    print("US Market Engine | 5D Hybrid | $1,000/stock | Ctrl+C to stop")
    while True:
        try:main();print(f"  [Next: 5min]\n");time.sleep(300)
        except KeyboardInterrupt:print("\n  Stopped");break
        except Exception as e:print(f"  Error: {e}");time.sleep(60)
