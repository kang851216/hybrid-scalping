"""2026 Top 2 x 3mo x $1K"""
import yfinance as yf, pandas as pd, numpy as np, warnings, random
warnings.filterwarnings('ignore')
random.seed(42)
FIXED_FEE=2.00;SLIPPAGE=0.00005;INIT_CAP=1000.0
CANDIDATES=['NVDA','AMD','AVGO','INTC','MU','TXN','QCOM','AMAT','META','GOOGL','NFLX','AMZN','CRM','ADBE','AAPL','MSFT','TSLA','DELL','JPM','GS','MS','BAC','C','BLK','SCHW','V','MA','PFE','JNJ','MRK','ABBV','LLY','UNH','CVS','KO','PEP','PG','WMT','COST','HD','MCD','SBUX','NKE','XOM','CVX','COP','CAT','DE','GE','BA','VZ','T','TMUS','SO','DUK','NEE']

dates_pool=[]
for m in range(1,4):
    for d in range(2,29,2):
        try:pd.Timestamp(f'2026-{m:02d}-{d:02d}');dates_pool.append(f'2026-{m:02d}-{d:02d}')
        except:pass
random.shuffle(dates_pool)
TEST_DATES=sorted(dates_pool[:20])

def screen_top2(date_str):
    end=pd.Timestamp(date_str);start=end-pd.Timedelta(days=60)
    results=[]
    for sym in CANDIDATES[:25]:
        try:
            df=yf.download(sym,start=start.strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'),interval='1h',progress=False,auto_adjust=True)
            if len(df)<100:continue
            if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0].lower() for c in df.columns]
            else:df.columns=[c.lower() for c in df.columns]
            cn=list(df.columns)
            cc=next((c for c in cn if 'close' in str(c).lower()),cn[3]);hc=next((c for c in cn if 'high' in str(c).lower()),cn[1]);lc=next((c for c in cn if 'low' in str(c).lower()),cn[2])
            c_arr=pd.to_numeric(df[cc],errors='coerce');h_arr=pd.to_numeric(df[hc],errors='coerce');l_arr=pd.to_numeric(df[lc],errors='coerce')
            price=float(c_arr.iloc[-1])
            if price<20 or price>600:continue
            tr=pd.concat([h_arr-l_arr,(h_arr-c_arr.shift(1)).abs(),(l_arr-c_arr.shift(1)).abs()],axis=1).max(axis=1)
            atr=tr.ewm(span=14,adjust=False).mean()
            up=(h_arr-h_arr.shift(1)).clip(0);dn=(l_arr.shift(1)-l_arr).clip(0)
            pDM=up.where((up>dn)&(up>0),0);mDM=dn.where((dn>up)&(dn>0),0)
            pDI=(pDM.ewm(span=14,adjust=False).mean()/atr.replace(0,1e-10))*100
            mDI=(mDM.ewm(span=14,adjust=False).mean()/atr.replace(0,1e-10))*100
            dx=(abs(pDI-mDI)/(pDI+mDI).replace(0,1))*100
            adx=float(dx.ewm(span=14,adjust=False).mean().iloc[-1])
            if adx>25:continue
            atr_pct=float(atr.iloc[-1]/price*100)
            if atr_pct<0.40:continue
            ret_30d=(c_arr.iloc[-1]/c_arr.iloc[-min(len(c_arr),130)]-1)*100
            adx_score=max(0,25-adx)/25*30;vol_score=min(atr_pct/1.0,1)*25
            gap_score=10;price_score=(1-abs(price-200)/400)*10;ret_score=max(0,(ret_30d+15)/30)*10
            total_score=adx_score+vol_score+gap_score+price_score+ret_score
            results.append({'symbol':sym,'price':round(price,2),'score':round(total_score,1)})
        except:continue
    results.sort(key=lambda x:x['score'],reverse=True)
    return results[:2]

def hybrid_5d(sym,start_str,end_str):
    df_raw=yf.download(sym,start=start_str,end=end_str,interval='1h',progress=False,auto_adjust=True)
    if len(df_raw)<50:return None,None
    if isinstance(df_raw.columns,pd.MultiIndex):df_raw.columns=[c[0].lower() for c in df_raw.columns]
    else:df_raw.columns=[c.lower() for c in df_raw.columns]
    cn=list(df_raw.columns)
    oc=next((c for c in cn if 'open' in str(c).lower()),cn[0]);hc_=next((c for c in cn if 'high' in str(c).lower()),cn[1]);lc_=next((c for c in cn if 'low' in str(c).lower()),cn[2]);cc_=next((c for c in cn if 'close' in str(c).lower()),cn[3])
    df=pd.DataFrame({'open':pd.to_numeric(df_raw[oc],errors='coerce'),'high':pd.to_numeric(df_raw[hc_],errors='coerce'),'low':pd.to_numeric(df_raw[lc_],errors='coerce'),'close':pd.to_numeric(df_raw[cc_],errors='coerce')}).dropna()
    ca=df['close'].values.ravel();ha=df['high'].values.ravel();la=df['low'].values.ravel();oa=df['open'].values.ravel();n=len(ca)
    if n<20:return None,None
    bh_ret=(ca[-1]/oa[0]-1)*100
    d=np.diff(ca,prepend=ca[0]);g=np.where(d>0,d,0.);lo=np.where(d<0,-d,0.)
    ag=np.zeros(n);al=np.zeros(n)
    if n>2:ag[2]=np.mean(g[1:3]);al[2]=np.mean(lo[1:3])
    for i in range(3,n):ag[i]=(ag[i-1]+g[i])/2;al[i]=(al[i-1]+lo[i])/2
    with np.errstate(divide='ignore',invalid='ignore'):rs=np.where(al!=0,ag/al,100.);rsi=np.where(al!=0,100.-100./(1.+rs),100.)
    rsi=np.nan_to_num(rsi,nan=50.)
    dfd=df['close'].resample('1D').last().dropna();cd=dfd.values
    regime=np.zeros(n,dtype=int)
    for i in range(n):
        bar_day=df.index[i].date();day_idx=None
        for j in range(len(dfd.index)):
            if dfd.index[j].date()==bar_day:day_idx=j;break
        if day_idx is not None and day_idx>=5:
            if cd[day_idx]>cd[max(0,day_idx-5)]:regime[i]=1
    sig=np.zeros(n,dtype=int)
    for i in range(10,n):
        if regime[i]==0:
            if rsi[i]<25:sig[i]=1
            elif rsi[i]>75:sig[i]=-1
    tr=np.maximum(ha-la,np.maximum(np.abs(ha-np.roll(ca,1)),np.abs(la-np.roll(ca,1))));tr[0]=ha[0]-la[0]
    atr=pd.Series(tr).ewm(span=14,adjust=False).mean().values;atr_pct=atr/np.where(ca>0,ca,1)
    fee=FIXED_FEE/2;slp=SLIPPAGE;cap=float(INIT_CAP);ip=False;posd=None;in_bh=False;bh_ep=None;bh_sh=None
    for i in range(1,n):
        cr=regime[i]
        if cr==1 and not in_bh:
            if ip:p=posd;xp=ca[i-1];gp=p['sh']*(xp-p['ep']) if p['d']=='LONG' else p['sh']*(p['ep']-xp);gp-=p['sh']*p['ep']*slp;cap+=gp-fee;ip=False
            bh_ep=oa[i];bh_sh=cap*0.90/bh_ep;cap-=fee;in_bh=True
        elif cr==0 and in_bh:xp=oa[i];cap+=bh_sh*(xp-bh_ep)-fee;in_bh=False
        if in_bh:continue
        if not ip:
            s=sig[i-1]
            if s==0:continue
            ep=oa[i];dsl=np.clip(atr_pct[i-1]*0.5,0.001,0.008) if i>0 else 0.002;dtp=dsl*10
            if s==1:ee=ep*(1+slp);slp_=ee*(1-dsl);tpp=ee*(1+dtp);dd='LONG'
            else:ee=ep*(1-slp);slp_=ee*(1+dsl);tpp=ee*(1-dtp);dd='SHORT'
            pz=cap*0.20;sh=pz/ee;cap-=fee;posd={'d':dd,'ep':ee,'sl':slp_,'tp':tpp,'sh':sh};ip=True
        else:
            p=posd
            if p['d']=='LONG':
                if la[i]<=p['sl']:xp=p['sl']
                elif ha[i]>=p['tp']:xp=p['tp']
                else:continue
            else:
                if ha[i]>=p['sl']:xp=p['sl']
                elif la[i]<=p['tp']:xp=p['tp']
                else:continue
            gp=p['sh']*(xp-p['ep']) if p['d']=='LONG' else p['sh']*(p['ep']-xp);gp-=p['sh']*p['ep']*slp;cap+=gp-fee;ip=False
    if in_bh and bh_sh is not None:cap+=bh_sh*(ca[-1]-bh_ep)-fee
    return(cap/INIT_CAP-1)*100,bh_ret

wins=0;total=0;grand_h=[];grand_b=[]
print(f'2026 Top2 x3mo x$1K | {len(TEST_DATES)} dates')
print(f'{"#":<4} {"Date":<12} {"Stocks":<16} {"Hybrid":>9} {"BH":>9} {"Winner":<10}')
print('-'*60)

for sd in TEST_DATES:
    ed=(pd.Timestamp(sd)+pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    picks=screen_top2(sd)
    if len(picks)<2:continue
    port_h=0;port_b=0
    ok=0
    for pick in picks:
        nr,bh=hybrid_5d(pick['symbol'],sd,ed)
        if nr is not None:port_h+=nr;port_b+=bh;ok+=1
    if ok<2:continue
    port_h/=ok;port_b/=ok
    grand_h.append(port_h);grand_b.append(port_b)
    w='HYBRID' if port_h>port_b else 'BH'
    if port_h>port_b:wins+=1
    total+=1
    m='*' if port_h>port_b else ' '
    ss=','.join(p['symbol'] for p in picks)
    print(f'{m}{total:>3} {sd:<12} {ss:<16} {port_h:>+8.2f}% {port_b:>+8.2f}% {w:<10}')

if grand_h:
    print('-'*60)
    print(f'Result: {wins}/{total} wins ({wins/total*100:.0f}%) | Avg H={np.mean(grand_h):+.1f}% | Avg BH={np.mean(grand_b):+.1f}%')
