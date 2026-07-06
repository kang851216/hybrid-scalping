"""$1,000 Validation — All 23 tests"""
import yfinance as yf, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
FIXED_FEE=2.00;SLIPPAGE=0.00005;INIT_CAP=1000.0
CANDIDATES=['NVDA','AMD','AVGO','INTC','MU','TXN','QCOM','AMAT','META','GOOGL','NFLX','AMZN','CRM','ADBE','AAPL','MSFT','TSLA','DELL','JPM','GS','MS','BAC','C','BLK','SCHW','V','MA','PFE','JNJ','MRK','ABBV','LLY','UNH','CVS','KO','PEP','PG','WMT','COST','HD','MCD','SBUX','NKE','XOM','CVX','COP','CAT','DE','GE','BA','VZ','T','TMUS','SO','DUK','NEE']

# All dates from both validations
ALL_DATES={
    '2025_01':'2025-01-15','2025_02':'2025-03-25','2025_03':'2025-04-05','2025_04':'2025-06-05','2025_05':'2025-07-05',
    '2025_06':'2025-07-25','2025_07':'2025-08-05','2025_08':'2025-09-05','2025_09':'2025-09-25','2025_10':'2025-10-05',
    '2025_11':'2025-10-15','2025_12':'2025-11-05','2025_13':'2025-11-25','2025_14':'2025-12-05','2025_15':'2025-12-05',
    'COVID':'2020-05-15','Bear':'2022-01-15','BearR':'2022-06-15','Side1':'2023-03-15','Side2':'2023-08-15',
    'AI1':'2024-03-15','AI2':'2024-06-15',
}

def screen_top(date_str):
    end=pd.Timestamp(date_str);start=end-pd.Timedelta(days=90)
    results=[]
    for sym in CANDIDATES[:25]:
        try:
            df=yf.download(sym,start=start.strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'),interval='1d',progress=False,auto_adjust=True)
            if len(df)<50:continue
            if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0].lower() for c in df.columns]
            else:df.columns=[c.lower() for c in df.columns]
            cn=list(df.columns)
            cc=next((c for c in cn if 'close' in str(c).lower()),cn[3])
            c_arr=pd.to_numeric(df[cc],errors='coerce')
            price=float(c_arr.iloc[-1])
            if price<5 or price>600:continue
            hc=next((c for c in cn if 'high' in str(c).lower()),cn[1])
            lc=next((c for c in cn if 'low' in str(c).lower()),cn[2])
            h_arr=pd.to_numeric(df[hc],errors='coerce');l_arr=pd.to_numeric(df[lc],errors='coerce')
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
            total_score=max(0,25-adx)/25*30+min(atr_pct/2.0,1)*25+(1-abs(price-200)/400)*15
            results.append({'symbol':sym,'price':round(price,2),'score':round(total_score,1)})
        except:continue
    results.sort(key=lambda x:x['score'],reverse=True)
    return results[0] if results else None

def hybrid_5d(sym,start_str,end_str):
    df=yf.download(sym,start=start_str,end=end_str,interval='1d',progress=False,auto_adjust=True)
    if len(df)<20:return None,None
    if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0].lower() for c in df.columns]
    else:df.columns=[c.lower() for c in df.columns]
    cn=list(df.columns)
    oc=next((c for c in cn if 'open' in str(c).lower()),cn[0])
    cc_=next((c for c in cn if 'close' in str(c).lower()),cn[3])
    close=pd.to_numeric(df[cc_],errors='coerce').values
    open_=pd.to_numeric(df[oc],errors='coerce').values;n=len(close)
    bh_ret=(close[-1]/open_[0]-1)*100
    d=np.diff(close,prepend=close[0]);g=np.where(d>0,d,0.);lo=np.where(d<0,-d,0.)
    ag=np.zeros(n);al=np.zeros(n)
    if n>2:ag[2]=np.mean(g[1:3]);al[2]=np.mean(lo[1:3])
    for i in range(3,n):ag[i]=(ag[i-1]+g[i])/2;al[i]=(al[i-1]+lo[i])/2
    with np.errstate(divide='ignore',invalid='ignore'):rs=np.where(al!=0,ag/al,100.);rsi=np.where(al!=0,100.-100./(1.+rs),100.)
    rsi=np.nan_to_num(rsi,nan=50.)
    regime=np.zeros(n,dtype=int)
    for i in range(5,n):
        if close[i]>close[i-5]:regime[i]=1
    sig=np.zeros(n,dtype=int)
    for i in range(5,n):
        if regime[i]==0:
            if rsi[i]<25:sig[i]=1
            elif rsi[i]>75:sig[i]=-1
    cap=float(INIT_CAP);in_bh=False;bh_ep=None;bh_sh=None;ip=False;posd=None
    for i in range(1,n):
        if regime[i]==1 and not in_bh:
            if ip:xp=open_[i];gp=posd['sh']*(xp-posd['ep']) if posd['d']=='LONG' else posd['sh']*(posd['ep']-xp);cap+=gp-FIXED_FEE;ip=False
            bh_ep=open_[i];bh_sh=cap*0.90/bh_ep;cap-=FIXED_FEE/2;in_bh=True
        elif regime[i]==0 and in_bh:
            xp=open_[i];cap+=bh_sh*(xp-bh_ep)-FIXED_FEE/2;in_bh=False
        if in_bh:continue
        if not ip:
            s=sig[i-1]
            if s==0:continue
            ep=open_[i]
            if s==1:ee=ep*(1+SLIPPAGE);slp_=ee*0.98;tpp=ee*1.20;dd='LONG'
            else:ee=ep*(1-SLIPPAGE);slp_=ee*1.02;tpp=ee*0.80;dd='SHORT'
            pz=cap*0.20;sh=pz/ee;cap-=FIXED_FEE/2;posd={'d':dd,'ep':ee,'sl':slp_,'tp':tpp,'sh':sh};ip=True
        else:
            if posd['d']=='LONG':
                if close[i]<=posd['sl']:xp=posd['sl']
                elif close[i]>=posd['tp']:xp=posd['tp']
                else:continue
            else:
                if close[i]>=posd['sl']:xp=posd['sl']
                elif close[i]<=posd['tp']:xp=posd['tp']
                else:continue
            gp=posd['sh']*(xp-posd['ep']) if posd['d']=='LONG' else posd['sh']*(posd['ep']-xp);cap+=gp-FIXED_FEE/2;ip=False
    if in_bh and bh_sh is not None:cap+=bh_sh*(close[-1]-bh_ep)-FIXED_FEE/2
    return (cap/INIT_CAP-1)*100,bh_ret

# Filter duplicates
seen=set();unique_dates=[]
for k,v in ALL_DATES.items():
    if v not in seen:unique_dates.append((k,v));seen.add(v)

wins=0;total=0;tot_h=0;tot_b=0
for label,sd in unique_dates:
    ed=(pd.Timestamp(sd)+pd.Timedelta(days=183)).strftime('%Y-%m-%d')
    pick=screen_top(sd)
    if pick is None:continue
    nr,bh=hybrid_5d(pick['symbol'],sd,ed)
    if nr is None:continue
    w='HYBRID' if nr>bh else 'BH'
    if nr>bh:wins+=1
    total+=1;tot_h+=nr;tot_b+=bh
    m='*' if nr>bh else ' '
    print(f'{m}{label:<20} {sd} {pick["symbol"]:<6} H={nr:>+7.1f}% BH={bh:>+7.1f}%')

print(f'---')
print(f'$1,000/stock: {wins}/{total} wins | Avg H={tot_h/total:+.1f}% | Avg BH={tot_b/total:+.1f}%' if total else 'No data')
