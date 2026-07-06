"""10 Random Dates Validation — Hybrid Strategy vs Buy & Hold"""
import yfinance as yf, pandas as pd, numpy as np, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

FIXED_FEE=2.00;SLIPPAGE=0.00005;INIT_CAP=5000.0
CANDIDATES=['NVDA','AMD','AVGO','INTC','MU','TXN','QCOM','AMAT','META','GOOGL','NFLX','AMZN','CRM','ADBE','AAPL','MSFT','TSLA','DELL','JPM','GS','MS','BAC','C','BLK','SCHW','V','MA','PFE','JNJ','MRK','ABBV','LLY','UNH','CVS','KO','PEP','PG','WMT','COST','HD','MCD','SBUX','NKE','XOM','CVX','COP','CAT','DE','GE','BA','VZ','T','TMUS','SO','DUK','NEE']

DATES = [
    ('2025-01-15','2025-07-15'),('2025-02-20','2025-08-20'),('2025-03-10','2025-09-10'),
    ('2025-04-25','2025-10-25'),('2025-05-15','2025-11-15'),('2025-06-20','2025-12-20'),
    ('2025-07-10','2026-01-10'),('2025-08-15','2026-02-15'),('2025-09-20','2026-03-20'),
    ('2025-10-10','2026-04-10'),
]

def screen_on_date(date_str):
    end=pd.Timestamp(date_str);start=end-pd.Timedelta(days=60)
    results=[]
    for sym in CANDIDATES:
        try:
            df=yf.download(sym,start=start.strftime('%Y-%m-%d'),end=end.strftime('%Y-%m-%d'),interval='1h',progress=False,auto_adjust=True)
            if len(df)<100:continue
            if isinstance(df.columns,pd.MultiIndex):df.columns=[c[0].lower() for c in df.columns]
            else:df.columns=[c.lower() for c in df.columns]
            cn=list(df.columns)
            cc=next((c for c in cn if 'close' in str(c).lower()),cn[3])
            hc=next((c for c in cn if 'high' in str(c).lower()),cn[1])
            lc=next((c for c in cn if 'low' in str(c).lower()),cn[2])
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
            gap=abs(c_arr-c_arr.shift(1))/c_arr.shift(1)*100;max_gap=float(gap.max())
            if max_gap>12:continue
            ret_30d=(c_arr.iloc[-1]/c_arr.iloc[-min(len(c_arr),130)]-1)*100
            adx_score=max(0,25-adx)/25*30;vol_score=min(atr_pct/1.0,1)*25
            gap_score=max(0,12-max_gap)/12*15;price_score=(1-abs(price-200)/400)*10
            ret_score=max(0,(ret_30d+15)/30)*10
            total_score=adx_score+vol_score+gap_score+price_score+ret_score
            results.append({'symbol':sym,'price':round(price,2),'adx':round(adx,1),'score':round(total_score,1)})
        except:continue
    results.sort(key=lambda x:x['score'],reverse=True)
    return results[0] if results else None

def hybrid_backtest(sym,start_str,end_str):
    df_raw=yf.download(sym,start=start_str,end=end_str,interval='1h',progress=False,auto_adjust=True)
    if len(df_raw)<50:return None
    if isinstance(df_raw.columns,pd.MultiIndex):df_raw.columns=[c[0].lower() for c in df_raw.columns]
    else:df_raw.columns=[c.lower() for c in df_raw.columns]
    cn=list(df_raw.columns)
    oc=next((c for c in cn if 'open' in str(c).lower()),cn[0])
    hc=next((c for c in cn if 'high' in str(c).lower()),cn[1])
    lc=next((c for c in cn if 'low' in str(c).lower()),cn[2])
    cc_=next((c for c in cn if 'close' in str(c).lower()),cn[3])
    df=pd.DataFrame({'open':pd.to_numeric(df_raw[oc],errors='coerce'),'high':pd.to_numeric(df_raw[hc],errors='coerce'),'low':pd.to_numeric(df_raw[lc],errors='coerce'),'close':pd.to_numeric(df_raw[cc_],errors='coerce')})
    df=df.dropna()
    ca=df['close'].values.ravel();ha=df['high'].values.ravel();la=df['low'].values.ravel();oa=df['open'].values.ravel();n=len(ca)
    if n<20:return None
    
    bh_entry=float(df['open'].iloc[0]);bh_exit=float(df['close'].iloc[-1]);bh_ret=(bh_exit/bh_entry-1)*100
    
    # RSI
    d=np.diff(ca,prepend=ca[0]);g=np.where(d>0,d,0.);lo=np.where(d<0,-d,0.)
    ag=np.zeros(n);al=np.zeros(n)
    if n>2:ag[2]=np.mean(g[1:3]);al[2]=np.mean(lo[1:3])
    for i in range(3,n):ag[i]=(ag[i-1]+g[i])/2;al[i]=(al[i-1]+lo[i])/2
    with np.errstate(divide='ignore',invalid='ignore'):rs=np.where(al!=0,ag/al,100.);rsi=np.where(al!=0,100.-100./(1.+rs),100.)
    rsi=np.nan_to_num(rsi,nan=50.)
    
    # Daily regime detection
    dfd=df['close'].resample('1D').last().dropna()
    regime_arr=np.zeros(n,dtype=int)
    if len(dfd)>=50:
        cd_d=dfd.values
        e20=pd.Series(cd_d).ewm(span=20,adjust=False).mean().values
        e50=pd.Series(cd_d).ewm(span=50,adjust=False).mean().values
        hd=df['high'].resample('1D').max().dropna().values;ld=df['low'].resample('1D').min().dropna().values
        cdd=df['close'].resample('1D').last().dropna().values
        trd=np.maximum(hd-ld,np.maximum(np.abs(hd-np.roll(cdd,1)),np.abs(ld-np.roll(cdd,1))));trd[0]=hd[0]-ld[0]
        atrd=pd.Series(trd).ewm(span=14,adjust=False).mean().values
        upd=(hd-np.roll(hd,1)).clip(0);dnd=(np.roll(ld,1)-ld).clip(0);upd[0]=dnd[0]=0
        pDMd=np.where((upd>dnd)&(upd>0),upd,0);mDMd=np.where((dnd>upd)&(dnd>0),dnd,0)
        pDId=(pd.Series(pDMd).ewm(span=14,adjust=False).mean()/np.where(atrd>0,atrd,1e-10))*100
        mDId=(pd.Series(mDMd).ewm(span=14,adjust=False).mean()/np.where(atrd>0,atrd,1e-10))*100
        dxd=np.abs(pDId-mDId)/np.where(pDId+mDId>0,pDId+mDId,1)*100
        adxd=pd.Series(dxd).ewm(span=14,adjust=False).mean().values
        for i in range(n):
            bar_day=df.index[i].date()
            day_idx=None
            for j in range(len(dfd.index)):
                if dfd.index[j].date()==bar_day:day_idx=j;break
            if day_idx is not None and day_idx>=50:
                if e20[day_idx]>e50[day_idx] and adxd[day_idx]>18:regime_arr[i]=1
    
    sig=np.zeros(n,dtype=int)
    for i in range(10,n):
        if regime_arr[i]==0:
            if rsi[i]<25:sig[i]=1
            elif rsi[i]>75:sig[i]=-1
    
    tr=np.maximum(ha-la,np.maximum(np.abs(ha-np.roll(ca,1)),np.abs(la-np.roll(ca,1))));tr[0]=ha[0]-la[0]
    atr=pd.Series(tr).ewm(span=14,adjust=False).mean().values;atr_pct=atr/np.where(ca>0,ca,1)
    
    fee=FIXED_FEE/2;slp=SLIPPAGE;cap=float(INIT_CAP);eq=[cap];ip,posd=False,None
    in_bh=False;bh_ep=None;bh_sh=None

    for i in range(1,n):
        cr=regime_arr[i]
        if cr==1 and not in_bh:
            if ip:
                p=posd;xp=ca[i-1]
                gp=p['sh']*(xp-p['ep']) if p['d']=='LONG' else p['sh']*(p['ep']-xp)
                gp-=p['sh']*p['ep']*slp;cap+=gp-fee;ip,posd=False,None
            bh_ep=oa[i];bh_sh=cap*0.90/bh_ep;cap-=fee;in_bh=True
        elif cr==0 and in_bh:
            xp=oa[i];cap+=bh_sh*(xp-bh_ep)-fee;in_bh=False;eq.append(cap)
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
            gp=p['sh']*(xp-p['ep']) if p['d']=='LONG' else p['sh']*(p['ep']-xp)
            gp-=p['sh']*p['ep']*slp;cap+=gp-fee;eq.append(cap);ip,posd=False,None
    
    if in_bh and bh_sh is not None:cap+=bh_sh*(ca[-1]-bh_ep)-fee
    nr=(cap/INIT_CAP-1)*100
    eqa=np.array(eq);pk=np.maximum.accumulate(eqa)
    with np.errstate(divide='ignore',invalid='ignore'):dd=(eqa-pk)/pk*100
    dd=np.where(np.isfinite(dd),dd,0);mdd=abs(np.min(dd))
    return nr,mdd,bh_ret

print('10 Random Dates — 6-Month Hybrid Strategy Validation')
print(f'{"#":<4} {"Start":<12} {"Stock":<8} {"Price":>8} {"Score":>6} {"Hybrid":>9} {"MDD":>7} {"BH":>9} {"Winner":<10}')
print('-'*80)
results=[]
for idx,(start,end) in enumerate(DATES):
    pick=screen_on_date(start)
    if pick is None:
        print(f'{idx+1:<4} {start:<12} NO PICK')
        continue
    sym=pick['symbol'];price=pick['price'];score=pick['score']
    r=hybrid_backtest(sym,start,end)
    if r is None:
        print(f'{idx+1:<4} {start:<12} {sym:<8} ${price:>7,.0f} {score:>5.1f} NO DATA')
        continue
    nr,mdd,bh=r
    win='HYBRID' if nr>bh else 'BH'
    mark='*' if nr>bh else ' '
    print(f'{mark}{idx+1:<3} {start:<12} {sym:<8} ${price:>7,.0f} {score:>5.1f} {nr:>+8.2f}% {mdd:>6.2f}% {bh:>+8.2f}% {win:<10}')
    results.append((nr,mdd,bh,nr>bh))

if results:
    wins=sum(1 for r in results if r[3])
    avg_hyb=np.mean([r[0] for r in results])
    avg_mdd=np.mean([r[1] for r in results])
    avg_bh=np.mean([r[2] for r in results])
    print('-'*80)
    print(f'RESULTS: {wins}/{len(results)} wins for Hybrid')
    print(f'Avg Hybrid: {avg_hyb:+.1f}% | Avg MDD: {avg_mdd:.1f}% | Avg BH: {avg_bh:+.1f}%')
