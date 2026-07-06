#!/usr/bin/env python3
"""
KR Market Engine — 5D Hybrid Scalping
한국 주식시장(KOSPI) 대상 하이브리드 스캘핑 + 중기 추세추종 자동매매 엔진.
- 5분봉 기반으로 5분마다 실행되며, RSI 과매도 시 스캘핑 진입,
- ADX < 25 구간에서 5일 이동평균 돌파 시 중기 보유(BH) 진입.
- 종목당 초기 자본 1,000,000원, 최대 3종목 동시 운용.
"""
import yfinance as yf, pandas as pd, numpy as np, json, os, time, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 설정값 (Configuration)
# ---------------------------------------------------------------------------
CONFIG = {
    "fee_per_side": 2000.0,      # 2,000 KRW 편도 (왕복 4,000 KRW)
    "slippage": 0.00005,        # 슬리피지 (0.005%)
    "init_cap": 1000000.0,       # 1,000,000 KRW
    "max_stocks": 3,            # 최대 동시 보유 종목 수
    "state_file": "kr_state.json",  # 상태 저장 파일
}

# ---------------------------------------------------------------------------
# 스크리닝 후보 종목 리스트 (코스피 우량주 약 100개)
# ---------------------------------------------------------------------------
CANDIDATES = [
    '005930.KS','000660.KS','373220.KS','207940.KS','005380.KS','000270.KS',
    '068270.KS','005490.KS','105560.KS','055550.KS','035420.KS','006400.KS',
    '035720.KS','051910.KS','012330.KS','028260.KS','086790.KS','138040.KS',
    '033780.KS','017670.KS','032830.KS','010130.KS','066570.KS','096770.KS',
    '259960.KS','047050.KS','009540.KS','402340.KS','034020.KS','009150.KS',
    '012450.KS','030200.KS','316140.KS','086280.KS','375500.KS','010140.KS',
    '003550.KS','011200.KS','042660.KS','267260.KS','247540.KS','323410.KS',
    '326030.KS','329180.KS','241560.KS','352820.KS','047810.KS','251270.KS',
    '000720.KS','047040.KS','006360.KS','009830.KS','015760.KS','011790.KS',
    '000880.KS','139480.KS','004170.KS','097950.KS','271560.KS','021240.KS',
    '003490.KS','302440.KS','000100.KS','128940.KS','039130.KS','008770.KS',
    '011170.KS','011780.KS','018260.KS','012510.KS','307950.KS','029780.KS',
    '024110.KS','138930.KS','139130.KS','175330.KS','000810.KS','005830.KS',
    '001450.KS','088350.KS','035250.KS','114090.KS','032350.KS','034230.KS',
    '034220.KS','383220.KS','005070.KS','022100.KS','042700.KS','007660.KS',
    '058470.KS','403870.KS','035900.KS','041510.KS','122870.KS','263750.KS',
    '036570.KS','293490.KS','267250.KS','001120.KS','011210.KS',
]

# ---------------------------------------------------------------------------
# 상태 저장/불러오기 (State Persistence)
# ---------------------------------------------------------------------------
def load_state():
    """파일에서 직전 운용 상태를 불러온다. 없으면 빈 상태 반환."""
    if os.path.exists(CONFIG['state_file']):
        with open(CONFIG['state_file']) as f: return json.load(f)
    return {'stocks': [], 'last_screen': None, 'last_rebalance_month': None}

def save_state(state):
    """현재 운용 상태를 원자적으로 파일에 저장 (.tmp 쓰기 후 rename)."""
    tmp = CONFIG['state_file'] + '.tmp'
    with open(tmp, 'w') as f: json.dump(state, f, indent=2, default=str)
    os.replace(tmp, CONFIG['state_file'])

# ---------------------------------------------------------------------------
# 기술적 지표 계산 함수들
# ---------------------------------------------------------------------------
def compute_rsi(close, period=2):
    """
    Wilder's Smoothing 방식으로 RSI(Relative Strength Index) 계산.
    기본 period=2로 단기 과매도/과매수 포착에 최적화.
    RSI < 25 → 과매도(스캘핑 매수 신호).
    """
    n = len(close); d = np.diff(close, prepend=close[0])
    g = np.where(d > 0, d, 0.); lo = np.where(d < 0, -d, 0.)  # 상승분 / 하락분 분리
    ag = np.zeros(n); al = np.zeros(n)  # 평균 상승 / 평균 하락
    if n > period: ag[period] = np.mean(g[1:period+1]); al[period] = np.mean(lo[1:period+1])
    for i in range(period+1, n):
        ag[i] = (ag[i-1]*(period-1) + g[i]) / period  # Wilder's smoothing
        al[i] = (al[i-1]*(period-1) + lo[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al != 0, ag/al, 100.)  # RS = 평균상승 / 평균하락
        return np.nan_to_num(100. - 100./(1.+np.where(al != 0, rs, 100.)), nan=50.)


def compute_regime(df):
    """
    5일 이동평균 돌파 여부로 시장 국면(regime) 판단.
    - regime=1 (추세): 현재 일봉 종가가 5일 전 종가보다 높음 → 중기 보유(BH) 모드 진입.
    - regime=0 (비추세): 횡보/하락장 → 스캘핑 모드만 활성.
    """
    dfd = df['close'].resample('1D').last().dropna(); cd = dfd.values; n = len(df)
    regime = np.zeros(n, dtype=int)
    for i in range(n):
        bar_day = df.index[i].date(); day_idx = None
        for j in range(len(dfd.index)):
            if dfd.index[j].date() == bar_day: day_idx = j; break
        if day_idx is not None and day_idx >= 5:
            if cd[day_idx] > cd[max(0, day_idx - 5)]: regime[i] = 1  # 5일 전 대비 상승
    return regime


def screen_market(candidates):
    """
    종목 스크리닝: ADX, ATR, 30일 수익률 기반 점수화.
    - ADX < 25 (추세 약함, 횡보장) 선호 → 스캘핑에 유리.
    - ATR% ≥ 0.40% (변동성 충분) → 스캘핑 수익 가능.
    - 점수 = (25-ADX 가중치 30) + (ATR% 가중치 25) + (30일 수익률 가중치 10).
    - 상위 max_stocks(3)개 종목 선정.
    """
    end = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
    start = end - pd.Timedelta(days=30)  # 30일이면 ADX+ATR 충분
    results = []
    for sym in candidates:
        try:
            df = yf.download(sym, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                           interval='1h', progress=False, auto_adjust=True)
            if len(df) < 100: continue  # 데이터 부족 시 제외
            # yfinance 컬럼명 정규화 (MultiIndex 대응)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
            else: df.columns = [c.lower() for c in df.columns]
            cn = list(df.columns)
            cc = next((c for c in cn if 'close' in str(c).lower()), cn[3])
            hc = next((c for c in cn if 'high' in str(c).lower()), cn[1])
            lc = next((c for c in cn if 'low' in str(c).lower()), cn[2])
            c_arr = pd.to_numeric(df[cc], errors='coerce'); h_arr = pd.to_numeric(df[hc], errors='coerce'); l_arr = pd.to_numeric(df[lc], errors='coerce')
            price = float(c_arr.iloc[-1])
            # ATR% 계산 (True Range → 14기간 EMA)
            tr_raw = pd.concat([h_arr-l_arr, (h_arr-c_arr.shift(1)).abs(),
                              (l_arr-c_arr.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr_raw.ewm(span=14, adjust=False).mean()
            atr_pct = float(atr.iloc[-1] / price * 100)
            if atr_pct < 0.40: continue  # 변동성 너무 낮으면 제외
            # ADX 계산 (Directional Movement Index → DX → 14기간 EMA)
            up = (h_arr - h_arr.shift(1)).clip(0); dn = (l_arr.shift(1) - l_arr).clip(0)
            pDM = up.where((up > dn) & (up > 0), 0); mDM = dn.where((dn > up) & (dn > 0), 0)  # +DM / -DM
            pDI = (pDM.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10)) * 100  # +DI
            mDI = (mDM.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10)) * 100  # -DI
            dx = (abs(pDI - mDI) / (pDI + mDI).replace(0, 1)) * 100  # DX
            adx = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])  # ADX
            if adx > 25: continue  # 추세 강하면 스캘핑 부적합 → 제외
            ret_30d = (c_arr.iloc[-1] / c_arr.iloc[-min(len(c_arr), 130)] - 1) * 100  # 30일 수익률
            # 종합 점수: ADX 낮을수록 + ATR% 높을수록 + 30일 수익률 높을수록
            score = max(0, 25 - adx) / 25 * 30 + min(atr_pct / 1.0, 1) * 25 + max(0, (ret_30d + 15) / 30) * 10
            results.append({'symbol': sym, 'price': round(price, 2), 'score': round(score, 1)})
        except: continue
    results.sort(key=lambda x: x['score'], reverse=True)  # 점수 높은 순 정렬
    return results[:CONFIG['max_stocks']]  # 상위 N개 반환


def get_current_price(symbol):
    """현재가 조회: yfinance 1시간봉 최근 종가 반환."""
    df = yf.download(symbol, period='1d', interval='1h', progress=False, auto_adjust=True)
    if len(df) > 0:
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0].lower() for c in df.columns]
        else: df.columns = [c.lower() for c in df.columns]
        for c in df.columns:
            if 'close' in str(c).lower(): return float(df[c].iloc[-1])
    return 0.0

# ---------------------------------------------------------------------------
# 핵심 매매 로직 (process): 단일 종목에 대한 시그널 처리
# ---------------------------------------------------------------------------
def process(st):
    """
    단일 종목 상태(st)를 받아 최신 데이터로 매매 시그널을 평가하고 실행.
    두 가지 전략을 동시 운용:
      1) 중기 보유 (Buy & Hold, BH): regime=1 → 자본의 90%로 진입, regime=0 → 청산.
      2) 단기 스캘핑 (Scalp): regime=0 & RSI<25 → 자본의 20%로 진입,
         ATR 기반 손절/익절로 청산.
    BH 우선: BH 진입 전에 스캘핑 포지션 자동 청산.
    """
    sym = st['symbol']; end = datetime.now(); start = end - timedelta(days=14)  # 최근 14일 데이터
    try:
        df_raw = yf.download(sym, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                           interval='1h', progress=False, auto_adjust=True)
        if len(df_raw) < 20: return st  # 데이터 부족 시 스킵
        # 컬럼명 정규화
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = [c[0].lower() for c in df_raw.columns]
        else: df_raw.columns = [c.lower() for c in df_raw.columns]
        cn = list(df_raw.columns)
        oc = next((c for c in cn if 'open' in str(c).lower()), cn[0])
        hc_ = next((c for c in cn if 'high' in str(c).lower()), cn[1])
        lc_ = next((c for c in cn if 'low' in str(c).lower()), cn[2])
        cc_ = next((c for c in cn if 'close' in str(c).lower()), cn[3])
        df = pd.DataFrame({
            'open': pd.to_numeric(df_raw[oc], errors='coerce'),
            'high': pd.to_numeric(df_raw[hc_], errors='coerce'),
            'low': pd.to_numeric(df_raw[lc_], errors='coerce'),
            'close': pd.to_numeric(df_raw[cc_], errors='coerce'),
        }).dropna()
    except: return st


    # 기술적 지표 계산
    ca = df['close'].values; ha = df['high'].values; la = df['low'].values; oa = df['open'].values; n = len(ca)
    slp = CONFIG['slippage']; fee = CONFIG['fee_per_side'] / 2  # 편도 수수료 (매수/매도 각각 적용)
    rsi = compute_rsi(ca); regime = compute_regime(df)
    tr = np.maximum(ha-la, np.maximum(np.abs(ha-np.roll(ca,1)), np.abs(la-np.roll(ca,1))))  # True Range
    tr[0] = ha[0]-la[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values  # ATR (14기간 EMA)
    atr_pct = atr / np.where(ca > 0, ca, 1)  # ATR% (변동성 비율)
    i = n - 1; cr = regime[i] if i < len(regime) else 0  # 최신 봉의 regime 값
    rsi_now = rsi[-1] if len(rsi) > 0 else 50
    mode_before = 'BH' if st['in_bh'] else ('SCALP' if st['in_scalp'] else 'CASH')

    def log_detail(action, reason, trade_px=0, trade_sh=0, pnl_val=0):
        tv = st['cash']
        if st['in_bh']: tv += st['bh_shares'] * ca[i]
        if st['in_scalp']: tv += st['scalp_shares'] * ca[i]
        append_detail_log(sym, {
            'time': str(df.index[i])[:16],
            'symbol': sym,
            'price': round(ca[i], 0),
            'rsi': round(rsi_now, 1),
            'regime': int(cr),
            'mode_before': mode_before,
            'action': action,
            'reason': reason,
            'trade_price': round(trade_px, 0) if trade_px else '',
            'trade_shares': round(trade_sh, 4) if trade_sh else '',
            'pnl': round(pnl_val, 0) if pnl_val else '',
            'cash': round(st['cash'], 0),
            'equity': round(tv, 0),
            'return_pct': round((tv / CONFIG['init_cap'] - 1) * 100, 2),
        })

    # ---- 중기 보유 (Buy & Hold) 전략 ----
    # regime=1 (추세 상승): BH 진입, regime=0 (비추세): BH 청산
    if cr == 1 and not st['in_bh']:
        # 추세 전환 → BH 진입 전, 먼저 스캘핑 포지션 강제 청산
        if st['in_scalp']:
            s = st['scalp_shares']; xp = ca[i-1]; pnl = s*(xp-st['scalp_entry'])-CONFIG['fee_per_side']
            st['cash'] += s*xp - fee; st['total_pnl'] += pnl; st['in_scalp'] = False; st['scalp_shares'] = 0
            st['trades'].append({'type': 'SCALP', 'entry': round(st['scalp_entry'],2), 'exit': round(xp,2), 'shares': round(s,4), 'pnl': round(pnl, 2)})
            log_detail('SCALP_EXIT', f'BH entry force close', round(xp, 0), round(s, 4), round(pnl, 0))
        # 현금의 90%로 BH 진입
        st['bh_shares'] = st['cash'] * 0.90 / oa[i]
        st['cash'] -= st['bh_shares'] * oa[i] + fee
        st['bh_entry'] = oa[i]; st['in_bh'] = True
        st['trades'].append({'type': 'BH_ENTER', 'price': round(oa[i],2), 'shares': round(st['bh_shares'],4)})
        log_detail('BH_ENTER', '5D Ret>0', round(oa[i], 0), round(st['bh_shares'], 4))
    elif cr == 0 and st['in_bh']:
        # 비추세 전환 → BH 전량 청산
        s = st['bh_shares']; pnl = s*(oa[i]-st['bh_entry'])-CONFIG['fee_per_side']
        st['cash'] += s*oa[i] - fee; st['total_pnl'] += pnl; st['in_bh'] = False; st['bh_shares'] = 0
        st['trades'].append({'type': 'BH_EXIT', 'entry': round(st['bh_entry'],2), 'exit': round(oa[i],2), 'shares': round(s,4), 'pnl': round(pnl, 2)})
        log_detail('BH_EXIT', '5D Ret<=0', round(oa[i], 0), round(s, 4), round(pnl, 0))
    # BH 보유 중이면 스캘핑은 스킵 (전략 충돌 방지)
    if st['in_bh']: st['last_trade_time'] = str(df.index[i]); return st


    # ---- 단기 스캘핑 (Scalping) 전략 ----
    # 조건: BH 미보유, 현금>$10, RSI 데이터 충분, 비추세, RSI<25 (과매도)
    if not st['in_scalp'] and st['cash'] > 10 and len(rsi) > 10 and cr == 0 and rsi[-1] < 25:
        dsl = np.clip(atr_pct[i]*0.5, 0.001, 0.008) if i > 0 else 0.002  # 손절 = ATR%의 50%, 최소 0.1%~최대 0.8%
        dtp = dsl*10  # 익절 = 손절의 10배 (리스크:리워드 1:10)
        ee = oa[i] * (1 + slp)  # 슬리피지 적용 진입가
        pz = st['cash'] * 0.20; s = pz / ee  # 현금의 20%로 진입
        st['cash'] -= pz + fee; st['scalp_shares'] = s; st['scalp_entry'] = ee
        st['scalp_sl'] = ee*(1-dsl); st['scalp_tp'] = ee*(1+dtp); st['in_scalp'] = True
        st['trades'].append({'type': 'SCALP_ENTER', 'price': round(ee,2), 'shares': round(s,4)})
        log_detail('SCALP_ENTER', f'RSI={rsi_now:.1f}<25 SL={ee*(1-dsl):.0f} TP={ee*(1+dtp):.0f}', round(ee, 0), round(s, 4))
    elif st['in_scalp']:
        # 스캘핑 청산 로직: 저가 ≤ 손절가 → 손절, 고가 ≥ 익절가 → 익절
        sl = st['scalp_sl']; tp = st['scalp_tp']; entry = st['scalp_entry']; xp = None
        if la[i] <= sl: xp = sl  # 손절선 터치 시 청산
        elif ha[i] >= tp: xp = tp  # 익절선 터치 시 청산
        if xp:
            s = st['scalp_shares']; pnl = s*(xp-entry) - CONFIG['fee_per_side']
            st['cash'] += s*xp - fee; st['total_pnl'] += pnl; st['in_scalp'] = False; st['scalp_shares'] = 0
            st['trades'].append({'type': 'SCALP_EXIT', 'entry': round(entry,2), 'exit': round(xp,2), 'shares': round(s,4), 'pnl': round(pnl, 2)})
            log_detail('SCALP_EXIT', f'{"SL" if la[i]<=sl else "TP"} hit', round(xp, 0), round(s, 4), round(pnl, 0))
    st['last_trade_time'] = str(df.index[i])
    return st


# ---------------------------------------------------------------------------
# 종목 상태 초기화
# ---------------------------------------------------------------------------
def append_detail_log(sym, row):
    """Append a detailed trade log entry to kr_trade_log.json"""
    log_file = 'kr_trade_log.json'
    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f: existing = json.load(f)
        except: pass
    existing.append(row)
    with open(log_file, 'w') as f: json.dump(existing, f, indent=2, default=str)

def init_stock(symbol):
    """Create new stock state with initial capital."""
    return {'symbol': symbol, 'cash': CONFIG['init_cap'], 'shares': 0, 'in_bh': False,
            'bh_entry': None, 'bh_shares': 0, 'in_scalp': False, 'scalp_entry': None,
            'scalp_shares': 0, 'scalp_sl': None, 'scalp_tp': None, 'trades': [], 'total_pnl': 0.0,
            'last_trade_time': None}

# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------
def main():
    """
    실행 주기(5분)마다 호출되는 메인 함수.
    1) 매월 첫 실행 시 스크리닝으로 종목 교체 (리밸런싱).
    2) 기존 종목은 기존 자산가치를 유지한 채 재분배.
    3) 각 종목에 대해 process() 실행 → 매매 시그널 처리.
    4) 포트폴리오 상태 및 로그 저장.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] KR Engine")
    state = load_state(); now = datetime.now()
    # 월간 리밸런싱: 매월 첫 실행 또는 상태가 비어있으면 스크리닝으로 종목 교체
    if not state['stocks'] or state.get('last_rebalance_month') != now.month:
        picks = screen_market(CANDIDATES)
        # 기존 종목들의 현재 총 가치 계산
        old_val = 0
        for s in state['stocks']:
            tv = s['cash']
            try:
                p = get_current_price(s['symbol'])
                if s['in_bh']: tv += s['bh_shares'] * p
                if s['in_scalp']: tv += s['scalp_shares'] * p
            except: pass
            old_val += tv
        # 기존 자산을 신규 종목에 균등 분배
        per_stock = old_val / len(picks) if picks and old_val > 0 else CONFIG['init_cap']
        state['stocks'] = []
        for pick in picks:
            st = init_stock(pick['symbol']); st['cash'] = per_stock; state['stocks'].append(st)
        state['last_screen'] = now.strftime('%Y-%m-%d'); state['last_rebalance_month'] = now.month  # 리밸런싱 완료 기록
        print(f"  Rebalanced: {[p['symbol'] for p in picks]}")
    else:
        print(f"  Processing {len(state['stocks'])} stocks")

    # ---- 각 종목 처리 및 포트폴리오 집계 ----
    totals=[]       # 각 종목별 총 가치
    log_entries=[]  # 로그 저장용
    now_ts=datetime.now().strftime('%Y-%m-%d %H:%M')
    for st in state['stocks']:
        st = process(st)  # 매매 로직 실행
        tv = st['cash'];cur_p=0
        try:
            cur_p = get_current_price(st['symbol'])
            if st['in_bh']: tv += st['bh_shares'] * cur_p     # BH 보유분 평가
            if st['in_scalp']: tv += st['scalp_shares'] * cur_p  # 스캘핑 보유분 평가
        except: pass
        ret = (tv / CONFIG['init_cap'] - 1) * 100  # 초기 자본 대비 수익률
        mode = 'BH' if st['in_bh'] else ('SCALP' if st['in_scalp'] else 'CASH')
        price_str=f'{cur_p:,.0f}' if cur_p else 'N/A'
        print(f"  {st['symbol']:<12} {price_str:<10} {mode:<6} ₩{tv:,.0f} ({ret:+.1f}%) | {len(st['trades'])} trades")
        totals.append(tv)
        log_entries.append({
            'time': now_ts,'symbol': st['symbol'],'price': cur_p,'mode': mode,
            'value': tv,'return_pct': round(ret,2),'cash': round(st['cash'],2),
            'bh_shares': round(st.get('bh_shares',0),4),
            'scalp_shares': round(st.get('scalp_shares',0),4),
            'total_trades': len(st['trades']),
        })
    # 포트폴리오 총 가치 출력
    total_port_val = sum(totals)
    print(f"  Total: ₩{total_port_val:,.0f} ({(total_port_val/(len(state['stocks'])*CONFIG['init_cap'])-1)*100:+.1f}%)")
    # 상태 저장 & 로그 누적
    save_state(state)
    log_file = CONFIG['state_file'].replace('state.json','log.json')
    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f: existing = json.load(f)
        except: pass
    existing.extend(log_entries)  # 기존 로그에 이번 실행 로그 추가
    with open(log_file, 'w') as f: json.dump(existing, f, indent=2, default=str)
    print(f"  State saved | Log: {log_file}")


# ---------------------------------------------------------------------------
# 엔트리 포인트: 5분 간격 무한 루프 실행
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("KR Market Engine | 5D Hybrid | ₩1,000,000/stock | Ctrl+C to stop")
    while True:
        try:
            main()                    # 메인 로직 실행
            print(f"  [Next: 5min]\n")
            time.sleep(300)           # 5분(300초) 대기 후 재실행
        except KeyboardInterrupt:
            print("\n  Stopped"); break  # Ctrl+C 시 안전 종료
        except Exception as e:
            print(f"  Error: {e}"); time.sleep(60)  # 예외 발생 시 1분 후 재시도
