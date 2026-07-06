#!/usr/bin/env python3
"""Live Dashboard — US + KR Portfolio Monitor"""
import json, os, time
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

def get_portfolio(market):
    """Get current portfolio state for a market"""
    state = load_json(f'{market}_state.json')
    log = load_json(f'{market}_log.json')
    stocks = state.get('stocks', [])
    # Determine init_cap from state file or market type
    init_cap = 1000000.0 if market == 'kr' else 1000.0
    result = []
    for s in stocks:
        # Use last known value from log
        tv = s['cash']
        for entry in reversed(log):
            if entry.get('symbol') == s['symbol']:
                tv = entry.get('value', tv)
                break
        ret = (tv / init_cap - 1) * 100
        mode = 'BH' if s.get('in_bh') else ('SCALP' if s.get('in_scalp') else 'CASH')
        result.append({
            'symbol': s['symbol'],
            'mode': mode,
            'value': round(tv, 0),
            'return': round(ret, 1),
            'trades': len(s.get('trades', [])),
            'init_cap': round(init_cap, 0),
        })
    return result

def get_trade_history(market):
    """Get all trades from state files"""
    state = load_json(f'{market}_state.json')
    all_trades = []
    for s in state.get('stocks', []):
        for t in s.get('trades', []):
            all_trades.append({
                'symbol': s['symbol'],
                'type': t.get('type', ''),
                'price': t.get('price', t.get('entry', '')),
                'exit': t.get('exit', ''),
                'shares': t.get('shares', ''),
                'pnl': t.get('pnl', ''),
            })
    return all_trades

def get_log_entries(market, limit=50):
    """Get recent log entries"""
    log = load_json(f'{market}_log.json')
    return log[-limit:]

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:-apple-system,system-ui,sans-serif;padding:20px}
h1{font-size:1.5em;margin-bottom:16px;color:#38bdf8}
h2{font-size:1.1em;margin:16px 0 8px;color:#818cf8}
.tables{display:flex;gap:20px;flex-wrap:wrap}
.market{flex:1;min-width:400px}
table{width:100%;border-collapse:collapse;margin-bottom:12px;font-size:0.85em}
th{background:#1e293b;padding:8px 10px;text-align:left;border-bottom:2px solid #334155}
td{padding:6px 10px;border-bottom:1px solid #1e293b}
tr:hover{background:#1e293b}
.green{color:#4ade80}.red{color:#f87171}.yellow{color:#fbbf24}.gray{color:#94a3b8}
.badge{padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold}
.badge-BH{background:#166534;color:#4ade80}.badge-SCALP{background:#713f12;color:#fbbf24}.badge-CASH{background:#1e293b;color:#94a3b8}
.updated{color:#64748b;font-size:0.75em;margin-top:8px}
</style></head><body>
<h1>5D Hybrid Trading Dashboard</h1>
<div class="tables">
<div class="market"><h2>US Market</h2><div id="us-portfolio"></div><div id="us-detail"></div></div>
<div class="market"><h2>KR Market</h2><div id="kr-portfolio"></div><div id="kr-detail"></div></div>
</div>
<div class="updated" id="updated"></div>
<script>
async function load() {
  const [usP, krP, usD, krD] = await Promise.all([
    fetch('/api/portfolio/us').then(r=>r.json()),
    fetch('/api/portfolio/kr').then(r=>r.json()),
    fetch('/api/detail/us').then(r=>r.json()),
    fetch('/api/detail/kr').then(r=>r.json())
  ]);
  renderPortfolio('us', usP); renderPortfolio('kr', krP);
  renderDetail('us', usD); renderDetail('kr', krD);
  document.getElementById('updated').textContent = 'Last update: ' + new Date().toLocaleTimeString();
}
function renderPortfolio(market, data) {
  let html = '<table><tr><th>Symbol</th><th>Mode</th><th>Value</th><th>Return</th><th>Trades</th></tr>';
  let total = 0;
  for (const s of data) {
    total += s.value;
    const cls = s.return > 0 ? 'green' : s.return < 0 ? 'red' : 'gray';
    html += `<tr><td>${s.symbol}</td><td><span class="badge badge-${s.mode}">${s.mode}</span></td>`
      + `<td>${s.value.toLocaleString()}</td><td class="${cls}">${s.return > 0 ? '+' : ''}${s.return}%</td>`
      + `<td>${s.trades}</td></tr>`;
  }
  html += `<tr style="font-weight:bold;border-top:2px solid #334155"><td colspan="2">Total</td><td>${total.toLocaleString()}</td><td class="${total>data[0].init_cap*data.length?'green':'red'}">${((total/(data[0].init_cap*data.length)-1)*100).toFixed(1)}%</td><td></td></tr>`;
  html += '</table>';
  document.getElementById(market + '-portfolio').innerHTML = html;
}
function renderDetail(market, data) {
  if (!data || !data.length) { document.getElementById(market+'-detail').innerHTML='<p class="gray">No trade log yet</p>'; return; }
  let html = '<table style="font-size:0.75em"><tr><th>Time</th><th>Sym</th><th>Price</th><th>RSI</th><th>Reg</th><th>Mode</th><th>Action</th><th>Reason</th><th>TrdPrice</th><th>Shares</th><th>P&L</th><th>Cash</th><th>Eqty</th><th>Ret%</th></tr>';
  for (const t of data.slice(-30).reverse()) {
    const pnlCls = t.pnl > 0 ? 'green' : t.pnl < 0 ? 'red' : 'gray';
    html += `<tr><td>${(t.time||'').slice(-11)}</td><td>${(t.symbol||'').replace('.KS','')}</td><td>${t.price||''}</td><td>${t.rsi||''}</td><td>${t.regime||''}</td><td>${t.mode_before||''}</td><td>${t.action||''}</td><td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reason||''}">${(t.reason||'').slice(0,25)}</td><td>${t.trade_price||''}</td><td>${t.trade_shares||''}</td><td class="${pnlCls}">${t.pnl||''}</td><td>${t.cash||''}</td><td>${t.equity||''}</td><td>${t.return_pct||''}</td></tr>`;
  }
  html += '</table>';
  document.getElementById(market + '-detail').innerHTML = html;
}
load(); setInterval(load, 60000);
</script></body></html>'''

@app.route('/api/portfolio/<market>')
def api_portfolio(market):
    return jsonify(get_portfolio(market))

@app.route('/api/detail/<market>')
def api_detail(market):
    log = load_json(f'{market}_trade_log.json')
    return jsonify(log[-50:])  # last 50 entries

if __name__ == '__main__':
    print("Dashboard: http://localhost:8081")
    app.run(host='0.0.0.0', port=8081, debug=False)
