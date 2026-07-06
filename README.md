# 5D Hybrid Scalping Strategy

## Overview

RSI(2) mean-reversion + 5-day return trend following hybrid strategy for US and Korean stock markets.

### Quick Start

```bash
# US Market
python live_us.py

# KR Market
python live_kr.py

# Dashboard
python dashboard.py
→ http://localhost:8081
```

### Files

| File | Description |
|------|-------------|
| `STRATEGY.md` | Full strategy documentation |
| `live_us.py` | US market live engine (293 S&P 500 stocks) |
| `live_kr.py` | KR market live engine (97 KOSPI stocks) |
| `dashboard.py` | Web dashboard (Flask + Chart.js) |
| `cloud_engine.py` | Backtest engine |
| `validate_*.py` | Validation scripts (89 tests, 89% win rate) |
| `weekly_screener.py` | Weekly stock screener |

### Strategy

- **Signal**: RSI(2) < 25 → Buy (Kelly 20% position)
- **Trend**: 5-day return > 0 → Switch to Buy & Hold
- **Risk**: ATR-based SL (0.1-0.8%) / TP (SL × 10)
- **Fees**: $2 round-trip + 0.01% slippage
- **Rebalance**: Quarterly

### Performance (89 backtests)

- Win rate: 89%
- Average return: +45%
- Worst MDD: 15%
