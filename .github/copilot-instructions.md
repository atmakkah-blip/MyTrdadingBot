# AI Coding Agent Instructions

## Project Overview
A Streamlit-based quantitative trading bot for cryptocurrency market analysis and algorithmic strategy backtesting. The bot connects to CCXT exchanges (Binance), executes ML-based predictions, and provides institutional-grade risk metrics.

## Architecture & Data Flow

### Core Components
1. **DataFeed** - Exchanges OHLCV data via CCXT (Binance), fetches crypto sentiment (Fear & Greed Index)
2. **QuantCore** - ML strategy engine: trains RandomForest on technical indicators, runs backtests with equity tracking
3. **GUI (Streamlit)** - Two modes:
   - `🧪 Backlab (Research)`: Historical backtesting with metrics (return, max drawdown, Sharpe ratio)
   - `📡 Live Trading`: Real-time data feeds, market regime detection, AI model training

### Data Pipeline
```
CCXT/API → DataFeed.fetch_ohlcv() → DataFrame(time, open, high, low, close, volume)
→ QuantCore.prepare_data() → Technical Features (RSI, SMA50/200, ATR, Vol_MA)
→ Market Regime (close > SMA200 = Bull/Bear) → QuantCore.train_ai() or backtest()
```

## Key Patterns & Conventions

### Feature Engineering (QuantCore.prepare_data)
- Use **pandas_ta** for all technical indicators: `df.ta.rsi()`, `df.ta.sma()`, `df.ta.atr()`
- Always call `.dropna()` after feature calculation to remove initialization NaNs
- **Market Regime Filter**: Bull = `close > SMA_200`, Bear = opposite (binary encoded as 1/-1)
- Features scale implicitly; no explicit normalization (RandomForest is tree-based)

### Strategy Logic Pattern (Backtest)
- **Buy**: `(Regime == Bull) AND (RSI < 35)` → Deploy 99% of available balance
- **Sell**: `(RSI > 75) OR (close < SMA_50)` → Exit to cash
- Equity tracking via `balance + (position * current_price)` at each bar
- Trades logged as `{'time', 'type': 'buy'|'sell', 'price'}` for later analysis

### Metric Calculations
- **Total Return**: `((final_equity - initial_capital) / initial_capital) * 100`
- **Max Drawdown**: `min((equity - peak) / peak)` across equity curve
- **Sharpe Ratio**: `(mean_returns / std_returns) * sqrt(24)` for hourly bars (annualized approx)

### Streamlit UI Conventions
- Use `with st.sidebar:` block for controls (symbol, timeframe, risk params)
- Metric cards styled with dark theme CSS (`.1e1e1e` background, `#4caf50` green accent)
- Use `st.session_state['bt_results']` to persist backtest results between reruns
- Tabs for multi-view viz: `st.tabs(["Equity Curve", "Drawdown"])` + `go.Scatter` for Plotly
- Two-column layouts for data vs. controls: `col1, col2 = st.columns([3, 1])`

## Critical Implementation Details

### Dependencies (requirements.txt)
- **streamlit**: UI framework (rerun-based, no persistent state)
- **pandas + pandas_ta**: Data manipulation + technical indicators
- **ccxt**: Exchange API abstraction (Binance hardcoded in DataFeed.__init__)
- **scikit-learn**: RandomForest (n_estimators=100, max_depth=5, seed=42)
- **plotly**: Interactive charts (use `go.Scatter` with fill='tozeroy')
- **requests**: HTTP calls (sentiment API: `api.alternative.me/fng/`)

### Known Limitations
- No fee simulation in backtest (simplified for speed)
- RandomForest train accuracy ~65% (acknowledged in code)
- Sharpe calculation is annualized approximation (not true time-series Sharpe)
- AI prediction only returns win probability, not directly used in live mode

## Common Development Tasks

### Add a New Technical Indicator
1. In `QuantCore.prepare_data()`, add `df['NewIndicator'] = df.ta.<method>()`
2. Update `features` list in `train_ai()` if using for ML
3. Ensure no NaN rows post-calculation

### Modify Strategy Logic
1. Edit conditions in `backtest()` loop: `if position == 0 and <conditions>`
2. Test via Backlab tab (🧪) with BTC/USDT, 1h timeframe
3. Monitor equity curve & max drawdown; Sharpe ratios < 1.0 suggest poor fit

### Add Real-Time Alerts
1. In `Live Trading` mode, extend condition checks after `prepare_data()`
2. Use `st.warning()` / `st.success()` for alert styling
3. Example: `if last_close > sma_200: st.success("Bull signal")`

## Integration Points
- **CCXT**: Hardcoded Binance; swap via `ccxt.<exchange_name>()` in DataFeed.__init__
- **Sentiment API**: `api.alternative.me/fng/` (no auth required); wrap in try/except
- **ML Model Persistence**: Currently none; model retrains on each "Train AI" button click
- **Backtesting Timeline**: Fixed 1000-bar lookback; modify `limit=1000` in `backtest()` call

## Testing Patterns
- Use Backlab mode with small historical windows (default 500-1000 bars)
- Monitor equity curve shape (smooth growth = healthy strategy)
- Max drawdown > 50% indicates over-leveraging or poor entry logic
- Sharpe < 0 signals losing strategy

---
*Last updated: 2025-12-04*
