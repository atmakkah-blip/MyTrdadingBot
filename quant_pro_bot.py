import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import time
import requests
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 0. INSTITUTIONAL CONFIG
# ==========================================

st.set_page_config(
    layout="wide",
    page_title="Institutional Quant System",
    initial_sidebar_state="expanded"
)

# CSS for Pro Dashboard
st.markdown(
    """
<style>
    .metric-card { background: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .metric-val { font-size: 24px; font-weight: bold; color: #4caf50; }
    .metric-lbl { font-size: 14px; color: #aaa; }
    .risk-alert { color: #ff4b4b; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True
)

# ==========================================
# 1. DATA & SENTIMENT LAYER
# ==========================================

class DataFeed:
    def __init__(self):
        # If you need API keys, configure here:
        self.exchange = ccxt.binance()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=["time", "open", "high", "low", "close", "vol"]
            )
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_sentiment(self):
        """Fetches Crypto Fear & Greed Index"""
        try:
            r = requests.get("https://api.alternative.me/fng/")
            data = r.json()["data"][0]
            return int(data["value"]), data["value_classification"]
        except Exception:
            return 50, "Neutral"

# ==========================================
# 2. QUANT CORE (STRATEGY & BACKTESTER)
# ==========================================

class QuantCore:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators and regime filters."""
        if df.empty:
            return df

        # Institutional Feature Engineering
        df["RSI"] = df.ta.rsi(length=14)
        df["SMA_50"] = df.ta.sma(length=50)
        df["SMA_200"] = df.ta.sma(length=200)
        df["ATR"] = df.ta.atr(length=14)
        df["Vol_MA"] = df["vol"].rolling(20).mean()

        # Market Regime Filter (Price > 200 SMA = Bull)
        df["Regime"] = np.where(df["close"] > df["SMA_200"], 1, -1)
        df = df.dropna().copy()
        return df

    def train_ai(self, raw_df: pd.DataFrame) -> str:
        """Train Random Forest on passed data."""
        if raw_df is None or raw_df.empty:
            return "Training failed: no data."

        df = self.prepare_data(raw_df.copy())
        if df.empty or len(df) < 100:
            return "Training failed: not enough candles after indicator calculation."

        # Target: will next candle close higher than current?
        df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna().copy()

        features = ["RSI", "SMA_50", "ATR", "vol"]
        X = df[features]
        y = df["Target"]

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X, y)
        self.is_trained = True

        acc = self.model.score(X, y)
        return f"Model Trained (In-sample accuracy: {acc:.2%})"

    def predict(self, row: pd.Series) -> float:
        if not self.is_trained or self.model is None:
            return 0.5

        try:
            features = [row["RSI"], row["SMA_50"], row["ATR"], row["vol"]]
            proba = self.model.predict_proba([features])[0][1]
            return float(proba)
        except Exception:
            return 0.5

    def backtest(self, symbol: str, tf: str, initial_capital: float = 10000):
        """The Time Machine: Simulates strategy over history."""
        feed = DataFeed()
        df = feed.fetch_ohlcv(symbol, tf, limit=1000)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = self.prepare_data(df)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        balance = initial_capital
        position = 0.0
        equity_curve = []
        trades = []

        for _, row in df.iterrows():
            price = row["close"]

            # Buy Condition: Bull Regime + RSI Dip
            if position == 0 and row["Regime"] == 1 and row["RSI"] < 35:
                qty = (balance * 0.99) / price  # 99% equity
                position = qty
                balance -= qty * price
                trades.append(
                    {"time": row["time"], "type": "buy", "price": price}
                )

            # Sell Condition: RSI Overbought OR Trend Break
            elif position > 0 and (row["RSI"] > 75 or row["close"] < row["SMA_50"]):
                val = position * price
                balance += val
                position = 0
                trades.append(
                    {"time": row["time"], "type": "sell", "price": price}
                )

            curr_val = balance + (position * price)
            equity_curve.append({"time": row["time"], "equity": curr_val})

        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades)
        return equity_df, trades_df

# ==========================================
# 3. GUI & DASHBOARD
# ==========================================

feed = DataFeed()
quant = QuantCore()

# Sidebar Config
with st.sidebar:
    st.header("🏦 Institutional Controls")
    mode = st.radio(
        "System Mode",
        ["📡 Live Trading", "🧪 Backlab (Research)"]
    )
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["1h", "4h", "1d"])

    st.divider()
    st.caption("Risk Parameters")
    sl_atr = st.slider("ATR Stop Loss Multiplier", 1.0, 5.0, 2.0)

# ================= BACKLAB MODE ==================

if mode == "🧪 Backlab (Research)":
    st.title(f"🧪 Quantitative Research Lab: {symbol}")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.info(
            "Run a simulation on historical data to verify strategy robustness."
        )
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Crunching numbers..."):
                equity, trades = quant.backtest(symbol, tf)

                if equity.empty:
                    st.error(
                        "Backtest failed: no data or not enough candles."
                    )
                else:
                    # METRICS CALCULATION
                    total_return = (
                        (equity["equity"].iloc[-1] - 10000) / 10000
                    ) * 100

                    equity["peak"] = equity["equity"].cummax()
                    equity["dd"] = (
                        (equity["equity"] - equity["peak"])
                        / equity["peak"]
                    )
                    max_dd = equity["dd"].min() * 100

                    returns = equity["equity"].pct_change().dropna()
                    if returns.empty or returns.std() == 0:
                        sharpe = 0.0
                    else:
                        # simplified Sharpe, assume ~daily
                        sharpe = (
                            returns.mean() / returns.std()
                        ) * np.sqrt(252)

                    st.session_state.bt_results = {
                        "eq": equity,
                        "tr": trades,
                        "ret": total_return,
                        "mdd": max_dd,
                        "shp": sharpe,
                    }

    with col1:
        if "bt_results" in st.session_state:
            res = st.session_state.bt_results

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Total Return",
                f"{res['ret']:.2f}%",
                delta_color="normal"
            )
            m2.metric(
                "Max Drawdown",
                f"{res['mdd']:.2f}%",
                delta_color="inverse"
            )
            m3.metric("Sharpe Ratio", f"{res['shp']:.2f}")
            m4.metric("Trades Executed", len(res["tr"]))

            tab_eq, tab_dd = st.tabs(["Equity Curve", "Drawdown"])

            with tab_eq:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=res["eq"]["time"],
                        y=res["eq"]["equity"],
                        fill="tozeroy",
                        name="Equity",
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_dd:
                st.area_chart(res["eq"].set_index("time")["dd"])

            st.dataframe(res["tr"], use_container_width=True)

# ================= LIVE TRADING MODE ==================

elif mode == "📡 Live Trading":
    st.title("📡 Live Institutional Desk")

    # SENTIMENT HEADER
    sent_val, sent_class = feed.get_sentiment()
    s_color = (
        "red" if sent_val < 30 else "green" if sent_val > 70 else "orange"
    )

    st.markdown(
        f"""
        <div style='background: #111; padding: 20px; border-radius: 10px; 
                    display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <h3 style='margin:0'>Market Sentiment</h3>
                <span style='color: #888'>Fear & Greed Index</span>
            </div>
            <div style='text-align: right'>
                <h1 style='margin:0; color: {s_color}'>{sent_val}</h1>
                <span style='text-transform: uppercase; font-weight: bold; color: {s_color}'>{sent_class}</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Live Market Data")
        if st.button("Connect to Exchange Feed"):
            df_live = feed.fetch_ohlcv(symbol, tf, limit=500)
            if df_live.empty:
                st.error("No data returned. Check ticker or timeframe.")
            else:
                df_live = quant.prepare_data(df_live)
                if df_live.empty:
                    st.error(
                        "Not enough data after indicator calculation. Try another timeframe or symbol."
                    )
                else:
                    last = df_live.iloc[-1]
                    atr_stop = last["close"] - (last["ATR"] * sl_atr)

                    st.dataframe(df_live.tail(5), use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"${last['close']:.2f}")
                    c2.metric(
                        "Market Regime",
                        "BULL 🐂"
                        if last["Regime"] == 1
                        else "BEAR 🐻",
                    )
                    c3.metric(
                        "Suggested Stop Loss",
                        f"${atr_stop:.2f}"
                    )

    with col_r:
        st.subheader("AI Validator")
        if st.button("Train AI Model"):
            with st.spinner("Training on last 500 candles..."):
                raw_df = feed.fetch_ohlcv(symbol, tf, limit=500)
                msg = quant.train_ai(raw_df)
                if "failed" in msg.lower():
                    st.warning(msg)
                else:
                    st.success(msg)

        if quant.is_trained:
            st.info("AI is watching...")
        else:
            st.warning("Model Untrained")