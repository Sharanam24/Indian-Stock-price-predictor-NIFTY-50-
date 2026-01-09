import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
import pytz
import ta

from datetime import timedelta
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FUTURE_MINUTES = 5
TRAIN_WINDOW = 300
IST = pytz.timezone("Asia/Kolkata")

# --------------------------------------------------
# NIFTY 50 STOCKS
# --------------------------------------------------
NSE_STOCKS = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Divis Labs": "DIVISLAB.NS",
    "Dr Reddys": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance": "RELIANCE.NS",
    "SBI Life": "SBILIFE.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "Wipro": "WIPRO.NS"
}

# --------------------------------------------------
# SAFE DATA FETCH
# --------------------------------------------------
@st.cache_data(ttl=60)
def fetch_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="1m",
            progress=False,
            threads=False
        )

        if df.empty:
            df = yf.download(
                symbol,
                period="1d",
                interval="1m",
                progress=False,
                threads=False
            )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "Time", "Date": "Time"}, inplace=True)

        df = (
            df.set_index("Time")
            .between_time("09:15", "15:30")
            .reset_index()
        )

        return df.dropna()

    except Exception:
        return pd.DataFrame()

# --------------------------------------------------
# SAFE SERIES
# --------------------------------------------------
def safe_series(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def add_features(df):
    df = df.copy()
    df["Close"] = safe_series(df, "Close")

    df["return"] = df["Close"].pct_change()

    for lag in [1, 2, 3, 5]:
        df[f"ret_lag_{lag}"] = df["return"].shift(lag)

    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], 20)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], 20)
    df["RSI"] = ta.momentum.rsi(df["Close"], 14)

    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)

    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], 14
    )

    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df["target"] = df["Close"].shift(-FUTURE_MINUTES)

    return df.dropna()

FEATURES = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_5",
    "SMA_20",
    "EMA_20",
    "RSI"
]

# --------------------------------------------------
# FUTURE PREDICTION
# --------------------------------------------------
def predict_future(df):
    train = df.iloc[-TRAIN_WINDOW:]

    scaler = StandardScaler()
    X = scaler.fit_transform(train[FEATURES])

    model = SVR(C=200, gamma="scale", epsilon=0.01)
    model.fit(X, train["target"])

    last_row = scaler.transform(df.iloc[-1:][FEATURES])
    return model.predict(last_row)[0]

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 NIFTY 50 AI Stock Predictor")

company = st.sidebar.selectbox("Select Stock", NSE_STOCKS.keys())
symbol = NSE_STOCKS[company]

data = fetch_data(symbol)

if data.empty:
    st.error("❌ Data unavailable. Try another stock.")
    st.stop()

data = add_features(data)

if len(data) < TRAIN_WINDOW:
    st.error("❌ Not enough clean data.")
    st.stop()

future_price = predict_future(data)
current_price = data["Close"].iloc[-1]

# --------------------------------------------------
# AUTO-ADJUSTING SIGNAL LOGIC (FIXED)
# --------------------------------------------------
price_change = abs(future_price - current_price)
adx = data["ADX"].iloc[-1]
vol_ratio = data["vol_ratio"].iloc[-1]
atr = data["ATR"].iloc[-1]

min_move = 0.4 * atr
adx_threshold = 15 + (atr / current_price) * 50
vol_threshold = 0.9 + (atr / current_price)

score = 0
if price_change >= min_move:
    score += 1
if adx >= adx_threshold:
    score += 1
if vol_ratio >= vol_threshold:
    score += 1

if score >= 2:
    if future_price > current_price:
        signal = "BUY"
    elif future_price < current_price:
        signal = "SELL"
    else:
        signal = "HOLD"
else:
    signal = "NO TRADE"

# --------------------------------------------------
# METRICS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"₹ {current_price:.2f}")
c2.metric(
    "Predicted Price (+5 min)",
    f"₹ {future_price:.2f}",
    f"{future_price - current_price:.2f}"
)
c3.metric("AI Signal", signal)

# --------------------------------------------------
# CHART
# --------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["Time"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

future_time = data["Time"].iloc[-1] + timedelta(minutes=FUTURE_MINUTES)

fig.add_trace(go.Scatter(
    x=[data["Time"].iloc[-1], future_time],
    y=[current_price, future_price],
    mode="lines+markers",
    name="Predicted Move",
    line=dict(dash="dot")
))

fig.update_layout(
    title=f"{company} – Adaptive AI Prediction",
    xaxis_title="Time (IST)",
    yaxis_title="Price (₹)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.warning("⚠️ Educational purpose only. Not financial advice.")
