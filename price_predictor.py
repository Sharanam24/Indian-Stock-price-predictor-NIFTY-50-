import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import timedelta
import pytz
import ta

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# NSE STOCK LIST
# --------------------------------------------------
NSE_STOCKS = {
    "Nestle India": "NESTLEIND.NS",
    "BEL": "BEL.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Axis Bank": "AXISBANK.NS",
    "Tata Steel": "TATASTEEL.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Titan": "TITAN.NS",
    "Cipla": "CIPLA.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "SBI Life": "SBILIFE.NS",
    "NTPC": "NTPC.NS",
    "State Bank of India": "SBIN.NS",
    "Hindalco": "HINDALCO.NS",
    "Power Grid": "POWERGRID.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Trent": "TRENT.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Grasim": "GRASIM.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Coal India": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "L&T": "LT.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Tech Mahindra": "TECHM.NS",
    "Zomato": "ZOMATO.NS",
    "TCS": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Jio Financial": "JIOFIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Wipro": "WIPRO.NS",
    "HCL Tech": "HCLTECH.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# --------------------------------------------------
# FETCH DATA (REAL-TIME SAFE)
# --------------------------------------------------
@st.cache_data(ttl=60)
def fetch_stock_data(ticker):
    data = yf.download(ticker, period="1d", interval="1m")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# --------------------------------------------------
# PROCESS DATA (IST + NSE HOURS)
# --------------------------------------------------
def process_data(data):
    if data.empty:
        return data

    ist = pytz.timezone("Asia/Kolkata")
    if data.index.tz is None:
        data.index = data.index.tz_localize(ist)
    else:
        data.index = data.index.tz_convert(ist)

    data.reset_index(inplace=True)
    data.rename(columns={"Date": "Datetime"}, inplace=True)

    data = data.set_index("Datetime").between_time("09:15", "15:30").reset_index()
    return data


# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------
def add_technical_indicators(data):
    close = data["Close"].squeeze()
    data["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    data["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    return data


# --------------------------------------------------
# PREDICT NEXT 5 MINUTES (SVR)
# --------------------------------------------------
def predict_next_5_minutes(data):
    df = data[["Close", "SMA_20", "EMA_20"]].dropna()

    if len(df) < 30:
        return None, None

    X = df[["SMA_20", "EMA_20"]].values
    y = df["Close"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = SVR(kernel="rbf")
    model.fit(X_scaled, y_scaled)

    last_input = X_scaled[-1].reshape(1, -1)

    predictions = []
    for _ in range(5):
        pred_scaled = model.predict(last_input)
        pred_price = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0][0]
        predictions.append(pred_price)

    avg_prediction = sum(predictions) / len(predictions)
    return predictions, avg_prediction


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Indian Stock Real-Time Predictor (NSE)")

st.sidebar.header("📡 Stock Selection")
company = st.sidebar.selectbox("Select Company", list(NSE_STOCKS.keys()))
symbol = NSE_STOCKS[company]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = fetch_stock_data(symbol)
data = process_data(data)

if data.empty:
    st.warning("⚠️ Market closed or data unavailable.")
    st.stop()

data = add_technical_indicators(data)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
last_price = float(data["Close"].iloc[-1])
open_price = float(data["Close"].iloc[0])
change = last_price - open_price
pct = (change / open_price) * 100

predictions, avg_predicted_price = predict_next_5_minutes(data)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"₹ {last_price:.2f}",
    f"{change:.2f} ({pct:.2f}%)"
)

if predictions:
    col2.metric(
        "Predicted Avg (Next 5 min)",
        f"₹ {avg_predicted_price:.2f}",
        f"{avg_predicted_price - last_price:.2f}"
    )
else:
    col2.metric("Predicted Avg (Next 5 min)", "Not enough data")

col3.metric("Stock", company)

# --------------------------------------------------
# BUY / SELL / HOLD SIGNAL
# --------------------------------------------------
if predictions:
    if avg_predicted_price > last_price * 1.003:
        st.success("📈 AI Signal: BUY")
    elif avg_predicted_price < last_price * 0.997:
        st.error("📉 AI Signal: SELL")
    else:
        st.info("⏸️ AI Signal: HOLD")

# --------------------------------------------------
# CHART
# --------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["Datetime"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=data["Datetime"],
    y=data["SMA_20"],
    name="SMA 20"
))

fig.add_trace(go.Scatter(
    x=data["Datetime"],
    y=data["EMA_20"],
    name="EMA 20"
))

# Predicted future prices
if predictions:
    future_times = [
        data["Datetime"].iloc[-1] + timedelta(minutes=i)
        for i in range(1, 6)
    ]

    fig.add_trace(go.Scatter(
        x=future_times,
        y=predictions,
        mode="lines+markers",
        name="Predicted (Next 5 min)",
        line=dict(dash="dot")
    ))

fig.update_layout(
    title=f"{company} – Real-Time NSE Chart",
    xaxis_title="Time (IST)",
    yaxis_title="Price (₹)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.warning(
    "⚠️ Educational purpose only. Not SEBI registered. "
    "Do NOT use for real trading decisions."
)
