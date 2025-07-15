# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from report.report_generator import generate_summary_pdf
from pdf.pdf_parser import extract_pdf_insights
from utils.sentiment import generate_mock_sentiment
from utils.metrics import calculate_backtest_metrics
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ CONFIG ------------------
st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞💵 FinCaster: Financial Forecasting App")

# ------------------ SIDEBAR ------------------
uploaded_file = st.sidebar.file_uploader("📤 Upload OHLCV CSV (w/ optional 'Ticker')", type=["csv"])
uploaded_pdf = st.sidebar.file_uploader("📄 Upload optional PDF report", type=["pdf"])
use_sentiment = st.sidebar.checkbox("🧠 Include Sentiment Overlay", value=True)
use_live = st.sidebar.checkbox("🌐 Use Live Market Data", value=False)
auto_clean = st.sidebar.checkbox("🧹 Auto-clean uploaded data", value=True)
tickers = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")

# ------------------ DATA LOADING ------------------
if use_live:
    symbols = [t.strip().upper() for t in tickers.split(",")]
    raw = yf.download(symbols, period="6mo", auto_adjust=True)
    if raw.empty:
        st.error("⚠️ Failed to fetch data from yfinance.")
        st.stop()
    close = raw['Close'].reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Close')
    volume = raw['Volume'].reset_index().melt(id_vars='Date', value_name='Volume')['Volume']
    close['Volume'] = volume
    for col in ['Open', 'High', 'Low']:
        close[col] = close['Close']
    df = close.copy()
elif uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns:
        st.error("❌ CSV must include a 'Date' column.")
        st.stop()
    
    if 'Ticker' not in df.columns:
        st.warning("⚠️ No 'Ticker' column found. Defaulting to 'ASSET'")
        df['Ticker'] = 'ASSET'
else:
    st.warning("❌ Please upload a merged CSV or enable live mode.")
    st.stop()

# ------------------ DATE CLEANING ------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

# ------------------ OPTIONAL AUTO CLEAN ------------------
if auto_clean:
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

# ------------------ COLUMN VALIDATION ------------------
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.columns = [col.strip() for col in df.columns]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {', '.join(missing)}")
    st.stop()

# Assign default ticker if missing
if 'Ticker' not in df.columns:
    df['Ticker'] = 'ASSET'

# ------------------ PREPROCESSING ------------------
df = preprocess_data(df)

# ------------------ FILTER VALID ASSETS ------------------
min_rows = 30
valid_assets = df.groupby('Ticker').filter(lambda x: len(x) >= min_rows)['Ticker'].unique()
if len(valid_assets) == 0:
    st.error("❌ No assets have enough clean data to process (min 30 rows).")
    st.stop()
df = df[df['Ticker'].isin(valid_assets)]
assets = df['Ticker'].unique()

# ------------------ PREVIEW & DOWNLOAD CLEANED DATA ------------------
st.subheader("📋 Preview of Loaded Data")
st.dataframe(df.head(20), use_container_width=True)

@st.cache_data
def convert_df_to_csv(data):
    return data.to_csv(index=False).encode('utf-8')

csv_cleaned = convert_df_to_csv(df)
st.download_button("📥 Download Cleaned Data", csv_cleaned, file_name="cleaned_data.csv", mime='text/csv')

if use_sentiment:
    df = generate_mock_sentiment(df)

# ------------------ PDF INSIGHTS ------------------
pdf_summary = ""
if uploaded_pdf:
    try:
        pdf_summary = extract_pdf_insights(uploaded_pdf)
        st.info("📄 Insights from uploaded PDF:\n\n" + pdf_summary)
    except Exception as e:
        st.warning(f"⚠️ Could not parse PDF: {e}")

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Multivariate LSTM",
    "📉 GARCH Risk + VaR",
    "📊 Strategy Backtest + PnL",
    "📑 Report + Export"
])

# ------------------ TAB 1: LSTM ------------------
with tab1:
    st.subheader("🔮 Multivariate LSTM Forecast")
    selected_asset = st.selectbox("📌 Select Asset", assets)
    df_asset = df[df['Ticker'] == selected_asset].copy()

    features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
    try:
        X, y = create_sequences(df_asset[features], target_col='Close')
        if X.shape[0] == 0:
            st.warning("⚠️ Not enough data to train LSTM.")
        else:
            split = int(len(X) * 0.8)
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            model.fit(X[:split], y[:split], epochs=10, batch_size=16,
                      validation_data=(X[split:], y[split:]), callbacks=[EarlyStopping(patience=3)], verbose=0)

            preds = model.predict(X[split:]).flatten()

            # Forecast 5 steps ahead
            last_input = X[-1:]
            future_preds = []
            for _ in range(5):
                pred = model.predict(last_input)[0][0]
                future_preds.append(pred)
                next_input = np.roll(last_input, -1, axis=1)
                next_input[0, -1, :] = last_input[0, -1, :]
                next_input[0, -1, features.index('Close')] = pred
                last_input = next_input

            st.plotly_chart(go.Figure([
                go.Scatter(y=y[split:], name="Actual"),
                go.Scatter(y=preds, name="Predicted")
            ]), use_container_width=True)

            st.metric("📌 Next Day Forecasted Price", f"${future_preds[0]:.2f}")
            st.write("📆 5-Day Forecast:", [f"${p:.2f}" for p in future_preds])

    except Exception as e:
        st.error(f"❌ LSTM error: {e}")

# ------------------ TAB 2: GARCH ------------------
with tab2:
    st.subheader("📉 GARCH Volatility + 1-Day VaR")
    try:
        df_garch = df[df['Ticker'] == selected_asset].copy()
        vol_forecast, var_1d = forecast_garch_var(df_garch)

        if isinstance(var_1d, (np.ndarray, pd.Series)):
            var_1d = var_1d.item()

        if isinstance(vol_forecast, (np.ndarray, pd.Series, list)) and len(vol_forecast) > 0:
            latest_vol = vol_forecast[-1]
            annualized_vol = np.sqrt(252) * latest_vol
            st.metric("🔻 1-Day VaR (95%)", f"{var_1d:.2f}%")
            st.metric("📈 Annualized Volatility", f"{annualized_vol:.2f}%")
            st.line_chart(vol_forecast)
        else:
            st.warning("⚠️ No volatility forecast returned.")
    except Exception as e:
        st.error(f"❌ GARCH Error: {e}")

# ------------------ TAB 3: Strategy ------------------
with tab3:
    st.subheader("⚙️ Strategy Backtest + Portfolio PnL")
    result_df = []
    for ticker in assets:
        dft = df[df['Ticker'] == ticker].copy()
        dft['Signal'] = np.where(
            (dft['MACD'] > dft['MACD_Signal']) & (dft['RSI'] < 70) &
            ((dft['Sentiment'] > 0) if use_sentiment and 'Sentiment' in dft else True), 1, 0)
        dft['PnL'] = dft['Returns'] * dft['Signal']
        result_df.append(dft)
    result_df = pd.concat(result_df)
    portfolio_pnl = result_df.groupby('Date')['PnL'].mean()
    st.line_chart(portfolio_pnl.cumsum())

    sharpe, sortino, max_dd = calculate_backtest_metrics(portfolio_pnl)
    st.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("📉 Sortino Ratio", f"{sortino:.2f}")
    st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")
    st.download_button("📥 Download Signals", result_df.to_csv(index=False), "strategy_signals.csv")

# ------------------ TAB 4: REPORT ------------------
with tab4:
    st.subheader("📄 Export Summary + PDF")
    if st.button("📝 Generate Report"):
        report_text = generate_summary_pdf(result_df, pdf_summary)
        st.success("✅ Summary ready!")
        st.download_button("📥 Download Summary", report_text, "FinCaster_Report.txt")
