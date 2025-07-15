# streamlit_app.py (Full Polished Version)
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
import time

st.set_page_config(page_title="FinCaster", layout="wide")

# ------------------ Session Init ------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

# ------------------ Header ------------------
st.markdown(
    "<h1 style='text-align: center; color: green;'>🌞💵 FInCaster</h1>",
    unsafe_allow_html=True
)
st.markdown("##### A next-gen financial forecasting suite using LSTM + GARCH + Strategy Backtests")

# ------------------ Sidebar ------------------
st.sidebar.header("🔧 Configuration")
uploaded_file = st.sidebar.file_uploader("📤 Upload your OHLCV CSV", type=["csv"])
uploaded_pdf = st.sidebar.file_uploader("📄 Upload optional PDF report", type=["pdf"])
use_sentiment = st.sidebar.checkbox("🧠 Include Sentiment Overlay", value=True)
use_live = st.sidebar.checkbox("🌐 Use Live Market Data", value=False)
tickers = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, MSFT")
# ------------------ Data Load ------------------
if use_live:
    symbols = [t.strip().upper() for t in tickers.split(",")]
    raw_data = yf.download(symbols, period="6mo", auto_adjust=True)
    if raw_data.empty:
        st.error("⚠️ Failed to fetch data from yfinance.")
        st.stop()
    data = raw_data['Close'].reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Close')
    data['Volume'] = raw_data['Volume'].reset_index().melt(id_vars='Date')['value']
    for col in ['Open', 'High', 'Low']:
        data[col] = data['Close']
    df = data.copy()

    # ✅ Clean Date column
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    except Exception as e:
        st.error(f"❌ Error parsing dates: {e}")
        st.stop()

elif uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Check for Date column
    if 'Date' not in df.columns:
        st.error("❌ Your CSV must have a 'Date' column.")
        st.stop()

    # ✅ Assign default ticker if missing
    if 'Ticker' not in df.columns:
        st.warning("⚠️ 'Ticker' column missing — assigning default value 'ASSET'")
        df['Ticker'] = 'ASSET'

    # ✅ Clean Date column
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().all():
            st.error("❌ All values in the 'Date' column are invalid.")
            st.stop()
        df = df.dropna(subset=['Date'])
    except Exception as e:
        st.error(f"❌ Error parsing dates: {e}")
        st.stop()

    # ✅ Clean numeric columns (remove commas from '1,000.25' style)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close'])

else:
    st.warning("❌ Please upload a merged CSV or enable live mode.")
    st.stop()


# ------------------ Preprocess ------------------
df['Date'] = pd.to_datetime(df['Date'])
df = preprocess_data(df)
if use_sentiment:
    df = generate_mock_sentiment(df)
assets = df['Ticker'].unique()

# ------------------ PDF Insights ------------------
pdf_summary = ""
if uploaded_pdf:
    try:
        pdf_summary = extract_pdf_insights(uploaded_pdf)
        st.info("📄 Insights from uploaded PDF:\n\n" + pdf_summary)
    except Exception as e:
        st.warning(f"⚠️ Could not parse PDF: {e}")

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Multivariate LSTM",
    "📉 GARCH Risk + VaR",
    "📊 Strategy Backtest + PnL",
    "📑 Report + Export"
])

# ------------------ TAB 1 ------------------
with tab1:
    st.subheader("🔮 LSTM Forecast")
    selected_asset = st.selectbox("📌 Select Asset", assets)
    df_asset = df[df['Ticker'] == selected_asset].copy()
    features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']

    try:
        with st.spinner("📚 Training LSTM model..."):
            X, y = create_sequences(df_asset[features], target_col='Close')
            split = int(len(X) * 0.8)
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            model.fit(X[:split], y[:split], epochs=10, batch_size=16,
                      validation_data=(X[split:], y[split:]),
                      callbacks=[EarlyStopping(patience=3)], verbose=0)
            preds = model.predict(X[split:]).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y[split:], name="Actual"))
        fig.add_trace(go.Scatter(y=preds, name="Predicted"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ LSTM error: {e}")

# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("📉 GARCH Volatility + 1-Day VaR")
    try:
        df_garch = df[df['Ticker'] == selected_asset].copy()
        vol_forecast, var_1d = forecast_garch_var(df_garch)
        st.metric("🔻 1-Day VaR (95%)", f"{var_1d:.2f}%")
        st.line_chart(vol_forecast)
    except Exception as e:
        st.error(f"GARCH Error: {e}")

# ------------------ TAB 3 ------------------
with tab3:
    st.subheader("⚙️ Backtest Strategy + Portfolio PnL")
    df_grouped = []
    for ticker in assets:
        dft = df[df['Ticker'] == ticker].copy()
        dft['Signal'] = np.where(
            (dft['MACD'] > dft['MACD_Signal']) & (dft['RSI'] < 70) &
            ((dft['Sentiment'] > 0) if use_sentiment and 'Sentiment' in dft else True), 1, 0)
        dft['PnL'] = dft['Returns'] * dft['Signal']
        dft['Ticker'] = ticker
        df_grouped.append(dft)
    result_df = pd.concat(df_grouped)
    portfolio_pnl = result_df.groupby('Date')['PnL'].mean()
    st.line_chart(portfolio_pnl.cumsum())

    sharpe, sortino, max_dd = calculate_backtest_metrics(portfolio_pnl)
    st.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("📉 Sortino Ratio", f"{sortino:.2f}")
    st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")

    st.download_button("📥 Download Signals", result_df.to_csv(index=False), "strategy_signals.csv")

# ------------------ TAB 4 ------------------
with tab4:
    st.subheader("📄 Generate Report")
    if st.button("📝 Generate Report"):
        with st.spinner("🧾 Compiling..."):
            report_text = generate_summary_pdf(result_df, pdf_summary)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.success("✅ Report ready!")
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"FinCaster_Report_{selected_asset}_{timestamp}.txt"
            )
