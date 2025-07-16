import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from utils.metrics import calculate_backtest_metrics
from report.report_generator import generate_summary_pdf
from pdf.pdf_parser import extract_pdf_insights
from utils.sentiment import generate_mock_sentiment
from tensorflow.keras.callbacks import EarlyStopping
from io import BytesIO
from PIL import Image

# ------------------ CONFIG & THEME ------------------
st.set_page_config(page_title="FinCaster", layout="wide", page_icon="💹")
st.markdown("""
    <style>
    /* Hide default sidebar */
    [data-testid="stSidebar"] {visibility: hidden;}
    /* Custom background & theme */
    .main {background-color: #0D1117; color: white;}
    .stApp {background-color: #0D1117;}
    h1, h2, h3, .stMarkdown, .stDataFrame, .stMetricValue { color: #F2F2F2; }
    .css-1rs6os.edgvbvh3 { background-color: #1A1A1A; border-radius: 10px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ------------------ ICON + TITLE ------------------
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("download.jpeg", width=60)
with col2:
    st.title("FinCaster ⚡ | Your Smart Financial Forecasting Tool")

# ------------------ SESSION INIT ------------------
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# ------------------ HOME DASHBOARD ------------------
with st.expander("🏠 Create New Analysis Task", expanded=True):
    task_name = st.text_input("Enter Task Name")
    uploaded_file = st.file_uploader("Upload OHLCV CSV", type=['csv'])
    uploaded_pdf = st.file_uploader("Optional: Upload Company Report PDF", type=['pdf'])
    enable_lstm = st.checkbox("Enable LSTM Forecasting")
    enable_garch = st.checkbox("Enable GARCH Risk Analysis")
    enable_backtest = st.checkbox("Enable Backtest + Strategy")
    enable_report = st.checkbox("Enable Report Generator")
    use_sentiment = st.checkbox("Use Sentiment Overlay", value=True)
    auto_clean = st.checkbox("🧹 Clean Uploaded Data Automatically", value=False)

# ------------------ DATA LOADING ------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns:
        st.error("❌ CSV must have a 'Date' column.")
        st.stop()

    if 'Ticker' not in df.columns:
        df['Ticker'] = 'ASSET'

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Optional Clean
    if auto_clean:
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

    # Only keep required columns
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    df = df[[col for col in df.columns if col in required]]

    df = preprocess_data(df)

    if use_sentiment:
        df = generate_mock_sentiment(df)

    assets = df['Ticker'].unique()
    st.success(f"✅ Task '{task_name}' loaded with {len(df)} rows.")
    st.session_state.tasks.append({'name': task_name, 'data': df})

    st.dataframe(df.head(20), use_container_width=True)

# ------------------ TABS ------------------
if uploaded_file and task_name:
    tab_dict = {}
    if enable_lstm:
        tab_dict["🔮 LSTM"] = "lstm"
    if enable_garch:
        tab_dict["📉 GARCH"] = "garch"
    if enable_backtest:
        tab_dict["📊 Backtest"] = "backtest"
    if enable_report:
        tab_dict["📑 Report"] = "report"

    tabs = st.tabs(list(tab_dict.keys()))

    # ------------------ TAB: LSTM ------------------
    if "🔮 LSTM" in tab_dict:
        with tabs[list(tab_dict.keys()).index("🔮 LSTM")]:
            st.subheader("🔮 Multivariate LSTM Forecast")
            selected = st.selectbox("Select Asset", assets)
            data = df[df['Ticker'] == selected]
            features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
            try:
                X, y = create_sequences(data[features], target_col='Close')
                if len(X) == 0:
                    st.warning("⚠️ Not enough data.")
                else:
                    split = int(0.8 * len(X))
                    model = build_lstm_model((X.shape[1], X.shape[2]))
                    model.fit(X[:split], y[:split], epochs=10, batch_size=16,
                              validation_data=(X[split:], y[split:]), verbose=0,
                              callbacks=[EarlyStopping(patience=3)])
                    preds = model.predict(X[split:]).flatten()
                    # Forecast next 5 days
                    last_input = X[-1:]
                    future_preds = []
                    for _ in range(5):
                        pred = model.predict(last_input)[0][0]
                        future_preds.append(pred)
                        next_input = np.roll(last_input, -1, axis=1)
                        next_input[0, -1, :] = last_input[0, -1, :]
                        next_input[0, -1, features.index('Close')] = pred
                        last_input = next_input
                    st.line_chart(pd.DataFrame({'Actual': y[split:], 'Predicted': preds}))
                    st.metric("📌 Next Day Forecast", f"${future_preds[0]:.2f}")
                    st.write("📆 5-Day Forecast:", [f"${p:.2f}" for p in future_preds])
            except Exception as e:
                st.error(f"LSTM Error: {e}")

    # ------------------ TAB: GARCH ------------------
    if "📉 GARCH" in tab_dict:
        with tabs[list(tab_dict.keys()).index("📉 GARCH")]:
            st.subheader("📉 GARCH Volatility + VaR")
            selected = st.selectbox("Select Asset", assets)
            garch_df = df[df['Ticker'] == selected].dropna(subset=['Returns'])
            if garch_df.empty:
                st.warning("⚠️ Not enough returns data.")
            else:
                try:
                    vol_forecast, var_1d = forecast_garch_var(garch_df)
                    st.metric("🔻 1-Day VaR (95%)", f"{var_1d:.2f}%")
                    st.metric("📈 Annualized Volatility", f"{(np.sqrt(252)*vol_forecast[-1]):.2f}%")
                    st.line_chart(vol_forecast)
                except Exception as e:
                    st.error(f"GARCH Error: {e}")

    # ------------------ TAB: BACKTEST ------------------
    if "📊 Backtest" in tab_dict:
        with tabs[list(tab_dict.keys()).index("📊 Backtest")]:
            st.subheader("⚙️ Strategy Backtest + PnL")
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
            st.metric("📈 Sharpe", f"{sharpe:.2f}")
            st.metric("📉 Sortino", f"{sortino:.2f}")
            st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")

    # ------------------ TAB: REPORT ------------------
    if "📑 Report" in tab_dict:
        with tabs[list(tab_dict.keys()).index("📑 Report")]:
            st.subheader("📄 Export Summary Report")
            if st.button("📝 Generate Report"):
                report_text = generate_summary_pdf(result_df, extract_pdf_insights(uploaded_pdf) if uploaded_pdf else "")
                st.download_button("📥 Download Report", report_text, "FinCaster_Report.txt")
