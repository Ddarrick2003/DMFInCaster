import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import os
from model.lstm_model import run_lstm
from model.garch_model import run_garch
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

# App Branding
st.set_page_config(page_title="FinCaster", page_icon="📈", layout="wide")
st.markdown("""
    <style>
        body { background-color: #f9f9f9; }
        .main { padding: 1rem; }
        .stApp { background-color: #ffffff; }
        header, footer { visibility: hidden; }
        .block-container { padding-top: 2rem; }
        .css-1d391kg { padding: 2rem 1rem 1rem 1rem; }
        .css-1kyxreq { padding: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🌤️ FinCaster: Forecast the Future of Finance")

# User config panel
st.sidebar.header("🔧 Configure Analysis")
task_name = st.sidebar.text_input("Analysis Task Name", value="My Forecast")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=10)
run_clean = st.sidebar.checkbox("Auto-clean uploaded data", value=False)
currency = st.sidebar.selectbox("Select Pricing Currency", ["KSh", "USD"])

# Upload
st.sidebar.subheader("📤 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if run_clean:
        df = df.dropna().copy()
        df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    ticker_col = 'Ticker' if 'Ticker' in df.columns else None
    unique_tickers = df[ticker_col].unique() if ticker_col else ["Single Asset"]
    selected_ticker = st.sidebar.selectbox("Select Ticker", unique_tickers) if ticker_col else None
    df_ticker = df[df[ticker_col] == selected_ticker] if selected_ticker else df.copy()

    st.subheader(f"📊 Data Preview: {selected_ticker if selected_ticker else 'Uploaded Data'}")
    st.dataframe(df_ticker.tail())

    # Show module toggles
    module_tabs = st.tabs(["📈 LSTM", "📉 GARCH", "🌲 XGBoost", "🤖 Informer", "🔁 Autoformer/TFT"])

    with module_tabs[0]:
        st.header("📈 LSTM Forecast")
        run_lstm(df_ticker.copy(), forecast_days, currency)

    with module_tabs[1]:
        st.header("📉 GARCH Volatility Forecast & VaR")
        run_garch(df_ticker.copy(), forecast_days, currency)

    with module_tabs[2]:
        st.header("🌲 XGBoost Forecast with SHAP")
        run_xgboost_with_shap(df_ticker.copy(), forecast_days, currency)

    with module_tabs[3]:
        st.header("🤖 Informer Transformer Forecast")
        run_informer(df_ticker.copy(), forecast_days, currency)

    with module_tabs[4]:
        st.header("🔁 Autoformer/TFT Transformer Forecast")
        run_autoformer(df_ticker.copy(), forecast_days, currency)

else:
    st.info("Please upload OHLCV data with Date, Open, High, Low, Close, Volume [+ optional Ticker] columns to get started.")
