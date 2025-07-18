import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model.lstm_model import run_lstm
from model.garch_model import run_garch
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer
import base64

# Set Streamlit page config
st.set_page_config(page_title="FinCaster", page_icon=":chart_with_upwards_trend:", layout="wide")

# Theme styling
st.markdown("""
    <style>
        body {background-color: #f8fbfd;}
        h1 {color: #035d36;}
        .stButton>button {
            background-color: #035d36;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💹 FinCaster – AI-Powered Financial Forecasting")

# Currency and theme toggles
col1, col2 = st.columns(2)
with col1:
    currency = st.radio("Currency", ["KSh", "USD"], horizontal=True)
with col2:
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)

# File upload section
st.subheader("📂 Upload Financial Data")
uploaded_file = st.file_uploader("Upload a CSV file with columns: Date, Open, High, Low, Close, Volume, (optional) Ticker", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df.columns = df.columns.str.strip()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    if "Ticker" not in df.columns:
        df["Ticker"] = "Asset_1"

    tickers = df["Ticker"].unique().tolist()
    selected_ticker = st.selectbox("Select Asset", tickers)
    df_ticker = df[df["Ticker"] == selected_ticker].copy()

    # Display sample data
    st.write("📊 Sample Data Preview", df_ticker.tail())

    # Forecast window
    forecast_days = st.slider("Forecast Horizon (Days)", 5, 30, 10)

    # Model selection
    st.markdown("### 🔍 Select Models to Run")
    run_lstm_flag = st.checkbox("LSTM Model", value=True)
    run_garch_flag = st.checkbox("GARCH Volatility Model")
    run_xgb_flag = st.checkbox("XGBoost + SHAP")
    run_inf_flag = st.checkbox("Informer Transformer")
    run_autoformer_flag = st.checkbox("Autoformer/TFT")

    if st.button("🚀 Run Forecasting"):
        with st.spinner("Running selected models..."):

            # LSTM
            if run_lstm_flag:
                st.subheader("📈 LSTM Forecast")
                run_lstm(df_ticker, forecast_days, currency)

            # GARCH
            if run_garch_flag:
                st.subheader("📉 GARCH Volatility & VaR")
                try:
                    run_garch(df_ticker, currency)
                except Exception as e:
                    st.error(f"GARCH Error: {e}")

            # XGBoost + SHAP
            if run_xgb_flag:
                st.subheader("🌳 XGBoost Forecast with SHAP")
                df_clean = df_ticker.select_dtypes(include=[np.number]).dropna()
                if df_clean.empty:
                    st.error("❌ Cleaned data is empty. Check for NaNs or non-numeric issues.")
                else:
                    run_xgboost_with_shap(df_clean, forecast_days, currency)

            # Informer
            if run_inf_flag:
                st.subheader("🤖 Informer Transformer Forecast")
                df_ticker = df_ticker.copy()
df_ticker.dropna(inplace=True)
df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
df_ticker.set_index('Date', inplace=True)
df_ticker.sort_index(inplace=True)

                run_informer(df_ticker, forecast_days, currency)

            # Autoformer/TFT
            if run_autoformer_flag:
                st.subheader("🔁 Autoformer/TFT Transformer Forecast")
                run_autoformer(df_ticker, forecast_days, currency)

else:
    st.info("📥 Please upload a CSV file to begin.")

# Footer
st.markdown("---")
st.markdown("© 2025 FinCaster | Built with 💚 using Streamlit, LSTM, GARCH, Transformers & XGBoost.")
