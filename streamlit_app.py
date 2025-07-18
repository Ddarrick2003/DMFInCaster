import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Local imports for all models (make sure these files exist)
from lstm_model import run_lstm
from garch_model import run_garch
from xgboost_model import run_xgboost_with_shap
from transformer_models import run_informer, run_autoformer
from utils import preprocess_data

# App Config
st.set_page_config(page_title="FinCaster", page_icon=":chart_with_upwards_trend:", layout="wide")

# Theme settings
st.markdown(
    """
    <style>
        .main {background-color: #f5f8fb;}
        h1 {color: #035d36;}
        .stButton>button {
            background-color: #035d36;
            color: white;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("💹 FinCaster – AI-Powered Financial Forecasting")

# UI toggles
col1, col2 = st.columns(2)
with col1:
    currency = st.radio("Currency", ["KSh", "USD"], horizontal=True)
with col2:
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)

# Upload section
st.subheader("📂 Upload Financial Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read the file: {e}")
    else:
        df.columns = df.columns.str.strip()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date")

        if "Ticker" not in df.columns:
            df["Ticker"] = "Asset_1"

        # Select asset
        tickers = df["Ticker"].unique().tolist()
        selected_ticker = st.selectbox("Select Asset", tickers)

        df_ticker = df[df["Ticker"] == selected_ticker].copy()

        st.write("📊 Sample Data", df_ticker.tail())

        forecast_days = st.slider("Forecast Horizon (Days)", 5, 30, 10)

        # Select models
        st.markdown("### 🔍 Select Models to Run")
        run_lstm_flag = st.checkbox("LSTM Model", value=True)
        run_garch_flag = st.checkbox("GARCH Volatility Model")
        run_xgb_flag = st.checkbox("XGBoost + SHAP")
        run_inf_flag = st.checkbox("Informer Transformer")
        run_autoformer_flag = st.checkbox("Autoformer/TFT")

        if st.button("🚀 Run Forecasting"):
            with st.spinner("Running selected models..."):

                if run_lstm_flag:
                    st.subheader("📈 LSTM Forecast")
                    run_lstm(df_ticker, forecast_days, currency)

                if run_garch_flag:
                    st.subheader("📉 GARCH Volatility & VaR")
                    try:
                        run_garch(df_ticker, currency)
                    except Exception as e:
                        st.error(f"GARCH Error: {e}")

                if run_xgb_flag:
                    st.subheader("🌳 XGBoost Forecast with SHAP")
                    df_clean = df_ticker.select_dtypes(include=[np.number]).dropna()
                    if df_clean.empty:
                        st.error("❌ Cleaned data is empty. Check for NaNs or non-numeric issues.")
                    else:
                        run_xgboost_with_shap(df_clean, forecast_days, currency)

                if run_inf_flag:
                    st.subheader("🤖 Informer Forecast")
                    run_informer(df_ticker, forecast_days, currency)

                if run_autoformer_flag:
                    st.subheader("🔁 Autoformer/TFT Forecast")
                    run_autoformer(df_ticker, forecast_days, currency)
else:
    st.info("📥 Please upload a CSV file with columns like Date, Open, High, Low, Close, Volume, Ticker.")

# Footer
st.markdown("---")
st.markdown("© 2025 FinCaster | Built with 💚 using Streamlit, LSTM, GARCH, Transformers & XGBoost.")
