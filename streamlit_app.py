import streamlit as st
import pandas as pd
import numpy as np
from lstm_model import run_lstm_forecast
from garch_model import run_garch_forecast
from xgboost_model import run_xgboost_with_shap
from transformer_models import run_informer, run_autoformer

# App title
st.set_page_config(page_title="FinCaster", layout="wide", page_icon="📈")
st.title("🌍 FinCaster: AI-Powered Financial Forecasting App")

# Task configuration
st.sidebar.header("⚙️ Settings")
currency = st.sidebar.radio("Select Currency", ["KSh", "USD"], key="currency_toggle")
forecast_days = st.sidebar.slider("📅 Forecast Horizon (Days)", 5, 30, 10, key="horizon_days")
selected_model = st.sidebar.radio("Select Forecasting Model", ["LSTM", "GARCH", "XGBoost + SHAP", "Informer", "Autoformer"], key="model_toggle")

# Data upload
uploaded_file = st.file_uploader("📤 Upload your CSV file (Date, Open, High, Low, Close, Volume)", type=["csv"], key="file_upload")

# Validate and load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ Uploaded CSV must contain at least 'Date' and 'Close' columns.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Preview data
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.tail(10), use_container_width=True)

        # Run selected model
        st.markdown("---")
        if selected_model == "LSTM":
            run_lstm_forecast(df, forecast_days, currency)

        elif selected_model == "GARCH":
            run_garch_forecast(df, forecast_days, currency)

        elif selected_model == "XGBoost + SHAP":
            run_xgboost_with_shap(df, forecast_days, currency)

        elif selected_model == "Informer":
            run_informer(df, forecast_days, currency)

        elif selected_model == "Autoformer":
            run_autoformer(df, forecast_days, currency)
else:
    st.info("⬆️ Upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 FinCaster | AI-Powered Financial Forecasting Suite</div>",
    unsafe_allow_html=True
)
