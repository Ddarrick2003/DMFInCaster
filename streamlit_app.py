import streamlit as st
import pandas as pd
import numpy as np
from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

# App Config
st.set_page_config(page_title="FinCaster", layout="wide", page_icon="📈")
st.title("🌍 FinCaster: AI-Powered Financial Forecasting App")

# Sidebar: Settings
st.sidebar.header("⚙️ Settings")
currency = st.sidebar.radio("Select Currency", ["KSh", "USD"], key="currency_toggle")
forecast_days = st.sidebar.slider("📅 Forecast Horizon (Days)", 5, 30, 10, key="horizon_days")

# NEW: Option to run all models
run_all = st.sidebar.checkbox("Run All Models at Once", value=False, key="run_all_models")

# Individual model toggle only shown if not running all
if not run_all:
    selected_model = st.sidebar.radio("Select Forecasting Model", [
        "LSTM", "GARCH", "XGBoost + SHAP", "Informer", "Autoformer"
    ], key="model_toggle")

# File upload
uploaded_file = st.file_uploader("📤 Upload your CSV file (Date, Open, High, Low, Close, Volume)", type=["csv"], key="file_upload")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ Uploaded CSV must contain at least 'Date' and 'Close' columns.")
    else:
        # Preprocess
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Optional cleaning
        st.sidebar.markdown("🧹 **Data Options**")
        clean_option = st.sidebar.checkbox("Auto-clean missing data (drop NaNs)?", value=False, key="clean_toggle")
        if clean_option:
            df = df.dropna()
            st.success("✅ Missing values dropped automatically.")

        # Preview
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.tail(10), use_container_width=True)

        # Run selected model or all
        st.markdown("---")

        if run_all:
            st.subheader("📊 LSTM Forecast")
            run_lstm_forecast(df.copy(), forecast_days, currency)

            st.subheader("📈 GARCH Volatility Forecast")
            run_garch_forecast(df.copy(), forecast_days, currency)

            st.subheader("📉 XGBoost Forecast + SHAP")
            run_xgboost_with_shap(df.copy(), forecast_days, currency)

            st.subheader("🔮 Informer Forecast")
            run_informer(df.copy(), forecast_days, currency)

            st.subheader("🧠 Autoformer Forecast")
            run_autoformer(df.copy(), forecast_days, currency)

        else:
            # Only one selected model
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
