# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_forecast
from transformer_model import run_transformer_forecast

from utils.helpers import convert_currency, display_mae_chart
from utils.plotting import plot_forecast_chart, plot_volatility_chart
from utils.theme import set_page_config, inject_custom_css

# ------------------ SETUP ------------------ #
set_page_config()
inject_custom_css()
st.title("📈 FinCaster - AI Forecasting App")

# ------------------ SIDEBAR ------------------ #
st.sidebar.header("⚙️ Configuration")
model_selection = st.sidebar.multiselect("Select Models", ["LSTM", "GARCH", "XGBoost", "Transformer"], default=["LSTM"])
currency = st.sidebar.radio("Currency", ["KSh", "USD"], horizontal=True)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 10)

# ------------------ FILE UPLOAD ------------------ #
st.subheader("1. Upload Market Data (CSV)")
file = st.file_uploader("Upload CSV", type="csv")

if file:
    try:
        df = pd.read_csv(file)
        if 'Date' not in df.columns:
            st.error("❌ The file must contain a 'Date' column.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else "Asset"
            st.success(f"✅ Data loaded for: {ticker}")
            st.dataframe(df.tail())

            # ------------------ RUN MODELS ------------------ #
            results = {}

            if "LSTM" in model_selection:
                with st.spinner("Running LSTM..."):
                    lstm_pred, lstm_actual, mae = run_lstm_forecast(df, forecast_days)
                    results['LSTM'] = (lstm_pred, lstm_actual, mae)

            if "GARCH" in model_selection:
                with st.spinner("Running GARCH..."):
                    garch_vol, garch_var = run_garch_forecast(df)
                    results['GARCH'] = (garch_vol, garch_var)

            if "XGBoost" in model_selection:
                with st.spinner("Running XGBoost..."):
                    xgb_pred, xgb_actual, shap_fig, mae = run_xgboost_forecast(df, forecast_days)
                    results['XGBoost'] = (xgb_pred, xgb_actual, shap_fig, mae)

            if "Transformer" in model_selection:
                with st.spinner("Running Transformer..."):
                    transformer_pred, transformer_actual, mae = run_transformer_forecast(df, forecast_days)
                    results['Transformer'] = (transformer_pred, transformer_actual, mae)

            # ------------------ DISPLAY ------------------ #
            st.subheader("2. Forecast Results")
            for model_name, output in results.items():
                st.markdown(f"### 🔹 {model_name} Forecast")

                if model_name == "GARCH":
                    vol_fig, var_fig = plot_volatility_chart(output[0], output[1])
                    st.plotly_chart(vol_fig, use_container_width=True)
                    st.plotly_chart(var_fig, use_container_width=True)
                else:
                    pred, actual, *extras = output
                    fig = plot_forecast_chart(actual, pred, model_name, currency)
                    st.plotly_chart(fig, use_container_width=True)
                    display_mae_chart(actual, pred, model_name)

                    if model_name == "XGBoost" and len(extras) > 0:
                        shap_fig = extras[0]
                        st.subheader("XGBoost SHAP Feature Importance")
                        st.pyplot(shap_fig)

    except Exception as e:
        st.error(f"Data processing error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
