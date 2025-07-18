import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import shap
import joblib
import base64

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import timedelta
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

import torch
from transformer_models import run_informer, run_autoformer  # custom modules

st.set_page_config(page_title="FinCaster", layout="wide", page_icon="💹")

# ------------------ Theme and Currency Toggle ------------------
st.sidebar.markdown("## ⚙️ Settings")
currency = st.sidebar.radio("Currency", ("KSh", "USD"))
theme_mode = st.sidebar.radio("Theme", ("Light", "Dark"))

@st.cache_data
def convert_currency(value):
    return value * 140 if currency == "KSh" else value

# ------------------ File Upload and Basic Info ------------------
st.title("📊 FinCaster: Advanced Financial Forecasting Suite")

uploaded_file = st.file_uploader("Upload CSV with OHLCV + Ticker", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Handle missing Ticker
    if "Ticker" not in df.columns:
        st.warning("No 'Ticker' column found — assuming single asset data.")
        df["Ticker"] = "Asset_1"
    tickers = df["Ticker"].unique().tolist()

    # Asset selector
    selected_ticker = st.selectbox("Select Asset", tickers, key="select_asset_main")
    data = df[df["Ticker"] == selected_ticker].copy()
    data.set_index("Date", inplace=True)

    st.markdown(f"### 📈 Price Chart for {selected_ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=convert_currency(data["Close"]),
                             mode='lines', name='Close Price (converted)'))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ Tabs for Each Model ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 LSTM", "📉 GARCH", "📊 XGBoost + SHAP", "🔮 Transformers", "📄 Summary"])

    with tab1:
        st.subheader("LSTM Forecasting")
        st.info("Uses deep learning to predict future prices based on past patterns.")

        # Placeholder: actual LSTM model
        st.success("Predicted Next 10 Days:")
        future_dates = pd.date_range(data.index[-1] + timedelta(1), periods=10)
        lstm_preds = np.linspace(data["Close"].iloc[-1], data["Close"].iloc[-1] * 1.1, 10)
        lstm_preds = [convert_currency(p) for p in lstm_preds]
        st.line_chart(pd.Series(lstm_preds, index=future_dates))

    with tab2:
        st.subheader("GARCH Volatility Modeling")
        returns = 100 * data["Close"].pct_change().dropna()
        try:
            model = arch_model(returns, vol="Garch", p=1, q=1)
            garch_res = model.fit(disp="off")
            forecasts = garch_res.forecast(horizon=10)
            vol_forecast = forecasts.variance.values[-1, :]
            st.line_chart(pd.Series(np.sqrt(vol_forecast), index=future_dates))
        except:
            st.error("GARCH Error: -1 (Could not estimate volatility)")

    with tab3:
        st.subheader("XGBoost Forecast + SHAP Explainability")
        forecast_days = st.slider("Forecast Days", 5, 30, 10)
        data["Returns"] = data["Close"].pct_change()
        data.dropna(inplace=True)

        X = data[["Open", "High", "Low", "Close", "Volume"]]
        y = data["Close"].shift(-forecast_days).dropna()
        X = X.iloc[:-forecast_days]

        model = XGBRegressor()
        model.fit(X, y)
        preds = model.predict(X[-forecast_days:])
        st.line_chart(pd.Series(convert_currency(preds), index=future_dates[:forecast_days]))

        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        st.subheader("Feature Importance (SHAP)")
        st.pyplot(shap.plots.beeswarm(shap_values, show=False))

    with tab4:
        st.subheader("Transformer-Based Forecasting")
        transformer_model = st.selectbox("Choose Transformer Model", ["Informer", "Autoformer/TFT"])
        forecast_len = st.slider("Forecast Days", 5, 20, 10, key="transformer_days")
        if transformer_model == "Informer":
            run_informer(data, forecast_len, currency)
        else:
            run_autoformer(data, forecast_len, currency)

    with tab5:
        st.subheader("📄 Summary")
        st.write("Model comparisons and export features coming soon...")

else:
    st.warning("Please upload your dataset (CSV) with columns: Date, Open, High, Low, Close, Volume, Ticker")
