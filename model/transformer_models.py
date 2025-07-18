import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_transformer_input(df, lookback):
    df = df.copy()
    df = df.dropna(subset=["Close"])
    df["Close"] = df["Close"].astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1)).astype("float32")

    X = []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i + lookback])
    return np.array(X), scaler

def run_informer(df, forecast_days, currency):
    lookback = 30
    X, scaler = preprocess_transformer_input(df, lookback)

    if len(X) == 0:
        st.error("Not enough data for Informer forecast.")
        return

    X_torch = torch.tensor(X[-1].reshape(1, lookback, 1)).float()

    # Dummy Informer: Replace with real model
    prediction_scaled = torch.randn(forecast_days).numpy() * 0.05 + X_torch[0, -1, 0].item()
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_forecast = pd.DataFrame({f"Informer Forecast ({currency})": prediction}, index=future_dates)

    st.subheader("🤖 Informer Transformer Forecast")
    st.line_chart(df_forecast)
    st.metric("📌 Final Forecasted Price", f"{prediction[-1]:,.2f} {currency}")

def run_autoformer(df, forecast_days, currency):
    lookback = 30
    X, scaler = preprocess_transformer_input(df, lookback)

    if len(X) == 0:
        st.error("Not enough data for Autoformer forecast.")
        return

    X_torch = torch.tensor(X[-1].reshape(1, lookback, 1)).float()

    # Dummy Autoformer: Replace with real model
    prediction_scaled = torch.randn(forecast_days).numpy() * 0.04 + X_torch[0, -1, 0].item()
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_forecast = pd.DataFrame({f"Autoformer Forecast ({currency})": prediction}, index=future_dates)

    st.subheader("🔁 Autoformer/TFT Transformer Forecast")
    st.line_chart(df_forecast)
    st.metric("📌 Final Forecasted Price", f"{prediction[-1]:,.2f} {currency}")
