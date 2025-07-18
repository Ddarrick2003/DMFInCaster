import numpy as np
import pandas as pd
import streamlit as st
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def run_xgboost_with_shap(df, forecast_days, currency):
    df = df.copy()
    df = df.dropna()
    df["Close"] = df["Close"].astype("float32")

    # Lag features
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    df.dropna(inplace=True)

    features = [col for col in df.columns if col.startswith("lag")]
    X = df[features].astype("float32").values
    y = df["Close"].astype("float32").values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Forecast next days using last known values
    last_known = df[features].iloc[-1].values.reshape(1, -1)
    predictions = []
    for _ in range(forecast_days):
        next_pred = model.predict(last_known)[0]
        predictions.append(next_pred)
        last_known = np.roll(last_known, -1)
        last_known[0, -1] = next_pred  # insert new prediction

    # Create SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    st.subheader("📊 XGBoost Forecast")
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({f"Forecast ({currency})": predictions}, index=future_dates)
    st.line_chart(forecast_df)
    st.metric("📌 Final Predicted Price", f"{predictions[-1]:,.2f} {currency}")

    st.subheader("🔍 SHAP Feature Importance")
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    st.bar_chart(mean_shap)
