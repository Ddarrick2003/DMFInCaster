import pandas as pd
import numpy as np
import shap
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def run_xgboost_forecast(df, forecast_days, currency):
    df = df.copy()
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Volatility"] = df["Return"].rolling(window=5).std().fillna(0)

    df['Target'] = df['Close'].shift(-forecast_days)
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    future_input = df[features].iloc[-forecast_days:]
    future_pred = model.predict(future_input)

    # Plot
    st.subheader("📉 XGBoost Forecast")
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({f"Forecast ({currency})": future_pred}, index=future_dates)
    st.line_chart(forecast_df)

    st.metric("📌 RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")

    # SHAP
    st.subheader("🔍 Feature Importance (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, max_display=6, show=False)
    st.pyplot(fig)
