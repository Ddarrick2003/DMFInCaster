import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from models.lstm_model import run_lstm_forecast
from models.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_forecast
from model.transformer_models import run_transformer_model
from utils.preprocessing import preprocess_data


# --------------------------- Page Config ---------------------------
st.set_page_config(page_title="FinCaster", layout="wide")

# --------------------------- Utility Functions ---------------------------
def plot_pred_vs_actual(actual, predicted, forecast_dates, currency="KSh"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='Actual Price',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predicted,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title='Predicted vs Actual Prices',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(actual) >= len(predicted):
        mae = mean_absolute_error(actual[-len(predicted):], predicted)
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} {currency}")

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.title("🔍 FinCaster")
    selected_model = st.radio("Choose Forecasting Model:", ["LSTM", "XGBoost", "Informer", "Autoformer"])
    currency = st.radio("Select Currency:", ["KSh", "USD"])
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# --------------------------- Load and Clean Data ---------------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    df = preprocess_data(data)

    if df is not None and not df.empty:
        st.success("✅ Data uploaded and cleaned successfully.")
    else:
        st.error("❌ Data format is incorrect or empty.")
        st.stop()
else:
    st.warning("📂 Please upload a CSV file to begin.")
    st.stop()

# --------------------------- LSTM Tab ---------------------------
if selected_model == "LSTM":
    st.header("📈 LSTM Forecasting")

    actual_prices, predicted_prices, forecast_dates = run_lstm_forecast(df)

    st.subheader("Forecasted Prices")
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        f"Predicted Price ({currency})": predicted_prices
    })
    st.dataframe(forecast_df)

    # Plot comparison
    plot_pred_vs_actual(actual_prices, predicted_prices, forecast_dates, currency)

# --------------------------- XGBoost Tab ---------------------------
elif selected_model == "XGBoost":
    st.header("📉 XGBoost Forecasting")

    actual_prices, predicted_prices, forecast_dates = run_xgboost_forecast(df)

    st.subheader("Forecasted Prices")
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        f"Predicted Price ({currency})": predicted_prices
    })
    st.dataframe(forecast_df)

    # Plot comparison
    plot_pred_vs_actual(actual_prices, predicted_prices, forecast_dates, currency)

# --------------------------- Transformer (Informer / Autoformer) ---------------------------
elif selected_model in ["Informer", "Autoformer"]:
    st.header(f"🔮 Transformer-Based Forecasting: {selected_model}")

    actual_prices, predicted_prices, forecast_dates = run_transformer_model(df, model_type=selected_model.lower())

    st.subheader("Forecasted Prices")
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        f"Predicted Price ({currency})": predicted_prices
    })
    st.dataframe(forecast_df)

    # Plot comparison
    plot_pred_vs_actual(actual_prices, predicted_prices, forecast_dates, currency)

# --------------------------- Footer ---------------------------
st.markdown("---")
st.caption("📊 FinCaster – Forecast smarter.")
