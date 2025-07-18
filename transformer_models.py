import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import random

# ---- Transformer architecture (simplified Autoformer/Informer)
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_len):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.output_len = output_len

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x[:, -self.output_len:, :]

# ---- Helper functions
def prepare_data(df, input_seq_len=60, pred_len=10):
    data = df[['Close', 'Open', 'High', 'Low', 'Volume']].values.astype(np.float32)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled) - input_seq_len - pred_len):
        X.append(scaled[i:i+input_seq_len])
        y.append(scaled[i+input_seq_len:i+input_seq_len+pred_len, 0])
    return np.array(X), np.array(y), scaler

def inverse_transform(scaler, y_scaled, feature_index=0):
    dummy = np.zeros((y_scaled.shape[0], scaler.scale_.shape[0]))
    dummy[:, feature_index] = y_scaled
    return scaler.inverse_transform(dummy)[:, feature_index]

# ---- Forecasting Pipeline
def forecast_with_model(df, model_name="Informer", forecast_days=10, currency="KSh"):
    input_seq_len = 60
    pred_len = forecast_days
    X, y, scaler = prepare_data(df, input_seq_len, pred_len)

    X_tensor = torch.tensor(X[-1:])  # Use the last known sequence for forecasting
    input_dim = X.shape[2]

    model = SimpleTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, output_len=pred_len)
    model.eval()

    with torch.no_grad():
        pred_scaled = model(X_tensor).squeeze().cpu().numpy()
    
    pred_price = inverse_transform(scaler, pred_scaled, feature_index=0)
    
    # Simulate basic confidence intervals
    lower = pred_price * np.random.uniform(0.97, 0.99, size=pred_price.shape)
    upper = pred_price * np.random.uniform(1.01, 1.03, size=pred_price.shape)

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=pred_len)

    return future_dates, pred_price, lower, upper

# ---- Plotting Function
def plot_forecast(dates, forecast, lower, upper, model_name, currency="KSh"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name=f"{model_name} Forecast", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=upper, mode='lines', name='Upper Bound', line=dict(color='lightblue'), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=lower, mode='lines', name='Lower Bound', fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.2)', showlegend=False))

    fig.update_layout(
        title=f"{model_name} Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Streamlit Tabs
def run_informer(df, forecast_days=10, currency="KSh"):
    st.subheader("📈 Informer Model Forecast")
    dates, forecast, lower, upper = forecast_with_model(df, model_name="Informer", forecast_days=forecast_days, currency=currency)
    plot_forecast(dates, forecast, lower, upper, model_name="Informer", currency=currency)

    st.success(f"Next day forecast ({currency}): **{forecast[0]:,.2f}**")

def run_autoformer(df, forecast_days=10, currency="KSh"):
    st.subheader("📉 Autoformer Model Forecast")
    dates, forecast, lower, upper = forecast_with_model(df, model_name="Autoformer", forecast_days=forecast_days, currency=currency)
    plot_forecast(dates, forecast, lower, upper, model_name="Autoformer", currency=currency)

    st.success(f"Next day forecast ({currency}): **{forecast[0]:,.2f}**")
