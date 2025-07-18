import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model.informer_model import InformerModel
from model.autoformer_model import AutoformerModel

def preprocess_series(df):
    df = df.copy()
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.fillna(method='ffill')
    return df

def run_informer(df, forecast_days, currency):
    df = preprocess_series(df)
    sequence = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    model = InformerModel(input_dim=sequence.shape[2], forecast_steps=forecast_days)
    model.eval()

    with torch.no_grad():
        output = model(sequence)

    forecast = output.squeeze().numpy()
    forecast = forecast.flatten()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast, index=future_dates)

    plot_forecast(df['Close'], forecast_series, 'Informer Forecast', currency)

def run_autoformer(df, forecast_days, currency):
    df = preprocess_series(df)
    sequence = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    model = AutoformerModel(input_dim=sequence.shape[2], forecast_steps=forecast_days)
    model.eval()

    with torch.no_grad():
        output = model(sequence)

    forecast = output.squeeze().numpy().flatten()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast, index=future_dates)

    plot_forecast(df['Close'], forecast_series, 'Autoformer Forecast', currency)

def plot_forecast(actual_series, forecast_series, title, currency):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_series.index, y=actual_series.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines+markers', name='Forecast'))

    # Confidence band (dummy ±5%)
    ci_upper = forecast_series * 1.05
    ci_lower = forecast_series * 0.95
    fig.add_trace(go.Scatter(x=forecast_series.index, y=ci_upper, name='Upper CI',
                             line=dict(width=0), mode='lines',
                             showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=ci_lower, name='Lower CI',
                             fill='tonexty', line=dict(width=0),
                             mode='lines', fillcolor='rgba(0,100,80,0.2)',
                             showlegend=False))

    fig.update_layout(title=title,
                      xaxis_title='Date',
                      yaxis_title=f'Price ({currency})',
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
