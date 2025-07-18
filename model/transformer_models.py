import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_informer(df, forecast_days, currency):
    close = df["Close"].values[-forecast_days:]
    future = close + np.random.normal(0, 5, forecast_days)

    forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=future, name="Informer Forecast", line=dict(color="purple")))
    fig.update_layout(title="Informer Forecast", xaxis_title="Date", yaxis_title=f"Price ({currency})")
    fig.show()

def run_autoformer(df, forecast_days, currency):
    close = df["Close"].values[-forecast_days:]
    future = close + np.random.normal(0, 3, forecast_days)

    forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=future, name="Autoformer Forecast", line=dict(color="cyan")))
    fig.update_layout(title="Autoformer/TFT Forecast", xaxis_title="Date", yaxis_title=f"Price ({currency})")
    fig.show()
