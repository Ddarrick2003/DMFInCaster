# utils/helpers.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

# Convert price based on selected currency
def convert_currency(prices, currency="KSh"):
    return prices * 142 if currency == "KSh" else prices

# Compute Mean Absolute Error
def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Standard preprocessing function
def preprocess_data(df):
    df = df.copy()
    df.columns = [col.strip().capitalize() for col in df.columns]
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.dropna()
    return df

# Format currency for display
def format_currency(value, currency='KSh'):
    return f"KSh {value:,.2f}" if currency == 'KSh' else f"${value:,.2f}"

# Comparative Forecast vs Actual Plot
def plot_comparison(pred_df, actual_df, title, currency='KSh'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_df['Date'],
        y=convert_currency(actual_df['Actual'], currency),
        mode='lines+markers',
        name='Actual',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=convert_currency(pred_df['Forecast'], currency),
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        template='plotly_white',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    return fig

# Clean uploaded file based on opt-in
def auto_clean_dataframe(df, clean=True):
    if clean:
        df = df.copy()
        df.dropna(inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        df.reset_index(drop=True, inplace=True)
    return df

# Light theme color palette
def get_theme_colors():
    return {
        "primary": "#28a745",
        "secondary": "#f8f9fa",
        "background": "#ffffff",
        "text": "#000000"
    }

# Compute future dates for custom forecasting
def generate_future_dates(start_date, steps):
    return pd.date_range(start=start_date + pd.Timedelta(days=1), periods=steps)
