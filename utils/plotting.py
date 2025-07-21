import plotly.graph_objs as go
import pandas as pd
import streamlit as st

# Plot forecast vs actual prices (LSTM, XGBoost, Transformer, etc.)
def plot_forecast_chart(dates, actual, predicted, model_name="", lower=None, upper=None, currency="KSh"):
    fig = go.Figure()

    # Actual Prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    ))

    # Predicted Prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=4)
    ))

    # Confidence Interval (optional)
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

    fig.update_layout(
        title=f"{model_name} Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        template="plotly_white",
        height=500,
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

# Plot GARCH Volatility Forecast
def plot_volatility_chart(dates, volatility, var=None, currency="KSh"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=volatility,
        mode='lines',
        name='Forecasted Volatility',
        line=dict(color='orange', width=2)
    ))

    if var is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=var,
            mode='lines',
            name='Value at Risk (VaR)',
            line=dict(color='red', width=2, dash='dot')
        ))

    fig.update_layout(
        title="Volatility and Value at Risk Forecast",
        xaxis_title="Date",
        yaxis_title=f"Volatility / VaR ({currency})",
        template="plotly_white",
        height=500,
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)
