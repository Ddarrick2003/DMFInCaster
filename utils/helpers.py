import pandas as pd
import plotly.graph_objs as go
import streamlit as st

# Converts USD prices to KSh based on a fixed exchange rate or from user input
def convert_currency(df, currency="KSh", exchange_rate=145.0):
    df_converted = df.copy()
    if currency == "KSh":
        price_cols = ['Open', 'High', 'Low', 'Close', 'Predicted', 'Actual']
        for col in price_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col] * exchange_rate
    return df_converted

# Display Mean Absolute Error (MAE) as a comparison chart
def display_mae_chart(mae_scores, currency="KSh"):
    if not mae_scores:
        st.warning("No MAE data to display.")
        return

    models = list(mae_scores.keys())
    mae_values = list(mae_scores.values())

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=mae_values,
            marker=dict(color='rgba(0, 128, 0, 0.7)'),
            text=[f"{v:.2f}" for v in mae_values],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title=f"Mean Absolute Error (MAE) Comparison",
        xaxis_title="Model",
        yaxis_title=f"MAE ({currency})",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
