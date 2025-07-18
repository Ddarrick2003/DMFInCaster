# model/garch_model.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

def run_garch(df, currency='KSh'):
    st.subheader("📊 GARCH Volatility Forecast")

    try:
        if 'Close' not in df.columns:
            st.error("'Close' column missing in uploaded data.")
            return

        df = df.copy()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()

        returns = df['LogReturns'].dropna() * 100  # convert to percentage scale

        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        forecasts = res.forecast(horizon=5)
        vol_forecast = np.sqrt(forecasts.variance.values[-1, :])

        # Plot volatility forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(len(vol_forecast))],
            y=vol_forecast,
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='orange')
        ))
        fig.update_layout(title='Forecasted Volatility (5-day horizon)',
                          xaxis_title='Future Days',
                          yaxis_title=f'Volatility (%) in {currency}')
        st.plotly_chart(fig, use_container_width=True)

        # Calculate Value-at-Risk (VaR)
        confidence_level = 0.95
        z_score = 1.65  # 95% confidence
        latest_price = df['Close'].iloc[-1]
        VaRs = latest_price * (vol_forecast / 100) * z_score

        st.markdown("### 📉 Value-at-Risk (VaR)")
        for i, var in enumerate(VaRs):
            st.write(f"Day {i+1}: {currency} {var:.2f}")

        st.success("✅ GARCH forecast and VaR calculated successfully.")

    except Exception as e:
        st.error(f"GARCH Error: {e}")
