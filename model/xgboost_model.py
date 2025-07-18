# model/xgboost_model.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_xgboost_with_shap(df, forecast_days=10, currency='KSh'):
    st.subheader("📈 XGBoost Forecast with SHAP Explainability")

    try:
        df = df.copy()
        df = df.dropna()

        if 'Close' not in df.columns:
            st.error("Data must contain 'Close' column.")
            return

        df['Target'] = df['Close'].shift(-forecast_days)
        df = df.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
        model.fit(X_train, y_train)

        # Predict future
        future_input = df[features].iloc[-forecast_days:]
        future_preds = model.predict(future_input)

        # SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        st.markdown("### 🔍 SHAP Feature Importance")
        st_shap(shap.plots.beeswarm(shap_values, show=False), height=300)

        # Forecast chart
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            name='XGBoost Forecast',
            line=dict(color='green')
        ))
        fig.update_layout(title='XGBoost Price Forecast',
                          xaxis_title='Date',
                          yaxis_title=f'Predicted Price ({currency})')
        st.plotly_chart(fig, use_container_width=True)

        # Display predictions
        results = pd.DataFrame({'Date': future_dates, f'Forecasted Price ({currency})': future_preds})
        st.dataframe(results.set_index('Date'))

    except Exception as e:
        st.error(f"XGBoost Error: {e}")


# Utility to display SHAP plot in Streamlit
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 400, scrolling=True)
