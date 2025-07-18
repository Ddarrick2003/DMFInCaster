import numpy as np
import shap
import xgboost as xgb
import plotly.graph_objects as go
import pandas as pd

def run_xgboost_with_shap(df, forecast_days, currency):
    df = df.dropna()
    X = df.drop("Close", axis=1)
    y = df["Close"]

    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X, y)

    future_preds = model.predict(X.tail(forecast_days))
    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=y, name="Historical"))
    fig.add_trace(go.Scatter(x=dates, y=future_preds, name="XGBoost Forecast", line=dict(color="orange")))
    fig.update_layout(title="XGBoost Forecast", xaxis_title="Date", yaxis_title=f"Price ({currency})")
    fig.show()

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
