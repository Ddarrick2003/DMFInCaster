import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

def run_garch(df, currency):
    df["Return"] = 100 * df["Close"].pct_change().dropna()
    returns = df["Return"].dropna()

    model = arch_model(returns, vol='GARCH', p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=5)

    vol = np.sqrt(forecast.variance.values[-1, :])
    var_99 = res.conditional_volatility[-1] * 2.33  # 99% VaR

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=returns, name="Returns", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[-var_99], name="99% VaR", mode='markers', marker=dict(color="red", size=10)))
    fig.update_layout(title="GARCH Volatility Forecast & VaR", xaxis_title="Date", yaxis_title="Returns (%)")
    fig.show()
