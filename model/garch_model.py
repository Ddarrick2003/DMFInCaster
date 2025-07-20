import pandas as pd
from arch import arch_model
from datetime import timedelta

def run_garch_forecast(df, forecast_horizon=10):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']].dropna()
    
    returns = 100 * df['Close'].pct_change().dropna()

    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)

    vol_forecast = forecast.variance.values[-1] ** 0.5

    forecast_dates = pd.date_range(start=returns.index[-1] + timedelta(days=1), periods=forecast_horizon)

    return returns, vol_forecast, forecast_dates

