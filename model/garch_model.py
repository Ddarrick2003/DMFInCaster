from arch import arch_model
import pandas as pd

def forecast_garch_var(df):
    returns = df['Returns'].dropna() * 100
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=5)
    vol = forecast.variance.values[-1]
    var_1d = 1.65 * (vol[0] ** 0.5)
    return pd.Series(vol), var_1d