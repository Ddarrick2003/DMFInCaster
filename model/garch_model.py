from arch import arch_model

def forecast_garch_var(df):
    returns = df['Returns'] * 100  # scale
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')
    forecast = fitted.forecast(horizon=1)
    vol = forecast.variance.values[-1, 0] ** 0.5
    var_1d = -1.65 * vol
    return forecast.variance[-30:], var_1d

