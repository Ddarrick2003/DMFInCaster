import numpy as np

def calculate_backtest_metrics(pnl_series):
    pnl = pnl_series.dropna()
    returns = pnl.values
    if len(returns) == 0:
        return 0, 0, 0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(252)
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak)
    max_dd = np.min(drawdown) * -100
    return sharpe, sortino, max_dd
