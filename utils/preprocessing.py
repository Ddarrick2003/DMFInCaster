def preprocess_data(df):
    df = df.copy()
    df['Log_Volume'] = (df['Volume'] + 1).apply(np.log)
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short).mean()
    ema_long = series.ewm(span=long).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal
