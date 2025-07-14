import pandas as pd
import numpy as np

def preprocess_data(df):
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
    df['Log_Volume'] = np.log(df['Volume'].replace(0, np.nan)).fillna(0)
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(14).apply(calc_rsi)).fillna(0)
    df['EMA12'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12).mean())
    df['EMA26'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26).mean())
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=9).mean())
    return df

def calc_rsi(series):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))
