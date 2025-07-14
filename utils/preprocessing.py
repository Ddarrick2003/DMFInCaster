import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    # Clean and convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('--', '', regex=False)
            .str.extract(r'([0-9.]+)')[0]
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing critical data
    df.dropna(subset=numeric_cols + ['Date'], inplace=True)

    # Sort for consistency
    df.sort_values(['Ticker', 'Date'], inplace=True)

    # Feature Engineering
    df['Log_Volume'] = np.log1p(df['Volume'])

    # Technical indicators
    df['RSI'] = df.groupby('Ticker')['Close'].transform(
        lambda x: RSIIndicator(close=x, window=14).rsi()
    )
    macd = df.groupby('Ticker')['Close'].transform(
        lambda x: MACD(close=x).macd_diff()
    )
    df['MACD'] = macd
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(lambda x: x.rolling(9).mean())

    # Returns
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)

    return df
