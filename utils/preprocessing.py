import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    # Convert prices to float (handle commas, symbols, blanks)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')  # NaNs allowed
        else:
            df[col] = np.nan  # Ensure all price columns exist

    df = df.dropna(subset=['Close'])  # Drop rows with no price data

    # Sort values for time series calculations
    df = df.sort_values(by=['Ticker', 'Date'])

    # Compute log volume
    df['Log_Volume'] = np.log1p(df['Volume'].fillna(0))

    # Compute daily returns per asset
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)

    # Add RSI
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(close=x, window=14).rsi())

    # Add MACD and signal
    macd = df.groupby('Ticker').apply(
        lambda g: MACD(close=g['Close'], window_slow=26, window_fast=12, window_sign=9)
    )
    df['MACD'] = df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd())
    df['MACD_Signal'] = df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd_signal())

    # Fill remaining NaNs safely
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df
