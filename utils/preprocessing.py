import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    df = df.copy()

    # Clean numeric columns: remove commas and convert to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

    # Ensure date is datetime and sorted
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
    df = df.sort_values(by=['Ticker', 'Date'])

    # Log volume
    df['Log_Volume'] = np.log1p(df['Volume'])

    # RSI and MACD
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(close=x).rsi())
    macd = df.groupby('Ticker')['Close'].transform(lambda x: MACD(close=x).macd())
    macd_signal = df.groupby('Ticker')['Close'].transform(lambda x: MACD(close=x).macd_signal())
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal

    # Returns
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)

    return df.dropna()
