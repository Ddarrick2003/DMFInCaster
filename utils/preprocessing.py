import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    # Ensure proper types
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # Sort before feature generation
    df = df.sort_values(['Ticker', 'Date'])

    # Add returns
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)

    # Log volume
    df['Log_Volume'] = np.log1p(df['Volume'])

    # RSI and MACD
    df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(x, window=14).rsi())
    df['MACD'] = df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd())
    df['MACD_Signal'] = df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd_signal())

    # Final cleanup
    df = df.dropna(subset=['RSI', 'MACD', 'MACD_Signal'])

    return df
