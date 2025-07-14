import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    # Ensure 'Ticker' column exists
    if 'Ticker' not in df.columns:
        df['Ticker'] = 'ASSET'

    # Ensure 'Volume' exists and is numeric
    if 'Volume' not in df.columns:
        df['Volume'] = 1  # Assign dummy value
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(1)

    df['Log_Volume'] = np.log1p(df['Volume'])

    # Calculate Returns
    df['Returns'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)

    # RSI
    df['RSI'] = df.groupby('Ticker')['Close'].transform(
        lambda x: RSIIndicator(x, window=14).rsi().fillna(50)
    )

    # MACD
    df['MACD'] = df.groupby('Ticker')['Close'].transform(
        lambda x: MACD(x).macd().fillna(0)
    )
    df['MACD_Signal'] = df.groupby('Ticker')['Close'].transform(
        lambda x: MACD(x).macd_signal().fillna(0)
    )

    return df
