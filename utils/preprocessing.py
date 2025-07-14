import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    # Ensure required columns exist
    if 'Ticker' not in df.columns:
        df['Ticker'] = 'ASSET'

    if 'Volume' not in df.columns:
        df['Volume'] = 1

    if 'Close' not in df.columns:
        raise ValueError("❌ 'Close' column is required in your dataset.")

    # Coerce to numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Drop rows with missing Close values (important for pct_change)
    df = df.dropna(subset=['Close']).copy()
    df['Volume'] = df['Volume'].fillna(1)

    df['Log_Volume'] = np.log1p(df['Volume'])

    # Calculate Returns safely
    df['Returns'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().fillna(0))

    # RSI (relative strength index)
    df['RSI'] = df.groupby('Ticker')['Close'].transform(
        lambda x: RSIIndicator(x, window=14).rsi().fillna(50)
    )

    # MACD (trend momentum)
    df['MACD'] = df.groupby('Ticker')['Close'].transform(
        lambda x: MACD(x).macd().fillna(0)
    )
    df['MACD_Signal'] = df.groupby('Ticker')['Close'].transform(
        lambda x: MACD(x).macd_signal().fillna(0)
    )

    return df
