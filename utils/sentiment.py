import numpy as np

def generate_mock_sentiment(df):
    np.random.seed(42)
    df['Sentiment'] = np.random.uniform(-1, 1, size=len(df))
    return df
