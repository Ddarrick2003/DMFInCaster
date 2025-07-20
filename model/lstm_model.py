import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def run_lstm_forecast(df, forecast_horizon=10):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    lookback = 20
    for i in range(lookback, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i:i + forecast_horizon, 0])
    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    last_input = scaled_data[-lookback:]
    last_input = last_input.reshape((1, lookback, 1))
    predicted_scaled = model.predict(last_input)[0]
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon)

    return df['Close'], predicted, forecast_dates
