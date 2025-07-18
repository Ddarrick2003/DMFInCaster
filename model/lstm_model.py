import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_lstm_forecast(df, forecast_days, currency):
    df = df.copy()
    df = df.sort_values("Date")

    data = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    lookback = 30
    for i in range(lookback, len(scaled_data) - forecast_days):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i:i + forecast_days].flatten())

    X, y = np.array(X), np.array(y)
    X_train, y_train = X[:-1], y[:-1]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(forecast_days))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    last_sequence = scaled_data[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    forecast_scaled = model.predict(last_sequence)[0]
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    st.subheader("📈 LSTM Forecast")
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({f"Forecast ({currency})": forecast}, index=future_dates)
    st.line_chart(forecast_df)

    st.metric("📌 Final Predicted Price", f"{forecast[-1]:,.2f} {currency}")
