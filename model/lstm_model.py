# model/lstm_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def run_lstm(df, forecast_days, currency='KSh'):
    st.subheader("📈 LSTM Forecast")

    price_column = 'Close'
    if price_column not in df.columns:
        st.error(f"'{price_column}' column not found in the dataset.")
        return

    df = df.copy()
    df[price_column] = df[price_column].fillna(method='ffill')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[price_column]])

    # Create training data
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    look_back = 60
    X, Y = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=10, batch_size=32, verbose=0)

    # Forecast
    inputs = scaled_data[-look_back:].reshape(1, look_back, 1)
    predictions = []
    for _ in range(forecast_days):
        next_pred = model.predict(inputs, verbose=0)
        predictions.append(next_pred[0, 0])
        inputs = np.append(inputs[:, 1:, :], [[next_pred]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Date range
    last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date']).iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_column], name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_prices, name="Forecast", line=dict(color='green')))
    fig.update_layout(title="LSTM Forecast", xaxis_title="Date", yaxis_title=f"Price ({currency})")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ LSTM predicted next {forecast_days} day(s) in {currency}")
