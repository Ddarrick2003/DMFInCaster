import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import timedelta

def run_lstm(df, forecast_days, currency):
    required_cols = ['Date', 'Close']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: '{col}' in uploaded data.")
            st.stop()

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    price_column = 'Close'
    df[price_column] = df[price_column].fillna(method='ffill')

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[price_column]])

    # Prepare sequences
    sequence_length = 60
    x_train, y_train = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Forecasting
    input_sequence = scaled_data[-sequence_length:]
    forecast = []

    for _ in range(forecast_days):
        input_seq = input_sequence.reshape(1, sequence_length, 1)
        predicted_price = model.predict(input_seq, verbose=0)
        forecast.append(predicted_price[0][0])
        input_sequence = np.append(input_sequence[1:], predicted_price, axis=0)

    forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_column], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices.flatten(), mode='lines+markers',
                             name='Forecast', line=dict(color='green')))
    fig.update_layout(title=f'{forecast_days}-Day Price Forecast using LSTM',
                      xaxis_title='Date', yaxis_title=f'Price ({currency})')

    st.plotly_chart(fig)

    # Display next day forecast
    st.success(f"📈 Next predicted price: **{forecast_prices[0][0]:,.2f} {currency}**")
