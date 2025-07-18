import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def run_lstm(df, forecast_days, currency):
    data = df[["Date", "Close"]].copy()
    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
    data.dropna(inplace=True)
    data.set_index("Date", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_sequence = scaled_data[-sequence_length:]
    forecast_scaled = []

    for _ in range(forecast_days):
        input_seq = last_sequence.reshape(1, sequence_length, 1)
        pred = model.predict(input_seq, verbose=0)
        forecast_scaled.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred[0][0]).reshape(sequence_length)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecast"], name="Forecast", line=dict(color="green")))
    fig.update_layout(title=f"{forecast_days}-Day LSTM Forecast", xaxis_title="Date", yaxis_title=f"Price ({currency})")
    fig.update_traces(line=dict(width=2))
    return fig.show()
