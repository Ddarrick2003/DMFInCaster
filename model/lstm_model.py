import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_sequences(df, target_col, lookback=10):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(lookback, len(df_scaled)):
        X.append(df_scaled[i - lookback:i])
        y.append(df_scaled[i, df.columns.get_loc(target_col)])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential(name="FinCaster_LSTM_Model")
    model.add(LSTM(64, input_shape=input_shape, name="lstm_layer"))
    model.add(Dense(1, name="output_layer"))
    model.compile(loss='mse', optimizer='adam')
    return model
