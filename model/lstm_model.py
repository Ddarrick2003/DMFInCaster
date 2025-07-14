import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def create_sequences(data, target_col='Close', window_size=30):
    """
    Create sequences of features and target for LSTM.
    """
    # Drop rows with NaNs to avoid Tensor conversion errors
    data = data.dropna().copy()

    # Convert all values to float32 for TensorFlow compatibility
    values = data.astype(np.float32).values

    target_idx = data.columns.get_loc(target_col)

    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size][target_idx])

    return np.array(X), np.array(y)



def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
