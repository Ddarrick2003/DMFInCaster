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
    from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import uuid

def build_lstm_model(input_shape):
    model = Sequential()
    unique_suffix = str(uuid.uuid4())[:8]  # e.g., '3f2c5a1b'
    model.add(LSTM(64, input_shape=input_shape, name=f"lstm_{unique_suffix}"))
    model.add(Dense(1, name=f"dense_{unique_suffix}"))
    model.compile(optimizer='adam', loss='mse')
    return model





