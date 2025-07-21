# models/transformer_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_layer(src)
        src = self.transformer(src)
        output = self.output_layer(src)
        return output

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_transformer_forecast(df, forecast_days=10):
    df = df.copy()

    # Use Close price
    if 'Close' not in df.columns:
        raise ValueError("Dataset must contain a 'Close' column")

    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    # Parameters
    seq_len = 20
    forecast_window = forecast_days

    # Prepare data
    X, y = create_sequences(scaled, seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Model
    model = TimeSeriesTransformer(input_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(X_tensor.transpose(0, 1)).transpose(0, 1)[:, -1]
        loss = criterion(out.squeeze(), y_tensor.squeeze())
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    recent_seq = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)

    preds = []
    input_seq = recent_seq.clone()

    for _ in range(forecast_window):
        with torch.no_grad():
            output = model(input_seq.transpose(0, 1))
        next_val = output[-1, 0, 0].item()
        preds.append(next_val)

        # Append new prediction to input sequence
        next_input = torch.tensor([[[next_val]]], dtype=torch.float32).to(device)
        input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

    # Inverse scale
    forecast_scaled = np.array(preds).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled).flatten()

    # Actual values for MAE (last known + future)
    actual = df['Close'].values[-forecast_window:]

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(actual[:len(forecast)], forecast[:len(actual)])

    pred_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_window)
    pred_df = pd.DataFrame({'Date': pred_dates, 'Forecast': forecast})
    actual_df = pd.DataFrame({'Date': df['Date'].iloc[-forecast_window:], 'Actual': actual})

    return pred_df, actual_df, mae
