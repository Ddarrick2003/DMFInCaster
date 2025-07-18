# transformer_models.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class Informer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Informer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x[:, -1, :])
        return x


class Autoformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Autoformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x[:, -1, :])
        return x


def run_transformer(df, model_type="informer", forecast_horizon=10):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    scaler = MinMaxScaler()
    price_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    seq_length = 20
    X, y = create_sequences(price_data, seq_length)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    X_tensor = X_tensor.reshape(X_tensor.shape[0], seq_length, 1)

    model = Informer(input_dim=1) if model_type == "informer" else Autoformer(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    last_seq = X_tensor[-1].unsqueeze(0)
    preds = []
    for _ in range(forecast_horizon):
        with torch.no_grad():
            pred = model(last_seq)
        preds.append(pred.item())
        last_seq = torch.cat([last_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_horizon+1, freq='D')[1:]

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})
    return forecast_df


def run_informer(df, forecast_horizon=10):
    return run_transformer(df, model_type="informer", forecast_horizon=forecast_horizon)


def run_autoformer(df, forecast_horizon=10):
    return run_transformer(df, model_type="autoformer", forecast_horizon=forecast_horizon)
