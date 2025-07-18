import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))
        return self.linear(output[:, -1, :])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_lstm(df, forecast_days=10, price_column='Close'):
    df = df.copy()
    df[price_column] = df[price_column].fillna(method='ffill')
    values = df[price_column].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)

    seq_length = 20
    X, y = create_sequences(scaled_data, seq_length)
    X = torch.from_numpy(X).float().unsqueeze(-1)
    y = torch.from_numpy(y).float()

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()

    # Forecasting
    model.eval()
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    forecast = []
    current_seq = torch.from_numpy(last_seq).float()

    for _ in range(forecast_days):
        with torch.no_grad():
            next_val = model(current_seq)
        forecast.append(next_val.item())
        current_seq = torch.cat((current_seq[:, 1:, :], next_val.view(1, 1, 1)), dim=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_days)]

    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    return forecast_df
