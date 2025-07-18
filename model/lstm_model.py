import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_lstm(df, forecast_days=10):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    SEQ_LEN = 60
    X, y = create_sequences(scaled_data, SEQ_LEN)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    last_seq = torch.tensor(scaled_data[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    forecasts = []

    with torch.no_grad():
        for _ in range(forecast_days):
            pred = model(last_seq)
            forecasts.append(pred.item())
            new_input = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
            last_seq = new_input

    forecast_prices = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': forecast_prices})
    forecast_df['Predicted_Close'] = forecast_df['Predicted_Close'].round(2)

    return forecast_df
