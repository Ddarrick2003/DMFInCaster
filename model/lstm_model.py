# lstm_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
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

def run_lstm(df, forecast_days=10, price_col='Close'):
    df = df.copy()
    scaler = MinMaxScaler()
    df[price_col] = scaler.fit_transform(df[price_col].values.reshape(-1, 1))

    seq_length = 20
    data = df[price_col].values
    X, y = create_sequences(data, seq_length)

    X_train = torch.from_numpy(X).float().unsqueeze(-1).to(device)
    y_train = torch.from_numpy(y).float().unsqueeze(-1).to(device)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    input_seq = torch.tensor(X[-1:]).float().unsqueeze(-1).to(device)
    preds = []

    for _ in range(forecast_days):
        with torch.no_grad():
            pred = model(input_seq)
            preds.append(pred.item())
            input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)

    preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': preds_rescaled})
    return forecast_df
