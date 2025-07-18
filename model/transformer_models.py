# In transformer_models.py

def run_autoformer(df, forecast_days, currency):
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    class SimpleAutoformer(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=64, output_dim=5):
            super(SimpleAutoformer, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            enc_out, _ = self.encoder(x)
            out = self.decoder(enc_out[:, -1, :])
            return out

    # Preprocess data
    df = df.copy()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    sequence_length = 30
    X, y = [], []
    for i in range(len(scaled) - sequence_length - forecast_days):
        X.append(scaled[i:i+sequence_length])
        y.append(scaled[i+sequence_length:i+sequence_length+forecast_days])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    model = SimpleAutoformer(input_dim=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop (quick & small epochs for Streamlit)
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.unsqueeze(1), y[:, 0])
        loss.backward()
        optimizer.step()

    # Forecast next N days
    model.eval()
    recent_seq = torch.tensor(scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(recent_seq).numpy()

    # Inverse scale and build forecast DataFrame
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=forecast_days + 1, freq='D')[1:]
    forecast = scaler.inverse_transform(np.tile(pred, (forecast_days, 1)))
    forecast_df = pd.DataFrame(forecast, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=future_dates)

    st.subheader("🔁 Autoformer Forecast Output")
    st.dataframe(forecast_df)

    st.line_chart(forecast_df[['Close']].rename(columns={'Close': f'Predicted Close ({currency})'}))
