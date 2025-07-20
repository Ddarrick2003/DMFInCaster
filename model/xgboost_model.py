import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from datetime import timedelta

def run_xgboost_forecast(df, forecast_horizon=10):
    df = df.copy()
    
    # Ensure datetime index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Use 'Close' price for prediction
    df = df[['Close']].dropna()

    # Create features: lag values
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    df.dropna(inplace=True)

    # Features and target
    X = df.drop('Close', axis=1)
    y = df['Close']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=forecast_horizon)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train_scaled, y_train)

    # Predict future values
    predicted = model.predict(X_test_scaled)

    # Get the last actual price date
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=y_test.index[0], periods=forecast_horizon, freq='D')

    return y, predicted, forecast_dates
