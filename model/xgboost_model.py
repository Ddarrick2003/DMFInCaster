def run_xgboost_with_shap(df, forecast_days, currency):
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    st.subheader("📈 XGBoost Forecast with SHAP")
    st.line_chart(pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    }, index=y_test.index))

    # SHAP summary
    st.subheader("🔍 SHAP Feature Importance")
    shap_fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(shap_fig)

    # Generate forecast
    future_input = X.iloc[-forecast_days:]
    future_preds = model.predict(future_input)

    # Ensure date index for predictions
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'Forecasted Price': future_preds
    }, index=future_dates)

    # Currency formatting
    symbol = "KSh" if currency == "KES" else "$"
    st.subheader(f"🔮 {forecast_days}-Day Forecast in {symbol}")
    st.line_chart(forecast_df)

    # Display numeric forecast
    st.dataframe(forecast_df.applymap(lambda x: f"{symbol}{x:,.2f}"))
