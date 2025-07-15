with tab1:
    st.subheader("🔮 Multivariate LSTM Forecast")
    selected_asset = st.selectbox("📌 Select Asset", assets)
    df_asset = df[df['Ticker'] == selected_asset].copy()

    features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
    try:
        X, y = create_sequences(df_asset[features], target_col='Close')

        st.write("📊 LSTM Input Shape:", X.shape)  # Debug: Check shape

        if len(X) == 0 or len(y) == 0:
            st.warning("⚠️ Not enough data to train LSTM. Please upload more historical rows.")
        else:
            split = int(len(X) * 0.8)
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            model.fit(
                X[:split], y[:split],
                epochs=10, batch_size=16,
                validation_data=(X[split:], y[split:]),
                callbacks=[EarlyStopping(patience=3)],
                verbose=0
            )
            preds = model.predict(X[split:]).flatten()
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y[split:], name="Actual"))
            fig.add_trace(go.Scatter(y=preds, name="Predicted"))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ LSTM error: {e}")
