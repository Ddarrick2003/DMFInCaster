import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from report.report_generator import generate_summary_pdf
from pdf.pdf_parser import extract_pdf_insights
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞💵 FinCaster: Financial Forecasting App")

uploaded_file = st.file_uploader("📤 Upload your OHLCV CSV file", type=["csv"])
uploaded_pdf = st.file_uploader("📄 Upload optional PDF report", type=["pdf"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(str).str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        df = preprocess_data(df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    st.success(f"✅ Data Loaded: {df.shape[0]} rows")

    pdf_summary = ""
    if uploaded_pdf:
        try:
            pdf_summary = extract_pdf_insights(uploaded_pdf)
            st.info(f"📄 Insights from uploaded report:\n{pdf_summary}")
        except Exception as e:
            st.warning(f"⚠️ Could not read PDF report: {e}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 LSTM Forecast", "📉 GARCH Risk", "📊 Strategy + PnL", "📑 Summary + PDF"])

    with tab1:
        st.subheader("LSTM Forecasting")
        features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
        try:
            X, y = create_sequences(df[features], target_col='Close')
            if len(X) == 0:
                st.warning("⚠️ Not enough data.")
            else:
                split = int(len(X) * 0.8)
                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X[:split], y[:split], epochs=10, batch_size=16,
                          validation_data=(X[split:], y[split:]), callbacks=[EarlyStopping(patience=3)], verbose=0)
                preds = model.predict(X[split:]).flatten()
                st.line_chart({"Actual": y[split:], "Predicted": preds})
        except Exception as e:
            st.error(f"LSTM Error: {e}")

    with tab2:
        st.subheader("GARCH Forecast")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)
            st.metric("1-Day VaR (95%)", f"{var_1d:.2f}%")
            st.line_chart(vol_forecast)
        except Exception as e:
            st.error(f"GARCH Error: {e}")

    with tab3:
        st.subheader("Strategy Backtest")
        df['Signal'] = np.where((df['MACD'] > df['MACD_Signal']) & (df['RSI'] < 70), 1, 0)
        df['PnL'] = df['Returns'] * df['Signal']
        st.line_chart(df['PnL'].cumsum())
        st.download_button("📥 Download Signals", df.to_csv(index=False), "signals.csv")

    with tab4:
        st.subheader("Summary + PDF")
        if st.button("📄 Generate Summary PDF"):
            summary_text = generate_summary_pdf(df, pdf_summary)
            st.success("✅ Summary generated.")
            st.download_button("📥 Download Summary", summary_text, file_name="FinCaster_Summary.txt")

else:
    st.info("📥 Please upload data to begin.")
