import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from report.report_generator import generate_full_pdf_report
from pdf.pdf_parser import extract_pdf_insights
from utils.sentiment import generate_mock_sentiment
from tensorflow.keras.callbacks import EarlyStopping
import io

# -------------------- App Config --------------------
st.set_page_config(page_title="FinCaster", layout="wide")
st.title("🌞💵 FinCaster: Financial Forecasting App")

# -------------------- Uploads --------------------
uploaded_file = st.file_uploader("📤 Upload OHLCV CSV", type=["csv"])
uploaded_pdf = st.file_uploader("📄 Upload PDF Report (optional)", type=["pdf"])
use_sentiment = st.sidebar.checkbox("🧠 Overlay Sentiment", value=False)

# -------------------- Main Workflow --------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = preprocess_data(df)
        if use_sentiment:
            df = generate_mock_sentiment(df)
            st.success("✅ Sentiment overlay added.")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    st.success(f"✅ Data Loaded: {df.shape[0]} rows")

    # -------------------- PDF Insight --------------------
    pdf_summary = ""
    if uploaded_pdf:
        try:
            pdf_summary = extract_pdf_insights(uploaded_pdf)
            st.info("📄 Report Insights:\n\n" + pdf_summary)
        except Exception as e:
            st.warning(f"⚠️ Could not parse PDF: {e}")

    # -------------------- TABS --------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 LSTM Forecast", "📉 GARCH Risk",
        "📊 Strategy + PnL", "📑 Summary + PDF"
    ])

    # -------------------- TAB 1: LSTM --------------------
    with tab1:
        st.subheader("Multivariate LSTM Forecast")
        features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
        try:
            X, y = create_sequences(df[features], target_col='Close')
            if len(X) == 0:
                st.warning("⚠️ Not enough data for LSTM")
            else:
                split = int(len(X) * 0.8)
                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X[:split], y[:split], epochs=10, batch_size=16,
                          validation_data=(X[split:], y[split:]),
                          callbacks=[EarlyStopping(patience=3)], verbose=0)
                preds = model.predict(X[split:]).flatten()

                # Plotly interactive chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y[split:], name="Actual"))
                fig.add_trace(go.Scatter(y=preds, name="Predicted"))

                # Sentiment overlay
                if use_sentiment and 'Sentiment' in df.columns:
                    sentiment_sub = df.iloc[-len(y[split:]):]['Sentiment']
                    fig.add_trace(go.Scatter(
                        y=sentiment_sub,
                        name="Sentiment",
                        yaxis='y2',
                        line=dict(dash='dot', color='gray'),
                        opacity=0.5
                    ))
                    fig.update_layout(
                        yaxis2=dict(overlaying='y', side='right', title='Sentiment')
                    )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"LSTM Error: {e}")

    # -------------------- TAB 2: GARCH --------------------
    with tab2:
        st.subheader("GARCH Volatility Forecast")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)
            st.metric("1-Day VaR (95%)", f"{var_1d:.2f}%")
            st.line_chart(vol_forecast)
        except Exception as e:
            st.error(f"GARCH Error: {e}")

    # -------------------- TAB 3: Strategy --------------------
    with tab3:
        st.subheader("Strategy + Backtest")
        df['Signal'] = np.where(
            (df['MACD'] > df['MACD_Signal']) & (df['RSI'] < 70), 1, 0
        )
        if use_sentiment and 'Sentiment' in df.columns:
            df['Signal'] = df['Signal'] * (df['Sentiment'] > 0).astype(int)

        df['PnL'] = df['Returns'] * df['Signal'].shift(1)
        st.line_chart(df['PnL'].cumsum())
        st.download_button("📥 Download Signals", df.to_csv(index=False), "signals.csv")

        if 'Sentiment' in df.columns:
            st.subheader("💬 Sentiment Over Time")
            st.line_chart(df['Sentiment'])

    # -------------------- TAB 4: PDF Export --------------------
    with tab4:
        st.subheader("📄 Generate Full PDF Report")
        if st.button("📝 Export Analysis to PDF"):
            pdf_buffer = generate_full_pdf_report(
                df=df,
                lstm_predictions=preds if 'preds' in locals() else [],
                actuals=y[split:] if 'y' in locals() else [],
                pdf_summary=pdf_summary,
                var=var_1d if 'var_1d' in locals() else None
            )
            st.success("✅ PDF generated.")
            st.download_button(
                "📥 Download PDF",
                data=pdf_buffer,
                file_name="FinCaster_Report.pdf",
                mime="application/pdf"
            )

else:
    st.info("📥 Please upload your OHLCV data to begin.")
