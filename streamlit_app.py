import streamlit as st
import pandas as pd
from model.transformer_models import run_autoformer, run_informer
from model.xgboost_model import run_xgboost_forecast
from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast

# App settings
st.set_page_config(page_title="FinCaster", layout="wide", page_icon="💹")

# Styling
def set_theme():
    dark = st.session_state.get('theme_dark', False)
    if dark:
        st.markdown(
            """<style>
            body {background-color: #121212; color: white;}
            .stApp {background-color: #121212;}
            </style>""",
            unsafe_allow_html=True
        )

# Sidebar toggles
with st.sidebar:
    st.title("⚙️ FinCaster Settings")
    st.session_state['currency'] = st.radio("Currency", ['KSh', 'USD'], index=0)
    st.session_state['theme_dark'] = st.toggle("🌙 Dark Mode", value=False)
    model_selected = st.selectbox("📊 Select Model", ['Informer', 'Autoformer', 'XGBoost', 'LSTM', 'GARCH'])
    forecast_days = st.slider("📆 Forecast Days", 1, 30, 10)

set_theme()

# Main area
st.title("📈 FinCaster - Forecasting Dashboard")

uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["Date"])
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique().tolist()
        selected_ticker = st.selectbox("Choose Ticker", tickers)
        df = df[df['Ticker'] == selected_ticker]

    df = df.sort_values("Date")

    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    if st.button("Run Forecast 🚀"):
        st.success(f"Running {model_selected} forecast for {forecast_days} days...")

        try:
            if model_selected == "Informer":
                run_informer(df, forecast_days, st.session_state['currency'])
            elif model_selected == "Autoformer":
                run_autoformer(df, forecast_days, st.session_state['currency'])
            elif model_selected == "XGBoost":
                run_xgboost_forecast(df, forecast_days, st.session_state['currency'])
            elif model_selected == "LSTM":
                run_lstm_forecast(df, forecast_days, st.session_state['currency'])
            elif model_selected == "GARCH":
                run_garch_forecast(df, forecast_days, st.session_state['currency'])
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    st.info("📂 Upload a CSV file to begin.")

st.markdown("---")
st.caption("© 2025 FinCaster by Mboya Darrick")
