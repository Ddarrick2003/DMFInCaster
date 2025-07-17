import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# -------- PAGE CONFIGURATION --------
st.set_page_config(
    page_title="FinCaster | Financial Forecasting Tool",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------- SESSION STATE INIT --------
if "currency" not in st.session_state:
    st.session_state.currency = "KSh"

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "data" not in st.session_state:
    st.session_state.data = None

if "task_name" not in st.session_state:
    st.session_state.task_name = ""

# --------- STYLING ---------
light_style = """
<style>
/* Hide sidebar by default */
.css-18ni7ap.e8zbici2 { display: none; }
section[data-testid="stSidebar"] { background-color: #f8f9fa; }

/* Main Title and Dashboard Design */
h1 { font-family: 'Segoe UI', sans-serif; font-weight: 700; color: #00704A; }
.stButton > button { background-color: #00704A; color: white; border-radius: 6px; }
</style>
"""

dark_style = """
<style>
h1 { font-family: 'Segoe UI', sans-serif; font-weight: 700; color: #00FFCC; }
.stButton > button { background-color: #00FFCC; color: #222; border-radius: 6px; }
</style>
"""

st.markdown(light_style if st.session_state.theme == "Light" else dark_style, unsafe_allow_html=True)

# --------- HEADER & SETTINGS ---------
with st.container():
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown("### 💹")
    with col2:
        st.title("FinCaster | Your Smart Financial Forecasting Tool")

    with st.expander("⚙️ Settings", expanded=True):
        st.session_state.task_name = st.text_input("Task Name", st.session_state.task_name, key="task_name_input")
        st.session_state.theme = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, key="theme_toggle")
        st.session_state.currency = st.radio("Currency", ["KSh", "USD"], index=0 if st.session_state.currency == "KSh" else 1, key="currency_toggle")

st.divider()

# --------- DATA UPLOAD ---------
with st.expander("📁 Upload Your Data", expanded=False):
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="data_uploader")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values(by='Date', inplace=True)
                df.reset_index(drop=True, inplace=True)
            st.session_state.data = df
            st.success(f"✅ {len(df)} records loaded for task: {st.session_state.task_name}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.data = None

# --------- SIDEBAR NAVIGATION ---------
with st.sidebar:
    st.header("📌 FinCaster Modules")
    tabs = {
        "LSTM Forecast": st.checkbox("🔮 LSTM Forecast", key="tab_lstm"),
        "GARCH Risk": st.checkbox("📊 GARCH Risk", key="tab_garch"),
        "Backtesting": st.checkbox("📈 Backtesting", key="tab_backtest"),
        "Report": st.checkbox("📄 Generate Report", key="tab_report")
    }

# --------- MODULES ---------
def format_price(value):
    currency = st.session_state.currency
    if pd.isna(value):
        return f"{currency} NaN"
    return f"{currency} {value:,.2f}"

if st.session_state.data is not None:
    df = st.session_state.data
    # --- Only use relevant columns ---
    if not {'Close'}.issubset(df.columns):
        st.warning("⚠️ 'Close' column is required for forecasting modules.")
    else:
        close_prices = df['Close'].values

        # ----- LSTM FORECASTING -----
        if tabs["LSTM Forecast"]:
            st.subheader("🔮 LSTM Price Forecast")
            try:
                # Dummy logic for now, replace with your model
                forecast = [close_prices[-1] + i*5 for i in range(1, 6)]
                next_day = forecast[0]
                st.metric("📌 Next Day Forecasted Price", format_price(next_day))
                st.write("📆 5-Day Forecast:")
                st.json([format_price(val) for val in forecast])
            except Exception as e:
                st.error(f"❌ LSTM Error: {e}")

        # ----- GARCH RISK ESTIMATION -----
        if tabs["GARCH Risk"]:
            st.subheader("📊 GARCH Volatility & VaR")
            try:
                returns = np.diff(close_prices) / close_prices[:-1]
                if np.var(returns) == 0:
                    raise ValueError("GARCH Error: -1 (Zero variance in returns)")
                # Dummy volatility
                volatility = np.std(returns) * 100
                var = -np.percentile(returns, 5) * close_prices[-1]
                st.metric("⚠️ Forecasted Volatility", f"{volatility:.2f}%")
                st.metric("📉 Value-at-Risk (95%)", format_price(var))
            except Exception as e:
                st.warning(f"⚠️ GARCH Error: {e}")

        # ----- BACKTESTING -----
        if tabs["Backtesting"]:
            st.subheader("📈 Backtesting Results")
            try:
                returns = np.diff(close_prices) / close_prices[:-1]
                sharpe = np.mean(returns) / np.std(returns)
                sortino = np.mean(returns) / np.std(returns[returns < 0]) if any(returns < 0) else 0
                max_dd = np.max(np.maximum.accumulate(close_prices) - close_prices) / np.max(close_prices)
                st.metric("📈 Sharpe Ratio", round(sharpe, 2))
                st.metric("📉 Sortino Ratio", round(sortino, 2))
                st.metric("📉 Max Drawdown", f"{max_dd*100:.2f}%")
            except Exception as e:
                st.error(f"❌ Backtest Error: {e}")

        # ----- REPORT GENERATION -----
        if tabs["Report"]:
            st.subheader("📄 Generate Report")
            st.write("🚧 Report generator in progress...")

else:
    st.info("📥 Please upload a dataset via the Upload section above.")

