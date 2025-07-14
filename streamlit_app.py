import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from scipy.stats import sem
from scipy import stats
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from utils.sentiment import generate_mock_sentiment
from report.report_generator import (
    generate_full_pdf_report,
    generate_full_pdf_report_all
)
from pdf.pdf_parser import extract_pdf_insights
from tensorflow.keras.callbacks import EarlyStopping
import io

# ------------------ Helper Metrics ------------------
def sharpe_ratio(returns, rf=0.0):
    excess = returns.mean() - rf/252
    return np.sqrt(252) * excess / returns.std() if returns.std()>0 else np.nan

def sortino_ratio(returns, rf=0.0):
    downside = returns[returns<0].std() * np.sqrt(252)
    excess = returns.mean() - rf/252
    return excess / downside if downside>0 else np.nan

def max_drawdown(cum):
    peak = cum.cummax()
    return ((cum - peak)/peak).min()

# ------------------ App Setup ------------------
st.set_page_config(page_title="FinCaster", layout="wide")
st.title("FinCaster:Forecasting & Portfolio Manager")

# ------------------ Sidebar Inputs ------------------
use_live = st.sidebar.checkbox("Use Live Market Data (yfinance)")
tickers = st.sidebar.text_input("Tickers (comma separated)", "AAPL, MSFT, GOOGL")
uploaded = st.file_uploader("OR upload merged CSV with 'Ticker' column", type=["csv"])
use_sentiment = st.sidebar.checkbox("Enable Sentiment Overlay")
auth_toggle = st.sidebar.checkbox("Enable Dummy Authentication (future)")

# ---- Authentication Placeholder ----
if auth_toggle:
    st.sidebar.info("Authentication not yet implemented (Tier 4 placeholder)")

# ------------------ Data Load ------------------
import yfinance as yf

if use_live:
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        st.warning("Please enter at least one ticker.")
        st.stop()

    try:
        data = yf.download(symbols, period="6mo")['Adj Close']
        df = data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Close')
        df['Volume'] = df['Open'] = df['High'] = df['Low'] = df['Close']  # placeholders for structure
        st.success("Live data loaded successfully.")
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        st.stop()

elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Ticker' not in df.columns:
            st.error("Uploaded file must contain a 'Ticker' column.")
            st.stop()
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to load uploaded CSV: {e}")
        st.stop()
else:
    st.stop("Please upload merged CSV or enable live mode to continue.")

# Standardize Date format
df['Date'] = pd.to_datetime(df['Date'])

# ------------------ Feature Engineering ------------------
def compute_indicators(group):
    group = group.sort_values('Date')
    group['Returns'] = group['Close'].pct_change()
    group['Log_Volume'] = np.log(group['Volume'].replace(0, np.nan)).fillna(0)

    delta = group['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    group['RSI'] = 100 - (100 / (1 + rs))

    exp1 = group['Close'].ewm(span=12, adjust=False).mean()
    exp2 = group['Close'].ewm(span=26, adjust=False).mean()
    group['MACD'] = exp1 - exp2
    group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
    return group

df = df.groupby('Ticker', group_keys=False).apply(compute_indicators)
df.dropna(subset=['Returns', 'RSI', 'MACD', 'MACD_Signal'], inplace=True)

# ------------------ Compute Per-Asset Stats ------------------
all_stats = {}
for asset in assets:
    sub = df[df['Ticker']==asset].sort_values('Date').copy()
    sub = preprocess_data(sub)
    if use_sentiment:
        sub = generate_mock_sentiment(sub)
    sub['Returns'] = sub['Close'].pct_change().fillna(0)

    vol_forecast, var_1d = forecast_garch_var(sub)
    X, y = create_sequences(sub[['Open','High','Low','Close','Log_Volume','RSI','MACD','Returns']], 'Close')
    preds = None
    if len(X) > 60:
        model = build_lstm_model((X.shape[1],X.shape[2]))
        model.fit(X[:-60], y[:-60], epochs=5, batch_size=16, verbose=0)
        preds = model.predict(X[-60:]).flatten()

    cum = sub['Returns'].cumsum()
    sr = sharpe_ratio(sub['Returns'])
    sor = sortino_ratio(sub['Returns'])
    mdd = max_drawdown(cum)

    all_stats[asset] = {
        'df': sub,
        'vars': (vol_forecast, var_1d),
        'lstm': (preds, y[-len(preds):] if preds is not None else None),
        'sr': sr,
        'sor': sor,
        'mdd': mdd
    }

# ------------------ Portfolio Aggregates ------------------
port = pd.concat({a: s['df'].set_index('Date')['Returns'] for a,s in all_stats.items()}, axis=1)
port['Portfolio'] = port.mean(axis=1)
cum_pf = port['Portfolio'].cumsum()
pf_sr = sharpe_ratio(port['Portfolio'])
pf_sor = sortino_ratio(port['Portfolio'])
pf_mdd = max_drawdown(cum_pf)

# ------------------ UI Tabs ------------------
tabs = st.tabs([
    "📈 LSTM Forecast", "📉 GARCH Risk", 
    "📊 Strategy & Signals", "📉 Portfolio Risk", 
    "📄 PDF Reports"
])

# ---- Tab 1: LSTM Forecast ----
with tabs[0]:
    st.header("Multivariate LSTM Forecast per Asset")
    for asset, info in all_stats.items():
        st.subheader(asset)
        sub = info['df']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub['Date'], y=sub['Close'], name='Actual'))
        if info['lstm'][0] is not None:
            last60 = sub['Date'].iloc[-len(info['lstm'][0]):]
            fig.add_trace(go.Scatter(x=last60, y=info['lstm'][0], name='Forecast'))
        if use_sentiment and 'Sentiment' in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub['Date'], y=sub['Sentiment'], name='Sentiment',
                yaxis='y2', line=dict(dash='dot', color='gray'), opacity=0.5))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: GARCH Risk ----
with tabs[1]:
    st.header("GARCH Volatility Forecast")
    for asset, info in all_stats.items():
        st.subheader(asset)
        vol, var = info['vars']
        st.metric(f"{asset} - 1d VaR(95%)", f"{var:.2f}%")
        st.line_chart(vol)

# ---- Tab 3: Strategy & Signals ----
with tabs[2]:
    st.header("Strategy Builder & Signals")
    st.info("Using simple MACD>Signal AND RSI<70 (with optional sentiment)")
    for asset, info in all_stats.items():
        st.subheader(asset)
        sub = info['df']
        sub['Signal'] = np.where(
            (sub['MACD'] > sub['MACD_Signal']) & (sub['RSI'] < 70) & 
            ((~use_sentiment) | (sub['Sentiment'] > 0)), 1, 0)
        sub['PnL'] = sub['Returns'] * sub['Signal'].shift(1)
        sub['CumPnL'] = sub['PnL'].cumsum()
        st.line_chart(pd.DataFrame({'CumPnL': sub['CumPnL']}))
        st.download_button(f"Download {asset} Signals", sub[['Date','Signal','PnL']].to_csv(index=False), f"{asset}_signals.csv")

# ---- Tab 4: Portfolio Risk Metrics ----
with tabs[3]:
    st.header("📉 Portfolio Analytics")
    st.metric("Sharpe", round(pf_sr,2))
    st.metric("Sortino", round(pf_sor,2))
    st.metric("Max Drawdown", f"{pf_mdd:.2%}")
    st.line_chart(cum_pf)

# ---- Tab 5: PDF & Report Generation ----
with tabs[4]:
    st.header("Generate PDF Report")
    if st.button("📝 Export Single Asset PDF"):
        buf = generate_full_pdf_report(
            df=all_stats[assets[0]]['df'],
            lstm_predictions=all_stats[assets[0]]['lstm'][0] or [],
            actuals=all_stats[assets[0]]['lstm'][1] or [],
            pdf_summary=extract_pdf_insights(uploaded) if uploaded else "",
            var=all_stats[assets[0]]['vars'][1]
        )
        st.download_button("Download PDF", buf, f"{assets[0]}_report.pdf","application/pdf")

    if st.button("🗂 Export Portfolio PDF"):
        buf = generate_full_pdf_report_all(port, all_stats, pf_sr, pf_sor, pf_mdd)
        st.download_button("Download Portfolio PDF", buf, "portfolio_report.pdf", "application/pdf")
