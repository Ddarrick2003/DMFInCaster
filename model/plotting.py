import plotly.graph_objects as go
import pandas as pd

# Currency Conversion Helper
def convert_currency(prices, currency="KSh"):
    rate = 142 if currency == "KSh" else 1
    return prices * rate

# Main Forecast vs Actual Plot
def plot_comparison(pred_df, actual_df, title, currency='KSh', color_scheme='light'):
    colors = get_theme_colors(color_scheme)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_df['Date'],
        y=convert_currency(actual_df['Actual'], currency),
        mode='lines+markers',
        name='Actual',
        line=dict(color=colors['actual'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=convert_currency(pred_df['Forecast'], currency),
        mode='lines+markers',
        name='Forecast',
        line=dict(color=colors['forecast'], width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        template=colors['template'],
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font=dict(color=colors['text']),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
        legend=dict(bgcolor=colors['bg'], bordercolor=colors['text'], borderwidth=1)
    )
    return fig

# Confidence Interval Plot (for Transformer models)
def plot_with_confidence(pred_df, actual_df, title, currency='KSh', color_scheme='light'):
    colors = get_theme_colors(color_scheme)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_df['Date'],
        y=convert_currency(actual_df['Actual'], currency),
        mode='lines',
        name='Actual',
        line=dict(color=colors['actual'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=convert_currency(pred_df['Forecast'], currency),
        mode='lines',
        name='Forecast',
        line=dict(color=colors['forecast'], width=2, dash='dot')
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pred_df['Date'].tolist() + pred_df['Date'][::-1].tolist(),
        y=(convert_currency(pred_df['Upper'], currency).tolist() + 
           convert_currency(pred_df['Lower'], currency)[::-1].tolist()),
        fill='toself',
        fillcolor=colors['conf_fill'],
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        template=colors['template'],
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font=dict(color=colors['text']),
        margin=dict(l=30, r=30, t=40, b=30),
        height=460,
        legend=dict(bgcolor=colors['bg'], bordercolor=colors['text'], borderwidth=1)
    )
    return fig
