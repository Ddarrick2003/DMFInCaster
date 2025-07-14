def generate_summary_pdf(df, report_text):
    summary = f'''
📘 FinCaster Summary Report 📘

Data points: {len(df)}
Forecast window: {len(df) - 10}

🧠 Model Insights:
- LSTM predicted prices show clear directional patterns.
- GARCH forecast indicates market volatility patterns.

📊 Strategy Summary:
- Net Strategy PnL: {df['PnL'].sum():.2f}
- Avg Return: {df['Returns'].mean():.4f}

📄 Report Notes:
{report_text if report_text else "No PDF report uploaded."}

'''
    return summary