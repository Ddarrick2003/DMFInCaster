def generate_summary_pdf(df, pdf_summary=""):
    summary = []

    summary.append("📊 FinCaster Forecast Report")
    summary.append("Generated automatically from uploaded data.\n")
    summary.append("Total Forecast Days: {}".format(len(df)))
    summary.append("Total Assets Analyzed: {}".format(df['Ticker'].nunique()))

    total_signals = df['Signal'].sum() if 'Signal' in df else 0
    summary.append(f"Total Buy Signals: {int(total_signals)}")

    if 'PnL' in df.columns:
        pnl_cumsum = df['PnL'].cumsum().iloc[-1]
        summary.append(f"Total Strategy PnL: {pnl_cumsum:.2f}")

    if pdf_summary:
        summary.append("\n📄 External PDF Summary Extract:")
        summary.append(pdf_summary)

    return "\n".join(summary)
