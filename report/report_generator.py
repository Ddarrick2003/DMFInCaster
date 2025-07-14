from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_full_pdf_report(df, lstm_predictions, actuals, pdf_summary, var):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "📄 FinCaster Report Summary")
    c.drawString(50, 730, "Executive Summary:")
    c.drawString(70, 710, pdf_summary[:300] + "..." if pdf_summary else "No summary available.")

    c.drawString(50, 680, "📈 LSTM Forecast:")
    if len(lstm_predictions):
        c.drawString(70, 660, f"Forecasted {len(lstm_predictions)} points. Sample: {round(lstm_predictions[0], 2)}")
    else:
        c.drawString(70, 660, "No forecast generated.")

    c.drawString(50, 630, "📉 GARCH Risk Estimate:")
    c.drawString(70, 610, f"1-Day VaR (95%): {round(var, 2)}%" if var else "Not calculated.")

    c.drawString(50, 580, "📊 Strategy Result:")
    total_return = df['PnL'].cumsum().iloc[-1]
    c.drawString(70, 560, f"Total Strategy Return: {round(total_return * 100, 2)}%")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
