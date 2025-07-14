from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

def generate_full_pdf_report(df, lstm_predictions, actuals, pdf_summary, var):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("FinCaster Asset Report", styles['Heading1']),
        Spacer(1, 12),
        Paragraph(f"Latest VaR Forecast: {var:.2f}%", styles['Normal']),
        Spacer(1, 12),
        Paragraph("LSTM Forecast Accuracy", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Predictions vs Actuals: {len(lstm_predictions)} points", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Insights from Uploaded Report", styles['Normal']),
        Paragraph(pdf_summary or "No summary found", styles['Normal'])
    ]
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_full_pdf_report_all(port, stats, sr, sort, mdd):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("FinCaster Portfolio Summary", styles['Heading1']),
        Spacer(1, 12),
        Paragraph(f"Portfolio Sharpe: {sr:.2f}", styles['Normal']),
        Paragraph(f"Portfolio Sortino: {sort:.2f}", styles['Normal']),
        Paragraph(f"Max Drawdown: {mdd:.2%}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Assets: {', '.join(stats.keys())}", styles['Normal']),
    ]
    doc.build(elements)
    buffer.seek(0)
    return buffer
