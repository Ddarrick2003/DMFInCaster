import fitz

def extract_pdf_insights(uploaded_pdf):
    text = ""
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text[:1000]