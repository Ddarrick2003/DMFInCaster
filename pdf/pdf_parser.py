import fitz  # PyMuPDF

def extract_pdf_insights(uploaded_pdf):
    if uploaded_pdf is None:
        return ""
    try:
        with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text[:1000]  # limit for display
    except Exception as e:
        return f"Error extracting PDF text: {e}"
