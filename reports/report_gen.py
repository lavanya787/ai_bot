from fpdf import FPDF

def generate_pdf_report(username, summary_text, predictions=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"{username}'s Analysis Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, summary_text)

    if predictions is not None:
        pdf.ln()
        pdf.cell(0, 10, "Prediction Summary", ln=True)
        for k, v in predictions.items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)

    path = f"reports/{username}_report.pdf"
    pdf.output(path)
    return path
