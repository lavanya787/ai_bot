from fpdf import FPDF
from io import BytesIO

def export_chat_to_pdf(title, messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title(title)

    pdf.cell(200, 10, txt=f"Chat: {title}", ln=True, align="L")
    pdf.ln(5)

    for msg in messages:
        role = "User:" if msg["role"] == "user" else "Assistant:"
        pdf.multi_cell(0, 10, f"{role} {msg['content']}", border=0)
        pdf.ln(1)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output
