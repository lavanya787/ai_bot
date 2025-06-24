from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io
import re

def generate_chat_pdf(title, messages):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=1*inch, rightMargin=1*inch, topMargin=1*inch, bottomMargin=1*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"<b>{title}</b>", styles['Title']), Spacer(1, 12)]

    for msg in messages:
        role = "You" if msg["role"] == "user" else "AI"
        content = msg["content"]
        story.append(Paragraph(f"<b>{role}:</b> {content}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

def sanitize_filename(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"\s+", "_", title)
    return title[:50] or "chat"
