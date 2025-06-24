import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import datetime

# 🔒 Replace these with your actual email credentials and receiver
EMAIL_ADDRESS = "lavanyarajganeshan1004@gmail.com"
EMAIL_PASSWORD = "wksu bdoz mhue vsut"
EMAIL_RECEIVER = "lavanyarajganeshan123@gmail.com"

def send_alert_email(subject, html_content, attachment_paths=None, log_file=None):
    """
    Sends an HTML email with optional file attachments and logs the result.

    :param subject: Email subject line
    :param html_content: Body of the email in HTML
    :param attachment_paths: List of file paths to attach (optional)
    :param log_file: Path to save log entries (optional)
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject

        # Attach HTML message
        msg.attach(MIMEText(html_content, "html"))

        # Attach files
        if attachment_paths:
            for path in attachment_paths:
                with open(path, "rb") as file:
                    part = MIMEApplication(file.read(), Name=os.path.basename(path))
                    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
                    msg.attach(part)

        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        # Log result
        if log_file:
            with open(log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"[{timestamp}] ✅ Email sent: {subject}\n")

    except Exception as e:
        if log_file:
            with open(log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"[{timestamp}] ❌ Email error: {e}\n")
        print(f"❌ Failed to send email: {e}")
