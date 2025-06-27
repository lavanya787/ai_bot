import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import datetime
import time

EMAIL_ADDRESS = "lavanyarajganeshan1004@gmail.com"
EMAIL_PASSWORD = "wksubdozmhuevsut"
EMAIL_RECEIVER = "lavanyarajganeshan123@gmail.com"

def send_alert_email(subject, html_content, attachment_paths=None, log_file=None, max_retries=3, retry_delay=5):
    """
    Sends an HTML email with optional file attachments and logs the result.
    Automatically retries on failure.
    """
    timestamp = datetime.datetime.now().isoformat()

    # Compose the email once (outside the retry loop)
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(html_content, "html"))

    if attachment_paths:
        for path in attachment_paths:
            with open(path, "rb") as file:
                part = MIMEApplication(file.read(), Name=os.path.basename(path))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
                msg.attach(part)

    # Retry logic
    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)

            # Log success
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] ✅ Email sent: {subject} (Attempt {attempt})\n")
            return  # success — exit function

        except Exception as e:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] ❌ Attempt {attempt} failed: {e}\n")
            print(f"❌ Attempt {attempt} failed: {e}")

            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                print("❌ All retries failed.")
