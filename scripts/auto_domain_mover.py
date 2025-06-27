import os
import time
import json
import fitz  # PyMuPDF
import docx
import pandas as pd
import shutil
from datetime import datetime
from rag_domain_trainer import store_and_train, get_file_hash
from utils.domain_detector import detect_domain, log_domain_usage
from utils.alerts_email import send_alert_email  # ✅ IMPORT ALERTS

WATCH_DIR = "rag_data"
ERROR_DIR = os.path.join(WATCH_DIR, "_errors")
LOG_FILE = f"logs/runner_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CHECK_INTERVAL = 10  # seconds

def get_processed_hashes(domain_folder):
    metadata_path = os.path.join(domain_folder, "metadata.json")
    if not os.path.exists(metadata_path):
        return set()
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {entry["hash"] for entry in metadata}
    except json.JSONDecodeError:
        return set()

def extract_text_from_file(file_path, ext):
    try:
        if ext.endswith(".pdf"):
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)

        elif ext.endswith((".txt", ".csv", ".json")):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext.endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        elif ext.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            return df.astype(str).apply(" ".join).str.cat(sep=" ")

        elif ext.endswith(".doc"):
            print(f"⚠️ .doc (legacy format) not supported: {file_path}")
            return ""

        else:
            print(f"⚠️ Unsupported file type: {file_path}")
            return ""

    except Exception as e:
        print(f"❌ Error reading file '{file_path}': {e}")
        return ""

def move_to_error_and_alert(file_path, reason):
    os.makedirs(ERROR_DIR, exist_ok=True)
    fname = os.path.basename(file_path)
    dest_path = os.path.join(ERROR_DIR, fname)
    shutil.move(file_path, dest_path)

    print(f"⚠️ Moved to error folder: {fname} | Reason: {reason}")
    send_alert_email(
        subject=f"🚨 File Failed: {fname}",
        html_content=f"<p><b>Reason:</b> {reason}</p><p><b>Original Path:</b> {file_path}</p>",
        attachment_paths=[dest_path],
        log_file=LOG_FILE
    )

def scan_and_move_files():
    print(f"👀 Watching '{WATCH_DIR}/' for new documents...")
    os.makedirs(WATCH_DIR, exist_ok=True)

    while True:
        files = [
            f for f in os.listdir(WATCH_DIR)
            if f.lower().endswith((".pdf", ".txt", ".csv", ".json", ".docx", ".xlsx", ".doc"))
            and os.path.isfile(os.path.join(WATCH_DIR, f))
        ]

        for fname in files:
            full_path = os.path.join(WATCH_DIR, fname)
            ext = fname.lower()

            try:
                file_hash = get_file_hash(full_path)
            except Exception as e:
                print(f"⚠️ Error hashing {fname}: {e}")
                move_to_error_and_alert(full_path, "hashing_failed")
                continue

            text = extract_text_from_file(full_path, ext)
            if not text.strip():
                print(f"⚠️ No content extracted from: {fname}")
                move_to_error_and_alert(full_path, "empty_or_invalid_content")
                continue

            try:
                domain = detect_domain(text)
            except Exception as e:
                print(f"❌ Failed domain detection for {fname}: {e}")
                move_to_error_and_alert(full_path, "domain_detection_failed")
                continue

            domain_folder = os.path.join(WATCH_DIR, domain)
            os.makedirs(domain_folder, exist_ok=True)

            processed_hashes = get_processed_hashes(domain_folder)
            if file_hash in processed_hashes:
                print(f"✔️ Already processed: {fname}")
                os.remove(full_path)
                continue

            new_path = os.path.join(domain_folder, fname)
            os.rename(full_path, new_path)
            print(f"📥 Moved '{fname}' to domain: {domain}")

            try:
                log_domain_usage(domain)
                store_and_train(new_path)
            except Exception as e:
                print(f"❌ Training failed for {fname}: {e}")
                move_to_error_and_alert(new_path, "training_failed")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    scan_and_move_files()
