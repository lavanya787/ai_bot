import os
import time
import json
import shutil
from datetime import datetime
from rag_domain_trainer import store_and_train, get_file_hash
from utils.domain_detector import detect_domain, log_domain_usage
from utils.alerts_email import send_alert_email
from file_processing.processor import extract_text_from_file

WATCH_DIR = "rag_data"
ERROR_DIR = os.path.join(WATCH_DIR, "_errors")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"runner_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

def move_file_to_domain_folder():
    print(f"👀 Watching '{WATCH_DIR}/' for new documents...")
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    while True:
        files = [
            f for f in os.listdir(WATCH_DIR)
            if f.lower().endswith((".pdf", ".txt", ".csv", ".json", ".docx", ".xlsx", ".doc"))
            and os.path.isfile(os.path.join(WATCH_DIR, f))
        ]

        for fname in files:
            full_path = os.path.join(WATCH_DIR, fname)

            try:
                file_hash = get_file_hash(full_path)
            except Exception as e:
                print(f"⚠️ Error hashing {fname}: {e}")
                move_to_error_and_alert(full_path, "hashing_failed")
                continue

            try:
                text = extract_text_from_file(full_path, fname)
                if not text.strip():
                    raise ValueError("Empty content")
            except Exception as e:
                print(f"⚠️ Error extracting text from {fname}: {e}")
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

            # Move the file to domain folder
            new_path = os.path.join(domain_folder, fname)
            try:
                os.rename(full_path, new_path)
                print(f"📥 Moved '{fname}' to domain: {domain}")
            except Exception as e:
                print(f"❌ Failed to move {fname}: {e}")
                move_to_error_and_alert(full_path, "move_failed")
                continue

            # Log usage and train
            try:
                log_domain_usage(domain)
                store_and_train(new_path)
            except Exception as e:
                print(f"❌ Training failed for {fname}: {e}")
                move_to_error_and_alert(new_path, "training_failed")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    move_file_to_domain_folder()
