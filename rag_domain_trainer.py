import os
import json
import shutil
import hashlib
from datetime import datetime
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from llm_handler import LLMHandler
from utils.domain_detector import detect_domain
from utils.preprocessing import Preprocessor
from utils.logger import Logger
import logging
from file_processing.processor import extract_text

log = Logger().logger
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log.handlers = [stream_handler]
log.handlers[0].stream.reconfigure(encoding='utf-8')

def get_file_hash(filepath):
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        log.error(f"Failed to compute hash for {filepath}: {e}")
        return ""

def fallback_ocr(file_path):
    try:
        images = convert_from_path(file_path, dpi=300)
        ocr_text = [pytesseract.image_to_string(img).strip() for img in images]
        return "\n".join([text for text in ocr_text if text])
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return ""

def extract_table_text(file_path):
    try:
        rows = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    try:
                        for row in table:
                            cleaned_row = [str(cell[0]) if isinstance(cell, tuple) else str(cell or "") for cell in row]
                            rows.append(" | ".join(cleaned_row))
                    except Exception as e:
                        log.warning(f"Table extraction failed for page: {e}")
                        continue
        return "\n".join(rows)
    except Exception as e:
        log.warning(f"Table extraction failed: {e}")
        return ""

def update_metadata(domain_folder, file_name, file_path):
    metadata_path = os.path.join(domain_folder, "metadata.json")
    file_hash = get_file_hash(file_path)
    metadata = []

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    metadata = [entry for entry in loaded if isinstance(entry, dict) and all(k in entry for k in ["file", "uploaded_at", "source", "hash"])]
                    if len(metadata) != len(loaded):
                        log.warning(f"Filtered out {len(loaded) - len(metadata)} invalid metadata entries")
                else:
                    log.warning(f"Invalid metadata format in {metadata_path}, resetting to empty list")
        except json.JSONDecodeError as e:
            log.warning(f"JSON decode error in {metadata_path}: {e}, resetting to empty list")
        except Exception as e:
            log.warning(f"Failed to read {metadata_path}: {e}, resetting to empty list")

    for entry in metadata:
        if isinstance(entry, dict) and entry.get("hash") == file_hash:
            log.info("Duplicate file detected. Skipping retraining.")
            return False

    metadata.append({
        "file": file_name,
        "uploaded_at": datetime.now().isoformat(),
        "source": "user_upload",
        "hash": file_hash
    })

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        log.debug(f"Updated metadata: {metadata}")
    except Exception as e:
        log.error(f"Failed to write metadata to {metadata_path}: {e}")
        return False

    return True

def store_and_train(file_path, domain, cache_dir, output_dir):
    doc_id = hashlib.md5(os.path.basename(file_path).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{doc_id}.txt")
    text = ""

    if os.path.exists(cache_path):
        logging.info(f"Duplicate file detected: {file_path}. Loading cached text.")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        start_time = datetime.now()
        with open(file_path, "rb") as f:
            text = extract_text(f)  # Use the processor function
        if not text.startswith("[ERROR]"):
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
            logging.info(f"Saved cached text: {cache_path}, size: {len(text)} bytes, took {(datetime.now() - start_time).total_seconds()} seconds")
        else:
            logging.error(f"Failed to extract text for {file_path}")
            return text

    output_path = os.path.join(output_dir, f"{doc_id}.txt")
    if os.path.exists(output_path):
        logging.info(f"Duplicate file detected. Skipping copy to {output_path}.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Copied text file to {output_path}")

    if not os.path.exists(os.path.join(output_dir, f"{doc_id}_indexed.txt")):
        logging.info(f"Training and indexing for {file_path}")
        with open(os.path.join(output_dir, f"{doc_id}_indexed.txt"), "w") as f:
            f.write("Indexed")
    else:
        logging.info(f"Skipping indexing and training for {file_path} (already indexed)")

    return text