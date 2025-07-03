import os
import re
import json
import fitz  # PyMuPDF
import docx
import PyPDF2
import numpy as np
import logging
import tempfile
import pandas as pd
from PIL import Image
from langdetect import detect
from contextlib import contextmanager
from typing import List
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from datetime import datetime

# Optional imports with fallbacks
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    USE_PDFPLUMBER = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 100_000_000
    SPACY_AVAILABLE = True
except Exception:
    nlp = None
    SPACY_AVAILABLE = False

try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=False for compatibility
    EASY_OCR_AVAILABLE = True
except Exception:
    EASY_OCR_AVAILABLE = False

# ---------------------- Logging Setup ----------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "processor.log")

logger = logging.getLogger("processor")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='a', encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------------- Utility Functions ----------------------

@contextmanager
def open_tempfile(file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        yield tmp.name
    try:
        os.remove(tmp.name)
    except Exception as e:
        logger.warning(f"Failed to delete temporary file {tmp.name}: {e}")

def detect_language(text: str) -> str:
    try:
        if not text.strip():
            return "unknown"
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception:
        return "unknown"

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def preprocess_extracted_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.{4,}', '.', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    return text.strip()

# ---------------------- Extractors ----------------------

def extract_text_from_image(file):
    try:
        if EASY_OCR_AVAILABLE:
            result = easyocr_reader.readtext(np.array(Image.open(file)), detail=0)
            return " ".join(str(item) for item in result if item)
        else:
            return pytesseract.image_to_string(Image.open(file)).strip()
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"[OCR ERROR] {e}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        return file.read().decode("utf-8").strip()
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return f"[TXT ERROR] {e}"

def extract_text_from_csv(file):
    try:
        file.seek(0)
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).apply(" | ".join, axis=1))
    except Exception as e:
        logger.error(f"CSV extraction error: {e}")
        return f"[CSV ERROR] {e}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        data = json.load(file)
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"JSON extraction error: {e}")
        return f"[JSON ERROR] {e}"

def extract_text_from_docx(file):
    try:
        with open_tempfile(file, ".docx") as path:
            doc = docx.Document(path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return f"[DOCX ERROR] {e}"

def ocr_pdf_images(file):
    try:
        file.seek(0)
        images = []
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=150)  # Lower DPI for speed
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

        ocr_text = []
        def process_image(img):
            if EASY_OCR_AVAILABLE:
                result = easyocr_reader.readtext(np.array(img), detail=0)
                return " ".join(str(item) for item in result if item)
            return pytesseract.image_to_string(img).strip()

        with ThreadPoolExecutor(max_workers=4) as executor:
            ocr_text = list(executor.map(process_image, images))
        
        text = "\n".join(t for t in ocr_text if t.strip())
        return text if text.strip() else "[OCR fallback failed]"
    except Exception as e:
        logger.error(f"OCR fallback failed: {e}")
        return "[OCR fallback failed]"
    
def extract_text_from_pdf(file):
    text = ""
    try:
        file.seek(0)
        start_time = datetime.now()
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            logger.info(f"Processing PDF with {doc.page_count} pages")
            text = "\n".join(page.get_text("text") for page in doc if page.get_text("text").strip())
        if text.strip():
            logger.info(f"[fitz] Extracted text, took {(datetime.now() - start_time).total_seconds()} seconds")
            return preprocess_extracted_text(text)
        else:
            logger.warning("[fitz] Extracted text is empty, trying OCR fallback...")
    except Exception as e:
        logger.warning(f"[fitz] failed: {e}")

    # OCR fallback only if necessary
    try:
        file.seek(0)
        text = ocr_pdf_images(file)
        if text.strip():
            logger.info(f"[OCR] Extracted text, took {(datetime.now() - start_time).total_seconds()} seconds")
            return preprocess_extracted_text(text)
        else:
            logger.warning("[OCR] Extracted text is empty")
    except Exception as e:
        logger.warning(f"[OCR] failed: {e}")

    logger.warning("All extraction methods failed")
    return "[ERROR] No text extracted"

# ---------------------- Main Extraction Function ----------------------

def extract_text(file):
    """
    Main text extraction function.
    Accepts a file-like object with a .name attribute.
    Supports PNG, JPG, PDF, DOCX, TXT, CSV, JSON.
    """
    name = file.name.lower()
    logger.info(f"Extracting text from {name}")

    if name.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(file)
    elif name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif name.endswith(".txt"):
        text = extract_text_from_txt(file)
    elif name.endswith(".csv"):
        text = extract_text_from_csv(file)
    elif name.endswith(".json"):
        text = extract_text_from_json(file)
    else:
        logger.error(f"Unsupported file type: {file.name}")
        return "[Unsupported file type: only PDF, DOCX, TXT, CSV, JSON, PNG, JPG]"

    text = preprocess_extracted_text(text)

    if not text.strip() or text.startswith("[ERROR]") or "Unsupported file type" in text:
        logger.warning(f"Extracted text is empty or invalid for: {file.name}")
        return "[ERROR] Empty or invalid content"

    lang = detect_language(text)
    if lang != "en" and lang != "unknown":
        logger.info("Translating document to English...")
        text = translate_to_english(text)

    return text

# ---------------------- File-Path Based Extraction ----------------------

def extract_text_from_file(file_path, file_name):
    """
    Extract text from a file on disk by path.
    Supports common file types: PDF, DOCX, TXT, CSV, XLSX.
    """
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Extracting text from file: {file_path}")

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            return preprocess_extracted_text(df.to_string(index=False))
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            return preprocess_extracted_text(df.to_string(index=False))
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                return extract_text_from_pdf(f)
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return preprocess_extracted_text(text)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return preprocess_extracted_text(f.read())
        else:
            logger.error(f"Unsupported file type: {ext}")
            return "[ERROR] Unsupported file type"
    except Exception as e:
        logger.error(f"Failed to extract text from file {file_path}: {e}")
        return f"[ERROR] Could not read file: {e}"

# ---------------------- Text Chunking ----------------------

def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    if not text or len(text.strip()) < 50:
        logger.warning("Text is empty or too short to chunk.")
        return []

    try:
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    except Exception as e:
        logger.warning(f"Chunk fallback due to error: {e}")
        sentences = text.split(". ")

    chunks = []

    def process_chunk(start_idx):
        chunk = ""
        idx = start_idx
        while idx < len(sentences) and len(chunk) + len(sentences[idx]) < chunk_size:
            chunk += sentences[idx] + " "
            idx += 1
        return chunk.strip()

    with ThreadPoolExecutor(max_workers=4) as executor:
        i = 0
        futures = []
        while i < len(sentences):
            futures.append(executor.submit(process_chunk, i))
            j = i
            chunk_len = 0
            while j < len(sentences) and chunk_len + len(sentences[j]) < chunk_size:
                chunk_len += len(sentences[j])
                j += 1
            i = j

        for future in futures:
            chunk = future.result()
            if len(chunk.split()) >= 10:
                chunks.append(chunk)

    return chunks

# ------------------ Data Extraction for QA Models ------------------

def extract_rows_from_file(file_path):
    ext = file_path.split('.')[-1].lower()

    try:
        if ext == 'csv':
            df = pd.read_csv(file_path)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            df = pd.DataFrame({'question': lines, 'context': lines, 'is_answer': 1})
        elif ext == 'pdf':
            with open(file_path, 'rb') as f:
                text = extract_text_from_pdf(f)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            df = pd.DataFrame({'question': lines, 'context': lines, 'is_answer': 1})
        elif ext == 'docx':
            doc = docx.Document(file_path)
            lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            df = pd.DataFrame({'question': lines, 'context': lines, 'is_answer': 1})
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if all(col in df.columns for col in ['question', 'context', 'is_answer']):
            return df
        else:
            df = pd.DataFrame({'question': df.iloc[:, 0], 'context': df.iloc[:, 0], 'is_answer': 1})
            return df
    except Exception as e:
        logger.error(f"Failed to extract rows from {file_path}: {e}")
        return pd.DataFrame({'question': [], 'context': [], 'is_answer': []})