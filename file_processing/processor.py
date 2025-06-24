import os
import pandas as pd
import json
import fitz  # PyMuPDF
import numpy as np
from docx import Document
from PIL import Image
import pytesseract
from contextlib import contextmanager
import tempfile
import re
from typing import List, Dict, Any
import logging
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import docx

DetectorFactory.seed = 0

logger = logging.getLogger("processor")
logging.basicConfig(level=logging.INFO)

# Optional NLP tools
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 100_000_000  # Increased from default 1M
    SPACY_AVAILABLE = True
except Exception:
    nlp = None
    SPACY_AVAILABLE = False


try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=True)
    EASY_OCR_AVAILABLE = True
except Exception:
    EASY_OCR_AVAILABLE = False

# ---------------------- UTILS ----------------------

@contextmanager
def open_tempfile(file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        yield tmp.name
    os.remove(tmp.name)

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except:
        return "unknown"

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

# ------------------ TEXT EXTRACTION ------------------

def extract_text_from_image(file):
    try:
        if EASY_OCR_AVAILABLE:
            result = easyocr_reader.readtext(np.array(Image.open(file)), detail=0)
            return " ".join(result)
        else:
            image = Image.open(file)
            return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"[OCR ERROR] {e}"

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"[PDF ERROR] {e}"

def extract_text_from_pdf_plumber(file):
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        return f"[PDFPlumber ERROR] {e}"

def extract_text_from_docx(file):
    try:
        with open_tempfile(file, ".docx") as path:
            doc = Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"[DOCX ERROR] {e}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        return file.read().decode("utf-8")
    except Exception as e:
        return f"[TXT ERROR] {e}"

def extract_text_from_csv(file):
    try:
        file.seek(0)
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).apply(" | ".join, axis=1))
    except Exception as e:
        return f"[CSV ERROR] {e}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        data = json.load(file)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"[JSON ERROR] {e}"

def extract_rows_from_file(file_path):
    ext = file_path.split('.')[-1].lower()

    if ext == 'csv':
        df = pd.read_csv(file_path)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        df = pd.DataFrame({'question': lines, 'context': lines, 'is_answer': 1})
    elif ext == 'pdf':
        reader = PyPDF2.PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
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

def extract_text(file):
    name = file.name.lower()
    if name.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(file)
    elif name.endswith(".pdf"):
        text = extract_text_from_pdf_plumber(file) if PDFPLUMBER_AVAILABLE else extract_text_from_pdf(file)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(file)
    elif name.endswith(".txt"):
        text = extract_text_from_txt(file)
    elif name.endswith(".csv"):
        text = extract_text_from_csv(file)
    elif name.endswith(".json"):
        text = extract_text_from_json(file)
    else:
        return "[Unsupported file type: only PDF, DOCX, TXT, CSV, JSON, PNG, JPG]"

    lang = detect_language(text)
    if lang != "en" and lang != "unknown":
        logger.info("Translating document to English...")
        text = translate_to_english(text)

    return text

# ------------------ PARALLEL CHUNKING ------------------

def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    if not text or len(text.strip()) < 50:
        return []

    try:
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    except Exception as e:
        logger.warning(f"Chunk fallback due to error: {e}")
        sentences = text.split(". ")

    chunks = []
    current_chunk = ""

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
