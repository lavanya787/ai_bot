import re
import chardet
import pandas as pd
import PyPDF2
import pdfplumber
import nltk
import os
import logging
import tempfile
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from langdetect import detect
from datetime import datetime
from pathlib import Path
# Logging configuration
logging.basicConfig(
    filename='logs/app.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]
logger.handlers[0].stream.reconfigure(encoding='utf-8')

for res in ["punkt", "wordnet", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {
            "en": set(stopwords.words("english")),
            "fr": set(stopwords.words("french"))
        }
        self.section_keywords = ["chapter", "definition", "section", "example", "note", "theorem", "lemma", "corollary", "proof"]
        self.equation_pattern = r"[A-Za-z]+\s*=\s*[^.\n]+"
        self.domain_patterns = {
            "physics": [r"law of \w+", r"equation of motion", r"gravitational constant"],
            "mathematics": [r"proof", r"lemma", r"theorem", r"corollary", r"integral of", r"matrix of"],
            "biology": [r"cell structure", r"photosynthesis", r"dna replication"],
            "chemistry": [r"chemical reaction", r"periodic table", r"molecular structure"]
        }
        self.max_text_size = 5 * 1024 * 1024  # 5MB
        self.chunk_size = 50000  # 50KB
    def clean_text(self, text):
        try:
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[^\w\s.,?!]', '', text)
            tokens = nltk.word_tokenize(text.lower())
            tokens = [t for t in tokens if t not in self.stopwords]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def extract_metadata(self, text, file_path):
        metadata = {
            "filename": Path(file_path).name,
            "word_count": len(text.split()),
            "domain": "physics"  # Default domain
        }

    def detect_language(self, text):
        try:
            if not text.strip():
                return "en"
            return detect(text)
        except:
            logger.warning("Language detection failed, defaulting to English")
            return "en"

    def translate_to_english(self, text):
        try:
            if len(text.encode('utf-8')) > 100000:
                text = text[:100000 // 4]
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

    def preserve_patterns(self, text, domain):
        sections = re.findall(r"\b(" + "|".join(self.section_keywords) + r")\s+\d+", text, flags=re.IGNORECASE)
        equations = re.findall(self.equation_pattern, text)
        domain_hits = []
        for pattern in self.domain_patterns.get(domain, []):
            domain_hits += re.findall(pattern, text, flags=re.IGNORECASE)
        return sections, equations, domain_hits
    def generate_intent_data(self, text):
        try:
            sentences = nltk.sent_tokenize(text)
            intent_data = {"sentence": [], "intent": []}
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if any(keyword in sentence.lower() for keyword in ["define", "what is", "explain"]):
                    intent = "ask_question"
                elif any(keyword in sentence.lower() for keyword in ["summarize", "summary"]):
                    intent = "summarize_document"
                else:
                    intent = "default"
                intent_data["sentence"].append(sentence)
                intent_data["intent"].append(intent)
            
            df = pd.DataFrame(intent_data)
            if df.empty:
                logger.warning("Generated empty intent_data. Adding default entry.")
                df = pd.DataFrame({
                    "sentence": ["Default sentence for training"],
                    "intent": ["default"]
                })
            return df
        except Exception as e:
            logger.error(f"Error generating intent data: {e}")
            return pd.DataFrame({
                "sentence": ["Default sentence for training"],
                "intent": ["default"]
            })
    def general_preprocessing(self, text, domain="general"):
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(item[0]) if isinstance(item, tuple) else str(item) for item in text if item)
        if not isinstance(text, str):
            text = str(text)
        if isinstance(text, bytes):
            encoding = chardet.detect(text)["encoding"]
            text = text.decode(encoding or "utf-8", errors="replace")
    
        text_size = len(text.encode('utf-8'))
        if text_size > self.max_text_size:
            logger.warning(f"Text size {text_size} bytes exceeds {self.max_text_size} bytes, truncating")
            text = text[:self.max_text_size // 4]
    
        lang = self.detect_language(text)
        if lang != "en" and lang != "unknown":
            text = self.translate_to_english(text)
    
        sections, equations, domain_patterns = self.preserve_patterns(text, domain)
    
        text = re.sub(r"[^\w\s=^+*/().-]", "", text)
        text = re.sub(r"\s+", " ", text.strip())
    
        tokens = []
        self.chunk_size = 10000
        for i in range(0, len(text), self.chunk_size - 100):
            chunk = text[i:i + self.chunk_size].strip()
            if not chunk:
                continue
            try:
                chunk_tokens = re.findall(r"\w+|[=^+*/().-]", chunk.lower())
                chunk_tokens = [t for t in chunk_tokens if t not in self.stop_words.get("en", set())]
                tokens.extend(chunk_tokens)
            except MemoryError:
                logger.warning(f"MemoryError in chunk {i//self.chunk_size + 1}, skipping")
                continue
            except Exception as e:
                logger.warning(f"Tokenization failed for chunk {i//self.chunk_size + 1}: {e}")
                return text, {
                    "language": lang,
                    "sections": [],
                    "equations": [],
                    "domain_hits": [],
                    "cleaned_at": datetime.now().isoformat(),
                    "error": str(e)
                }
    
        cleaned = " ".join(tokens)
        annotated = cleaned
        if sections:
            annotated += "\n\n# Sections:\n" + "\n".join(set(sections))
        if equations:
            annotated += "\n\n# Equations:\n" + "\n".join(set(equations))
        if domain_patterns:
            annotated += "\n\n# Domain Patterns:\n" + "\n".join(set(domain_patterns))
    
        metadata = {
            "language": lang,
            "sections": list(set(sections)),
            "equations": list(set(equations)),
            "domain_hits": list(set(domain_patterns)),
            "cleaned_at": datetime.now().isoformat()
        }
        logger.debug(f"Metadata from general_preprocessing: {metadata}")
        return annotated.strip(), metadata

    def preprocess_for_intent(self, text, domain="general"):
        if not text.strip():
            logger.warning("Empty text provided for intent preprocessing")
            return pd.DataFrame({"sentence": [], "intent": []})

        sentences = sent_tokenize(text)
        intents = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in ["define", "what is", "explain", "describe"]):
                intents.append("ask_question")  # Matches IntentClassifier's label2id
            elif any(keyword in sentence_lower for keyword in ["chapter", "summarize", "overview"]):
                intents.append("summarize_document")
            else:
                intents.append("default")
        df = pd.DataFrame({"sentence": sentences, "intent": intents})
        logger.info(f"Generated intent DataFrame with {len(df)} sentences")
        logger.debug(f"Sample intent data: {df.head().to_dict()}")
        return df

    def preprocess_file(self, file_path, domain="general"):
        log = logging.getLogger(__name__)
        text = ""
        
        if not os.path.isfile(file_path):
            log.info("Input is text content, not a file path; processing directly")
            text = file_path
        else:
            ext = os.path.splitext(file_path)[-1].lower()
            try:
                if ext == ".pdf":
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            page_count = len(pdf.pages)
                            log.info(f"Processing PDF with {page_count} pages: {file_path}")
                            for i, page in enumerate(pdf.pages):
                                try:
                                    page_text = page.extract_text() or ""
                                    if page_text.strip():
                                        text += page_text + "\n"
                                        log.debug(f"Page {i+1} text length: {len(page_text)}")
                                    tables = page.extract_tables() or []
                                    for table in tables:
                                        try:
                                            def flatten_cell(cell, depth=0, max_depth=5):
                                                if depth > max_depth or not cell:
                                                    return str(cell or "")
                                                if isinstance(cell, (list, tuple)):
                                                    return " ".join(flatten_cell(item, depth + 1, max_depth) for item in cell if item)
                                                return str(cell)
                                            cleaned_table = [[flatten_cell(cell) for cell in row] for row in table if row]
                                            for row in cleaned_table:
                                                row_text = " | ".join(str(item) for item in row if item) + "\n"
                                                text += row_text
                                                log.debug(f"Page {i+1} table row length: {len(row_text)}")
                                        except Exception as e:
                                            log.warning(f"Skipping problematic table on page {i+1} in {file_path}: {e}")
                                            continue
                                except Exception as e:
                                    log.warning(f"Skipping problematic page {i+1} in {file_path}: {e}")
                                    continue
                    except Exception as e:
                        log.warning(f"pdfplumber failed for {file_path}: {e}, falling back to PyPDF2")
                        with open(file_path, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            page_count = len(reader.pages)
                            log.info(f"PyPDF2 processing PDF with {page_count} pages: {file_path}")
                            for i, page in enumerate(reader.pages):
                                try:
                                    page_text = page.extract_text() or ""
                                    if page_text.strip():
                                        text += page_text + "\n"
                                        log.debug(f"Page {i+1} text length: {len(page_text)}")
                                except Exception as e:
                                    log.warning(f"Skipping problematic page {i+1} in {file_path}: {e}")
                                    continue
                elif ext == ".csv":
                    df = pd.read_csv(file_path).fillna("")
                    text = df.to_string(index=False)
                    log.info(f"Processed CSV: {file_path}, text length: {len(text)}")
                elif ext in [".xls", ".xlsx"]:
                    df = pd.read_excel(file_path).fillna("")
                    text = df.to_string(index=False)
                    log.info(f"Processed Excel: {file_path}, text length: {len(text)}")
                elif ext == ".txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        log.info(f"Processed text file: {file_path}, text length: {len(text)}")
                else:
                    log.error(f"Unsupported file type: {ext}")
                    return "", {"error": f"Unsupported file type: {ext}"}, pd.DataFrame({"sentence": [], "intent": []})
            except Exception as e:
                log.error(f"Failed to extract text from {file_path}: {e}")
                return "", {"error": str(e)}, pd.DataFrame({"sentence": [], "intent": []})
    
        if not text.strip():
            log.warning(f"No text extracted from {file_path}")
            return "", {"error": "No text extracted"}, pd.DataFrame({"sentence": [], "intent": []})
    
        cleaned_text, metadata = self.general_preprocessing(text, domain)
        intent_data = self.preprocess_for_intent(text, domain)
        
        log.debug(f"preprocess_file output: cleaned_text length={len(cleaned_text)}, metadata={metadata}, intent_data rows={len(intent_data)}")
        return cleaned_text, metadata, intent_data