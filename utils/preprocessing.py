import os
import re
import logging
import pandas as pd
import pdfplumber
import docx
import pptx
import json
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from langdetect import detect
from datetime import datetime
from utils.domain_detector import detect_domain
from sklearn.model_selection import train_test_split
import nltk
for res in ["punkt", "wordnet", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

# Logger configuration - only to file, no console output
logging.basicConfig(
    filename='app.log',
    encoding='utf-8',
    level=logging.INFO,
    filemode='a'  # Append to existing log file
)
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {
            "en": set(stopwords.words("english")),
            "fr": set(stopwords.words("french"))
        }

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "en"

    def translate_to_english(self, text):
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except:
            return text

    def extract_text_from_pdf(self, file_path):
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        if not text.strip():
            text = self.extract_text_with_ocr(file_path)
        return text

    def extract_text_with_ocr(self, file_path):
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            logger.error(f"OCR failed: {e}")
        return text

    def extract_text_from_docx(self, file_path):
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"DOCX read failed: {e}")
        return text

    def extract_text_from_pptx(self, file_path):
        text = ""
        try:
            ppt = pptx.Presentation(file_path)
            for slide in ppt.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        text += shape.text + "\n"
        except Exception as e:
            logger.error(f"PPTX read failed: {e}")
        return text

    def extract_text_from_json(self, file_path):
        text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON read failed: {e}")
        return text

    def extract_text_from_image(self, file_path):
        text = ""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
        return text

    def extract_text_from_excel(self, file_path):
        text = ""
        try:
            df = pd.read_excel(file_path)
            text = df.astype(str).to_string(index=False)
        except Exception as e:
            logger.error(f"Excel read failed: {e}")
        return text

    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT read failed: {e}")
            return ""

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E]", "", text)  # Remove non-printable
        return text.strip()

    def get_next_file_number(self, domain, output_dir):
        """Get the next available numeric suffix for the given domain."""
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"{domain}_") and f.endswith("_preprocessed.csv")]
        if not existing_files:
            return 1
        numbers = [int(f.split("_")[1]) for f in existing_files if f.split("_")[1].isdigit()]
        return max(numbers, default=0) + 1

    def preprocess_file(self, file_path, domain="general"):
        logger.info(f"📄 Processing: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "", {"error": f"File not found {file_path}"}, pd.DataFrame()

        ext = Path(file_path).suffix.lower()
        raw_text = ""

        if not ext or ext == "":
            logger.warning(f"No recognizable extension for {file_path}. Attempting to process as text.")
            raw_text = self.extract_text_from_txt(file_path)
        elif ext in [".pdf"]:
            raw_text = self.extract_text_from_pdf(file_path)
        elif ext in [".docx"]:
            raw_text = self.extract_text_from_docx(file_path)
        elif ext in [".pptx"]:
            raw_text = self.extract_text_from_pptx(file_path)
        elif ext in [".json"]:
            raw_text = self.extract_text_from_json(file_path)
        elif ext in [".jpeg", ".jpg", ".png"]:
            raw_text = self.extract_text_from_image(file_path)
        elif ext in [".xls", ".xlsx"]:
            raw_text = self.extract_text_from_excel(file_path)
        elif ext in [".txt"]:
            raw_text = self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return "", {"error": f"Unsupported type {ext}"}, pd.DataFrame()

        if not raw_text.strip():
            logger.warning("🛑 No content found in file after extraction.")
            return "", {"error": "Empty file"}, pd.DataFrame()

        lang = self.detect_language(raw_text)
        logger.info(f"🌐 Detected Language: {lang}")

        if lang != "en":
            logger.info("🌍 Translating to English...")
            raw_text = self.translate_to_english(raw_text)

        cleaned_text = self.clean_text(raw_text)

        if not cleaned_text:
            logger.warning("⚠️ Cleaned text is empty.")
            return "", {"error": "No clean text"}, pd.DataFrame()

        sentences = sent_tokenize(cleaned_text)
        intent_data = {
            "sentence": [],
            "intent": []
        }

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(k in sentence_lower for k in ["define", "explain", "what is"]):
                intent = "ask_question"
            elif any(k in sentence_lower for k in ["summarize", "summary", "overview"]):
                intent = "summarize_document"
            elif any(k in sentence_lower for k in ["calculate", "solve", "determine"]):
                intent = "calculation"
            else:
                intent = "default"

            intent_data["sentence"].append(sentence)
            intent_data["intent"].append(intent)

        df = pd.DataFrame(intent_data)
        domain_detected = detect_domain(cleaned_text)
        logger.info(f"📌 Detected domain: {domain_detected}")
        metadata = {
            "filename": Path(file_path).name,
            "language": lang,
            "domain": domain_detected,
            "word_count": len(cleaned_text.split()),
            "processed_at": datetime.now().isoformat()
        }

        # Create domain-specific folder
        rag_dir = os.path.join("rag_data", domain_detected)
        os.makedirs(rag_dir, exist_ok=True)
        logger.info(f"📁 Created folder: {rag_dir}")

        # Save preprocessed data with domain, number, and readable timestamp
        output_dir = "preprocessed_data"
        os.makedirs(output_dir, exist_ok=True)
        file_number = self.get_next_file_number(domain_detected, output_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(output_dir, f"{domain_detected}_{file_number}_{timestamp}_preprocessed.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved preprocessed data to {output_file}")

        logger.info(f"✅ Preprocessing completed: {metadata}")
        return cleaned_text, metadata, df  # Return 3 values to match caller expectation

    def prepare_training_data(self, file_paths):
        all_data = pd.DataFrame()
        for file_path in file_paths:
            _, _, df = self.preprocess_file(file_path)
            all_data = pd.concat([all_data, df], ignore_index=True)
        
        # Save combined dataset
        output_dir = "preprocessed_data"
        os.makedirs(output_dir, exist_ok=True)
        combined_file = os.path.join(output_dir, "combined_dataset.csv")
        all_data.to_csv(combined_file, index=False)
        logger.info(f"💾 Saved combined dataset to {combined_file}")

        # Split into train and validation
        train_df, val_df = train_test_split(all_data, test_size=0.2, random_state=42)
        train_file = os.path.join(output_dir, "train_dataset.csv")
        val_file = os.path.join(output_dir, "val_dataset.csv")
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        logger.info(f"✅ Prepared training data: Train {train_file}, Val {val_file}")

        return train_file, val_file


if __name__ == "__main__":
    preprocessor = Preprocessor()
    sample_files = [
        "C:\\Users\\lavan\\AppData\\Local\\Temp\\tmpwy4kjq8i.pdf"  # Example from your input
    ]
    for file in sample_files:
        cleaned_text, metadata, df = preprocessor.preprocess_file(file)
    train_file, val_file = preprocessor.prepare_training_data(sample_files)
    logger.info(f"✅ Data prepared for custom model training. Use {train_file} and {val_file} to train your model with domain {metadata['domain']}.")