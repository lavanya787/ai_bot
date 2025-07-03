# utils/text_utils.py
import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize_text(text):
    """
    Tokenize, remove stopwords, lemmatize, and filter non-alphabetic tokens.
    """
    if not isinstance(text, str):
        return []

    words = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return tokens

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    """Clean the input text by removing special characters, numbers, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_process(text):
    """Tokenize, remove stopwords, and lemmatize the text."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_text(text):
    """Full preprocessing pipeline."""
    cleaned = clean_text(text)
    processed = tokenize_and_process(cleaned)
    return processed

def create_training_data(chunks, domain="education"):
    """Create a DataFrame with synthetic labels for the education domain."""
    labels = []
    for chunk in chunks:
        if '?' in chunk:
            labels.append('interrogative')
        elif '!' in chunk or any(word in chunk.lower() for word in ['explain', 'describe', 'discuss']):
            labels.append('imperative')
        else:
            labels.append('declarative')
    
    return pd.DataFrame({
        "text": chunks,
        "label": labels,
        "domain": [domain] * len(chunks)
    })

import os
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader  # pip install PyPDF2
from docx import Document     # pip install python-docx

def load_documents_from_folder(folder_path, allowed_exts=None):
    allowed_exts = allowed_exts or [".txt", ".md", ".csv", ".json", ".pdf", ".docx"]
    documents = []

    for filename in tqdm(os.listdir(folder_path), desc="📄 Loading files"):
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in allowed_exts:
            continue

        try:
            if ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            elif ext == ".csv":
                df = pd.read_csv(file_path)
                content = df.to_string()

            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            elif ext == ".pdf":
                content = ""
                reader = PdfReader(file_path)
                for page in reader.pages:
                    content += page.extract_text() or ""

            elif ext == ".docx":
                doc = Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])

            else:
                continue

            documents.append((filename, content))

        except Exception as e:
            print(f"❌ Skipped {filename}: {e}")

    return documents
