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

def load_documents_from_folder(folder_path, allowed_exts=None):
    allowed_exts = allowed_exts or [".txt", ".md", ".csv", ".json"]
    documents = []

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in allowed_exts):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append((filename, f.read()))
    
    return documents

