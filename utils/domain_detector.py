import os
import re
import json
import joblib
import logging
import hashlib
import csv
import pickle
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from googletrans import Translator

# Initialize logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        filename="logs/domain_detector.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(stream)

# Constants
LOG_PATH_JSON = "logs/domain_usage.json"
LOG_PATH_CSV = "logs/domain_scores.csv"
BASE_MODEL_DIR = Path("domain_models")
BASE_MODEL_DIR.mkdir(exist_ok=True)
VECTORIZER_FILE = "vectorizer.pkl"
MODEL_FILE = "logistic_model.pkl"
CHECKPOINT_PATH = "checkpoint.pt"
#Create or append to a file:
TRAINING_DATA_PATH = Path("domain_models/training_dataset.csv")
TRAINING_DATA_PATH.parent.mkdir(exist_ok=True)

# Keywords for hybrid scoring
DOMAIN_KEYWORDS = {
    "education": ["student", "exam", "grade", "syllabus", "university", "teacher", "assignment"],
    "qa": ["question", "answer", "faq", "support", "chatbot", "inquiry", "ticket"],
    "sentiment": ["review", "rating", "feedback", "opinion", "emotion", "positive", "negative"],
    "physics": ["motion", "velocity", "force", "energy", "gravity", "mass", "newton", "ohm"],
    "chemistry": ["reaction", "molecule", "acid", "compound", "atom", "catalyst", "bond"],
    "mathematics": ["equation", "algebra", "theorem", "calculus", "function", "matrix", "proof"],
    "biology": ["cell", "photosynthesis", "dna", "gene", "organism", "anatomy", "evolution"]
}

translator = Translator()


# ----------------------- Clean / Hash -----------------------

def clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(text)).lower()


def hash_text(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


# ----------------------- Dataset Builder -----------------------

def build_training_dataset() -> pd.DataFrame:
    rows = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            for phrase in [f"What is {kw}?", f"Define {kw}", f"Explain {kw}"]:
                rows.append((phrase, domain))
    return pd.DataFrame(rows, columns=["text", "label"])


# ----------------------- ML Training & Loading -----------------------
def append_to_training_data(text: str, label: str):
    """Add a new labeled sample to the training set."""
    if not text.strip() or not label.strip():
        logger.warning("⚠️ Skipped adding empty training data")
        return

    is_new = not TRAINING_DATA_PATH.exists()
    with open(TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        if is_new:
            writer.writeheader()
        writer.writerow({"text": text.strip(), "label": label.strip().lower()})
    logger.info(f"✅ Added training sample: [{label}] {text[:60]}...")

def evaluate_model(model, vectorizer, X_raw, y_true):
    try:
        X_vec = vectorizer.transform(X_raw)
        y_pred = model.predict(X_vec)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        report = classification_report(y_true, y_pred, target_names=model.classes_)

        logger.info(f"🎯 Accuracy: {acc:.4f}")
        logger.info(f"🧮 Confusion Matrix:\n{cm}")
        logger.info(f"📊 Classification Report:\n{report}")

        return acc, cm, report
    except Exception as e:
        logger.error(f"⚠️ Evaluation failed: {e}")
        return 0.0, None, ""

def train_ml_model():
    logger.info("🧠 Retraining domain model...")
    base_df = build_training_dataset()

    if TRAINING_DATA_PATH.exists():
        try:
            user_df = pd.read_csv(TRAINING_DATA_PATH).dropna()
            full_df = pd.concat([base_df, user_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"⚠️ Failed loading user training data: {e}")
            full_df = base_df
    else:
        full_df = base_df

    full_df = full_df.dropna()
    X_raw = full_df["text"].values
    y = full_df["label"].values

    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    except Exception as e:
        logger.warning(f"⚠️ Train/Test split failed, using full data for training: {e}")
        X_train_raw, y_train = X_raw, y
        X_test_raw, y_test = [], []

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Evaluate
    if len(X_test_raw) > 1:
        evaluate_model(model, vectorizer, X_test_raw, y_test)
    else:
        logger.warning("⚠️ Not enough data to evaluate model")

    # Save artifacts
    joblib.dump(model, BASE_MODEL_DIR / MODEL_FILE)
    joblib.dump(vectorizer, BASE_MODEL_DIR / VECTORIZER_FILE)
    logger.info("✅ Updated model and vectorizer saved.")

    return model, vectorizer

def load_ml_model() -> tuple:
    model_path = BASE_MODEL_DIR / MODEL_FILE
    vectorizer_path = BASE_MODEL_DIR / VECTORIZER_FILE
    if not model_path.exists() or not vectorizer_path.exists():
        logger.warning("❗ Model not found, retraining...")
        return train_ml_model()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# ----------------------- Hybrid Detection -----------------------

def keyword_score(text: str) -> dict:
    tokens = word_tokenize(clean_text(text))
    scores = defaultdict(int)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] += sum(tokens.count(k) for k in keywords)
    return scores


def detect_domain(text: str, top_n: int = 1, session_store: dict = None) -> Union[str, list]:
    try:
        model, vectorizer = load_ml_model()
        X = vectorizer.transform([text])
        ml_pred = model.predict(X)[0]
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        ml_pred = "general"

    kw_scores = keyword_score(text)
    kw_scores[ml_pred] += 2  # Boost ML prediction
    sorted_domains = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
    detected = sorted_domains[0][0] if sorted_domains and sorted_domains[0][1] > 0 else "general"

    if session_store is not None:
        session_store.setdefault("recent_domains", [])
        if detected not in session_store["recent_domains"]:
            session_store["recent_domains"].append(detected)

    logger.info(f"📌 Detected domain: {detected} (Top scores: {sorted_domains[:3]})")
    log_domain_usage(detected)
    log_domain_scores(detected, "hybrid", text[:100])
    ensure_domain_folder(detected)
    return detected if top_n == 1 else sorted_domains[:top_n]


# ----------------------- Logging -----------------------

def log_domain_usage(domain: str):
    os.makedirs(os.path.dirname(LOG_PATH_JSON), exist_ok=True)
    usage = defaultdict(int)
    if os.path.exists(LOG_PATH_JSON):
        with open(LOG_PATH_JSON, "r") as f:
            usage.update(json.load(f))
    usage[domain] += 1
    with open(LOG_PATH_JSON, "w") as f:
        json.dump(dict(usage), f, indent=2)


def log_domain_scores(domain: str, method: str, text_snippet: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH_CSV), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH_CSV)
    row = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "predicted_domain": domain,
        "text_preview": text_snippet.replace("\n", " ")[:100]
    }
    with open(LOG_PATH_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ----------------------- Directory & Session -----------------------

def ensure_domain_folder(domain: str) -> str:
    folder = Path("rag_data") / domain
    folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created folder: {folder}")
    return str(folder)


# ----------------------- Translate if Needed -----------------------

def translate_to_english(text: str) -> str:
    try:
        return translator.translate(text, dest="en").text
    except Exception:
        logger.warning("⚠️ Translation failed, using original text")
        return text


# ----------------------- CLI Test -----------------------

if __name__ == "__main__":
    input_text = input("Enter sample text:\n")
    session = {}
    domain = detect_domain(input_text, session_store=session)
    print("Detected:", domain)
    print("Session state:", session)
