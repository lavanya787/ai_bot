import json
import os
import re
import csv
from collections import defaultdict
from datetime import datetime
from typing import Union
from googletrans import Translator
import logging

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

LOG_PATH_JSON = "logs/domain_usage.json"
LOG_PATH_CSV = "logs/domain_scores.csv"
translator = Translator()

# Domain-wise keyword mappings
DOMAIN_KEYWORDS = {
    "education": [
        "student", "exam", "grade", "subject", "marks", "syllabus", "teacher", "university",
        "assignment", "semester", "attendance", "course", "question paper", "school"
    ],
    "qa": [
        "question", "answer", "faq", "support", "helpdesk", "ticket", "chatbot",
        "inquiry", "troubleshooting", "resolution"
    ],
    "sentiment": [
        "review", "rating", "feedback", "sentiment", "opinion", "comment", "reaction",
        "emotion", "positive", "negative", "neutral"
    ],
    "physics": [
        "physics", "motion", "velocity", "force", "energy", "power", "quantum",
        "relativity", "gravity", "mass", "newton", "thermodynamics", "friction", "acceleration"
    ],
    "chemistry": [
        "chemistry", "reaction", "molecule", "compound", "acid", "base", "organic",
        "inorganic", "element", "atom", "bond", "catalyst", "pH", "solvent"
    ],
    "mathematics": [
        "math", "algebra", "geometry", "calculus", "trigonometry", "equation", "variable",
        "theorem", "proof", "derivative", "integration", "statistics", "function", "matrix"
    ],
    "biology": [
        "biology", "cell", "organism", "dna", "rna", "gene", "evolution", "photosynthesis",
        "mitochondria", "bacteria", "virus", "anatomy", "respiration", "taxonomy", "protein"
    ]
}

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest="en")
        return translated.text
    except Exception:
        logger.warning(f"Translation failed: {text[:50]}...")
        return text  # Fallback to original

def detect_domain(text: Union[str, list, tuple], top_n: int = 2) -> Union[str, list]:
    """
    Detect domain from raw text content using keyword TF-IDF style scoring.
    """
    try:
        def flatten_input(item, depth=0, max_depth=10):
            if depth > max_depth:
                logger.warning(f"Max recursion depth {max_depth} reached in flatten_input")
                return str(item)
            if isinstance(item, (list, tuple)):
                return " ".join(flatten_input(subitem, depth + 1, max_depth) for subitem in item if subitem)
            elif isinstance(item, dict):
                return " ".join(flatten_input(v, depth + 1, max_depth) for v in item.values() if v)
            else:
                return str(item)

        text = flatten_input(text)
        if not isinstance(text, str):
            text = str(text)
        logger.debug(f"Input to detect_domain: type={type(text)}, sample={text[:100]}")

        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        word_freq = defaultdict(int)
        for word in tokens:
            word_freq[word] += 1

        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            domain_score = 0
            for keyword in keywords:
                tf = word_freq.get(keyword, 0)
                idf = 1 + 1 / (1 + sum(1 for d in DOMAIN_KEYWORDS.values() if keyword in d))
                domain_score += tf * idf
            scores[domain] = domain_score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [(d, round(s, 4)) for d, s in sorted_scores if s > 0]

        if filtered:
            log_domain_usage(filtered)
            log_domain_scores(filtered, text_snippet=text)
        else:
            log_domain_usage("general")
            log_domain_scores([("general", 0.0)], text_snippet=text)

        return filtered[0][0] if top_n == 1 else filtered[:top_n]
    except Exception as e:
        logger.error(f"Domain detection failed: {e}")
        raise

def get_domain_from_file(df) -> str:
    text = " ".join(map(str, df.columns.tolist()))
    if not df.empty:
        text += " " + " ".join(map(str, df.iloc[0].astype(str).tolist()))
    return detect_domain(text)

def log_domain_usage(predictions: Union[str, list]):
    os.makedirs(os.path.dirname(LOG_PATH_JSON), exist_ok=True)
    usage = defaultdict(int)

    if os.path.exists(LOG_PATH_JSON):
        try:
            with open(LOG_PATH_JSON, "r") as f:
                usage.update(json.load(f))
        except Exception:
            pass

    if isinstance(predictions, list):
        for domain, _ in predictions:
            usage[domain] += 1
    else:
        usage[predictions] += 1

    with open(LOG_PATH_JSON, "w") as f:
        json.dump(dict(usage), f, indent=2)

def log_domain_scores(predictions: list, text_snippet: str = ""):
    """
    Log domain prediction scores into a CSV file.
    """
    os.makedirs(os.path.dirname(LOG_PATH_CSV), exist_ok=True)
    is_new_file = not os.path.exists(LOG_PATH_CSV)

    row = {
        "timestamp": datetime.now().isoformat(),
        "text_preview": text_snippet.strip().replace('\n', ' ')[:100]
    }
    for domain, score in predictions:
        row[domain] = score

    all_domains = set(row.keys())

    if not is_new_file:
        with open(LOG_PATH_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                all_domains.update(reader.fieldnames)

    fieldnames = sorted([d for d in all_domains if d not in ["timestamp", "text_preview"]])
    fieldnames = ["timestamp", "text_preview"] + fieldnames

    with open(LOG_PATH_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)

# Test
if __name__ == "__main__":
    text = input("Enter text to detect domain: ")
    result = detect_domain(text, top_n=3)
    print("Predicted:", result)