import json
import os
import re
from collections import defaultdict
from googletrans import Translator

LOG_PATH = "logs/domain_usage.json"
translator = Translator()

# 🔍 Domain-wise keyword mappings
DOMAIN_KEYWORDS = {
    "education": [
        "student", "exam", "grade", "subject", "marks", "syllabus", "teacher", "university",
        "assignment", "semester", "attendance", "course", "question paper", "school"
    ],
    "finance": [
        "investment", "stock", "loan", "balance", "profit", "loss", "equity",
        "revenue", "bank", "account", "financial", "budget", "tax", "income", "expense"
    ],
    "healthcare": [
        "patient", "diagnosis", "doctor", "symptom", "treatment", "hospital",
        "prescription", "medical", "medication", "nurse", "clinic", "disease", "therapy"
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
        return text  # Fallback to original if translation fails


def detect_domain(text: str) -> str:
    """
    Detect domain from raw text content using keyword match count.
    """
    text = text.lower()
    counts = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\\b{re.escape(keyword)}\\b", text):
                counts[domain] += 1

    best_match = max(counts, key=counts.get)
    return best_match if counts[best_match] > 0 else "general"


def get_domain_from_file(df) -> str:
    """
    Detect domain from dataframe columns + first row.
    """
    text = " ".join(map(str, df.columns.tolist()))
    if not df.empty:
        text += " " + " ".join(map(str, df.iloc[0].astype(str).tolist()))
    return detect_domain(text)


def log_domain_usage(domain: str):
    """Append domain usage to a persistent log file."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    usage = defaultdict(int)

    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                usage.update(json.load(f))
        except Exception:
            pass  # Skip if corrupted

    usage[domain] += 1

    with open(LOG_PATH, "w") as f:
        json.dump(dict(usage), f, indent=2)


# 🔧 Sample usage for test/debug
if __name__ == "__main__":
    sample_text = "The student appeared for the semester exam and submitted their assignment."
    domain = detect_domain(sample_text)
    print("Detected Domain:", domain)

    # Optional: log usage and create folder
    log_domain_usage(domain)
    folder_path = f"./trained_data/{domain}"
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created/verified folder: {folder_path}")
