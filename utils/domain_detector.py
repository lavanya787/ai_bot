import json
import os
from collections import defaultdict
from googletrans import Translator

# Define keywords per domain
DOMAIN_KEYWORDS = {
    "physics": ["physics", "motion", "gravity", "energy", "quantum", "relativity"],
    "chemistry": ["chemistry", "reaction", "molecule", "compound", "acid", "base", "organic"],
    "mathematics": ["math", "algebra", "calculus", "geometry", "equation", "theorem"],
    "biology": ["biology", "cell", "organism", "evolution", "photosynthesis", "dna"]
}

LOG_PATH = "logs/domain_usage.json"
translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest="en")
        return translated.text
    except Exception:
        return text  # Fallback to original if translation fails

def detect_domain(text, log=True, translate=True, return_scores=False):
    """
    Detects domain of a given text, with optional logging and multilingual support.

    Args:
        text (str): The input text.
        log (bool): Whether to log domain usage count.
        translate (bool): Whether to translate non-English text to English.
        return_scores (bool): Whether to return score breakdown.

    Returns:
        str or (str, dict): Domain or (domain, scores)
    """
    if translate:
        text = translate_to_english(text)
    text = text.lower()

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for word in keywords if word in text)
        scores[domain] = score

    # Confidence thresholding (choose domain with highest score or default to 'general')
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        best_domain = "general"

    if log:
        log_domain_usage(best_domain)

    if return_scores:
        return best_domain, scores
    return best_domain

def log_domain_usage(domain):
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
