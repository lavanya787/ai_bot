# utils/nltk_setup.py

import nltk

def download_nltk_resources():
    resources = [
        "punkt",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words",
        "stopwords"
    ]
    for res in resources:
        try:
            if res == "punkt":
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
