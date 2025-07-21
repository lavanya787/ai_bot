# utils/nltk_setup.py

import nltk

def download_nltk_resources():
    resources = [
        "punkt",
        "averaged_perceptron_tagger",
        "stopwords",
        "maxent_ne_chunker",
        "words"
    ]

    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "words":
                nltk.data.find("corpora/words")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
