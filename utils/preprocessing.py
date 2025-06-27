import re
import chardet
import pandas as pd
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from langdetect import detect

# Ensure required NLTK resources are available
for res in ["punkt", "wordnet", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res)


class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {}
        self.supported_langs = ["en", "hi", "fr"]

        # Map language codes to valid NLTK stopword sets
        lang_map = {
            "en": "english",
            "hi": "english",   # fallback since 'hi' not in nltk
            "fr": "french"
        }

        for lang in self.supported_langs:
            try:
                mapped_lang = lang_map.get(lang, "english")
                self.stop_words[lang] = set(stopwords.words(mapped_lang))
            except LookupError:
                try:
                    nltk.download("stopwords")
                    self.stop_words[lang] = set(stopwords.words(mapped_lang))
                except Exception:
                    print(f"⚠️ Warning: Stopwords for '{lang}' not available after retry.")
                    self.stop_words[lang] = set()
            except Exception as e:
                print(f"⚠️ Unexpected error loading stopwords for '{lang}': {e}")
                self.stop_words[lang] = set()

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "en"

    def translate_to_english(self, text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            return text

    def general_preprocessing(self, text):
        if isinstance(text, bytes):
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding or 'utf-8', errors='replace')

        lang = self.detect_language(text)
        if lang != "en":
            text = self.translate_to_english(text)
            lang = "en"

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        text = text.lower()

        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words.get(lang, set())]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def preprocess_pdf(self, file):
        try:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            return self.general_preprocessing(text)
        except Exception as e:
            return f"[PDF Error] {e}"

    def preprocess_csv(self, file):
        try:
            df = pd.read_csv(file).fillna("")
            df.columns = [c.lower().strip() for c in df.columns]
            return self.general_preprocessing(df.to_string(index=False))
        except Exception as e:
            return f"[CSV Error] {e}"

    def preprocess_excel(self, file):
        try:
            df = pd.read_excel(file).fillna("")
            df.columns = [c.lower().strip() for c in df.columns]
            return self.general_preprocessing(df.to_string(index=False))
        except Exception as e:
            return f"[Excel Error] {e}"

    def preprocess_text(self, file):
        try:
            text = file.read().decode("utf-8", errors="replace")
            return self.general_preprocessing(text)
        except Exception as e:
            return f"[Text Error] {e}"
