import re
import pandas as pd
import PyPDF2
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def is_valid_uuid(val):
        pattern = r'^[0-9a-fA-F\-]{36}$'
        return re.fullmatch(pattern, val) is not None

    def general_preprocessing(self, text):
        if isinstance(text, bytes):
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding, errors='replace')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def preprocess_pdf(self, file):
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return self.general_preprocessing(text)

    def preprocess_excel(self, file):
        df = pd.read_excel(file).fillna('').dropna(how='all').dropna(axis=1, how='all')
        df.columns = [c.lower().strip() for c in df.columns]
        return self.general_preprocessing(df.to_string(index=False))

    def preprocess_csv(self, file):
        df = pd.read_csv(file).fillna('')
        df.columns = [c.lower().strip() for c in df.columns]
        return self.general_preprocessing(df.to_string(index=False))

    def preprocess_text(self, file):
        text = file.read().decode('utf-8', errors='replace')
        return self.general_preprocessing(text)
