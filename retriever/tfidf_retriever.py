from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDFRetriever:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.documents = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = (self.doc_vectors * query_vector.T).toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

# Debugging: Confirm file is loaded
print("Successfully loaded tfidf_retriever.py")