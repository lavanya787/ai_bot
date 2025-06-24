from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.bm25_retriever import BM25Retriever

class MetaRetriever:
    def __init__(self, retrievers=None, documents=None):
        if documents is None:
            documents = []
        if retrievers is None:
            retrievers = {
                "bm25": BM25Retriever(documents),
                "tfidf": TFIDFRetriever(documents)
            }
        self.retrievers = retrievers

    def retrieve(self, query, top_k=3):
        results = []
        for name, retriever in self.retrievers.items():
            try:
                retrieved = retriever.retrieve(query, top_k)
                results.extend(retrieved)
            except Exception as e:
                print(f"Retriever {name} failed: {e}")
        return list(set(results))[:top_k]


# Debugging: Confirm file is loaded
print("Successfully loaded meta_retriever.py")