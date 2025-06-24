# retrieval/bm25_retriever.py
from collections import Counter
from math import log
from utils.text_utils import tokenize_text

class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus  # list of dicts: {"query": ..., "response": ...}
        self.k1 = 1.5
        self.b = 0.75
        self.index = []
        self.doc_lens = []
        self.avgdl = 0
        self.df = Counter()
        self.idf = {}
        self.build_index()

    def build_index(self):
        for doc in self.corpus:
            tokens = tokenize_text(doc["query"])
            self.index.append(tokens)
            self.doc_lens.append(len(tokens))
            self.df.update(set(tokens))
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        total_docs = len(self.corpus)
        for word, freq in self.df.items():
            self.idf[word] = log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens, doc_tokens):
        score = 0
        freq = Counter(doc_tokens)
        dl = len(doc_tokens)
        for word in query_tokens:
            if word in self.idf:
                f = freq[word]
                idf = self.idf[word]
                score += idf * ((f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)))
        return score

    def retrieve(self, query, top_k=1):
        query_tokens = tokenize_text(query)
        scored = []
        for i, doc_tokens in enumerate(self.index):
            s = self.score(query_tokens, doc_tokens)
            scored.append((s, i))
        top = sorted(scored, reverse=True)[:top_k]
        return [self.corpus[i]["response"] for _, i in top if _ > 0.1] or ["No relevant info found."]

# Debugging: Confirm file is loaded
print("Successfully loaded bm25_retriever.py")