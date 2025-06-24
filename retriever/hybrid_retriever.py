import re
from collections import Counter
import math

class HybridRetriever:
    def __init__(self, chunks, weight_bm25=0.7, weight_jaccard=0.3):
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]
        self.df = self._compute_df(self.tokenized_chunks)
        self.avgdl = sum(len(doc) for doc in self.tokenized_chunks) / len(self.tokenized_chunks)
        self.k1 = 1.5
        self.b = 0.75
        self.weight_bm25 = weight_bm25
        self.weight_jaccard = weight_jaccard

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _compute_df(self, docs):
        df = {}
        for doc in docs:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        return df

    def _bm25_score(self, query_tokens, doc_tokens):
        score = 0
        freqs = Counter(doc_tokens)
        N = len(self.chunks)
        doc_len = len(doc_tokens)

        for term in query_tokens:
            if term in freqs:
                df = self.df.get(term, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * tf * (self.k1 + 1) / denom
        return score

    def _jaccard_similarity(self, query_tokens, doc_tokens):
        set_q = set(query_tokens)
        set_d = set(doc_tokens)
        intersection = set_q & set_d
        union = set_q | set_d
        return len(intersection) / len(union) if union else 0

    def retrieve(self, query, top_k=3):
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.tokenized_chunks):
            bm25 = self._bm25_score(query_tokens, doc_tokens)
            jaccard = self._jaccard_similarity(query_tokens, doc_tokens)
            combined = (self.weight_bm25 * bm25) + (self.weight_jaccard * jaccard)
            scores.append((idx, combined))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        return [self.chunks[i] for i in top_indices]
