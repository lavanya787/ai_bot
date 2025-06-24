import re

class CustomRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]

    def _tokenize(self, text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _jaccard_similarity(self, query_tokens, doc_tokens):
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        return len(intersection) / len(union) if union else 0

    def retrieve(self, query, top_k=3):
        query_tokens = self._tokenize(query)
        scores = []
        for idx, doc_tokens in enumerate(self.tokenized_chunks):
            score = self._jaccard_similarity(query_tokens, doc_tokens)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]
        return [self.chunks[i] for i in top_indices]
