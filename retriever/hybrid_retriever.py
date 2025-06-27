from retriever.bm25_retriever import BM25Retriever
from retriever.tfidf_retriever import TFIDFRetriever
from vector_db.faiss_indexer import VectorDB
import numpy as np
import hashlib
import json
import os

CACHE_FILE = "retriever/cache/retrieval_cache.json"
os.makedirs("retriever/cache", exist_ok=True)


def _query_hash(query):
    return hashlib.md5(query.encode()).hexdigest()

def cache_result(query, results):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    cache[_query_hash(query)] = results
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def get_cached(query):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        return cache.get(_query_hash(query))
    return None
    
def deduplicate_results(results):
    seen = set()
    deduped = []
    for r in results:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped

def retrieve_context(query, k=5):
    # 1. BM25
    bm25 = BM25Retriever()
    bm25_results = bm25.retrieve(query, top_k=k)

    # 2. TF-IDF
    tfidf = TFIDFRetriever()
    tfidf_results = tfidf.retrieve(query, top_k=k)

    # 3. Vector-based (FAISS)
    vector_db = VectorDB(dim=384)  # or 768 or whatever your embedding size is
    from models.qa_model import get_embedding  # you should define this
    query_vec = get_embedding(query).reshape(1, -1).astype(np.float32)
    vector_results = vector_db.search(query_vec, k=k)

    # Combine and deduplicate
    all_results = bm25_results + tfidf_results + vector_results
    hybrid_context = deduplicate_results(all_results)[:k]

    return hybrid_context
