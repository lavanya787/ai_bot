import os
import sys
import pickle
import json
import numpy as np
from file_processing.processor import extract_text_from_file
from file_processing.chunker import chunk_text
from models.qa_model import get_embedding
from vector_db.faiss_indexer import VectorDB

# Output storage
CHUNK_PKL = "embedding_store/chunks.pkl"
META_FILE = "embedding_store/meta.json"
EMBED_FILE = "embedding_store/embeddings.npy"

def file_to_vectors(file_path, chunk_size=300, overlap=50, dim=384):
    # 1. Extract text
    print(f"📂 Extracting text from: {file_path}")
    text = extract_text_from_file(file_path)
    if not text:
        raise ValueError("❌ No text extracted from file.")

    # 2. Chunking
    print("🔪 Chunking text...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    with open(CHUNK_PKL, "wb") as f:
        pickle.dump(chunks, f)

    # 3. Embedding
    print(f"🧠 Embedding {len(chunks)} chunks...")
    embeddings = []
    metadata = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embeddings.append(emb)
        metadata.append({
            "chunk_id": i,
            "text": chunk,
            "source_file": os.path.basename(file_path)
        })

    embeddings = np.array(embeddings).astype(np.float32)
    np.save(EMBED_FILE, embeddings)
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Index in FAISS
    print("📦 Building FAISS index...")
    vector_db = VectorDB(dim=dim)
    vector_db.add(embeddings, metadata)
    print("✅ Done. Chunks indexed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/file_to_vector.py <your_file>")
    else:
        file_to_vectors(sys.argv[1])
