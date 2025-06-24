import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def faiss_indexer(chunks, docname):
    os.makedirs(f"faiss_logs/{docname}", exist_ok=True)

    # Embed chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index + chunks
    faiss.write_index(index, f"faiss_logs/{docname}/faiss.index")
    with open(f"faiss_logs/{docname}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
