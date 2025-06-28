import os
import mimetypes
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
import time
import hashlib
import torch
import numpy as np
import json
import tempfile

from models.qa_model import get_embedding
from vector_db.faiss_indexer import VectorDB
from tokenizer.vocab_builder import build_vocab
from models.sentiment_model import SentimentClassifier
from scripts.auto_domain_mover import move_file_to_domain_folder

from file_processing.processor import (
    extract_text,
    detect_language,
    translate_to_english,
    chunk_text
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_bot")

# ----------------- Utility -----------------
def compute_file_hash(file):
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

# ----------------- Auto Embed (Batch-Aware) -----------------
def auto_embed_and_train(result):
    chunks = result.get("chunks", [])
    if not chunks:
        result["embedding_count"] = 0
        result["trained"] = False
        return result

    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []

    st.session_state.all_chunks.extend(chunks)
    return result

# ----------------- Train Trigger -----------------
def train_on_all_documents():
    chunks = st.session_state.get("all_chunks", [])
    if not chunks:
        st.warning("No documents to train on.")
        return

    st.info(f"🔍 Training on {len(chunks)} total chunks...")

    vocab = build_vocab(chunks)
    tokenizer_path = "tokenizer/vocab.json"
    os.makedirs("tokenizer", exist_ok=True)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    tokenizer_size = len(vocab)
    embeddings = [get_embedding(c) for c in chunks]
    embeddings = np.array(embeddings).astype("float32")

    os.makedirs("embedding_store", exist_ok=True)
    np.save("embedding_store/embeddings.npy", embeddings)

    db = VectorDB(dim=embeddings.shape[1])
    db.add(embeddings, chunks)

    model = SentimentClassifier(vocab_size=tokenizer_size, embedding_dim=128, hidden_dim=256)
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/sentiment_model.pt")

    st.session_state.model_status = {
        "trained": True,
        "device": "cpu",
        "documents": len(chunks),
        "tokenizer_vocab": tokenizer_size,
        "embeddings": len(embeddings)
    }

    st.success("✅ Multi-document training completed!")

# ----------------- File Content Processing -----------------
def extract_and_process(file, ext):
    result = {
        "filename": file.name.lower(),
        "type": None,
        "content": "",
        "data": None,
        "chunks": [],
        "lang": "en",
        "error": None
    }

    try:
        text = extract_text(file)
        if not text or text.startswith("[ERROR]"):
            raise ValueError("Extraction failed or empty content.")

        lang = detect_language(text)
        if lang != "en":
            text = translate_to_english(text)

        result.update({
            "type": "structured" if ext in [".csv", ".xls", ".xlsx", ".json"] else "document",
            "content": text,
            "lang": lang,
            "chunks": chunk_text(text)
        })

    except Exception as e:
        result["error"] = f"❌ Extraction failed: {e}"

    return auto_embed_and_train(result)

# ----------------- File Metadata -----------------
def build_file_metadata(file, result, doc_info, file_hash=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = {
        "filename": file.name,
        "processed_at": timestamp,
        "hash": file_hash or compute_file_hash(file),
    }

    if result["type"] == "structured":
        meta.update({
            "type": "dataset",
            "shape": result.get("data", pd.DataFrame()).shape,
            "columns": result.get("data", pd.DataFrame()).shape[1],
            "domain": doc_info.get("domain", "generic")
        })
    else:
        meta.update({
            "type": "document",
            "size": len(str(result.get("content", ""))),
            "chunks": doc_info.get("chunk_count", 0),
        })

    return meta

# ----------------- Upload Handler -----------------
def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    processed_hashes = {meta["hash"] for meta in st.session_state.processed_files if "hash" in meta}
    success_count = 0

    for file in uploaded_files:
        file_hash = compute_file_hash(file)

        if file_hash in processed_hashes:
            st.info(f"🔁 Skipping `{file.name}` — already processed.")
            continue

        ext = os.path.splitext(file.name.lower())[1]
        st.markdown(f"### 📄 Processing: `{file.name}`")

        try:
            with st.spinner("📥 Extracting content..."):
                start = time.time()
                result = extract_and_process(file, ext)
                elapsed = time.time() - start
                st.success(f"✅ Extracted in {elapsed:.2f} sec")

            if result.get("error"):
                st.error(f"❌ Error: {result['error']}")
                continue

            if len(str(result.get("content", "")).strip()) < 10:
                st.warning(f"⚠️ File '{file.name}' contains insufficient content.")
                continue

            result["raw_file"] = file
            doc_info = st.session_state.chatbot.add_document(file.name, result)
            file_info = build_file_metadata(file, result, doc_info, file_hash)
            st.session_state.processed_files.append(file_info)
            success_count += 1

            # ✅ Save to temp and move to domain folder
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            move_result = move_file_to_domain_folder(tmp_path)
            os.remove(tmp_path)

            st.info(f"📂 Moved to domain: {move_result['detected_domain']}")

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            logger.error(f"Error processing {file.name}: {e}")

        st.markdown("---")

    if success_count > 0:
        st.success(f"✅ Successfully processed {success_count} file(s)!")

# ----------------- Optional Manual Log + Train -----------------
def log_and_train(file_path):
    text = extract_text(file_path)
    chunks = chunk_text(text)
    auto_embed_and_train({"chunks": chunks})
    move_result = move_file_to_domain_folder(file_path)
    print(f"[INFO] Moved to: {move_result['new_path']} (Domain: {move_result['detected_domain']})")

# ----------------- Button Trigger -----------------
if st.button("🚀 Train on All Uploaded Documents"):
    train_on_all_documents()