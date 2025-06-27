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

from models.qa_model import get_embedding
from vector_db.faiss_indexer import VectorDB
from tokenizer.vocab_builder import build_vocab
from models.sentiment_model import SentimentClassifier

from file_processing.processor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image,
    extract_text_from_txt,
    extract_text_from_csv,
    extract_text_from_json,
    detect_language,
    translate_to_english,
    chunk_text
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_bot")

# ---------- Auto Embed, Vocab, Dummy Train ----------

def auto_embed_and_train(result):
    chunks = result.get("chunks", [])
    print(f"🔍 Extracted {len(chunks)} chunks")

    # Show first few chunk samples
    for i, c in enumerate(chunks[:3]):
        print(f"[Chunk {i+1}] {c[:150].strip()}...")

    if not chunks:
        print("⚠️ No chunks found, skipping embedding.")
        result["embedding_count"] = 0
        result["trained"] = False
        return result

    # === Build vocab
    vocab = build_vocab(chunks)
    tokenizer_path = "tokenizer/vocab.json"
    os.makedirs("tokenizer", exist_ok=True)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    tokenizer_size = len(vocab)
    print(f"📘 Vocab built with {tokenizer_size} tokens.")

    # === Generate embeddings
    embeddings = [get_embedding(c) for c in chunks]
    embeddings = np.array(embeddings).astype("float32")
    print(f"📦 Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    # === Save embeddings
    os.makedirs("embedding_store", exist_ok=True)
    np.save("embedding_store/embeddings.npy", embeddings)

    # === Add to FAISS index (optional: persist later)
    db = VectorDB(dim=embeddings.shape[1])
    db.add(embeddings, chunks)

    # === Dummy train + save model
    model = SentimentClassifier(vocab_size=tokenizer_size, embedding_dim=128, hidden_dim=256)
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/sentiment_model.pt")

    # === Update UI model status
    st.session_state.model_status = {
        "trained": True,
        "device": "cpu",
        "documents": st.session_state.get("documents_count", 1),
        "tokenizer_vocab": tokenizer_size,
        "embeddings": len(embeddings)
    }

    # Attach metrics to result
    result["trained"] = True
    result["tokenizer_vocab"] = tokenizer_size
    result["embedding_count"] = len(embeddings)
    return result


# ---------- Main Extraction Logic ----------

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
        if ext in [".csv", ".xls", ".xlsx"]:
            df = pd.read_csv(file)
            text = extract_text_from_csv(file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_to_english(text)
            result.update({
                "type": "structured",
                "data": df,
                "content": text,
                "lang": lang,
                "chunks": chunk_text(text)
            })

        elif ext in [".txt", ".md"]:
            text = extract_text_from_txt(file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_to_english(text)
            result.update({
                "type": "text",
                "content": text,
                "lang": lang,
                "chunks": chunk_text(text)
            })

        elif ext == ".pdf":
            text = extract_text_from_pdf(file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_to_english(text)
            result.update({
                "type": "document",
                "content": text,
                "lang": lang,
                "chunks": chunk_text(text)
            })

        elif ext == ".docx":
            text = extract_text_from_docx(file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_to_english(text)
            result.update({
                "type": "document",
                "content": text,
                "lang": lang,
                "chunks": chunk_text(text)
            })

        elif ext in [".png", ".jpg", ".jpeg"]:
            text = extract_text_from_image(file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_to_english(text)
            result.update({
                "type": "ocr",
                "content": text,
                "lang": lang,
                "chunks": chunk_text(text)
            })

        elif ext == ".json":
            text = extract_text_from_json(file)
            result.update({
                "type": "structured",
                "content": text
            })

        else:
            result["error"] = f"❌ Unsupported file format: {ext}"

    except Exception as e:
        result["error"] = f"❌ Extraction failed: {e}"

    result = auto_embed_and_train(result)
    return result

# ---------- Used for Local or CLI Calls ----------

def extract_text_from_file(file):
    """
    Works with either Streamlit-uploaded or file-like object.
    """
    ext = os.path.splitext(file.name.lower())[1]
    result = extract_and_process(file, ext)

    if result.get("error"):
        raise ValueError(result["error"])

    return result.get("content", "")

# ---------- Streamlit Use Only ----------

def log_and_train(file, result, doc_info):
    if result["type"] == "structured":
        doc_id = hashlib.md5(file.name.encode()).hexdigest()
        if doc_id in st.session_state.chatbot.datasets:
            st.markdown("🤖 Training model...")
            with st.spinner("Training in progress..."):
                st.session_state.chatbot.datasets[doc_id]["raw_file"] = file

def build_file_metadata(file, result, doc_info):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if result["type"] == "structured":
        return {
            "filename": file.name,
            "type": "dataset",
            "shape": result["data"].shape,
            "columns": len(result["data"].columns),
            "domain": doc_info.get("domain", "generic"),
            "processed_at": timestamp
        }
    else:
        return {
            "filename": file.name,
            "type": "document",
            "size": len(str(result.get("content", ""))),
            "chunks": doc_info.get("chunk_count", 0),
            "processed_at": timestamp
        }

# ---------- Main Batch Handler ----------

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return

    success_count = 0

    for file in uploaded_files:
        name = file.name.lower()
        ext = os.path.splitext(name)[1]
        mime, _ = mimetypes.guess_type(name)

        st.markdown(f"### 📄 Processing: `{file.name}`")

        try:
            with st.spinner("📥 Extracting content..."):
                start = time.time()
                result = extract_and_process(file, ext)
                elapsed = time.time() - start
                st.success(f"✅ Content extracted in {elapsed:.2f} seconds")

            if result.get("error"):
                st.error(f"❌ Error: {result['error']}")
                continue

            if len(str(result.get("content", "")).strip()) < 10:
                st.warning(f"⚠️ File '{file.name}' contains insufficient content.")
                continue

            result["raw_file"] = file
            doc_info = st.session_state.chatbot.add_document(file.name, result)

            log_and_train(file, result, doc_info)

            file_info = build_file_metadata(file, result, doc_info)
            st.session_state.processed_files.append(file_info)
            success_count += 1

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            logger.error(f"Error processing {file.name}: {e}")

        st.markdown("---")

    if success_count > 0:
        st.success(f"✅ Successfully processed {success_count} file(s)!")

# ---------- Summary Utility ----------

def generate_summary(result):
    if not isinstance(result, dict):
        return "❌ Invalid result structure."

    if 'error' in result:
        return f"❌ Error: {result['error']}"

    if result.get('type') == 'dataset':
        data = result.get('data')
        return f"""🗂️ **Dataset Summary**  
- Shape: {data.shape}  
- Columns: {', '.join(data.columns)}  
- Domain: {result.get('domain', 'unknown')}"""

    elif result.get('type') == 'document':
        return f"""📄 **Document Summary**  
- Words: {result.get('word_count')}  
- Chunks: {result.get('chunk_count')}"""

    return "⚠️ Unknown document type"
