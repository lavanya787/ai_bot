import os
import mimetypes
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
import time
import hashlib

from model_trainer_batch import train_and_store_model
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

# ---------- Utility Sub-functions ----------

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
    
    return result


def log_and_train(file, result, doc_info):
    if result["type"] == "structured":
        doc_id = hashlib.md5(file.name.encode()).hexdigest()
        if doc_id in st.session_state.chatbot.datasets:
            st.markdown("🤖 Training model...")
            with st.spinner("Training in progress..."):
                start_train = time.time()
                st.session_state.chatbot.datasets[doc_id]["raw_file"] = file
                train_and_store_model(st.session_state.chatbot, doc_id)
                train_time = time.time() - start_train
                st.success(f"✅ Model training completed in {train_time:.2f} seconds")


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

# ---------- Main Controller Function ----------

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


# ---------- Summary Function ----------

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
