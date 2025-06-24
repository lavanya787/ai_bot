import streamlit as st
import os
import pickle
import hashlib
import numpy as np
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from utils.document_processor import DocumentProcessor
from file_processing.processor import chunk_text
from utils.domain_detector import detect_domain
from models.train_lstm import train_model  # <-- new fixed import

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_bot")


class ChatBot:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.documents = {}
        self.datasets = {}
        self.models = {}
        self.embeddings = {}

        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def add_document(self, filename: str, file_result: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = hashlib.md5(filename.encode()).hexdigest()

        if file_result["type"] == "structured":
            data = file_result["data"]

            doc_info = {
                "filename": filename,
                "type": "dataset",
                "data": data,
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "processed_at": datetime.now().isoformat(),
                "domain": "generic"
            }

            self.datasets[doc_id] = doc_info
            self.auto_train_models(doc_id)

        else:
            content = file_result["content"]
            chunks = chunk_text(content)

            doc_info = {
                "filename": filename,
                "type": "document",
                "content": content,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "word_count": len(content.split()),
                "processed_at": datetime.now().isoformat()
            }

            if self.embedding_model and chunks:
                try:
                    embeddings = self.embedding_model.encode(chunks)
                    self.embeddings[doc_id] = embeddings
                except Exception as e:
                    logger.error(f"Error creating embeddings: {e}")

            self.documents[doc_id] = doc_info

        return doc_info

    def auto_train_models(self, doc_id: str):
        if doc_id not in self.datasets:
            logger.error(f"Dataset {doc_id} not found for training")
            return

        dataset = self.datasets[doc_id]
        data = dataset["data"]
        filename = dataset["filename"]

        domain = detect_domain(filename) or "general"
        dataset["domain"] = domain

        domain_folder = os.path.join("models", domain)
        os.makedirs(domain_folder, exist_ok=True)
        model_path = os.path.join(domain_folder, f"{filename}_model.pkl")
        file_path = os.path.join(domain_folder, filename)

        if os.path.exists(model_path):
            if 'retrain_confirmed' not in st.session_state:
                st.session_state.retrain_confirmed = False

            if not st.session_state.retrain_confirmed:
                if st.checkbox(f"⚠️ Retrain existing model for {filename}?"):
                    st.session_state.retrain_confirmed = True
                else:
                    st.warning("Training skipped. Check the box to retrain.")
                    return

        if "raw_file" in dataset:
            with open(file_path, "wb") as f:
                f.write(dataset["raw_file"].getbuffer())

        target_column = next((col for col in data.columns if data[col].dtype == 'object'), None)
        if not target_column:
            logger.warning("No target column found.")
            return

        try:
            trained_model, label_encoder = train_model(
                data, target_column,
                epochs=5, batch_size=32, learning_rate=0.001
            )
            with open(model_path, "wb") as f:
                pickle.dump((trained_model, label_encoder), f)

            st.success(f"✅ Model trained and saved to `{model_path}`")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            st.error(f"❌ Training error: {e}")
