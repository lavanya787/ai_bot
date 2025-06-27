# chatbot_module.py 
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
from file_processing.processor import chunk_text, extract_text
from utils.domain_detector import detect_domain
from utils.preprocessing import Preprocessor
from utils.search_google import search_google
from utils.logger import Logger

from retriever.meta_retriever import MetaRetriever

log_manager = Logger()
logger = log_manager.logger

class ChatBot:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.documents = {}
        self.datasets = {}
        self.models = {}
        self.embeddings = {}
        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        from llm_handler import LLMHandler
        self.llm_handler = LLMHandler()
        self.preprocessor = Preprocessor()

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
                "domain": "generic",
                "raw_file": file_result.get("file")
            }

            self.datasets[doc_id] = doc_info
            self.auto_train_models(doc_id)

        else:
            file_obj = file_result.get("file")
            raw_text = extract_text(file_obj)
            cleaned_text = self.preprocessor.general_preprocessing(raw_text)
            chunks = chunk_text(cleaned_text)

            doc_info = {
                "filename": filename,
                "type": "document",
                "content": cleaned_text,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "word_count": len(cleaned_text.split()),
                "processed_at": datetime.now().isoformat()
            }

            if self.embedding_model and chunks:
                try:
                    embeddings = self.embedding_model.encode(chunks)
                    self.embeddings[doc_id] = embeddings
                except Exception as e:
                    logger.error(f"Error creating embeddings: {e}")

            self.documents[doc_id] = doc_info

            # Auto-train RAG domain model
            domain = detect_domain(cleaned_text)
            logger.info(f"Detected domain: {domain}")
            os.makedirs(f"domain_models/{domain}", exist_ok=True)
            file_path = os.path.join(f"domain_models/{domain}", filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # LLM index
            if self.llm_handler:
                self.llm_handler.index_document(filename, cleaned_text)

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

        from models.train_lstm import train_model
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

    def generate_response(self, prompt):
        if not self.documents:
            return "Please upload some documents first."

        if not self.llm_handler:
            return "LLMHandler not available."

        if "search online" in prompt.lower() or "latest" in prompt.lower():
            try:
                web_snippets = search_google(prompt)
                return f"🔍 Here's what I found online:\n\n{web_snippets}"
            except Exception as e:
                return f"Error performing search: {e}"

        try:
            retriever = MetaRetriever()
            all_text = "\n".join(doc["content"] for doc in self.documents.values())
            context, used_retriever = retriever.retrieve_with_best(prompt, all_text, top_k=5)

            enhanced_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
            response = self.llm_handler.generate_response(enhanced_prompt, task="answer")

            Logger.logger(
                query=prompt,
                response=response,
                status="success",
                chunk_used=used_retriever
            )
            return response

        except Exception as e:
            return f"Error: {str(e)}"
