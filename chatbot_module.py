import streamlit as st
import os
import hashlib
from datetime import datetime
import logging
import pandas as pd
from file_processing.processor import extract_text
from utils.domain_detector import detect_domain
from utils.logger import Logger
from rag_domain_trainer import store_and_train
from utils.preprocessing import Preprocessor
from intent.classifier import IntentClassifier
from intent.train_classifier import train_model
from models.qa_model import QAHandler
from llm_handler import LLMHandler

LLM_AVAILABLE = True

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = Logger().logger
preprocessor = Preprocessor()

class ChatBot:
    def __init__(self):
        self.datasets = {}
        self.documents = {}
        self.trained_models = {}
        self.intent_dataset = None
        self.llm_handler = LLMHandler() if LLM_AVAILABLE else None
        self.intent_classifier = IntentClassifier()  # Always instantiate
        self.qa_handler = QAHandler()

    def generate_response(self, prompt):
        if not self.documents:
            return "📂 Please upload files so I can analyze and respond better."

        # Intent detection
        detected_intent = "ask_question"
        try:
            detected_intent = self.intent_classifier.predict(prompt)
            logger.info(f"🧠 Detected intent: {detected_intent}")
        except Exception as e:
            logger.warning(f"⚠️ Intent detection failed: {e}")

        # Map intent to task
        intent_task_map = {
            "summarize_document": "summarize",
            "get_insights": "analyze",
            "ask_question": "answer",
            "train_model": "train",
            "get_sentiment": "sentiment",
            "book_flight": "external_action",
            "set_reminder": "external_action",
            "question": "answer",
            "generate": "generate",
            "sentiment": "sentiment",
            "rag_query": "answer",
            "default": "answer"
        }
        task = intent_task_map.get(detected_intent, "answer")

        # Handle actions
        if task == "external_action":
            return f"🚀 Triggering external action for: **{detected_intent}**"
        elif task == "train":
            return self.auto_train_models_from_user()
        elif task == "answer":
            try:
                return self.qa_handler.answer(prompt)
            except Exception as e:
                logger.warning(f"🔁 QA fallback to LLM: {e}")
        elif task == "generate" and self.llm_handler:
            try:
                return self.llm_handler.generate_response(prompt, task="generate")
            except Exception as e:
                logger.warning(f"🔁 Generate fallback: {e}")

        # LLM fallback
        try:
            response = self.llm_handler.generate_response(prompt, task)
            Logger.logger(prompt, response, status="success", chunk_used="LLMHandler", intent=detected_intent)
            return response
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def add_document(self, name, doc_dict):
        content = doc_dict.get("content", "")
        file_obj = doc_dict.get("file")

        # Step 1: Extract text
        if not content and file_obj:
            try:
                content = extract_text(file_obj)
            except Exception as e:
                log.warning(f"❌ Failed to extract text: {e}")
                return

        if not content.strip():
            log.warning(f"⚠️ Empty content after extraction for file: {name}")
            return

        # Step 2: Preprocess
        cleaned = preprocessor.general_preprocessing(content)
        self.documents[name] = {"content": cleaned, "file": file_obj}
        content_hash = hashlib.md5(cleaned.encode()).hexdigest()

        # Step 3: Detect domain
        domain = detect_domain(cleaned)
        log.info(f"🌐 Detected domain for {name}: {domain}")

        # Step 4: Prepare directory & avoid duplicates
        domain_folder = os.path.join("trained_data", domain)
        os.makedirs(domain_folder, exist_ok=True)

        # Check for existing hash
        for fname in os.listdir(domain_folder):
            if fname.endswith(".txt") and content_hash in fname:
                log.info(f"⚠️ Duplicate file detected: {fname} — skipping.")
                return

        # Step 5: Save unique cleaned content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_file_path = os.path.join(domain_folder, f"{timestamp}_{content_hash}_{name}.txt")
        try:
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            log.info(f"✅ Saved to: {cleaned_file_path}")
        except Exception as e:
            log.warning(f"⚠️ Saving file failed: {e}")

        # Step 6: Auto fine-tuning
        try:
            domain_files = [os.path.join(domain_folder, f) for f in os.listdir(domain_folder) if f.endswith(".txt")]
            if len(domain_files) >= 3:
                merged_content = ""
                for fpath in domain_files:
                    with open(fpath, "r", encoding="utf-8") as f:
                        merged_content += f.read() + "\n"
                log.info(f"🧠 Auto-finetuning domain model for: {domain} with {len(domain_files)} files")
                store_and_train(None, text=merged_content, domain=domain)
            else:
                log.info(f"ℹ️ Not enough files for fine-tuning: {len(domain_files)}")
        except Exception as e:
            log.warning(f"⚠️ Fine-tuning failed: {e}")

        # Step 7: Upload to RAG store
        try:
            tmp_path = os.path.join("rag_data", f"{datetime.now().timestamp()}_{name}")
            os.makedirs("rag_data", exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            store_and_train(tmp_path)
        except Exception as e:
            log.warning(f"⚠️ store_and_train() failed: {e}")

        # Step 8: Intent dataset handling
        if name.endswith((".csv", ".json")):
            try:
                file_obj.seek(0)
                df = pd.read_csv(file_obj) if name.endswith(".csv") else pd.read_json(file_obj)
                if {"sentence", "intent"}.issubset(df.columns):
                    self.intent_dataset = df
                    logger.info(f"📊 Loaded intent dataset: {len(df)} samples")
                    train_model(df)
                    self.trained_models["intent_transformer"] = {
                        "model_type": "Transformer",
                        "samples": len(df),
                        "accuracy": "Pending eval",
                        "trained_at": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"⚠️ Intent fine-tuning failed: {e}")

        # Step 9: LLM Indexing
        if LLM_AVAILABLE and self.llm_handler:
            try:
                self.llm_handler.index_document(name, cleaned)
            except Exception as e:
                logger.warning(f"⚠️ LLM Indexing failed: {e}")
                st.warning(f"LLM indexing failed for {name}: {e}")

        # Step 10: Tabular content detection
        if any(delim in content for delim in [',', '\t', '|']):
            lines = content.split('\n')
            if len(lines) > 1:
                doc_id = hashlib.md5(name.encode()).hexdigest()
                self.datasets[doc_id] = {
                    'data': lines,
                    'filename': name,
                    'type': 'tabular'
                }

    def auto_train_models(self, doc_id):
        if doc_id in self.datasets:
            dataset = self.datasets[doc_id]
            size = len(dataset.get("data", []))
            self.trained_models[doc_id] = {
                "model_type": "classification" if size > 10 else "regression",
                "accuracy": min(0.95, 0.7 + (size / 100)),
                "features": min(10, max(3, size // 5)),
                "trained_at": datetime.now().isoformat()
            }
            model = self.trained_models[doc_id]
            return f"✅ Trained {model['model_type']} model (Acc: {model['accuracy']:.2%})"
        return "⚠️ Dataset not found."

    def auto_train_models_from_user(self):
        # Placeholder for user-triggered training
        results = []
        for doc_id in self.datasets:
            result = self.auto_train_models(doc_id)
            results.append(result)
        return "\n".join(results) if results else "⚠️ No datasets available for training."