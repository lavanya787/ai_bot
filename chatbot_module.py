import streamlit as st
import os
import json
import hashlib
import logging
from datetime import datetime
import pandas as pd
from file_processing.processor import extract_text
from model_orchestrator import ModelOrchestrator
from utils.logger import Logger
from rag_domain_trainer import store_and_train
from utils.preprocessing import Preprocessor
from intent.classifier import IntentClassifier
from intent.train_classifier import train_model
from models.qa_model import QAHandler
from llm_handler import LLMHandler
from typing import List, Dict, Optional
import time
from datetime import datetime

LLM_AVAILABLE = True
# At the top of chatbot_module.py
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear existing handlers

# File handler (UTF-8)
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

log = logger  # Replace Logger().logger

preprocessor = Preprocessor()

class ChatBot:
    def __init__(self):
        self.datasets = {}
        self.documents = {}
        self.trained_models = {}
        self.intent_dataset = None
        self.llm_handler = LLMHandler() if LLM_AVAILABLE else None
        self.intent_classifier = IntentClassifier()
        self.qa_handler = QAHandler()
        self.trained_models_path = "trained_models.json"
        logger.info("ChatBot initialized")

    def generate_response(self, prompt: str, task: Optional[str] = None) -> str:
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Check if documents are loaded
        if not self.documents:
            logger.warning("No documents loaded")
            return "Hello! Please upload some files first so I can analyze them and provide better responses."
        
        if not LLM_AVAILABLE or not self.llm_handler:
            logger.error("LLMHandler not available")
            return "LLMHandler is not available. Please ensure llm_handler.py is installed."

        try:
            # Determine task based on prompt content or explicit task
            if not task:
                prompt_lower = prompt.lower()
                if "analyze" in prompt_lower or "data" in prompt_lower:
                    task = "analyze"
                elif "summary" in prompt_lower or "summarize" in prompt_lower:
                    task = "summarize"
                elif "help" in prompt_lower:
                    return self._get_help_response()
                else:
                    task = "answer"
            logger.info(f"Selected task: {task}")

            # Use intent classifier to determine if prompt matches a trained model
            intent = self.intent_classifier.predict(prompt)
            logger.info(f"Predicted intent: {intent}")

            # Check if there's a trained model for the relevant dataset
            relevant_doc_id = None
            for doc_id, dataset in self.datasets.items():
                if dataset['filename'] in self.documents:
                    relevant_doc_id = doc_id
                    break

            if relevant_doc_id and relevant_doc_id in self.trained_models:
                model_info = self.trained_models[relevant_doc_id]
                logger.info(f"Using trained model for doc_id: {relevant_doc_id}, model_type: {model_info['model_type']}")

                if model_info['model_type'] == "classification":
                    # Example: Use trained classification model to predict a label
                    prediction = self._apply_classification_model(prompt, relevant_doc_id)
                    response = f"Classification result: {prediction}"
                elif model_info['model_type'] == "regression":
                    # Example: Use trained regression model to predict a value
                    prediction = self._apply_regression_model(prompt, relevant_doc_id)
                    response = f"Regression result: {prediction:.2f}"
                else:
                    # Fallback to LLM if model type is unknown
                    response = self.llm_handler.generate_response(prompt, task)
            else:
                # No trained model; use LLM
                logger.info("No trained model found; using LLMHandler")
                response = self.llm_handler.generate_response(prompt, task)

            logger.info(f"Response generated: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return f"Error generating response: {str(e)}. Try rephrasing or uploading more documents."

    def _apply_classification_model(self, prompt: str, doc_id: str) -> str:
        """Apply a trained classification model to the prompt."""
        # Placeholder: Implement actual model inference here
        # For example, load the model from self.trained_models[doc_id] and predict
        dataset = self.datasets.get(doc_id, {}).get('data', [])
        if not dataset:
            return "No dataset available for classification."
        
        # Simulate classification (replace with actual model inference)
        return "Positive"  # Example output; replace with real model prediction

    def _apply_regression_model(self, prompt: str, doc_id: str) -> float:
        """Apply a trained regression model to the prompt."""
        # Placeholder: Implement actual model inference here
        dataset = self.datasets.get(doc_id, {}).get('data', [])
        if not dataset:
            return 0.0
        
        # Simulate regression (replace with actual model inference)
        return 0.85  # Example output; replace with real model prediction

    def add_document(self, name: str, doc_dict: Dict):
        logger.info(f"Adding document: {name}")
        self.documents[name] = doc_dict
        content = doc_dict.get("content", "")
        logger.info(f"Document content length: {len(content)}")
               
        # Store tabular data for potential model training
        if any(delimiter in content for delimiter in [',', '\t', '|']):
            lines = content.split('\n')
            if len(lines) > 1:
                doc_id = hashlib.md5(name.encode()).hexdigest()
                self.datasets[doc_id] = {
                    'data': lines,
                    'filename': name,
                    'type': 'tabular'
                }
                # Automatically train a model for this dataset
                self.auto_train_models(doc_id)

    def load_saved_model(self, model_version: str) -> str:
        try:
            with open(self.trained_models_path, "r") as f:
                models = json.load(f)

            if model_version not in models:
                return f"❌ Model version `{model_version}` not found."

            model_info = models[model_version]
            vectorstore_path = model_info["vectorstore_path"]

            self.llm_handler.load_index(vectorstore_path)
            self.documents[model_info["doc_name"]] = {
                "content": "",
                "metadata": model_info.get("metadata", {})
            }
            # Load trained model metadata into self.trained_models
            doc_id = hashlib.md5(model_info["doc_name"].encode()).hexdigest()
            self.trained_models[doc_id] = {
                "model_type": model_info.get("model_type", "classification"),
                "accuracy": model_info.get("accuracy", 0.0),
                "features": model_info.get("features", 0),
                "trained_at": model_info.get("trained_at", datetime.now().isoformat())
            }
            log.info(f"✅ Loaded model `{model_version}` and retriever")
            return f"✅ Model `{model_version}` loaded successfully"
        except Exception as e:
            log.error(f"❌ Failed loading model {model_version}: {e}")
            return f"❌ Failed loading model: {e}"

    def auto_train_models(self, doc_id: str) -> str:
        log = logging.getLogger(__name__)
        if doc_id not in self.datasets:
            log.error(f"Dataset not found for {doc_id}")
            return "Dataset not found."

        dataset = self.datasets[doc_id]
        data = dataset.get("data", [])
        doc_name = dataset.get("filename", "")
        content = self.documents.get(doc_name, {}).get("content", "")
        size = len(data)

        try:
            model_type = "classification" if size > 10 else "regression"
            log.info(f"Starting {model_type} training for doc `{doc_name}` with {size} lines")

            start_time = time.time()  # Fix: Use time.time()

            # Preprocessing
            log.info("Preprocessing data...")
            t0 = time.time()  # Fix: Use time.time()
            processed_data, metadata = preprocessor.preprocess_file(content)  # Expect tuple (text, metadata)
            log.info(f"Processed data sample: {processed_data[:100]}")
            log.info(f"Preprocessing metadata: {metadata}")
            # Convert processed_data to DataFrame
            processed_data = pd.DataFrame({"text": processed_data.split("\n")})
            if processed_data.empty:
                log.error(f"Preprocessing returned empty DataFrame for {doc_name}")
                return f"Failed to preprocess document: empty data"
            log.info(f"Preprocessing done in {time.time() - t0:.2f} seconds")  # Fix: Use time.time()

            # Training
            log.info("Training model...")
            t1 = time.time()  # Fix: Use time.time()
            if model_type == "classification":
                model, accuracy = train_model(processed_data)
            else:
                log.warning(f"Regression not supported; falling back to classification")
                model, accuracy = train_model(processed_data)  # Replace with regression function if available
            log.info(f"Training done in {time.time() - t1:.2f} seconds")  # Fix: Use time.time()

            self.trained_models[doc_id] = {
                "model_type": model_type,
                "accuracy": accuracy,
                "features": len(processed_data.columns),
                "trained_at": datetime.now().isoformat(),
                "model": model,
                "metadata": metadata
            }

            total_time = time.time() - start_time  # Fix: Use time.time()
            log.info(f"Trained {model_type} model for `{doc_name}` in {total_time:.2f} seconds")

            # Indexing
            if LLM_AVAILABLE and self.llm_handler and content:
                log.info("Indexing document after training...")
                t2 = time.time()  # Fix: Use time.time()
                success = self.llm_handler.index_documents([(doc_name, content)])
                if success:
                    log.info(f"Indexed document `{doc_name}` in {time.time() - t2:.2f} seconds")  # Fix: Use time.time()
                else:
                    log.warning(f"Failed to index `{doc_name}`")

            return f"Trained {model_type} model (Acc: {accuracy:.2%})"
        except Exception as e:
            log.error(f"Failed to train model for {doc_id}: {str(e)}")
            return f"Failed to train model: {str(e)}"
            
    def _get_help_response(self) -> str:
        return (
        "I can help you with the following tasks:\n"
        "- **Answer**: Ask questions about uploaded documents.\n"
        "- **Analyze**: Perform data analysis on tabular data.\n"
        "- **Summarize**: Generate summaries of documents.\n"
        "Please upload documents or specify a task in your prompt."
        )