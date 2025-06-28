# model_orchestrator.py

import os
import torch
import streamlit as st

from models.qa_model import QAHandler
from models.sentiment_model import SentimentClassifier
from models.lstm_generator import LSTMGenerator
from models.transformer_generator import TransformerGenerator
from intent.classifier import predict_intent
from llm_components.RAGModel import RAGModel
from file_processing.auto_pipeline import process_uploaded_files
from training.model_runner import train_all_models  # Assuming this integrates LLM, RAG, retriever, etc.

# Load all models globally (use caching in production)

@st.cache_resource
def load_models():
    models = {}
    models["qa"] = QAHandler()
    models["sentiment"] = SentimentClassifier()
    models["lstm"] = LSTMGenerator()
    models["transformer"] = TransformerGenerator()
    models["rag"] = RAGModel()  # Includes retriever
    return models

models = load_models()


# Process query based on intent
def handle_query(text):
    intent = predict_intent(text)

    if intent == "question":
        return models["qa"].answer(text)

    elif intent == "generate":
        return models["transformer"].generate(text)

    elif intent == "sentiment":
        return models["sentiment"].predict(text)

    elif intent == "rag_query":
        return models["rag"].generate(text)

    else:
        return models["lstm"].generate(text)  # fallback


def model_training_ui():
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training all models..."):
            results = train_all_models()  # Call the actual unified model runner
        st.success("✅ Training completed!")
        st.json(results)
