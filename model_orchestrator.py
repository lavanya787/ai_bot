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
from training.model_runner import train_all_models

# Load vocabulary for models
def load_vocab_for_models():
    from models.qa_model import load_vocab
    return load_vocab()

@st.cache_resource
def load_models():
    models = {}
    print("Loading QAHandler...")
    models["qa"] = QAHandler()
    vocab = load_vocab_for_models()
    vocab_size = len(vocab)
    print("Loading SentimentClassifier...")
    models["sentiment"] = SentimentClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    print("Loading LSTMGenerator...")
    models["lstm"] = LSTMGenerator(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    print("Loading TransformerGenerator...")
    models["transformer"] = TransformerGenerator(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    print("Loading RAGModel...")
    models["rag"] = RAGModel(vocab_size=vocab_size)
    return models

models = load_models()

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
        return models["lstm"].generate(text)

def model_training_ui():
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training all models..."):
            results = train_all_models()
        st.success("✅ Training completed!")
        st.json(results)