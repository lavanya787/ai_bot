# model_orchestrator.py

import streamlit as st
from models.qa_model import QAHandler
from models.sentiment_model import SentimentClassifier
from models.lstm_generator import LSTMGenerator
from models.transformer_generator import TransformerGenerator
from intent.classifier import predict_intent
from llm_components.RAGModel import RAGModel
from training.model_runner import train_all_models
from models.qa_model import load_vocab


class ModelOrchestrator:
    def __init__(self):
        self.models = {}
        self.vocab = load_vocab()
        self.vocab_size = len(self.vocab)
        self.load_all_models()

    def load_all_models(self):
        self.models["qa"] = QAHandler()
        self.models["sentiment"] = SentimentClassifier(
            vocab_size=self.vocab_size, embedding_dim=128, hidden_dim=256
        )
        self.models["lstm"] = LSTMGenerator(
            vocab_size=self.vocab_size, embedding_dim=128, hidden_dim=256
        )
        self.models["transformer"] = TransformerGenerator(
            vocab_size=self.vocab_size, embedding_dim=128, hidden_dim=256
        )
        self.models["rag"] = RAGModel(vocab_size=self.vocab_size)

    def route_intent(self, text: str) -> str:
        return predict_intent(text)

    def respond(self, text: str) -> str:
        intent = self.route_intent(text)

        if intent == "question":
            return self.models["qa"].answer(text)

        elif intent == "generate":
            return self.models["transformer"].generate(text)

        elif intent == "sentiment":
            return self.models["sentiment"].predict(text)

        elif intent == "rag_query":
            return self.models["rag"].generate(text)

        else:
            # Fallback to LSTM-based generation
            return self.models["lstm"].generate(text)

    def train_models(self):
        return train_all_models()
    
    def model_training_ui():
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training all models..."):
                results = train_all_models()
            st.success("✅ Training completed!")
            st.json(results)
