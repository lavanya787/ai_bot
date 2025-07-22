import logging
import streamlit as st
import os
import tempfile
import json
import hashlib
import requests
import sys
import pandas as pd
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import traceback
from chatbot_module import ChatBot
from llm_handler import LLMHandler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pickle
import random
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.preprocessing import Preprocessor
from utils.domain_detector import detect_domain
from utils.visualizer import visualize_model_performance, load_training_log
from evaluation.evaluate import calculate_dynamic_accuracy, calculate_processing_speed, calculate_memory_usage, get_model_accuracy
from utils.nltk_setup import download_nltk_resources

download_nltk_resources()

# Setup
#sys.stdout.reconfigure(encoding='utf-8')
#logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')  # Changed to WARNING
#logger = logging.getLogger(__name__)
# Setup
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Supabase setup
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

SUPABASE_URL = "https://swrhcsfuorjqszqoohfb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3cmhjc2Z1b3JqcXN6cW9vaGZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MDkxODA3MCwiZXhwIjoyMDY2NDk0MDcwfQ.aYmADn3cpUhPlNSUiKlb-EveEyyE7-8FgYVq7L4A2OA"

supabase = None
if SUPABASE_AVAILABLE:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}", exc_info=True)
        st.error(f"Failed to connect to database: {e}")

# Safe imports
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# nltk
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Only download if not already present
    def download_nltk_resource(resource):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1], quiet=True)

    download_nltk_resource("tokenizers/punkt")
    download_nltk_resource("taggers/averaged_perceptron_tagger")
    download_nltk_resource("chunkers/maxent_ne_chunker")
    download_nltk_resource("corpora/words")
    download_nltk_resource("corpora/stopwords")

    NLTK_AVAILABLE = True
    logger.info("NLTK resources downloaded and loaded.")
except Exception as e:
    NLTK_AVAILABLE = False
    logger.warning(f"NLTK initialization failed: {e}")

# Mock classes for missing modules
class MockPreprocessor:
    def preprocess_file(self, file_path, domain):
        logger.debug(f"Mock preprocessing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content, {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return "", {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})

class MockIntentClassifier:
    def predict(self, text): return "default"
    def train_model(self, data): return True

class MockTokenizer:
    def save(self, path): pass
    
tokenizer = MockTokenizer()

# Import fallbacks
try:
    from utils.preprocessing import Preprocessor
    preprocessor = Preprocessor()
except ImportError:
    preprocessor = MockPreprocessor()
    logger.warning("Preprocessor not found, using mock")

try:
    from intent.classifier import IntentClassifier
except ImportError:
    IntentClassifier = MockIntentClassifier
    logger.warning("Preprocessor not found, using mock")

# Local file storage for queries
def save_query_to_file(user_id, prompt, response):
    try:
        queries_file = "query_history.json"
        try:
            with open(queries_file, 'r') as f:
                queries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            queries = []

        queries.append({
            "user_id": user_id,
            "query": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        with open(queries_file, 'w') as f:
            json.dump(queries, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving to file: {e}")
        st.warning(f"⚠️ Error saving query/response to file: {e}")

# Model management functions
def save_trained_model(model_name, model_data, llm_handler=None):
    logger.debug(f"Saving model: {model_name}")
    try:
        models_file = "trained_models.json"
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")

        try:
            with open(models_file, 'r') as f:
                models = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            models = {}

        version = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        model_dir = os.path.join("saved_models", version)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "checkpoint.pt")
        tokenizer_path = os.path.join(model_dir, "uml_tokenizer.pkl")

        if version in models:
            logger.info(f"Model version {version} already exists")
            return version

        if llm_handler:
            model_data["doc_texts"] = getattr(llm_handler, "doc_texts", {})
            model_data["is_trained"] = getattr(llm_handler, "is_trained", False)
            model_data["all_chunks"] = getattr(llm_handler, "all_chunks", [])
            model_data["is_trained"] = getattr(llm_handler, "is_trained", False)
            model_data["vocab_size"] = getattr(llm_handler, "vocab_size", 0)
            model_data["tokenizer_vocab"] = getattr(llm_handler.tokenizer, "token_to_id", {}).get("vocab_size", 0)
            torch.save({
                'model_state_dict': llm_handler.model.state_dict(),
                'vocab_size': llm_handler.vocab_size,
                'doc_texts': llm_handler.doc_texts,
                'is_trained': llm_handler.is_trained
            }, model_path)
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            if hasattr(llm_handler, "tokenizer") and hasattr(llm_handler.tokenizer, "save_pickle"):
                llm_handler.tokenizer.save_pickle(tokenizer_path)

        models[version] = {
            "name": model_name,
            "version": version,
            "timestamp": timestamp,
            "data": model_data,
            "path": model_path,
            "tokenizer_path": tokenizer_path
        }

        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
    
        logger.info(f"Model saved successfully: {model_path}")
        return version

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def load_trained_models():
    logger.debug("Loading trained models")
    models = {}
    try:
        with open("trained_models.json", "r", encoding="utf-8") as f:
            models = json.load(f)
        for k, v in models.items():
            v.setdefault("name", k)
            v.setdefault("version", k)
            v.setdefault("timestamp", "Unknown")
            v.setdefault("data", {})
    except (FileNotFoundError, json.JSONDecodeError):
        logger.debug("No trained_models.json found or invalid")
        pass

    saved_models_dir = "saved_models"
    allowed_extensions = {'.pt', '.pth', '.pkl', '.bin'}
    if os.path.isdir(saved_models_dir):
        for fname in os.listdir(saved_models_dir):
            name, ext = os.path.splitext(fname)
            if ext.lower() in allowed_extensions:
                file_path = os.path.join(saved_models_dir, fname)
                if fname not in models:
                    models[fname] = {
                        "name": name,
                        "version": fname,
                        "timestamp": "Manual Upload",
                        "data": {},
                        "path": file_path
                    }
    return models

def get_model_dropdown_options():
    models = load_trained_models()
    return [(f"{model['name']} ({model['version']}) - {model['timestamp']}", key) for key, model in models.items()]

def get_latest_model(models):
    logger.debug("Getting latest model")
    if not models:
        return None
    try:
        sorted_models = sorted(models.items(), 
                              key=lambda x: datetime.strptime(x[1]['timestamp'], "%Y-%m-%d %I:%M %p") 
                              if x[1]['timestamp'] not in ['Unknown', 'Manual Upload'] else datetime.min, reverse=True)
        return sorted_models[0][0] if sorted_models else None
    except Exception:
        return list(models.keys())[0]

def auto_load_latest_model(model_key):
    logger.debug(f"Auto-loading model: {model_key}")
    try:
        models = load_trained_models()
        model_info = models.get(model_key)
        if not model_info:
            logger.warning(f"Model metadata for {model_key} not found")
            st.warning(f"⚠️ Model metadata for {model_key} not found.")
            return False

        model_path = model_info.get("path")
        tokenizer_path = model_info.get("tokenizer_path", "uml_tokenizer.pkl")
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist")
            st.error(f"❌ Model file {model_path} does not exist.")
            return False
        
        # Detect domain from model metadata or documents
        domain_name = model_info.get("domain", "general")
        if not domain_name or domain_name == "general":
            # Use document content to detect domain
            doc_texts = model_info.get("data", {}).get("doc_texts", {})
            if doc_texts:
                content = " ".join(doc.get("content", "") for doc in doc_texts.values())
                domain_name = detect_domain(content) or "general"

        llm_handler = LLMHandler(model_path=model_path, tokenizer_path=tokenizer_path, model_version=model_key)
        llm_handler.load_model(model_path, tokenizer_path, model_key)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        doc_texts = model_data.get("doc_texts", {})
        is_trained = model_data.get("is_trained", False)
        llm_handler.doc_texts = doc_texts
        llm_handler.is_trained = is_trained
        llm_handler.index_documents([(fname, doc["content"]) for fname, doc in doc_texts.items()], force_reindex=True)

        if not doc_texts:
            st.warning("⚠️ No documents stored in the model file.")
        else:
            st.session_state.llm_handler.doc_texts = doc_texts
            st.session_state.llm_handler.index_documents([(fname, doc["content"]) for fname, doc in doc_texts.items()], force_reindex=True)

        if len(llm_handler.tokenizer.token_to_id) <= 4:
            logger.error(f"Tokenizer vocabulary too small ({len(llm_handler.tokenizer.token_to_id)} tokens)")
            st.error(f"❌ Tokenizer vocabulary too small ({len(llm_handler.tokenizer.token_to_id)} tokens).")
            return False

        st.session_state.llm_handler = llm_handler
        st.session_state.model_loaded = True
        st.session_state.selected_model = model_key
        logger.info(f"Model loaded successfully: {model_key}")
        st.success(f"✅ Model loaded: {model_key}")
        return True

    except Exception as e:
        logger.error(f"Failed to load model {model_key}: {e}", exc_info=True)
        st.error(f"❌ Failed to load model {model_key}: {e}")
        st.session_state.model_loaded = False
        return False
        
# Chat title generation
def generate_chat_title(messages):
    if not messages or not NLTK_AVAILABLE:
        return "New Chat"
    try:
        # Get the first user message
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        if not user_messages:
            return "New Chat"
        
        text = user_messages[0]
        # Use NLTK for basic keyword extraction
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        keywords = [word for word, pos in tagged if pos.startswith(('NN', 'VB', 'JJ'))]
        
        # Remove stopwords and short words
        stopwords = set(nltk.corpus.stopwords.words('english') if NLTK_AVAILABLE else [])
        keywords = [word for word in keywords if word.lower() not in stopwords and len(word) > 3]
        
        # Generate title from top keywords
        title = " ".join(keywords[:3]).title()
        return title[:30] + ("..." if len(title) > 30 else "") or "New Chat"
    except Exception as e:
        logger.error(f"Error generating chat title: {e}")
        return "New Chat"
    
# Session state initialization
def initialize_session_state():
    defaults = {
        'chat_history': [{"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}],
        'selected_chat_index': 0,
        'processed_files': [],
        'current_view': 'chat',
        'show_welcome': True,
        'selected_model': None,
        'trained_models': {},
        'model_loaded': False,
        'vocab_comparison_done': False,
        'vocab_results': [],
        'log_placeholder': st.empty()  # Added for log display
    }
    
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler(model_path="checkpoint.pt", tokenizer_path="uml_tokenizer.pkl")
    
        # Ensure existing chats have archived status
    for chat in st.session_state.chat_history:
        if "archived" not in chat:
            chat["archived"] = False

    st.session_state.chatbot.llm_handler = st.session_state.llm_handler

# Welcome UI (unchanged)
def show_welcome():
    logger.debug("Rendering welcome page")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🤖 AI Document Assistant</h1>
        <h3>Upload your documents and start intelligent conversations. Get AI-powered insights, summaries, and answers from your files with advanced machine learning.</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    features = [
        ("📄", "Document Processing", "Upload PDFs, Word docs, text files, and spreadsheets for intelligent analysis"),
        ("💬", "Smart Conversations", "Ask questions about your documents and get contextual, intelligent responses"),
        ("🧠", "AI Training", "Train custom models on your data for enhanced performance and accuracy")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; margin: 1rem 0;'>
                <div style='font-size: 2rem; margin-bottom: 1rem;'>{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start New Conversation →", type="primary", use_container_width=True):
            st.session_state.update({'show_welcome': False, 'current_view': 'chat'})
            st.rerun()

def render_model_dashboard():
    logger.debug("Rendering model dashboard")
    st.markdown("### 🧠 Model Dashboard")

    # Load models and check for documents
    models = load_trained_models()
    if not models:
        st.info("📭 No trained models available. Upload documents and train a model to get started.")
        return

    llm_handler = st.session_state.get("llm_handler")
    doc_texts = getattr(llm_handler, 'doc_texts', {}) if llm_handler else {}
    
    doc_count = len(doc_texts)
    if doc_count == 0:
        st.warning("⚠️ No document data available in the current model session.")
        return

    # Performance Metrics
    accuracy = get_model_accuracy() if hasattr(llm_handler, 'model') else calculate_dynamic_accuracy()
    processing_speed = calculate_processing_speed(doc_texts)
    memory_usage = calculate_memory_usage()

    # Domain Detection
    document_domains = {}
    all_domains = set()

    for doc_name, content in doc_texts.items():
        text_content = content.get('content', str(content)) if isinstance(content, dict) else str(content)
        domain = detect_domain(text_content) if text_content else "general"
        document_domains[doc_name] = domain
        all_domains.add(domain)

    primary_domain = max(all_domains, key=lambda x: list(document_domains.values()).count(x)) if all_domains else "general"

    # Main Metric Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documents", doc_count)
    with col2:
        st.metric("📊 Accuracy", f"{accuracy:.1f}%", delta=f"{accuracy-85:+.1f}%")
    with col3:
        if len(all_domains) > 1:
            st.metric("🎯 Domains", f"{len(all_domains)}")
            st.caption(f"Primary: {primary_domain.title()}")
        else:
            st.metric("🎯 Domain", primary_domain.title())
    with col4:
        st.metric("💾 Memory", f"{memory_usage:.1f}%")
        st.caption(f"Speed: {processing_speed:.0f} ms")

    # System Status
    status_cols = st.columns(4)
    with status_cols[0]:
        st.success("🟢 System Online")
    with status_cols[1]:
        st.info(f"🔄 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    with status_cols[2]:
        st.metric("⚡ Speed", f"{processing_speed:.0f} ms")
    with status_cols[3]:
        st.success("🟢 Model: Active" if st.session_state.get("model_loaded") else "⚠️ Model: Inactive")

    # 📚 Document Tools
    st.markdown("### 📚 Documents & Training")
    
    # Bulk Retrain
    col_bulk1, col_bulk2 = st.columns(2)
    with col_bulk1:
        if st.button("🔄 Retrain All Documents", use_container_width=True):
            with st.spinner("Retraining all documents..."):
                retrain_model(list(doc_texts.keys()))
            st.success("✅ All documents retrained")
            st.balloons()
            st.rerun()

    with col_bulk2:
        selected_docs = st.multiselect("Select documents for bulk retrain", list(doc_texts.keys()))
        if st.button("🔄 Retrain Selected", disabled=not selected_docs, use_container_width=True):
            with st.spinner(f"Retraining {len(selected_docs)} documents..."):
                retrain_model(selected_docs)
            st.success(f"✅ {len(selected_docs)} documents retrained")
            st.rerun()

    # Individual Document Display
    for doc_name, content in doc_texts.items():
        text_content = content.get('content', str(content)) if isinstance(content, dict) else str(content)
        domain = document_domains.get(doc_name, "general")
        word_count = len(text_content.split())
        confidence = calculate_domain_confidence(text_content, domain)
        complexity = calculate_text_complexity(text_content)
        
        with st.expander(f"📄 {doc_name} • {domain.title()} • {word_count:,} words"):
            col_info, col_retrain = st.columns([4, 1])
            with col_info:
                metric_cols = st.columns(4)
                metric_cols[0].metric("📝 Words", f"{word_count:,}")
                metric_cols[1].metric("🎯 Domain", domain.title())
                metric_cols[2].metric("📊 Confidence", f"{confidence:.0f}%")
                metric_cols[3].metric("🔍 Complexity", complexity)

            with col_retrain:
                if st.button(f"🔄 Retrain", key=f"retrain_{doc_name}", use_container_width=True):
                    with st.spinner(f"Retraining {doc_name}..."):
                        retrain_model([doc_name])
                    st.success(f"✅ {doc_name} retrained")
                    st.rerun()

            preview = text_content[:300] + "..." if len(text_content) > 300 else text_content
            st.text_area("Preview", preview, height=80, disabled=True, label_visibility="collapsed")

    # 📊 Domain Pie Chart
    if len(all_domains) > 1:
        st.markdown("### 🎯 Domain Distribution")
        domain_counts = {d: list(document_domains.values()).count(d) for d in all_domains}

        col_chart, col_stats = st.columns([3, 2])
        with col_chart:
            fig = go.Figure(data=[go.Pie(
                labels=[d.title() for d in domain_counts],
                values=list(domain_counts.values()),
                hole=0.3
            )])
            fig.update_layout(title="Document Domain Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            for d, c in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                st.metric(d.title(), f"{c} docs", f"{(c / doc_count) * 100:.1f}%")

    # ⚙️ Model Manager
    st.markdown("### ⚙️ Model Management")
    if models:
        col_model, col_actions = st.columns([3, 2])
        with col_model:
            if not st.session_state.get("selected_model"):
                st.session_state.selected_model = list(models.keys())[0]

            model_options = [(f"📋 {name} ({info.get('doc_count', 0)} docs)", name) for name, info in models.items()]
            current_display = [opt[0] for opt in model_options]
            current_index = next((i for i, (_, key) in enumerate(model_options)
                                  if key == st.session_state.selected_model), 0)

            selected_display = st.selectbox("Active Model", current_display, index=current_index)
            model_key = next(k for label, k in model_options if label == selected_display)

            if model_key != st.session_state.selected_model:
                st.session_state.selected_model = model_key
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded: {model_key}")
                st.rerun()

        with col_actions:
            if st.button("🔄 Reload", use_container_width=True):
                st.session_state.model_loaded = True
                st.success("✅ Model reloaded")
                st.rerun()

            if st.button("🗑️ Delete", use_container_width=True):
                if st.session_state.selected_model in models:
                    path = models[st.session_state.selected_model].get("path")
                    if path and os.path.exists(path):
                        os.remove(path)
                    del models[st.session_state.selected_model]
                    save_trained_model(models)
                    st.session_state.update({'selected_model': None, 'model_loaded': False})
                    st.success("✅ Model deleted")
                    st.rerun()

    # 📈 Training Performance Chart
    if st.session_state.get("model_loaded") and os.path.exists("logs/training_log.txt"):
        st.markdown("### 📈 Performance Monitor")
        try:
            df = load_training_log("logs/training_log.txt")
            if not df.empty and {'TrainLoss', 'ValLoss'}.issubset(df.columns):
                col_chart, col_metrics = st.columns([3, 1])
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['TrainLoss'], name='Train Loss', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['ValLoss'], name='Val Loss', line=dict(color='red')))
                    fig.update_layout(title="🎯 Training Progress", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col_metrics:
                    st.metric("📊 Epochs", len(df))
                    st.metric("📉 Best Loss", f"{df['ValLoss'].min():.4f}")
                    st.metric("🎯 Final Loss", f"{df['ValLoss'].iloc[-1]:.4f}")
                    improvement = df['ValLoss'].iloc[0] - df['ValLoss'].iloc[-1]
                    if improvement > 0:
                        st.success(f"🟢 Improved by {improvement:.4f}")
                    else:
                        st.warning("🟡 Needs optimization")
        except Exception as e:
            st.error(f"Performance data unavailable: {e}")

    # Navigation Controls
    col_nav1, col_nav2, col_refresh = st.columns(3)
    with col_nav1:
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    with col_nav2:
        if st.button("💬 Start Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    with col_refresh:
        if st.button("🔄 Auto-Refresh", use_container_width=True, help="Enable auto-refresh"):
            st.rerun()



# Helper functions for real-time calculations
def calculate_domain_confidence(text, detected_domain):
    """Calculate confidence score for domain detection"""
    if not text:
        return 0
    
    domain_keywords = {
        'technical': ['algorithm', 'implementation', 'code', 'system', 'technology', 'software', 'programming'],
        'business': ['revenue', 'strategy', 'market', 'customer', 'profit', 'sales', 'management'],
        'academic': ['research', 'study', 'analysis', 'methodology', 'conclusion', 'hypothesis'],
        'medical': ['patient', 'treatment', 'diagnosis', 'clinical', 'medical', 'health'],
        'legal': ['contract', 'agreement', 'law', 'legal', 'regulation', 'compliance'],
        'general': ['information', 'data', 'content', 'text', 'document']
    }
    
    keywords = domain_keywords.get(detected_domain.lower(), domain_keywords['general'])
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword in text_lower)
    
    # Calculate confidence based on keyword matches and text length
    base_confidence = 50
    keyword_boost = min(45, matches * 8)  # Max 45% boost from keywords
    length_factor = min(1.0, len(text.split()) / 100)  # Text length factor
    
    confidence = base_confidence + (keyword_boost * length_factor)
    return min(95, max(10, confidence))  # Keep between 10-95%

def calculate_text_complexity(text):
    """Calculate text complexity level"""
    if not text:
        return "Low"
    
    words = text.split()
    if not words:
        return "Low"
    
    # Calculate average word length and sentence complexity
    avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
    sentences = text.split('.')
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Complexity scoring
    complexity_score = (avg_word_length * 0.6) + (avg_sentence_length * 0.4)
    
    if complexity_score > 8:
        return "High"
    elif complexity_score > 5:
        return "Medium"
    else:
        return "Low"

def retrain_model(selected_docs):
    """Retrain model with selected documents"""
    try:
        llm_handler = st.session_state.get("llm_handler")
        if llm_handler and hasattr(llm_handler, 'retrain'):
            llm_handler.retrain(selected_docs)
        else:
            st.warning("Retrain functionality not available")
    except Exception as e:
        st.error(f"Retrain failed: {e}")

def render_sidebar():
    logger.debug("Rendering sidebar")
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant")
        # User info and logout - ADD THIS
        full_name= st.session_state.get("full_name", "Guest")
        st.success(f"👋 Welcome, {full_name}!")
        
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session
            for key in ['user_id', 'full_name', 'user_email', 'authenticated']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        # Navigation
        col1, col2 = st.columns(2)
        nav_buttons = [("💬 Chat", "chat"), ("🧠 Models", "dashboard")]
        for i, (label, view) in enumerate(nav_buttons):
            with [col1, col2][i]:
                if st.button(label, use_container_width=True, 
                           type="primary" if st.session_state.current_view == view else "secondary"):
                    st.session_state.current_view = view
                    if view == "chat":
                        st.session_state.show_welcome = False
                    st.rerun()

        # SEPARATED TRAINING BUTTON
        if st.button("🚀 Train Model", use_container_width=True):
            if not NLTK_AVAILABLE:
                st.error("NLTK not installed. Please install with 'pip install nltk' and restart the app.")
                return
        
            if st.session_state.chatbot and st.session_state.chatbot.documents:
                with st.spinner("Training model..."):
                    try:
                        llm_handler = st.session_state.llm_handler
                        intent_classifier = st.session_state.chatbot.intent_classifier

                        # Validate document content
                        valid_documents = []
                        for filename, doc in st.session_state.chatbot.documents.items():
                            content = doc["content"]
                            if not content.strip():
                                st.warning(f"⚠️ Skipping empty document: {filename}")
                                logger.warning(f"Skipping empty document: {filename}")
                                continue
                            token_count = len(content.split())  # Approximate token count
                            if token_count < 1024:  # 2 * seq_len (512)
                                st.warning(f"⚠️ Document {filename} too short ({token_count} tokens)")
                                logger.warning(f"Document {filename} too short ({token_count} tokens)")
                                continue
                            valid_documents.append((filename, content))
                        if not valid_documents:
                            st.error("❌ No valid documents for training")
                            logger.error("No valid documents for training")
                            return
        
                        # Index documents
                        logger.debug(f"Indexing {len(valid_documents)} documents")
                        llm_handler.index_documents(valid_documents, force_reindex=True)

                        # Train intent classifier
                        for filename, doc in st.session_state.chatbot.documents.items():
                            if "intent_data" in doc and isinstance(doc["intent_data"], pd.DataFrame):
                                intent_classifier.train_model(doc["intent_data"])
        
        
                        #  TRAIN LLM
                                                # Train LLM
                        st.info("🔄 Training model... This may take a few minutes.")
                        logger.debug("Starting LLM training")
                        success = llm_handler.train_on_documents(epochs=15, batch_size=8, save_path="checkpoint.pt")                        

                        if success and llm_handler.is_trained and len(llm_handler.tokenizer.token_to_id) > 4:
                            st.success("✅ Model trained successfully")
                            logger.info("Model trained successfully")
                        else:
                            st.error("❌ Model training failed or tokenizer vocabulary too small")
                            logger.error("Model training failed or tokenizer vocabulary too small")
                            return
        
                        doc_names = "_".join([os.path.splitext(name)[0] for name in st.session_state.chatbot.documents.keys()])
                        model_name = f"model_{doc_names}"[:50]
                        model_data = {
                            "documents": list(st.session_state.chatbot.documents.keys()),
                            "doc_texts": llm_handler.doc_texts,
                            "is_trained": llm_handler.is_trained
                        }
                        model_key = save_trained_model(model_name, model_data, llm_handler)
        
                        if model_key:
                            st.session_state.update({'selected_model': model_key, 'model_loaded': True})
                            st.success(f"✅ Model trained and saved: {model_key}")
                            logger.info(f"Model saved: {model_key}")
                        else:
                            st.error("❌ Failed to save trained model")
                            logger.error("Failed to save trained model")
        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        logger.error(f"Training error: {e}", exc_info=True)            
            else:
                st.warning("📁 Upload documents first!")

        # New Chat
        if st.button("+ New Chat", use_container_width=True):
            new_chat = {"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}
            st.session_state.chat_history.append(new_chat)
            st.session_state.update({
                'selected_chat_index': len(st.session_state.chat_history) - 1,
                'show_welcome': False,
                'current_view': 'chat'
            })
            st.rerun()

        st.markdown("---")

        # Chat History
        st.markdown("**📝 RECENT CHATS**")
        for i, chat in enumerate(st.session_state.chat_history):
            chat_title = chat["title"] if chat["title"] != "New Chat" else f"Chat {i+1}"
            is_selected = i == st.session_state.selected_chat_index

            if st.button(f"💬 {chat_title[:20]}", key=f"chat_{i}", use_container_width=True,
                        type="primary" if is_selected else "secondary"):
                st.session_state.update({
                    'selected_chat_index': i,
                    'show_welcome': False,
                    'current_view': 'chat'
                })
                st.rerun()

        st.markdown("---")

        # Document Upload
        st.markdown("**📁 DOCUMENTS**")
        uploaded_files = st.file_uploader("Upload Files", 
                                         type=["pdf", "txt", "docx", "csv", "xlsx", "pptx", "ppt", "doc"],
                                         accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            st.caption(f"📁 {len(uploaded_files)} files selected")

        st.caption("PDF, Word, Text, CSV, Excel files supported")

        # SIMPLIFIED PROCESSING - NO TRAINING
        if uploaded_files and st.button("📊 Process Files", type="primary", use_container_width=True):
            if not NLTK_AVAILABLE:
                st.error("NLTK not installed. Please install with 'pip install nltk' and restart the app.")
                return
            
            with st.spinner("Processing files..."):
                try:
                    processed_count = 0
                    for file in uploaded_files:
                        if any(f["filename"] == file.name for f in st.session_state.processed_files):
                            st.warning(f"⚠️ {file.name} already processed. Skipping.")
                            continue

                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            # PREPROCESSING
                            cleaned, metadata, intent_data = preprocessor.preprocess_file(tmp_path, domain="general")
                            content = cleaned if isinstance(cleaned, str) else str(cleaned)
                            if not content.strip():
                                st.error(f"❌ No valid content extracted from {file.name}")
                                continue
                            
                            st.session_state.chatbot.add_document(file.name, {
                                "content": content,
                                "file": file,
                                "type": file.type,
                                "metadata": metadata,
                                "intent_data": intent_data
                            })
                            
                            # Index in LLM_handler
                            st.session_state.llm_handler.index_documents([(file.name, content)], force_reindex=True)
                            st.session_state.processed_files.append({
                                "filename": file.name,
                                "type": file.type,
                                "size": len(content),
                                "processed_at": datetime.now().isoformat()
                            })
                            processed_count += 1
                        except Exception as e:
                            st.error(f"Failed to process {file.name}: {e}")
                        finally:
                            os.unlink(tmp_path)
                    if processed_count > 0:
                        st.success(f"✅ {processed_count} files processed successfully! Now click 'Train Model' to train.")
                        st.rerun()
                    else:
                        st.warning("No new files were processed.")
                except Exception as e:
                    st.error(f"Processing failed: {e}")


# CHAT INTERFACE 
def render_chat():
    logger.debug("Rendering chat interface")

    if st.session_state.show_welcome:
        show_welcome()
        return

    user_id = st.session_state.get("user_id")
    chat_history = st.session_state.get("chat_history", [])
    selected_index = st.session_state.get("selected_chat_index", 0)

    if not chat_history or selected_index >= len(chat_history):
        st.warning("No chat history available or invalid index.")
        logger.warning("No chat history or invalid index")
        return

    chat = chat_history[selected_index]

    # Update chat title based on content
    if chat["title"] == "New Chat" and chat["messages"]:
        chat["title"] = generate_chat_title(chat["messages"])

    # Display previous messages
    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show placeholder if no messages
    if not chat["messages"]:
        messages = [
            "Ready to explore your documents? Ask me anything!",
            "I'm here to help you discover insights from your files.",
            "What would you like to know about your documents today?",
            "Let's dive into your content together!",
            "I can help you find patterns and connections across files."
        ]
        current_msg = messages[int(time.time() // 5) % len(messages)]
        with st.chat_message("assistant"):
            st.markdown(current_msg)

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = ""
                    model_status = ""

                    # Safe domain detection
                    try:
                        domain_name = detect_domain(prompt)
                    except Exception as e:
                        domain_name = "general"
                        logger.warning(f"Domain detection failed: {e}")

                    # Use trained model if available
                    if (st.session_state.model_loaded and 
                        st.session_state.llm_handler and 
                        st.session_state.llm_handler.is_trained):

                        logger.info(f"Using trained model for query: {prompt}")
                        context = st.session_state.llm_handler.get_relevant_context(prompt)
                        response = st.session_state.llm_handler.generate_response(
                            prompt, max_length=200, temperature=0.7, context=context
                        )
                        model_name = st.session_state.selected_model or "Trained Model"
                        model_status = f"*[Using trained model: {model_name}]*\n\n"

                    # Fallback to basic chatbot
                    elif st.session_state.chatbot and st.session_state.chatbot.documents:
                        st.session_state.chatbot.llm_handler = st.session_state.llm_handler
                        response = st.session_state.chatbot.generate_response(prompt)
                        model_status = "*[Using basic chatbot - train a model for better results]*\n\n"

                    else:
                        response = "Please upload and process documents, then train a model to get better responses."
                        model_status = "*[No documents or model available]*\n\n"
                        logger.warning("No documents or trained model available")

                    if not response.strip():
                        response = "I couldn't generate a meaningful response to your query. Please try rephrasing."
                        logger.warning("Empty response generated")

                    # Display response
                    full_response = model_status + response
                    st.markdown(full_response)
                    chat["messages"].append({"role": "assistant", "content": full_response})

                    # Save the query
                    if user_id:
                        try:
                            payload = {
                                "user_id": user_id,
                                "query": prompt,
                                "response": response,
                                "timestamp": datetime.now().isoformat()
                            }
                            if supabase and SUPABASE_AVAILABLE:
                                result = supabase.table("queries").insert(payload).execute()
                                if not result.data:
                                    st.warning("⚠️ Failed to save to Supabase")
                            else:
                                save_query_to_file(user_id, prompt, response)
                        except Exception as e:
                            logger.warning(f"Error saving query: {e}")
                            save_query_to_file(user_id, prompt, response)
                    else:
                        user_id_from_url = st.query_params.get("user_id")
                        if user_id_from_url:
                            st.session_state["user_id"] = user_id_from_url
                            logger.info(f"✅ user_id from URL: {user_id_from_url}")
                        else:
                            st.warning("⚠️ Login required. Please log in from the main page.")

                    # Update chat title if it's still 'New Chat'
                    if len(chat["messages"]) >= 2 and chat["title"] == "New Chat":
                        chat["title"] = generate_chat_title(chat["messages"])

                    st.rerun()

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    logger.error(f"Chat error: {traceback.format_exc()}")
                    st.error(error_msg)
                    chat["messages"].append({"role": "assistant", "content": error_msg})


# Main application
def main():
    logger.debug("Starting app.py main")
    st.set_page_config(
        page_title="AI Document Assistant",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
        # Authentication check - ADD THIS BLOCK
    if "user_id" not in st.session_state or not st.session_state["user_id"]:
        user_id_from_url = st.query_params.get("user_id")
        if user_id_from_url:
            st.session_state["user_id"] = user_id_from_url
            logger.info(f"✅ user_id received from URL: {user_id_from_url}")
        else:
            # Redirect to authentication
            st.error("🔐 Please login to access the AI Document Assistant")
            if st.button("🚀 Go to Login"):
                st.markdown('<meta http-equiv="refresh" content="0;url=/">', unsafe_allow_html=True)
            st.stop()
    # Custom rounded UI elements
    st.markdown("""
        <style>
        .stButton > button { border-radius: 8px; }
        .stSelectbox > div > div { border-radius: 8px; }
        .stTextInput > div > div { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

    if not NLTK_AVAILABLE:
        st.error("NLTK is not installed. Please run 'pip install nltk' and restart the app.")
        return

    initialize_session_state()
    render_sidebar()

    models = load_trained_models()

    # Clear invalid selection
    if st.session_state.selected_model not in models and st.session_state.selected_model is not None:
        st.session_state.selected_model = None
        st.session_state.model_loaded = False

    # Auto-select latest model if nothing is selected
    if not st.session_state.selected_model and models:
        latest_model = get_latest_model(models)
        if latest_model:
            st.session_state.selected_model = latest_model
            st.session_state.model_loaded = auto_load_latest_model(latest_model)

    # Load selected model (if not already loaded)
    elif st.session_state.selected_model and not st.session_state.model_loaded:
        st.session_state.model_loaded = auto_load_latest_model(st.session_state.selected_model)

    # UI routing
    if st.session_state.current_view == "dashboard":
        render_model_dashboard()
    else:
        render_chat()

if __name__ == "__main__":
    main()