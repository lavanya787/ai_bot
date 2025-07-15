import logging
import streamlit as st
import os
import tempfile
import json
import hashlib
import sys
import pandas as pd
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import traceback
from chatbot_module import ChatBot
from llm_handler import LLMHandler

# Setup - REDUCED LOGGING LEVEL
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')  # Changed to WARNING
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Safe imports
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Mock classes for missing modules
class MockPreprocessor:
    def preprocess_file(self, file_path, domain):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content, {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
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

try:
    from intent.classifier import IntentClassifier
except ImportError:
    IntentClassifier = MockIntentClassifier

# REMOVED store_and_train import - using only preprocessing
def simple_preprocess_file(file_path, domain):
    """Simple preprocessing without training"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return "", {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})

# Model management functions (unchanged)
def save_trained_model(model_name, model_data, llm_handler=None):
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
            return version

        if llm_handler:
            model_data["doc_texts"] = getattr(llm_handler, "doc_texts", {})
            model_data["tokenizer_vocab"] = getattr(llm_handler, "tokenizer_vocab", {})
            model_data["all_chunks"] = getattr(llm_handler, "all_chunks", [])
            model_data["is_trained"] = getattr(llm_handler, "is_trained", False)

            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            if hasattr(llm_handler, "tokenizer") and hasattr(llm_handler.tokenizer, "save"):
                llm_handler.tokenizer.save(tokenizer_path)

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

        return version

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def load_trained_models():
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
    import pickle
    try:
        models = load_trained_models()
        model_info = models.get(model_key)
        if not model_info:
            st.warning(f"⚠️ Model metadata for {model_key} not found.")
            return False

        model_path = model_info.get("path")
        tokenizer_path = model_info.get("tokenizer_path", "uml_tokenizer.pkl")
        if not model_path or not os.path.exists(model_path):
            st.error(f"❌ Model file {model_path} does not exist.")
            return False

        llm_handler = LLMHandler(model_path=model_path, tokenizer_path=tokenizer_path, model_version=model_key)

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
            st.error(f"❌ Tokenizer vocabulary too small ({len(llm_handler.tokenizer.token_to_id)} tokens).")
            return False

        st.session_state.llm_handler = llm_handler
        st.session_state.model_loaded = True
        st.session_state.selected_model = model_key
        return True

    except Exception as e:
        st.error(f"❌ Failed to load model {model_key}: {e}")
        st.session_state.model_loaded = False
        return False
        
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
        'vocab_results': []
    }
    
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler(model_path="checkpoint.pt", tokenizer_path="uml_tokenizer.pkl")
        
    st.session_state.chatbot.llm_handler = st.session_state.llm_handler

# Welcome UI (unchanged)
def show_welcome():
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
    st.markdown("### 🧠 Model Dashboard")
    models = load_trained_models()
    if not models:
        st.info("📭 No trained models available. Upload documents and train a model to get started.")
        return

    model_options = get_model_dropdown_options()

    if model_options:
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.selected_model:
                st.session_state.selected_model = get_latest_model(models) or model_options[0][1]

            current_selection = next((i for i, (_, key) in enumerate(model_options) if key == st.session_state.selected_model), 0)
            selected_display = st.selectbox(
                "Select Trained Model",
                options=[opt[0] for opt in model_options],
                index=current_selection,
                key="model_selector"
            )

            model_key = next(opt[1] for opt in model_options if opt[0] == selected_display)
            if model_key != st.session_state.selected_model:
                st.session_state.selected_model = model_key
                st.session_state.model_loaded = False
                if auto_load_latest_model(model_key):
                    st.success(f"✅ Model loaded: {model_key}")
                else:
                    st.error(f"❌ Failed to load model: {model_key}")
                st.rerun()

        with col2:
            if st.button("🗑️ Delete Model") and st.session_state.selected_model in models:
                model_path = models[st.session_state.selected_model].get("path")
                if model_path and os.path.exists(model_path):
                    os.remove(model_path)
                del models[st.session_state.selected_model]
                with open("trained_models.json", 'w') as f:
                    json.dump(models, f, indent=2)
                st.success("✅ Model deleted successfully")
                st.session_state.update({'selected_model': None, 'model_loaded': False})
                st.rerun()

        if st.session_state.selected_model and st.session_state.model_loaded:
            llm_handler = st.session_state.get("llm_handler")
            if llm_handler and hasattr(llm_handler, "get_status"):
                status = llm_handler.get_status()
                doc_count = len(getattr(llm_handler, 'doc_texts', {}))

                st.markdown("#### Model Status")
                cols = st.columns(5)
                metrics = [
                    ("Status", "✅ Loaded" if status.get("trained", False) else "⚠️ Not Trained"),
                    ("Trained", "Yes" if status.get("trained", False) else "No"),
                    ("Documents", doc_count),
                    ("Device", status.get("device", "Unknown")),
                    ("Vocab Size", status.get("tokenizer_vocab", "Unknown")),
                ]

                for i, (label, value) in enumerate(metrics):
                    with cols[i]:
                        if label == "Status":
                            if "✅" in value:
                                st.success(value)
                            else:
                                st.warning(value)
                        else:
                            st.metric(label, value)

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant")

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
        
                        # Index documents without training
                        for filename, doc in st.session_state.chatbot.documents.items():
                            content = doc["content"]
                            if not content.strip():
                                st.warning(f"⚠️ Skipping empty document: {filename}")
                                continue
                            llm_handler.index_documents([(filename, content)], force_reindex=True)
                            if "intent_data" in doc and isinstance(doc["intent_data"], pd.DataFrame):
                                intent_classifier.train_model(doc["intent_data"])
        
                        # ACTUAL TRAINING HAPPENS HERE
                        st.info("🔄 Training model... This may take a few minutes.")
                        llm_handler.train_on_documents(epochs=3, batch_size=8, save_path="checkpoint.pt")
                        
                        if llm_handler.is_trained and len(llm_handler.tokenizer.token_to_id) > 4:
                            st.success("✅ Model trained successfully")
                        else:
                            st.error("❌ Model training failed or tokenizer vocabulary too small")
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
                        else:
                            st.error("Failed to save trained model")
        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        logger.error(f"Training error: {e}")
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

        # Configuration
        with st.expander("⚙️ Configuration"):
            domain = st.text_input("Domain", value=os.getenv("DOMAIN", "physics"))

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
                            # ONLY PREPROCESSING - NO TRAINING
                            cleaned, metadata, intent_data = simple_preprocess_file(tmp_path, domain)
                            
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

                            # Index in llm_handler
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
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                            
                    if processed_count > 0:
                        st.success(f"✅ {processed_count} files processed successfully! Now click 'Train Model' to train.")
                        st.rerun()
                    else:
                        st.warning("No new files were processed.")

                except Exception as e:
                    st.error(f"Processing failed: {e}")

# FIXED CHAT INTERFACE - USES TRAINED MODEL
# FIXED CHAT INTERFACE - IMPROVED MODEL RESPONSE GENERATION
def render_chat():
    if st.session_state.show_welcome:
        show_welcome()
        return

    chat_history = st.session_state.get("chat_history", [])
    selected_index = st.session_state.get("selected_chat_index", 0)

    if not chat_history or selected_index >= len(chat_history):
        st.warning("No chat history available or invalid index.")
        return

    chat = chat_history[selected_index]
    
    # Display chat messages
    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                    
                    # Check if we have a trained model loaded
                    if (st.session_state.model_loaded and 
                        st.session_state.llm_handler and 
                        st.session_state.llm_handler.is_trained):
                        
                        try:
                            # Use the trained model for generation
                            logger.info(f"Using trained model for query: {prompt}")
                            
                            # Get relevant context from documents first
                            relevant_context = st.session_state.llm_handler.get_relevant_context(prompt)
                            
                            # Generate response using the trained model
                            response = st.session_state.llm_handler.generate_response(
                                prompt, 
                                max_length=200,
                                temperature=0.7,
                                context=relevant_context
                            )
                            
                            model_name = st.session_state.selected_model or "Trained Model"
                            model_status = f"*[Using trained model: {model_name}]*\n\n"
                            
                            logger.info(f"Generated response length: {len(response)}")
                            
                        except Exception as e:
                            logger.error(f"Error with trained model generation: {e}")
                            # Fallback to document search
                            response = st.session_state.llm_handler._search_documents(prompt)
                            model_status = f"*[Using document search - model generation failed]*\n\n"
                    
                    # Fallback to chatbot if no trained model
                    elif st.session_state.chatbot and st.session_state.chatbot.documents:
                        try:
                            # Ensure chatbot has access to llm_handler
                            st.session_state.chatbot.llm_handler = st.session_state.llm_handler
                            response = st.session_state.chatbot.generate_response(prompt)
                            model_status = f"*[Using basic chatbot - train a model for better results]*\n\n"
                            
                        except Exception as e:
                            logger.error(f"Error with chatbot generation: {e}")
                            response = "I apologize, but I'm having trouble generating a response. Please try again."
                            model_status = f"*[Error in response generation]*\n\n"
                    
                    else:
                        response = "Please upload and process some documents first, then train a model to get better responses."
                        model_status = f"*[No documents or model available]*\n\n"
                    
                    # Ensure we have a valid response
                    if not response or response.strip() == "":
                        response = "I couldn't generate a meaningful response to your query. Please try rephrasing your question."
                    
                    # Combine status and response
                    full_response = model_status + response
                    
                    st.markdown(full_response)
                    chat["messages"].append({"role": "assistant", "content": full_response})
                    
                    # Log the interaction
                    logger.info(f"User query: {prompt}")
                    logger.info(f"Response generated: {len(response)} characters")

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    logger.error(f"Chat error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    st.error(error_msg)
                    chat["messages"].append({"role": "assistant", "content": error_msg})

        st.rerun()


# Main application
def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )

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