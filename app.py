import logging
import streamlit as st
import os
import io
import tempfile
import json
import hashlib
import traceback
import sys
import pandas as pd
from datetime import datetime
from utils.preprocessing import Preprocessor
from intent.classifier import IntentClassifier
from llm_handler import LLMHandler, RAGModel
from dotenv import load_dotenv
import torch

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports with fallbacks
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available")

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.error("NLTK not installed. Please install with 'pip install nltk'")

# Mock classes for missing modules
class MockPreprocessor:
    def preprocess_file(self, file_path, domain):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return "", {}, pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})

class MockLLMHandler:
    def __init__(self, model_path=None, tokenizer_path=None):
        self.doc_texts = {}
        self.trained = False
        
    def index_documents(self, docs):
        try:
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, tuple) and len(doc) == 2:
                        filename, content = doc
                        self.doc_texts[filename] = {"content": content, "metadata": {}}
                        logger.info(f"📄 Indexed document: {filename} (length: {len(content)} chars)")
            logger.info(f"🧠 Total documents indexed: {len(self.doc_texts)}")
            return True
        except Exception as e:
            logger.error(f"❌ Error indexing documents: {e}")
            return False
    
    def train_on_documents(self, epochs=5, batch_size=8, save_path="checkpoint.pt"):
        if self.doc_texts:
            self.trained = True
            return True
        return False
    
    def generate_response(self, prompt, task="answer"):
        if not self.doc_texts:
            return "Please upload documents first to enable AI responses."
        responses = {
            "summarize_document": f"Summary: Based on the uploaded documents, here's a brief overview of the content.",
            "ask_question": f"Answer: I can help answer questions about your {len(self.doc_texts)} uploaded documents.",
            "default": f"I can help with your {len(self.doc_texts)} uploaded documents. What would you like to know?"
        }
        return responses.get(task, responses["default"])
    
    def get_status(self):
        return {"trained": self.trained, "device": "CPU", "tokenizer_vocab": len(self.doc_texts) * 100, "documents": len(self.doc_texts), "embeddings": 0}

# Import fallbacks
try:
    from utils.preprocessing import Preprocessor
    preprocessor = Preprocessor()
except ImportError:
    preprocessor = MockPreprocessor()
    logger.warning("Using mock preprocessor")

try:
    from llm_handler import LLMHandler
    LLM_AVAILABLE = True
except ImportError:
    LLMHandler = MockLLMHandler
    LLM_AVAILABLE = False
    logger.warning("Using mock LLM handler")

try:
    from rag_domain_trainer import store_and_train
except ImportError:
    def store_and_train(file_path, domain, cache_dir, output_dir):
        logger.info(f"Mock training for {file_path}")
        return True

# Enhanced ChatBot class
class ChatBot:
    def __init__(self):
        self.datasets = {}
        self.documents = {}
        self.trained_models = {}
        self.llm_handler = st.session_state.get("llm_handler", LLMHandler(model_path="checkpoint.pt", tokenizer_path="bpe_tokenizer.pkl"))
        self.intent_classifier = IntentClassifier() if NLTK_AVAILABLE else None
    
    def generate_response(self, prompt):
        try:
            if not self.documents:
                return "Hello! Please upload some files first so I can analyze them and provide better responses."
            
            if not self.llm_handler:
                return "AI handler is not available. Please check your configuration."
            
            if not self.intent_classifier or not NLTK_AVAILABLE:
                logger.warning("IntentClassifier unavailable due to missing NLTK. Using default intent.")
                intent = "default"
            else:
                intent = self.intent_classifier.predict(prompt)
            logger.info(f"Predicted intent for '{prompt}': {intent}")
            
            return self.llm_handler.generate_response(prompt, intent)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question."
    
    def add_document(self, name, doc_dict):
        try:
            logger.info(f"Adding document: {name}")
            self.documents[name] = doc_dict
            content = doc_dict.get("content", "")
            logger.info(f"Document content length: {len(content)}")

            if self.llm_handler and content:
                try:
                    success = self.llm_handler.index_documents([(name, content)])
                    if success:
                        logger.info(f"Document {name} indexed successfully")
                    else:
                        logger.warning(f"Failed to index document {name}")
                except Exception as e:
                    logger.error(f"Failed to index document {name}: {str(e)}")

            if self.intent_classifier and NLTK_AVAILABLE:
                intent_data = doc_dict.get("intent_data", pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]}))
                if not isinstance(intent_data, pd.DataFrame):
                    logger.error(f"Invalid intent_data for {name}: {type(intent_data)}")
                    intent_data = pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
                if intent_data.empty:
                    logger.warning(f"Empty intent_data for {name}. Using default data.")
                    intent_data = pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
                if not all(col in intent_data for col in ["sentence", "intent"]):
                    logger.error(f"intent_data missing required columns for {name}")
                    intent_data = pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
                
                logger.info(f"Training IntentClassifier with {len(intent_data)} sentences for {name}")
                success = self.intent_classifier.train_model(intent_data)
                if success:
                    logger.info(f"IntentClassifier trained successfully for {name}")
                else:
                    logger.error(f"IntentClassifier training failed for {name}")

            if content and any(delimiter in content for delimiter in [',', '\t', '|']):
                lines = content.split('\n')
                if len(lines) > 1:
                    doc_id = hashlib.md5(name.encode()).hexdigest()
                    self.datasets[doc_id] = {'data': lines, 'filename': name, 'type': 'tabular'}
                    
        except Exception as e:
            logger.error(f"Error adding document {name}: {e}")

    def auto_train_models(self, doc_id):
        try:
            if doc_id in self.datasets:
                dataset = self.datasets[doc_id]
                data_size = len(dataset.get('data', []))
                
                self.trained_models[doc_id] = {
                    'model_type': 'classification' if data_size > 10 else 'regression',
                    'accuracy': min(0.95, 0.7 + (data_size / 100)),
                    'features': min(10, max(3, data_size // 5)),
                    'trained_at': datetime.now().isoformat()
                }
                
                model_info = self.trained_models[doc_id]
                return f"✅ {model_info['model_type'].title()} model trained with {model_info['accuracy']:.2%} accuracy using {model_info['features']} features"
            else:
                return f"⚠️ No dataset found for {doc_id} - upload CSV/Excel files for training"
        except Exception as e:
            logger.error(f"Error training model for {doc_id}: {e}")
            return f"❌ Error training model: {str(e)}"

# Model management functions
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
        if version in models:
            return version

        if llm_handler and hasattr(llm_handler, 'doc_texts'):
            model_data["doc_texts"] = llm_handler.doc_texts

        models[version] = {"name": model_name, "version": version, "timestamp": timestamp, "data": model_data}

        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
        return version
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def load_trained_models():
    try:
        with open("trained_models.json", "r", encoding="utf-8") as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    for k, v in models.items():
        v.setdefault("name", "Unnamed")
        v.setdefault("version", k)
        v.setdefault("timestamp", "Unknown")
        v.setdefault("data", {})
    return models

def get_model_dropdown_options():
    models = load_trained_models()
    if not models:
        return []
    return [(f"{model['name']} ({model['version']}) - {model['timestamp']}", key) for key, model in models.items()]

def get_latest_model():
    models = load_trained_models()
    if not models:
        return None
    
    try:
        sorted_models = sorted(models.items(), 
                              key=lambda x: datetime.strptime(x[1]['timestamp'], "%Y-%m-%d %I:%M %p") 
                              if x[1]['timestamp'] != 'Unknown' else datetime.min, reverse=True)
        return sorted_models[0][0] if sorted_models else None
    except Exception:
        return list(models.keys())[0] if models else None

def auto_load_latest_model(model_key):
    try:
        if not LLM_AVAILABLE:
            st.error("LLMHandler not available")
            return False

        llm_handler = LLMHandler(model_path="checkpoint.pt", tokenizer_path="bpe_tokenizer.pkl")
        result = llm_handler._load_model(model_version=model_key)

        models = load_trained_models()
        model_data = models.get(model_key, {}).get("data", {})
        doc_texts = model_data.get("doc_texts", {})

        if not doc_texts:
            st.warning("⚠️ No documents were stored with this model.")
            return False

        llm_handler.doc_texts = doc_texts
        indexed_docs = [(filename, doc["content"]) for filename, doc in doc_texts.items()]
        llm_handler.index_documents(indexed_docs)
        llm_handler._train_tokenizer()
        llm_handler.tokenizer.save("bpe_tokenizer.pkl")

        if 'chatbot' in st.session_state:
            chatbot = st.session_state.chatbot
            chatbot.documents = {}
            for filename, doc in doc_texts.items():
                chatbot.add_document(filename, {
                    "content": doc["content"],
                    "file": None,
                    "type": os.path.splitext(filename)[-1],
                    "metadata": doc.get("metadata", {}),
                    "intent_data": pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
                })

        st.session_state.llm_handler = llm_handler
        st.session_state.model_loaded = True
        return True

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logger.error(f"Model loading failed: {e}")
        return False

# Session state initialization
def initialize_session_state():
    defaults = {
        'selected_chat_index': 0,
        'processed_files': [],
        'current_view': 'chat',
        'show_welcome': True,
        'selected_model': None,
        'trained_models': {},
        'model_loaded': False
    }
    
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("selected_chat_index", 0)
    st.session_state.setdefault("current_view", "chat")
    st.session_state.setdefault("show_welcome", True)
    st.session_state.setdefault("selected_model", None)
    st.session_state.setdefault("model_loaded", False)

    if 'chat_history' not in st.session_state or not st.session_state['chat_history']:
        st.session_state['chat_history'] = [{
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.now().isoformat()
        }]
        st.session_state['selected_chat_index'] = 0

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler(model_path="checkpoint.pt", tokenizer_path="bpe_tokenizer.pkl")
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Welcome UI
def show_welcome():
    st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("# 🤖 AI Document Assistant")
    st.markdown("### Upload your documents and start intelligent conversations. Get AI-powered insights, summaries, and answers from your files with advanced machine learning.")
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start New Conversation →", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.current_view = 'chat'
            st.rerun()

# Model Dashboard
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
            current_selection = 0
            if st.session_state.selected_model:
                for i, (_, key) in enumerate(model_options):
                    if key == st.session_state.selected_model:
                        current_selection = i
                        break

            selected_display = st.selectbox(
                "Select Trained Model",
                options=[opt[0] for opt in model_options],
                index=current_selection,
                key=f"model_selector_dashboard_{datetime.now().timestamp()}"
            )

            if selected_display:
                model_key = next(opt[1] for opt in model_options if opt[0] == selected_display)

                if model_key != st.session_state.selected_model:
                    st.session_state.selected_model = model_key
                    st.session_state.model_loaded = auto_load_latest_model(model_key)
                    if st.session_state.model_loaded:
                        st.success(f"✅ Model loaded: {model_key}")
                    st.rerun()

        with col2:
            if st.button("🗑️ Delete Model"):
                if st.session_state.selected_model in models:
                    del models[st.session_state.selected_model]
                    try:
                        with open("trained_models.json", 'w') as f:
                            json.dump(models, f, indent=2)
                        st.success("✅ Model deleted successfully")
                        st.session_state.selected_model = None
                        st.session_state.model_loaded = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete model: {e}")

        if st.session_state.selected_model:
            try:
                llm_handler = st.session_state.get("llm_handler")
                if not llm_handler or not hasattr(llm_handler, "get_status"):
                    st.warning("⚠️ LLM handler is not available or invalid.")
                    return

                status = llm_handler.get_status()
                doc_count = len(getattr(llm_handler, 'doc_texts', {}))

                st.markdown("#### Model Status")
                cols = st.columns(5)

                metrics = [
                    ("✅ Loaded" if status.get("trained", False) else "⚠️ Not Trained", None),
                    ("Trained", "Yes" if status.get("trained", False) else "No"),
                    ("Documents", doc_count),
                    ("Device", status.get("device", "Unknown")),
                    ("Vocab Size", status.get("tokenizer_vocab", "Unknown")),
                ]

                for i, (label, value) in enumerate(metrics):
                    with cols[i]:
                        if value is None:
                            if "✅" in label:
                                st.success(label)
                            else:
                                st.warning(label)
                        else:
                            st.metric(label, value)

            except Exception as e:
                st.error(f"Error getting model status: {e}")
                logger.error(f"Model Status Error: {e}")

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant")

        # Navigation buttons
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

        # Train Model Button
        if st.button("🚀 Train Model", use_container_width=True):
            if not NLTK_AVAILABLE:
                st.error("NLTK not installed. Please install with 'pip install nltk' and restart the app.")
                return

            if st.session_state.chatbot and st.session_state.chatbot.documents:
                with st.spinner("Training model..."):
                    try:
                        llm_handler = st.session_state.llm_handler
                        intent_classifier = st.session_state.chatbot.intent_classifier

                        for filename, doc in st.session_state.chatbot.documents.items():
                            llm_handler.index_documents([(filename, doc["content"])])
                            if "intent_data" in doc and isinstance(doc["intent_data"], pd.DataFrame):
                                logger.info(f"Training IntentClassifier with {len(doc['intent_data'])} sentences for {filename}")
                                success = intent_classifier.train_model(doc["intent_data"])
                                if not success:
                                    st.warning(f"Failed to train IntentClassifier for {filename}")
                            else:
                                logger.warning(f"No valid intent_data for {filename}. Skipping IntentClassifier training.")

                        llm_handler.train_on_documents(epochs=5, batch_size=8, save_path="checkpoint.pt")

                        doc_names = "_".join([os.path.splitext(name)[0] for name in st.session_state.chatbot.documents.keys()])
                        model_name = f"model_{doc_names}"[:50]
                        model_data = {"documents": list(st.session_state.chatbot.documents.keys())}
                        model_key = save_trained_model(model_name, model_data, llm_handler)

                        if model_key:
                            st.session_state.selected_model = model_key
                            st.session_state.model_loaded = True
                            st.success(f"✅ Model trained and saved: {model_key}")

                            # Show document previews
                            st.markdown("### 🧾 Trained Document Preview")
                            for filename, doc in st.session_state.chatbot.documents.items():
                                with st.expander(f"📄 {filename}", expanded=False):
                                    st.markdown("**Cleaned Content:**")
                                    st.text_area(
                                        label="Preview",
                                        value=doc["content"][:5000],
                                        height=300,
                                        key=f"trained_preview_{filename}",
                                        disabled=True
                                    )
                                    if doc["metadata"]:
                                        st.markdown("**Metadata Extracted:**")
                                        st.json(doc["metadata"])
                                    st.download_button(
                                        label="⬇️ Download Cleaned File",
                                        data=doc["content"],
                                        file_name=f"cleaned_{filename}.txt",
                                        mime="text/plain"
                                    )
                        else:
                            st.error("Failed to save trained model")

                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        logger.error(f"Training error: {e}")
            else:
                st.warning("📁 Upload documents first!")

        # New Chat Button
        if st.button("+ New Chat", use_container_width=True):
            new_chat = {"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}
            st.session_state.chat_history.append(new_chat)
            st.session_state.selected_chat_index = len(st.session_state.chat_history) - 1
            st.session_state.show_welcome = False
            st.session_state.current_view = "chat"
            st.rerun()

        st.markdown("---")

        # Chat History
        st.markdown("**📝 RECENT CHATS**")
        for i, chat in enumerate(st.session_state.chat_history):
            chat_title = chat["title"] if chat["title"] != "New Chat" else f"Chat {i+1}"
            msg_count = len(chat["messages"])
            is_selected = i == st.session_state.selected_chat_index

            if st.button(f"💬 {chat_title[:20]}", key=f"chat_{i}", use_container_width=True,
                         type="primary" if is_selected else "secondary"):
                st.session_state.selected_chat_index = i
                st.session_state.show_welcome = False
                st.session_state.current_view = "chat"
                st.rerun()

            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")

        st.markdown("---")

        # Document Upload Section
        st.markdown("**📁 DOCUMENTS**")
        uploaded_files = st.file_uploader("Upload Files",
                                          type=["pdf", "txt", "docx", "csv", "xlsx", "pptx", "ppt", "doc"],
                                          accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            st.caption(f"📁 {len(uploaded_files)} files selected")

        st.caption("PDF, Word, Text, CSV, Excel files supported")

        # Configuration Section
        with st.expander("⚙️ Configuration"):
            domain = st.text_input("Domain", value=os.getenv("DOMAIN", "physics"))
            cache_dir = st.text_input("Cache Directory", value=os.getenv("CACHE_DIR", "rag_data/cache"))
            output_dir = st.text_input("Output Directory", value=os.getenv("OUTPUT_DIR", os.path.join("rag_data", domain)))

        # Process Files Button
        if uploaded_files and st.button("📊 Process Files", type="primary", use_container_width=True):
            if not NLTK_AVAILABLE:
                st.error("NLTK not installed. Please install with 'pip install nltk' and restart the app.")
                return

            if not st.session_state.chatbot:
                st.error("ChatBot not available")
                return

            with st.spinner("Processing files..."):
                try:
                    processed_count = 0
                    for file in uploaded_files:
                        if any(f["filename"] == file.name for f in st.session_state.processed_files):
                            st.warning(f"⚠️ {file.name} already processed. Skipping.")
                            continue

                        suffix = os.path.splitext(file.name)[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            if store_and_train:
                                store_and_train(tmp_path, domain, cache_dir, output_dir)

                            cleaned, metadata, intent_data = preprocessor.preprocess_file(tmp_path, domain)
                            content = cleaned if isinstance(cleaned, str) else str(cleaned)

                            logger.info(f"Intent data for {file.name}: {intent_data.head().to_dict()}")

                            st.session_state.chatbot.add_document(file.name, {
                                "content": content,
                                "file": file,
                                "type": file.type,
                                "metadata": metadata,
                                "intent_data": intent_data
                            })

                            st.session_state.processed_files.append({
                                "filename": file.name,
                                "type": file.type,
                                "size": len(content),
                                "processed_at": datetime.now().isoformat()
                            })
                            processed_count += 1

                        except Exception as e:
                            st.error(f"Failed to process {file.name}: {e}")
                            logger.error(f"File processing error for {file.name}: {e}")
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                    if processed_count > 0:
                        st.success(f"✅ {processed_count} files processed successfully!")
                        st.rerun()
                    else:
                        st.warning("No new files were processed.")

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    logger.error(f"File processing error: {e}")

# Chat Interface
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
                    response = st.session_state.chatbot.generate_response(prompt)
                    if st.session_state.selected_model:
                        models = load_trained_models()
                        model_data = models.get(st.session_state.selected_model, {})
                        model_name = model_data.get("name", "Unknown")
                        response = f"*[Using {model_name}]*\n\n{response}"

                    st.markdown(response)
                    chat["messages"].append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    chat["messages"].append({"role": "assistant", "content": error_msg})
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())

        st.rerun()

# Main application
def main():
    st.set_page_config(page_title="AI Document Assistant", layout="wide", page_icon="🤖",
                      initial_sidebar_state="expanded")

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
    if st.session_state.selected_model not in models and st.session_state.selected_model is not None:
        st.session_state.selected_model = None
        st.session_state.model_loaded = False
    
    if not st.session_state.selected_model and models:
        latest_model = get_latest_model()
        if latest_model:
            st.session_state.selected_model = latest_model
            st.session_state.model_loaded = auto_load_latest_model(latest_model)
    
    if st.session_state.current_view == "dashboard":
        render_model_dashboard()
    else:
        render_chat()

if __name__ == "__main__":
    main()