import streamlit as st
import os
import json
import hashlib
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, Optional
import time
from utils.domain_detector import detect_domain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration flags
LLM_AVAILABLE = True
NLTK_AVAILABLE = True

# Mock classes for missing dependencies
class MockIntentClassifier:
    def predict(self, text): return "default"
    def train_model(self, data): pass

class MockLLMHandler:
    def generate_response(self, prompt, task=None): return f"Mock response for: {prompt}"
    def index_documents(self, docs): return True
    def load_index(self, path): pass

class MockQAHandler:
    def answer_question(self, question, context): return f"Mock answer for: {question}"

class MockPreprocessor:
    def preprocess_file(self, content): return content, {"rows": len(content.split('\n'))}

# Try to import real classes, fall back to mocks
try:
    from llm_handler import LLMHandler
    from intent.classifier import IntentClassifier
    from models.qa_model import QAHandler
    from utils.preprocessing import Preprocessor
    from intent.classifier import IntentClassifier
except ImportError as e:
    logger.warning(f"Import error: {e}. Using mock classes.")
    LLMHandler = MockLLMHandler
    IntentClassifier = MockIntentClassifier
    QAHandler = MockQAHandler
    Preprocessor = MockPreprocessor
    def train_model(data): return None, 0.85

class ChatBot:
    def __init__(self):
        self.datasets = {}
        self.documents = {}
        self.trained_models = {}
        self.trained_models_path = "trained_models.json"
        
        # Initialize handlers
        self.LLM_AVAILABLE = True
        self.llm_handler = LLMHandler() if LLM_AVAILABLE else None
        self.intent_classifier = IntentClassifier() if NLTK_AVAILABLE else MockIntentClassifier()
        self.qa_handler = QAHandler()
        self.preprocessor = Preprocessor()
        
        logger.info("ChatBot initialized")
            # FIX 1: Load previously saved documents and models on initialization
        self._load_persistent_state()
        
        logger.info("ChatBot initialized")
    def _load_persistent_state(self):
        """Load documents and trained models from persistent storage"""
        try:
            # Load documents from session state if available (Streamlit)
            if 'st' in globals() and hasattr(st, 'session_state'):
                if 'chatbot_documents' in st.session_state:
                    self.documents = st.session_state.chatbot_documents
                    logger.info(f"Loaded {len(self.documents)} documents from session state")
                
                if 'chatbot_trained_models' in st.session_state:
                    self.trained_models = st.session_state.chatbot_trained_models
                    logger.info(f"Loaded {len(self.trained_models)} trained models from session state")
            
            # Load from file system as backup
            documents_path = "persistent_documents.json"
            if os.path.exists(documents_path):
                with open(documents_path, 'r', encoding='utf-8') as f:
                    file_documents = json.load(f)
                    # Merge with session state, prioritizing session state
                    for name, doc in file_documents.items():
                        if name not in self.documents:
                            self.documents[name] = doc
                    logger.info(f"Loaded {len(file_documents)} documents from file system")
            
            # Restore documents to LLMHandler
            if self.llm_handler and self.documents:
                docs_to_index = [(name, doc.get('content', '')) for name, doc in self.documents.items()]
                success = self.llm_handler.index_documents(docs_to_index, force_reindex=True)
                if success:
                    logger.info(f"Restored {len(docs_to_index)} documents to LLMHandler")
                else:
                    logger.warning("Failed to restore documents to LLMHandler")
                    
        except Exception as e:
            logger.error(f"Error loading persistent state: {e}")

    def _save_persistent_state(self):
        """Save documents and trained models to persistent storage"""
        try:
            # Save to session state if available (Streamlit)
            if 'st' in globals() and hasattr(st, 'session_state'):
                st.session_state.chatbot_documents = self.documents
                st.session_state.chatbot_trained_models = self.trained_models
                logger.info("Saved state to session state")
            
            # Save to file system as backup
            documents_path = "persistent_documents.json"
            with open(documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.documents)} documents to file system")
                
        except Exception as e:
            logger.error(f"Error saving persistent state: {e}")

    def generate_response(self, prompt: str, task: Optional[str] = None) -> str:
        """Generate response based on user prompt and optional task."""
        logger.info(f"[USER] {prompt}")

        # FIX: Check both ChatBot documents AND LLMHandler documents
        has_chatbot_docs = bool(self.documents)
        has_llm_docs = bool(self.llm_handler and hasattr(self.llm_handler, 'doc_texts') and self.llm_handler.doc_texts)

        # Log the state for debugging
        logger.info(f"ChatBot documents: {len(self.documents)}, LLMHandler documents: {len(getattr(self.llm_handler, 'doc_texts', {}))}")

        # Only show upload message if BOTH are empty
        if not has_chatbot_docs and not has_llm_docs:
            return "Hello! Please upload some files first so I can analyze them and provide better responses."

        if not self.llm_handler:
            return "LLM Handler is not available. Please check your configuration."

        try:
            # Detect domain from prompt
            domain_name = detect_domain(prompt)
            logger.info(f"Detected domain: {domain_name}")

            # Load domain-specific model if available
            self.llm_handler.load_model(model_path=f"rag_data/{domain_name}/{domain_name}_checkpoint.pt",
                                     tokenizer_path=f"rag_data/{domain_name}/uml_tokenizer.pkl")
            # Auto-detect task if not provided
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

            # Predict intent
            intent = self.intent_classifier.predict(prompt)
            logger.info(f"[TASK] {task}, [INTENT] {intent}")

            # Check for trained models
            relevant_doc_id = self._get_relevant_doc_id()
            if relevant_doc_id and relevant_doc_id in self.trained_models:
                response = self._use_trained_model(prompt, relevant_doc_id)
            else:
                # This is where the actual LLM response generation happens
                response = self.llm_handler.generate_response(prompt, task)

            logger.info(f"[BOT] {response}")
            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I apologize, but I encountered an error. Please try rephrasing your question."

    def _get_relevant_doc_id(self) -> Optional[str]:
        """Get the most relevant document ID for the current context."""
        for doc_id, dataset in self.datasets.items():
            if dataset['filename'] in self.documents:
                return doc_id
        return None

    def _use_trained_model(self, prompt: str, doc_id: str) -> str:
        """Use trained model for prediction."""
        model_info = self.trained_models[doc_id]
        model_type = model_info['model_type']
        
        if model_type == "classification":
            prediction = self._apply_classification_model(prompt, doc_id)
            return f"Classification result: {prediction}"
        elif model_type == "regression":
            prediction = self._apply_regression_model(prompt, doc_id)
            return f"Regression result: {prediction:.2f}"
        else:
            return self.llm_handler.generate_response(prompt)

    def _apply_classification_model(self, prompt: str, doc_id: str) -> str:
        """Apply classification model (placeholder implementation)."""
        # In a real implementation, load and apply the actual model
        return "Positive"  # Mock prediction

    def _apply_regression_model(self, prompt: str, doc_id: str) -> float:
        """Apply regression model (placeholder implementation)."""
        # In a real implementation, load and apply the actual model
        return 0.85  # Mock prediction

    def add_document(self, name: str, doc_dict: Dict):
        """Add document to the chatbot's knowledge base."""
        try:
            logger.info(f"Adding document: {name}")
            content = doc_dict.get("content", "")
            if not content.strip():
                logger.error(f"Empty content for {name}")
                return False
            
            # Store document in self.documents
            self.documents[name] = {
                "content": content,
                "metadata": doc_dict.get("metadata", {})
            }
            logger.info(f"Document {name} added to self.documents with {len(content)} characters")

            # Save to persistent storage
            self._save_persistent_state()
            # Index document if LLM handler is available
            if self.llm_handler and content:
                success = self.llm_handler.index_documents([(name, content)], force_reindex=True)
                if success:
                    logger.info(f"Document {name} indexed successfully")
                else:
                    logger.error(f"Failed to index document {name}")
                    return False
            # Train intent classifier if available
            if NLTK_AVAILABLE:
                intent_data = doc_dict.get("intent_data", 
                    pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]}))
                if not isinstance(intent_data, pd.DataFrame) or intent_data.empty:
                    intent_data = pd.DataFrame({"sentence": ["Default sentence"], "intent": ["default"]})
                    logger.warning(f"Using default intent data for {name}")
                
                self.intent_classifier.train_model(intent_data)
                logger.info(f"IntentClassifier trained for {name}")
            

            # Check if document contains tabular data
            if content and any(delimiter in content for delimiter in [',', '\t', '|']):
                doc_id = hashlib.md5(name.encode()).hexdigest()
                lines = content.split('\n')
                self.datasets[doc_id] = {
                    'data': lines,
                    'filename': name,
                    'type': 'tabular'
                }
                # Auto-train model for tabular data
                self.auto_train_models(doc_id)
                        # Save persistent state again after indexing and training
            self._save_persistent_state()
            return True
    
        except Exception as e:
            logger.error(f"Error adding document {name}: {e}", exc_info=True)
            return False
    
    def debug_document_state(self):
        """Debug method to check the state of documents in both ChatBot and LLMHandler"""
        print("=== DOCUMENT STATE DEBUG ===")

        # Check ChatBot documents
        print(f"ChatBot.documents: {len(self.documents)} items")
        for name, doc in self.documents.items():
            content_length = len(doc.get('content', ''))
            print(f"  - {name}: {content_length} characters")

        # Check LLMHandler documents
        if self.llm_handler:
            llm_docs = getattr(self.llm_handler, 'doc_texts', {})
            print(f"LLMHandler.doc_texts: {len(llm_docs)} items")
            for name, doc in llm_docs.items():
                content_length = len(doc.get('content', ''))
                sentences = len(doc.get('sentences', []))
                print(f"  - {name}: {content_length} characters, {sentences} sentences")

            # Check if LLMHandler is trained
            print(f"LLMHandler.is_trained: {getattr(self.llm_handler, 'is_trained', False)}")
            print(f"LLMHandler.tokenizer.trained: {getattr(self.llm_handler.tokenizer, 'trained', False) if self.llm_handler.tokenizer else False}")
        else:
            print("LLMHandler: Not available")

        print("=== END DEBUG ===")
    
    def force_reindex_documents(self):
        """Force re-indexing of all documents in LLMHandler"""
        if not self.llm_handler:
            logger.error("LLMHandler not available")
            return False
        
        if not self.documents:
            logger.warning("No documents to re-index")
            return False
        
        try:
            docs_to_index = [(name, doc.get('content', '')) for name, doc in self.documents.items()]
            success = self.llm_handler.index_documents(docs_to_index, force_reindex=True)
            if success:
                logger.info(f"Successfully re-indexed {len(docs_to_index)} documents")
                return True
            else:
                logger.error("Failed to re-index documents")
                return False
        except Exception as e:
            logger.error(f"Error re-indexing documents: {e}")
            return False
        
    def auto_train_models(self, doc_id: str) -> str:
        """Automatically train models for the given document."""
        try:
            if doc_id not in self.datasets:
                return f"Dataset not found for {doc_id}"

            dataset = self.datasets[doc_id]
            data = dataset.get("data", [])
            doc_name = dataset.get("filename", "")
            size = len(data)

            # Determine model type based on data size
            model_type = "classification" if size > 10 else "regression"
            logger.info(f"Training {model_type} model for {doc_name} with {size} lines")

            start_time = time.time()

            # Get document content for preprocessing
            content = self.documents.get(doc_name, {}).get("content", "")
            if not content:
                return f"No content found for {doc_name}"

            # Preprocess data
            processed_data, metadata, intent_data = self.preprocessor.preprocess_file(content)

            # Convert to DataFrame if needed
            if isinstance(processed_data, str):
                processed_data = pd.DataFrame({"text": processed_data.split("\n")})

            if processed_data.empty:
                return f"Failed to preprocess document: empty data"

            # Log intent data for debugging zero-loss issue
            logger.info(f"Intent data for {doc_name}: shape={intent_data.shape}, unique intents={intent_data['intent'].unique()}")

            # Skip training if only one intent is present (to avoid zero-loss issue)
            if intent_data['intent'].nunique() <= 1:
                logger.warning(f"Skipping IntentClassifier training for {doc_name}: only one intent found")
                return f"⚠️ No training performed for {doc_name}: insufficient intent variety"

            # Train model
            success, accuracy, model = self.intent_classifier.train_model(intent_data)
            if not success:
                logger.error(f"Failed to train IntentClassifier for {doc_name}")
                return f"❌ Failed to train model for {doc_name}"

            # Store model information
            self.trained_models[doc_id] = {
                "model_type": model_type,
                "accuracy": accuracy,
                "features": len(processed_data.columns) if hasattr(processed_data, 'columns') else 1,
                "trained_at": datetime.now().isoformat(),
                "model": model,
                "metadata": metadata
            }
            #st.session_state.losses = losses  # Store for visualization
            total_time = time.time() - start_time
            logger.info(f"Trained {model_type} model for {doc_name} in {total_time:.2f} seconds")

            return f"✅ {model_type.title()} model trained with {accuracy:.2%} accuracy"

        except Exception as e:
            logger.error(f"Error training model for {doc_id}: {e}")
            return f"❌ Error training model: {str(e)}"

    def load_saved_model(self, model_version: str) -> str:
        """Load a previously saved model."""
        try:
            if not os.path.exists(self.trained_models_path):
                return f"❌ No saved models found"

            with open(self.trained_models_path, "r") as f:
                models = json.load(f)

            if model_version not in models:
                return f"❌ Model version `{model_version}` not found"

            model_info = models[model_version]
            
            # Load model components
            if self.llm_handler and "vectorstore_path" in model_info:
                self.llm_handler.load_index(model_info["vectorstore_path"])
            
            # Restore document info
            self.documents[model_info["doc_name"]] = {
                "content": "",
                "metadata": model_info.get("metadata", {})
            }
            
            # Restore trained model info
            doc_id = hashlib.md5(model_info["doc_name"].encode()).hexdigest()
            self.trained_models[doc_id] = {
                "model_type": model_info.get("model_type", "classification"),
                "accuracy": model_info.get("accuracy", 0.0),
                "features": model_info.get("features", 0),
                "trained_at": model_info.get("trained_at", datetime.now().isoformat())
            }
            
            logger.info(f"✅ Loaded model `{model_version}`")
            return f"✅ Model `{model_version}` loaded successfully"
            
        except Exception as e:
            logger.error(f"❌ Failed loading model {model_version}: {e}")
            return f"❌ Failed loading model: {e}"

    def _get_help_response(self) -> str:
        """Return help message for available commands."""
        return (
            "I can help you with the following tasks:\n"
            "- **Answer**: Ask questions about uploaded documents\n"
            "- **Analyze**: Perform data analysis on tabular data\n"
            "- **Summarize**: Generate summaries of documents\n"
            "Please upload documents or specify a task in your prompt."
        )

# Usage example
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Example document
    sample_doc = {
        "content": "This is a sample document with some text content.",
        "metadata": {"type": "text", "source": "sample"}
    }
    
    # Add document
    chatbot.add_document("sample.txt", sample_doc)
    
    # Generate response
    response = chatbot.generate_response("What is this document about?")
    print(response)