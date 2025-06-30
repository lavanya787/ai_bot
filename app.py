import streamlit as st
import os
import tempfile
from datetime import datetime
import logging
from io import BytesIO
import sys
# Safe imports
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Custom modules
from file_processing.processor import extract_text_from_file
from utils.preprocessing import Preprocessor
from utils.logger import Logger
from chatbot_module import ChatBot
from model_orchestrator import model_training_ui
from scripts.auto_domain_mover import move_file_to_domain_folder
from utils.pdf_export import export_chat_to_pdf
from utils.voice_input import record_and_transcribe

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding='utf-8')

log = Logger().logger
preprocessor = Preprocessor()

# Session state
def initialize_session_state():
    defaults = {
        'chat_history': [{"title": "Welcome Chat", "messages": [], "created_at": datetime.now().isoformat()}],
        'selected_chat_index': 0,
        'processed_files': [],
        'current_view': 'chat',
        'show_welcome': True
    }
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = ChatBot()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Welcome UI - Updated to match the landing page design
def show_welcome():
    st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("# 🤖 AI Document Assistant")
    st.markdown("### Upload your documents and start intelligent conversations. Get AI-powered insights, summaries, and answers from your files with advanced machine learning.")
    
    # Feature cards in columns
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; margin: 1rem 0;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>📄</div>
            <h4>Document Processing</h4>
            <p>Upload PDFs, Word docs, text files, and spreadsheets for intelligent analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; margin: 1rem 0;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>💬</div>
            <h4>Smart Conversations</h4>
            <p>Ask questions about your documents and get contextual, intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; margin: 1rem 0;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>🧠</div>
            <h4>AI Training</h4>
            <p>Train custom models on your data for enhanced performance and accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start New Conversation →", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

# Sidebar - Simplified and cleaner
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant")
        
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            new_chat = {"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}
            st.session_state.chat_history.append(new_chat)
            st.session_state.selected_chat_index = len(st.session_state.chat_history) - 1
            st.session_state.show_welcome = False
            st.rerun()

        st.markdown("---")
        st.markdown("**RECENT CHATS**")
        
        for i, chat in enumerate(st.session_state.chat_history):
            chat_title = chat["title"] if chat["title"] != "New Chat" else f"Chat {i+1}"
            msg_count = len(chat["messages"])
            
            if st.button(f"💬 {chat_title[:20]}", key=f"chat_{i}", use_container_width=True):
                st.session_state.selected_chat_index = i
                st.session_state.show_welcome = False
                st.rerun()
            
            if msg_count > 0:
                st.caption(f"{msg_count} messages")

        st.markdown("---")
        st.markdown("**DOCUMENTS**")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Files", 
            type=["pdf", "txt", "docx", "csv", "xlsx"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.caption("PDF, Word, Text, CSV, Excel files supported")

        if uploaded_files and st.button("Process Files", type="secondary", use_container_width=True):
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    content = extract_text_from_file(tmp_path, file.name)
                    st.session_state.chatbot.add_document(file.name, {"content": content, "file": file, "type": file.type})

                    st.session_state.processed_files.append({
                        "filename": file.name,
                        "type": file.type,
                        "size": len(content),
                        "processed_at": datetime.now().isoformat()
                    })

                    try:
                        move_file_to_domain_folder(tmp_path)
                    except Exception as e:
                        st.error(f"Error moving file: {e}")

                    os.unlink(tmp_path)

            st.success("✅ Files processed!")
            st.rerun()

# Chat area - Updated to match the conversation interface
def render_chat():
    if st.session_state.show_welcome:
        show_welcome()
        return

    chat_idx = st.session_state.selected_chat_index
    chat = st.session_state.chat_history[chat_idx]

    # Chat header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 💬 {chat['title']}")
        st.caption(f"0 messages • Last updated {datetime.now().strftime('%I:%M %p')}")
    with col2:
        if chat["messages"] and st.button("🧹", help="Clear Chat"):
            chat["messages"] = []
            st.rerun()

    # Empty state when no messages
    if not chat["messages"]:
        st.markdown("<div style='text-align: center; padding: 3rem 0;'>", unsafe_allow_html=True)
        st.markdown("### 💬 Start a conversation")
        st.markdown("Ask me anything about your uploaded documents. I can help analyze, summarize, and answer questions about your content.")
        
        # Quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        quick_actions = [
            ("Summarize my documents", "Summarize my documents"),
            ("What are the key points?", "What are the key points?"),
            ("Analyze the content", "Analyze the content"),
            ("Find specific information", "Find specific information")
        ]
        
        for i, (label, prompt) in enumerate(quick_actions):
            with [col1, col2, col3, col4][i]:
                if st.button(label, key=f"quick_{i}"):
                    # Add the quick action as user message and generate response
                    chat["messages"].append({"role": "user", "content": prompt})
                    response = st.session_state.chatbot.generate_response(prompt)
                    chat["messages"].append({"role": "assistant", "content": response})
                    if chat["title"] == "New Chat":
                        chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

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
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        
        chat["messages"].append({"role": "assistant", "content": response})
        
        if len(chat["messages"]) == 2 and chat["title"] == "New Chat":
            chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
        st.rerun()

    # Mode selector at bottom
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat Mode", use_container_width=True, type="primary" if st.session_state.current_view == "chat" else "secondary"):
            st.session_state.current_view = "chat"
            st.rerun()
    with col2:
        if st.button("🔬 Training Mode", use_container_width=True, type="primary" if st.session_state.current_view == "train" else "secondary"):
            st.session_state.current_view = "train"
            st.rerun()

# Training UI - Simplified
def render_training():
    st.markdown("### 🧠 Model Training")
    
    if not st.session_state.chatbot.documents:
        st.info("📁 Please upload documents first to start training models.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Documents", len(st.session_state.chatbot.documents))
    with col2:
        st.metric("🤖 Datasets", len(getattr(st.session_state.chatbot, 'datasets', {})))

    if hasattr(st.session_state.chatbot, 'datasets') and st.session_state.chatbot.datasets:
        st.markdown("**📊 Available Datasets**")
        for doc_id, dataset in st.session_state.chatbot.datasets.items():
            with st.expander(f"📊 {dataset.get('filename', doc_id[:20])}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Type", dataset.get('type', 'Unknown'))
                with col2:
                    st.metric("Records", len(dataset.get('data', [])))
                with col3:
                    if st.button("🚀 Train", key=f"train_{doc_id}"):
                        with st.spinner("Training model..."):
                            result = st.session_state.chatbot.auto_train_models(doc_id)
                            st.success(f"✅ {result}")

# Entry Point
def main():
    st.set_page_config(
        page_title="AI Document Assistant", 
        layout="wide", 
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stButton > button {
        border-radius: 8px;
    }
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    .stTextInput > div > div {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    render_sidebar()

    if st.session_state.current_view == "chat":
        render_chat()
    elif st.session_state.current_view == "train":
        render_training()

if __name__ == "__main__":
    main()