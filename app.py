import streamlit as st
import os
import tempfile
from datetime import datetime
import logging
from io import BytesIO

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
from utils.pdf_export import export_chat_to_pdf  # ✅
from utils.voice_input import record_and_transcribe  # ✅ Add this at the top

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

log = Logger().logger
preprocessor = Preprocessor()

# Session state
def initialize_session_state():
    defaults = {
        'chat_history': [{"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}],
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

# Welcome UI
def show_welcome():
    st.title("🤖 AI Document Assistant")
    st.write("Upload files and start chatting with your documents!")
    col1, col2, col3 = st.columns(3)
    with col1: st.info("📄 Upload documents for analysis")
    with col2: st.info("💬 Ask questions about your files")
    with col3: st.info("🧠 Get AI-powered insights")
    if st.button("Start New Conversation", type="primary"):
        st.session_state.show_welcome = False
        st.rerun()

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.header("🤖 AI Assistant")
        if st.button("➕ New Chat", use_container_width=True):
            new_chat = {"title": "New Chat", "messages": [], "created_at": datetime.now().isoformat()}
            st.session_state.chat_history.append(new_chat)
            st.session_state.selected_chat_index = len(st.session_state.chat_history) - 1
            st.session_state.show_welcome = False
            st.rerun()

        st.divider()
        st.subheader("💬 Chats")
        for i, chat in enumerate(st.session_state.chat_history):
            is_selected = (i == st.session_state.selected_chat_index)
            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(chat["title"][:25], key=f"chat_{i}", use_container_width=True):
                    st.session_state.selected_chat_index = i
                    st.session_state.show_welcome = False
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"rename_{i}"):
                    new_title = st.text_input("Rename chat", chat["title"], key=f"title_{i}")
                    if new_title:
                        st.session_state.chat_history[i]["title"] = new_title
                        st.rerun()

        st.divider()
        st.subheader("📤 Upload Files")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "docx", "csv", "xlsx"], accept_multiple_files=True)

        if uploaded_files and st.button("Process Files", type="primary"):
            with st.spinner("Processing files..."):
                progress = st.progress(0)
                for i, file in enumerate(uploaded_files):
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
                    progress.progress((i + 1) / len(uploaded_files))

            st.success("✅ Files processed successfully!")
            st.rerun()

        if st.session_state.processed_files:
            st.subheader("📁 Recently Processed")
            for f in st.session_state.processed_files[-3:]:
                st.markdown(f"✅ `{f['filename']}` | {f.get('size', 0):,} chars | {f.get('type', 'Unknown')}")

# Chat area
def render_chat():
    if st.session_state.show_welcome:
        show_welcome()
        return

    chat_idx = st.session_state.selected_chat_index
    chat = st.session_state.chat_history[chat_idx]

    st.header(f"💬 {chat['title']}")
    if chat["messages"] and st.button("🧹 Clear Chat"):
        chat["messages"] = []
        st.rerun()

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.chat_input("Ask me anything...")
    with col2:
        audio_bytes = st.file_uploader("🎙️ Voice Input", type=["webm"], label_visibility="collapsed")
        if audio_bytes is not None:
            with st.spinner("Transcribing..."):
                prompt = record_and_transcribe(audio_bytes)
                st.success(f"📣 You said: {prompt}")
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        chat["messages"].append({"role": "assistant", "content": response})

        if len(chat["messages"]) == 2 and chat["title"] == "New Chat":
            chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
        st.rerun()

    # Export
    if chat["messages"]:
        with st.expander("📥 Download Chat History"):
            md = f"# Chat: {chat['title']}\n\n"
            for msg in chat["messages"]:
                md += f"{'**User**' if msg['role']=='user' else '**Assistant**'}: {msg['content']}\n\n"
            st.download_button("⬇️ Markdown", md, file_name=f"{chat['title']}.md", mime="text/markdown")

            if st.button("⬇️ Generate PDF"):
                pdf_bytes = export_chat_to_pdf(chat["title"], chat["messages"])
                st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"{chat['title']}.pdf", mime="application/pdf")

# Training UI
def render_training():
    st.header("🧠 Model Training")
    if not st.session_state.chatbot.documents:
        st.warning("Please upload documents first.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Documents", len(st.session_state.chatbot.documents))
    col2.metric("Datasets", len(getattr(st.session_state.chatbot, 'datasets', {})))

    if hasattr(st.session_state.chatbot, 'datasets'):
        st.subheader("📊 Available Datasets")
        for doc_id, dataset in st.session_state.chatbot.datasets.items():
            with st.expander(f"Dataset: {dataset.get('filename', doc_id[:10])}"):
                st.write(f"- Type: {dataset.get('type', 'Unknown')}")
                st.write(f"- Records: {len(dataset.get('data', []))}")
                if st.button("Train", key=f"train_{doc_id}"):
                    with st.spinner("Training..."):
                        res = st.session_state.chatbot.auto_train_models(doc_id)
                        st.success(res)

    if hasattr(st.session_state.chatbot, 'trained_models'):
        st.subheader("📌 Trained Models")
        for doc_id, model in st.session_state.chatbot.trained_models.items():
            with st.expander(f"Model: {doc_id[:10]}"):
                st.metric("Type", model.get("model_type", "N/A"))
                st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
                st.caption(f"Trained: {model.get('trained_at', '')[:10]}")

# Entry Point
def main():
    st.set_page_config(page_title="AI Bot Assistant", layout="wide", page_icon="🤖")
    initialize_session_state()
    render_sidebar()

    if st.session_state.current_view == "chat":
        render_chat()
    elif st.session_state.current_view == "train":
        render_training()

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("💬 Chat Mode", use_container_width=True):
        st.session_state.current_view = "chat"
        st.rerun()
    if col2.button("📊 Train Mode", use_container_width=True):
        st.session_state.current_view = "train"
        st.rerun()

if __name__ == "__main__":
    main()
