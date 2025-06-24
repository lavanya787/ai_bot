import json
import os
import pandas as pd
import PyPDF2
import logging
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_bot")
CHAT_HISTORY_FILE = "logs/chat_history.json"

def save_chat_history(chat_history):
    os.makedirs("logs", exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2)
def save_chat_state():
    try:
        success = save_chat_history(st.session_state.chat_history)
        if not success:
            st.error("Failed to save chat history")
    except Exception as e:
        st.error(f"Error saving chat: {e}")
        logger.error(f"Error saving chat: {e}")
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
def fallback_load_chat_history():
    try:
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
    return []

def fallback_save_chat_history(chat_history):
    try:
        with open("chat_history.json", "w") as f:
            json.dump(chat_history, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        return False

def fallback_extract_text(file):
    try:
        if file.type == "text/plain":
            content = str(file.read(), "utf-8")
            return {"type": "text", "content": content}
        elif file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            return {"type": "text", "content": content}
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return {"type": "structured", "data": df, "content": df.to_string()}
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
            return {"type": "structured", "data": df, "content": df.to_string()}
        else:
            return {"error": f"Unsupported file type: {file.type}"}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}
