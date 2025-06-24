# model_trainer_patch.py

import os
import pickle
import hashlib
import streamlit as st
from datetime import datetime
from utils.domain_detector import detect_domain
from models.education_models import EducationModel

def train_and_store_model(chatbot, doc_id: str):
    """
    Trains a model for the given document ID and stores the model and uploaded file
    in domain-specific subdirectories.
    """
    if doc_id not in chatbot.datasets:
        st.error(f"❌ Dataset with ID {doc_id} not found.")
        return

    dataset = chatbot.datasets[doc_id]
    data = dataset['data']
    filename = dataset['filename']
    raw_file = dataset.get('raw_file')

    # Detect domain using column names and first row values
    sample_text = " ".join(list(data.columns))
    if not data.empty:
        sample_text += " " + " ".join(map(str, data.iloc[0].values))

    domain = detect_domain(sample_text) or "general"
    dataset['domain'] = domain

    # Create folders
    model_dir = os.path.join("models", domain, "trained_models")
    file_dir = os.path.join("files", domain, "uploaded_files")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(file_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{filename}_model.pkl")
    file_path = os.path.join(file_dir, filename)

    # Save uploaded file
    if raw_file:
        with open(file_path, "wb") as f:
            f.write(raw_file.getbuffer())

    # Avoid retraining if already exists
    retrain_key = f"retrain_{doc_id}"
    if os.path.exists(model_path):
        if retrain_key not in st.session_state:
            st.session_state[retrain_key] = False

        if not st.session_state[retrain_key]:
            if st.checkbox(f"⚠️ Model already exists for '{filename}' in {domain.upper()}. Retrain?"):
                st.session_state[retrain_key] = True
            else:
                st.warning("⚠️ Training skipped.")
                return

    # Train model
    target_column = next((col for col in data.columns if data[col].dtype == 'object'), None)
    if not target_column:
        st.error("❌ No suitable target column found.")
        return

    try:
        model = EducationModel(len(data), embedding_dim=128, lstm_units=64)
        trained_model = model.train(data, target_column,
                                    max_words=5000,
                                    max_len=250,
                                    epochs=5,
                                    batch_size=32,
                                    learning_rate=0.001)

        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)

        st.success(f"✅ Model trained and saved: `{model_path}`")

        chatbot.models[f"{doc_id}_education"] = {
            "model": trained_model,
            "metadata": {"target": target_column, "domain": domain},
            "trained_at": datetime.now().isoformat(),
            "model_type": "education"
        }

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
