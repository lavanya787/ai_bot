import os
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
from utils.preprocessing import Preprocessor

# Initialize NLTK downloads once (you can move this to main.py if you want)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

preprocessor = Preprocessor()

def file_upload_interface():
    st.title("File Upload Interface")

    query_params = st.experimental_get_query_params()
    user_id = query_params.get("user_id", [""])[0]

    if not user_id or not preprocessor.is_valid_uuid(user_id):
        st.error("Invalid or missing User ID.")
        st.markdown("[Go to Login/Register](http://localhost:5000)", unsafe_allow_html=True)
        st.stop()

    uploaded_file = st.file_uploader("Upload your file (CSV, Excel, PDF, Text)", type=["csv", "xlsx", "xls", "pdf", "txt"])

    if uploaded_file:
        file_id = str(uuid.uuid4())
        file_type = uploaded_file.type
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use Preprocessor methods
        if file_type == "application/pdf":
            preprocessed_data = preprocessor.preprocess_pdf(uploaded_file)
        elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            preprocessed_data = preprocessor.preprocess_excel(uploaded_file)
        elif file_type == "text/csv":
            preprocessed_data = preprocessor.preprocess_csv(uploaded_file)
        elif file_type == "text/plain":
            preprocessed_data = preprocessor.preprocess_text(uploaded_file)
        else:
            st.error("Unsupported file type!")
            st.stop()

        os.makedirs("preprocessed", exist_ok=True)
        cleaned_filename = uploaded_file.name.rsplit('.', 1)[0] + "_preprocessed.csv"
        storage_path = f"preprocessed/{cleaned_filename}"
        df_clean = pd.DataFrame({"cleaned_text": [preprocessed_data]})
        df_clean.to_csv(storage_path, index=False)


    user_message = st.text_input("Your Message:")
    if user_message:
        st.write("📩 Message received.")
