import streamlit as st
import sys
from pathlib import Path
import logging
import json
import auth
import os
import app as doc_app
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv  # ✅ Add this
from utils.nltk_setup import download_nltk_resources

load_dotenv()  # ✅ Load environment variables

# Optional: for Google Drive
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
download_nltk_resources()

# Ensure current directory is in sys.path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# ✅ Google Drive Authentication & Model Download

def download_models_from_drive():
    print("🔐 Authenticating with Google Drive...")

    try:
        gauth = GoogleAuth()
        gauth.ServiceAuth()  # ✅ Use service account auth (requires settings.yaml + service_account.json)

        drive = GoogleDrive(gauth)

        # Folder ID from your environment
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not folder_id:
            raise ValueError("Environment variable GOOGLE_DRIVE_FOLDER_ID not set.")

        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        if not file_list:
            print("⚠️ No files found in the specified Google Drive folder.")
            return

        for file in file_list:
            file_path = os.path.join("saved_models", file['title'])
            print(f"⬇️  Downloading {file['title']}...")
            file.GetContentFile(file_path)

        print("✅ All model files downloaded successfully.")

    except Exception as e:
        print("❌ Failed to download model files from Google Drive.\n")
        print(f"{type(e).__name__}: {e}")



def main():
    logger.debug("Starting main.py")

    st.set_page_config(
        page_title="IntelliDoc",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ✅ Download models at startup
    download_models_from_drive()

    # Session and URL-based authentication
    user_id_from_url = st.query_params.get("user_id")
    is_authenticated = st.session_state.get('authenticated', False)

    if user_id_from_url:
        if not st.session_state.get('user_id'):
            st.session_state['user_id'] = user_id_from_url
        if not is_authenticated:
            st.session_state['authenticated'] = True
            st.experimental_rerun()

    if st.session_state.get('authenticated', False):
        logger.debug("User is authenticated. Running main app.")
        doc_app.main()
    else:
        logger.debug("User not authenticated. Showing login.")
        auth.render_auth_page()

if __name__ == "__main__":
    main()
