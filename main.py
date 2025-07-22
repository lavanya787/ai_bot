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
from dotenv import load_dotenv
from utils.nltk_setup import download_nltk_resources

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download NLTK data
download_nltk_resources()

# Ensure current directory is in sys.path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


# ✅ Recursive download function
def download_folder_contents(drive, folder_id, parent_path="saved_models/models"):
    try:
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        for item in file_list:
            item_path = os.path.join(parent_path, item['title'])

            if item['mimeType'] == 'application/vnd.google-apps.folder':
                os.makedirs(item_path, exist_ok=True)
                download_folder_contents(drive, item['id'], item_path)
            else:
                print(f"⬇️  Downloading: {item_path}")
                item.GetContentFile(item_path)
    except Exception as e:
        print(f"❌ Error downloading folder contents: {e}")

def download_models_from_drive():
    print("🔐 Authenticating with Google Drive...")
    try:
        gauth = GoogleAuth()
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)

        parent_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not parent_folder_id:
            raise ValueError("Environment variable GOOGLE_DRIVE_FOLDER_ID not set.")

        download_folder_contents(drive, parent_folder_id)
        print("✅ All non-empty model folders downloaded.")
    except Exception as e:
        print("❌ Failed to download model files.")
        print(f"{type(e).__name__}: {e}")


# ✅ Main Streamlit app entry point
def main():
    logger.debug("Starting main.py")

    st.set_page_config(
        page_title="IntelliDoc",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Download models from Google Drive recursively
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
