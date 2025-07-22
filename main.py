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
            logger.debug(f"Processing item: {item_path}")

            if item['mimeType'] == 'application/vnd.google-apps.folder':
                os.makedirs(item_path, exist_ok=True)
                download_folder_contents(drive, item['id'], item_path)
            else:
                logger.debug(f"⬇️  Downloading: {item_path}")
                item.GetContentFile(item_path)
                logger.debug(f"Downloaded file exists: {os.path.exists(item_path)}")
    except Exception as e:
        logger.error(f"❌ Error downloading folder contents: {e}")

def download_models_from_drive():
    logger.debug("🔐 Authenticating with Google Drive...")
    try:
        gauth = GoogleAuth()
        # Load client secrets for OAuth2
        client_secrets_path = os.getenv("GOOGLE_CLIENT_SECRETS")
        if client_secrets_path and os.path.exists(client_secrets_path):
            gauth.LoadClientConfigFile(client_secrets_path)
        else:
            raise ValueError("GOOGLE_CLIENT_SECRETS environment variable not set or file not found.")

        # Authenticate using local web server flow
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        parent_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not parent_folder_id:
            raise ValueError("Environment variable GOOGLE_DRIVE_FOLDER_ID not set.")

        download_folder_contents(drive, parent_folder_id)
        logger.info("✅ All non-empty model folders downloaded.")

        # Create symlink to latest folder
        model_dir = os.path.join("saved_models", "models")
        if os.path.exists(model_dir):
            model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
            if model_folders:
                latest_model = max(model_folders, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
                latest_link = os.path.join("saved_models", "latest")
                if os.path.exists(latest_link):
                    os.unlink(latest_link)
                os.symlink(os.path.join(model_dir, latest_model), latest_link)
                logger.info(f"Created latest symlink to: {latest_model}")
    except Exception as e:
        logger.error("❌ Failed to download model files.")
        logger.error(f"{type(e).__name__}: {e}")

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

    # Find the latest model folder
    model_dir = os.path.join("saved_models", "models")
    latest_model_path = None
    if os.path.exists(model_dir):
        model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
        if model_folders:
            latest_model_path = max(model_folders, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            logger.debug(f"Latest model folder found: {latest_model_path}")
            checkpoint_path = os.path.join(model_dir, latest_model_path, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                logger.info(f"Found checkpoint file: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint file not found in {checkpoint_path}")
        else:
            logger.warning(f"No model folders found in {model_dir}")
    else:
        logger.warning(f"Model directory {model_dir} does not exist.")

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
        # Pass the latest model path to doc_app.main() if available
        if latest_model_path:
            st.session_state['model_path'] = os.path.join(model_dir, latest_model_path)
        doc_app.main()
    else:
        logger.debug("User not authenticated. Showing login.")
        auth.render_auth_page()

if __name__ == "__main__":
    main()