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
# Create local model folder if not exists
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)
# Ensure current directory is in sys.path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import pickle

def load_models(models_dir=MODELS_DIR):
    loaded_models = {}

    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.suffix in ['.pkl', '.model']:
                    try:
                        with open(file, 'rb') as f:
                            model = pickle.load(f)
                        model_key = f"{subdir.name}/{file.name}"
                        loaded_models[model_key] = model
                        print(f"✅ Loaded model: {model_key}")
                    except Exception as e:
                        print(f"❌ Failed to load model {file.name}: {e}")

    return loaded_models

# ✅ Recursive download function
def download_folder_contents(drive, folder_id, parent_path="saved_models"):
    try:
        # Query contents of the current folder
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        for item in file_list:
            item_path = os.path.join(parent_path, item['title'])

            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # Create local subfolder
                os.makedirs(item_path, exist_ok=True)
                # Recurse into subfolder
                download_folder_contents(drive, item['id'], item_path)
            else:
                # Download file
                print(f"⬇️  Downloading: {item_path}")
                item.GetContentFile(item_path)

    except Exception as e:
        print(f"❌ Error downloading folder contents: {e}")


# ✅ Authentication and model download
def download_models_from_drive():
    print("🔐 Authenticating with Google Drive...")

    try:
        gauth = GoogleAuth()
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)

        parent_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not parent_id:
            raise ValueError("Environment variable GOOGLE_DRIVE_FOLDER_ID not set.")

        # Get all subfolders inside 'models'
        folder_list = drive.ListFile({
            'q': f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }).GetList()

        if not folder_list:
            print("⚠️ No subfolders found inside the models folder.")
            return

        for subfolder in folder_list:
            subfolder_id = subfolder['id']
            subfolder_name = subfolder['title']
            local_subfolder_path = MODELS_DIR / subfolder_name
            local_subfolder_path.mkdir(parents=True, exist_ok=True)

            print(f"\n📁 Processing folder: {subfolder_name}")

            files = drive.ListFile({
                'q': f"'{subfolder_id}' in parents and trashed=false"
            }).GetList()

            if not files:
                print(f"⚠️ Skipping empty subfolder: {subfolder_name}")
                continue

            for file in files:
                file_name = file['title']
                destination_path = local_subfolder_path / file_name
                print(f"⬇️  Downloading: {file_name}")
                file.GetContentFile(str(destination_path))

        print("✅ All model files downloaded successfully.")

    except Exception as e:
        print("❌ Failed to download model files from Google Drive.")
        print(f"{type(e).__name__}: {e}")




# ✅ Main Streamlit app entry point
def main():
    logger.debug("Starting main.py")

    st.set_page_config(
        page_title="IntelliDoc",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Step 1: Download models
    download_models_from_drive()

    # Step 2: Load models into session_state
    if 'models' not in st.session_state:
        st.session_state.models = load_models()

    # Step 3: Handle authentication
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
