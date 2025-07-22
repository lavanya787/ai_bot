import streamlit as st
import sys
from pathlib import Path
import logging
import json
import auth
import app as doc_app
from utils.nltk_setup import download_nltk_resources

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
def download_models_from_drive(drive_folder_id, destination_folder):
    print("Authenticating with Google Drive...")

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("service_account.json")
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({
        'q': f"'{drive_folder_id}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        file_path = Path(destination_folder) / file['title']

        if file_path.exists():
            print(f"✅ Already downloaded: {file['title']}")
            continue

        print(f"\n📥 Downloading model: {file['title']}")

        # ✅ Skip Google Docs/Sheets etc.
        if 'application/vnd.google-apps' in file['mimeType']:
            print(f"⚠️ Skipping unsupported Google-native file: {file['title']}")
            continue

        try:
            file.GetContentFile(str(file_path))
            print(f"✅ Downloaded: {file['title']}")
        except Exception as e:
            print(f"❌ Failed to download {file['title']}\n{e}")


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
