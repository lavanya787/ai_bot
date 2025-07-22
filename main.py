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
def download_models_from_drive():
    try:
        st.info("Authenticating with Google Drive...")

        # Load service account credentials from Streamlit secrets
        service_account_info = json.loads(st.secrets["GOOGLE_DRIVE_SERVICE_ACCOUNT"])
        credentials_file_path = current_dir / "service_account.json"
        with open(credentials_file_path, "w") as f:
            json.dump(service_account_info, f)

        # ✅ Set up PyDrive2 to use the service account
        gauth = GoogleAuth()
        gauth.settings['get_refresh_token'] = True
        gauth.LoadCredentialsFile(str(credentials_file_path))
        gauth.ServiceAuth()

        drive = GoogleDrive(gauth)
        folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        for file in file_list:
            file_path = current_dir / file['title']
            if not file_path.exists():
                st.write(f"📥 Downloading model: {file['title']}")
                file.GetContentFile(str(file_path))
            else:
                st.write(f"✅ Already downloaded: {file['title']}")

    except Exception as e:
        logger.error("Error during Google Drive authentication or download")
        st.error("❌ Failed to download model files from Google Drive.")
        st.exception(e)

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
