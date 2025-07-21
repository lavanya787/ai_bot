import streamlit as st
import sys
from pathlib import Path
import logging
import auth
import app as doc_app
from utils.nltk_setup import download_nltk_resources

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

def main():
    logger.debug("Starting main.py")

    st.set_page_config(
        page_title="IntelliDoc",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

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
