import streamlit as st
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for the application"""
    logger.debug("Starting main.py")

    # Configure the page
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Check authentication status
    user_id_from_url = st.query_params.get("user_id")
    is_authenticated = st.session_state.get('authenticated', False)
    
    # If user is authenticated or has user_id in URL, show main app
    if is_authenticated or user_id_from_url:
        logger.debug(f"User authenticated: {is_authenticated}, user_id: {user_id_from_url}")
        # Set user_id in session if from URL
        if user_id_from_url and not st.session_state.get('user_id'):
            st.session_state['user_id'] = user_id_from_url
            st.session_state['authenticated'] = True
            
        # Import and run the main document assistant app
        try:
            import app
            app.main()
        except ImportError as e:
            logger.error(f"Failed to import app: {e}", exc_info=True)
            st.error(f"❌ Main application not found: {e}")
            st.info("Please ensure app.py is in the same directory.")
    else:
        logger.debug("User not authenticated, rendering auth page")
        # Show authentication page
        try:
            import auth
            auth.render_auth_page()
        except ImportError as e:
            logger.error(f"Failed to import auth: {e}", exc_info=True)
            st.error(f"❌ Authentication module not found: {e}")
            st.info("Please ensure auth.py is in the same directory.")

if __name__ == "__main__":
    main()