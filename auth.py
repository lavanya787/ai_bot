import streamlit as st
import hashlib
import re
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Supabase setup
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

SUPABASE_URL = "https://swrhcsfuorjqszqoohfb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3cmhjc2Z1b3JqcXN6cW9vaGZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MDkxODA3MCwiZXhwIjoyMDY2NDk0MDcwfQ.aYmADn3cpUhPlNSUiKlb-EveEyyE7-8FgYVq7L4A2OA"

supabase = None
if SUPABASE_AVAILABLE:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}", exc_info=True)
        st.error(f"Failed to connect to database: {e}")

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    return True, "Valid password"

def register_user(email, password, full_name):
    """Register new user in Supabase"""
    logger.debug(f"Registering user: {email}")
    if not supabase:
        logger.error("Supabase connection not available")
        st.error("Database connection not available")
        return False, "Database error"
    
    try:
        # Check if user already exists
        existing = supabase.table("users").select("*").eq("email", email).execute()
        if existing.data:
            return False, "User with this email already exists"
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_pw = hash_password(password)
        
        user_data = {
            "user_id": user_id,
            "email": email,
            "password": hashed_pw,
            "full_name": full_name,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        result = supabase.table("users").insert(user_data).execute()
        if result.data:
            logger.info(f"User registered successfully: {user_id}")
            return True, user_id
        else:
            logger.error("Registration failed")
            return False, "Registration failed"
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return False, f"Error: {str(e)}"

def login_user(email, password):
    """Authenticate user login"""
    logger.debug(f"Logging in user: {email}")
    if not supabase:
        logger.error("Supabase connection not available")
        st.error("Database connection not available")
        return False, None, "Database error"
    
    try:
        hashed_pw = hash_password(password)
        result = supabase.table("users").select("*").eq("email", email).eq("password", hashed_pw).execute()
        
        if result.data:
            user = result.data[0]
            if user.get("is_active", True):
                logger.info(f"User logged in successfully: {user['user_id']}")
                return True, user["user_id"], user["full_name"]
            else:
                logger.warning("Account is deactivated")
                return False, None, "Account is deactivated"
        else:
            logger.warning("Invalid email or password")
            return False, None, "Invalid email or password"
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return False, None, f"Login error: {str(e)}"

def render_auth_page():
    """Render the authentication page"""
    logger.debug("Rendering auth page")
    st.set_page_config(
        page_title="IntelliDoc - Login", 
        page_icon="📚", 
        layout="wide"
    )
    
    # Custom CSS for exact UI match
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    .auth-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0 auto;
        max-width: 400px;
    }
    .feature-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.9) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #333 !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4285f4, #34a853) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: bold !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    .secondary-btn button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: white !important;
    }
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">🤖</div>
                <div>
                    <h1 style="color: white; font-size: 2.5rem; margin: 0;">IntelliDoc</h1>
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 1.1rem;">Intelligent document analysis powered by AI</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 1])
    
    # Auth form
    with col1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        
        if st.session_state.auth_mode == 'login':
            st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 0.5rem;">Welcome Back! 👋</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 1.5rem;">Sign in to access your AI-powered workspace</p>', unsafe_allow_html=True)
            
            with st.form("login_form"):
                email = st.text_input("", placeholder="Enter your email", label_visibility="collapsed")
                password = st.text_input("", placeholder="Enter your password", type="password", label_visibility="collapsed")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_btn = st.form_submit_button("🚀 Sign In")
                with col_b:
                    if st.form_submit_button("📝 Need Account?"):
                        st.session_state.auth_mode = 'register'
                        st.rerun()
                
                if login_btn and email and password:
                    if validate_email(email):
                        success, user_id, full_name = login_user(email, password)
                        if success:
                            st.session_state.update({'authenticated': True, 'user_id': user_id, 'user_email': email, 'user_name': full_name})
                            st.success(f"✅ Welcome back, {full_name}!")
                            #st.balloons()
                        else:
                            st.error(f"❌ {full_name}")
                    else:
                        st.error("⚠️ Invalid email format")
        
        else:  # Register mode
            st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 0.5rem;">Join Our Community! 🎉</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 1.5rem;">Create your account and start exploring AI-powered document analysis</p>', unsafe_allow_html=True)
            
            with st.form("register_form"):
                full_name = st.text_input("", placeholder="Enter your full name", label_visibility="collapsed")
                email = st.text_input("", placeholder="Enter your email", label_visibility="collapsed")
                password = st.text_input("", placeholder="Create a secure password", type="password", label_visibility="collapsed")
                confirm_password = st.text_input("", placeholder="Confirm your password", type="password", label_visibility="collapsed")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    register_btn = st.form_submit_button("✨ Create Account")
                with col_b:
                    if st.form_submit_button("🔙 Back to Login"):
                        st.session_state.auth_mode = 'login'
                        st.rerun()
                
                if register_btn and all([full_name, email, password, confirm_password]):
                    if not validate_email(email):
                        st.error("⚠️ Invalid email format")
                    elif password != confirm_password:
                        st.error("⚠️ Passwords do not match")
                    elif not validate_password(password)[0]:
                        st.error(f"⚠️ {validate_password(password)[1]}")
                    else:
                        success, result = register_user(email, password, full_name)
                        if success:
                            st.success("✅ Account created successfully!")
                            st.info("🎯 Please login with your new credentials")
                            st.session_state.auth_mode = 'login'
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    with col2:
        st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 1rem;">✨ Why Choose Our AI Assistant?</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 2rem;">Experience the future of document processing with our cutting-edge AI technology</p>', unsafe_allow_html=True)
        
        features = [
            ("⚡", "Lightning Fast", "Upload and analyze documents in seconds with our optimized AI processing"),
            ("🧠", "Smart AI", "Advanced machine learning algorithms provide accurate insights and analysis"),
            ("🔒", "Secure & Private", "Your documents are processed safely with enterprise-grade security"),
            ("📄", "Trusted Platform", "Join thousands of users who trust our platform for their document analysis needs")
        ]
        
        for icon, title, desc in features:
            st.markdown(f'''
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 1rem;">{icon}</div>
                <div style="color: white; font-weight: bold; margin-bottom: 0.5rem;">{title}</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem; line-height: 1.4;">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)
            
def render_login_form():
    """Render login form"""
    logger.debug("Rendering login form")
    st.markdown('<div class="form-title">👋 Welcome Back!</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Sign in to access your AI-powered workspace</div>', unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="Enter your email", key="login_email", label_visibility="collapsed")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_password", label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.form_submit_button("🚀 Sign In", type="primary")
        with col2:
            if st.form_submit_button("📝 Need Account?"):
                st.session_state.auth_mode = 'register'
                st.rerun()
        
        if login_btn:
            if not email or not password:
                st.error("⚠️ Please fill in all fields")
                logger.warning("Login attempted with empty fields")
                return
            
            if not validate_email(email):
                st.error("⚠️ Please enter a valid email address")
                logger.warning("Invalid email format")
                return
            
            with st.spinner("🔄 Signing you in..."):
                success, user_id, full_name = login_user(email, password)
                
                if success:
                    st.session_state.update({
                        'authenticated': True,
                        'user_id': user_id,
                        'user_email': email,
                        'user_name': full_name
                    })
                    logger.info(f"User {full_name} logged in successfully")
                    st.success(f"✅ Welcome back, {full_name}!")
                    #st.balloons()
                    
                    # Redirect to main app
                    st.markdown(f"""
                        <meta http-equiv="refresh" content="1;url=?user_id={user_id}">
                        <script>window.location.href = '?user_id={user_id}';</script>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Login failed: {full_name}")
                    logger.error(f"Login failed for {email}: {full_name}")

def render_register_form():
    """Render registration form"""
    logger.debug("Rendering register form")
    st.markdown('<div class="form-title">🎉 Join Our Community!</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-subtitle">Create your account and start exploring AI-powered document analysis</div>', unsafe_allow_html=True)
    
    with st.form("register_form", clear_on_submit=False):
        full_name = st.text_input("Full Name", placeholder="Enter your full name", key="reg_name", label_visibility="collapsed")
        email = st.text_input("Email", placeholder="Enter your email", key="reg_email", label_visibility="collapsed")
        password = st.text_input("Password", placeholder="Create a secure password", type="password", key="reg_password", label_visibility="collapsed")
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", key="reg_confirm", label_visibility="collapsed")

        col1, col2 = st.columns(2)
        with col1:
            register_btn = st.form_submit_button("✨ Create Account", type="primary")
        with col2:
            if st.form_submit_button("🔙 Back to Login"):
                st.session_state.auth_mode = 'login'
                st.rerun()
        
        if register_btn:
            if not all([full_name, email, password, confirm_password]):
                st.error("⚠️ Please fill in all fields")
                logger.warning("Registration attempted with empty fields")
                return
            
            if not validate_email(email):
                st.error("⚠️ Please enter a valid email address")
                logger.warning("Invalid email format")
                return
            
            if password != confirm_password:
                st.error("⚠️ Passwords do not match")
                logger.warning("Passwords do not match")
                return
            
            is_valid, msg = validate_password(password)
            if not is_valid:
                st.error(f"⚠️ {msg}")
                logger.warning(f"Invalid password: {msg}")
                return
            
            with st.spinner("🔄 Creating your account..."):
                success, result = register_user(email, password, full_name)
                
                if success:
                    st.success("✅ Account created successfully!")
                    logger.info("Account created successfully")
                    st.info("🎯 Please login with your new credentials")
                    #st.balloons()
                    st.session_state.auth_mode = 'login'
                    st.rerun()
                else:
                    st.error(f"❌ Registration failed: {result}")
                    logger.error(f"Registration failed: {result}")

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def logout_user():
    """Logout current user"""
    logger.debug("Logging out user")
    keys_to_clear = ['authenticated', 'user_id', 'user_email', 'full_name']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def main():
    """Main authentication app"""
    # Check if user is already authenticated or has user_id in URL
    user_id_from_url = st.query_params.get("user_id")
    logger.debug("Starting auth.py main")

    if check_authentication() or user_id_from_url:
        # Import and run main app
        if user_id_from_url and not st.session_state.get('user_id'):
            st.session_state.user_id = user_id_from_url
        
        try:
            # Import your main app here
            import app
            app.main()
        except ImportError:
            st.error("❌ Main application not found. Please ensure app.py is in the same directory.")
        if st.button("Logout"):
            for key in ['authenticated', 'user_id', 'user_email', 'user_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        # Show authentication page
        render_auth_page()

if __name__ == "__main__":
    main()