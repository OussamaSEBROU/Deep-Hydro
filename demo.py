import streamlit as st

# --- Page Configuration & Initialization ---
# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from fpdf import FPDF
import google.generativeai as genai
import io
import base64
import time
import os
import json
import datetime
import uuid
import firebase_admin
from firebase_admin import credentials, db
import requests
import plotly.express as px
from dotenv import load_dotenv
import hashlib
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component

# --- Load Configuration from Single Environment Variable --- 
APP_CONFIG = {}
config_load_error = None # Store potential errors here

def load_app_config():
    """Load configuration from APP_CONFIG_JSON environment variable."""
    global APP_CONFIG, config_load_error
    load_dotenv() # Load .env for local development first
    config_json_str = os.getenv("APP_CONFIG_JSON")
    if not config_json_str:
        config_load_error = "Critical: APP_CONFIG_JSON environment variable not found."
        return False
    try:
        APP_CONFIG = json.loads(config_json_str)
        # Basic validation
        if not isinstance(APP_CONFIG.get("firebase_service_account"), dict) or \
           not isinstance(APP_CONFIG.get("google_oauth"), dict) or \
           not APP_CONFIG.get("google_api_key") or \
           not APP_CONFIG["google_oauth"].get("client_id") or \
           not APP_CONFIG["google_oauth"].get("client_secret") or \
           not APP_CONFIG["google_oauth"].get("redirect_uri"):
            config_load_error = "APP_CONFIG_JSON is missing required keys (firebase_service_account, google_oauth, google_api_key, etc.). Check the structure."
            return False
        # print("DEBUG: APP_CONFIG loaded successfully.") # Debug
        return True
    except json.JSONDecodeError as e:
        config_load_error = f"Error decoding APP_CONFIG_JSON: {e}. Check if it is valid JSON."
        return False
    except Exception as e:
        config_load_error = f"Unexpected error loading APP_CONFIG_JSON: {e}"
        return False

config_loaded = load_app_config()

# --- Initialize Services (Firebase, Gemini, OAuth, Admin Pass) --- 
firebase_initialized = False
gemini_configured = False
google_oauth_configured = False
admin_password = "admin123" # Default fallback
service_init_errors = [] # Store initialization errors

# --- Google OAuth Global Variables (needed by streamlit-oauth) ---
CLIENT_ID = None
CLIENT_SECRET = None
REDIRECT_URI = None
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

def initialize_services():
    """Initialize Firebase, Gemini, OAuth, Admin Pass from loaded config."""
    global firebase_initialized, gemini_configured, google_oauth_configured, admin_password, service_init_errors
    global CLIENT_ID, CLIENT_SECRET, REDIRECT_URI # Make sure globals are modified

    if not config_loaded: # Don't proceed if config didn't load
        return

    # --- Firebase --- 
    if not firebase_admin._apps:
        try:
            firebase_creds_dict = APP_CONFIG.get("firebase_service_account")
            if not firebase_creds_dict or not isinstance(firebase_creds_dict, dict):
                service_init_errors.append("Firebase service account details not found or invalid in config.")
            else:
                cred = credentials.Certificate(firebase_creds_dict)
                project_id = firebase_creds_dict.get("project_id")
                if not project_id:
                    service_init_errors.append("Firebase Service Account JSON in config is missing 'project_id'.")
                else:
                    firebase_url = APP_CONFIG.get("firebase_database_url", f"https://{project_id}-default-rtdb.firebaseio.com/")
                    firebase_admin.initialize_app(cred, {"databaseURL": firebase_url})
                    firebase_initialized = True
        except Exception as e:
            service_init_errors.append(f"Firebase initialization error: {e}")
            firebase_initialized = False
    else:
        firebase_initialized = True # Already initialized

    # --- Gemini --- 
    GEMINI_API_KEY = APP_CONFIG.get("google_api_key")
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Keep model setup simple here, move config details later if needed
            gemini_configured = True
        except Exception as e:
            service_init_errors.append(f"Error configuring Gemini API: {e}")
            gemini_configured = False
    else:
        service_init_errors.append("Gemini API Key not found or is placeholder in config.")
        gemini_configured = False

    # --- Google OAuth --- 
    oauth_config = APP_CONFIG.get("google_oauth")
    if oauth_config and isinstance(oauth_config, dict):
        CLIENT_ID = oauth_config.get("client_id")
        CLIENT_SECRET = oauth_config.get("client_secret")
        REDIRECT_URI = oauth_config.get("redirect_uri")
        if all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
            google_oauth_configured = True
        else:
            service_init_errors.append("Google OAuth config missing client_id, client_secret, or redirect_uri.")
            google_oauth_configured = False
    else:
        service_init_errors.append("Google OAuth config section ('google_oauth') missing or invalid.")
        google_oauth_configured = False

    # --- Admin Password --- 
    admin_password = APP_CONFIG.get("admin_password", "admin123")

# --- Main App Logic --- 
if not config_loaded:
    st.error(config_load_error) # Display config load error AFTER set_page_config
else:
    initialize_services() # Initialize services AFTER set_page_config

    # Display service init errors if any
    if service_init_errors:
        for error_msg in service_init_errors:
            st.warning(error_msg) # Use warning for non-critical init issues

    # --- Constants ---
    ADVANCED_FEATURE_LIMIT = 3

    # --- Gemini Model Setup (moved here) --- 
    gemini_model_report = None
    gemini_model_chat = None
    if gemini_configured:
        try:
            generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
            gemini_model_report = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
            gemini_model_chat = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
        except Exception as e:
            st.warning(f"Failed to initialize Gemini models: {e}")
            gemini_configured = False # Mark as not configured if models fail

    # --- Custom CSS (No changes needed here) ---
    def apply_custom_css():
        st.markdown("""
        <style>
        .stApp { background-color: #f0f2f6; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #FFFFFF; border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px 15px; transition: background-color 0.3s ease; }
        .stTabs [aria-selected="true"] { background-color: #e6f7ff; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #f0f0f0; }
        [data-testid="stSidebar"] { background-color: #ffffff; padding: 1rem; }
        [data-testid="stSidebar"] h2 { color: #1890ff; }
        [data-testid="stSidebar"] .stButton>button { width: 100%; border-radius: 5px; background-color: #1890ff; color: white; transition: background-color 0.3s ease; }
        [data-testid="stSidebar"] .stButton>button:hover { background-color: #40a9ff; }
        [data-testid="stSidebar"] .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #52c41a; color: white; transition: background-color 0.3s ease; }
        [data-testid="stSidebar"] .stDownloadButton>button:hover { background-color: #73d13d; }
        .chat-message { padding: 0.8rem 1rem; margin-bottom: 0.8rem; border-radius: 8px; position: relative; word-wrap: break-word; }
        .user-message { background-color: #e6f7ff; border-left: 4px solid #1890ff; text-align: left; }
        .ai-message { background-color: #f0f0f0; border-left: 4px solid #8c8c8c; text-align: left; }
        .copy-tooltip { position: absolute; top: 0.5rem; right: 0.5rem; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; display: none; background-color: #555; color: white; }
        .chat-message:active .copy-tooltip { display: block; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; border-radius: 4px 4px 0 0; }
        [data-testid="stMetricValue"] { font-weight: 600; }
        .about-us-header { cursor: pointer; padding: 0.5rem; border-radius: 4px; margin-top: 1rem; font-weight: 500; }
        .about-us-content { padding: 0.8rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; }
        .app-intro { padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #1E88E5; }
        </style>
        """, unsafe_allow_html=True)

    # --- JavaScript (No changes needed here) ---
    def add_javascript_functionality():
        st.markdown("""
        <script>
        function copyToClipboard(text) { const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta); }
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                const chatMessages = document.querySelectorAll('.chat-message');
                chatMessages.forEach(function(msg) {
                    if (!msg.querySelector('.copy-tooltip')) { const tt = document.createElement('span'); tt.className = 'copy-tooltip'; tt.textContent = 'Copied!'; msg.appendChild(tt); }
                    let timer;
                    msg.addEventListener('touchstart', function(e) { timer = setTimeout(() => { const txt = this.innerText.replace('Copied!', '').trim(); copyToClipboard(txt); const tt = this.querySelector('.copy-tooltip'); if (tt) { tt.style.display = 'block'; setTimeout(() => { tt.style.display = 'none'; }, 1500); } }, 500); });
                    msg.addEventListener('touchend', function() { clearTimeout(timer); });
                    msg.addEventListener('touchmove', function() { clearTimeout(timer); });
                    msg.addEventListener('click', function(e) { const txt = this.innerText.replace('Copied!', '').trim(); copyToClipboard(txt); const tt = this.querySelector('.copy-tooltip'); if (tt) { tt.style.display = 'block'; setTimeout(() => { tt.style.display = 'none'; }, 1500); } });
                });
                const aboutH = document.querySelector('.about-us-header'); const aboutC = document.querySelector('.about-us-content');
                if (aboutH && aboutC) { if (!aboutC.classList.contains('initialized')) { aboutC.style.display = 'none'; aboutC.classList.add('initialized'); } aboutH.addEventListener('click', function() { aboutC.style.display = (aboutC.style.display === 'none' ? 'block' : 'none'); }); }
            }, 1000);
        });
        </script>
        """, unsafe_allow_html=True)

    # --- Apply CSS & JS --- 
    apply_custom_css()
    add_javascript_functionality()

    # --- Capture User Agent --- 
    def capture_user_agent():
        if "user_agent" not in st.session_state:
            try:
                component_value = components.html(
                    "<script>window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', key: 'user_agent_capture', value: navigator.userAgent}, '*')</script>",
                    height=0, key="user_agent_capture"
                )
                if component_value: st.session_state.user_agent = component_value
                elif "user_agent_capture" in st.session_state and st.session_state.user_agent_capture: st.session_state.user_agent = st.session_state.user_agent_capture
                else: st.session_state.user_agent = "Unknown (Pending)"
            except Exception: st.session_state.user_agent = "Unknown (Failed)"
    capture_user_agent()

    # --- Initialize Session State --- 
    def initialize_session_state():
        defaults = {
            "df": None, "forecast_df": None, "metrics": None, "forecast_fig": None, 
            "loss_fig": None, "model_trained": False, "selected_model_type": "Standard", 
            "custom_model": None, "custom_model_seq_len": None, 
            "standard_model": None, "standard_model_seq_len": None, # Will be set after loading
            "ai_report": None, "chat_active": False, "messages": [], "chat_model": None,
            "admin_authenticated": False, "user_profile": None, 
            "auth_status": "anonymous", "user_email": None, "user_name": None, "token": None,
            "persistent_user_id": None, "user_agent": "Unknown", "session_id": None, "session_visit_logged": False
        }
        for key, value in defaults.items():
            if key not in st.session_state: st.session_state[key] = value
    initialize_session_state()

    # --- Load Standard Model --- 
    STANDARD_MODEL_PATH = "standard_model.h5"
    STANDARD_MODEL_SEQUENCE_LENGTH = 60 # Default fallback
    if os.path.exists(STANDARD_MODEL_PATH):
        try:
            if st.session_state.standard_model is None:
                _std_model_temp = load_model(STANDARD_MODEL_PATH, compile=False)
                st.session_state.standard_model = _std_model_temp
                st.session_state.standard_model_seq_len = _std_model_temp.input_shape[1]
                STANDARD_MODEL_SEQUENCE_LENGTH = st.session_state.standard_model_seq_len
            else:
                 STANDARD_MODEL_SEQUENCE_LENGTH = st.session_state.standard_model_seq_len
        except Exception as e:
            st.warning(f"Could not load standard model {STANDARD_MODEL_PATH}: {e}. Using default seq len {STANDARD_MODEL_SEQUENCE_LENGTH}.")
            # log_visitor_activity("Model Handling", action="load_standard_model_fail", details={"path": STANDARD_MODEL_PATH, "error": str(e)}) # Logging moved
    else:
        st.warning(f"Standard model file not found: {STANDARD_MODEL_PATH}. Standard model option disabled.")
        # Disable standard model if file not found
        if st.session_state.selected_model_type == "Standard":
             st.session_state.selected_model_type = "Train New" # Default to train new

    # --- User Identification & Tracking (No changes needed here) ---
    def get_client_ip():
        """Get the client's IP address if available."""
        try:
            response = requests.get("https://httpbin.org/ip", timeout=3)
            response.raise_for_status()
            return response.json().get("origin", "Unknown")
        except requests.exceptions.RequestException:
            return "Unknown"
        except Exception:
            return "Unknown"

    def get_persistent_user_id():
        """Generate or retrieve a persistent user ID."""
        if st.session_state.get("auth_status") == "authenticated" and st.session_state.get("user_email"):
            user_id = st.session_state.user_email
            if st.session_state.get("persistent_user_id") != user_id:
                 st.session_state.persistent_user_id = user_id
            return user_id
        if "persistent_user_id" in st.session_state and st.session_state.persistent_user_id:
            return st.session_state.persistent_user_id
        ip_address = get_client_ip()
        user_agent = st.session_state.get("user_agent", "Unknown")
        hash_input = f"{ip_address}-{user_agent}"
        hashed_id = hashlib.sha256(hash_input.encode()).hexdigest()
        persistent_id = f"anon_{hashed_id}"
        st.session_state.persistent_user_id = persistent_id
        return persistent_id

    # --- Visitor Analytics Functions --- (No changes needed here)
    def get_session_id():
        """Create or retrieve a unique session ID."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id

    def log_visitor_activity(page_name, action="page_view", feature_used=None, details=None):
        """Log visitor activity to Firebase."""
        if not firebase_initialized:
            return
        try:
            user_id = get_persistent_user_id()
            profile, is_new = get_or_create_user_profile(user_id)
            should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"]
            access_granted, _ = check_feature_access()
            action_status = "success"
            if should_increment:
                if access_granted:
                    increment_feature_usage(user_id)
                else:
                    action_status = "denied_limit_reached"
            full_action_name = f"{action}_{action_status}"
            ref = db.reference("visitors_log")
            log_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            session_id = get_session_id()
            user_agent = st.session_state.get("user_agent", "Unknown")
            ip_address = get_client_ip()
            log_data = {
                "timestamp": timestamp,
                "persistent_user_id": user_id,
                "is_authenticated": st.session_state.get("auth_status") == "authenticated",
                "visit_count": profile.get("visit_count", 1) if profile else 1,
                "ip_address": ip_address,
                "page": page_name,
                "action": full_action_name,
                "feature_used": feature_used,
                "session_id": session_id,
                "user_agent": user_agent
            }
            if details and isinstance(details, dict):
                log_data["details"] = details
            ref.child(log_id).set(log_data)
        except Exception as e:
            # print(f"Error logging visitor activity: {e}")
            pass

    def update_firebase_profile_on_login(email):
        """Update Firebase profile when a user logs in via Google OAuth."""
        if not firebase_initialized or not email:
            return
        try:
            ref = db.reference(f"users/{email}")
            profile = ref.get()
            now_iso = datetime.datetime.now().isoformat()
            update_data = {
                "is_authenticated": True,
                "last_login_google": now_iso,
                "user_id": email
            }
            if profile is None:
                update_data["first_visit"] = now_iso
                update_data["visit_count"] = 1
                update_data["feature_usage_count"] = 0
                update_data["last_visit"] = now_iso
                ref.set(update_data)
                st.session_state.user_profile = update_data
            else:
                ref.update(update_data)
                st.session_state.user_profile = ref.get()
            log_visitor_activity("Authentication", action="google_login_success", details={"email": email})
        except Exception as e:
            # st.error(f"Firebase error updating profile after Google login for {email}: {e}") # Error shown globally
            log_visitor_activity("Authentication", action="google_login_firebase_update_fail", details={"email": email, "error": str(e)})

    def get_or_create_user_profile(user_id):
        """Get user profile from Firebase or create a new one."""
        if not firebase_initialized:
            return None, False
        try:
            ref = db.reference(f"users/{user_id}")
            profile = ref.get()
            is_new_user = False
            now_iso = datetime.datetime.now().isoformat()
            current_auth_status = st.session_state.get("auth_status") == "authenticated"
            current_email = st.session_state.get("user_email")
            if profile is None:
                is_new_user = True
                profile = {
                    "user_id": user_id,
                    "first_visit": now_iso,
                    "visit_count": 1,
                    "is_authenticated": current_auth_status,
                    "feature_usage_count": 0,
                    "last_visit": now_iso,
                    "email": current_email if current_auth_status else None
                }
                ref.set(profile)
            else:
                if "session_visit_logged" not in st.session_state:
                    profile["visit_count"] = profile.get("visit_count", 0) + 1
                    profile["last_visit"] = now_iso
                    profile["is_authenticated"] = current_auth_status
                    if current_auth_status:
                         profile["email"] = current_email
                    else:
                         profile["email"] = profile.get("email")
                    ref.update({
                        "visit_count": profile["visit_count"], 
                        "last_visit": profile["last_visit"],
                        "is_authenticated": profile["is_authenticated"],
                        "email": profile.get("email")
                    })
                    st.session_state.session_visit_logged = True
            return profile, is_new_user
        except Exception as e:
            # st.warning(f"Firebase error getting/creating user profile for {user_id}: {e}") # Warning shown globally
            return None, False

    def increment_feature_usage(user_id):
        """Increment the feature usage count for the user in Firebase."""
        if not firebase_initialized:
            return False
        try:
            ref = db.reference(f"users/{user_id}/feature_usage_count")
            current_count = ref.get() or 0
            ref.set(current_count + 1)
            if "user_profile" in st.session_state and st.session_state.user_profile:
                st.session_state.user_profile["feature_usage_count"] = current_count + 1
            return True
        except Exception as e:
            # st.warning(f"Firebase error incrementing usage count for {user_id}: {e}") # Warning shown globally
            return False

    # --- Authentication Check & Google OAuth --- 
    def check_feature_access():
        """Check if user can access advanced features based on usage count and auth status."""
        is_authenticated = st.session_state.get("auth_status") == "authenticated"
        if is_authenticated:
            return True, "Access granted (Authenticated)."
        if "user_profile" not in st.session_state or st.session_state.user_profile is None:
            user_id = get_persistent_user_id()
            st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
            if st.session_state.user_profile is None:
                # st.warning("Could not retrieve user profile. Feature access may be limited.") # Warning shown globally
                return False, "Cannot verify usage limit. Access denied."
        usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
        if usage_count < ADVANCED_FEATURE_LIMIT:
            return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
        else:
            return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in to continue."

    def get_user_info_from_google(token):
        """Fetch user info using the access token."""
        if not token or "access_token" not in token:
            return None
        try:
            response = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {token['access_token']}"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching user info from Google: {e}") # Show error here
            return None
        except Exception as e:
             st.error(f"Unexpected error fetching user info: {e}") # Show error here
             return None

    def show_google_login():
        """Handles the Google OAuth login flow using streamlit-oauth."""
        if not google_oauth_configured: # Check if OAuth config loaded successfully
            st.sidebar.error("Google Sign-In not configured.")
            log_visitor_activity("Authentication", action="google_login_fail_config_missing")
            return

        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
        
        if "token" not in st.session_state:
            usage_count = st.session_state.user_profile.get("feature_usage_count", 0) if st.session_state.user_profile else 0
            if usage_count >= ADVANCED_FEATURE_LIMIT:
                st.sidebar.warning(f"Usage limit reached.")
                st.sidebar.info("Log in with Google to continue.")
                result = oauth2.authorize_button(
                    name="Login with Google",
                    icon="https://www.google.com/favicon.ico",
                    redirect_uri=REDIRECT_URI,
                    scope="openid email profile",
                    key="google_login",
                    extras_params={"prompt": "consent", "access_type": "offline"}
                )
                if result:
                    st.session_state.token = result.get("token")
                    user_info = get_user_info_from_google(st.session_state.token)
                    if user_info and user_info.get("email"):
                        st.session_state.auth_status = "authenticated"
                        st.session_state.user_email = user_info.get("email")
                        st.session_state.user_name = user_info.get("name")
                        st.session_state.persistent_user_id = user_info.get("email")
                        update_firebase_profile_on_login(user_info.get("email"))
                        st.success(f"Logged in as {st.session_state.user_name} ({st.session_state.user_email}).")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Login successful, but failed to retrieve user information from Google.")
                        if "token" in st.session_state: del st.session_state.token 
                        log_visitor_activity("Authentication", action="google_login_fail_userinfo")
        else:
            if not st.session_state.get("user_email"):
                 user_info = get_user_info_from_google(st.session_state.token)
                 if user_info and user_info.get("email"):
                     st.session_state.auth_status = "authenticated"
                     st.session_state.user_email = user_info.get("email")
                     st.session_state.user_name = user_info.get("name")
                     st.session_state.persistent_user_id = user_info.get("email")
                 else:
                     st.error("Could not verify login status. Please log in again.")
                     if "token" in st.session_state: del st.session_state.token
                     st.session_state.auth_status = "anonymous"
                     st.session_state.user_email = None
                     st.session_state.user_name = None
                     log_visitor_activity("Authentication", action="google_reauth_fail_userinfo")
                     st.rerun()
                     
            if st.session_state.get("user_email"):
                st.sidebar.success(f"Logged in as: {st.session_state.user_name} ({st.session_state.user_email})")
                if st.sidebar.button("Logout", key="google_logout"):
                    log_visitor_activity("Authentication", action="google_logout", details={"email": st.session_state.user_email})
                    if "token" in st.session_state: del st.session_state.token
                    st.session_state.auth_status = "anonymous"
                    st.session_state.user_email = None
                    st.session_state.user_name = None
                    st.session_state.persistent_user_id = None
                    st.rerun()

    # --- Initialize user profile (needs session state) ---
    if "user_profile" not in st.session_state or st.session_state.user_profile is None:
        if firebase_initialized:
            user_id = get_persistent_user_id()
            st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        else:
            st.session_state.user_profile = None

    # --- Helper Functions (Data Loading, Model Building, Prediction - unchanged) ---
    @st.cache_data
    def load_and_clean_data(uploaded_file_content):
        try:
            df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
            if df.shape[1] < 2: st.error("File must have Date and Level columns."); return None
            date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time"])), None)
            level_col = next((c for c in df.columns if any(k in c.lower() for k in ["level", "groundwater", "gwl"])), None)
            if not date_col or not level_col: st.error("Cannot find Date or Level columns."); return None
            st.success(f"Using columns: Date='{date_col}', Level='{level_col}'.")
            df = df.rename(columns={date_col: "Date", level_col: "Level"})[["Date", "Level"]]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
            init_rows = len(df)
            df.dropna(subset=["Date", "Level"], inplace=True)
            if len(df) < init_rows: st.warning(f"Dropped {init_rows - len(df)} rows with invalid values.")
            if df.empty: st.error("No valid data remaining."); return None
            df = df.sort_values("Date").reset_index(drop=True).drop_duplicates("Date", keep="first")
            if df["Level"].isnull().any():
                missing = df["Level"].isnull().sum()
                df["Level"] = df["Level"].interpolate(method="linear", limit_direction="both")
                st.warning(f"Filled {missing} missing levels via interpolation.")
            if df["Level"].isnull().any(): st.error("Could not fill all missing values."); return None
            st.success("Data loaded & cleaned.")
            return df
        except Exception as e: st.error(f"Error loading/cleaning data: {e}"); return None

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    @st.cache_resource
    def load_keras_model_from_file(uploaded_file_obj, model_name_for_log):
        temp_path = f"temp_{model_name_for_log}.h5"
        try:
            with open(temp_path, "wb") as f: f.write(uploaded_file_obj.getbuffer())
            model = load_model(temp_path, compile=False)
            seq_len = model.input_shape[1]
            st.success(f"Loaded {model_name_for_log}. Seq len: {seq_len}")
            log_visitor_activity("Model Handling", action="load_custom_model", details={"name": model_name_for_log, "seq_len": seq_len})
            return model, seq_len
        except Exception as e: 
            st.error(f"Error loading model {model_name_for_log}: {e}")
            log_visitor_activity("Model Handling", action="load_custom_model_fail", details={"name": model_name_for_log, "error": str(e)})
            return None, None
        finally: 
            if os.path.exists(temp_path): os.remove(temp_path)

    def build_lstm_model(sequence_length, n_features=1):
        model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
        all_preds = []
        current_seq = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
        @tf.function
        def predict_step(inp): return model(inp, training=True)
        prog_bar = st.progress(0)
        stat_txt = st.empty()
        for i in range(n_iterations):
            iter_preds_scaled = []
            temp_seq = current_seq.copy()
            for _ in range(n_steps):
                next_pred_s = predict_step(temp_seq).numpy()[0,0]
                iter_preds_scaled.append(next_pred_s)
                temp_seq = np.append(temp_seq[:, 1:, :], np.array([[next_pred_s]]).reshape(1,1,1), axis=1)
            all_preds.append(iter_preds_scaled)
            prog_bar.progress((i + 1) / n_iterations)
            stat_txt.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
        prog_bar.empty(); stat_txt.empty()
        preds_arr_s = np.array(all_preds)
        mean_preds_s = np.mean(preds_arr_s, axis=0)
        std_devs_s = np.std(preds_arr_s, axis=0)
        ci_mult = 2.5
        mean_preds = scaler.inverse_transform(mean_preds_s.reshape(-1, 1)).flatten()
        lower_b = scaler.inverse_transform((mean_preds_s - ci_mult * std_devs_s).reshape(-1, 1)).flatten()
        upper_b = scaler.inverse_transform((mean_preds_s + ci_mult * std_devs_s).reshape(-1, 1)).flatten()
        min_uncert_pct = 0.05
        for i in range(len(mean_preds)):
            curr_range_pct = (upper_b[i] - lower_b[i]) / mean_preds[i] if mean_preds[i] != 0 else 0
            if curr_range_pct < min_uncert_pct:
                uncert_val = mean_preds[i] * min_uncert_pct / 2
                lower_b[i] = mean_preds[i] - uncert_val
                upper_b[i] = mean_preds[i] + uncert_val
        return mean_preds, lower_b, upper_b

    def calculate_metrics(y_true, y_pred):
        y_t, y_p = np.array(y_true), np.array(y_pred)
        if len(y_t) == 0 or len(y_p) == 0 or len(y_t) != len(y_p): return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        mape = np.inf if not np.all(y_t != 0) else mean_absolute_percentage_error(y_t, y_p) * 100
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    # --- Plotting Functions (unchanged) ---
    def create_forecast_plot(hist_df, fc_df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Level"], mode="lines", name="Historical", line=dict(color="rgb(31, 119, 180)")))
        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Upper_CI"], mode="lines", name="Upper CI", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Lower_CI"], mode="lines", name="Lower CI", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=False))
        fig.update_layout(title="Groundwater Level Forecast", xaxis_title="Date", yaxis_title="Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
        return fig

    def create_loss_plot(hist_dict):
        if not hist_dict or "loss" not in hist_dict or "val_loss" not in hist_dict:
            fig = go.Figure().update_layout(title="No Training History", xaxis_title="Epoch", yaxis_title="Loss")
            fig.add_annotation(text="Training history not found.", showarrow=False)
            return fig
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist_dict["loss"], mode="lines", name="Training Loss"))
        fig.add_trace(go.Scatter(y=hist_dict["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Model Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified", template="plotly_white")
        return fig

    # --- AI Report Generation --- 
    def generate_ai_report(data_summary, forecast_summary, metrics, model_details):
        if not gemini_configured or gemini_model_report is None: return "AI report disabled or model not initialized."
        try:
            prompt = f"""Generate concise scientific report: Groundwater Level Forecast.
            Data: {data_summary}
            Forecast: {forecast_summary}
            Metrics: {metrics}
            Model: {model_details}
            Structure: Intro, Data Overview, Methodology (LSTM type: {model_details.get('type', 'N/A')}), Results (trend, range, metrics), Discussion (interpretation, uncertainty, performance), Conclusion. 
            Instructions: Objective, concise (300-500 words), interpret provided info only.
            """
            response = gemini_model_report.generate_content(prompt)
            return response.text
        except Exception as e: return f"Error generating AI report: {e}"

    # --- AI Chat Functionality --- 
    def initialize_chat():
        if not gemini_configured or gemini_model_chat is None: return
        if "messages" not in st.session_state: st.session_state.messages = []
        if "chat_model" not in st.session_state or st.session_state.chat_model is None:
             try:
                 st.session_state.chat_model = gemini_model_chat.start_chat(history=[])
             except Exception as e:
                 st.warning(f"Failed to start chat model: {e}")
                 st.session_state.chat_model = None

    def display_chat_history():
        for msg in st.session_state.messages:
            role_cls = "user-message" if msg["role"] == "user" else "ai-message"
            st.markdown(f'<div class="chat-message {role_cls}">{msg["content"]}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)

    def handle_chat_input(prompt):
        if not gemini_configured or st.session_state.get("chat_model") is None: st.error("Chat disabled or model not initialized."); return
        if not prompt: return
        log_visitor_activity("AI Chat", action="send_chat_message")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-message user-message">{prompt}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
        try:
            with st.spinner("AI thinking..."):
                response = st.session_state.chat_model.send_message(prompt)
            st.session_state.messages.append({"role": "model", "content": response.text})
            st.markdown(f'<div class="chat-message ai-message">{response.text}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
        except Exception as e:
            err_msg = f"Chat error: {e}"
            st.session_state.messages.append({"role": "model", "content": err_msg})
            st.error(err_msg)
            log_visitor_activity("AI Chat", action="chat_error", details={"error": str(e)})

    # --- PDF Generation (No changes needed here) ---
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "DeepHydro AI Forecasting Report", 0, 1, "C")
            self.ln(5)
        def chapter_title(self, title):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, title, 0, 1, "L")
            self.ln(4)
        def chapter_body(self, body):
            self.set_font("Arial", "", 10)
            self.multi_cell(0, 5, body)
            self.ln()
        def add_plot(self, plot_bytes, title):
            self.chapter_title(title)
            try:
                temp_img_path = "temp_plot.png"
                with open(temp_img_path, "wb") as f: f.write(plot_bytes)
                self.image(temp_img_path, x=10, w=self.w - 20)
                os.remove(temp_img_path)
                self.ln(5)
            except Exception as e: self.chapter_body(f"[Error adding plot: {e}]")
        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def create_pdf_report(forecast_fig, loss_fig, metrics, ai_report_text):
        pdf = PDF()
        pdf.add_page()
        if forecast_fig: pdf.add_plot(forecast_fig.to_image(format="png", scale=2), "Forecast Visualization")
        else: pdf.chapter_body("[Forecast plot not available]")
        if loss_fig: pdf.add_plot(loss_fig.to_image(format="png", scale=2), "Model Training Loss")
        else: pdf.chapter_body("[Training loss plot not available]")
        pdf.chapter_title("Performance Metrics")
        if metrics: pdf.chapter_body("\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]))
        else: pdf.chapter_body("[Metrics not available]")
        pdf.chapter_title("AI Generated Analysis")
        if ai_report_text: pdf.chapter_body(ai_report_text)
        else: pdf.chapter_body("[AI analysis not available]")
        try: return pdf.output(dest="S").encode("latin-1")
        except Exception as e: st.error(f"Error generating PDF bytes: {e}"); return None

    # --- Streamlit App UI --- 
    st.sidebar.image("logo.png", width=100)
    st.sidebar.title("DeepHydro AI")
    st.sidebar.markdown("--- User Access ---")
    show_google_login()
    st.sidebar.markdown("--- Options ---")
    uploaded_file = st.sidebar.file_uploader("Upload Excel Data (.xlsx)", type=["xlsx"], key="data_uploader")
    if uploaded_file:
        if st.session_state.df is None or uploaded_file.name != st.session_state.get("uploaded_file_name"):
            st.session_state.df = load_and_clean_data(uploaded_file.getvalue())
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.forecast_df = None; st.session_state.metrics = None; st.session_state.forecast_fig = None
            st.session_state.ai_report = None; st.session_state.chat_active = False
            log_visitor_activity("Sidebar", action="upload_data", details={"filename": uploaded_file.name})
            st.rerun()

    model_options = ["Train New", "Upload Custom (.h5)"]
    default_model_index = 0 # Default to Train New
    if st.session_state.standard_model is not None: # Only add Standard if loaded
        model_options.insert(0, "Standard")
        if st.session_state.selected_model_type == "Standard": default_model_index = 0
        elif st.session_state.selected_model_type == "Train New": default_model_index = 1
        else: default_model_index = 2
    else:
        if st.session_state.selected_model_type == "Train New": default_model_index = 0
        else: default_model_index = 1
        
    model_choice = st.sidebar.radio("Select Model", model_options, key="model_selector", index=default_model_index)
    if model_choice != st.session_state.selected_model_type:
        log_visitor_activity("Sidebar", action="select_model_type", details={"type": model_choice})
        st.session_state.selected_model_type = model_choice
        st.session_state.forecast_df = None; st.session_state.metrics = None; st.session_state.forecast_fig = None
        st.session_state.ai_report = None; st.session_state.chat_active = False
        st.rerun()

    custom_model_file = None
    if st.session_state.selected_model_type == "Upload Custom (.h5)":
        custom_model_file = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"], key="model_uploader")
        if custom_model_file:
            if st.session_state.custom_model is None or custom_model_file.name != st.session_state.get("custom_model_file_name"):
                st.session_state.custom_model, st.session_state.custom_model_seq_len = load_keras_model_from_file(custom_model_file, custom_model_file.name)
                st.session_state.custom_model_file_name = custom_model_file.name
                st.rerun()

    st.sidebar.markdown("--- Forecast Settings ---")
    mc_iterations = st.sidebar.number_input("MC Dropout Iterations (C.I.)", min_value=10, max_value=500, value=100, step=10, key="mc_iter")
    forecast_horizon = st.sidebar.number_input("Forecast Horizon (steps)", min_value=1, max_value=365, value=12, step=1, key="horizon")
    can_access_forecast, reason_forecast = check_feature_access()
    can_access_report, reason_report = check_feature_access()
    can_access_chat, reason_chat = check_feature_access()
    st.sidebar.markdown("--- Actions ---")
    run_forecast_disabled = st.session_state.df is None or \
                            (st.session_state.selected_model_type == "Upload Custom (.h5)" and st.session_state.custom_model is None) or \
                            (st.session_state.selected_model_type == "Standard" and st.session_state.standard_model is None) or \
                            not can_access_forecast
    if st.sidebar.button("Run Forecast", key="run_forecast_btn", disabled=run_forecast_disabled):
        log_visitor_activity("Sidebar", action="run_forecast", feature_used="Forecast", details={"type": st.session_state.selected_model_type, "horizon": forecast_horizon, "mc": mc_iterations})
        model_to_use, sequence_length, model_type_log = None, None, st.session_state.selected_model_type
        if st.session_state.selected_model_type == "Standard":
            model_to_use, sequence_length = st.session_state.standard_model, st.session_state.standard_model_seq_len
        elif st.session_state.selected_model_type == "Upload Custom (.h5)":
            model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len
        elif st.session_state.selected_model_type == "Train New":
            if st.session_state.model_trained and st.session_state.custom_model:
                 model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len
                 model_type_log = "Trained New"
            else: st.warning("Train model first."); st.stop()
        if model_to_use is None or sequence_length is None: st.error("Model unavailable."); log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Model unavailable"}); st.stop()
        if st.session_state.df is None or len(st.session_state.df) <= sequence_length: st.error(f"Need > {sequence_length} data points."); log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Insufficient data"}); st.stop()
        with st.spinner("Forecasting..."):
            try:
                df_fc = st.session_state.df.copy()
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df_fc["Level"].values.reshape(-1, 1))
                last_seq_s = scaled_data[-sequence_length:]
                mean_p, lower_b, upper_b = predict_with_dropout_uncertainty(model_to_use, last_seq_s, forecast_horizon, mc_iterations, scaler, sequence_length)
                last_date = df_fc["Date"].iloc[-1]
                fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
                st.session_state.forecast_df = pd.DataFrame({"Date": fc_dates, "Forecast": mean_p, "Lower_CI": lower_b, "Upper_CI": upper_b})
                st.session_state.metrics = st.session_state.get("training_metrics", {"Info": "Metrics from training"})
                st.session_state.forecast_fig = create_forecast_plot(df_fc, st.session_state.forecast_df)
                st.success("Forecast complete!")
                log_visitor_activity("Sidebar", action="run_forecast_success", feature_used="Forecast", details={"type": model_type_log, "horizon": forecast_horizon})
                st.rerun()
            except Exception as e: st.error(f"Forecast error: {e}"); log_visitor_activity("Sidebar", action="run_forecast_fail", feature_used="Forecast", details={"type": model_type_log, "error": str(e)})
    if not can_access_forecast and not run_forecast_disabled: st.sidebar.warning(reason_forecast)

    gen_report_disabled = st.session_state.forecast_df is None or not gemini_configured or not can_access_report
    if st.sidebar.button("Generate AI Report", key="gen_report_btn", disabled=gen_report_disabled):
        log_visitor_activity("Sidebar", action="generate_report", feature_used="AI Report")
        with st.spinner("Generating AI analysis..."):
            try:
                data_sum = f"Hist data: {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}, {len(st.session_state.df)} points. Avg level: {st.session_state.df['Level'].mean():.2f}."
                fc_sum = f"Forecast: {forecast_horizon} steps. Range: {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. CI: {st.session_state.forecast_df['Lower_CI'].min():.2f} to {st.session_state.forecast_df['Upper_CI'].max():.2f}."
                met_sum = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in st.session_state.metrics.items()]) if st.session_state.metrics else "Metrics N/A."
                model_det = {"type": st.session_state.selected_model_type, "seq_len": sequence_length, "horizon": forecast_horizon, "mc_iter": mc_iterations}
                st.session_state.ai_report = generate_ai_report(data_sum, fc_sum, met_sum, model_det)
                st.success("AI Report generated!")
                log_visitor_activity("Sidebar", action="generate_report_success", feature_used="AI Report")
                st.rerun()
            except Exception as e: st.error(f"Report generation error: {e}"); log_visitor_activity("Sidebar", action="generate_report_fail", feature_used="AI Report", details={"error": str(e)})
    if not can_access_report and not gen_report_disabled: st.sidebar.warning(reason_report)

    pdf_bytes = None
    if st.session_state.forecast_fig or st.session_state.ai_report:
        try: pdf_bytes = create_pdf_report(st.session_state.forecast_fig, st.session_state.loss_fig, st.session_state.metrics, st.session_state.ai_report)
        except Exception as e: st.sidebar.warning(f"PDF prep error: {e}"); log_visitor_activity("Sidebar", action="prepare_pdf_fail", details={"error": str(e)})
    st.sidebar.download_button("Download PDF Report", data=pdf_bytes if pdf_bytes else b"", file_name="DeepHydro_AI_Report.pdf", mime="application/pdf", key="download_pdf_btn", disabled=not pdf_bytes, help="Generate forecast/report first.", on_click=log_visitor_activity if pdf_bytes else None, args=("Sidebar",) if pdf_bytes else None, kwargs={"action": "download_pdf"} if pdf_bytes else None)

    activate_chat_disabled = st.session_state.forecast_df is None or not gemini_configured or not can_access_chat
    if st.sidebar.button("Activate Chat", key="activate_chat_btn", disabled=activate_chat_disabled):
        log_visitor_activity("Sidebar", action="activate_chat", feature_used="AI Chat")
        if not st.session_state.chat_active:
            initialize_chat()
            st.session_state.chat_active = True
            if st.session_state.forecast_df is not None:
                context = f"Chat Context: Forecast generated. Hist data: {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. Forecast: {forecast_horizon} steps, range {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. Ask questions."
                st.session_state.messages = []
                # Re-initialize chat model if needed
                if st.session_state.chat_model is None and gemini_configured:
                    try:
                        st.session_state.chat_model = gemini_model_chat.start_chat(history=[])
                    except Exception as e:
                        st.warning(f"Failed to re-initialize chat model: {e}")
                if st.session_state.chat_model:
                    st.session_state.messages.append({"role": "model", "content": context})
                    st.success("Chat activated with context!")
                else:
                    st.warning("Chat activated, but model failed to initialize.")
            else: st.warning("Chat activated, no forecast context.")
            st.rerun()
        else: st.sidebar.info("Chat already active.")
    if not can_access_chat and not activate_chat_disabled: st.sidebar.warning(reason_chat)

    st.title("Groundwater Level Forecasting & Analysis")
    tab_titles = ["Home / Data View", "Forecast Results", "Train Model", "AI Report", "AI Chatbot", "Admin Analytics"]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # Home / Data View
        log_visitor_activity("Tab: Home", "page_view")
        st.header("Welcome to DeepHydro AI")
        st.markdown("<div class='app-intro'>Forecast groundwater levels using LSTM AI. Upload data, choose model, run forecast, generate reports, or chat.</div>", unsafe_allow_html=True)
        st.subheader("Uploaded Data Overview")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head())
            st.metric("Data Points", len(st.session_state.df))
            st.metric("Date Range", f"{st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}")
            st.line_chart(st.session_state.df.set_index('Date')['Level'])
        else: st.info("Upload Excel file via sidebar.")

    with tabs[1]: # Forecast Results
        log_visitor_activity("Tab: Forecast", "page_view")
        st.header("Forecast Results")
        if st.session_state.forecast_fig: st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
        else: st.info("Run forecast from sidebar.")
        if st.session_state.forecast_df is not None: st.dataframe(st.session_state.forecast_df)
        if st.session_state.metrics: st.subheader("Performance Metrics"); st.json(st.session_state.metrics)

    with tabs[2]: # Train Model
        log_visitor_activity("Tab: Train Model", "page_view")
        st.header("Train New LSTM Model")
        if st.session_state.selected_model_type != "Train New": st.info("Select 'Train New' in sidebar.")
        elif st.session_state.df is None: st.warning("Upload data first.")
        else:
            st.subheader("Training Parameters")
            col1, col2, col3 = st.columns(3)
            with col1: seq_len_tr = st.number_input("Seq Len", 10, 180, 60, 5, key="tr_seq")
            with col2: epochs_tr = st.number_input("Epochs", 1, 200, 20, 1, key="tr_epochs")
            with col3: batch_tr = st.number_input("Batch Size", 8, 128, 32, 8, key="tr_batch")
            test_size_tr = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="tr_split")
            if st.button("Train New LSTM Model", key="train_btn"):
                log_visitor_activity("Tab: Train Model", action="train_model", details={"seq": seq_len_tr, "epochs": epochs_tr, "batch": batch_tr, "split": test_size_tr})
                with st.spinner("Training..."):
                    try:
                        df_tr = st.session_state.df.copy()
                        scaler_tr = MinMaxScaler(feature_range=(0, 1))
                        scaled_data_tr = scaler_tr.fit_transform(df_tr["Level"].values.reshape(-1, 1))
                        X, y = create_sequences(scaled_data_tr, seq_len_tr)
                        if len(X) == 0: st.error(f"Need > {seq_len_tr} data points."); log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"reason": "Insufficient data"}); st.stop()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_tr, random_state=42)
                        X_train, X_test = X_train.reshape((-1, seq_len_tr, 1)), X_test.reshape((-1, seq_len_tr, 1))
                        model_tr = build_lstm_model(seq_len_tr)
                        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                        hist = model_tr.fit(X_train, y_train, epochs=epochs_tr, batch_size=batch_tr, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
                        y_pred_s = model_tr.predict(X_test)
                        y_pred_tr = scaler_tr.inverse_transform(y_pred_s)
                        y_test_tr = scaler_tr.inverse_transform(y_test.reshape(-1, 1))
                        st.session_state.metrics = calculate_metrics(y_test_tr, y_pred_tr)
                        st.session_state.training_metrics = st.session_state.metrics
                        st.session_state.loss_fig = create_loss_plot(hist.history)
                        st.session_state.custom_model = model_tr
                        st.session_state.custom_model_seq_len = seq_len_tr
                        st.session_state.model_trained = True
                        st.session_state.trained_model_seq_len = seq_len_tr
                        st.success("Training complete!"); st.balloons()
                        log_visitor_activity("Tab: Train Model", action="train_model_success", details={"metrics": st.session_state.metrics})
                        st.rerun()
                    except Exception as e: st.error(f"Training error: {e}"); log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"error": str(e)})
            if st.session_state.model_trained: st.subheader("Training Results"); st.plotly_chart(st.session_state.loss_fig, use_container_width=True); st.json(st.session_state.metrics); st.info("Model ready for forecasting.")

    with tabs[3]: # AI Report
        log_visitor_activity("Tab: AI Report", "page_view")
        st.header("AI-Generated Scientific Report")
        if not gemini_configured: st.warning("AI report disabled (Gemini not configured).")
        elif st.session_state.ai_report: st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
        else: st.info("Generate report from sidebar after forecast.")

    with tabs[4]: # AI Chatbot
        log_visitor_activity("Tab: AI Chatbot", "page_view")
        st.header("AI Chatbot")
        if not gemini_configured: st.warning("AI chat disabled (Gemini not configured).")
        elif st.session_state.chat_active: display_chat_history(); user_prompt = st.chat_input("Ask about forecast..."); handle_chat_input(user_prompt)
        elif st.session_state.forecast_df is None: st.info("Run forecast, then activate chat.")
        else: st.info("Activate chat from sidebar.")

    with tabs[5]: # Admin Analytics
        log_visitor_activity("Tab: Admin Analytics", "page_view")
        st.header("Admin Analytics Dashboard")
        if not st.session_state.admin_authenticated:
            password = st.text_input("Admin Password:", type="password", key="admin_pass")
            if st.button("Login Admin", key="admin_login"):
                if password == admin_password:
                    st.session_state.admin_authenticated = True; log_visitor_activity("Admin Auth", action="admin_login_success"); st.rerun()
                else: st.error("Incorrect password."); log_visitor_activity("Admin Auth", action="admin_login_fail")
        else:
            st.success("Admin access granted.")
            if st.button("Logout Admin", key="admin_logout"):
                st.session_state.admin_authenticated = False; log_visitor_activity("Admin Auth", action="admin_logout"); st.rerun()
            st.markdown("--- Visitor Data ---")
            if firebase_initialized:
                visitor_df = fetch_visitor_logs()
                if not visitor_df.empty:
                    st.dataframe(visitor_df)
                    st.markdown("--- Visualizations ---")
                    charts = create_visitor_charts(visitor_df)
                    for chart in charts: st.plotly_chart(chart, use_container_width=True)
                    try:
                        csv = visitor_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Analytics CSV", csv, "visitor_analytics.csv", "text/csv", key="dl_analytics", on_click=log_visitor_activity, args=("Admin Analytics",), kwargs={"action": "download_analytics_csv"})
                    except Exception as e_csv: st.warning(f"CSV download error: {e_csv}"); log_visitor_activity("Admin Analytics", action="download_analytics_csv_fail", details={"error": str(e_csv)})
                else: st.info("No visitor logs found.")
            else: st.warning("Firebase not initialized. Cannot display analytics.")

    st.markdown("<div class='about-us-header'>About DeepHydro AI ▼</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="about-us-content">
    DeepHydro AI uses LSTM AI for groundwater forecasts. Upload data, choose/train model, run forecast, get reports, chat.
    **Config Note:** This version reads config from the `APP_CONFIG_JSON` environment variable. Ensure it's set securely in your deployment (e.g., Render.com Secret File) with the correct JSON structure including Firebase, Google OAuth, Gemini API key, and admin password.
    </div>
    """, unsafe_allow_html=True)

# --- (End of Script) --- 

