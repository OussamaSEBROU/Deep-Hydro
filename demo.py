# flake8: noqa
import streamlit as st
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
import hashlib
import html
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component
import logging

# --- Setup Logging --- 
# Configure logging (basic setup)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Prevent duplicate logging in Streamlit console if handlers are added elsewhere
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Add a handler to ensure logs go somewhere (e.g., stderr, visible in Render logs)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Logging configured.")

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 3
STANDARD_MODEL_PATH = "standard_model.h5"
DEFAULT_SEQUENCE_LENGTH = 60

# --- Global Flags & Configuration --- 
# These are set by load_configuration
APP_CONFIG = {}
google_oauth_configured = False
admin_password_configured = False
config_load_error = None

# --- Configuration Loading --- 
# @st.cache_data # Caching config might be okay, but env vars can change on redeploy
def load_configuration():
    """Loads configuration from APP_CONFIG_JSON environment variable."""
    global APP_CONFIG, google_oauth_configured, admin_password_configured, config_load_error
    # Reset state in case of re-run with changed env var
    APP_CONFIG = {}
    google_oauth_configured = False
    admin_password_configured = False
    config_load_error = None
    
    logging.info("Attempting to load configuration from environment variable...")
    config_json_str = os.getenv("APP_CONFIG_JSON")
    if not config_json_str:
        config_load_error = "Environment variable APP_CONFIG_JSON not found. Application configuration is missing."
        logging.error(config_load_error)
        return {}

    try:
        APP_CONFIG = json.loads(config_json_str)
        logging.info("Configuration loaded successfully from APP_CONFIG_JSON.")
        
        # Validate essential config sections
        if "firebase_service_account" not in APP_CONFIG:
            logging.warning("Firebase configuration (\"firebase_service_account\") missing.")
        if "google_oauth" not in APP_CONFIG or not all(k in APP_CONFIG["google_oauth"] for k in ["client_id", "client_secret", "redirect_uri"]):
            logging.warning("Google OAuth configuration (\"google_oauth\") incomplete or missing.")
            google_oauth_configured = False
        else:
            google_oauth_configured = True
            logging.info("Google OAuth config found.")
        if "google_api_key" not in APP_CONFIG:
            logging.warning("Google API Key (\"google_api_key\") missing.")
        if "admin_password" not in APP_CONFIG:
             logging.warning("Admin Password (\"admin_password\") missing.")
             admin_password_configured = False
        else:
            admin_password_configured = True
            logging.info("Admin password config found.")
        return APP_CONFIG

    except json.JSONDecodeError as e:
        config_load_error = f"Error decoding APP_CONFIG_JSON: {e}. Check the JSON format."
        logging.error(config_load_error)
        APP_CONFIG = {}
        google_oauth_configured = False
        admin_password_configured = False
        return {}

# --- Service Initializations (Cached) --- 
@st.cache_resource
def initialize_firebase_cached(_config):
    """Initializes Firebase Admin SDK. Cached resource."""
    logging.info("Attempting to initialize Firebase...")
    if firebase_admin._apps:
        logging.info("Firebase already initialized.")
        return True
    if not _config or "firebase_service_account" not in _config:
        logging.warning("Firebase Service Account details not found in config. Firebase disabled.")
        return False

    try:
        cred_dict = _config["firebase_service_account"]
        if not all(k in cred_dict for k in ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]):
             logging.error("Firebase Service Account JSON in configuration is incomplete.")
             return False

        cred = credentials.Certificate(cred_dict)
        firebase_url = _config.get("firebase_database_url")
        if not firebase_url or firebase_url == "OPTIONAL_YOUR_FIREBASE_DB_URL":
            project_id = cred_dict.get("project_id")
            if project_id:
                firebase_url = f"https://{project_id}-default-rtdb.firebaseio.com/"
                logging.info(f"Firebase Database URL not set, using default: {firebase_url}")
            else:
                logging.error("Cannot determine Firebase DB URL: \"firebase_database_url\" missing and \"project_id\" missing.")
                return False

        firebase_admin.initialize_app(cred, {"databaseURL": firebase_url})
        logging.info("Firebase initialized successfully.")
        return True

    except ValueError as e:
        logging.error(f"Error initializing Firebase with provided credentials: {e}")
        return False
    except Exception as e:
        logging.error(f"Firebase initialization failed: {e}. Analytics may be disabled.")
        return False

@st.cache_resource
def configure_gemini_cached(_config):
    """Configures the Gemini API client. Cached resource."""
    logging.info("Attempting to configure Gemini API...")
    GEMINI_API_KEY = _config.get("google_api_key")
    if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSy..." and GEMINI_API_KEY != "Gemini_api_key":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Configure models separately if needed, or just return True/False
            # Example: Check if model exists
            # genai.get_model("gemini-pro") 
            logging.info("Gemini API configured successfully.")
            return True
        except Exception as e:
            logging.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
            return False
    else:
        logging.warning("Gemini API Key (\"google_api_key\") not found or is placeholder. AI features disabled.")
        return False

@st.cache_resource
def load_standard_model_cached(model_path):
    """Loads the standard Keras model from path. Cached resource."""
    logging.info(f"Attempting to load standard model from {model_path}...")
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            sequence_length = model.input_shape[1]
            logging.info(f"Standard model loaded successfully. Sequence length: {sequence_length}")
            return model, sequence_length
        except Exception as e:
            logging.error(f"Could not load standard model from {model_path}: {e}.")
            return None, None
    else:
        logging.warning(f"Standard model file not found: {model_path}. Standard model option disabled.")
        return None, None

# --- Run Initializations --- 
# Load config first, as it\"s needed by others
APP_CONFIG = load_configuration()

# Initialize services using cached functions
fbase_init_success = initialize_firebase_cached(APP_CONFIG)
gemini_init_success = configure_gemini_cached(APP_CONFIG)

# Load standard model into session state using cached function
if "standard_model_data" not in st.session_state:
    model, seq_len = load_standard_model_cached(STANDARD_MODEL_PATH)
    st.session_state.standard_model_data = {"model": model, "seq_len": seq_len}

# Retrieve model and seq len from session state
standard_model = st.session_state.standard_model_data.get("model")
STANDARD_MODEL_SEQUENCE_LENGTH = st.session_state.standard_model_data.get("seq_len") or DEFAULT_SEQUENCE_LENGTH

# --- User Identification & Tracking (Keep Original Logic) --- 
# (get_client_ip, get_persistent_user_id, update_firebase_profile_on_login, 
#  get_or_create_user_profile, increment_feature_usage - kept as is)
def get_client_ip():
    try:
        response = requests.get("https://httpbin.org/ip", timeout=3)
        response.raise_for_status()
        return response.json().get("origin", "Unknown")
    except Exception:
        return "Unknown"

def get_persistent_user_id():
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

def update_firebase_profile_on_login(email):
    if not fbase_init_success or not email: return
    try:
        safe_email_key = email.replace(".", "_dot_").replace("@", "_at_")
        ref = db.reference(f"users/{safe_email_key}")
        profile = ref.get()
        now_iso = datetime.datetime.now().isoformat()
        update_data = {"is_authenticated": True, "last_login_google": now_iso, "user_id": email, "email": email}
        if profile is None:
            update_data.update({"first_visit": now_iso, "visit_count": 1, "feature_usage_count": 0, "last_visit": now_iso})
            ref.set(update_data)
            st.session_state.user_profile = update_data
        else:
            ref.update(update_data)
            st.session_state.user_profile = ref.get()
        log_visitor_activity("Authentication", action="google_login_success", details={"email": email})
    except Exception as e:
        logging.error(f"Firebase error updating profile for {email}: {e}")
        log_visitor_activity("Authentication", action="google_login_firebase_update_fail", details={"email": email, "error": str(e)})

def get_or_create_user_profile(user_id):
    if not fbase_init_success: return None, False
    try:
        safe_user_id_key = user_id.replace(".", "_dot_").replace("@", "_at_") if "@" in user_id else user_id
        ref = db.reference(f"users/{safe_user_id_key}")
        profile = ref.get()
        is_new_user = False
        now_iso = datetime.datetime.now().isoformat()
        current_auth_status = st.session_state.get("auth_status") == "authenticated"
        current_email = st.session_state.get("user_email")
        if profile is None:
            is_new_user = True
            profile = {"user_id": user_id, "first_visit": now_iso, "visit_count": 1, "is_authenticated": current_auth_status, "feature_usage_count": 0, "last_visit": now_iso, "email": current_email if current_auth_status else None}
            ref.set(profile)
        else:
            if "session_visit_logged" not in st.session_state:
                profile["visit_count"] = profile.get("visit_count", 0) + 1
                profile["last_visit"] = now_iso
                profile["is_authenticated"] = current_auth_status
                profile["email"] = current_email if current_auth_status else profile.get("email")
                ref.update({"visit_count": profile["visit_count"], "last_visit": profile["last_visit"], "is_authenticated": profile["is_authenticated"], "email": profile.get("email")})
                st.session_state.session_visit_logged = True
        return profile, is_new_user
    except Exception as e:
        logging.warning(f"Firebase error getting/creating profile for {user_id}: {e}")
        return None, False

def increment_feature_usage(user_id):
    if not fbase_init_success: return False
    try:
        safe_user_id_key = user_id.replace(".", "_dot_").replace("@", "_at_") if "@" in user_id else user_id
        ref = db.reference(f"users/{safe_user_id_key}/feature_usage_count")
        current_count = ref.get() or 0
        ref.set(current_count + 1)
        if "user_profile" in st.session_state and st.session_state.user_profile:
            st.session_state.user_profile["feature_usage_count"] = current_count + 1
        return True
    except Exception as e:
        logging.warning(f"Firebase error incrementing usage for {user_id}: {e}")
        return False

# --- Authentication Check & Google OAuth --- 
def check_feature_access():
    is_authenticated = st.session_state.get("auth_status") == "authenticated"
    if is_authenticated: return True, "Access granted (Authenticated)."
    if not fbase_init_success: # If firebase failed, allow limited access anyway
        logging.warning("Firebase not initialized, cannot check usage count. Allowing limited access.")
        # We need a way to track usage without firebase, maybe session state? 
        # For now, let's just allow ADVANCED_FEATURE_LIMIT uses per session.
        if "session_usage_count" not in st.session_state:
            st.session_state.session_usage_count = 0
        if st.session_state.session_usage_count < ADVANCED_FEATURE_LIMIT:
             return True, f"Access granted (Session Usage: {st.session_state.session_usage_count}/{ADVANCED_FEATURE_LIMIT})."
        else:
             return False, f"Session usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in (if available) or restart session."

    # If firebase is working, use the profile
    if "user_profile" not in st.session_state or st.session_state.user_profile is None:
        user_id = get_persistent_user_id()
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        if st.session_state.user_profile is None:
            logging.warning("Could not retrieve user profile. Feature access limited.")
            return False, "Cannot verify usage limit. Access denied."
            
    usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
    if usage_count < ADVANCED_FEATURE_LIMIT:
        return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
    else:
        return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in."

# Define OAuth endpoints (constants)
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

def get_user_info_from_google(token):
    if not token or "access_token" not in token: return None
    try:
        # Use the token to fetch user info
        auth_header = {"Authorization": f"Bearer {token['access_token']}"}
        response = requests.get("https://www.googleapis.com/oauth2/v1/userinfo", headers=auth_header)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        user_info = response.json()
        logging.info(f"Successfully fetched user info for: {user_info.get('email')}")
        return user_info
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching user info from Google: {e}")
        if e.response is not None:
            logging.error(f"Google API Response Status: {e.response.status_code}")
            logging.error(f"Google API Response Body: {e.response.text}")
        return None
    except Exception as e:
         logging.error(f"Unexpected error fetching user info: {e}")
         return None

def show_google_login():
    """Displays the Google Login button and handles the OAuth flow."""
    if not google_oauth_configured:
        st.sidebar.warning("Google Sign-In unavailable (Configuration missing).", icon="⚠️")
        if fbase_init_success: log_visitor_activity("Authentication", action="google_login_fail_config_missing")
        return

    # Retrieve OAuth details safely from APP_CONFIG
    oauth_config = APP_CONFIG.get("google_oauth", {})
    client_id = oauth_config.get("client_id")
    client_secret = oauth_config.get("client_secret")
    redirect_uri = oauth_config.get("redirect_uri")

    # Crucial Check: Ensure all OAuth params are present
    if not all([client_id, client_secret, redirect_uri]):
        st.sidebar.error("Google OAuth configuration incomplete in settings.", icon="❌")
        logging.error("OAuth component cannot be initialized: client_id, client_secret, or redirect_uri is missing.")
        return
        
    # Add debugging info (visible in sidebar expander)
    with st.sidebar.expander("OAuth Debug Info (for admin)", expanded=False):
        st.text(f"Client ID: {client_id}")
        st.text(f"Redirect URI: {redirect_uri}")
        st.caption("Ensure the Redirect URI above EXACTLY matches one listed in your Google Cloud Console Authorized redirect URIs for this Client ID.")

    try:
        oauth2 = OAuth2Component(client_id, client_secret, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OAuth component: {e}", icon="❌")
        logging.error(f"OAuth2Component initialization failed: {e}")
        return

    # --- Login Flow ---
    if "token" not in st.session_state or st.session_state.token is None:
        # Check usage limit before showing login prompt
        usage_count = -1
        if fbase_init_success and "user_profile" in st.session_state and st.session_state.user_profile:
            usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
        elif "session_usage_count" in st.session_state: # Fallback if firebase failed
            usage_count = st.session_state.session_usage_count
            
        show_login_prompt = False
        if usage_count != -1 and usage_count >= ADVANCED_FEATURE_LIMIT:
            st.sidebar.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached.", icon="✋")
            st.sidebar.info("Please log in with Google to continue.")
            show_login_prompt = True
        
        # Always show login button if not authenticated
        logging.info(f"Attempting to display authorize button. Redirect URI: {redirect_uri}")
        try:
            result = oauth2.authorize_button(
                name="Login with Google",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=redirect_uri,
                scope="openid email profile",
                key="google_login_button",
                extras_params={"prompt": "consent", "access_type": "offline"}
            )
        except Exception as e:
            st.sidebar.error(f"Error displaying login button: {e}", icon="❌")
            logging.error(f"oauth2.authorize_button failed: {e}")
            result = None

        if result and "token" in result:
            st.session_state.token = result.get("token")
            logging.info("Received token from OAuth component. Fetching user info...")
            user_info = get_user_info_from_google(st.session_state.token)
            if user_info and user_info.get("email"):
                st.session_state.auth_status = "authenticated"
                st.session_state.user_email = user_info.get("email")
                st.session_state.user_name = user_info.get("name")
                st.session_state.user_picture = user_info.get("picture")
                logging.info(f"Authentication successful for {st.session_state.user_email}")
                update_firebase_profile_on_login(st.session_state.user_email)
                st.sidebar.success(f"Logged in as {st.session_state.user_name}")
                time.sleep(1) # Short delay to show message
                st.rerun()
            else:
                st.session_state.auth_status = "error"
                st.session_state.token = None # Clear token on error
                st.sidebar.error("Failed to get user info from Google after login.", icon="❌")
                logging.error("OAuth login flow failed: Could not retrieve user info from Google.")
                log_visitor_activity("Authentication", action="google_login_fail_userinfo")
        elif result: # Handle potential errors from the component (e.g., user cancels)
            logging.warning(f"OAuth result received but no token: {result}")
            # Check for specific error codes if the component provides them
            error_desc = result.get("error_description") or result.get("error")
            if error_desc:
                 st.sidebar.error(f"Login failed: {error_desc}", icon="❌")
            log_visitor_activity("Authentication", action="google_login_fail_no_token", details=result)
            
    # --- Logout Flow ---
    elif st.session_state.get("auth_status") == "authenticated":
        if st.session_state.get("user_picture"):
            st.sidebar.image(st.session_state.user_picture, width=50)
        st.sidebar.write(f"Logged in as: {st.session_state.get('user_name', st.session_state.get('user_email'))}")
        if st.sidebar.button("Logout", key="logout_button"):
            logging.info(f"User {st.session_state.get("user_email")} initiated logout.")
            # Clear session state related to auth
            st.session_state.auth_status = "logged_out"
            st.session_state.token = None
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.user_picture = None
            # Optionally revoke token (if needed and supported by component)
            # try: oauth2.revoke_token(st.session_state.token) except: pass
            log_visitor_activity("Authentication", action="logout")
            st.rerun()

# --- Visitor Analytics Functions (Keep Original Logic, use fbase_init_success flag) ---
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None, details=None):
    if not fbase_init_success: return # Check if Firebase initialized successfully
    try:
        user_id = get_persistent_user_id()
        profile, _ = get_or_create_user_profile(user_id)
        
        # Increment usage count if a limited feature is used successfully
        should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"]
        if should_increment and action not in ["run_forecast_denied", "generate_report_denied", "activate_chat_denied"]:
            increment_feature_usage(user_id)
        elif not fbase_init_success and should_increment: # Increment session counter if firebase failed
             if "session_usage_count" in st.session_state:
                 st.session_state.session_usage_count += 1

        ref = db.reference("visitor_logs")
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
            "action": action,
            "feature_used": feature_used,
            "session_id": session_id,
            "user_agent": user_agent,
            "details": details if details else {}
        }
        
        ref.child(log_id).set(log_data)
    except Exception as e:
        logging.warning(f"Error logging visitor activity: {e}")
        pass # Silently fail logging

def fetch_visitor_logs():
    if not fbase_init_success: return pd.DataFrame()
    try:
        ref = db.reference("visitor_logs")
        visitors_data = ref.get()
        if not visitors_data: return pd.DataFrame()
        visitors_list = []
        for log_id, data in visitors_data.items():
            if isinstance(data, dict):
                data["log_id"] = log_id
                if "details" in data and isinstance(data["details"], dict):
                    for k, v in data["details"].items():
                        safe_key = f"detail_{str(k).replace(".", "_").replace("$", "_")}"
                        data[safe_key] = v
                    del data["details"]
                visitors_list.append(data)
        df = pd.DataFrame(visitors_list)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        return df
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
    # (Keep original logic)
    if visitor_df.empty: return []
    figures = []
    try:
        df = visitor_df.copy()
        if "timestamp" not in df.columns: return [] # Need timestamp
        df["date"] = df["timestamp"].dt.date
        
        # 1. Daily visitors
        if "persistent_user_id" in df.columns:
            daily_visitors = df.groupby("date")["persistent_user_id"].nunique().reset_index(name="unique_users")
            daily_visitors["date"] = pd.to_datetime(daily_visitors["date"])
            fig1 = px.line(daily_visitors, x="date", y="unique_users", title="Daily Unique Visitors")
            figures.append(fig1)
        
        # 2. Action counts
        if "action" in df.columns:
            action_counts = df["action"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            fig2 = px.bar(action_counts, x="action", y="count", title="Activity Counts by Action")
            figures.append(fig2)

        # 3. Authenticated vs Anonymous
        if "persistent_user_id" in df.columns and "is_authenticated" in df.columns:
            latest_status = df.sort_values("timestamp").groupby("persistent_user_id")["is_authenticated"].last().reset_index()
            auth_counts = latest_status["is_authenticated"].value_counts().reset_index()
            auth_counts.columns = ["is_authenticated", "count"]
            auth_counts["status"] = auth_counts["is_authenticated"].map({True: "Authenticated", False: "Anonymous"})
            fig3 = px.pie(auth_counts, values="count", names="status", title="User Authentication Status (Latest Known)")
            figures.append(fig3)
            
        # 4. Hourly activity heatmap
        try:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()
            day_order = [\"Monday\", \"Tuesday\", \"Wednesday\", \"Thursday\", \"Friday\", \"Saturday\", \"Sunday\"]
            hourly_activity = df.groupby([\"day_of_week\", \"hour\"]).size().reset_index(name=\"count\")
            
            if len(hourly_activity) > 0:
                hourly_pivot = hourly_activity.pivot_table(values=\"count\", index=\"day_of_week\", columns=\"hour\", fill_value=0)
                available_days = set(hourly_pivot.index) & set(day_order)
                ordered_available_days = [day for day in day_order if day in available_days]
                hourly_pivot = hourly_pivot.reindex(ordered_available_days)
                available_hours = sorted(hourly_pivot.columns)
                
                fig4 = px.imshow(hourly_pivot, 
                                labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
                                x=[str(h) for h in available_hours],
                                y=ordered_available_days,
                                title="Visitor Activity by Hour and Day")
                figures.append(fig4)
            else:
                fig4 = go.Figure().update_layout(title="Visitor Activity by Hour and Day (No Data)")
                fig4.add_annotation(text="Not enough data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                figures.append(fig4)
        except Exception as heatmap_err:
            st.warning(f"Could not generate hourly activity heatmap: {heatmap_err}")
            fig4 = go.Figure().update_layout(title="Visitor Activity by Hour and Day (Error)")
            fig4.add_annotation(text="Error generating heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            figures.append(fig4)
            
    except Exception as e:
        st.error(f"Error creating visitor charts: {e}")
        return []
    return figures

# --- Admin Analytics Dashboard (Keep Original Logic, use fbase_init_success flag) ---
def render_admin_analytics():
    st.header("Admin Analytics Dashboard")
    
    if not admin_password_configured:
        st.warning("Admin password not configured. Access disabled.")
        return
        
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.info("Admin access required.")
        admin_password_input = st.text_input("Admin Password", type="password", key="admin_pass_input")
        if st.button("Login", key="admin_login_btn"):
            correct_password = APP_CONFIG.get("admin_password")
            if admin_password_input == correct_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
                log_visitor_activity("Admin Login", action="login_fail")
    else:
        if not fbase_init_success:
            st.warning("Firebase not initialized. Cannot display visitor logs.")
            if st.button("Logout Admin", key="admin_logout_btn_no_fb"):
                st.session_state.admin_authenticated = False
                log_visitor_activity("Admin Dashboard", action="logout_admin")
                st.rerun()
            return
            
        visitor_df = fetch_visitor_logs()
        
        if visitor_df.empty:
            st.info("No visitor data available yet.")
        else:
            st.subheader("Visitor Statistics")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Activities Logged", len(visitor_df))
            with col2: st.metric("Unique Visitors", visitor_df["persistent_user_id"].nunique() if "persistent_user_id" in visitor_df else 0)
            with col3:
                today_visitors = 0
                if "timestamp" in visitor_df:
                     today = datetime.datetime.now().date()
                     today_visitors = visitor_df[visitor_df["timestamp"].dt.date == today]["persistent_user_id"].nunique() if "persistent_user_id" in visitor_df else 0
                st.metric("Today\"s Unique Visitors", today_visitors)
            
            st.subheader("Visitor Analytics")
            try:
                charts = create_visitor_charts(visitor_df)
                for fig in charts:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as chart_err: st.error(f"Error displaying charts: {chart_err}")
            
            st.subheader("Raw Visitor Data")
            # (Filtering and display logic kept as is)
            col1_filter, col2_filter = st.columns(2)
            date_range = None; user_id_filter = \"All\"
            try:
                if "timestamp" in visitor_df and not visitor_df["timestamp"].isnull().all():
                    min_date = visitor_df["timestamp"].min().date(); max_date = visitor_df["timestamp"].max().date()
                    with col1_filter: date_range = st.date_input("Date Range", [min_date, max_date], key="admin_date_filter")
                else: 
                    with col1_filter: st.info("No timestamp data for date filter.")
                if "persistent_user_id" in visitor_df and not visitor_df["persistent_user_id"].isnull().all():
                    user_id_options = [\"All\"] + visitor_df["persistent_user_id"].unique().tolist()
                    with col2_filter: user_id_filter = st.selectbox("Filter by User ID", options=user_id_options, index=0, key="admin_user_filter")
                else: 
                    with col2_filter: st.info("No user ID data for filter.")
            except Exception as filter_setup_err: st.warning(f"Error setting up filters: {filter_setup_err}")
            try:
                filtered_df = visitor_df.copy()
                if date_range and len(date_range) == 2 and "timestamp" in filtered_df:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[(filtered_df["timestamp"].dt.date >= start_date) & (filtered_df["timestamp"].dt.date <= end_date)]
                if user_id_filter != \"All\" and "persistent_user_id" in filtered_df:
                    filtered_df = filtered_df[filtered_df["persistent_user_id"] == user_id_filter]
                display_cols = [col for col in [\"timestamp\", \"persistent_user_id\", \"is_authenticated\", \"visit_count\", \"page\", \"action\", \"feature_used\", \"ip_address\", \"session_id\", \"log_id\"] if col in filtered_df.columns]
                st.dataframe(filtered_df[display_cols])
                if st.button("Export Filtered to CSV", key="admin_export_btn"):
                    csv = filtered_df[display_cols].to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f\"<a href=\"data:file/csv;base64,{b64}\" download=\"filtered_visitor_logs.csv\">Download CSV File</a>\"
                    st.markdown(href, unsafe_allow_html=True)
                    log_visitor_activity("Admin Dashboard", action="export_csv")
            except Exception as filter_err:
                st.error(f"Error applying filters or displaying data: {filter_err}")
                fallback_cols = [col for col in [\"timestamp\", \"persistent_user_id\", \"action\"] if col in visitor_df.columns]
                st.dataframe(visitor_df[fallback_cols])
                
        if st.button("Logout Admin", key="admin_logout_btn"):
            st.session_state.admin_authenticated = False
            log_visitor_activity("Admin Dashboard", action="logout_admin")
            st.rerun()

# --- Custom CSS (Keep Original) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* Existing CSS rules */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-weight: 600; font-size: 1.8rem; }
    h2 { font-weight: 600; font-size: 1.5rem; }
    h3, h4 { font-weight: 500; }
    .stButton > button { border-radius: 4px; font-weight: 500; transition: all 0.3s; padding: 0.5rem 1rem; }
    .stButton > button:hover { opacity: 0.8; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    .css-1d391kg, .css-12oz5g7 { padding: 1rem; } /* Adjust padding if needed */
    .sidebar .block-container { font-size: 0.9rem; }
    .sidebar h1 { font-size: 1.4rem; }
    .sidebar h2 { font-size: 1.2rem; }
    .card-container { border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .chat-message { padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; position: relative; }
    .user-message { border-left: 4px solid #1E88E5; background-color: #e3f2fd; }
    .ai-message { border-left: 4px solid #78909C; background-color: #eceff1; }
    .chat-message:active { opacity: 0.7; }
    .copy-tooltip { position: absolute; top: 0.5rem; right: 0.5rem; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; display: none; background-color: #555; color: white; z-index: 10; }
    .chat-message:active .copy-tooltip { display: block; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; border-radius: 4px 4px 0 0; }
    [data-testid="stMetricValue"] { font-weight: 600; }
    .about-us-header { cursor: pointer; padding: 0.5rem; border-radius: 4px; margin-top: 1rem; font-weight: 500; background-color: #f5f5f5; border: 1px solid #e0e0e0; }
    .about-us-content { padding: 0.8rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; border: 1px solid #e0e0e0; border-top: none; }
    .app-intro { padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #1E88E5; background-color: #e3f2fd; }
    </style>
    """, unsafe_allow_html=True) --- JavaScript (Keep Original) ---
def add_javascript_functionality():
    st.markdown("""
    <script>
    // Function to copy text to clipboard
    function copyToClipboard(text) {
        const textarea = document.createElement(\"textarea\");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand(\"copy\");
        document.body.removeChild(textarea);
    }
    
    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    const setupInteractions = () => {
        // Copy functionality for chat messages
        const chatMessages = document.querySelectorAll(\".chat-message\");
        chatMessages.forEach(function(message) {
            if (!message.querySelector(\".copy-tooltip\")) {
                const tooltip = document.createElement(\"span\");
                tooltip.className = \"copy-tooltip\";
                tooltip.textContent = \"Copied!\";
                message.appendChild(tooltip);
            }
            let longPressTimer;
            const handleStart = (e) => {
                longPressTimer = setTimeout(() => {
                    const textToCopy = message.innerText.replace(\"Copied!\", \"\").trim();
                    copyToClipboard(textToCopy);
                    const tooltip = message.querySelector(\".copy-tooltip\");
                    if (tooltip) {
                        tooltip.style.display = \"block\";
                        setTimeout(() => { tooltip.style.display = \"none\"; }, 1500);
                    }
                }, 500);
            };
            const handleEnd = () => { clearTimeout(longPressTimer); };
            message.removeEventListener(\"mousedown\", handleStart);
            message.removeEventListener(\"mouseup\", handleEnd);
            message.removeEventListener(\"mouseleave\", handleEnd);
            message.removeEventListener(\"touchstart\", handleStart);
            message.removeEventListener(\"touchend\", handleEnd);
            message.removeEventListener(\"touchmove\", handleEnd);
            message.addEventListener(\"mousedown\", handleStart);
            message.addEventListener(\"mouseup\", handleEnd);
            message.addEventListener(\"mouseleave\", handleEnd);
            message.addEventListener(\"touchstart\", handleStart, { passive: true });
            message.addEventListener(\"touchend\", handleEnd);
            message.addEventListener(\"touchmove\", handleEnd, { passive: true });
        });
        
        // Collapsible About Us
        const aboutUsHeader = document.querySelector(\".about-us-header\");
        const aboutUsContent = document.querySelector(\".about-us-content\");
        if (aboutUsHeader && aboutUsContent) {
            if (!aboutUsContent.classList.contains(\"initialized\")) {
                 aboutUsContent.style.display = \"none\";
                 aboutUsContent.classList.add(\"initialized\");
            }
            aboutUsHeader.removeEventListener(\"click\", toggleAboutUs);
            aboutUsHeader.addEventListener(\"click\", toggleAboutUs);
        }
    };
    
    const toggleAboutUs = () => {
        const aboutUsContent = document.querySelector(\".about-us-content\");
        if (aboutUsContent) {
            aboutUsContent.style.display = (aboutUsContent.style.display === \"none\") ? \"block\" : \"none\";
        }
    };

    const observer = new MutationObserver(debounce((mutationsList, observer) => {
        for(const mutation of mutationsList) {
            if (mutation.type === \"childList\" && mutation.addedNodes.length > 0) {
                 setupInteractions();
                 break;
            }
        }
    }, 250));

    const targetNode = window.parent.document.querySelector(\"section.main\");
    if (targetNode) {
        observer.observe(targetNode, { childList: true, subtree: true });
    } else {
        window.addEventListener(\"load\", () => {
             const fallbackNode = window.parent.document.querySelector(\"section.main\");
             if(fallbackNode) observer.observe(fallbackNode, { childList: true, subtree: true });
        });
    }

    if (document.readyState === \"complete\") {
        setupInteractions();
    } else {
        window.addEventListener(\"load\", setupInteractions);
    }
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration --- 
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css()

# --- Capture User Agent --- 
def capture_user_agent():
    if \"user_agent\" not in st.session_state:
        try:
            user_agent_val = components.html(
                """
                <script>
                window.parent.postMessage({
                    isStreamlitMessage: true,
                    type: "streamlit:setComponentValue",
                    key: "user_agent_capture_component", 
                    value: navigator.userAgent
                }, "*");
                </script>
                """,
                height=0,
                key="user_agent_capture_component"
            )
            component_value = st.session_state.get("user_agent_capture_component")
            st.session_state.user_agent = component_value or user_agent_val or "Unknown (Capture Pending)"
        except Exception as e:
            logging.warning(f"User agent capture failed: {e}")
            st.session_state.user_agent = "Unknown (Capture Failed)"

capture_user_agent() # Attempt capture early

# --- Helper Functions (Keep Original - Data Loading, Model Building, Prediction, Plotting, Gemini) ---
# (load_and_clean_data, create_sequences, load_keras_model_from_file, 
#  build_lstm_model, predict_with_dropout_uncertainty, calculate_metrics, 
#  create_forecast_plot, create_loss_plot, generate_gemini_report, 
#  get_gemini_chat_response, run_forecast_pipeline, StreamlitCallback - kept as is)
@st.cache_data
def load_and_clean_data(uploaded_file_content):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        if df.shape[1] < 2: st.error("File must have at least two columns (Date, Level)."); return None
        date_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["date", "time"])), None)
        level_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["level", "groundwater", "gwl"])), None)
        if not date_col: st.error("Cannot find Date column (e.g., named \"Date\", \"Time\")."); return None
        if not level_col: st.error("Cannot find Level column (e.g., named \"Level\", \"Groundwater Level\")."); return None
        st.success(f"Identified columns: Date=\"{date_col}\", Level=\"{level_col}\". Renaming to \"Date\" and \"Level\".")
        df = df.rename(columns={date_col: "Date", level_col: "Level"})[["Date", "Level"]]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
        initial_rows = len(df)
        df.dropna(subset=["Date", "Level"], inplace=True)
        if len(df) < initial_rows: st.warning(f"Dropped {initial_rows - len(df)} rows with invalid/missing date or level values.")
        if df.empty: st.error("No valid data remaining after cleaning."); return None
        df = df.sort_values(by="Date").reset_index(drop=True).drop_duplicates(subset=["Date"], keep="first")
        if df["Level"].isnull().any():
            missing_before = df["Level"].isnull().sum()
            df["Level"] = df["Level"].interpolate(method="linear", limit_direction="both")
            st.warning(f"Filled {missing_before} missing level values using linear interpolation.")
        if df["Level"].isnull().any(): st.error("Could not fill all missing values even after interpolation."); return None
        st.success("Data loaded and cleaned successfully!")
        return df
    except Exception as e: st.error(f"An unexpected error occurred during data loading/cleaning: {e}"); return None

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource
def load_keras_model_from_file(uploaded_file_obj, model_name_for_log):
    temp_model_path = f"temp_{model_name_for_log.replace(\" \", \"_\")}.h5"
    try:
        with open(temp_model_path, "wb") as f: f.write(uploaded_file_obj.getbuffer())
        model = load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded {model_name_for_log}. Inferred sequence length: {sequence_length}")
        return model, sequence_length
    except Exception as e: st.error(f"Error loading Keras model {model_name_for_log}: {e}"); return None, None
    finally: 
        if os.path.exists(temp_model_path): os.remove(temp_model_path)

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    has_dropout = any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)

    @tf.function
    def predict_step_training_mode(inp):
        return model(inp, training=has_dropout)
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting MC Dropout ({n_iterations} iterations)...")
    
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step_training_mode(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            new_step = np.array([[next_pred_scaled]]).reshape(1, 1, 1)
            temp_sequence = np.append(temp_sequence[:, 1:, :], new_step, axis=1)
        all_predictions.append(iteration_predictions_scaled)
        progress_fraction = (i + 1) / n_iterations
        progress_bar.progress(progress_fraction)
        status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
        
    progress_bar.empty(); status_text.empty()
    predictions_array_scaled = np.array(all_predictions)
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    ci_multiplier = 1.96
    lower_bound_scaled = mean_preds_scaled - ci_multiplier * std_devs_scaled
    upper_bound_scaled = mean_preds_scaled + ci_multiplier * std_devs_scaled
    lower_bound = scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()
    lower_bound = np.minimum(lower_bound, mean_preds)
    upper_bound = np.maximum(upper_bound, mean_preds)
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true_valid = y_true[valid_indices]; y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    non_zero_true = y_true_valid[y_true_valid != 0]; non_zero_pred = y_pred_valid[y_true_valid != 0]
    mape = np.inf
    if len(non_zero_true) > 0: mape = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def create_forecast_plot(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=True))
    fig.update_layout(title="Groundwater Level: Historical Data & LSTM Forecast", xaxis_title="Date", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_loss_plot(history_dict):
    if not history_dict or not isinstance(history_dict, dict) or "loss" not in history_dict or "val_loss" not in history_dict:
        fig = go.Figure().update_layout(title="No Training History Available", xaxis_title="Epoch", yaxis_title="Loss")
        fig.add_annotation(text="Training history is not available.",xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    history_df = pd.DataFrame(history_dict); history_df["Epoch"] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", template="plotly_white")
    return fig

def get_gemini_model(model_type="report"):
    """Gets a configured Gemini model instance."""
    if not gemini_init_success: return None
    try:
        model_name = "gemini-pro" # Or choose based on type
        generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        return model
    except Exception as e:
        logging.error(f"Failed to get Gemini model ({model_type}): {e}")
        return None

def generate_gemini_report(hist_df, forecast_df, metrics, language):
    model = get_gemini_model("report")
    if model is None: return "AI report generation disabled or model not configured."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    try:
        prompt = f"""Act as a professional hydrologist analyzing groundwater level data and LSTM forecast results. Provide a concise scientific report in {language}. 
        
        **Analysis Context:**
        - Historical Data Period: {hist_df["Date"].min():%Y-%m-%d} to {hist_df["Date"].max():%Y-%m-%d}
        - Forecast Period: {forecast_df["Date"].min():%Y-%m-%d} to {forecast_df["Date"].max():%Y-%m-%d}
        - Model Evaluation Metrics (Validation/Pseudo-Validation):
            - RMSE: {metrics.get("RMSE", "N/A"):.4f}
            - MAE: {metrics.get("MAE", "N/A"):.4f}
            - MAPE: {metrics.get("MAPE", "N/A"):.2f}%

        **Instructions:**
        1.  **Introduction:** Briefly state the purpose - analyzing historical groundwater levels and evaluating an LSTM forecast.
        2.  **Historical Data Insights:** Describe key trends, seasonality (if apparent), and notable high/low periods observed in the historical data. Mention the overall range (min/max levels).
        3.  **Forecast Evaluation:** Comment on the model\"s performance based on the provided metrics (RMSE, MAE, MAPE). Interpret what these values imply about the forecast accuracy (e.g., low MAE suggests good average prediction).
        4.  **Forecast Analysis:** Describe the predicted trend for the forecast period. Mention the confidence interval (Lower/Upper CI) and what it suggests about the forecast uncertainty.
        5.  **Conclusion & Recommendations:** Summarize the findings. Briefly mention potential applications or limitations based on the analysis and forecast uncertainty. Avoid definitive statements; use cautious language.

        **Historical Data Summary:**
        {hist_df["Level"].describe().to_string()}
        
        **Forecast Data Summary:**
        {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}
        """
        response = model.generate_content(prompt)
        # Add safety check if needed
        # if response.prompt_feedback.block_reason:
        #    logging.warning(f"Gemini report generation blocked: {response.prompt_feedback.block_reason}")
        #    return f"Report generation blocked due to safety settings."
        return response.text
    except Exception as e: 
        logging.error(f"Error generating AI report: {e}")
        st.error(f"Error generating AI report: {e}")
        return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    model = get_gemini_model("chat")
    if model is None: return "AI chat disabled or model not configured."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient context for AI chat."
    try:
        history_context = "\n".join([f"{sender}: {str(message)}" for sender, message in chat_hist])
        context = f"""You are an AI assistant helping a user understand groundwater level data and forecast results. 
        Use the following context to answer the user\"s query accurately and concisely. 
        
        **Available Context:**
        - Historical Data Period: {hist_df["Date"].min():%Y-%m-%d} to {hist_df["Date"].max():%Y-%m-%d}
        - Forecast Period: {forecast_df["Date"].min():%Y-%m-%d} to {forecast_df["Date"].max():%Y-%m-%d}
        - Model Evaluation Metrics: RMSE={metrics.get("RMSE", "N/A"):.4f}, MAE={metrics.get("MAE", "N/A"):.4f}, MAPE={metrics.get("MAPE", "N/A"):.2f}%
        - AI Report Summary (if available): {str(ai_report)[:500] if ai_report else \"Not generated yet.\"}...
        - Historical Data Summary:
        {hist_df["Level"].describe().to_string()}
        - Forecast Data Summary:
        {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}

        **Chat History:**
        {history_context}
        
        **User Query:** {user_query}
        
        **Instructions:** Answer the user\"s query based *only* on the provided context and chat history. Be helpful and informative. If the answer isn\"t in the context, say so.
        
        AI Assistant:"""
        response = model.generate_content(context)
        # Add safety check if needed
        # if response.prompt_feedback.block_reason:
        #    logging.warning(f"Gemini chat response blocked: {response.prompt_feedback.block_reason}")
        #    return f"Chat response blocked due to safety settings."
        return response.text
    except Exception as e: 
        logging.error(f"Error in AI chat: {e}")
        st.error(f"Error in AI chat: {e}")
        return f"Error in AI chat: {e}"

def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    st.info(f"Starting forecast pipeline with model: {model_choice}")
    model = None; history_data = None
    # Determine sequence length based on choice
    model_sequence_length = DEFAULT_SEQUENCE_LENGTH # Fallback default
    if model_choice == "Standard Pre-trained Model":
        model, seq_len = st.session_state.standard_model_data.get("model"), st.session_state.standard_model_data.get("seq_len")
        if model is None or seq_len is None:
            st.error("Standard model not loaded correctly. Cannot proceed.")
            return None, None, None, None
        model_sequence_length = seq_len
    elif model_choice == "Train New Model":
        model_sequence_length = sequence_length_train_param
    elif model_choice == "Upload Custom .h5 Model":
        if custom_model_file_obj is None:
            st.error("Custom model selected, but no file uploaded.")
            return None, None, None, None
        # Load the custom model (cached within the function)
        model, inferred_seq_len = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
        if model is None or inferred_seq_len is None:
            st.error("Failed to load custom model.")
            return None, None, None, None
        model_sequence_length = inferred_seq_len
    else:
        st.error("Invalid model choice."); return None, None, None, None
        
    st.session_state.model_sequence_length = model_sequence_length # Store the length being used
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    try:
        st.info(f"Using sequence length: {model_sequence_length}")
        st.info("Step 1: Preprocessing Data (Scaling)...")
        scaler_obj.fit(df["Level"].values.reshape(-1, 1))
        if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
             scaler_obj.data_min_ = np.array([custom_scaler_min_param])
             scaler_obj.data_max_ = np.array([custom_scaler_max_param])
             scaler_obj.data_range_ = scaler_obj.data_max_ - scaler_obj.data_min_
             # Recalculate internal params based on new min/max
             scaler_obj.min_ = np.array([custom_scaler_min_param * scaler_obj.scale_ + scaler_obj.feature_range[0]])
             scaler_obj.scale_ = (scaler_obj.feature_range[1] - scaler_obj.feature_range[0]) / (scaler_obj.data_range_)
             st.warning("Applied custom scaling parameters. Ensure they match the model\"s training data.")
        scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
        st.info("Data scaling complete.")

        st.info(f"Step 2: Creating sequences (length {model_sequence_length})...")
        if len(df) <= model_sequence_length: 
            st.error(f"Not enough data ({len(df)}) for sequence length {model_sequence_length}. Need at least {model_sequence_length + 1} points."); return None, None, None, None
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0: 
            st.error("Could not create sequences from the data."); return None, None, None, None
        st.info(f"Sequences created: {len(X)}")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        if model_choice == "Train New Model":
            st.info(f"Step 3a: Training New Model (Epochs: {epochs_train_param})...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_train) == 0 or len(X_val) == 0: 
                st.error("Not enough sequences for train/validation split."); return None, None, None, None
            model = build_lstm_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            st_callback = StreamlitCallback(epochs_train_param)
            history_obj = model.fit(X_train, y_train, epochs=epochs_train_param, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, st_callback], verbose=0)
            st_callback.close()
            history_data = history_obj.history
            st.success("Training complete.")
            st.info("Evaluating trained model on validation set...")
            val_predictions_scaled = model.predict(X_val)
            val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
            y_val_actual = scaler_obj.inverse_transform(y_val)
            evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
            st.success("Evaluation complete.")
        else: # Pre-trained or Custom Uploaded (model is already loaded)
            st.info("Step 3b: Evaluating Pre-trained/Uploaded Model (Pseudo-Validation)...")
            if len(X) > 5:
                val_split_idx = max(1, int(len(X) * 0.8))
                X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                if len(X_val_pseudo) > 0:
                    val_predictions_scaled = model.predict(X_val_pseudo)
                    val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo)
                    evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation complete.")
                else: st.warning("Not enough data for pseudo-validation split.")
            else: st.warning("Not enough sequences for pseudo-validation.")

        st.info(f"Step 4: Forecasting {forecast_horizon} Steps (MC Dropout: {mc_iterations_param})...")
        last_sequence_scaled_for_pred = scaled_data[-model_sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_dropout_uncertainty(model, last_sequence_scaled_for_pred, forecast_horizon, mc_iterations_param, scaler_obj, model_sequence_length)
        st.success("Forecasting complete.")

        last_date = df["Date"].iloc[-1]
        try: 
            freq = pd.infer_freq(df["Date"].dropna()); freq = freq or "D"
            date_offset = pd.tseries.frequencies.to_offset(freq)
        except Exception as freq_err:
            logging.warning(f"Could not infer frequency or create offset: {freq_err}. Defaulting to daily.")
            date_offset = pd.DateOffset(days=1)
        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=date_offset)
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": mean_forecast, "Lower_CI": lower_bound, "Upper_CI": upper_bound})
        
        st.info("Forecast pipeline finished successfully.")
        return forecast_df, evaluation_metrics, history_data, scaler_obj

    except Exception as e:
        logging.error(f"An error occurred in the forecast pipeline: {e}")
        st.error(f"An error occurred in the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc())
        return None, None, None, None

# --- Streamlit Callback for Training Progress ---
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}; progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        loss = logs.get(\"loss\", \"N/A\"); val_loss = logs.get(\"val_loss\", \"N/A\")
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        self.status_text.text(f"Epoch {epoch+1}/{self.total_epochs} | Loss: {loss_str} | Val Loss: {val_loss_str}")
    def close(self):
        self.progress_bar.empty(); self.status_text.empty()

# --- Initialize Session State --- 
def initialize_session_state():
    defaults = {
        "cleaned_data": None, "forecast_results": None, "evaluation_metrics": None, 
        "training_history": None, "ai_report": None, "scaler_object": None, 
        "forecast_plot_fig": None, "uploaded_data_filename": None,
        "active_tab": 0, "report_language": "English", "chat_history": [], 
        "chat_active": False, "model_sequence_length": STANDARD_MODEL_SEQUENCE_LENGTH, 
        "run_forecast_triggered": False, "about_us_expanded": False,
        "persistent_user_id": None, "user_profile": None, 
        "token": None, "auth_status": "logged_out", "user_email": None, 
        "user_name": None, "user_picture": None, "admin_authenticated": False, 
        "session_visit_logged": False, "user_agent": "Unknown",
        "custom_model_seq_len": None, "last_model_choice": None, 
        "user_agent_capture_component": None, "session_usage_count": 0 # Fallback usage counter
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Get User Profile (Ensure it runs after Firebase init attempt) ---
if fbase_init_success and st.session_state.persistent_user_id is None:
    st.session_state.persistent_user_id = get_persistent_user_id()
    st.session_state.user_profile, _ = get_or_create_user_profile(st.session_state.persistent_user_id)
elif not fbase_init_success and st.session_state.persistent_user_id is None:
    st.session_state.persistent_user_id = get_persistent_user_id() # Get ID even if firebase failed

# --- Sidebar --- 
with st.sidebar:
    st.title("DeepHydro AI")
    if fbase_init_success: log_visitor_activity("Sidebar", "view")
    
    st.header("1. Upload Data")
    uploaded_data_file = st.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")
    
    st.header("2. Model & Forecast")
    model_choice = st.selectbox("Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select")
    if fbase_init_success and st.session_state.get(\"last_model_choice\") != model_choice:
         log_visitor_activity("Sidebar", "select_model", feature_used=model_choice)
         st.session_state.last_model_choice = model_choice

    # Input Parameters
    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False
    current_sequence_length = st.session_state.get("model_sequence_length", STANDARD_MODEL_SEQUENCE_LENGTH)
    sequence_length_train_sidebar = current_sequence_length
    epochs_train_sidebar = 50

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload .h5 model", type="h5", key="custom_h5_uploader")
        st.info(f"Custom model selected. Sequence length will be inferred upon running.")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_custom_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="custom_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="custom_scaler_max_in")
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using standard model (Seq Len: {STANDARD_MODEL_SEQUENCE_LENGTH})")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_std_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values standard model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="std_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="std_scaler_max_in")
    elif model_choice == "Train New Model":
        try:
            sequence_length_train_sidebar = st.number_input("LSTM Sequence Length", min_value=10, max_value=365, value=current_sequence_length, step=10, key="seq_len_train_in")
        except Exception as e:
            st.warning(f"Using default sequence length {current_sequence_length} due to input error: {e}")
            sequence_length_train_sidebar = current_sequence_length
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in")

    mc_iterations_sidebar = st.number_input("MC Dropout Iterations (C.I.)", min_value=20, max_value=500, value=100, step=10, key="mc_iter_in")
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in")

    # Run Forecast Button
    run_forecast_button = st.button("Run Forecast", key="run_forecast_main_btn", use_container_width=True)
    if run_forecast_button:
        access_granted, message = check_feature_access()
        if access_granted:
            st.session_state.run_forecast_triggered = True
            if st.session_state.cleaned_data is not None:
                if model_choice == "Upload Custom .h5 Model" and custom_model_file_obj_sidebar is None:
                    st.error("Please upload a custom .h5 model file.")
                    st.session_state.run_forecast_triggered = False
                else:
                    log_visitor_activity("Sidebar", "run_forecast", feature_used=\"Forecast\")
                    with st.spinner(f"Running forecast ({model_choice})..."):
                        forecast_df, metrics, history, scaler_obj = run_forecast_pipeline(
                            st.session_state.cleaned_data, model_choice, forecast_horizon_sidebar, 
                            custom_model_file_obj_sidebar, sequence_length_train_sidebar, epochs_train_sidebar, 
                            mc_iterations_sidebar, use_custom_scaler_sidebar, custom_scaler_min_sidebar, custom_scaler_max_sidebar
                        )
                    st.session_state.forecast_results = forecast_df
                    st.session_state.evaluation_metrics = metrics
                    st.session_state.training_history = history
                    st.session_state.scaler_object = scaler_obj
                    if forecast_df is not None and metrics is not None:
                        st.session_state.forecast_plot_fig = create_forecast_plot(st.session_state.cleaned_data, forecast_df)
                        st.success("Forecast complete! Results updated.")
                        st.session_state.ai_report = None; st.session_state.chat_history = []; st.session_state.chat_active = False
                        st.session_state.active_tab = 1
                        st.rerun()
                    else:
                        st.error("Forecast pipeline failed. Check messages above.")
                        st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                        st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
            else:
                st.error("Please upload data first.")
                st.session_state.run_forecast_triggered = False
        else:
            st.warning(message)
            show_google_login()
            log_visitor_activity("Sidebar", "run_forecast_denied", feature_used=\"Forecast\")

    st.header("3. AI Analysis")
    st.session_state.report_language = st.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_init_success)
    
    # Generate AI Report Button
    generate_report_button = st.button("Generate AI Report", key="show_report_btn", disabled=not gemini_init_success, use_container_width=True)
    if generate_report_button:
        access_granted, message = check_feature_access()
        if access_granted:
            if not gemini_init_success: st.error("AI Report disabled. Gemini not configured.")
            elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
                log_visitor_activity("Sidebar", "generate_report", feature_used=\"AI Report\")
                with st.spinner(f"Generating AI report ({st.session_state.report_language})..."):
                    st.session_state.ai_report = generate_gemini_report(
                        st.session_state.cleaned_data, st.session_state.forecast_results,
                        st.session_state.evaluation_metrics, st.session_state.report_language
                    )
                if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:") and not st.session_state.ai_report.startswith("AI report generation disabled"):
                    st.success("AI report generated.")
                    st.session_state.active_tab = 3
                    st.rerun()
                else: 
                    st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
            else: 
                st.error("Data, forecast, and metrics needed. Run forecast first.")
        else:
            st.warning(message)
            show_google_login()
            log_visitor_activity("Sidebar", "generate_report_denied", feature_used=\"AI Report\")

    # Download PDF Button
    if st.button("Download Report (PDF)", key="download_report_btn", use_container_width=True):
        log_visitor_activity("Sidebar", "download_pdf")
        if (st.session_state.forecast_results is not None and 
            st.session_state.evaluation_metrics is not None and 
            st.session_state.ai_report is not None and 
            st.session_state.forecast_plot_fig is not None):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf = FPDF(); pdf.add_page()
                    font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    report_font = "Arial"
                    try: 
                        if os.path.exists(font_path_dejavu):
                            pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True); report_font = "DejaVu"
                        else: logging.warning(f"DejaVu font not found at {font_path_dejavu}, using Arial.")
                    except RuntimeError as font_err: logging.warning(f"Failed to add DejaVu font ({font_err}), using Arial.")
                    pdf.set_font(report_font, size=12); pdf.cell(0, 10, txt="DeepHydro AI Forecasting Report", new_x="LMARGIN", new_y="NEXT", align="C"); pdf.ln(5)
                    plot_filename = "forecast_plot.png"; img_embedded = False
                    try:
                        if st.session_state.forecast_plot_fig:
                            st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                            if os.path.exists(plot_filename):
                                pdf.image(plot_filename, x=pdf.get_x(), y=pdf.get_y(), w=190); pdf.ln(125); img_embedded = True
                            else: logging.warning("Plot image file was not created.")
                        else: logging.warning("Forecast plot figure not found in session state.")
                    except Exception as img_err: logging.warning(f"Could not embed plot image: {img_err}.")
                    finally: 
                        if os.path.exists(plot_filename): os.remove(plot_filename)
                    if not img_embedded: pdf.ln(10)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Model Evaluation Metrics", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10); metrics_dict = st.session_state.evaluation_metrics or {}
                    for key, value in metrics_dict.items():
                        val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else str(value)
                        pdf.cell(0, 8, txt=f"{key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Forecast Data (First 10)", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=8); col_widths = [35, 35, 35, 35]
                    pdf.cell(col_widths[0], 7, txt="Date", border=1); pdf.cell(col_widths[1], 7, txt="Forecast", border=1); pdf.cell(col_widths[2], 7, txt="Lower CI", border=1); pdf.cell(col_widths[3], 7, txt="Upper CI", border=1, new_x="LMARGIN", new_y="NEXT")
                    forecast_table_df = st.session_state.forecast_results.head(10) if st.session_state.forecast_results is not None else pd.DataFrame()
                    for _, row in forecast_table_df.iterrows():
                        pdf.cell(col_widths[0], 6, txt=str(row["Date"].date()), border=1); pdf.cell(col_widths[1], 6, txt=f"{row[\"Forecast\"]:.2f}", border=1); pdf.cell(col_widths[2], 6, txt=f"{row[\"Lower_CI\"]:.2f}", border=1); pdf.cell(col_widths[3], 6, txt=f"{row[\"Upper_CI\"]:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt=f"AI Report ({st.session_state.report_language})", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10); ai_report_text = st.session_state.ai_report or "AI Report not available."
                    try: pdf.multi_cell(0, 5, txt=ai_report_text)
                    except UnicodeEncodeError: pdf.multi_cell(0, 5, txt=ai_report_text.encode(\"latin-1\", \"replace\").decode(\"latin-1\"))
                    pdf.ln(5)
                    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                    st.download_button(label="Download PDF Now", data=pdf_output_bytes, file_name="deephydro_forecast_report.pdf", mime="application/octet-stream", key="pdf_download_final_btn", use_container_width=True)
                    st.success("PDF ready. Click download button above.")
                    log_visitor_activity("Sidebar", "download_pdf_success")
                except Exception as pdf_err:
                    st.error(f"Failed to generate PDF: {pdf_err}")
                    log_visitor_activity("Sidebar", "download_pdf_failure", details={"error": str(pdf_err)})
        else: st.error("Required data missing. Run forecast and generate AI report first.")

    st.header("4. AI Assistant")
    chat_button_label = "Deactivate Chat" if st.session_state.chat_active else "Activate Chat"
    activate_chat_button = st.button(chat_button_label, key="chat_ai_btn", disabled=not gemini_init_success, use_container_width=True)
    if activate_chat_button:
        if st.session_state.chat_active:
            st.session_state.chat_active = False; st.session_state.chat_history = []
            log_visitor_activity("Sidebar", "deactivate_chat"); st.rerun()
        else:
            access_granted, message = check_feature_access()
            if access_granted:
                if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
                    st.session_state.chat_active = True; st.session_state.active_tab = 4
                    log_visitor_activity("Sidebar", "activate_chat", feature_used=\"AI Chat\"); st.rerun()
                else: st.error("Cannot activate chat. Run a successful forecast first.")
            else:
                st.warning(message); show_google_login()
                log_visitor_activity("Sidebar", "activate_chat_denied", feature_used=\"AI Chat\")

    # About Us
    st.markdown(\"<div class=\"about-us-header\">👥 About Us</div>\", unsafe_allow_html=True)
    st.markdown(\"<div class=\"about-us-content\">\", unsafe_allow_html=True)
    st.markdown("Specializing in groundwater forecasting using AI.")
    st.markdown("**Contact:** [deephydro@example.com](mailto:deephydro@example.com)")
    st.markdown("© 2025 DeepHydro AI Team")
    st.markdown(\"</div>\", unsafe_allow_html=True)
    
    # Admin Analytics Access
    st.header("5. Admin")
    if st.button("Analytics Dashboard", key="admin_analytics_btn", use_container_width=True):
        log_visitor_activity("Sidebar", "access_admin")
        st.session_state.active_tab = 5; st.rerun()
        
    # Authentication Section
    st.header("6. Authentication")
    show_google_login()

# --- Main Application Area --- 
st.title("DeepHydro AI Forecasting")
log_visitor_activity("Main Page", "view")

# App Introduction
st.markdown(\"<div class=\"app-intro\">\", unsafe_allow_html=True)
st.markdown("""
### Welcome to DeepHydro AI Forecasting
Advanced groundwater forecasting platform using deep learning.
**Features:** LSTM forecasting, MC Dropout uncertainty, AI interpretation, Interactive visualization.
Upload your data using the sidebar to begin.
""")
st.markdown(\"</div>\", unsafe_allow_html=True)

# Handle data upload and cleaning
if uploaded_data_file is not None:
    if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
        st.session_state.uploaded_data_filename = uploaded_data_file.name
        with st.spinner("Loading and cleaning data..."):
            cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
        if cleaned_df_result is not None:
            st.session_state.cleaned_data = cleaned_df_result
            st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
            st.session_state.training_history = None; st.session_state.ai_report = None
            st.session_state.chat_history = []; st.session_state.chat_active = False
            st.session_state.scaler_object = None; st.session_state.forecast_plot_fig = None
            st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
            st.session_state.run_forecast_triggered = False; st.session_state.active_tab = 0
            log_visitor_activity("Data Upload", "upload_success", details={"filename": uploaded_data_file.name})
            st.rerun()
        else:
            st.session_state.cleaned_data = None
            st.error("Data loading failed. Please check the file format and content.")
            log_visitor_activity("Data Upload", "upload_failure", details={"filename": uploaded_data_file.name})
            st.session_state.uploaded_data_filename = None 

# Define tabs
tab_titles = ["Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# --- Tab Content --- 

# Data Preview Tab (Tab 0)
with tabs[0]:
    log_visitor_activity("Tab: Data Preview", "view")
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        col1, col2 = st.columns(2)
        with col1: 
            min_date = st.session_state.cleaned_data[\"Date\"].min(); max_date = st.session_state.cleaned_data[\"Date\"].max()
            st.metric("Time Range", f"{min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}")
        with col2: st.metric("Data Points", len(st.session_state.cleaned_data))
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], mode="lines", name="Level"))
        fig_data.update_layout(title="Historical Groundwater Levels", xaxis_title="Date", yaxis_title="Level", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=400)
        st.plotly_chart(fig_data, use_container_width=True)
    else: st.info("⬆️ Upload XLSX data using the sidebar.")

# Forecast Results Tab (Tab 1)
with tabs[1]:
    log_visitor_activity("Tab: Forecast Results", "view")
    st.header("Forecast Results")
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        if st.session_state.forecast_plot_fig is not None:
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else: 
            if st.session_state.cleaned_data is not None:
                 st.warning("Forecast plot figure missing, attempting to recreate...")
                 try:
                     recreated_fig = create_forecast_plot(st.session_state.cleaned_data, st.session_state.forecast_results)
                     st.plotly_chart(recreated_fig, use_container_width=True)
                 except Exception as plot_err: st.error(f"Could not recreate plot: {plot_err}")
            else: st.warning("Forecast plot unavailable (figure missing and no historical data).")
        st.subheader("Forecast Data Table")
        st.dataframe(st.session_state.forecast_results, use_container_width=True)
    elif st.session_state.run_forecast_triggered: st.warning("Forecast run attempted, but no results available.")
    else: st.info("⬅️ Run a forecast using the sidebar to see results here.")

# Model Evaluation Tab (Tab 2)
with tabs[2]:
    log_visitor_activity("Tab: Model Evaluation", "view")
    st.header("Model Evaluation")
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (Validation/Pseudo-Validation)")
        col1, col2, col3 = st.columns(3)
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan)
        mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan)
        mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        col1.metric("RMSE", f"{rmse_val:.4f}" if pd.notna(rmse_val) else "N/A")
        col2.metric("MAE", f"{mae_val:.4f}" if pd.notna(mae_val) else "N/A")
        col3.metric("MAPE", f"{mape_val:.2f}%" if pd.notna(mape_val) and np.isfinite(mape_val) else ("N/A" if pd.isna(mape_val) else "Inf"))
        st.subheader("Training Loss (if applicable)")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: st.info("No training history available (used pre-trained model or training failed/skipped).")
    elif st.session_state.run_forecast_triggered: st.warning("Forecast run attempted, but no evaluation metrics available.")
    else: st.info("⬅️ Run a forecast using the sidebar to see evaluation metrics.")

# AI Report Tab (Tab 3)
with tabs[3]:
    log_visitor_activity("Tab: AI Report", "view")
    st.header("AI-Generated Scientific Report")
    if not gemini_init_success: st.warning("AI features disabled. Gemini not configured.")
    elif st.session_state.ai_report: 
        st.markdown(f\"<div class=\"chat-message ai-message\">{st.session_state.ai_report}<span class=\"copy-tooltip\">Copied!</span></div>\", unsafe_allow_html=True)
    else: st.info("⬅️ Click \"Generate AI Report\" in the sidebar after running a forecast.")

# AI Chatbot Tab (Tab 4)
with tabs[4]:
    log_visitor_activity("Tab: AI Chatbot", "view")
    st.header("AI Chatbot Assistant")
    if not gemini_init_success: st.warning("AI features disabled. Gemini not configured.")
    elif st.session_state.chat_active:
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("Chat activated. Ask questions about the current forecast results below.")
            chat_container = st.container(height=400)
            with chat_container:
                for sender, message in st.session_state.chat_history:
                    msg_class = "user-message" if sender == "User" else "ai-message"
                    escaped_message = html.escape(str(message))
                    st.markdown(f\"<div class=\"chat-message {msg_class}\">{escaped_message}<span class=\"copy-tooltip\">Copied!</span></div>\", unsafe_allow_html=True)
            user_input = st.chat_input("Ask the AI assistant about the results...")
            if user_input:
                log_visitor_activity("Chat", "send_message")
                st.session_state.chat_history.append(("User", user_input))
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history,
                        st.session_state.cleaned_data, st.session_state.forecast_results, 
                        st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                st.session_state.chat_history.append(("AI", ai_response))
                st.rerun()
        else:
            st.warning("Context (data/forecast/metrics) is missing. Please run a successful forecast first.")
            st.session_state.chat_active = False
    else:
        st.info("⬅️ Click \"Activate Chat\" in the sidebar after running a forecast." if gemini_init_success else "AI Chat disabled.")

# Admin Analytics Tab (Tab 5)
with tabs[5]:
    log_visitor_activity("Tab: Admin Analytics", "view")
    render_admin_analytics()

# Add JavaScript at the end
add_javascript_functionality()

# --- Final Checks ---
if config_load_error:
    st.error(f"Configuration Error: {config_load_error}")

logging.info("Streamlit script execution finished.")

