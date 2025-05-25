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
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component
import logging # Import logging

# --- Setup Logging --- 
# Configure logging to not display messages in Streamlit frontend by default
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Prevent streamlit from showing logs directly unless explicitly configured
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Add a handler that doesn't output to streamlit console (e.g., could write to a file or stderr)
# For simplicity here, we'll just log INFO level and above, but not show it in UI
# If debugging is needed, one might add a StreamlitLogHandler or file handler.

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 3
STANDARD_MODEL_PATH = "standard_model.h5"
STANDARD_MODEL_SEQUENCE_LENGTH = 60 # Default

# --- Global Flags & Configuration --- 
APP_CONFIG = {}
firebase_initialized = False
google_oauth_configured = False
gemini_configured = False
admin_password_configured = False
config_load_error = None # Store potential config load error message

# --- Load Configuration --- 
def load_configuration():
    global APP_CONFIG, google_oauth_configured, admin_password_configured, config_load_error
    config_json_str = os.getenv("APP_CONFIG_JSON")
    if not config_json_str:
        config_load_error = "Environment variable APP_CONFIG_JSON not found. Application configuration is missing."
        logging.error(config_load_error)
        APP_CONFIG = {}
        return

    try:
        APP_CONFIG = json.loads(config_json_str)
        logging.info("Configuration loaded successfully from APP_CONFIG_JSON.")
        # Validate essential config sections internally
        if "firebase_service_account" not in APP_CONFIG:
            logging.warning("Firebase configuration ('firebase_service_account') missing in APP_CONFIG_JSON.")
        if "google_oauth" not in APP_CONFIG or not all(k in APP_CONFIG["google_oauth"] for k in ["client_id", "client_secret", "redirect_uri"]):
            logging.warning("Google OAuth configuration ('google_oauth') incomplete or missing in APP_CONFIG_JSON.")
            google_oauth_configured = False
        else:
            google_oauth_configured = True
        if "google_api_key" not in APP_CONFIG:
            logging.warning("Google API Key ('google_api_key') missing in APP_CONFIG_JSON.")
        if "admin_password" not in APP_CONFIG:
             logging.warning("Admin Password ('admin_password') missing in APP_CONFIG_JSON.")
             admin_password_configured = False
        else:
            admin_password_configured = True
            
    except json.JSONDecodeError as e:
        config_load_error = f"Error decoding APP_CONFIG_JSON: {e}. Check the JSON format."
        logging.error(config_load_error)
        APP_CONFIG = {} # Reset on error
        google_oauth_configured = False
        admin_password_configured = False

# --- Firebase Initialization --- 
def initialize_firebase():
    global firebase_initialized
    if firebase_admin._apps: return True
    if "firebase_service_account" not in APP_CONFIG:
        logging.warning("Firebase Service Account details not found in config. Analytics disabled.")
        firebase_initialized = False
        return False
        
    try:
        cred_dict = APP_CONFIG["firebase_service_account"]
        if not all(k in cred_dict for k in ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]):
             logging.error("Firebase Service Account JSON in configuration is incomplete.")
             firebase_initialized = False
             return False
             
        cred = credentials.Certificate(cred_dict)
        firebase_url = APP_CONFIG.get("firebase_database_url")
        if not firebase_url or firebase_url == "OPTIONAL_YOUR_FIREBASE_DB_URL":
            project_id = cred_dict.get("project_id")
            if project_id:
                firebase_url = f"https://{project_id}-default-rtdb.firebaseio.com/"
                logging.info(f"Firebase Database URL not set, using default: {firebase_url}")
            else:
                logging.error("Cannot determine Firebase DB URL: 'firebase_database_url' missing and 'project_id' missing.")
                firebase_initialized = False
                return False
                
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_url})
        logging.info("Firebase initialized successfully.") 
        firebase_initialized = True
        return True
        
    except ValueError as e:
        logging.error(f"Error initializing Firebase with provided credentials: {e}")
        firebase_initialized = False
        return False
    except Exception as e:
        logging.warning(f"Firebase initialization error: {e}. Analytics may be disabled.")
        firebase_initialized = False
        return False

# --- Gemini API Configuration --- 
def configure_gemini():
    global gemini_configured, gemini_model_report, gemini_model_chat
    GEMINI_API_KEY = APP_CONFIG.get("google_api_key")
    gemini_model_report = None
    gemini_model_chat = None
    gemini_configured = False

    if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSy..." and GEMINI_API_KEY != "Gemini_api_key":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
            gemini_model_report = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
            gemini_model_chat = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
            gemini_configured = True
            logging.info("Gemini API configured successfully.")
        except Exception as e:
            logging.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
            gemini_configured = False
    else:
        logging.warning("Gemini API Key ('google_api_key') not found or is placeholder. AI features disabled.")
        gemini_configured = False

# --- Load Standard Model --- 
def load_standard_model():
    global STANDARD_MODEL_SEQUENCE_LENGTH
    if os.path.exists(STANDARD_MODEL_PATH):
        try:
            _model = load_model(STANDARD_MODEL_PATH, compile=False)
            STANDARD_MODEL_SEQUENCE_LENGTH = _model.input_shape[1]
            st.session_state.standard_model = _model
            st.session_state.standard_model_seq_len = STANDARD_MODEL_SEQUENCE_LENGTH
            logging.info(f"Standard model loaded. Sequence length: {STANDARD_MODEL_SEQUENCE_LENGTH}")
        except Exception as e:
            logging.warning(f"Could not load standard model from {STANDARD_MODEL_PATH}: {e}. Using default seq len {STANDARD_MODEL_SEQUENCE_LENGTH}.")
            if firebase_initialized: log_visitor_activity("Model Handling", action="load_standard_model_fail", details={"path": STANDARD_MODEL_PATH, "error": str(e)})
    else:
        logging.warning(f"Standard model file not found: {STANDARD_MODEL_PATH}. Standard model option disabled.")

# --- User Identification & Tracking --- 
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
    if not firebase_initialized or not email: return
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
        # Avoid showing error directly to user unless critical
        # st.error(f"Firebase error updating profile after Google login for {email}: {e}") 
        log_visitor_activity("Authentication", action="google_login_firebase_update_fail", details={"email": email, "error": str(e)})

def get_or_create_user_profile(user_id):
    if not firebase_initialized: return None, False
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
    if not firebase_initialized: return False
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

CLIENT_ID = APP_CONFIG.get("google_oauth", {}).get("client_id")
CLIENT_SECRET = APP_CONFIG.get("google_oauth", {}).get("client_secret")
REDIRECT_URI = APP_CONFIG.get("google_oauth", {}).get("redirect_uri")
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

def get_user_info_from_google(token):
    if not token or "access_token" not in token: return None
    try:
        response = requests.get("https://www.googleapis.com/oauth2/v1/userinfo", headers={"Authorization": f"Bearer {token['access_token']}"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching user info from Google: {e}")
        # Don't show error directly to user unless necessary
        # st.error(f"Error fetching user info from Google: {e}") 
        return None
    except Exception as e:
         logging.error(f"Unexpected error fetching user info: {e}")
         return None

def show_google_login():
    if not google_oauth_configured:
        st.sidebar.warning("Google Sign-In unavailable (not configured).")
        if firebase_initialized: log_visitor_activity("Authentication", action="google_login_fail_config_missing")
        return

    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    
    if "token" not in st.session_state:
        usage_count = st.session_state.user_profile.get("feature_usage_count", 0) if st.session_state.user_profile else 0
        if usage_count >= ADVANCED_FEATURE_LIMIT:
            st.sidebar.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached.")
            st.sidebar.info("Please log in with Google to continue.")
            result = oauth2.authorize_button(
                name="Login with Google",
                # icon="https://www.google.com/favicon.ico", # Removed icon
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
                    # Use st.toast for less intrusive message
                    st.toast(f"Logged in as {st.session_state.user_name}") 
                    time.sleep(1.5) # Keep delay for toast visibility
                    st.rerun()
                else:
                    # Use st.toast for error as well
                    st.toast("Login failed: Could not retrieve user info.", icon="🚨") 
                    if "token" in st.session_state: del st.session_state.token 
                    if firebase_initialized: log_visitor_activity("Authentication", action="google_login_fail_userinfo")
    else:
        if not st.session_state.get("user_email"):
             user_info = get_user_info_from_google(st.session_state.token)
             if user_info and user_info.get("email"):
                 st.session_state.auth_status = "authenticated"
                 st.session_state.user_email = user_info.get("email")
                 st.session_state.user_name = user_info.get("name")
                 st.session_state.persistent_user_id = user_info.get("email")
             else:
                 st.toast("Could not verify login. Please log in again.", icon="🚨")
                 if "token" in st.session_state: del st.session_state.token
                 st.session_state.auth_status = "anonymous"
                 st.session_state.user_email = None
                 st.session_state.user_name = None
                 if firebase_initialized: log_visitor_activity("Authentication", action="google_reauth_fail_userinfo")
                 st.rerun()
                 
        if st.session_state.get("user_email"):
            st.sidebar.success(f"Logged in: {st.session_state.user_name}")
            if st.sidebar.button("Logout", key="google_logout"):
                if firebase_initialized: log_visitor_activity("Authentication", action="google_logout", details={"email": st.session_state.user_email})
                if "token" in st.session_state: del st.session_state.token
                st.session_state.auth_status = "anonymous"
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.persistent_user_id = None
                st.toast("Logged out.")
                st.rerun()

# --- Visitor Analytics Functions --- 
def get_session_id():
    if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None, details=None):
    if not firebase_initialized: return
    try:
        user_id = get_persistent_user_id()
        profile, _ = get_or_create_user_profile(user_id)
        should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"]
        access_granted, _ = check_feature_access()
        action_status = "success"
        if should_increment:
            if access_granted: increment_feature_usage(user_id)
            else: action_status = "denied_limit_reached"
        full_action_name = f"{action}_{action_status}"
        safe_user_id_key = user_id.replace(".", "_dot_").replace("@", "_at_") if "@" in user_id else user_id
        ref = db.reference("visitors_log")
        log_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        session_id = get_session_id()
        user_agent = st.session_state.get("user_agent", "Unknown")
        ip_address = get_client_ip()
        log_data = {"timestamp": timestamp, "persistent_user_id": user_id, "is_authenticated": st.session_state.get("auth_status") == "authenticated", "visit_count": profile.get("visit_count", 1) if profile else 1, "ip_address": ip_address, "page": page_name, "action": full_action_name, "feature_used": feature_used, "session_id": session_id, "user_agent": user_agent}
        if details and isinstance(details, dict):
            try: json.dumps(details); log_data["details"] = details
            except TypeError: log_data["details"] = {"error": "Details not JSON serializable"}
        ref.child(log_id).set(log_data)
    except Exception as e: logging.error(f"Error logging visitor activity: {e}")

def fetch_visitor_logs():
    if not firebase_initialized: return pd.DataFrame()
    try:
        ref = db.reference("visitors_log")
        visitors_data = ref.get()
        if not visitors_data: return pd.DataFrame()
        visitors_list = []
        for log_id, data in visitors_data.items():
            if isinstance(data, dict):
                data["log_id"] = log_id
                if "details" in data and isinstance(data["details"], dict):
                    for k, v in data["details"].items():
                        safe_key = f"detail_{str(k).replace('.', '_').replace('$', '_')}"
                        data[safe_key] = v
                    del data["details"]
                visitors_list.append(data)
        df = pd.DataFrame(visitors_list)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", ascending=False, na_position="last")
        return df
    except Exception as e:
        logging.error(f"Error fetching visitor logs: {e}")
        st.error(f"Error fetching visitor logs: {e}") # Show error in admin panel
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
    if visitor_df.empty: return []
    figures = []
    df = visitor_df.copy()
    try:
        if "timestamp" in df.columns and df["timestamp"].dtype != "<M8[ns]":
             df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty: return []
        df["date"] = df["timestamp"].dt.date
        if "persistent_user_id" in df.columns:
            daily_visitors = df.groupby("date")["persistent_user_id"].nunique().reset_index(name="unique_users")
            daily_visitors["date"] = pd.to_datetime(daily_visitors["date"])
            fig1 = px.line(daily_visitors, x="date", y="unique_users", title="Daily Unique Visitors")
            figures.append(fig1)
        if "action" in df.columns:
            action_counts = df[~df["action"].str.contains("page_view", case=False, na=False)]["action"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            fig2 = px.bar(action_counts.head(20), x="action", y="count", title="Top 20 Activity Counts")
            figures.append(fig2)
        if "persistent_user_id" in df.columns and "is_authenticated" in df.columns:
            latest_status = df.sort_values("timestamp").groupby("persistent_user_id")["is_authenticated"].last().reset_index()
            auth_counts = latest_status["is_authenticated"].value_counts().reset_index()
            auth_counts.columns = ["is_authenticated", "count"]
            auth_counts["status"] = auth_counts["is_authenticated"].map({True: "Authenticated", False: "Anonymous"})
            fig3 = px.pie(auth_counts, values="count", names="status", title="User Auth Status (Latest)")
            figures.append(fig3)
        try:
            df["hour"] = df["timestamp"].dt.hour
            df["weekday"] = df["timestamp"].dt.day_name()
            hourly_activity = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
            all_hours = list(range(24)); all_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = pd.MultiIndex.from_product([all_weekdays, all_hours], names=["weekday", "hour"]).to_frame(index=False)
            heatmap_data = pd.merge(heatmap_data, hourly_activity, on=["weekday", "hour"], how="left").fillna(0)
            heatmap_pivot = heatmap_data.pivot(index="weekday", columns="hour", values="count").reindex(all_weekdays)
            fig4 = px.imshow(heatmap_pivot, labels=dict(x="Hour", y="Day", color="Count"), x=[str(h) for h in all_hours], y=all_weekdays, title="User Activity Heatmap", color_continuous_scale=px.colors.sequential.Viridis)
            fig4.update_xaxes(side="bottom")
            figures.append(fig4)
        except Exception as e_heatmap: logging.warning(f"Could not generate activity heatmap: {e_heatmap}")
    except Exception as e_charts: logging.error(f"Error creating visitor charts: {e_charts}")
    return figures

# --- PDF Generation --- 
class PDF(FPDF):
    def header(self): self.set_font("Arial", "B", 12); self.cell(0, 10, "DeepHydro AI Forecasting Report", 0, 1, "C"); self.ln(5)
    def chapter_title(self, title): self.set_font("Arial", "B", 12); self.cell(0, 10, title, 0, 1, "L"); self.ln(4)
    def chapter_body(self, body): self.set_font("Arial", "", 10); self.multi_cell(0, 5, body); self.ln()
    def add_plot(self, plot_bytes, title):
        self.chapter_title(title)
        try:
            temp_img_path = f"temp_plot_{uuid.uuid4()}.png"
            with open(temp_img_path, "wb") as f: f.write(plot_bytes)
            self.image(temp_img_path, x=10, w=self.w - 20)
            os.remove(temp_img_path)
            self.ln(5)
        except Exception as e: self.chapter_body(f"[Error adding plot: {e}]")
    def footer(self): self.set_y(-15); self.set_font("Arial", "I", 8); self.cell(0, 10, f"Page {self.page_no()}/{self.alias_nb_pages()}", 0, 0, "C")

def create_pdf_report(forecast_fig, loss_fig, metrics, ai_report_text):
    pdf = PDF(); pdf.alias_nb_pages(); pdf.add_page()
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
    except Exception as e: logging.error(f"Error generating PDF bytes: {e}"); return None

# --- Custom CSS --- 
def apply_custom_css():
    # Keep existing CSS but ensure it doesn't hide sidebar elements
    st.markdown("""
    <style>
    /* General Styling */
    .stApp { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px 15px;
        transition: background-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] { background-color: #e6f7ff; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #f0f0f0; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #ffffff; padding: 1rem; }
    /* Ensure sidebar title is visible */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
         color: #1890ff; /* Example color */
         display: block !important; /* Force display */
    }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%; border-radius: 5px; background-color: #1890ff;
        color: white; transition: background-color 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #40a9ff; }
    [data-testid="stSidebar"] .stDownloadButton>button {
        width: 100%; border-radius: 5px; background-color: #52c41a;
        color: white; transition: background-color 0.3s ease;
    }
     [data-testid="stSidebar"] .stDownloadButton>button:hover { background-color: #73d13d; }
    
    /* Chat Message Styling */
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

# --- JavaScript --- 
def add_javascript_functionality():
     # Keep existing JS
     st.markdown("""
    <script>
    // Function to copy text to clipboard
    function copyToClipboard(text) {
        const textarea = document.createElement('textarea'); textarea.value = text;
        document.body.appendChild(textarea); textarea.select();
        document.execCommand('copy'); document.body.removeChild(textarea);
    }
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const chatMessages = document.querySelectorAll('.chat-message');
            chatMessages.forEach(function(message) {
                if (!message.querySelector('.copy-tooltip')) {
                    const tooltip = document.createElement('span');
                    tooltip.className = 'copy-tooltip'; tooltip.textContent = 'Copied!';
                    message.appendChild(tooltip);
                }
                let longPressTimer;
                message.addEventListener('touchstart', function(e) {
                    longPressTimer = setTimeout(() => {
                        const textToCopy = this.innerText.replace('Copied!', '').trim();
                        copyToClipboard(textToCopy);
                        const tooltip = this.querySelector('.copy-tooltip');
                        if (tooltip) { tooltip.style.display = 'block'; setTimeout(() => { tooltip.style.display = 'none'; }, 1500); }
                    }, 500);
                });
                message.addEventListener('touchend', function() { clearTimeout(longPressTimer); });
                message.addEventListener('touchmove', function() { clearTimeout(longPressTimer); });
                 message.addEventListener('click', function(e) {
                     const textToCopy = this.innerText.replace('Copied!', '').trim();
                     copyToClipboard(textToCopy);
                     const tooltip = this.querySelector('.copy-tooltip');
                     if (tooltip) { tooltip.style.display = 'block'; setTimeout(() => { tooltip.style.display = 'none'; }, 1500); }
                 });
            });
            const aboutUsHeader = document.querySelector('.about-us-header');
            const aboutUsContent = document.querySelector('.about-us-content');
            if (aboutUsHeader && aboutUsContent) {
                if (!aboutUsContent.classList.contains('initialized')) {
                     aboutUsContent.style.display = 'none'; aboutUsContent.classList.add('initialized');
                }
                aboutUsHeader.addEventListener('click', function() {
                    aboutUsContent.style.display = (aboutUsContent.style.display === 'none') ? 'block' : 'none';
                });
            }
        }, 1000);
    });
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration & Initialization --- 
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css()
add_javascript_functionality()

# --- Run Initial Setup --- 
load_configuration() # Load config first
initialize_firebase() # Initialize Firebase
configure_gemini() # Configure Gemini

# --- Initialize Session State --- 
def initialize_session_state():
    # Load standard model *before* setting defaults if it exists
    load_standard_model()
    
    defaults = {
        "df": None, "forecast_df": None, "metrics": None, "forecast_fig": None, 
        "loss_fig": None, "model_trained": False, "selected_model_type": "Standard", 
        "custom_model": None, "custom_model_seq_len": None, 
        "standard_model": st.session_state.get("standard_model"), 
        "standard_model_seq_len": st.session_state.get("standard_model_seq_len", STANDARD_MODEL_SEQUENCE_LENGTH),
        "ai_report": None, "chat_active": False, "messages": [], "chat_model": None,
        "admin_authenticated": False, "user_profile": None, 
        "auth_status": "anonymous", "user_email": None, "user_name": None,
        "token": None, "persistent_user_id": None,
        "user_agent": st.session_state.get("user_agent", "Unknown"), 
        "session_id": None, "session_visit_logged": False,
        "uploaded_file_name": None, "custom_model_file_name": None,
        "training_metrics": None, "trained_model_seq_len": None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session_state()

# --- Capture User Agent --- 
def capture_user_agent():
    if "user_agent" not in st.session_state:
        try:
            component_value = components.html("""<script>window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', key: 'user_agent_capture', value: navigator.userAgent}, '*');</script>""", height=0, key="user_agent_capture")
            if component_value: st.session_state.user_agent = component_value
            elif "user_agent_capture" in st.session_state and st.session_state.user_agent_capture: st.session_state.user_agent = st.session_state.user_agent_capture
            else: st.session_state.user_agent = "Unknown (Pending)"
        except Exception: st.session_state.user_agent = "Unknown (Failed)"
capture_user_agent()

# Initialize user profile (depends on Firebase init and session state init)
if "user_profile" not in st.session_state or st.session_state.user_profile is None:
    if firebase_initialized:
        user_id = get_persistent_user_id()
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
    else:
        st.session_state.user_profile = None

# --- Helper Functions (Data Loading, Model Building, Prediction, Plotting, AI, PDF) ---
# These functions (load_and_clean_data, create_sequences, load_keras_model_from_file, 
# build_lstm_model, predict_with_dropout_uncertainty, calculate_metrics, 
# create_forecast_plot, create_loss_plot, generate_ai_report, initialize_chat, 
# display_chat_history, handle_chat_input, create_pdf_report) are defined above 
# and remain largely unchanged internally, only logging/error handling was adjusted.

# --- Streamlit App UI --- 

# --- Sidebar --- 
# Ensure sidebar is defined at the top level
st.sidebar.title("DeepHydro AI")

# Display config load error prominently if it occurred
if config_load_error:
    st.sidebar.error(f"Configuration Error: {config_load_error}")

# --- Authentication Section (Sidebar) ---
st.sidebar.markdown("--- User Access ---")
show_google_login() 

# --- Options Section (Sidebar) ---
st.sidebar.markdown("--- Options ---")

uploaded_file = st.sidebar.file_uploader("Upload Excel Data (.xlsx)", type=["xlsx"], key="data_uploader")
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.get("uploaded_file_name"):
        st.session_state.df = load_and_clean_data(uploaded_file.getvalue())
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.forecast_df = None; st.session_state.metrics = None
        st.session_state.forecast_fig = None; st.session_state.ai_report = None
        st.session_state.chat_active = False; st.session_state.messages = []
        if firebase_initialized: log_visitor_activity("Sidebar", action="upload_data", details={"filename": uploaded_file.name})
        st.rerun()

# Model Selection
model_options = []
if st.session_state.get("standard_model"): model_options.append("Standard")
model_options.extend(["Train New", "Upload Custom (.h5)"])
default_model_index = 0
if st.session_state.selected_model_type in model_options:
    try: default_model_index = model_options.index(st.session_state.selected_model_type)
    except ValueError: default_model_index = 0 # Fallback if somehow selected type isn't an option

model_choice = st.sidebar.radio(
    "Select Model", 
    options=model_options,
    key="model_selector", 
    index=default_model_index,
    disabled=not model_options
)

if model_choice and model_choice != st.session_state.selected_model_type:
    if firebase_initialized: log_visitor_activity("Sidebar", action="select_model_type", details={"model_type": model_choice})
    st.session_state.selected_model_type = model_choice
    st.session_state.forecast_df = None; st.session_state.metrics = None
    st.session_state.forecast_fig = None; st.session_state.ai_report = None
    st.session_state.chat_active = False; st.session_state.messages = []
    st.rerun()

custom_model_file = None
if st.session_state.selected_model_type == "Upload Custom (.h5)":
    custom_model_file = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"], key="model_uploader")
    if custom_model_file is not None:
        if custom_model_file.name != st.session_state.get("custom_model_file_name"):
            loaded_model, seq_len = load_keras_model_from_file(custom_model_file, custom_model_file.name)
            if loaded_model and seq_len:
                st.session_state.custom_model = loaded_model
                st.session_state.custom_model_seq_len = seq_len
                st.session_state.custom_model_file_name = custom_model_file.name
                st.rerun()
            else:
                st.session_state.custom_model = None
                st.session_state.custom_model_seq_len = None
                st.session_state.custom_model_file_name = None

# Forecast Parameters
st.sidebar.markdown("--- Forecast Settings ---")
mc_iterations = st.sidebar.number_input("MC Dropout Iterations (C.I.)", min_value=10, max_value=500, value=100, step=10, key="mc_iter")
forecast_horizon = st.sidebar.number_input("Forecast Horizon (steps)", min_value=1, max_value=365, value=12, step=1, key="horizon")

# --- Feature Access Control Check ---
can_access_forecast, reason_forecast = check_feature_access()
can_access_report, reason_report = check_feature_access()
can_access_chat, reason_chat = check_feature_access()

# --- Action Buttons (Sidebar) --- 
st.sidebar.markdown("--- Actions ---")

# Determine forecast readiness
model_available = False
if st.session_state.selected_model_type == "Standard" and st.session_state.get("standard_model"):
    model_available = True
elif st.session_state.selected_model_type == "Upload Custom (.h5)" and st.session_state.get("custom_model"):
    model_available = True
elif st.session_state.selected_model_type == "Train New" and st.session_state.get("model_trained") and st.session_state.get("custom_model"):
    model_available = True
forecast_ready = st.session_state.df is not None and model_available and can_access_forecast
run_forecast_disabled_reason = ""
if st.session_state.df is None: run_forecast_disabled_reason = "Upload data first."
elif not model_available: run_forecast_disabled_reason = "Select/Train/Upload a valid model."
elif not can_access_forecast: run_forecast_disabled_reason = reason_forecast

if st.sidebar.button("Run Forecast", key="run_forecast_btn", disabled=not forecast_ready, help=run_forecast_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast", feature_used="Forecast", details={"model_type": st.session_state.selected_model_type, "horizon": forecast_horizon, "mc_iterations": mc_iterations})
    model_to_use, sequence_length, model_type_log = None, None, st.session_state.selected_model_type
    if st.session_state.selected_model_type == "Standard": model_to_use, sequence_length = st.session_state.standard_model, st.session_state.standard_model_seq_len
    elif st.session_state.selected_model_type == "Upload Custom (.h5)": model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len
    elif st.session_state.selected_model_type == "Train New": model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len; model_type_log = "Trained New"
    if model_to_use is None or sequence_length is None: st.error("Model not available."); st.stop()
    if len(st.session_state.df) <= sequence_length: st.error(f"Not enough data ({len(st.session_state.df)}) for sequence length ({sequence_length})."); st.stop()
    with st.spinner("Forecasting..."):
        try:
            df_forecast_input = st.session_state.df.copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_forecast_input["Level"].values.reshape(-1, 1))
            last_sequence_scaled = scaled_data[-sequence_length:]
            mean_preds, lower_bound, upper_bound = predict_with_dropout_uncertainty(model_to_use, last_sequence_scaled, forecast_horizon, mc_iterations, scaler, sequence_length)
            last_date = df_forecast_input["Date"].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            st.session_state.forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": mean_preds, "Lower_CI": lower_bound, "Upper_CI": upper_bound})
            st.session_state.metrics = st.session_state.get("training_metrics", {"Info": "Metrics generated during training"})
            st.session_state.forecast_fig = create_forecast_plot(df_forecast_input, st.session_state.forecast_df)
            st.toast("Forecast completed!", icon="✅")
            if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_success", feature_used="Forecast", details={"model_type": model_type_log, "horizon": forecast_horizon})
            st.rerun()
        except Exception as e: st.error(f"Error during forecasting: {e}"); logging.error(f"Forecast error: {e}");
        if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_fail", feature_used="Forecast", details={"model_type": model_type_log, "error": str(e)})

# Generate AI Report Button
gen_report_ready = st.session_state.forecast_df is not None and gemini_configured and can_access_report
gen_report_disabled_reason = ""
if st.session_state.forecast_df is None: gen_report_disabled_reason = "Run forecast first."
elif not gemini_configured: gen_report_disabled_reason = "AI features disabled."
elif not can_access_report: gen_report_disabled_reason = reason_report

if st.sidebar.button("Generate AI Report", key="gen_report_btn", disabled=not gen_report_ready, help=gen_report_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report", feature_used="AI Report")
    with st.spinner("Generating AI analysis..."):
        try:
            data_summary = f"Historical data: {len(st.session_state.df)} points from {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. Avg Level: {st.session_state.df['Level'].mean():.2f}."
            forecast_summary = f"Forecast: {forecast_horizon} steps. Range: {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. 95% CI: [{st.session_state.forecast_df['Lower_CI'].min():.2f}, {st.session_state.forecast_df['Upper_CI'].max():.2f}]."
            metrics_summary = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in st.session_state.metrics.items()]) if st.session_state.metrics and "Info" not in st.session_state.metrics else "N/A"
            seq_len_detail = "N/A"
            if st.session_state.selected_model_type == "Standard": seq_len_detail = st.session_state.standard_model_seq_len
            elif st.session_state.custom_model_seq_len: seq_len_detail = st.session_state.custom_model_seq_len
            elif st.session_state.trained_model_seq_len: seq_len_detail = st.session_state.trained_model_seq_len
            model_details = {"type": st.session_state.selected_model_type, "sequence_length": seq_len_detail, "forecast_horizon": forecast_horizon, "mc_iterations_for_ci": mc_iterations}
            st.session_state.ai_report = generate_ai_report(data_summary, forecast_summary, metrics_summary, model_details)
            st.toast("AI Report generated!", icon="📄")
            if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report_success", feature_used="AI Report")
            st.rerun()
        except Exception as e: st.error(f"Failed to generate AI report: {e}"); logging.error(f"AI Report error: {e}");
        if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report_fail", feature_used="AI Report", details={"error": str(e)})

# Download PDF Report Button
pdf_bytes = None
pdf_ready = st.session_state.forecast_fig or st.session_state.ai_report
if pdf_ready:
    try: pdf_bytes = create_pdf_report(st.session_state.forecast_fig, st.session_state.loss_fig, st.session_state.metrics, st.session_state.ai_report)
    except Exception as e: logging.warning(f"Could not prepare PDF: {e}")

st.sidebar.download_button(
    label="Download PDF Report", data=pdf_bytes if pdf_bytes else b"",
    file_name="DeepHydro_AI_Forecast_Report.pdf", mime="application/pdf",
    key="download_pdf_btn", disabled=not pdf_bytes, help="Generate forecast or AI report first.",
    on_click=log_visitor_activity if pdf_bytes and firebase_initialized else None,
    args=("Sidebar",) if pdf_bytes and firebase_initialized else None,
    kwargs={"action": "download_pdf"} if pdf_bytes and firebase_initialized else None
)

# Activate Chat Button
chat_ready = st.session_state.forecast_df is not None and gemini_configured and can_access_chat
chat_disabled_reason = ""
if st.session_state.forecast_df is None: chat_disabled_reason = "Run forecast first."
elif not gemini_configured: chat_disabled_reason = "AI features disabled."
elif not can_access_chat: chat_disabled_reason = reason_chat

if st.sidebar.button("Activate Chat", key="activate_chat_btn", disabled=not chat_ready, help=chat_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="activate_chat", feature_used="AI Chat")
    if not st.session_state.chat_active:
        initialize_chat()
        if st.session_state.get("chat_model") is not None:
            st.session_state.chat_active = True
            context = f"Chat Context: Forecast generated. Data: {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. Forecast: {forecast_horizon} steps, range {st.session_state.forecast_df['Forecast'].min():.2f}-{st.session_state.forecast_df['Forecast'].max():.2f}. Ask questions."
            st.session_state.messages = [] 
            st.session_state.messages.append({"role": "model", "content": context})
            st.toast("Chat activated!", icon="💬")
            st.rerun()
        else: st.error("Could not activate chat. AI model init failed.")
    else: st.sidebar.info("Chat is already active.")

# --- Main Area Tabs --- 
st.title("Groundwater Level Forecasting & Analysis")

tab_titles = ["Home / Data View", "Forecast Results", "Train Model", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# Home / Data View Tab
with tabs[0]:
    if firebase_initialized: log_visitor_activity("Tab: Home", "page_view")
    st.header("Welcome to DeepHydro AI")
    st.markdown("""<div class="app-intro">This application uses AI (LSTM networks) to forecast groundwater levels. Upload data, choose a model, set parameters, and run the forecast. Generate reports or chat about results.</div>""", unsafe_allow_html=True)
    st.subheader("Uploaded Data Overview")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())
        st.metric("Total Data Points", len(st.session_state.df))
        st.metric("Date Range", f"{st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}")
        st.line_chart(st.session_state.df.set_index('Date')['Level'])
    else:
        st.info("Upload an Excel file (.xlsx) using the sidebar.")

# Forecast Results Tab
with tabs[1]:
    if firebase_initialized: log_visitor_activity("Tab: Forecast", "page_view")
    st.header("Forecast Results")
    if st.session_state.forecast_fig:
        st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecast_df)
        st.subheader("Performance Metrics (from last training)")
        if st.session_state.metrics: st.json(st.session_state.metrics)
        else: st.info("Metrics generated during model training.")
    else: st.info("Run a forecast to see results here.")

# Train Model Tab
with tabs[2]:
    if firebase_initialized: log_visitor_activity("Tab: Train Model", "page_view")
    st.header("Train a New LSTM Model")
    if st.session_state.selected_model_type != "Train New": st.info("Select 'Train New' in the sidebar.")
    elif st.session_state.df is None: st.warning("Upload data first.")
    else:
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        with col1: seq_len_train = st.number_input("Sequence Length", 10, 180, 60, 5, key="train_seq")
        with col2: epochs_train = st.number_input("Epochs", 1, 200, 20, 1, key="train_epochs")
        with col3: batch_size_train = st.number_input("Batch Size", 8, 128, 32, 8, key="train_batch")
        test_size_train = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05, key="train_test_split")
        if st.button("Train New LSTM Model", key="train_model_btn"):
            if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_start", details={"seq": seq_len_train, "epochs": epochs_train, "batch": batch_size_train, "test_split": test_size_train})
            with st.spinner("Training model..."):
                try:
                    df_train = st.session_state.df.copy()
                    scaler_train = MinMaxScaler(feature_range=(0, 1))
                    scaled_data_train = scaler_train.fit_transform(df_train["Level"].values.reshape(-1, 1))
                    X, y = create_sequences(scaled_data_train, seq_len_train)
                    if len(X) == 0: st.error(f"Not enough data ({len(df_train)}) for sequence length {seq_len_train}."); st.stop()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_train, random_state=42, shuffle=False)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)); X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    model_train = build_lstm_model(seq_len_train)
                    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                    history = model_train.fit(X_train, y_train, epochs=epochs_train, batch_size=batch_size_train, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
                    y_pred_scaled = model_train.predict(X_test)
                    y_pred_train = scaler_train.inverse_transform(y_pred_scaled); y_test_train = scaler_train.inverse_transform(y_test.reshape(-1, 1))
                    st.session_state.metrics = calculate_metrics(y_test_train, y_pred_train)
                    st.session_state.training_metrics = st.session_state.metrics
                    st.session_state.loss_fig = create_loss_plot(history.history)
                    st.session_state.custom_model = model_train
                    st.session_state.custom_model_seq_len = seq_len_train
                    st.session_state.model_trained = True
                    st.session_state.trained_model_seq_len = seq_len_train
                    st.toast("Model training completed!", icon="🎉")
                    st.balloons()
                    if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_success", details={"metrics": st.session_state.metrics})
                    st.rerun()
                except Exception as e: st.error(f"Training error: {e}"); logging.error(f"Training error: {e}");
                if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"error": str(e)})
        if st.session_state.model_trained:
            st.subheader("Last Training Results")
            if st.session_state.loss_fig: st.plotly_chart(st.session_state.loss_fig, use_container_width=True)
            if st.session_state.metrics: st.json(st.session_state.metrics)
            st.info("This trained model is now selected. Run forecast from the sidebar.")

# AI Report Tab
with tabs[3]:
    if firebase_initialized: log_visitor_activity("Tab: AI Report", "page_view")
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: st.warning("AI features disabled.")
    elif st.session_state.ai_report:
        report_escaped = st.session_state.ai_report.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="chat-message ai-message">{report_escaped}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    else: st.info("Click 'Generate AI Report' (sidebar) after a forecast.")

# AI Chatbot Tab
with tabs[4]:
    if firebase_initialized: log_visitor_activity("Tab: AI Chatbot", "page_view")
    st.header("AI Chatbot")
    if not gemini_configured: st.warning("AI features disabled.")
    elif st.session_state.chat_active:
        display_chat_history()
        user_prompt = st.chat_input("Ask about the forecast...")
        if user_prompt: handle_chat_input(user_prompt)
    else: st.info("Activate chat via the sidebar after running a forecast.")

# Admin Analytics Tab
with tabs[5]:
    if firebase_initialized: log_visitor_activity("Tab: Admin Analytics", "page_view")
    st.header("Admin Analytics Dashboard")
    ADMIN_PASSWORD = APP_CONFIG.get("admin_password", "admin123") if admin_password_configured else "admin123"
    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")
        if st.button("Login as Admin", key="admin_login_btn"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                if firebase_initialized: log_visitor_activity("Admin Auth", action="admin_login_success")
                st.rerun()
            else:
                st.error("Incorrect admin password.")
                if firebase_initialized: log_visitor_activity("Admin Auth", action="admin_login_fail")
    else:
        st.success("Admin access granted.")
        if st.button("Logout Admin", key="admin_logout_btn"):
            st.session_state.admin_authenticated = False
            if firebase_initialized: log_visitor_activity("Admin Auth", action="admin_logout")
            st.rerun()
        st.markdown("--- Visitor Data ---")
        if firebase_initialized:
            with st.spinner("Fetching visitor logs..."):
                visitor_df = fetch_visitor_logs()
            if not visitor_df.empty:
                st.dataframe(visitor_df)
                st.markdown("--- Visualizations ---")
                charts = create_visitor_charts(visitor_df)
                for chart in charts: st.plotly_chart(chart, use_container_width=True)
                try:
                    csv = visitor_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Analytics CSV", data=csv, file_name="visitor_analytics.csv", mime="text/csv", key="download_analytics_csv", on_click=log_visitor_activity if firebase_initialized else None, args=("Admin Analytics",) if firebase_initialized else None, kwargs={"action": "download_analytics_csv"} if firebase_initialized else None)
                except Exception as e_csv: st.warning(f"Could not generate CSV: {e_csv}");
                if firebase_initialized: log_visitor_activity("Admin Analytics", action="download_analytics_csv_fail", details={"error": str(e_csv)})
            else: st.info("No visitor logs found or error fetching logs.")
        else: st.warning("Firebase not initialized. Cannot display analytics.")

# --- About Us Section --- 
st.markdown("<div class='about-us-header'>About DeepHydro AI ▼</div>", unsafe_allow_html=True)
st.markdown("""<div class="about-us-content">DeepHydro AI leverages LSTM networks for groundwater forecasts. Upload data, choose/train models, and analyze results. \n**Configuration Note:** Reads config from `APP_CONFIG_JSON` env var.</div>""", unsafe_allow_html=True)

# --- (End of Script) --- 

