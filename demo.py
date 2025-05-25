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

# --- Load Configuration (Keep Original) --- 
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

# --- Firebase Initialization (Keep Original) --- 
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

# --- Gemini API Configuration (Keep Original) --- 
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

# --- Load Standard Model (Keep Original) --- 
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

# --- User Identification & Tracking (Keep Original) --- 
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

# --- Authentication Check & Google OAuth (Keep Original) --- 
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

def show_google_login(): # KEEP THIS ORIGINAL FUNCTION
    if not google_oauth_configured:
        st.sidebar.warning("Google Sign-In unavailable (not configured).")
        if firebase_initialized: log_visitor_activity("Authentication", action="google_login_fail_config_missing")
        return

    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    
    if "token" not in st.session_state or st.session_state.token is None:
        # Check usage count before showing login button if needed
        usage_count = -1 # Default to allow login attempt
        if "user_profile" in st.session_state and st.session_state.user_profile:
            usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
        
        if usage_count >= ADVANCED_FEATURE_LIMIT:
            st.sidebar.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached.")
            st.sidebar.info("Please log in with Google to continue.")
        
        # Always show login button if not authenticated
        result = oauth2.authorize_button(
            name="Login with Google",
            # icon="https://www.google.com/favicon.ico", # Optional icon
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_login_button", # Use a unique key
            extras_params={"prompt": "consent", "access_type": "offline"}
        )
        
        if result and "token" in result:
            st.session_state.token = result.get("token")
            user_info = get_user_info_from_google(st.session_state.token)
            if user_info and user_info.get("email"):
                st.session_state.auth_status = "authenticated"
                st.session_state.user_email = user_info.get("email")
                st.session_state.user_name = user_info.get("name")
                st.session_state.user_picture = user_info.get("picture")
                # Update Firebase profile upon successful login
                update_firebase_profile_on_login(st.session_state.user_email)
                st.sidebar.success(f"Logged in as {st.session_state.user_name}")
                time.sleep(1) # Short delay to show message
                st.rerun() # Rerun to update UI
            else:
                st.session_state.auth_status = "error"
                st.session_state.token = None # Clear token on error
                st.sidebar.error("Failed to get user info from Google.")
                log_visitor_activity("Authentication", action="google_login_fail_userinfo")
        elif result: # Handle potential errors from the component
            logging.warning(f"OAuth result received but no token: {result}")
            log_visitor_activity("Authentication", action="google_login_fail_no_token", details=result)
            
    elif st.session_state.get("auth_status") == "authenticated":
        # Show logged-in status and logout button
        if st.session_state.get("user_picture"):
            st.sidebar.image(st.session_state.user_picture, width=50)
        st.sidebar.write(f"Logged in as: {st.session_state.get('user_name', st.session_state.get('user_email'))}")
        if st.sidebar.button("Logout", key="logout_button"):
            # Clear session state related to auth
            st.session_state.auth_status = "logged_out"
            st.session_state.token = None
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.user_picture = None
            # Optionally revoke token (if needed, requires token endpoint)
            # oauth2.revoke_token(st.session_state.token) 
            log_visitor_activity("Authentication", action="logout")
            st.rerun()

# --- Visitor Analytics Functions (Keep Original) ---
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None, details=None):
    if not firebase_initialized: return
    try:
        user_id = get_persistent_user_id()
        profile, _ = get_or_create_user_profile(user_id)
        
        # Increment usage count if a limited feature is used successfully
        # Note: Access check happens *before* calling this in the UI for limited features
        should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"] # Define limited features
        if should_increment and action not in ["run_forecast_denied", "generate_report_denied", "activate_chat_denied"]:
            increment_feature_usage(user_id)

        ref = db.reference("visitor_logs") # Original collection name
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
    if not firebase_initialized: return pd.DataFrame()
    try:
        ref = db.reference("visitor_logs") # Original collection name
        visitors_data = ref.get()
        if not visitors_data: return pd.DataFrame()
        visitors_list = []
        for log_id, data in visitors_data.items():
            # Flatten details if they exist and are dicts
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
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        return df
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
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
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hourly_activity = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            
            if len(hourly_activity) > 0:
                hourly_pivot = hourly_activity.pivot_table(values='count', index='day_of_week', columns='hour', fill_value=0)
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

# --- Admin Analytics Dashboard (Keep Original Logic, Adapt UI) ---
def render_admin_analytics():
    st.header("Admin Analytics Dashboard")
    
    # Use original admin password check from config
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
        # Fetch visitor logs using original function
        visitor_df = fetch_visitor_logs()
        
        if visitor_df.empty:
            st.info("No visitor data available yet.")
            return
        
        # Display visitor statistics (Adapt from new design)
        st.subheader("Visitor Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_activities = len(visitor_df)
            st.metric("Total Activities Logged", total_activities)
        
        with col2:
            unique_visitors = visitor_df["persistent_user_id"].nunique() if "persistent_user_id" in visitor_df else 0
            st.metric("Unique Visitors", unique_visitors)
        
        with col3:
            today_visitors = 0
            if "timestamp" in visitor_df:
                 today = datetime.datetime.now().date()
                 # Count unique visitors today
                 today_visitors = visitor_df[visitor_df["timestamp"].dt.date == today]["persistent_user_id"].nunique() if "persistent_user_id" in visitor_df else 0
            st.metric("Today's Unique Visitors", today_visitors)
        
        # Create and display visualizations using original function
        st.subheader("Visitor Analytics")
        try:
            charts = create_visitor_charts(visitor_df)
            for fig in charts:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as chart_err:
            st.error(f"Error displaying charts: {chart_err}")
        
        # Display raw data with filters (Adapt from new design)
        st.subheader("Raw Visitor Data")
        col1_filter, col2_filter = st.columns(2)
        date_range = None
        user_id_filter = 'All'
        try:
            if "timestamp" in visitor_df and not visitor_df["timestamp"].isnull().all():
                min_date = visitor_df["timestamp"].min().date()
                max_date = visitor_df["timestamp"].max().date()
                with col1_filter:
                    date_range = st.date_input("Date Range", [min_date, max_date], key="admin_date_filter")
            else:
                 with col1_filter:
                     st.info("No timestamp data for date filter.")
                     
            if "persistent_user_id" in visitor_df and not visitor_df["persistent_user_id"].isnull().all():
                user_id_options = ['All'] + visitor_df["persistent_user_id"].unique().tolist()
                with col2_filter:
                    user_id_filter = st.selectbox("Filter by User ID", options=user_id_options, index=0, key="admin_user_filter")
            else:
                 with col2_filter:
                     st.info("No user ID data for filter.")
        except Exception as filter_setup_err:
            st.warning(f"Error setting up filters: {filter_setup_err}")

        try:
            filtered_df = visitor_df.copy()
            if date_range and len(date_range) == 2 and "timestamp" in filtered_df:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df["timestamp"].dt.date >= start_date) & (filtered_df["timestamp"].dt.date <= end_date)]
            if user_id_filter != 'All' and "persistent_user_id" in filtered_df:
                filtered_df = filtered_df[filtered_df["persistent_user_id"] == user_id_filter]
            
            # Define relevant columns (can customize)
            display_cols = [col for col in ['timestamp', 'persistent_user_id', 'is_authenticated', 'visit_count', 'page', 'action', 'feature_used', 'ip_address', 'session_id', 'log_id'] if col in filtered_df.columns]
            st.dataframe(filtered_df[display_cols])
            
            if st.button("Export Filtered to CSV", key="admin_export_btn"):
                csv = filtered_df[display_cols].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_visitor_logs.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                log_visitor_activity("Admin Dashboard", action="export_csv")
        except Exception as filter_err:
            st.error(f"Error applying filters or displaying data: {filter_err}")
            fallback_cols = [col for col in ['timestamp', 'persistent_user_id', 'action'] if col in visitor_df.columns]
            st.dataframe(visitor_df[fallback_cols]) # Fallback display
            
        if st.button("Logout Admin", key="admin_logout_btn"):
            st.session_state.admin_authenticated = False
            log_visitor_activity("Admin Dashboard", action="logout_admin")
            st.rerun()

# --- Custom CSS (Use New Design) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* New design CSS rules */
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
    """, unsafe_allow_html=True)

# --- JavaScript (Use New Design) ---
def add_javascript_functionality():
    st.markdown("""
    <script>
    // Function to copy text to clipboard
    function copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
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
        const chatMessages = document.querySelectorAll(".chat-message");
        chatMessages.forEach(function(message) {
            // Add tooltip element if not present
            if (!message.querySelector(".copy-tooltip")) {
                const tooltip = document.createElement('span');
                tooltip.className = 'copy-tooltip';
                tooltip.textContent = 'Copied!';
                message.appendChild(tooltip);
            }
            
            let longPressTimer;
            const handleStart = (e) => {
                // Prevent default behavior if needed (e.g., scrolling on touch)
                // e.preventDefault(); 
                longPressTimer = setTimeout(() => {
                    const textToCopy = message.innerText.replace('Copied!', '').trim(); // Exclude tooltip text
                    copyToClipboard(textToCopy);
                    const tooltip = message.querySelector(".copy-tooltip");
                    if (tooltip) {
                        tooltip.style.display = 'block';
                        setTimeout(() => { tooltip.style.display = 'none'; }, 1500);
                    }
                }, 500); // 500ms for long press
            };

            const handleEnd = () => {
                clearTimeout(longPressTimer);
            };

            // Use mouse events for desktop and touch events for mobile
            message.removeEventListener('mousedown', handleStart);
            message.removeEventListener('mouseup', handleEnd);
            message.removeEventListener('mouseleave', handleEnd);
            message.removeEventListener('touchstart', handleStart);
            message.removeEventListener('touchend', handleEnd);
            message.removeEventListener('touchmove', handleEnd); // Cancel long press if finger moves

            message.addEventListener('mousedown', handleStart);
            message.addEventListener('mouseup', handleEnd);
            message.addEventListener('mouseleave', handleEnd);
            message.addEventListener('touchstart', handleStart, { passive: true });
            message.addEventListener('touchend', handleEnd);
            message.addEventListener('touchmove', handleEnd, { passive: true });
        });
        
        // Collapsible About Us
        const aboutUsHeader = document.querySelector(".about-us-header");
        const aboutUsContent = document.querySelector(".about-us-content");
        if (aboutUsHeader && aboutUsContent) {
            // Check if already initialized to maintain state across reruns
            if (!aboutUsContent.classList.contains('initialized')) {
                 aboutUsContent.style.display = 'none'; // Start collapsed
                 aboutUsContent.classList.add('initialized');
            }
            // Remove previous listener to avoid duplicates
            aboutUsHeader.removeEventListener('click', toggleAboutUs);
            aboutUsHeader.addEventListener('click', toggleAboutUs);
        }
    };
    
    const toggleAboutUs = () => {
        const aboutUsContent = document.querySelector(".about-us-content");
        if (aboutUsContent) {
            if (aboutUsContent.style.display === 'none') {
                aboutUsContent.style.display = 'block';
            } else {
                aboutUsContent.style.display = 'none';
            }
        }
    };

    // Use MutationObserver to re-apply listeners when Streamlit updates the DOM
    const observer = new MutationObserver(debounce((mutationsList, observer) => {
        // Look for changes that might add new chat messages or affect the About Us section
        for(const mutation of mutationsList) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Check if relevant elements were added or modified
                 setupInteractions();
                 break; // No need to check further mutations in this batch
            }
        }
    }, 250)); // Debounce to avoid excessive calls

    // Start observing the Streamlit app container for changes
    const targetNode = window.parent.document.querySelector('section.main'); // Adjust selector if needed
    if (targetNode) {
        observer.observe(targetNode, { childList: true, subtree: true });
    } else {
        // Fallback if the main container isn't found immediately
        window.addEventListener('load', () => {
             const fallbackNode = window.parent.document.querySelector('section.main');
             if(fallbackNode) observer.observe(fallbackNode, { childList: true, subtree: true });
        });
    }

    // Initial setup on load
    if (document.readyState === 'complete') {
        setupInteractions();
    } else {
        window.addEventListener('load', setupInteractions);
    }
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration (Use New Design) ---
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css() # Apply new CSS
# JavaScript will be added at the end of the script execution

# --- Capture User Agent (Adapt from New Design) ---
def capture_user_agent():
    if 'user_agent' not in st.session_state:
        try:
            # Use Streamlit components to run JavaScript that sends the user agent
            # Ensure a unique key for the component
            user_agent_val = components.html(
                """
                <script>
                // Send the user agent back to Streamlit
                window.parent.postMessage({
                    isStreamlitMessage: true,
                    type: "streamlit:setComponentValue",
                    key: "user_agent_capture_component", 
                    value: navigator.userAgent
                }, "*");
                </script>
                """,
                height=0,
                key="user_agent_capture_component" # Unique key
            )
            # The component value might take a moment to arrive
            # Check component state directly if available
            component_value = st.session_state.get("user_agent_capture_component")
            if component_value:
                 st.session_state.user_agent = component_value
            elif user_agent_val: # Fallback to direct return value (less reliable)
                 st.session_state.user_agent = user_agent_val
            else:
                 st.session_state.user_agent = "Unknown (Capture Pending)"
        except Exception as e:
            logging.warning(f"User agent capture failed: {e}")
            st.session_state.user_agent = "Unknown (Capture Failed)"

# --- Initializations (Combine Original Logic with New Structure) ---
load_configuration() # Load config first
firebase_initialized = initialize_firebase() # Initialize Firebase using original logic
configure_gemini() # Configure Gemini using original logic
load_standard_model() # Load standard model using original logic
capture_user_agent() # Attempt to capture user agent

# Initialize user profile in session state if not already present (using original logic)
if 'user_profile' not in st.session_state:
    if firebase_initialized:
        user_id = get_persistent_user_id() # Get ID first using original logic
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id) # Fetch/create profile using original logic
    else:
        st.session_state.user_profile = None # No profile if Firebase fails

# --- Model Paths & Constants (Keep Original Logic) ---
# STANDARD_MODEL_PATH and ADVANCED_FEATURE_LIMIT already defined
# Re-check standard model sequence length (might have been updated by load_standard_model)
STANDARD_MODEL_SEQUENCE_LENGTH = st.session_state.get("standard_model_seq_len", 60)

# --- Helper Functions (Keep Original - Data Loading, Model Building, Prediction, Plotting, Gemini) ---
# All functions like load_and_clean_data, create_sequences, load_keras_model_from_file, 
# load_standard_model_cached, build_lstm_model, predict_with_dropout_uncertainty, 
# calculate_metrics, create_forecast_plot, create_loss_plot, generate_gemini_report, 
# get_gemini_chat_response, run_forecast_pipeline ARE KEPT AS THEY WERE IN THE ORIGINAL SCRIPT.
# (Code for these functions is omitted here for brevity but assumed to be present and unchanged)

@st.cache_data
def load_and_clean_data(uploaded_file_content):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        if df.shape[1] < 2: st.error("File must have at least two columns (Date, Level)."); return None
        date_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["date", "time"])), None)
        level_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["level", "groundwater", "gwl"])), None)
        if not date_col: st.error("Cannot find Date column (e.g., named 'Date', 'Time')."); return None
        if not level_col: st.error("Cannot find Level column (e.g., named 'Level', 'Groundwater Level')."); return None
        st.success(f"Identified columns: Date='{date_col}', Level='{level_col}'. Renaming to 'Date' and 'Level'.")
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
    temp_model_path = f"temp_{model_name_for_log.replace(' ', '_')}.h5"
    try:
        with open(temp_model_path, "wb") as f: f.write(uploaded_file_obj.getbuffer())
        model = load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded {model_name_for_log}. Inferred sequence length: {sequence_length}")
        return model, sequence_length
    except Exception as e: st.error(f"Error loading Keras model {model_name_for_log}: {e}"); return None, None
    finally: 
        if os.path.exists(temp_model_path): os.remove(temp_model_path)

@st.cache_resource
def load_standard_model_cached(path):
    # Use the model already loaded into session state if available
    if "standard_model" in st.session_state and st.session_state.standard_model is not None:
        model = st.session_state.standard_model
        seq_len = st.session_state.get("standard_model_seq_len", STANDARD_MODEL_SEQUENCE_LENGTH)
        # st.info(f"Using cached standard model from session state (Seq Len: {seq_len})")
        return model, seq_len
    # Fallback to loading from path if not in session state
    try:
        model = load_model(path, compile=False)
        sequence_length = model.input_shape[1]
        st.session_state.standard_model = model # Cache it
        st.session_state.standard_model_seq_len = sequence_length
        # st.info(f"Loaded standard model from path into cache (Seq Len: {sequence_length})")
        return model, sequence_length
    except Exception as e: 
        st.error(f"Error loading standard Keras model from {path}: {e}")
        return None, None

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    
    # Ensure model allows training=True during prediction if it has Dropout layers
    try:
        # Check if model has dropout layers that behave differently in training mode
        has_dropout = any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)
    except: has_dropout = True # Assume yes if check fails

    @tf.function
    def predict_step_training_mode(inp):
        return model(inp, training=has_dropout) # Only set training=True if dropout exists
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting MC Dropout ({n_iterations} iterations)...")
    
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step_training_mode(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            # Ensure the new prediction is added correctly
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
    
    # Inverse transform
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    # Calculate bounds in scaled space first, then inverse transform
    ci_multiplier = 1.96 # 95% CI
    lower_bound_scaled = mean_preds_scaled - ci_multiplier * std_devs_scaled
    upper_bound_scaled = mean_preds_scaled + ci_multiplier * std_devs_scaled
    lower_bound = scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()
    
    # Ensure bounds are reasonable (lower <= mean <= upper)
    lower_bound = np.minimum(lower_bound, mean_preds)
    upper_bound = np.maximum(upper_bound, mean_preds)
    
    # Optional: Add minimum uncertainty range if needed (from new design)
    # min_uncertainty_percent = 0.05 
    # for i in range(len(mean_preds)):
    #     if mean_preds[i] != 0:
    #         current_range_percent = (upper_bound[i] - lower_bound[i]) / mean_preds[i]
    #         if current_range_percent < min_uncertainty_percent:
    #             uncertainty_value = mean_preds[i] * min_uncertainty_percent / 2
    #             lower_bound[i] = mean_preds[i] - uncertainty_value
    #             upper_bound[i] = mean_preds[i] + uncertainty_value
    #     elif upper_bound[i] == lower_bound[i]: # Handle zero mean case
    #          # Add a small absolute uncertainty if desired
    #          pass 
             
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    # Ensure no NaNs or Infs in inputs
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0: return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    # Handle division by zero for MAPE
    non_zero_true = y_true_valid[y_true_valid != 0]
    non_zero_pred = y_pred_valid[y_true_valid != 0]
    mape = np.inf
    if len(non_zero_true) > 0:
        mape = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
        
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def create_forecast_plot(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
    # Ensure CIs are plotted correctly
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=True)) # Show legend for CI band
    
    fig.update_layout(title="Groundwater Level: Historical Data & LSTM Forecast", xaxis_title="Date", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_loss_plot(history_dict):
    if not history_dict or not isinstance(history_dict, dict) or "loss" not in history_dict or "val_loss" not in history_dict:
        fig = go.Figure()
        fig.update_layout(title="No Training History Available", xaxis_title="Epoch", yaxis_title="Loss")
        fig.add_annotation(text="Training history is not available.",xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    history_df = pd.DataFrame(history_dict); history_df["Epoch"] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", template="plotly_white")
    return fig

def generate_gemini_report(hist_df, forecast_df, metrics, language):
    global gemini_model_report # Ensure we use the globally configured model
    if not gemini_configured or gemini_model_report is None: return "AI report generation disabled or model not configured."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    try:
        # Construct prompt (Ensure language selection is handled if needed)
        prompt = f"""Act as a professional hydrologist analyzing groundwater level data and LSTM forecast results. Provide a concise scientific report in {language}. 
        
        **Analysis Context:**
        - Historical Data Period: {hist_df['Date'].min():%Y-%m-%d} to {hist_df['Date'].max():%Y-%m-%d}
        - Forecast Period: {forecast_df['Date'].min():%Y-%m-%d} to {forecast_df['Date'].max():%Y-%m-%d}
        - Model Evaluation Metrics (Validation/Pseudo-Validation):
            - RMSE: {metrics.get('RMSE', 'N/A'):.4f}
            - MAE: {metrics.get('MAE', 'N/A'):.4f}
            - MAPE: {metrics.get('MAPE', 'N/A'):.2f}%

        **Instructions:**
        1.  **Introduction:** Briefly state the purpose - analyzing historical groundwater levels and evaluating an LSTM forecast.
        2.  **Historical Data Insights:** Describe key trends, seasonality (if apparent), and notable high/low periods observed in the historical data. Mention the overall range (min/max levels).
        3.  **Forecast Evaluation:** Comment on the model's performance based on the provided metrics (RMSE, MAE, MAPE). Interpret what these values imply about the forecast accuracy (e.g., low MAE suggests good average prediction).
        4.  **Forecast Analysis:** Describe the predicted trend for the forecast period. Mention the confidence interval (Lower/Upper CI) and what it suggests about the forecast uncertainty.
        5.  **Conclusion & Recommendations:** Summarize the findings. Briefly mention potential applications or limitations based on the analysis and forecast uncertainty. Avoid definitive statements; use cautious language.

        **Historical Data Summary:**
        {hist_df["Level"].describe().to_string()}
        
        **Forecast Data Summary:**
        {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}
        """
        response = gemini_model_report.generate_content(prompt)
        # Check for safety ratings or blocks if necessary
        # if response.prompt_feedback.block_reason:
        #    return f"Report generation blocked: {response.prompt_feedback.block_reason}"
        return response.text
    except Exception as e: 
        st.error(f"Error generating AI report: {e}")
        return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    global gemini_model_chat # Ensure we use the globally configured model
    if not gemini_configured or gemini_model_chat is None: return "AI chat disabled or model not configured."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient context for AI chat."
    try:
        # Build context string including chat history
        # Ensure messages are strings before joining
        history_context = "\n".join([f"{sender}: {str(message)}" for sender, message in chat_hist])

        # Use triple quotes for the f-string
        context = f"""You are an AI assistant helping a user understand groundwater level data and forecast results.
        Use the following context to answer the user's query accurately and concisely.

        **Available Context:**
        - Historical Data Period: {hist_df['Date'].min():%Y-%m-%d} to {hist_df['Date'].max():%Y-%m-%d}
        - Forecast Period: {forecast_df['Date'].min():%Y-%m-%d} to {forecast_df['Date'].max():%Y-%m-%d}
        - Model Evaluation Metrics: RMSE={metrics.get('RMSE', 'N/A'):.4f}, MAE={metrics.get('MAE', 'N/A'):.4f}, MAPE={metrics.get('MAPE', 'N/A'):.2f}%
        - AI Report Summary (if available): {str(ai_report)[:500] if ai_report else 'Not generated yet.'}...
        - Historical Data Summary:
        {hist_df['Level'].describe().to_string()}
        - Forecast Data Summary:
        {forecast_df[['Forecast', 'Lower_CI', 'Upper_CI']].describe().to_string()}

        **Chat History:**
        {history_context}

        **User Query:** {user_query}

        **Instructions:** Answer the user's query based *only* on the provided context and chat history. Be helpful and informative. If the answer isn't in the context, say so.

        AI Assistant:""" # Ensure closing triple quotes are present and correct

        response = gemini_model_chat.generate_content(context)
        # Check for safety ratings or blocks if necessary
        # if response.prompt_feedback.block_reason:
        #    return f"Chat response blocked: {response.prompt_feedback.block_reason}"
        return response.text
    except Exception as e:
        st.error(f"Error in AI chat: {e}")
        return f"Error in AI chat: {e}"

def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    st.info(f"Starting forecast pipeline with model: {model_choice}")
    model = None; history_data = None
    # Use the globally determined sequence length for standard/custom models
    model_sequence_length = st.session_state.get("standard_model_seq_len", STANDARD_MODEL_SEQUENCE_LENGTH) 
    if model_choice == "Train New Model":
        model_sequence_length = sequence_length_train_param # Use user input for training
    elif model_choice == "Upload Custom .h5 Model" and "custom_model_seq_len" in st.session_state:
        model_sequence_length = st.session_state.custom_model_seq_len # Use inferred length for custom
        
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    try:
        st.info("Step 1: Preparing Model...")
        if model_choice == "Standard Pre-trained Model":
            if os.path.exists(STANDARD_MODEL_PATH):
                # Use cached model if available
                model, model_sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                if model is None: return None, None, None, None # Error handled in cached function
                st.session_state.model_sequence_length = model_sequence_length # Update session state
            else: 
                st.error(f"Standard model not found at {STANDARD_MODEL_PATH}."); return None, None, None, None
        elif model_choice == "Upload Custom .h5 Model" and custom_model_file_obj is not None:
            model, inferred_seq_len = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
            if model is None: return None, None, None, None
            model_sequence_length = inferred_seq_len # Use inferred length
            st.session_state.custom_model_seq_len = inferred_seq_len # Store for later use
            st.session_state.model_sequence_length = model_sequence_length # Update session state
        elif model_choice == "Train New Model":
            model_sequence_length = sequence_length_train_param # Already set
            st.session_state.model_sequence_length = model_sequence_length # Update session state
        else:
            st.error("Invalid model choice or missing file."); return None, None, None, None
        
        st.info(f"Model prep complete. Using sequence length: {model_sequence_length}")

        st.info("Step 2: Preprocessing Data (Scaling)...")
        # Fit scaler based on full historical data, apply custom params if provided *after* fitting
        scaler_obj.fit(df["Level"].values.reshape(-1, 1))
        if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
             # Override the scaler's fitted params - use with caution!
             scaler_obj.data_min_ = np.array([custom_scaler_min_param])
             scaler_obj.data_max_ = np.array([custom_scaler_max_param])
             scaler_obj.data_range_ = scaler_obj.data_max_ - scaler_obj.data_min_
             scaler_obj.min_ = np.array([custom_scaler_min_param * scaler_obj.scale_ + scaler_obj.feature_range[0]])
             scaler_obj.scale_ = (scaler_obj.feature_range[1] - scaler_obj.feature_range[0]) / (scaler_obj.data_range_)
             st.warning("Applied custom scaling parameters. Ensure they match the model's training data.")
             
        scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
        st.info("Data scaling complete.")

        st.info(f"Step 3: Creating sequences (length {model_sequence_length})...")
        if len(df) <= model_sequence_length: 
            st.error(f"Not enough data ({len(df)}) for sequence length {model_sequence_length}. Need at least {model_sequence_length + 1} points."); return None, None, None, None
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0: 
            st.error("Could not create sequences from the data."); return None, None, None, None
        st.info(f"Sequences created: {len(X)}")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        if model_choice == "Train New Model":
            st.info(f"Step 4a: Training New Model (Epochs: {epochs_train_param})...")
            # Split data for training/validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_train) == 0 or len(X_val) == 0: 
                st.error("Not enough sequences for train/validation split after creating sequences."); return None, None, None, None
            model = build_lstm_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            # Use a progress bar for training
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
        else: # Pre-trained or Custom Uploaded
            st.info("Step 4b: Evaluating Pre-trained/Uploaded Model (Pseudo-Validation)...")
            # Use last 20% of sequences for pseudo-validation if enough data
            if len(X) > 5:
                val_split_idx = max(1, int(len(X) * 0.8)) # Ensure at least 1 sample for validation if possible
                X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                if len(X_val_pseudo) > 0:
                    val_predictions_scaled = model.predict(X_val_pseudo)
                    val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo)
                    evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation complete.")
                else: st.warning("Not enough data for pseudo-validation split.")
            else: st.warning("Not enough sequences for pseudo-validation.")

        st.info(f"Step 5: Forecasting {forecast_horizon} Steps (MC Dropout: {mc_iterations_param})...")
        last_sequence_scaled_for_pred = scaled_data[-model_sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_dropout_uncertainty(model, last_sequence_scaled_for_pred, forecast_horizon, mc_iterations_param, scaler_obj, model_sequence_length)
        st.success("Forecasting complete.")

        # Generate forecast dates
        last_date = df["Date"].iloc[-1]
        try: 
            # Attempt to infer frequency, default to daily
            freq = pd.infer_freq(df["Date"].dropna())
            if freq is None: freq = "D"
            date_offset = pd.tseries.frequencies.to_offset(freq)
        except Exception as freq_err:
            logging.warning(f"Could not infer frequency or create offset: {freq_err}. Defaulting to daily.")
            freq = "D"
            date_offset = pd.DateOffset(days=1)
            
        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=date_offset)
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": mean_forecast, "Lower_CI": lower_bound, "Upper_CI": upper_bound})
        
        st.info("Forecast pipeline finished successfully.")
        return forecast_df, evaluation_metrics, history_data, scaler_obj

    except Exception as e:
        st.error(f"An error occurred in the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc()) # Show full traceback for debugging
        return None, None, None, None

# --- Streamlit Callback for Training Progress ---
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        loss = logs.get('loss', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        self.status_text.text(f"Epoch {epoch+1}/{self.total_epochs} | Loss: {loss_str} | Val Loss: {val_loss_str}")

    def close(self):
        self.progress_bar.empty()
        self.status_text.empty()

# --- Initialize Session State (Combine Original and New Keys, Set Defaults) ---
def initialize_session_state():
    # Keys from original + new design, ensure all are initialized
    keys_to_init = [
        "cleaned_data", "forecast_results", "evaluation_metrics", "training_history", 
        "ai_report", "scaler_object", "forecast_plot_fig", "uploaded_data_filename",
        "active_tab", "report_language", "chat_history", "chat_active", 
        "model_sequence_length", "run_forecast_triggered", "about_us_expanded",
        "persistent_user_id", "user_profile", 
        "token", "auth_status", "user_email", "user_name", "user_picture", # Original Google Auth keys
        "admin_authenticated", "session_visit_logged", "user_agent",
        "standard_model", "standard_model_seq_len", "custom_model_seq_len", # Model related
        "last_model_choice", "user_agent_capture_component" # UI interaction state
    ]
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = None 

    # Set specific defaults
    if st.session_state.chat_history is None: st.session_state.chat_history = []
    if st.session_state.chat_active is None: st.session_state.chat_active = False
    if st.session_state.model_sequence_length is None: st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
    if st.session_state.run_forecast_triggered is None: st.session_state.run_forecast_triggered = False
    if st.session_state.active_tab is None: st.session_state.active_tab = 0
    if st.session_state.about_us_expanded is None: st.session_state.about_us_expanded = False # New UI state
    if st.session_state.report_language is None: st.session_state.report_language = "English"
    if st.session_state.admin_authenticated is None: st.session_state.admin_authenticated = False
    if st.session_state.auth_status is None: st.session_state.auth_status = "logged_out" # Default to logged out for Google Auth

initialize_session_state() # Call initialization

# --- Sidebar (Use New Layout, Integrate Original Logic) ---
with st.sidebar:
    st.title("DeepHydro AI")
    
    # Log sidebar view activity using original function
    if firebase_initialized:
        log_visitor_activity("Sidebar", "view")
    
    st.header("1. Upload Data")
    uploaded_data_file = st.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")
    
    # Logging for upload success/failure is handled in the main area where cleaning happens

    st.header("2. Model & Forecast")
    model_choice = st.selectbox("Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select")
    
    # Log model selection change using original function
    if firebase_initialized:
        if st.session_state.get('last_model_choice') != model_choice:
             log_visitor_activity("Sidebar", "select_model", feature_used=model_choice)
             st.session_state.last_model_choice = model_choice

    # --- Input Parameters (Structure from new design) ---
    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False
    # Use sequence length from session state, updated by model loading/training choices
    current_sequence_length = st.session_state.get("model_sequence_length", STANDARD_MODEL_SEQUENCE_LENGTH)
    sequence_length_train_sidebar = current_sequence_length # Default for display/training input
    epochs_train_sidebar = 50 # Default

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload .h5 model", type="h5", key="custom_h5_uploader")
        if custom_model_file_obj_sidebar:
             # Attempt to load model here to get seq len early (optional, but improves UX)
             # This might be slow, consider doing it only when Run Forecast is clicked
             pass # Placeholder - loading happens in pipeline
        st.info(f"Custom model selected. Sequence length will be inferred upon running.")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_custom_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="custom_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="custom_scaler_max_in")
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using standard model (Seq Len: {current_sequence_length})")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_std_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values standard model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="std_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="std_scaler_max_in")
    elif model_choice == "Train New Model":
        try:
            # Allow user to set sequence length only when training
            sequence_length_train_sidebar = st.number_input("LSTM Sequence Length", min_value=10, max_value=365, value=current_sequence_length, step=10, key="seq_len_train_in")
        except Exception as e:
            st.warning(f"Using default sequence length {current_sequence_length} due to input error: {e}")
            sequence_length_train_sidebar = current_sequence_length
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in")

    mc_iterations_sidebar = st.number_input("MC Dropout Iterations (C.I.)", min_value=20, max_value=500, value=100, step=10, key="mc_iter_in")
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in")

    # --- Run Forecast Button (Use Original Access Check & Login) ---
    run_forecast_button = st.button("Run Forecast", key="run_forecast_main_btn", use_container_width=True)
    
    if run_forecast_button:
        access_granted, message = check_feature_access() # Use original access check
        if access_granted:
            st.session_state.run_forecast_triggered = True
            if st.session_state.cleaned_data is not None:
                if model_choice == "Upload Custom .h5 Model" and custom_model_file_obj_sidebar is None:
                    st.error("Please upload a custom .h5 model file.")
                    st.session_state.run_forecast_triggered = False
                else:
                    # Log successful access/usage *before* running (using original log function)
                    if firebase_initialized:
                        log_visitor_activity("Sidebar", "run_forecast", feature_used='Forecast')
                        
                    with st.spinner(f"Running forecast ({model_choice})..."):
                        # Call original pipeline function
                        forecast_df, metrics, history, scaler_obj = run_forecast_pipeline(
                            st.session_state.cleaned_data, model_choice, forecast_horizon_sidebar, 
                            custom_model_file_obj_sidebar, sequence_length_train_sidebar, epochs_train_sidebar, 
                            mc_iterations_sidebar, use_custom_scaler_sidebar, custom_scaler_min_sidebar, custom_scaler_max_sidebar
                        )
                    # Store results in session state
                    st.session_state.forecast_results = forecast_df
                    st.session_state.evaluation_metrics = metrics
                    st.session_state.training_history = history
                    st.session_state.scaler_object = scaler_obj
                    if forecast_df is not None and metrics is not None:
                        # Create plot and store it
                        st.session_state.forecast_plot_fig = create_forecast_plot(st.session_state.cleaned_data, forecast_df)
                        st.success("Forecast complete! Results updated.")
                        # Reset downstream results on new forecast
                        st.session_state.ai_report = None; 
                        st.session_state.chat_history = []; st.session_state.chat_active = False
                        st.session_state.active_tab = 1 # Switch to forecast tab
                        st.rerun()
                    else:
                        st.error("Forecast pipeline failed. Check messages above.")
                        # Clear potentially partial results
                        st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                        st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
            else:
                st.error("Please upload data first.")
                st.session_state.run_forecast_triggered = False
        else:
            # If access denied, show ORIGINAL Google login prompt
            st.warning(message) # Show reason for denial (e.g., usage limit)
            show_google_login() # Use original Google login function
            # Log denied access attempt
            if firebase_initialized:
                 log_visitor_activity("Sidebar", "run_forecast_denied", feature_used='Forecast')

    st.header("3. AI Analysis")
    st.session_state.report_language = st.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_configured)
    
    # --- Generate AI Report Button (Use Original Access Check & Login) ---
    generate_report_button = st.button("Generate AI Report", key="show_report_btn", disabled=not gemini_configured, use_container_width=True)
    
    if generate_report_button:
        access_granted, message = check_feature_access() # Use original access check
        if access_granted:
            if not gemini_configured: st.error("AI Report disabled. Configure Gemini API Key.")
            elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
                # Log successful access/usage (using original log function)
                if firebase_initialized:
                    log_visitor_activity("Sidebar", "generate_report", feature_used='AI Report')
                    
                with st.spinner(f"Generating AI report ({st.session_state.report_language})..."):
                    # Call original Gemini function
                    st.session_state.ai_report = generate_gemini_report(
                        st.session_state.cleaned_data, st.session_state.forecast_results,
                        st.session_state.evaluation_metrics, st.session_state.report_language
                    )
                if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:") and not st.session_state.ai_report.startswith("AI report generation disabled"):
                    st.success("AI report generated.")
                    st.session_state.active_tab = 3 # Switch to AI report tab
                    st.rerun()
                else: 
                    st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
            else: 
                st.error("Data, forecast, and metrics needed. Run forecast first.")
        else:
            # If access denied, show ORIGINAL Google login prompt
            st.warning(message)
            show_google_login() # Use original Google login function
            if firebase_initialized:
                 log_visitor_activity("Sidebar", "generate_report_denied", feature_used='AI Report')

    # --- Download PDF Button (Keep Original Logic) ---
    if st.button("Download Report (PDF)", key="download_report_btn", use_container_width=True):
        if firebase_initialized:
            log_visitor_activity("Sidebar", "download_pdf") # Log attempt
            
        # Check if all necessary components are available
        if (st.session_state.forecast_results is not None and 
            st.session_state.evaluation_metrics is not None and 
            st.session_state.ai_report is not None and 
            st.session_state.forecast_plot_fig is not None):
            
            with st.spinner("Generating PDF report..."):
                try:
                    pdf = FPDF(); pdf.add_page()
                    # Font handling (same as original)
                    font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    report_font = "Arial"
                    try: 
                        if os.path.exists(font_path_dejavu):
                            pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True); report_font = "DejaVu"
                        else: st.warning(f"DejaVu font not found at {font_path_dejavu}, using Arial.")
                    except RuntimeError as font_err:
                         st.warning(f"Failed to add DejaVu font ({font_err}), using Arial.")
                         report_font = "Arial"
                    
                    pdf.set_font(report_font, size=12); pdf.cell(0, 10, txt="DeepHydro AI Forecasting Report", new_x="LMARGIN", new_y="NEXT", align="C"); pdf.ln(5)
                    
                    # Embed plot (same as original)
                    plot_filename = "forecast_plot.png"
                    img_embedded = False
                    try:
                        # Ensure figure exists before writing
                        if st.session_state.forecast_plot_fig:
                            st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                            if os.path.exists(plot_filename):
                                pdf.image(plot_filename, x=pdf.get_x(), y=pdf.get_y(), w=190)
                                pdf.ln(125) # Adjust spacing based on image height
                                img_embedded = True
                            else: st.warning("Plot image file was not created.")
                        else: st.warning("Forecast plot figure not found in session state.")
                    except Exception as img_err: st.warning(f"Could not embed plot image: {img_err}.")
                    finally: 
                        if os.path.exists(plot_filename): os.remove(plot_filename)
                    if not img_embedded: pdf.ln(10) # Add some space if image failed
                        
                    # Metrics (same as original)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Model Evaluation Metrics", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10)
                    metrics_dict = st.session_state.evaluation_metrics or {}
                    for key, value in metrics_dict.items():
                        val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else str(value)
                        pdf.cell(0, 8, txt=f"{key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    
                    # Forecast Table (same as original)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Forecast Data (First 10)", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=8); col_widths = [35, 35, 35, 35]
                    pdf.cell(col_widths[0], 7, txt="Date", border=1); pdf.cell(col_widths[1], 7, txt="Forecast", border=1); pdf.cell(col_widths[2], 7, txt="Lower CI", border=1); pdf.cell(col_widths[3], 7, txt="Upper CI", border=1, new_x="LMARGIN", new_y="NEXT")
                    forecast_table_df = st.session_state.forecast_results.head(10) if st.session_state.forecast_results is not None else pd.DataFrame()
                    for _, row in forecast_table_df.iterrows():
                        pdf.cell(col_widths[0], 6, txt=str(row["Date"].date()), border=1); pdf.cell(col_widths[1], 6, txt=f"{row['Forecast']:.2f}", border=1); pdf.cell(col_widths[2], 6, txt=f"{row['Lower_CI']:.2f}", border=1); pdf.cell(col_widths[3], 6, txt=f"{row['Upper_CI']:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    
                    # AI Report (same as original)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt=f"AI Report ({st.session_state.report_language})", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10)
                    # Handle potential encoding issues for multi_cell
                    ai_report_text = st.session_state.ai_report or "AI Report not available."
                    try:
                        pdf.multi_cell(0, 5, txt=ai_report_text)
                    except UnicodeEncodeError:
                        st.warning("Encoding issue detected in AI report for PDF. Using fallback encoding.")
                        pdf.multi_cell(0, 5, txt=ai_report_text.encode('latin-1', 'replace').decode('latin-1'))
                    pdf.ln(5)
                    
                    # Generate PDF bytes
                    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                    
                    # Add download button
                    st.download_button(label="Download PDF Now", data=pdf_output_bytes, file_name="deephydro_forecast_report.pdf", mime="application/octet-stream", key="pdf_download_final_btn", use_container_width=True)
                    st.success("PDF ready. Click download button above.")
                    if firebase_initialized: log_visitor_activity("Sidebar", "download_pdf_success")
                except Exception as pdf_err:
                    st.error(f"Failed to generate PDF: {pdf_err}")
                    if firebase_initialized: log_visitor_activity("Sidebar", "download_pdf_failure", details={"error": str(pdf_err)})
        else:
            st.error("Required data missing. Run forecast and generate AI report first.")

    st.header("4. AI Assistant")
    # --- Activate Chat Button (Use Original Access Check & Login) ---
    chat_button_label = "Deactivate Chat" if st.session_state.chat_active else "Activate Chat"
    activate_chat_button = st.button(chat_button_label, key="chat_ai_btn", disabled=not gemini_configured, use_container_width=True)
    
    if activate_chat_button:
        if st.session_state.chat_active: # Deactivating
            st.session_state.chat_active = False
            st.session_state.chat_history = [] # Clear history on deactivate
            if firebase_initialized: log_visitor_activity("Sidebar", "deactivate_chat")
            st.rerun()
        else: # Activating
            access_granted, message = check_feature_access() # Use original access check
            if access_granted:
                # Check if context is available before activating
                if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
                    st.session_state.chat_active = True
                    st.session_state.active_tab = 4 # Switch to chat tab
                    # Log successful access/usage (using original log function)
                    if firebase_initialized:
                        log_visitor_activity("Sidebar", "activate_chat", feature_used='AI Chat')
                    st.rerun()
                else:
                    st.error("Cannot activate chat. Run a successful forecast first to provide context.")
            else:
                # If access denied, show ORIGINAL Google login prompt
                st.warning(message)
                show_google_login() # Use original Google login function
                if firebase_initialized:
                     log_visitor_activity("Sidebar", "activate_chat_denied", feature_used='AI Chat')

    # --- About Us (Use New Design Structure) ---
    # Use markdown with classes for JS control
    st.markdown('<div class="about-us-header">👥 About Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="about-us-content">', unsafe_allow_html=True)
    st.markdown("Specializing in groundwater forecasting using AI.")
    st.markdown("**Contact:** [deephydro@example.com](mailto:deephydro@example.com)")
    st.markdown("© 2025 DeepHydro AI Team")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Admin Analytics Access (Use New Design Button, Original Logic) ---
    st.header("5. Admin")
    if st.button("Analytics Dashboard", key="admin_analytics_btn", use_container_width=True):
        if firebase_initialized: log_visitor_activity("Sidebar", "access_admin")
        st.session_state.active_tab = 5 # Switch to admin tab
        st.rerun()
        
    # --- Display Login Status / Button (Using Original Google OAuth) ---
    st.header("6. Authentication")
    show_google_login() # Call the original Google login/logout display function

# --- Main Application Area (Use New Layout, Integrate Original Logic) ---
st.title("DeepHydro AI Forecasting")

# Log main page view activity using original function
if firebase_initialized:
    # Avoid logging on every rerun if possible, maybe only on initial load?
    # This basic log will fire on each rerun currently.
    log_visitor_activity("Main Page", "view")

# App Introduction (Use New Design)
st.markdown('<div class="app-intro">', unsafe_allow_html=True)
st.markdown("""
### Welcome to DeepHydro AI Forecasting
Advanced groundwater forecasting platform using deep learning.
**Features:** LSTM forecasting, MC Dropout uncertainty, AI interpretation, Interactive visualization.
Upload your data using the sidebar to begin.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Handle data upload and cleaning (Keep Original Logic)
if uploaded_data_file is not None:
    # Check if it's a new file to avoid reprocessing on every rerun
    if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
        st.session_state.uploaded_data_filename = uploaded_data_file.name
        with st.spinner("Loading and cleaning data..."):
            cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
        if cleaned_df_result is not None:
            st.session_state.cleaned_data = cleaned_df_result
            # Reset results on new data upload
            st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
            st.session_state.training_history = None; st.session_state.ai_report = None
            st.session_state.chat_history = []; st.session_state.chat_active = False
            st.session_state.scaler_object = None; st.session_state.forecast_plot_fig = None
            st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH # Reset sequence length
            st.session_state.run_forecast_triggered = False
            st.session_state.active_tab = 0 # Go back to data preview
            if firebase_initialized: log_visitor_activity("Data Upload", "upload_success", details={"filename": uploaded_data_file.name})
            st.rerun()
        else:
            st.session_state.cleaned_data = None # Ensure cleaned_data is None on failure
            st.error("Data loading failed. Please check the file format and content.")
            if firebase_initialized: log_visitor_activity("Data Upload", "upload_failure", details={"filename": uploaded_data_file.name})
            # Clear filename so user can retry with same file name
            st.session_state.uploaded_data_filename = None 

# Define tabs (Use New Design Titles)
tab_titles = ["Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# --- Tab Content (Integrate Original Logic into New Tab Structure) ---

# Data Preview Tab (Tab 0)
with tabs[0]:
    if firebase_initialized: log_visitor_activity("Tab: Data Preview", "view")
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        # Use new design metrics layout
        col1, col2 = st.columns(2)
        with col1: 
            min_date = st.session_state.cleaned_data['Date'].min()
            max_date = st.session_state.cleaned_data['Date'].max()
            st.metric("Time Range", f"{min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}")
        with col2: 
            st.metric("Data Points", len(st.session_state.cleaned_data))
        # Plot historical data (using original plotting logic)
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], mode="lines", name="Level"))
        fig_data.update_layout(title="Historical Groundwater Levels", xaxis_title="Date", yaxis_title="Level", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=400)
        st.plotly_chart(fig_data, use_container_width=True)
    else:
        st.info("⬆️ Upload XLSX data using the sidebar.")

# Forecast Results Tab (Tab 1)
with tabs[1]:
    if firebase_initialized: log_visitor_activity("Tab: Forecast Results", "view")
    st.header("Forecast Results")
    # Check if forecast ran successfully (using original session state keys)
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        # Display plot if available
        if st.session_state.forecast_plot_fig is not None:
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else: 
            # Try to recreate plot if figure is missing but data exists
            if st.session_state.cleaned_data is not None:
                 st.warning("Forecast plot figure missing, attempting to recreate...")
                 try:
                     recreated_fig = create_forecast_plot(st.session_state.cleaned_data, st.session_state.forecast_results)
                     st.plotly_chart(recreated_fig, use_container_width=True)
                 except Exception as plot_err:
                     st.error(f"Could not recreate plot: {plot_err}")
            else:
                st.warning("Forecast plot unavailable (figure missing and no historical data).")
        st.subheader("Forecast Data Table")
        st.dataframe(st.session_state.forecast_results, use_container_width=True)
    elif st.session_state.run_forecast_triggered: 
        st.warning("Forecast run attempted, but no results available. Check sidebar messages or previous steps for errors.")
    else: 
        st.info("⬅️ Run a forecast using the sidebar to see results here.")

# Model Evaluation Tab (Tab 2)
with tabs[2]:
    if firebase_initialized: log_visitor_activity("Tab: Model Evaluation", "view")
    st.header("Model Evaluation")
    # Check if evaluation metrics exist (using original session state keys)
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (Validation/Pseudo-Validation)")
        col1, col2, col3 = st.columns(3)
        # Safely get metrics
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan)
        mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan)
        mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        # Format metrics for display
        col1.metric("RMSE", f"{rmse_val:.4f}" if pd.notna(rmse_val) else "N/A")
        col2.metric("MAE", f"{mae_val:.4f}" if pd.notna(mae_val) else "N/A")
        col3.metric("MAPE", f"{mape_val:.2f}%" if pd.notna(mape_val) and np.isfinite(mape_val) else ("N/A" if pd.isna(mape_val) else "Inf"))
        
        st.subheader("Training Loss (if applicable)")
        # Display loss plot if training history exists
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: 
            st.info("No training history available (used pre-trained model or training failed/skipped).")
    elif st.session_state.run_forecast_triggered: 
        st.warning("Forecast run attempted, but no evaluation metrics available.")
    else: 
        st.info("⬅️ Run a forecast using the sidebar to see evaluation metrics.")

# AI Report Tab (Tab 3)
with tabs[3]:
    if firebase_initialized: log_visitor_activity("Tab: AI Report", "view")
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: 
        st.warning("AI features disabled. Configure Gemini API Key in environment variables.")
    # Check if AI report exists (using original session state key)
    if st.session_state.ai_report: 
        # Use new design chat message styling
        st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    else: 
        st.info("⬅️ Click 'Generate AI Report' in the sidebar after running a forecast.")

# AI Chatbot Tab (Tab 4)
with tabs[4]:
    if firebase_initialized: log_visitor_activity("Tab: AI Chatbot", "view")
    st.header("AI Chatbot Assistant")
    if not gemini_configured: 
        st.warning("AI features disabled. Configure Gemini API Key.")
    # Check if chat is active (using original session state key)
    elif st.session_state.chat_active:
        # Check if necessary context exists
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("Chat activated. Ask questions about the current forecast results below.")
            # Use container for scrollable chat history (from new design)
            chat_container = st.container(height=400) 
            with chat_container:
                # Display chat history using new design styling
                for sender, message in st.session_state.chat_history:
                    msg_class = "user-message" if sender == "User" else "ai-message"
                    # Ensure message is string and escape HTML potentially within the message
                    escaped_message = html.escape(str(message))
                    st.markdown(f'<div class="chat-message {msg_class}">{escaped_message}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
            
            # Chat input
            user_input = st.chat_input("Ask the AI assistant about the results...")
            if user_input:
                if firebase_initialized: log_visitor_activity("Chat", "send_message")
                # Append user message
                st.session_state.chat_history.append(("User", user_input))
                # Display user message immediately (will be redrawn on rerun anyway)
                # with chat_container:
                #      st.markdown(f'<div class="chat-message user-message">{html.escape(user_input)}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
                
                # Get AI response using original function
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history,
                        st.session_state.cleaned_data, st.session_state.forecast_results, 
                        st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                # Append AI response
                st.session_state.chat_history.append(("AI", ai_response))
                # Rerun to display the full history including the new AI response
                st.rerun()
        else:
            # If context is missing after activation, show warning and deactivate
            st.warning("Context (data/forecast/metrics) is missing. Please run a successful forecast first.")
            st.session_state.chat_active = False # Deactivate if context is missing
            # No rerun here, let the user reactivate via sidebar if they fix context
    else:
        st.info("⬅️ Click 'Activate Chat' in the sidebar after running a forecast to enable the AI assistant." if gemini_configured else "AI Chat disabled.")

# Admin Analytics Tab (Tab 5)
with tabs[5]:
    if firebase_initialized: log_visitor_activity("Tab: Admin Analytics", "view")
    # Render admin dashboard using the original logic wrapped in the function
    render_admin_analytics()

# Add JavaScript at the end of the script execution
add_javascript_functionality()

# --- Final Checks ---
# Display config load error if it occurred
if config_load_error:
    st.error(f"Configuration Error: {config_load_error}")

