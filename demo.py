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
# Removed dotenv import as config is now from APP_CONFIG_JSON
import hashlib
import streamlit.components.v1 as components # Import components
from streamlit_oauth import OAuth2Component # <-- Import for Google OAuth

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 3
STANDARD_MODEL_PATH = "standard_model.h5"
STANDARD_MODEL_SEQUENCE_LENGTH = 60 # Default, will be updated if standard model loads

# --- Load Configuration from Environment Variable --- 
APP_CONFIG = {}
firebase_initialized = False
google_oauth_configured = False
gemini_configured = False
admin_password_configured = False

config_json_str = os.getenv("APP_CONFIG_JSON")
if config_json_str:
    try:
        APP_CONFIG = json.loads(config_json_str)
        st.success("Configuration loaded successfully from APP_CONFIG_JSON.")
        # Check for essential config sections
        if "firebase_service_account" not in APP_CONFIG:
            st.warning("Firebase configuration (\"firebase_service_account\") missing in APP_CONFIG_JSON.")
        if "google_oauth" not in APP_CONFIG or not all(k in APP_CONFIG["google_oauth"] for k in ["client_id", "client_secret", "redirect_uri"]):
            st.warning("Google OAuth configuration (\"google_oauth\") incomplete or missing in APP_CONFIG_JSON.")
        else:
            google_oauth_configured = True
        if "google_api_key" not in APP_CONFIG:
            st.warning("Google API Key (\"google_api_key\") missing in APP_CONFIG_JSON.")
        if "admin_password" not in APP_CONFIG:
             st.warning("Admin Password (\"admin_password\") missing in APP_CONFIG_JSON.")
        else:
            admin_password_configured = True
            
    except json.JSONDecodeError as e:
        st.error(f"Error decoding APP_CONFIG_JSON: {e}. Check the JSON format in the environment variable.")
        APP_CONFIG = {} # Reset to empty dict on error
else:
    st.error("Environment variable APP_CONFIG_JSON not found. Application configuration is missing.")

# --- Firebase Configuration --- 
def initialize_firebase():
    """
    Initialize Firebase using credentials from the loaded APP_CONFIG.
    """
    global firebase_initialized # Allow modification of the global flag
    if firebase_admin._apps: # Already initialized
        return True
        
    if "firebase_service_account" not in APP_CONFIG:
        st.warning("Firebase Service Account details not found in configuration. Analytics disabled.")
        firebase_initialized = False
        return False
        
    try:
        cred_dict = APP_CONFIG["firebase_service_account"]
        # Basic validation of the cred_dict structure
        if not all(k in cred_dict for k in ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"]):
             st.error("Firebase Service Account JSON in configuration is incomplete.")
             firebase_initialized = False
             return False
             
        cred = credentials.Certificate(cred_dict)
        
        # Get Database URL: Prioritize from config, fallback to default based on project_id
        firebase_url = APP_CONFIG.get("firebase_database_url") # Optional key in config
        if not firebase_url or firebase_url == "OPTIONAL_YOUR_FIREBASE_DB_URL":
            project_id = cred_dict.get("project_id")
            if project_id:
                firebase_url = f"https://{project_id}-default-rtdb.firebaseio.com/"
                st.info(f"Firebase Database URL not explicitly set, using default: {firebase_url}")
            else:
                st.error("Cannot determine Firebase Database URL: 'firebase_database_url' not in config and 'project_id' missing in service account.")
                firebase_initialized = False
                return False
                
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_url
        })
        st.success("Firebase initialized successfully.") 
        firebase_initialized = True
        return True
        
    except ValueError as e:
        st.error(f"Error initializing Firebase with provided credentials: {e}. Check the service account details in the configuration.")
        firebase_initialized = False
        return False
    except Exception as e:
        st.warning(f"Firebase initialization error: {e}. Analytics and usage tracking may be disabled.")
        firebase_initialized = False
        return False

# --- User Identification & Tracking --- 
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

def update_firebase_profile_on_login(email):
    """Update Firebase profile when a user logs in via Google OAuth."""
    if not firebase_initialized or not email:
        return
    try:
        # Ensure user_id is URL-safe (Firebase keys cannot contain ., $, #, [, ], /)
        safe_email_key = email.replace('.', '_dot_').replace('@', '_at_') # Basic replacement
        ref = db.reference(f"users/{safe_email_key}")
        profile = ref.get()
        now_iso = datetime.datetime.now().isoformat()
        
        update_data = {
            "is_authenticated": True,
            "last_login_google": now_iso,
            "user_id": email, # Store the original email
            "email": email # Explicitly store email field
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
            st.session_state.user_profile = ref.get() # Refresh profile
            
        log_visitor_activity("Authentication", action="google_login_success", details={"email": email})
            
    except Exception as e:
        st.error(f"Firebase error updating profile after Google login for {email}: {e}")
        log_visitor_activity("Authentication", action="google_login_firebase_update_fail", details={"email": email, "error": str(e)})

def get_or_create_user_profile(user_id):
    """Get user profile from Firebase or create a new one."""
    if not firebase_initialized:
        return None, False
    
    try:
        # Ensure user_id is URL-safe for Firebase path
        safe_user_id_key = user_id.replace('.', '_dot_').replace('@', '_at_') if '@' in user_id else user_id
        ref = db.reference(f"users/{safe_user_id_key}")
        profile = ref.get()
        is_new_user = False
        now_iso = datetime.datetime.now().isoformat()
        current_auth_status = st.session_state.get("auth_status") == "authenticated"
        current_email = st.session_state.get("user_email")

        if profile is None:
            is_new_user = True
            profile = {
                "user_id": user_id, # Store original ID
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
                     profile["email"] = profile.get("email") # Keep last known
                     
                ref.update({
                    "visit_count": profile["visit_count"], 
                    "last_visit": profile["last_visit"],
                    "is_authenticated": profile["is_authenticated"],
                    "email": profile.get("email")
                })
                st.session_state.session_visit_logged = True
            
        return profile, is_new_user
    except Exception as e:
        st.warning(f"Firebase error getting/creating user profile for {user_id}: {e}")
        return None, False

def increment_feature_usage(user_id):
    """Increment the feature usage count for the user in Firebase."""
    if not firebase_initialized:
        return False
    
    try:
        safe_user_id_key = user_id.replace('.', '_dot_').replace('@', '_at_') if '@' in user_id else user_id
        ref = db.reference(f"users/{safe_user_id_key}/feature_usage_count")
        current_count = ref.get() or 0
        ref.set(current_count + 1)
        if "user_profile" in st.session_state and st.session_state.user_profile:
            st.session_state.user_profile["feature_usage_count"] = current_count + 1
        return True
    except Exception as e:
        st.warning(f"Firebase error incrementing usage count for {user_id}: {e}")
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
            # If Firebase isn't working, maybe allow limited access anyway?
            # For now, deny if profile can't be fetched and not authenticated.
            st.warning("Could not retrieve user profile (Firebase might be unavailable). Feature access limited.")
            return False, "Cannot verify usage limit. Access denied."

    usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
    
    if usage_count < ADVANCED_FEATURE_LIMIT:
        return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
    else:
        return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in to continue."

# --- Google OAuth Implementation --- 
# Get credentials from loaded config
CLIENT_ID = APP_CONFIG.get("google_oauth", {}).get("client_id")
CLIENT_SECRET = APP_CONFIG.get("google_oauth", {}).get("client_secret")
REDIRECT_URI = APP_CONFIG.get("google_oauth", {}).get("redirect_uri")

# OAuth endpoints remain constant
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

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
        user_info = response.json()
        return user_info
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching user info from Google: {e}")
        return None
    except Exception as e:
         st.error(f"Unexpected error fetching user info: {e}")
         return None

def show_google_login():
    """Handles the Google OAuth login flow using streamlit-oauth."""
    if not google_oauth_configured: # Check if config was loaded correctly
        st.sidebar.error("Google Sign-In is not configured correctly. Please check the application configuration.")
        # Optionally log this specific config issue if Firebase is up
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
                    update_firebase_profile_on_login(user_info.get("email")) # Update profile after successful login
                    st.success(f"Logged in as {st.session_state.user_name} ({st.session_state.user_email}).")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("Login successful, but failed to retrieve user information from Google.")
                    if "token" in st.session_state: del st.session_state.token 
                    if firebase_initialized: log_visitor_activity("Authentication", action="google_login_fail_userinfo")
    else:
        # User is logged in (token exists)
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
                 if firebase_initialized: log_visitor_activity("Authentication", action="google_reauth_fail_userinfo")
                 st.rerun()
                 
        if st.session_state.get("user_email"):
            st.sidebar.success(f"Logged in as: {st.session_state.user_name} ({st.session_state.user_email})")
            if st.sidebar.button("Logout", key="google_logout"):
                if firebase_initialized: log_visitor_activity("Authentication", action="google_logout", details={"email": st.session_state.user_email})
                if "token" in st.session_state: del st.session_state.token
                st.session_state.auth_status = "anonymous"
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.persistent_user_id = None
                st.rerun()

# --- Visitor Analytics Functions --- 
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
        profile, _ = get_or_create_user_profile(user_id)
        
        should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"]
        access_granted, _ = check_feature_access()
        action_status = "success"
        
        if should_increment:
            if access_granted:
                increment_feature_usage(user_id)
            else:
                action_status = "denied_limit_reached"

        full_action_name = f"{action}_{action_status}"
        safe_user_id_key = user_id.replace('.', '_dot_').replace('@', '_at_') if '@' in user_id else user_id
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
            # Ensure details are JSON serializable (basic check)
            try:
                json.dumps(details)
                log_data["details"] = details
            except TypeError:
                 log_data["details"] = {"error": "Details not JSON serializable"}
            
        ref.child(log_id).set(log_data)

    except Exception as e:
        # print(f"Error logging visitor activity: {e}") # Optional debug
        pass

def fetch_visitor_logs():
    """Fetch visitor logs from Firebase."""
    if not firebase_initialized:
        st.warning("Firebase not initialized. Cannot fetch logs.")
        return pd.DataFrame()
    
    try:
        ref = db.reference("visitors_log")
        visitors_data = ref.get()
        
        if not visitors_data:
            return pd.DataFrame()
            
        visitors_list = []
        for log_id, data in visitors_data.items():
            if isinstance(data, dict): # Basic check if data is a dictionary
                data["log_id"] = log_id
                if "details" in data and isinstance(data["details"], dict):
                    for k, v in data["details"].items():
                        # Sanitize key names for DataFrame columns
                        safe_key = f"detail_{str(k).replace('.', '_').replace('$', '_')}"
                        data[safe_key] = v
                    del data["details"]
                visitors_list.append(data)
            # else: skip malformed log entry
            
        df = pd.DataFrame(visitors_list)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce') # Handle potential errors
            df = df.sort_values("timestamp", ascending=False, na_position='last')
        
        return df
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
    """Create visualizations of visitor data."""
    if visitor_df.empty:
        return []
    
    figures = []
    df = visitor_df.copy()
    
    try:
        # Ensure timestamp is datetime
        if "timestamp" in df.columns and df["timestamp"].dtype != '<M8[ns]':
             df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True) # Drop rows where conversion failed
        if df.empty: return []
        
        df["date"] = df["timestamp"].dt.date
        
        # 1. Daily visitors
        if "persistent_user_id" in df.columns:
            daily_visitors = df.groupby("date")["persistent_user_id"].nunique().reset_index(name="unique_users")
            daily_visitors["date"] = pd.to_datetime(daily_visitors["date"])
            fig1 = px.line(daily_visitors, x="date", y="unique_users", title="Daily Unique Visitors", labels={"unique_users": "Unique Users", "date": "Date"})
            figures.append(fig1)
        
        # 2. Activity Counts
        if "action" in df.columns:
            action_counts = df[~df["action"].str.contains("page_view", case=False, na=False)]["action"].value_counts().reset_index()
            action_counts.columns = ["action", "count"]
            fig2 = px.bar(action_counts.head(20), x="action", y="count", title="Top 20 Activity Counts (Excluding Page Views)", labels={"count": "Count", "action": "Action"})
            figures.append(fig2)

        # 3. Auth Status
        if "persistent_user_id" in df.columns and "is_authenticated" in df.columns:
            latest_status = df.sort_values("timestamp").groupby("persistent_user_id")["is_authenticated"].last().reset_index()
            auth_counts = latest_status["is_authenticated"].value_counts().reset_index()
            auth_counts.columns = ["is_authenticated", "count"]
            auth_counts["status"] = auth_counts["is_authenticated"].map({True: "Authenticated", False: "Anonymous"})
            fig3 = px.pie(auth_counts, values="count", names="status", title="User Authentication Status (Latest Known)")
            figures.append(fig3)

        # 4. Activity Heatmap
        try:
            df["hour"] = df["timestamp"].dt.hour
            df["weekday"] = df["timestamp"].dt.day_name()
            hourly_activity = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
            all_hours = list(range(24))
            all_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = pd.MultiIndex.from_product([all_weekdays, all_hours], names=["weekday", "hour"]).to_frame(index=False)
            heatmap_data = pd.merge(heatmap_data, hourly_activity, on=["weekday", "hour"], how="left").fillna(0)
            heatmap_pivot = heatmap_data.pivot(index="weekday", columns="hour", values="count").reindex(all_weekdays)
            
            fig4 = px.imshow(heatmap_pivot, 
                           labels=dict(x="Hour", y="Day", color="Count"),
                           x=[str(h) for h in all_hours], y=all_weekdays,
                           title="User Activity Heatmap",
                           color_continuous_scale=px.colors.sequential.Viridis)
            fig4.update_xaxes(side="bottom")
            figures.append(fig4)
        except Exception as e_heatmap:
            st.warning(f"Could not generate activity heatmap: {e_heatmap}")
            
    except Exception as e_charts:
        st.error(f"Error creating visitor charts: {e_charts}")
        
    return figures

# --- PDF Generation --- (No changes needed here)
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
            temp_img_path = f"temp_plot_{uuid.uuid4()}.png" # Unique temp name
            with open(temp_img_path, "wb") as f: f.write(plot_bytes)
            self.image(temp_img_path, x=10, w=self.w - 20)
            os.remove(temp_img_path)
            self.ln(5)
        except Exception as e:
            self.chapter_body(f"[Error adding plot: {e}]")
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{self.alias_nb_pages()}", 0, 0, "C") # Added total pages

def create_pdf_report(forecast_fig, loss_fig, metrics, ai_report_text):
    pdf = PDF()
    pdf.alias_nb_pages() # Enable total page count
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
    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        st.error(f"Error generating PDF bytes: {e}"); return None

# --- Custom CSS --- (No changes needed here)
def apply_custom_css():
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keep existing CSS

# --- JavaScript --- (No changes needed here)
def add_javascript_functionality():
     st.markdown("""<script>...</script>""", unsafe_allow_html=True) # Keep existing JS

# --- Page Configuration & Initialization --- 
#st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css() # Apply CSS (assuming it's defined above or unchanged)
add_javascript_functionality() # Add JS (assuming it's defined above or unchanged)

# --- Initialize Firebase FIRST --- 
# Firebase needs to be initialized before other dependent functions are called
initialize_firebase() 

# --- Capture User Agent --- 
def capture_user_agent():
    if "user_agent" not in st.session_state:
        try:
            component_value = components.html("""<script>...</script>""", height=0, key="user_agent_capture") # Keep existing JS
            if component_value: st.session_state.user_agent = component_value
            elif "user_agent_capture" in st.session_state and st.session_state.user_agent_capture: st.session_state.user_agent = st.session_state.user_agent_capture
            else: st.session_state.user_agent = "Unknown (Capture Pending)"
        except Exception: st.session_state.user_agent = "Unknown (Capture Failed)"
capture_user_agent()

# --- Initialize Session State --- 
def initialize_session_state():
    # Update default sequence length based on loaded standard model if possible
    global STANDARD_MODEL_SEQUENCE_LENGTH 
    if os.path.exists(STANDARD_MODEL_PATH):
        try:
            _model = load_model(STANDARD_MODEL_PATH, compile=False)
            STANDARD_MODEL_SEQUENCE_LENGTH = _model.input_shape[1]
            st.session_state.standard_model = _model # Load into session state here
            st.session_state.standard_model_seq_len = STANDARD_MODEL_SEQUENCE_LENGTH
        except Exception as e:
            st.warning(f"Could not load standard model from {STANDARD_MODEL_PATH} to infer sequence length: {e}. Using default {STANDARD_MODEL_SEQUENCE_LENGTH}.")
            if firebase_initialized: log_visitor_activity("Model Handling", action="load_standard_model_fail", details={"path": STANDARD_MODEL_PATH, "error": str(e)})
    else:
        st.warning(f"Standard model file not found: {STANDARD_MODEL_PATH}. Standard model option disabled.")
        
    defaults = {
        "df": None, "forecast_df": None, "metrics": None, "forecast_fig": None, 
        "loss_fig": None, "model_trained": False, "selected_model_type": "Standard", 
        "custom_model": None, "custom_model_seq_len": None, 
        "standard_model": st.session_state.get("standard_model"), # Use pre-loaded model if available
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

# Initialize user profile (depends on Firebase init and session state init)
if "user_profile" not in st.session_state or st.session_state.user_profile is None:
    if firebase_initialized:
        user_id = get_persistent_user_id()
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
    else:
        st.session_state.user_profile = None

# --- Gemini API Configuration --- 
GEMINI_API_KEY = APP_CONFIG.get("google_api_key")
gemini_model_report = None
gemini_model_chat = None

if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSy..." and GEMINI_API_KEY != "Gemini_api_key": # Check for actual key
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
        gemini_model_report = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
        gemini_model_chat = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
        gemini_configured = True
        st.success("Gemini API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
        gemini_configured = False
else:
    st.warning("Gemini API Key (google_api_key) not found in configuration or is a placeholder. AI features disabled.")
    gemini_configured = False

# --- Helper Functions (Data Loading, Model Building, Prediction) --- 
@st.cache_data # Keep caching for data loading
def load_and_clean_data(uploaded_file_content):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        # ... (rest of the function remains the same) ...
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
    except Exception as e: st.error(f"An unexpected error during data loading/cleaning: {e}"); return None

def create_sequences(data, sequence_length):
    X, y = [], []
    if len(data) <= sequence_length:
        return np.array(X), np.array(y) # Return empty arrays if not enough data
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource # Keep caching for model loading
def load_keras_model_from_file(uploaded_file_obj, model_name_for_log):
    # Use a unique temp name based on file content hash? Or just UUID?
    temp_model_path = f"temp_model_{uuid.uuid4()}.h5"
    try:
        with open(temp_model_path, "wb") as f: f.write(uploaded_file_obj.getbuffer())
        model = load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded {model_name_for_log}. Inferred sequence length: {sequence_length}")
        if firebase_initialized: log_visitor_activity("Model Handling", action="load_custom_model_success", details={"model_name": model_name_for_log, "sequence_length": sequence_length})
        return model, sequence_length
    except Exception as e: 
        st.error(f"Error loading Keras model {model_name_for_log}: {e}")
        if firebase_initialized: log_visitor_activity("Model Handling", action="load_custom_model_fail", details={"model_name": model_name_for_log, "error": str(e)})
        return None, None
    finally: 
        if os.path.exists(temp_model_path): 
            try: os.remove(temp_model_path)
            except OSError: pass # Ignore error if file cannot be removed

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([
        LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), 
        Dropout(0.5), 
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    # ... (rest of the function remains the same) ...
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    # Ensure model allows training=True call
    try:
        _ = model(current_sequence, training=True) # Test call
    except TypeError:
        st.warning("Model does not support 'training' argument for prediction. Uncertainty estimation might be less reliable.")
        @tf.function
        def predict_step_internal(inp): return model(inp) # Fallback
    else:
        @tf.function
        def predict_step_internal(inp): return model(inp, training=True)
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step_internal(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            # Ensure sequence update dimension matches
            new_entry = np.array([[next_pred_scaled]]).reshape(1, 1, 1) if len(temp_sequence.shape) == 3 else np.array([next_pred_scaled]).reshape(1, 1)
            temp_sequence = np.append(temp_sequence[:, 1:, :], new_entry, axis=1)
            
        all_predictions.append(iteration_predictions_scaled)
        progress_bar.progress((i + 1) / n_iterations)
        status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
        
    progress_bar.empty(); status_text.empty()
    predictions_array_scaled = np.array(all_predictions)
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    ci_multiplier = 1.96 # Standard 95% CI
    
    # Inverse transform carefully
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    lower_bound_scaled = (mean_preds_scaled - ci_multiplier * std_devs_scaled).reshape(-1, 1)
    upper_bound_scaled = (mean_preds_scaled + ci_multiplier * std_devs_scaled).reshape(-1, 1)
    lower_bound = scaler.inverse_transform(lower_bound_scaled).flatten()
    upper_bound = scaler.inverse_transform(upper_bound_scaled).flatten()
    
    # Ensure minimum uncertainty range (optional, consider if needed)
    # min_uncertainty_percent = 0.05
    # ... (logic as before if desired) ...
    
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    # ... (rest of the function remains the same) ...
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    # Avoid division by zero in MAPE
    mask = y_true != 0
    if np.sum(mask) == 0: # All true values are zero
        mape = 0.0 if np.allclose(y_pred, 0) else np.inf
    else:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions --- (No changes needed here)
def create_forecast_plot(historical_df, forecast_df):
    # ... (function remains the same) ...
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
    # Ensure CI columns exist before plotting
    if "Upper_CI" in forecast_df.columns and "Lower_CI" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=True)) # Show legend for CI area
    fig.update_layout(title="Groundwater Level: Historical Data & LSTM Forecast", xaxis_title="Date", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_loss_plot(history_dict):
    # ... (function remains the same) ...
    if not history_dict or not isinstance(history_dict, dict) or "loss" not in history_dict or "val_loss" not in history_dict:
        fig = go.Figure()
        fig.update_layout(title="No Training History Available", xaxis_title="Epoch", yaxis_title="Loss")
        fig.add_annotation(text="Training history not found or incomplete.", showarrow=False)
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history_dict["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(y=history_dict["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified", template="plotly_white")
    return fig

# --- AI Report Generation --- 
def generate_ai_report(data_summary, forecast_summary, metrics, model_details):
    if not gemini_configured or gemini_model_report is None:
         return "AI report generation disabled. Gemini API not configured or model failed to load."
    try:
        prompt = f"""...""" # Keep existing prompt
        response = gemini_model_report.generate_content(prompt)
        # Add basic check for response content
        if response and response.text:
            return response.text
        else:
            return "AI report generation failed: Empty response from model."
    except Exception as e: 
        if firebase_initialized: log_visitor_activity("AI Report", action="generate_report_api_fail", details={"error": str(e)})
        return f"Error generating AI report: {e}"

# --- AI Chat Functionality --- 
def initialize_chat():
    if not gemini_configured or gemini_model_chat is None:
        st.warning("Chat disabled: Gemini API not configured or model failed to load.")
        return
    if "messages" not in st.session_state: st.session_state.messages = []
    # Ensure chat model is initialized if not already
    if "chat_model" not in st.session_state or st.session_state.chat_model is None:
        try:
            st.session_state.chat_model = gemini_model_chat.start_chat(history=[])
        except Exception as e:
            st.error(f"Failed to initialize chat model: {e}")
            st.session_state.chat_model = None # Ensure it's None on failure

def display_chat_history():
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "ai-message"
        # Basic escaping for HTML display
        content_escaped = message["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="chat-message {role_class}">{content_escaped}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)

def handle_chat_input(prompt):
    if not gemini_configured or st.session_state.get("chat_model") is None:
        st.error("Chat disabled. Gemini API not configured or chat model failed to initialize."); return
    if not prompt: return
    
    if firebase_initialized: log_visitor_activity("AI Chat", action="send_chat_message")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message immediately (escaped)
    prompt_escaped = prompt.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="chat-message user-message">{prompt_escaped}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.chat_model.send_message(prompt)
        # Add AI response (escaped)
        response_text = response.text if response and response.text else "AI returned an empty response."
        st.session_state.messages.append({"role": "model", "content": response_text})
        response_escaped = response_text.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="chat-message ai-message">{response_escaped}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
        
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "model", "content": error_msg})
        st.error(error_msg)
        if firebase_initialized: log_visitor_activity("AI Chat", action="chat_error", details={"error": str(e)})

# --- Streamlit App UI --- 

# --- Sidebar --- 
#st.sidebar.image("logo.png", width=100) # Assuming logo.png exists
st.sidebar.title("DeepHydro AI")

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
        # Reset dependent states
        st.session_state.forecast_df = None; st.session_state.metrics = None
        st.session_state.forecast_fig = None; st.session_state.ai_report = None
        st.session_state.chat_active = False; st.session_state.messages = []
        if firebase_initialized: log_visitor_activity("Sidebar", action="upload_data", details={"filename": uploaded_file.name})
        st.rerun()

# Model Selection
model_options = ["Standard"] if st.session_state.standard_model else []
model_options.extend(["Train New", "Upload Custom (.h5)"])

# Determine default index carefully
default_model_index = 0
if st.session_state.selected_model_type == "Train New":
    default_model_index = model_options.index("Train New") if "Train New" in model_options else 0
elif st.session_state.selected_model_type == "Upload Custom (.h5)":
    default_model_index = model_options.index("Upload Custom (.h5)") if "Upload Custom (.h5)" in model_options else 0

model_choice = st.sidebar.radio(
    "Select Model", 
    options=model_options,
    key="model_selector", 
    index=default_model_index,
    disabled=not model_options # Disable if no models available
)

if model_choice and model_choice != st.session_state.selected_model_type:
    if firebase_initialized: log_visitor_activity("Sidebar", action="select_model_type", details={"model_type": model_choice})
    st.session_state.selected_model_type = model_choice
    # Reset dependent states
    st.session_state.forecast_df = None; st.session_state.metrics = None
    st.session_state.forecast_fig = None; st.session_state.ai_report = None
    st.session_state.chat_active = False; st.session_state.messages = []
    st.rerun()

custom_model_file = None
if st.session_state.selected_model_type == "Upload Custom (.h5)":
    custom_model_file = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"], key="model_uploader")
    if custom_model_file is not None:
        if custom_model_file.name != st.session_state.get("custom_model_file_name"):
            # Load model and update state
            loaded_model, seq_len = load_keras_model_from_file(custom_model_file, custom_model_file.name)
            if loaded_model and seq_len:
                st.session_state.custom_model = loaded_model
                st.session_state.custom_model_seq_len = seq_len
                st.session_state.custom_model_file_name = custom_model_file.name
                st.rerun()
            else:
                # Handle loading failure (error shown in function)
                st.session_state.custom_model = None
                st.session_state.custom_model_seq_len = None
                st.session_state.custom_model_file_name = None # Reset filename on failure

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

# Determine if forecast can run
forecast_ready = False
model_available = False
if st.session_state.selected_model_type == "Standard" and st.session_state.standard_model:
    model_available = True
elif st.session_state.selected_model_type == "Upload Custom (.h5)" and st.session_state.custom_model:
    model_available = True
elif st.session_state.selected_model_type == "Train New" and st.session_state.model_trained and st.session_state.custom_model:
    model_available = True

if st.session_state.df is not None and model_available and can_access_forecast:
    forecast_ready = True

run_forecast_disabled_reason = ""
if st.session_state.df is None: run_forecast_disabled_reason = "Upload data first."
elif not model_available: run_forecast_disabled_reason = "Select/Train/Upload a valid model."
elif not can_access_forecast: run_forecast_disabled_reason = reason_forecast

if st.sidebar.button("Run Forecast", key="run_forecast_btn", disabled=not forecast_ready, help=run_forecast_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast", feature_used="Forecast", details={"model_type": st.session_state.selected_model_type, "horizon": forecast_horizon, "mc_iterations": mc_iterations})
    
    model_to_use, sequence_length, model_type_log = None, None, st.session_state.selected_model_type
    if st.session_state.selected_model_type == "Standard":
        model_to_use, sequence_length = st.session_state.standard_model, st.session_state.standard_model_seq_len
    elif st.session_state.selected_model_type == "Upload Custom (.h5)":
        model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len
    elif st.session_state.selected_model_type == "Train New":
        model_to_use, sequence_length = st.session_state.custom_model, st.session_state.custom_model_seq_len
        model_type_log = "Trained New"
        
    if model_to_use is None or sequence_length is None:
        st.error("Model not available. Cannot run forecast.")
        if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Model unavailable"})
        st.stop()

    if len(st.session_state.df) <= sequence_length:
        st.error(f"Not enough data ({len(st.session_state.df)} points) for sequence length ({sequence_length}).")
        if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Insufficient data"})
        st.stop()
        
    with st.spinner("Forecasting..."):
        try:
            df_forecast_input = st.session_state.df.copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_forecast_input["Level"].values.reshape(-1, 1))
            last_sequence_scaled = scaled_data[-sequence_length:]
            
            mean_preds, lower_bound, upper_bound = predict_with_dropout_uncertainty(
                model_to_use, last_sequence_scaled, forecast_horizon, mc_iterations, scaler, sequence_length
            )
            
            last_date = df_forecast_input["Date"].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            
            st.session_state.forecast_df = pd.DataFrame({
                "Date": forecast_dates, "Forecast": mean_preds,
                "Lower_CI": lower_bound, "Upper_CI": upper_bound
            })
            
            # Use training metrics if available, otherwise indicate N/A
            st.session_state.metrics = st.session_state.get("training_metrics", {"Info": "Metrics generated during training"})
            st.session_state.forecast_fig = create_forecast_plot(df_forecast_input, st.session_state.forecast_df)
            
            st.success("Forecast completed!")
            if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_success", feature_used="Forecast", details={"model_type": model_type_log, "horizon": forecast_horizon})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during forecasting: {e}")
            if firebase_initialized: log_visitor_activity("Sidebar", action="run_forecast_fail", feature_used="Forecast", details={"model_type": model_type_log, "error": str(e)})

# Generate AI Report Button
gen_report_ready = st.session_state.forecast_df is not None and gemini_configured and can_access_report
gen_report_disabled_reason = ""
if st.session_state.forecast_df is None: gen_report_disabled_reason = "Run forecast first."
elif not gemini_configured: gen_report_disabled_reason = "Gemini API not configured."
elif not can_access_report: gen_report_disabled_reason = reason_report

if st.sidebar.button("Generate AI Report", key="gen_report_btn", disabled=not gen_report_ready, help=gen_report_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report", feature_used="AI Report")
    with st.spinner("Generating AI analysis..."):
        try:
            # Prepare summaries and details for the report
            data_summary = f"Historical data: {len(st.session_state.df)} points from {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. Avg Level: {st.session_state.df['Level'].mean():.2f}."
            forecast_summary = f"Forecast: {forecast_horizon} steps. Range: {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. 95% CI: [{st.session_state.forecast_df['Lower_CI'].min():.2f}, {st.session_state.forecast_df['Upper_CI'].max():.2f}]."
            metrics_summary = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in st.session_state.metrics.items()]) if st.session_state.metrics and "Info" not in st.session_state.metrics else "N/A (Generated during training)"
            
            seq_len_detail = "N/A"
            if st.session_state.selected_model_type == "Standard": seq_len_detail = st.session_state.standard_model_seq_len
            elif st.session_state.custom_model_seq_len: seq_len_detail = st.session_state.custom_model_seq_len
            elif st.session_state.trained_model_seq_len: seq_len_detail = st.session_state.trained_model_seq_len
                
            model_details = {
                "type": st.session_state.selected_model_type,
                "sequence_length": seq_len_detail,
                "forecast_horizon": forecast_horizon,
                "mc_iterations_for_ci": mc_iterations
            }
            
            st.session_state.ai_report = generate_ai_report(data_summary, forecast_summary, metrics_summary, model_details)
            st.success("AI Report generated!")
            if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report_success", feature_used="AI Report")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate AI report: {e}")
            if firebase_initialized: log_visitor_activity("Sidebar", action="generate_report_fail", feature_used="AI Report", details={"error": str(e)})

# Download PDF Report Button
pdf_bytes = None
pdf_ready = st.session_state.forecast_fig or st.session_state.ai_report
if pdf_ready:
    try:
        pdf_bytes = create_pdf_report(
            st.session_state.forecast_fig, st.session_state.loss_fig, 
            st.session_state.metrics, st.session_state.ai_report
        )
    except Exception as e:
        st.sidebar.warning(f"Could not prepare PDF: {e}")
        if firebase_initialized: log_visitor_activity("Sidebar", action="prepare_pdf_fail", details={"error": str(e)})

st.sidebar.download_button(
    label="Download PDF Report",
    data=pdf_bytes if pdf_bytes else b"",
    file_name="DeepHydro_AI_Forecast_Report.pdf",
    mime="application/pdf",
    key="download_pdf_btn",
    disabled=not pdf_bytes,
    help="Generate forecast or AI report first.",
    on_click=log_visitor_activity if pdf_bytes and firebase_initialized else None,
    args=("Sidebar",) if pdf_bytes and firebase_initialized else None,
    kwargs={"action": "download_pdf"} if pdf_bytes and firebase_initialized else None
)

# Activate Chat Button
chat_ready = st.session_state.forecast_df is not None and gemini_configured and can_access_chat
chat_disabled_reason = ""
if st.session_state.forecast_df is None: chat_disabled_reason = "Run forecast first."
elif not gemini_configured: chat_disabled_reason = "Gemini API not configured."
elif not can_access_chat: chat_disabled_reason = reason_chat

if st.sidebar.button("Activate Chat", key="activate_chat_btn", disabled=not chat_ready, help=chat_disabled_reason):
    if firebase_initialized: log_visitor_activity("Sidebar", action="activate_chat", feature_used="AI Chat")
    if not st.session_state.chat_active:
        initialize_chat() # Ensure chat model is ready
        if st.session_state.get("chat_model") is not None: # Check if init was successful
            st.session_state.chat_active = True
            context = f"Chat Context: Forecast generated. Data: {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. Forecast: {forecast_horizon} steps, range {st.session_state.forecast_df['Forecast'].min():.2f}-{st.session_state.forecast_df['Forecast'].max():.2f}. Ask questions."
            st.session_state.messages = [] 
            st.session_state.messages.append({"role": "model", "content": context})
            st.success("Chat activated with forecast context!")
            st.rerun()
        else:
             st.error("Could not activate chat. Failed to initialize AI model.")
    else:
        st.sidebar.info("Chat is already active.")

# --- Main Area Tabs --- 
st.title("Groundwater Level Forecasting & Analysis")

tab_titles = ["Home / Data View", "Forecast Results", "Train Model", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# Home / Data View Tab
with tabs[0]:
    if firebase_initialized: log_visitor_activity("Tab: Home", "page_view")
    st.header("Welcome to DeepHydro AI")
    st.markdown("""<div class="app-intro">...</div>""", unsafe_allow_html=True) # Keep existing intro
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
        if st.session_state.metrics:
            st.json(st.session_state.metrics)
        else:
            st.info("Metrics generated during model training.")
    else:
        st.info("Run a forecast to see results here.")

# Train Model Tab
with tabs[2]:
    if firebase_initialized: log_visitor_activity("Tab: Train Model", "page_view")
    st.header("Train a New LSTM Model")
    if st.session_state.selected_model_type != "Train New":
        st.info("Select 'Train New' in the sidebar to enable training.")
    elif st.session_state.df is None:
        st.warning("Upload data first.")
    else:
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        with col1: seq_len_train = st.number_input("Sequence Length", min_value=10, max_value=180, value=60, step=5, key="train_seq")
        with col2: epochs_train = st.number_input("Epochs", min_value=1, max_value=200, value=20, step=1, key="train_epochs")
        with col3: batch_size_train = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8, key="train_batch")
        test_size_train = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="train_test_split")
        
        if st.button("Train New LSTM Model", key="train_model_btn"):
            if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_start", details={"seq": seq_len_train, "epochs": epochs_train, "batch": batch_size_train, "test_split": test_size_train})
            with st.spinner("Training model..."):
                try:
                    df_train = st.session_state.df.copy()
                    scaler_train = MinMaxScaler(feature_range=(0, 1))
                    scaled_data_train = scaler_train.fit_transform(df_train["Level"].values.reshape(-1, 1))
                    
                    X, y = create_sequences(scaled_data_train, seq_len_train)
                    if len(X) == 0:
                        st.error(f"Not enough data ({len(df_train)} points) for sequence length {seq_len_train}.")
                        if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"reason": "Insufficient data"})
                        st.stop()
                        
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_train, random_state=42, shuffle=False) # No shuffle for time series
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    model_train = build_lstm_model(seq_len_train)
                    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                    
                    history = model_train.fit(X_train, y_train, epochs=epochs_train, batch_size=batch_size_train, 
                                              validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
                    
                    y_pred_scaled = model_train.predict(X_test)
                    y_pred_train = scaler_train.inverse_transform(y_pred_scaled)
                    y_test_train = scaler_train.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Store results in session state
                    st.session_state.metrics = calculate_metrics(y_test_train, y_pred_train)
                    st.session_state.training_metrics = st.session_state.metrics # Keep a copy specifically for training
                    st.session_state.loss_fig = create_loss_plot(history.history)
                    st.session_state.custom_model = model_train # Store the trained model
                    st.session_state.custom_model_seq_len = seq_len_train
                    st.session_state.model_trained = True
                    st.session_state.trained_model_seq_len = seq_len_train # Record sequence length used for training
                    
                    st.success("Model training completed!")
                    st.balloons()
                    if firebase_initialized: log_visitor_activity("Tab: Train Model", action="train_model_success", details={"metrics": st.session_state.metrics})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
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
    if not gemini_configured: st.warning("AI features disabled (Gemini API not configured).")
    elif st.session_state.ai_report:
        report_escaped = st.session_state.ai_report.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="chat-message ai-message">{report_escaped}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    else: st.info("Click 'Generate AI Report' (sidebar) after running a forecast.")

# AI Chatbot Tab
with tabs[4]:
    if firebase_initialized: log_visitor_activity("Tab: AI Chatbot", "page_view")
    st.header("AI Chatbot")
    if not gemini_configured: st.warning("AI features disabled (Gemini API not configured).")
    elif st.session_state.chat_active:
        display_chat_history()
        user_prompt = st.chat_input("Ask about the forecast...")
        if user_prompt:
            handle_chat_input(user_prompt)
    else:
        st.info("Activate chat via the sidebar after running a forecast.")

# Admin Analytics Tab
with tabs[5]:
    if firebase_initialized: log_visitor_activity("Tab: Admin Analytics", "page_view")
    st.header("Admin Analytics Dashboard")
    
    # Get admin password from config, fallback to default if not set or config failed
    ADMIN_PASSWORD = APP_CONFIG.get("admin_password", "admin123") if admin_password_configured else "admin123"
    
    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")
        if st.button("Login as Admin", key="admin_login_btn"):
            # Use secure comparison if possible (hashlib) - basic comparison for now
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
                for chart in charts:
                    st.plotly_chart(chart, use_container_width=True)
                    
                try:
                    csv = visitor_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Analytics CSV", data=csv,
                        file_name="visitor_analytics.csv", mime="text/csv",
                        key="download_analytics_csv",
                        on_click=log_visitor_activity if firebase_initialized else None,
                        args=("Admin Analytics",) if firebase_initialized else None,
                        kwargs={"action": "download_analytics_csv"} if firebase_initialized else None
                    )
                except Exception as e_csv:
                    st.warning(f"Could not generate CSV: {e_csv}")
                    if firebase_initialized: log_visitor_activity("Admin Analytics", action="download_analytics_csv_fail", details={"error": str(e_csv)})
            else:
                st.info("No visitor logs found or error fetching logs.")
        else:
            st.warning("Firebase not initialized. Cannot display analytics.")

# --- About Us Section --- 
st.markdown("<div class='about-us-header'>About DeepHydro AI ▼</div>", unsafe_allow_html=True)
st.markdown("""<div class="about-us-content">... [Your About Us Text] ... 
**Configuration Note:** This version reads configuration from the `APP_CONFIG_JSON` environment variable. Ensure this variable contains valid JSON with keys like `firebase_service_account`, `google_oauth`, `google_api_key`, `admin_password`, and optionally `firebase_database_url`. 
</div>""", unsafe_allow_html=True)

# --- (End of Script) --- 

