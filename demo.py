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
from dotenv import load_dotenv
import hashlib
import streamlit.components.v1 as components # Import components
from streamlit_oauth import OAuth2Component # <-- Import for Google OAuth

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 3

# --- Firebase Configuration --- 
def initialize_firebase():
    """
    Initialize Firebase with secure credential management.
    Loads credentials from environment variables for secure deployment.
    """
    load_dotenv() # Load .env file if present (for local development)
    if not firebase_admin._apps:
        try:
            # Load Firebase credentials from environment variable
            firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            
            if firebase_creds_json:
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                # Construct Firebase URL safely, checking if project_id exists
                project_id = cred_dict.get("project_id")
                if not project_id:
                    st.error("Firebase Service Account JSON is missing 'project_id'. Cannot determine Database URL.")
                    return False
                firebase_url = os.getenv("FIREBASE_DATABASE_URL", f"https://{project_id}-default-rtdb.firebaseio.com/")
                
                firebase_admin.initialize_app(cred, {
                    "databaseURL": firebase_url
                })
                # st.success("Firebase initialized successfully.") # Optional: for debugging
                return True
            else:
                st.warning("Firebase credentials (FIREBASE_SERVICE_ACCOUNT) not found in environment variables. Analytics and usage tracking are disabled.")
                return False
        except json.JSONDecodeError:
             st.error("Error decoding Firebase Service Account JSON from environment variable. Check the format.")
             return False
        except Exception as e:
            st.warning(f"Firebase initialization error: {e}. Analytics and usage tracking are disabled.")
            return False
    return True # Already initialized

# --- User Identification & Tracking --- 
def get_client_ip():
    """Get the client's IP address if available."""
    try:
        # Use a reliable service to get IP
        response = requests.get("https://httpbin.org/ip", timeout=3)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json().get("origin", "Unknown")
    except requests.exceptions.RequestException as e:
        # print(f"Could not get IP: {e}") # Debug
        return "Unknown"
    except Exception as e:
        # print(f"Could not get IP (other error): {e}") # Debug
        return "Unknown"

def get_persistent_user_id():
    """Generate or retrieve a persistent user ID."""
    # Priority: Use real authenticated user email if available
    if st.session_state.get("auth_status") == "authenticated" and st.session_state.get("user_email"):
        user_id = st.session_state.user_email
        if st.session_state.get("persistent_user_id") != user_id:
             st.session_state.persistent_user_id = user_id
        return user_id
        
    # Fallback: Use existing persistent ID if set (could be anonymous or previously simulated)
    if "persistent_user_id" in st.session_state and st.session_state.persistent_user_id:
        return st.session_state.persistent_user_id

    # Generate anonymous ID if no other ID is available
    ip_address = get_client_ip()
    user_agent = st.session_state.get("user_agent", "Unknown")
    hash_input = f"{ip_address}-{user_agent}"
    hashed_id = hashlib.sha256(hash_input.encode()).hexdigest()
    persistent_id = f"anon_{hashed_id}"
    st.session_state.persistent_user_id = persistent_id
    return persistent_id

def update_firebase_profile_on_login(email):
    """Update Firebase profile when a user logs in via Google OAuth."""
    if not firebase_admin._apps or not email:
        return
    try:
        ref = db.reference(f"users/{email}")
        profile = ref.get()
        now_iso = datetime.datetime.now().isoformat()
        
        update_data = {
            "is_authenticated": True,
            "last_login_google": now_iso,
            "user_id": email # Ensure user_id field matches email
        }
        
        if profile is None:
            # If this is the first time seeing this email, create profile
            update_data["first_visit"] = now_iso
            update_data["visit_count"] = 1
            update_data["feature_usage_count"] = 0 # Start fresh or carry over? Start fresh.
            update_data["last_visit"] = now_iso
            ref.set(update_data)
            st.session_state.user_profile = update_data # Update session state
        else:
            # Update existing profile
            # Increment visit count only if it's a new session (handled by get_or_create)
            # Here, just mark as authenticated and update last login
            ref.update(update_data)
            # Refresh profile in session state
            st.session_state.user_profile = ref.get()
            
        # Log login event
        log_visitor_activity("Authentication", action="google_login_success", details={"email": email})
            
    except Exception as e:
        st.error(f"Firebase error updating profile after Google login for {email}: {e}")
        log_visitor_activity("Authentication", action="google_login_firebase_update_fail", details={"email": email, "error": str(e)})

def get_or_create_user_profile(user_id):
    """Get user profile from Firebase or create a new one."""
    if not firebase_admin._apps:
        return None, False # Indicate Firebase not available
    
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
                "email": current_email if current_auth_status else None # Store email if authenticated
            }
            ref.set(profile)
            # st.info(f"Created new user profile for {user_id}") # Debug
        else:
            # Update visit count and last visit time if it's a new session
            if "session_visit_logged" not in st.session_state:
                profile["visit_count"] = profile.get("visit_count", 0) + 1
                profile["last_visit"] = now_iso
                # Update auth status and email if they logged in this session
                profile["is_authenticated"] = current_auth_status
                if current_auth_status:
                     profile["email"] = current_email
                else: # If somehow they became unauthenticated, clear email? Or keep last known?
                     profile["email"] = profile.get("email") # Keep last known for now
                     
                ref.update({
                    "visit_count": profile["visit_count"], 
                    "last_visit": profile["last_visit"],
                    "is_authenticated": profile["is_authenticated"],
                    "email": profile.get("email")
                })
                st.session_state.session_visit_logged = True # Mark visit as logged for this session
                # st.info(f"Updated visit count for {user_id} to {profile['visit_count']}") # Debug
            
        return profile, is_new_user
    except Exception as e:
        st.warning(f"Firebase error getting/creating user profile for {user_id}: {e}")
        return None, False

def increment_feature_usage(user_id):
    """Increment the feature usage count for the user in Firebase."""
    if not firebase_admin._apps:
        return False
    
    try:
        ref = db.reference(f"users/{user_id}/feature_usage_count")
        # Atomically increment the count
        current_count = ref.get() or 0
        ref.set(current_count + 1)
        # Update session state as well
        if "user_profile" in st.session_state and st.session_state.user_profile:
            st.session_state.user_profile["feature_usage_count"] = current_count + 1
        return True
    except Exception as e:
        st.warning(f"Firebase error incrementing usage count for {user_id}: {e}")
        return False

# --- Authentication Check & Google OAuth --- 
def check_feature_access():
    """Check if user can access advanced features based on usage count and auth status."""
    # Use real auth status first
    is_authenticated = st.session_state.get("auth_status") == "authenticated"
    if is_authenticated:
        return True, "Access granted (Authenticated)."
        
    # If not authenticated, check usage count from profile
    if "user_profile" not in st.session_state or st.session_state.user_profile is None:
        user_id = get_persistent_user_id() # Get current ID (likely anonymous)
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        if st.session_state.user_profile is None:
            st.warning("Could not retrieve user profile. Feature access may be limited.")
            return False, "Cannot verify usage limit. Access denied."

    usage_count = st.session_state.user_profile.get("feature_usage_count", 0)
    
    if usage_count < ADVANCED_FEATURE_LIMIT:
        return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
    else:
        return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in to continue."

# --- Google OAuth Implementation --- 
# Load credentials from environment variables
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

def get_user_info_from_google(token):
    """Placeholder function to fetch user info using the access token."""
    if not token or "access_token" not in token:
        return None
    try:
        # print(f"DEBUG: Fetching user info with token: {token['access_token'][:10]}...") # Debug
        response = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {token['access_token']}"}
        )
        response.raise_for_status() # Raise an exception for bad status codes
        user_info = response.json()
        # print(f"DEBUG: User info received: {user_info}") # Debug
        return user_info
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching user info from Google: {e}")
        # Attempt to revoke token if user info fetch fails?
        # Maybe check response status code (e.g., 401 means token expired/invalid)
        return None
    except Exception as e:
         st.error(f"Unexpected error fetching user info: {e}")
         return None

def show_google_login():
    """Handles the Google OAuth login flow using streamlit-oauth."""
    if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
        st.error("Google OAuth credentials (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI) are not configured in environment variables. Cannot enable Google Sign-In.")
        log_visitor_activity("Authentication", action="google_login_fail_config_missing")
        return # Stop execution if config is missing

    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    
    # Check if token exists in session state (user is already logged in)
    if "token" not in st.session_state:
        # Display login prompt only if usage limit is reached
        usage_count = st.session_state.user_profile.get("feature_usage_count", 0) if st.session_state.user_profile else 0
        if usage_count >= ADVANCED_FEATURE_LIMIT:
            st.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached for advanced features.")
            st.info("Please log in with Google to continue using AI Report, Forecasting, and AI Chat.")
            
            # Create authorize button
            result = oauth2.authorize_button(
                name="Login with Google",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=REDIRECT_URI,
                scope="openid email profile", # Request email and profile info
                key="google_login",
                extras_params={"prompt": "consent", "access_type": "offline"} # Force consent screen, request refresh token
            )
            
            if result:
                # print(f"DEBUG: OAuth result received: {result}") # Debug
                st.session_state.token = result.get("token")
                # Immediately try to fetch user info after getting token
                user_info = get_user_info_from_google(st.session_state.token)
                if user_info and user_info.get("email"):
                    st.session_state.auth_status = "authenticated"
                    st.session_state.user_email = user_info.get("email")
                    st.session_state.user_name = user_info.get("name")
                    st.session_state.persistent_user_id = user_info.get("email") # Crucial: Update persistent ID
                    update_firebase_profile_on_login(user_info.get("email"))
                    st.success(f"Logged in as {st.session_state.user_name} ({st.session_state.user_email}).")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("Login successful, but failed to retrieve user information from Google.")
                    # Clear potentially invalid token
                    del st.session_state.token 
                    log_visitor_activity("Authentication", action="google_login_fail_userinfo")
        # else: No button clicked or usage limit not reached
    else:
        # User is already logged in (token exists)
        # Display user info and logout button
        if not st.session_state.get("user_email"):
             # If token exists but user info is missing, try fetching again
             user_info = get_user_info_from_google(st.session_state.token)
             if user_info and user_info.get("email"):
                 st.session_state.auth_status = "authenticated"
                 st.session_state.user_email = user_info.get("email")
                 st.session_state.user_name = user_info.get("name")
                 st.session_state.persistent_user_id = user_info.get("email")
             else:
                 # Still failed to get info, maybe token is bad? Clear it.
                 st.error("Could not verify login status. Please log in again.")
                 del st.session_state.token
                 st.session_state.auth_status = "anonymous"
                 st.session_state.user_email = None
                 st.session_state.user_name = None
                 log_visitor_activity("Authentication", action="google_reauth_fail_userinfo")
                 st.rerun()
                 
        # Display logged-in status and logout button if email is confirmed
        if st.session_state.get("user_email"):
            st.sidebar.success(f"Logged in as: {st.session_state.user_name} ({st.session_state.user_email})")
            if st.sidebar.button("Logout", key="google_logout"):
                # Log logout action before clearing state
                log_visitor_activity("Authentication", action="google_logout", details={"email": st.session_state.user_email})
                # Clear session state related to auth
                if "token" in st.session_state: del st.session_state.token
                st.session_state.auth_status = "anonymous"
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.persistent_user_id = None # Reset persistent ID to force re-generation of anonymous ID
                st.rerun()

# --- Visitor Analytics Functions --- (Modified for detailed actions)
def get_session_id():
    """Create or retrieve a unique session ID for the current user session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None, details=None):
    """
    Log visitor activity to Firebase Realtime Database, including persistent user ID.
    Args:
        page_name: The name of the page or section being viewed/interacted with.
        action: The action performed (e.g., page_view, run_forecast, generate_report).
        feature_used: Specific feature used (e.g., 'Forecast', 'AI Report', 'AI Chat') - used for usage counting.
        details: Optional dictionary for additional context (e.g., filename, model type).
    """
    if not firebase_admin._apps:
        return # Skip logging if Firebase is not initialized

    try:
        user_id = get_persistent_user_id() # Get persistent ID (anon hash or real email)
        profile, is_new = get_or_create_user_profile(user_id) # Ensure profile exists and update visit count
        
        # Check access and increment usage count *before* logging the successful action
        # for the specific features that count towards the limit.
        should_increment = feature_used in ["Forecast", "AI Report", "AI Chat"]
        access_granted, _ = check_feature_access() # Check access status
        
        # Default action status is success unless modified below
        action_status = "success"
        
        if should_increment:
            # Only increment if access is granted (either under limit or authenticated)
            if access_granted:
                increment_feature_usage(user_id)
            else:
                # If access was denied but they tried to use the feature, log the attempt but don't increment
                action_status = "denied_limit_reached"
                pass # Do not increment usage count

        # Construct the final action string with status
        full_action_name = f"{action}_{action_status}"
        
        # Proceed with logging the activity
        ref = db.reference("visitors_log") # Use the collection name
        log_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        session_id = get_session_id()
        user_agent = st.session_state.get("user_agent", "Unknown")
        ip_address = get_client_ip() # Log IP for geo-location, etc., but use hashed ID for tracking

        log_data = {
            "timestamp": timestamp,
            "persistent_user_id": user_id, # Track via persistent ID
            "is_authenticated": st.session_state.get("auth_status") == "authenticated", # Use real auth status
            "visit_count": profile.get("visit_count", 1) if profile else 1,
            "ip_address": ip_address, # Logged for info, not tracking ID
            "page": page_name,
            "action": full_action_name, # Use the action name with status
            "feature_used": feature_used, # Log which feature was used
            "session_id": session_id,
            "user_agent": user_agent
        }
        
        # Add optional details if provided
        if details and isinstance(details, dict):
            log_data["details"] = details
            
        ref.child(log_id).set(log_data)
        # st.info(f"Logged activity: {full_action_name} on {page_name} by {user_id}") # Debug

    except Exception as e:
        # Silently fail logging to not disrupt user experience, but maybe log locally?
        # print(f"Error logging visitor activity: {e}") # Optional: log to console/file
        pass

def fetch_visitor_logs():
    """
    Fetch visitor logs from Firebase for admin viewing.
    Returns a pandas DataFrame with the visitor data.
    """
    if not firebase_admin._apps:
        return pd.DataFrame()
    
    try:
        ref = db.reference("visitors_log") # Use the collection name
        visitors_data = ref.get()
        
        if not visitors_data:
            return pd.DataFrame()
        
        # Handle potential nested data (like 'details') during DataFrame creation
        visitors_list = []
        for log_id, data in visitors_data.items():
            data["log_id"] = log_id
            # Flatten details if they exist
            if "details" in data and isinstance(data["details"], dict):
                for k, v in data["details"].items():
                    data[f"detail_{k}"] = v
                del data["details"] # Remove original nested dict
            visitors_list.append(data)
        
        df = pd.DataFrame(visitors_list)
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Sort by timestamp (most recent first)
            df = df.sort_values("timestamp", ascending=False)
        
        return df
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
    """
    Create visualizations of visitor data using Plotly.
    Args: visitor_df: DataFrame containing visitor data
    Returns: List of Plotly figures
    """
    if visitor_df.empty:
        return []
    
    figures = []
    
    try:
        df = visitor_df.copy()
        if "timestamp" not in df.columns:
             st.warning("Timestamp column missing in visitor logs, cannot generate time-based charts.")
             return []
             
        df["date"] = df["timestamp"].dt.date
        
        # 1. Daily visitors (using unique persistent IDs)
        if "persistent_user_id" in df.columns:
            daily_visitors = df.groupby("date")["persistent_user_id"].nunique().reset_index(name="unique_users")
            daily_visitors["date"] = pd.to_datetime(daily_visitors["date"])
            fig1 = px.line(daily_visitors, x="date", y="unique_users", title="Daily Unique Visitors", labels={"unique_users": "Unique Users", "date": "Date"})
            figures.append(fig1)
        else: st.warning("Persistent User ID column missing, cannot generate daily visitors chart.")
        
        # 2. Page/Feature Popularity (using action)
        if "action" in df.columns:
            # Exclude basic page views for a cleaner action chart
            action_counts = df[~df["action"].str.contains("page_view", case=False, na=False)]["action"].value_counts().reset_index()
            # action_counts = df["action"].value_counts().reset_index() # Include page views
            action_counts.columns = ["action", "count"]
            fig2 = px.bar(action_counts, x="action", y="count", title="Activity Counts by Action (Excluding Page Views)", labels={"count": "Number of Times", "action": "Action Type"})
            figures.append(fig2)
        else: st.warning("Action column missing, cannot generate activity counts chart.")

        # 3. Authenticated vs Anonymous Users (based on last known status)
        if "persistent_user_id" in df.columns and "is_authenticated" in df.columns:
            latest_status = df.sort_values("timestamp").groupby("persistent_user_id")["is_authenticated"].last().reset_index()
            auth_counts = latest_status["is_authenticated"].value_counts().reset_index()
            auth_counts.columns = ["is_authenticated", "count"]
            auth_counts["status"] = auth_counts["is_authenticated"].map({True: "Authenticated", False: "Anonymous"})
            fig3 = px.pie(auth_counts, values="count", names="status", title="User Authentication Status (Latest Known)")
            figures.append(fig3)
        else: st.warning("User ID or Authentication status column missing, cannot generate authentication chart.")

        # 4. Hourly activity heatmap
        try:
            df["hour"] = df["timestamp"].dt.hour
            df["weekday"] = df["timestamp"].dt.day_name()
            hourly_activity = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
            # Ensure all hours/days are present for a full heatmap
            all_hours = list(range(24))
            all_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = pd.MultiIndex.from_product([all_weekdays, all_hours], names=["weekday", "hour"]).to_frame(index=False)
            heatmap_data = pd.merge(heatmap_data, hourly_activity, on=["weekday", "hour"], how="left").fillna(0)
            # Pivot for heatmap
            heatmap_pivot = heatmap_data.pivot(index="weekday", columns="hour", values="count")
            heatmap_pivot = heatmap_pivot.reindex(all_weekdays) # Ensure correct order
            
            fig4 = px.imshow(heatmap_pivot, 
                           labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
                           x=[str(h) for h in all_hours], 
                           y=all_weekdays,
                           title="User Activity Heatmap (Hour vs Day)",
                           color_continuous_scale=px.colors.sequential.Viridis)
            fig4.update_xaxes(side="bottom")
            figures.append(fig4)
        except Exception as e_heatmap:
            st.warning(f"Could not generate activity heatmap: {e_heatmap}")
            
    except Exception as e_charts:
        st.error(f"Error creating visitor charts: {e_charts}")
        
    return figures

# --- PDF Generation (unchanged) ---
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
            # Save bytes to a temporary file to add to PDF
            temp_img_path = "temp_plot.png"
            with open(temp_img_path, "wb") as f:
                f.write(plot_bytes)
            self.image(temp_img_path, x=10, w=self.w - 20) # Adjust width as needed
            os.remove(temp_img_path)
            self.ln(5)
        except Exception as e:
            self.chapter_body(f"[Error adding plot: {e}]")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def create_pdf_report(forecast_fig, loss_fig, metrics, ai_report_text):
    pdf = PDF()
    pdf.add_page()
    
    # Add Forecast Plot
    if forecast_fig:
        try:
            img_bytes = forecast_fig.to_image(format="png", scale=2)
            pdf.add_plot(img_bytes, "Forecast Visualization")
        except Exception as e:
            pdf.chapter_body(f"[Error adding forecast plot: {e}]")
    else:
        pdf.chapter_body("[Forecast plot not available]")

    # Add Training Loss Plot
    if loss_fig:
        try:
            img_bytes = loss_fig.to_image(format="png", scale=2)
            pdf.add_plot(img_bytes, "Model Training Loss (if applicable)")
        except Exception as e:
            pdf.chapter_body(f"[Error adding loss plot: {e}]")
    else:
        pdf.chapter_body("[Training loss plot not available]")

    # Add Metrics
    pdf.chapter_title("Performance Metrics")
    if metrics:
        metrics_text = "\n".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()])
        pdf.chapter_body(metrics_text)
    else:
        pdf.chapter_body("[Metrics not available]")

    # Add AI Report Text
    pdf.chapter_title("AI Generated Analysis")
    if ai_report_text:
        # Basic handling for markdown-like text (replace ** with bold?)
        # FPDF doesn't directly support Markdown. Need more complex parsing or library.
        pdf.chapter_body(ai_report_text)
    else:
        pdf.chapter_body("[AI analysis not generated or not available]")

    # Convert PDF to bytes
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1") # Use latin-1 for byte output
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF bytes: {e}")
        return None

# --- Custom CSS (unchanged from previous version) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* General Styling */
    .stApp { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 15px;
        transition: background-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] { background-color: #e6f7ff; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #f0f0f0; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #ffffff; padding: 1rem; }
    [data-testid="stSidebar"] h2 { color: #1890ff; }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%; 
        border-radius: 5px;
        background-color: #1890ff;
        color: white;
        transition: background-color 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #40a9ff; }
    [data-testid="stSidebar"] .stDownloadButton>button {
        width: 100%; 
        border-radius: 5px;
        background-color: #52c41a;
        color: white;
        transition: background-color 0.3s ease;
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

# --- JavaScript (unchanged from previous version) ---
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
    
    // Add event listeners
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            // Copy functionality
            const chatMessages = document.querySelectorAll('.chat-message');
            chatMessages.forEach(function(message) {
                // Add tooltip element if not present
                if (!message.querySelector('.copy-tooltip')) {
                    const tooltip = document.createElement('span');
                    tooltip.className = 'copy-tooltip';
                    tooltip.textContent = 'Copied!';
                    message.appendChild(tooltip);
                }
                
                let longPressTimer;
                message.addEventListener('touchstart', function(e) {
                    longPressTimer = setTimeout(() => {
                        const textToCopy = this.innerText.replace('Copied!', '').trim(); // Exclude tooltip text
                        copyToClipboard(textToCopy);
                        const tooltip = this.querySelector('.copy-tooltip');
                        if (tooltip) {
                            tooltip.style.display = 'block';
                            setTimeout(() => { tooltip.style.display = 'none'; }, 1500);
                        }
                    }, 500); // 500ms for long press
                });
                
                message.addEventListener('touchend', function() {
                    clearTimeout(longPressTimer);
                });
                message.addEventListener('touchmove', function() { // Cancel long press if finger moves
                    clearTimeout(longPressTimer);
                });
                // Add click listener for desktop
                 message.addEventListener('click', function(e) {
                     const textToCopy = this.innerText.replace('Copied!', '').trim();
                     copyToClipboard(textToCopy);
                     const tooltip = this.querySelector('.copy-tooltip');
                     if (tooltip) {
                         tooltip.style.display = 'block';
                         setTimeout(() => { tooltip.style.display = 'none'; }, 1500);
                     }
                 });
            });
            
            // Collapsible About Us
            const aboutUsHeader = document.querySelector('.about-us-header');
            const aboutUsContent = document.querySelector('.about-us-content');
            if (aboutUsHeader && aboutUsContent) {
                // Initial state: collapsed
                if (!aboutUsContent.classList.contains('initialized')) {
                     aboutUsContent.style.display = 'none';
                     aboutUsContent.classList.add('initialized');
                }
                aboutUsHeader.addEventListener('click', function() {
                    if (aboutUsContent.style.display === 'none') {
                        aboutUsContent.style.display = 'block';
                    } else {
                        aboutUsContent.style.display = 'none';
                    }
                });
            }
        }, 1000); // Delay to ensure elements are loaded
    });
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration --- 
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css()
add_javascript_functionality()

# --- Capture User Agent --- (Modified slightly for robustness)
def capture_user_agent():
    """Capture and store the user agent in session state."""
    if "user_agent" not in st.session_state:
        try:
            # Use Streamlit components to run JavaScript that sends the user agent
            component_value = components.html(
                """
                <script>
                // Send the user agent back to Streamlit
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: "streamlit:setComponentValue",
                    key: "user_agent_capture",
                    value: navigator.userAgent
                }}, "*");
                </script>
                """,
                height=0,
                key="user_agent_capture"
            )
            # Try to get the value from the component state if available
            if component_value:
                 st.session_state.user_agent = component_value
            elif "user_agent_capture" in st.session_state and st.session_state.user_agent_capture:
                 st.session_state.user_agent = st.session_state.user_agent_capture
            else:
                 st.session_state.user_agent = "Unknown (Capture Pending)"
        except Exception as e:
            st.session_state.user_agent = "Unknown (Capture Failed)"

# --- Initialize Firebase and User Profile --- 
firebase_initialized = initialize_firebase()
capture_user_agent() # Attempt to capture user agent early

# Initialize session state variables
def initialize_session_state():
    defaults = {
        "df": None, "forecast_df": None, "metrics": None, "forecast_fig": None, 
        "loss_fig": None, "model_trained": False, "selected_model_type": "Standard", 
        "custom_model": None, "custom_model_seq_len": None, 
        "standard_model": None, "standard_model_seq_len": STANDARD_MODEL_SEQUENCE_LENGTH,
        "ai_report": None, "chat_active": False, "messages": [], "chat_model": None,
        "admin_authenticated": False, "user_profile": None, 
        "auth_status": "anonymous", # Use 'anonymous', 'authenticated'
        "user_email": None,
        "user_name": None,
        "token": None, # Store OAuth token
        "persistent_user_id": None,
        "user_agent": "Unknown", "session_id": None, "session_visit_logged": False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session_state()

# Initialize user profile in session state if not already present
if "user_profile" not in st.session_state or st.session_state.user_profile is None:
    if firebase_initialized:
        user_id = get_persistent_user_id() # Get ID first
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id) # Fetch/create profile
    else:
        st.session_state.user_profile = None # No profile if Firebase fails

# --- Gemini API Configuration (unchanged) ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "Gemini_api_key":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
        gemini_model_report = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config) # Using standard gemini-pro
        gemini_model_chat = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)
        gemini_configured = True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
else:
    st.warning("Gemini API Key (GOOGLE_API_KEY) not found or is placeholder. AI features will be disabled.")

# --- Model Paths & Constants (unchanged) ---
STANDARD_MODEL_PATH = "standard_model.h5"
STANDARD_MODEL_SEQUENCE_LENGTH = 60
if os.path.exists(STANDARD_MODEL_PATH):
    try:
        # Only load if not already in session state (avoids reloading)
        if st.session_state.standard_model is None:
            _std_model_temp = load_model(STANDARD_MODEL_PATH, compile=False)
            STANDARD_MODEL_SEQUENCE_LENGTH = _std_model_temp.input_shape[1]
            st.session_state.standard_model = _std_model_temp # Store in session state
            st.session_state.standard_model_seq_len = STANDARD_MODEL_SEQUENCE_LENGTH
            # print("DEBUG: Standard model loaded into session state.") # Debug
        else:
            # Use length from session state if model already loaded
            STANDARD_MODEL_SEQUENCE_LENGTH = st.session_state.standard_model_seq_len
    except Exception as e:
        st.warning(f"Could not load standard model from {STANDARD_MODEL_PATH} to infer sequence length: {e}. Using default {STANDARD_MODEL_SEQUENCE_LENGTH}.")
        log_visitor_activity("Model Handling", action="load_standard_model_fail", details={"path": STANDARD_MODEL_PATH, "error": str(e)})
else:
    st.warning(f"Standard model file not found at path: {STANDARD_MODEL_PATH}. Please ensure it exists.")

# --- Helper Functions (Data Loading, Model Building, Prediction - unchanged) ---
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
        log_visitor_activity("Model Handling", action="load_custom_model", details={"model_name": model_name_for_log, "sequence_length": sequence_length})
        return model, sequence_length
    except Exception as e: 
        st.error(f"Error loading Keras model {model_name_for_log}: {e}")
        log_visitor_activity("Model Handling", action="load_custom_model_fail", details={"model_name": model_name_for_log, "error": str(e)})
        return None, None
    finally: 
        if os.path.exists(temp_model_path): os.remove(temp_model_path)

# Removed cached standard model loading here, handled during init

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    @tf.function
    def predict_step_training_true(inp): return model(inp, training=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step_training_true(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            temp_sequence = np.append(temp_sequence[:, 1:, :], np.array([[next_pred_scaled]]).reshape(1,1,1), axis=1)
        all_predictions.append(iteration_predictions_scaled)
        progress_bar.progress((i + 1) / n_iterations)
        status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
    progress_bar.empty(); status_text.empty()
    predictions_array_scaled = np.array(all_predictions)
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    ci_multiplier = 2.5 # Wider interval
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    lower_bound = scaler.inverse_transform((mean_preds_scaled - ci_multiplier * std_devs_scaled).reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform((mean_preds_scaled + ci_multiplier * std_devs_scaled).reshape(-1, 1)).flatten()
    min_uncertainty_percent = 0.05
    for i in range(len(mean_preds)):
        current_range_percent = (upper_bound[i] - lower_bound[i]) / mean_preds[i] if mean_preds[i] != 0 else 0
        if current_range_percent < min_uncertainty_percent:
            uncertainty_value = mean_preds[i] * min_uncertainty_percent / 2
            lower_bound[i] = mean_preds[i] - uncertainty_value
            upper_bound[i] = mean_preds[i] + uncertainty_value
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.inf
    if np.all(y_true != 0): mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions (unchanged) ---
def create_forecast_plot(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=False))
    fig.update_layout(title="Groundwater Level: Historical Data & LSTM Forecast", xaxis_title="Date", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_loss_plot(history_dict):
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

# --- AI Report Generation (unchanged, but logging added in sidebar call) ---
def generate_ai_report(data_summary, forecast_summary, metrics, model_details):
    if not gemini_configured: return "AI report generation disabled. Gemini API not configured."
    try:
        prompt = f"""Generate a concise scientific report summarizing the groundwater level forecast.

        **Data Summary:**
        {data_summary}

        **Forecast Summary:**
        {forecast_summary}

        **Performance Metrics:**
        {metrics}

        **Model Details:**
        {model_details}

        **Report Structure:**
        1.  **Introduction:** Briefly state the purpose - forecasting groundwater levels using the provided data and LSTM model.
        2.  **Data Overview:** Describe the key characteristics of the historical data (time range, general trend if obvious).
        3.  **Methodology:** Mention the use of an LSTM model (type: {model_details.get('type', 'N/A')}) and the forecast horizon.
        4.  **Results:** Present the key forecast findings (e.g., predicted trend, range of predicted values) and the calculated performance metrics (RMSE, MAE, MAPE).
        5.  **Discussion:** Briefly interpret the results. Mention the forecast uncertainty (referencing the confidence interval if available). Discuss the model's performance based on metrics.
        6.  **Conclusion:** Summarize the main findings and the reliability of the forecast.

        **Instructions:**
        - Be objective and scientific in tone.
        - Keep the report concise (around 300-500 words).
        - Focus on interpreting the provided summaries and metrics.
        - Do not invent data or make claims not supported by the input.
        """
        response = gemini_model_report.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error generating AI report: {e}"

# --- AI Chat Functionality (unchanged, but logging added in sidebar call and chat input) ---
def initialize_chat():
    if not gemini_configured: return
    if "messages" not in st.session_state: st.session_state.messages = []
    if "chat_model" not in st.session_state: st.session_state.chat_model = gemini_model_chat.start_chat(history=[])

def display_chat_history():
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "ai-message"
        st.markdown(f'<div class="chat-message {role_class}">{message["content"]}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)

def handle_chat_input(prompt):
    if not gemini_configured: st.error("Chat disabled. Gemini API not configured."); return
    if not prompt: return
    
    log_visitor_activity("AI Chat", action="send_chat_message")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user-message">{prompt}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.chat_model.send_message(prompt)
        st.session_state.messages.append({"role": "model", "content": response.text})
        st.markdown(f'<div class="chat-message ai-message">{response.text}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
        
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "model", "content": error_msg})
        st.error(error_msg)
        log_visitor_activity("AI Chat", action="chat_error", details={"error": str(e)})

# --- Streamlit App UI ---

# Initialize session state variables (already called above)

# --- Sidebar --- 
st.sidebar.image("logo.png", width=100)
st.sidebar.title("DeepHydro AI")

# --- Authentication Section (Sidebar) ---
st.sidebar.markdown("--- User Access ---")
show_google_login() # <-- Use the real Google Login function

# --- Options Section (Sidebar) ---
st.sidebar.markdown("--- Options ---")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel Data (.xlsx)", type=["xlsx"], key="data_uploader")
if uploaded_file is not None:
    if st.session_state.df is None or uploaded_file.name != st.session_state.get("uploaded_file_name"):
        st.session_state.df = load_and_clean_data(uploaded_file.getvalue())
        st.session_state.uploaded_file_name = uploaded_file.name
        # Reset forecast state when new data is uploaded
        st.session_state.forecast_df = None
        st.session_state.metrics = None
        st.session_state.forecast_fig = None
        st.session_state.ai_report = None
        st.session_state.chat_active = False
        log_visitor_activity("Sidebar", action="upload_data", details={"filename": uploaded_file.name})
        st.rerun() # Rerun to update main page with new data status

# Model Selection
model_choice = st.sidebar.radio(
    "Select Model", 
    ("Standard", "Train New", "Upload Custom (.h5)"), 
    key="model_selector", 
    index=0 if st.session_state.selected_model_type == "Standard" else 1 if st.session_state.selected_model_type == "Train New" else 2
)

if model_choice != st.session_state.selected_model_type:
    log_visitor_activity("Sidebar", action="select_model_type", details={"model_type": model_choice})
    st.session_state.selected_model_type = model_choice
    st.session_state.forecast_df = None 
    st.session_state.metrics = None
    st.session_state.forecast_fig = None
    st.session_state.ai_report = None
    st.session_state.chat_active = False
    st.rerun()

custom_model_file = None
if st.session_state.selected_model_type == "Upload Custom (.h5)":
    custom_model_file = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"], key="model_uploader")
    if custom_model_file is not None:
        if st.session_state.custom_model is None or custom_model_file.name != st.session_state.get("custom_model_file_name"):
            st.session_state.custom_model, st.session_state.custom_model_seq_len = load_keras_model_from_file(custom_model_file, custom_model_file.name)
            st.session_state.custom_model_file_name = custom_model_file.name
            st.rerun()

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

# Run Forecast Button
run_forecast_disabled = st.session_state.df is None or \
                        (st.session_state.selected_model_type == "Upload Custom (.h5)" and st.session_state.custom_model is None) or \
                        (st.session_state.selected_model_type == "Standard" and st.session_state.standard_model is None) or \
                        not can_access_forecast # Disable if access denied

if st.sidebar.button("Run Forecast", key="run_forecast_btn", disabled=run_forecast_disabled):
    # Access already checked by disabled state, proceed directly
    log_visitor_activity("Sidebar", action="run_forecast", feature_used="Forecast", details={"model_type": st.session_state.selected_model_type, "horizon": forecast_horizon, "mc_iterations": mc_iterations})
    
    # Determine model and sequence length
    model_to_use = None
    sequence_length = None
    model_type_log = st.session_state.selected_model_type
    
    if st.session_state.selected_model_type == "Standard":
        model_to_use = st.session_state.standard_model
        sequence_length = st.session_state.standard_model_seq_len
    elif st.session_state.selected_model_type == "Upload Custom (.h5)":
        model_to_use = st.session_state.custom_model
        sequence_length = st.session_state.custom_model_seq_len
    elif st.session_state.selected_model_type == "Train New":
        # Use the model trained in this session if available
        if st.session_state.model_trained and st.session_state.custom_model:
             model_to_use = st.session_state.custom_model
             sequence_length = st.session_state.custom_model_seq_len
             model_type_log = "Trained New"
        else:
            st.warning("Please train a new model first in the 'Train Model' tab.")
            st.stop()
        
    if model_to_use is None or sequence_length is None:
        st.error("Selected model is not available or sequence length is unknown. Cannot run forecast.")
        log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Model unavailable"})
        st.stop()

    if st.session_state.df is None or len(st.session_state.df) <= sequence_length:
        st.error(f"Not enough historical data ({len(st.session_state.df)} points) for the required sequence length ({sequence_length}).")
        log_visitor_activity("Sidebar", action="run_forecast_fail", details={"reason": "Insufficient data"})
        st.stop()
        
    with st.spinner("Forecasting in progress..."):
        try:
            df_forecast = st.session_state.df.copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_forecast["Level"].values.reshape(-1, 1))
            last_sequence_scaled = scaled_data[-sequence_length:]
            
            mean_preds, lower_bound, upper_bound = predict_with_dropout_uncertainty(
                model_to_use, last_sequence_scaled, forecast_horizon, mc_iterations, scaler, sequence_length
            )
            
            last_date = df_forecast["Date"].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            
            st.session_state.forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast": mean_preds,
                "Lower_CI": lower_bound,
                "Upper_CI": upper_bound
            })
            
            st.session_state.metrics = st.session_state.get("training_metrics", {"Info": "Metrics available after training"})
            st.session_state.forecast_fig = create_forecast_plot(df_forecast, st.session_state.forecast_df)
            
            st.success("Forecast completed!")
            log_visitor_activity("Sidebar", action="run_forecast_success", feature_used="Forecast", details={"model_type": model_type_log, "horizon": forecast_horizon})
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")
            log_visitor_activity("Sidebar", action="run_forecast_fail", feature_used="Forecast", details={"model_type": model_type_log, "error": str(e)})

# Display reason if forecast button is disabled due to access limit
if not can_access_forecast and not run_forecast_disabled: # Check if disabled *only* due to access
     st.sidebar.warning(reason_forecast)

# Generate AI Report Button
gen_report_disabled = st.session_state.forecast_df is None or not gemini_configured or not can_access_report
if st.sidebar.button("Generate AI Report", key="gen_report_btn", disabled=gen_report_disabled):
    log_visitor_activity("Sidebar", action="generate_report", feature_used="AI Report")
    with st.spinner("Generating AI analysis..."):
        try:
            data_summary = f"Historical data from {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}, with {len(st.session_state.df)} data points. Average level: {st.session_state.df['Level'].mean():.2f}."
            forecast_summary = f"Forecast for {forecast_horizon} steps. Predicted levels range from {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. Confidence interval suggests levels between {st.session_state.forecast_df['Lower_CI'].min():.2f} and {st.session_state.forecast_df['Upper_CI'].max():.2f}."
            metrics_summary = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in st.session_state.metrics.items()]) if st.session_state.metrics else "Metrics not available (generated during model training)."
            model_details = {
                "type": st.session_state.selected_model_type,
                "sequence_length": st.session_state.standard_model_seq_len if st.session_state.selected_model_type == "Standard" else st.session_state.custom_model_seq_len if st.session_state.custom_model else st.session_state.get("trained_model_seq_len", "N/A"),
                "forecast_horizon": forecast_horizon,
                "mc_iterations": mc_iterations
            }
            
            st.session_state.ai_report = generate_ai_report(data_summary, forecast_summary, metrics_summary, model_details)
            st.success("AI Report generated!")
            log_visitor_activity("Sidebar", action="generate_report_success", feature_used="AI Report")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate AI report: {e}")
            log_visitor_activity("Sidebar", action="generate_report_fail", feature_used="AI Report", details={"error": str(e)})

# Display reason if report button is disabled due to access limit
if not can_access_report and not gen_report_disabled: # Check if disabled *only* due to access
     st.sidebar.warning(reason_report)

# Download PDF Report Button
pdf_bytes = None
if st.session_state.forecast_fig or st.session_state.ai_report:
    try:
        pdf_bytes = create_pdf_report(
            st.session_state.forecast_fig, 
            st.session_state.loss_fig, 
            st.session_state.metrics, 
            st.session_state.ai_report
        )
    except Exception as e:
        st.sidebar.warning(f"Could not prepare PDF: {e}")
        log_visitor_activity("Sidebar", action="prepare_pdf_fail", details={"error": str(e)})

if pdf_bytes:
    st.sidebar.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="DeepHydro_AI_Forecast_Report.pdf",
        mime="application/pdf",
        key="download_pdf_btn",
        on_click=log_visitor_activity, # Log click directly
        args=("Sidebar",), # page_name
        kwargs={"action": "download_pdf"} # action
    )
else:
    st.sidebar.download_button(
        label="Download PDF Report",
        data=b"", # Empty bytes
        file_name="DeepHydro_AI_Forecast_Report.pdf",
        mime="application/pdf",
        key="download_pdf_btn_disabled",
        disabled=True,
        help="Generate a forecast or AI report first to enable download."
    )

# Activate Chat Button
activate_chat_disabled = st.session_state.forecast_df is None or not gemini_configured or not can_access_chat
if st.sidebar.button("Activate Chat", key="activate_chat_btn", disabled=activate_chat_disabled):
    log_visitor_activity("Sidebar", action="activate_chat", feature_used="AI Chat")
    if not st.session_state.chat_active:
        initialize_chat()
        st.session_state.chat_active = True
        if st.session_state.forecast_df is not None:
            context = f"Chat Context: A forecast has been generated. Historical data ranges from {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}. The forecast for the next {forecast_horizon} steps ranges from {st.session_state.forecast_df['Forecast'].min():.2f} to {st.session_state.forecast_df['Forecast'].max():.2f}. Ask me questions about this forecast or groundwater level analysis."
            st.session_state.messages = [] # Clear previous messages
            st.session_state.chat_model = gemini_model_chat.start_chat(history=[]) # Start fresh chat model
            st.session_state.messages.append({"role": "model", "content": context})
            st.success("Chat activated with forecast context!")
        else:
             st.warning("Chat activated, but no forecast context available.")
        st.rerun()
    else:
        st.sidebar.info("Chat is already active.")

# Display reason if chat button is disabled due to access limit
if not can_access_chat and not activate_chat_disabled: # Check if disabled *only* due to access
     st.sidebar.warning(reason_chat)

# --- Main Area Tabs --- 
st.title("Groundwater Level Forecasting & Analysis")

tab_titles = ["Home / Data View", "Forecast Results", "Train Model", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# Home / Data View Tab
with tabs[0]:
    log_visitor_activity("Tab: Home", "page_view") # Log tab view
    st.header("Welcome to DeepHydro AI")
    st.markdown("""
    <div class="app-intro">
    This application uses AI, specifically Long Short-Term Memory (LSTM) networks, to forecast groundwater levels.
    Upload your historical data (.xlsx format with 'Date' and 'Level' columns), choose a model (standard, train new, or upload custom), 
    set your forecast parameters, and run the forecast. You can then generate an AI-powered report or chat about the results.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Uploaded Data Overview")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head()) 
        st.metric("Total Data Points", len(st.session_state.df))
        st.metric("Date Range", f"{st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}")
        st.line_chart(st.session_state.df.set_index('Date')['Level'])
    else:
        st.info("Upload an Excel file using the sidebar to get started.")

# Forecast Results Tab
with tabs[1]:
    log_visitor_activity("Tab: Forecast", "page_view")
    st.header("Forecast Results")
    if st.session_state.forecast_fig:
        st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecast_df)
        st.subheader("Performance Metrics")
        if st.session_state.metrics:
            st.json(st.session_state.metrics) # Display metrics as JSON
        else:
            st.info("Metrics are calculated and displayed after model training.")
    else:
        st.info("Run a forecast using the sidebar options to see results here.")

# Train Model Tab
with tabs[2]:
    log_visitor_activity("Tab: Train Model", "page_view")
    st.header("Train a New LSTM Model")
    if st.session_state.selected_model_type != "Train New":
        st.info("Select 'Train New' in the sidebar model options to enable training.")
    elif st.session_state.df is None:
        st.warning("Upload data first before training a model.")
    else:
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        with col1: seq_len_train = st.number_input("Sequence Length", min_value=10, max_value=180, value=60, step=5, key="train_seq")
        with col2: epochs_train = st.number_input("Epochs", min_value=1, max_value=200, value=20, step=1, key="train_epochs")
        with col3: batch_size_train = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8, key="train_batch")
        
        test_size_train = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="train_test_split")
        
        if st.button("Train New LSTM Model", key="train_model_btn"):
            log_visitor_activity("Tab: Train Model", action="train_model", details={
                "sequence_length": seq_len_train, "epochs": epochs_train, 
                "batch_size": batch_size_train, "test_size": test_size_train
            })
            with st.spinner("Training model..."):
                try:
                    df_train = st.session_state.df.copy()
                    scaler_train = MinMaxScaler(feature_range=(0, 1))
                    scaled_data_train = scaler_train.fit_transform(df_train["Level"].values.reshape(-1, 1))
                    
                    X, y = create_sequences(scaled_data_train, seq_len_train)
                    if len(X) == 0:
                        st.error(f"Not enough data ({len(df_train)} points) to create sequences of length {seq_len_train}.")
                        log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"reason": "Insufficient data for sequences"})
                        st.stop()
                        
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_train, random_state=42)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    model_train = build_lstm_model(seq_len_train)
                    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                    
                    history = model_train.fit(X_train, y_train, epochs=epochs_train, batch_size=batch_size_train, 
                                              validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
                    
                    y_pred_scaled = model_train.predict(X_test)
                    y_pred_train = scaler_train.inverse_transform(y_pred_scaled)
                    y_test_train = scaler_train.inverse_transform(y_test.reshape(-1, 1))
                    
                    st.session_state.metrics = calculate_metrics(y_test_train, y_pred_train)
                    st.session_state.training_metrics = st.session_state.metrics
                    st.session_state.loss_fig = create_loss_plot(history.history)
                    
                    st.session_state.custom_model = model_train
                    st.session_state.custom_model_seq_len = seq_len_train
                    st.session_state.model_trained = True
                    st.session_state.trained_model_seq_len = seq_len_train
                    
                    st.success("Model training completed!")
                    st.balloons()
                    log_visitor_activity("Tab: Train Model", action="train_model_success", details={"metrics": st.session_state.metrics})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    log_visitor_activity("Tab: Train Model", action="train_model_fail", details={"error": str(e)})

        if st.session_state.model_trained:
            st.subheader("Training Results")
            if st.session_state.loss_fig:
                st.plotly_chart(st.session_state.loss_fig, use_container_width=True)
            if st.session_state.metrics:
                st.json(st.session_state.metrics)
            st.info("This trained model is now selected for forecasting. Run forecast from the sidebar.")

# AI Report Tab
with tabs[3]:
    log_visitor_activity("Tab: AI Report", "page_view")
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    if st.session_state.ai_report: 
        st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    else: st.info("Click 'Generate AI Report' (sidebar) after a forecast.")

# AI Chatbot Tab
with tabs[4]:
    log_visitor_activity("Tab: AI Chatbot", "page_view")
    st.header("AI Chatbot")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    elif st.session_state.chat_active:
        display_chat_history()
        user_prompt = st.chat_input("Ask about the forecast...")
        handle_chat_input(user_prompt)
    elif st.session_state.forecast_df is None:
        st.info("Run a forecast first, then click 'Activate Chat' in the sidebar.")
    else:
        st.info("Click 'Activate Chat' (sidebar) after a forecast." if gemini_configured else "AI Chat disabled.")

# Admin Analytics Tab
with tabs[5]:
    log_visitor_activity("Tab: Admin Analytics", "page_view")
    st.header("Admin Analytics Dashboard")
    
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    
    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter Admin Password:", type="password", key="admin_pass_input")
        if st.button("Login as Admin", key="admin_login_btn"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                log_visitor_activity("Admin Auth", action="admin_login_success")
                st.rerun()
            else:
                st.error("Incorrect admin password.")
                log_visitor_activity("Admin Auth", action="admin_login_fail")
    else:
        st.success("Admin access granted.")
        if st.button("Logout Admin", key="admin_logout_btn"):
            st.session_state.admin_authenticated = False
            log_visitor_activity("Admin Auth", action="admin_logout")
            st.rerun()
            
        st.markdown("--- Visitor Data ---")
        if firebase_initialized:
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
                        label="Download Analytics Data as CSV",
                        data=csv,
                        file_name="visitor_analytics.csv",
                        mime="text/csv",
                        key="download_analytics_csv",
                        on_click=log_visitor_activity, 
                        args=("Admin Analytics",), 
                        kwargs={"action": "download_analytics_csv"}
                    )
                except Exception as e_csv:
                    st.warning(f"Could not generate CSV for download: {e_csv}")
                    log_visitor_activity("Admin Analytics", action="download_analytics_csv_fail", details={"error": str(e_csv)})
            else:
                st.info("No visitor logs found in the database.")
        else:
            st.warning("Firebase not initialized. Cannot display analytics.")

# --- About Us Section (Collapsible) --- 
st.markdown("<div class='about-us-header'>About DeepHydro AI ▼</div>", unsafe_allow_html=True)
st.markdown("""
<div class="about-us-content">
DeepHydro AI leverages advanced LSTM neural networks to provide accurate groundwater level forecasts. 
Our platform allows users to utilize a pre-trained standard model, train a custom model on their specific data, or upload their own Keras models.
We aim to provide actionable insights for water resource management through cutting-edge AI technology combined with user-friendly tools 
for visualization, reporting, and interactive analysis.

**Note on Authentication:** This version integrates Google Sign-In using `streamlit-oauth`. Ensure `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `REDIRECT_URI` environment variables are set correctly in your deployment (e.g., Render.com).

**Note on Analytics:** User activity is logged for improvement purposes. Anonymous users are tracked via a hashed identifier; logged-in users are tracked by their Google email.
</div>
""", unsafe_allow_html=True)

# --- (End of Script) --- 

