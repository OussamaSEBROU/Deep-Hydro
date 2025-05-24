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
            "is_authenticated": st.session_state.get("auth_status") == "authenticated", 