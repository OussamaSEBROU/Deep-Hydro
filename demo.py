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
from firebase_admin import credentials, db, auth # Added auth
import requests
import plotly.express as px
from dotenv import load_dotenv
import streamlit.components.v1 as components # Ensure components is imported

# --- Firebase Configuration ---
def initialize_firebase():
    """
    Initialize Firebase with secure credential management.
    Loads credentials from environment variables for secure deployment.
    """
    load_dotenv()
    if not firebase_admin._apps:
        try:
            firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if firebase_creds_json:
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                firebase_url = os.getenv("FIREBASE_DATABASE_URL", 
                                        f"https://{cred_dict.get('project_id')}-default-rtdb.firebaseio.com/")
                firebase_admin.initialize_app(cred, {
                    'databaseURL': firebase_url
                })
                print("Firebase Initialized Successfully") # Added print statement
                return True
            else:
                st.warning("Firebase credentials not found. Analytics and Usage Tracking are disabled.")
                return False
        except Exception as e:
            st.error(f"Firebase initialization error: {e}. Analytics and Usage Tracking are disabled.") # Changed to error
            return False
    return True

# --- Authentication & User Management (NEW & MODIFIED) ---

# Define advanced features subject to usage limits
ADVANCED_FEATURES = ["AI Report Generator", "Predictive Model (Forecasting)", "AI Chat Assistant"]
USAGE_LIMIT = 3

def get_persistent_user_id():
    """Get a persistent ID for the user (email if authenticated, UUID otherwise)."""
    if is_user_authenticated():
        return get_authenticated_user_email()
    else:
        if "persistent_user_id" not in st.session_state:
            st.session_state.persistent_user_id = str(uuid.uuid4())
            # Optionally: Link UUID to IP on first creation in DB for analytics
            # log_anonymous_user_creation(st.session_state.persistent_user_id, get_client_ip())
        return st.session_state.persistent_user_id

def is_user_authenticated():
    """Check if the user is currently authenticated via Google."""
    return st.session_state.get("authenticated", False) and st.session_state.get("user_email") is not None

def get_authenticated_user_email():
    """Get the email of the authenticated user."""
    return st.session_state.get("user_email")

def is_admin_user():
    """Check if the authenticated user is an admin."""
    if not is_user_authenticated():
        return False
    admin_emails_str = os.getenv("ADMIN_EMAILS", "")
    admin_emails = [email.strip() for email in admin_emails_str.split(",") if email.strip()]
    return get_authenticated_user_email() in admin_emails

# Placeholder for Google Sign-In button/logic (Requires client-side implementation)
# In a real app, this would involve a frontend component sending an ID token
# which is then verified server-side using firebase_admin.auth.verify_id_token
def display_google_login():
    """Displays a placeholder for Google Login."""
    st.warning("Authentication Required: Please sign in with Google to continue using advanced features.")
    # Simulate login for testing (replace with actual Firebase Auth flow)
    st.markdown("_(Google Sign-In integration placeholder)_ ")
    test_email = st.text_input("Enter test email to simulate login:", key="test_login_email")
    if st.button("Simulate Google Login", key="simulate_login_btn"):
        if test_email and "@" in test_email:
            st.session_state.authenticated = True
            st.session_state.user_email = test_email
            st.session_state.persistent_user_id = test_email # Update persistent ID
            st.success(f"Simulated login as {test_email}")
            # Log authentication event
            log_visitor_activity("Authentication", "login_success")
            st.rerun()
        else:
            st.error("Please enter a valid email address to simulate login.")

def display_logout():
    """Displays logout button for authenticated users."""
    if is_user_authenticated():
        st.sidebar.write(f"Logged in as: {get_authenticated_user_email()}")
        if st.sidebar.button("Logout", key="logout_btn"):
            # Log logout event
            log_visitor_activity("Authentication", "logout")
            # Clear auth state
            st.session_state.authenticated = False
            st.session_state.user_email = None
            # Reset persistent ID to a new anonymous one
            st.session_state.persistent_user_id = str(uuid.uuid4())
            st.rerun()

# --- Usage Tracking Functions (NEW) ---
def get_feature_usage_count(user_id):
    """Get the total usage count for advanced features for a given user_id."""
    if not firebase_initialized:
        return 0
    try:
        ref = db.reference(f"user_usage/{user_id}/advanced_features")
        usage_data = ref.get()
        if usage_data and isinstance(usage_data, dict):
            # Sum up usage across all tracked features
            return sum(usage_data.values())
        return 0
    except Exception as e:
        st.warning(f"Error fetching usage count for {user_id}: {e}")
        return 0

def increment_feature_usage(user_id, feature_name):
    """Increment the usage count for a specific feature for a given user_id."""
    if not firebase_initialized:
        st.warning("Firebase not initialized. Cannot track feature usage.")
        return False # Indicate failure
    
    # Only track defined advanced features
    if feature_name not in ADVANCED_FEATURES:
        # st.info(f"Feature '{feature_name}' not tracked for usage limits.")
        return True # Not a failure, just not tracked

    try:
        ref = db.reference(f"user_usage/{user_id}/advanced_features/{feature_name}")
        
        # Use a transaction to safely increment the count
        def transaction_update(current_value):
            if current_value is None:
                return 1
            return current_value + 1

        new_count = ref.transaction(transaction_update)
        
        # After incrementing, check the total count for logging/display
        total_usage = get_feature_usage_count(user_id)
        st.toast(f"Advanced feature use recorded. Total uses: {total_usage}/{USAGE_LIMIT}") # User feedback
        
        # Log the specific feature engagement in analytics
        log_visitor_activity(feature_name, "feature_engaged")
        
        return True # Indicate success
    except Exception as e:
        st.error(f"Error incrementing usage count for {user_id} / {feature_name}: {e}")
        return False # Indicate failure

def check_usage_limit_and_auth(feature_name):
    """Check if user can access the feature based on usage and auth status."""
    # Admins always have access
    if is_admin_user():
        return True, "" # Allowed, no message
        
    user_id = get_persistent_user_id()
    usage_count = get_feature_usage_count(user_id)
    authenticated = is_user_authenticated()
    
    if usage_count < USAGE_LIMIT:
        return True, f"Free uses remaining: {USAGE_LIMIT - usage_count}/{USAGE_LIMIT}" # Allowed
    elif authenticated:
        return True, "Authenticated user access." # Allowed
    else:
        # Limit reached and not authenticated
        return False, f"Free usage limit ({USAGE_LIMIT}) reached. Please authenticate with Google to continue." # Blocked

# --- Visitor Analytics Functions (MODIFIED) ---
def get_client_ip():
    """Get the client's IP address if available."""
    # Try getting IP from Streamlit headers first (more reliable in deployment)
    try:
        from streamlit.web.server.server import Server
        session_info = Server.get_current()._get_session_info(st.session_state._session_id)
        if session_info:
            return session_info.client.address
    except Exception:
        pass # Fallback to external service
    
    # Fallback using external service
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=2)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json().get("ip", "Unknown")
    except requests.exceptions.RequestException as e:
        # st.warning(f"Could not get IP from ipify: {e}")
        return "Unknown"
    except Exception as e:
        # st.warning(f"Error getting client IP: {e}")
        return "Unknown"

def get_session_id(): # Kept for potential session-level analysis
    """Create or retrieve a unique session ID for the current user session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view"):
    """
    Log visitor activity to Firebase Realtime Database with enhanced user tracking.
    
    Args:
        page_name: The name of the page or feature being interacted with.
        action: The action performed (e.g., page_view, feature_engaged, login_success).
    """
    if not firebase_initialized:
        return
    
    try:
        user_id = get_persistent_user_id() # Use the persistent ID (email or UUID)
        is_auth = is_user_authenticated()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC
        ip_address = get_client_ip()
        session_id = get_session_id() # Still useful for session analysis
        user_agent = st.session_state.get("user_agent", "Unknown")
        
        # Reference to the specific user's activity log
        user_log_ref = db.reference(f"user_activity/{user_id}")
        
        # --- Track First Visit and Visit Count --- 
        user_metadata_ref = db.reference(f"user_metadata/{user_id}")
        metadata = user_metadata_ref.get()
        
        first_visit_timestamp = None
        visit_number = 1
        
        if metadata:
            first_visit_timestamp = metadata.get("first_visit")
            # Simple visit counting: increment if last visit was > 30 mins ago (adjust as needed)
            last_visit_timestamp_str = metadata.get("last_visit")
            if last_visit_timestamp_str:
                last_visit_dt = datetime.datetime.fromisoformat(last_visit_timestamp_str)
                if datetime.datetime.now(datetime.timezone.utc) - last_visit_dt > datetime.timedelta(minutes=30):
                    visit_number = metadata.get("visit_count", 0) + 1
                else:
                    visit_number = metadata.get("visit_count", 1)
            else:
                 visit_number = metadata.get("visit_count", 1) # Should not happen if first_visit exists
        else:
            # First time seeing this user_id
            first_visit_timestamp = timestamp
            visit_number = 1
            
        # Update metadata (first visit, last visit, visit count)
        user_metadata_ref.update({
            "first_visit": first_visit_timestamp or timestamp, # Set only if not exists
            "last_visit": timestamp,
            "visit_count": visit_number,
            "last_ip": ip_address, # Keep track of last known IP
            "last_user_agent": user_agent
        })
        # --- End Visit Tracking --- 
        
        # Generate a unique ID for this specific activity event
        activity_id = str(uuid.uuid4())
        
        activity_data = {
            "timestamp": timestamp,
            "page_or_feature": page_name,
            "action": action,
            "ip_address": ip_address,
            "session_id": session_id,
            "user_agent": user_agent,
            "is_authenticated": is_auth,
            "visit_number": visit_number
            # "user_id" is the parent key in this structure
        }
        
        # Add feature engagement details if applicable
        if action == "feature_engaged":
            activity_data["feature_name"] = page_name # Log which feature was engaged
            # We could add more details here if needed

        # Push the activity data under the user's log
        user_log_ref.child(activity_id).set(activity_data)
        
    except Exception as e:
        # Silently fail for analytics to not disrupt user experience
        # Consider logging this error server-side in production
        # print(f"Error logging visitor activity: {e}") 
        pass

def fetch_visitor_logs(): # MODIFIED for new structure
    """
    Fetch visitor logs from Firebase for admin viewing.
    Returns a pandas DataFrame with the visitor data.
    """
    if not firebase_initialized:
        return pd.DataFrame()
    
    try:
        activity_ref = db.reference("user_activity")
        all_user_activity = activity_ref.get()
        
        metadata_ref = db.reference("user_metadata")
        all_user_metadata = metadata_ref.get() or {}
        
        if not all_user_activity:
            return pd.DataFrame()
        
        visitors_list = []
        for user_id, activities in all_user_activity.items():
            user_metadata = all_user_metadata.get(user_id, {})
            first_visit = user_metadata.get("first_visit", "N/A")
            total_visits = user_metadata.get("visit_count", "N/A")
            
            if activities and isinstance(activities, dict):
                for activity_id, data in activities.items():
                    if isinstance(data, dict): # Ensure data is a dictionary
                        data["activity_id"] = activity_id
                        data["user_id"] = user_id
                        data["first_visit"] = first_visit
                        data["total_visits"] = total_visits
                        # Rename for clarity in dashboard
                        data["page"] = data.pop("page_or_feature", "N/A") 
                        visitors_list.append(data)
        
        if not visitors_list:
             return pd.DataFrame()
             
        df = pd.DataFrame(visitors_list)
        
        # Convert timestamp to datetime (handle potential errors)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["first_visit"] = pd.to_datetime(df["first_visit"], errors="coerce")
        
        # Sort by timestamp (most recent first)
        df = df.sort_values("timestamp", ascending=False)
        
        # Select and reorder columns for display
        cols_to_show = [
            "timestamp", "user_id", "is_authenticated", "visit_number", 
            "total_visits", "first_visit", "page", "action", 
            "ip_address", "session_id", "user_agent", "activity_id"
        ]
        # Ensure all expected columns exist, add missing ones with NaN
        for col in cols_to_show:
            if col not in df.columns:
                df[col] = pd.NA 
                
        return df[cols_to_show]
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        import traceback
        st.error(traceback.format_exc()) # More detailed error for debugging
        return pd.DataFrame()

def create_visitor_charts(visitor_df): # MODIFIED for new fields
    """
    Create visualizations of visitor data using Plotly.
    
    Args:
        visitor_df: DataFrame containing visitor data (new structure)
    
    Returns: List of Plotly figures
    """
    if visitor_df.empty:
        return []
    
    figures = []
    
    try:
        df = visitor_df.copy()
        df["date"] = df["timestamp"].dt.date
        
        # --- Enhanced Charts --- 
        
        # 1. Daily Active Users (Authenticated vs Anonymous)
        daily_users = df.drop_duplicates(subset=["date", "user_id"])
        daily_counts = daily_users.groupby(["date", "is_authenticated"]).size().unstack(fill_value=0).reset_index()
        daily_counts["date"] = pd.to_datetime(daily_counts["date"])
        daily_counts = daily_counts.rename(columns={True: "Authenticated", False: "Anonymous"})
        
        fig1 = go.Figure()
        if "Authenticated" in daily_counts.columns:
            fig1.add_trace(go.Bar(x=daily_counts["date"], y=daily_counts["Authenticated"], name="Authenticated Users"))
        if "Anonymous" in daily_counts.columns:
            fig1.add_trace(go.Bar(x=daily_counts["date"], y=daily_counts["Anonymous"], name="Anonymous Users"))
        fig1.update_layout(barmode="stack", title="Daily Active Users (Authenticated vs. Anonymous)", 
                           xaxis_title="Date", yaxis_title="Number of Unique Users")
        figures.append(fig1)
        
        # 2. Feature Engagement Chart (using 'action' == 'feature_engaged')
        feature_engagement = df[df["action"] == "feature_engaged"]["page"].value_counts().reset_index()
        feature_engagement.columns = ["feature", "count"]
        
        if not feature_engagement.empty:
            fig2 = px.bar(feature_engagement, x="feature", y="count",
                         title="Advanced Feature Engagement",
                         labels={"count": "Number of Engagements", "feature": "Feature"})
            figures.append(fig2)
        else:
            # Placeholder if no engagement data
            fig2 = go.Figure()
            fig2.update_layout(title="Advanced Feature Engagement (No Data)")
            fig2.add_annotation(text="No advanced feature engagement recorded yet.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            figures.append(fig2)
            
        # 3. User Actions Pie Chart (excluding page_view for clarity)
        action_counts = df[df["action"] != "page_view"]["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        if not action_counts.empty:
            fig3 = px.pie(action_counts, values="count", names="action", title="User Actions (Excluding Page Views)")
            figures.append(fig3)
            
        # 4. Hourly Activity Heatmap (Remains similar, uses timestamp)
        try:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            hourly_activity = df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
            
            if not hourly_activity.empty:
                hourly_pivot = hourly_activity.pivot_table(values="count", index="day_of_week", columns="hour", fill_value=0)
                available_days = list(set(hourly_pivot.index) & set(day_order))
                ordered_available_days = [day for day in day_order if day in available_days]
                hourly_pivot = hourly_pivot.reindex(ordered_available_days)
                available_hours = sorted(hourly_pivot.columns)
                
                fig4 = px.imshow(hourly_pivot, 
                                labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
                                x=[str(h) for h in available_hours],
                                y=ordered_available_days,
                                title="User Activity by Hour and Day")
                figures.append(fig4)
            else:
                raise ValueError("No hourly activity data to pivot.")
        except Exception as heatmap_err:
            st.warning(f"Could not generate hourly activity heatmap: {heatmap_err}")
            fig4 = go.Figure()
            fig4.update_layout(title="Visitor Activity by Hour and Day (No Data/Error)")
            fig4.add_annotation(text="Not enough data or error generating heatmap.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            figures.append(fig4)
            
        # 5. New vs Returning Users (based on visit_number)
        user_types = df.drop_duplicates(subset=["user_id"])
        user_types["user_type"] = user_types["visit_number"].apply(lambda x: "New" if x == 1 else "Returning")
        user_type_counts = user_types["user_type"].value_counts().reset_index()
        user_type_counts.columns = ["type", "count"]
        if not user_type_counts.empty:
             fig5 = px.pie(user_type_counts, values="count", names="type", title="New vs. Returning Users")
             figures.append(fig5)

    except Exception as e:
        st.error(f"Error creating visitor charts: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []
    
    return figures

# --- Admin Analytics Dashboard (MODIFIED for Auth & Data) ---
def render_admin_analytics():
    """Render the admin analytics dashboard, restricted to admin users."""
    st.header("Admin Analytics Dashboard")
    
    # REQUIREMENT 4: Restricted Access to Analytics Dashboard
    if not is_user_authenticated():
        st.warning("Access Denied. Please log in with an authorized admin Google account.")
        display_google_login() # Show login prompt
        return
        
    if not is_admin_user():
        st.error(f"Access Denied. User {get_authenticated_user_email()} is not authorized to view analytics.")
        # Optionally log unauthorized access attempt
        log_visitor_activity("Admin Analytics", "access_denied")
        return
        
    # If authenticated and is admin, proceed:
    st.success(f"Welcome, Admin {get_authenticated_user_email()}!")
    log_visitor_activity("Admin Analytics", "access_granted") # Log successful access
    
    # Fetch visitor logs (using the modified function)
    visitor_df = fetch_visitor_logs()
    
    if visitor_df.empty:
        st.info("No visitor data available yet.")
        return
    
    # Display visitor statistics (using new fields)
    st.subheader("Overall User Statistics")
    col1, col2, col3 = st.columns(3)
    
    unique_users = visitor_df["user_id"].nunique()
    auth_users = visitor_df[visitor_df["is_authenticated"] == True]["user_id"].nunique()
    anon_users = unique_users - auth_users
    
    with col1:
        st.metric("Total Unique Users", unique_users)
    with col2:
        st.metric("Authenticated Users", auth_users)
    with col3:
        st.metric("Anonymous Users", anon_users)
        
    # Display total activities logged
    st.metric("Total Activities Logged", len(visitor_df))

    # Create and display visualizations (using the modified function)
    st.subheader("User Analytics Visualizations")
    try:
        charts = create_visitor_charts(visitor_df)
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as chart_err:
        st.error(f"Error displaying charts: {chart_err}")
    
    # Display raw data with filters
    st.subheader("Raw Activity Data")
    
    # Add filters (adjust based on new columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            # Ensure timestamp column exists and has valid data before filtering
            if "timestamp" in visitor_df.columns and not visitor_df["timestamp"].isnull().all():
                min_date = visitor_df["timestamp"].min().date() if pd.notna(visitor_df["timestamp"].min()) else datetime.date.today()
                max_date = visitor_df["timestamp"].max().date() if pd.notna(visitor_df["timestamp"].max()) else datetime.date.today()
                date_range = st.date_input("Filter by Date Range", [min_date, max_date])
            else:
                st.warning("Timestamp data missing or invalid for date filtering.")
                date_range = None
        except Exception as date_err:
            st.warning(f"Could not set date range filter: {date_err}")
            date_range = None
    
    with col2:
        try:
            if "page" in visitor_df.columns:
                page_options = visitor_df["page"].unique()
                page_filter = st.multiselect("Filter by Page/Feature", options=page_options, default=[])
            else:
                page_filter = []
        except Exception as page_err:
            st.warning(f"Could not set page filter: {page_err}")
            page_filter = []
            
    with col3:
        try:
            auth_filter = st.selectbox("Filter by Auth Status", ["All", "Authenticated", "Anonymous"], index=0)
        except Exception as auth_err:
            st.warning(f"Could not set auth filter: {auth_err}")
            auth_filter = "All"

    # Apply filters
    try:
        filtered_df = visitor_df.copy()
        
        # Date filter
        if date_range and len(date_range) == 2 and "timestamp" in filtered_df.columns:
            start_date, end_date = date_range
            # Ensure comparison is between date objects
            filtered_df = filtered_df[filtered_df["timestamp"].notna() & 
                                      (filtered_df["timestamp"].dt.date >= start_date) & 
                                      (filtered_df["timestamp"].dt.date <= end_date)]
        
        # Page/Feature filter
        if page_filter and "page" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["page"].isin(page_filter)]
            
        # Auth Status filter
        if "is_authenticated" in filtered_df.columns:
            if auth_filter == "Authenticated":
                filtered_df = filtered_df[filtered_df["is_authenticated"] == True]
            elif auth_filter == "Anonymous":
                filtered_df = filtered_df[filtered_df["is_authenticated"] == False]
        
        # Display the filtered data
        st.dataframe(filtered_df)
        
        # Export options
        if st.button("Export Filtered Data to CSV"): 
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_activity_logs.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            log_visitor_activity("Admin Analytics", "export_csv") # Log export
            
    except Exception as filter_err:
        st.error(f"Error applying filters or displaying data: {filter_err}")
        st.dataframe(visitor_df) # Show unfiltered data on error

# --- Custom CSS (No changes needed based on requirements) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* ... (existing CSS rules) ... */
    .stButton>button {
        /* Add a slight visual cue for disabled buttons */
    }
    .stButton>button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .locked-feature {
        opacity: 0.6;
        pointer-events: none; /* Prevent interaction */
        border: 1px dashed grey;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- JavaScript (No changes needed based on requirements) ---
def add_javascript_functionality():
    st.markdown("""
    <script>
    // ... (existing JS functions) ... 
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration --- 
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css()
add_javascript_functionality()

# --- Capture User Agent (No changes needed) ---
def capture_user_agent():
    try:
        # Use Streamlit's experimental way to get headers if possible
        # headers = st.experimental_get_query_params()
        # user_agent = headers.get("User-Agent", ["Unknown"])[0]
        # st.session_state.user_agent = user_agent
        # Fallback using JS
        user_agent_script = """
            <script>
            const userAgent = navigator.userAgent;
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: userAgent, key: 'user_agent_val'}, '*');
            </script>
            """
        result = components.html(user_agent_script, height=0, key="ua_comp")
        # The value might take a moment to arrive, handle potential None
        if "user_agent_val" in st.session_state:
             st.session_state.user_agent = st.session_state.user_agent_val
        elif "user_agent" not in st.session_state: # Initialize if completely missing
             st.session_state.user_agent = "Unknown"

    except Exception:
        if "user_agent" not in st.session_state:
             st.session_state.user_agent = "Unknown"

# --- Initialize Firebase and Get User ID --- 
firebase_initialized = initialize_firebase()

# Initialize session state keys related to auth and usage
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "user_email" not in st.session_state: st.session_state.user_email = None
if "persistent_user_id" not in st.session_state:
    # Initialize persistent_user_id early
    st.session_state.persistent_user_id = str(uuid.uuid4())

# Get the user ID for the current session
current_user_id = get_persistent_user_id() 

# Log initial page view with user ID
if firebase_initialized:
    log_visitor_activity("App Initial Load", "page_view")

# --- Gemini API Configuration (No changes needed) ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "Gemini_api_key":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.types.GenerationConfig(
            temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000
        )
        gemini_model_report = genai.GenerativeModel(
            model_name="gemini-1.5-flash", # Updated model name
            generation_config=generation_config
        )
        gemini_model_chat = genai.GenerativeModel(
            model_name="gemini-1.5-flash", # Updated model name
            generation_config=generation_config
        )
        gemini_configured = True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
else:
    st.warning("Gemini API Key not found or is placeholder. AI features will be disabled. Set GOOGLE_API_KEY environment variable.")

# --- Model Paths & Constants (No changes needed) ---
STANDARD_MODEL_PATH = "standard_model.h5"
STANDARD_MODEL_SEQUENCE_LENGTH = 60
if os.path.exists(STANDARD_MODEL_PATH):
    try:
        _std_model_temp = load_model(STANDARD_MODEL_PATH, compile=False)
        STANDARD_MODEL_SEQUENCE_LENGTH = _std_model_temp.input_shape[1]
        del _std_model_temp
    except Exception as e:
        st.warning(f"Could not load standard model from {STANDARD_MODEL_PATH} to infer sequence length: {e}. Using default {STANDARD_MODEL_SEQUENCE_LENGTH}.")
else:
    st.warning(f"Standard model file not found at path: {STANDARD_MODEL_PATH}. Please ensure it exists.")

# --- Helper Functions (Minor modifications for usage tracking) ---
@st.cache_data
def load_and_clean_data(uploaded_file_content):
    # ... (existing code) ...
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        # ... (rest of the cleaning logic) ...
        st.success("Data loaded and cleaned successfully!")
        log_visitor_activity("Data Processing", "load_clean_success") # Log success
        return df
    except Exception as e: 
        st.error(f"An unexpected error occurred during data loading/cleaning: {e}"); 
        log_visitor_activity("Data Processing", "load_clean_error") # Log error
        return None

# ... (create_sequences, load_keras_model_from_file, load_standard_model_cached, build_lstm_model) ...
# No changes needed in these core ML functions

# --- MC Dropout (No changes needed) ---
def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    # ... (existing code) ...
    return mean_preds, lower_bound, upper_bound

# --- Metrics Calculation (No changes needed) ---
def calculate_metrics(y_true, y_pred):
    # ... (existing code) ...
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions (No changes needed) ---
def create_forecast_plot(historical_df, forecast_df):
    # ... (existing code) ...
    return fig

def create_loss_plot(history_dict):
    # ... (existing code) ...
    return fig

# --- Gemini API Functions (Integrate usage check/increment) ---
def generate_gemini_report(hist_df, forecast_df, metrics, language):
    feature_name = "AI Report Generator"
    allowed, message = check_usage_limit_and_auth(feature_name)
    if not allowed:
        st.error(message)
        display_google_login() # Prompt login if limit reached
        return None # Indicate failure due to limit
        
    if not gemini_configured: return "AI report generation disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    
    # Increment usage only if allowed and successful generation is attempted
    if not increment_feature_usage(get_persistent_user_id(), feature_name):
         st.warning("Failed to record feature usage.") # Warn but proceed if possible
         
    try:
        # ... (existing prompt generation) ...
        prompt = f"..."
        response = gemini_model_report.generate_content(prompt)
        log_visitor_activity(feature_name, "generation_success") # Log success
        return response.text
    except Exception as e: 
        st.error(f"Error generating AI report: {e}"); 
        log_visitor_activity(feature_name, "generation_error") # Log error
        return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    feature_name = "AI Chat Assistant"
    # Check usage only on the *first* message of a chat session (or maybe per query? Let's do per query for simplicity now)
    allowed, message = check_usage_limit_and_auth(feature_name)
    if not allowed:
        # Don't display login here, just return the error message
        return f"Access Denied: {message}"
        
    if not gemini_configured: return "AI chat disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient context for AI chat."
    
    # Increment usage for this query
    if not increment_feature_usage(get_persistent_user_id(), feature_name):
        st.warning("Failed to record chat usage.")
        
    try:
        # ... (existing context building) ...
        context = f"..."
        response = gemini_model_chat.generate_content(context)
        log_visitor_activity(feature_name, "response_success") # Log success
        return response.text
    except Exception as e: 
        st.error(f"Error in AI chat: {e}"); 
        log_visitor_activity(feature_name, "response_error") # Log error
        return f"Error in AI chat: {e}"

# --- Main Forecasting Pipeline (Integrate usage check/increment) ---
def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    feature_name = "Predictive Model (Forecasting)"
    allowed, message = check_usage_limit_and_auth(feature_name)
    if not allowed:
        st.error(message)
        display_google_login() # Prompt login if limit reached
        return None, None, None, None # Indicate failure

    st.info(f"Starting forecast pipeline with model: {model_choice}")
    # Increment usage *before* starting the potentially long process
    if not increment_feature_usage(get_persistent_user_id(), feature_name):
        st.error("Critical error: Failed to record feature usage. Cannot proceed.")
        return None, None, None, None # Stop if usage cannot be recorded
        
    model = None
    # ... (rest of the existing pipeline logic) ...
    try:
        # ... (pipeline steps) ...
        st.info("Forecast pipeline finished successfully.")
        log_visitor_activity(feature_name, "run_success") # Log success
        return forecast_df, evaluation_metrics, history_data, scaler_obj
    except Exception as e:
        st.error(f"An error occurred in the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc())
        log_visitor_activity(feature_name, "run_error") # Log error
        return None, None, None, None

# --- Initialize Session State (Add auth/usage keys) ---
for key in ["cleaned_data", "forecast_results", "evaluation_metrics", "training_history", 
            "ai_report", "scaler_object", "forecast_plot_fig", "uploaded_data_filename",
            "active_tab", "report_language", "authenticated", "user_email", 
            "persistent_user_id", "usage_message", "show_login_prompt"]:
    if key not in st.session_state: st.session_state[key] = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_active" not in st.session_state: st.session_state.chat_active = False
if "model_sequence_length" not in st.session_state: st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
if "run_forecast_triggered" not in st.session_state: st.session_state.run_forecast_triggered = False
if "about_us_expanded" not in st.session_state: st.session_state.about_us_expanded = False
# Ensure auth defaults are set correctly
if st.session_state.authenticated is None: st.session_state.authenticated = False
# Ensure persistent_user_id is initialized if somehow missed
if st.session_state.persistent_user_id is None: st.session_state.persistent_user_id = str(uuid.uuid4())

# --- Capture User Agent --- 
capture_user_agent() # Call it to try and capture

# --- Sidebar (MODIFIED for Auth & Usage Display) ---
with st.sidebar:
    st.title("DeepHydro AI Forecasting")
    log_visitor_activity("Sidebar", "view")
    
    # Display Login/Logout status
    display_logout() # Shows email and logout button if logged in
    if not is_user_authenticated():
        # Display usage count for anonymous users
        usage_count = get_feature_usage_count(current_user_id)
        st.info(f"Free advanced feature uses: {max(0, USAGE_LIMIT - usage_count)}/{USAGE_LIMIT}")
        if usage_count >= USAGE_LIMIT:
             display_google_login() # Show login prompt in sidebar if limit reached
             
    st.divider()
    
    st.header("1. Upload Data")
    # REQUIREMENT 2 Part B: Allow Home and XLSX view even if locked
    # Data upload is allowed regardless of auth/usage status
    uploaded_data_file = st.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")
    if uploaded_data_file is not None and firebase_initialized:
        log_visitor_activity("Data Upload", f"uploaded_{uploaded_data_file.name}")

    st.header("2. Model Selection & Configuration")
    # Check if forecasting section should be disabled
    can_run_forecast, forecast_message = check_usage_limit_and_auth("Predictive Model (Forecasting)")
    forecast_disabled = not can_run_forecast
    if forecast_disabled:
        st.warning(forecast_message)
        # Optionally wrap inputs in a div with 'locked-feature' class if you want visual locking

    model_choice = st.selectbox("Choose Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select", disabled=forecast_disabled)
    if firebase_initialized:
        log_visitor_activity("Model Selection", f"selected_{model_choice}")

    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False
    default_sequence_length = st.session_state.model_sequence_length
    sequence_length_train_sidebar = default_sequence_length
    epochs_train_sidebar = 50

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload your .h5 model", type="h5", key="custom_h5_uploader", disabled=forecast_disabled)
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler parameters?", value=False, key="use_custom_scaler_cb", disabled=forecast_disabled)
        if use_custom_scaler_sidebar:
            custom_scaler_min_sidebar = st.number_input("Original Data Min", value=0.0, format="%.4f", key="custom_scaler_min_in", disabled=forecast_disabled)
            custom_scaler_max_sidebar = st.number_input("Original Data Max", value=1.0, format="%.4f", key="custom_scaler_max_in", disabled=forecast_disabled)
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using standard model. Seq length: {st.session_state.model_sequence_length}", icon="ℹ️")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler parameters?", value=False, key="use_std_scaler_cb", disabled=forecast_disabled)
        if use_custom_scaler_sidebar:
            custom_scaler_min_sidebar = st.number_input("Original Data Min", value=0.0, format="%.4f", key="std_scaler_min_in", disabled=forecast_disabled)
            custom_scaler_max_sidebar = st.number_input("Original Data Max", value=1.0, format="%.4f", key="std_scaler_max_in", disabled=forecast_disabled)
    elif model_choice == "Train New Model":
        try:
            sequence_length_train_sidebar = st.number_input("LSTM Sequence Length", min_value=10, max_value=365, value=default_sequence_length, step=10, key="seq_len_train_in", disabled=forecast_disabled)
        except Exception as e:
            st.warning(f"Using default sequence length {default_sequence_length}: {e}")
            sequence_length_train_sidebar = default_sequence_length
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in", disabled=forecast_disabled)

    mc_iterations_sidebar = st.number_input("MC Dropout Iterations", min_value=20, max_value=500, value=100, step=10, key="mc_iter_in", disabled=forecast_disabled)
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in", disabled=forecast_disabled)

    if st.button("Run Forecast", key="run_forecast_main_btn", use_container_width=True, disabled=forecast_disabled):
        st.session_state.run_forecast_triggered = True
        if st.session_state.cleaned_data is not None:
            if model_choice == "Upload Custom .h5 Model" and custom_model_file_obj_sidebar is None:
                st.error("Please upload a custom .h5 model file.")
                st.session_state.run_forecast_triggered = False
            else:
                # Usage check is now inside run_forecast_pipeline
                with st.spinner(f"Running forecast with {model_choice}... This may take time."):
                    forecast_df, metrics, history, scaler_obj = run_forecast_pipeline(
                        st.session_state.cleaned_data, model_choice, forecast_horizon_sidebar, 
                        custom_model_file_obj_sidebar, sequence_length_train_sidebar, epochs_train_sidebar, 
                        mc_iterations_sidebar, use_custom_scaler_sidebar, custom_scaler_min_sidebar, custom_scaler_max_sidebar
                    )
                # Check if pipeline returned results (it returns None if usage limit hit)
                if forecast_df is not None:
                    st.session_state.forecast_results = forecast_df
                    st.session_state.evaluation_metrics = metrics
                    st.session_state.training_history = history
                    st.session_state.scaler_object = scaler_obj
                    st.session_state.forecast_plot_fig = create_forecast_plot(st.session_state.cleaned_data, forecast_df)
                    st.success("Forecast complete! Results available.")
                    st.session_state.ai_report = None; st.session_state.chat_history = []; st.session_state.chat_active = False
                    st.session_state.active_tab = 1 # Switch to forecast tab
                    st.rerun()
                else:
                    # Error message already shown in pipeline or usage check
                    st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                    st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
                    # Don't rerun if blocked by usage limit, login prompt is shown
        else:
            st.error("Please upload data first.")
            st.session_state.run_forecast_triggered = False

    st.header("3. View & Export")
    # Check if AI Report section should be disabled
    can_gen_report, report_message = check_usage_limit_and_auth("AI Report Generator")
    report_disabled = not can_gen_report or not gemini_configured or st.session_state.forecast_results is None
    if not can_gen_report and st.session_state.forecast_results is not None:
         st.warning(report_message)
         
    st.session_state.report_language = st.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=report_disabled)
    
    if st.button("Generate AI Report", key="show_report_btn", disabled=report_disabled, use_container_width=True):
        # Usage check is now inside generate_gemini_report
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            with st.spinner(f"Generating AI report ({st.session_state.report_language})..."):
                generated_report = generate_gemini_report(
                    st.session_state.cleaned_data, st.session_state.forecast_results,
                    st.session_state.evaluation_metrics, st.session_state.report_language
                )
            # Check if report generation was successful (not None)
            if generated_report is not None:
                 st.session_state.ai_report = generated_report
                 if not st.session_state.ai_report.startswith("Error:"):
                     st.success("AI report generated.")
                     st.session_state.active_tab = 3 # Switch to report tab
                 else: 
                     st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
                 st.rerun()
            # else: Error/login prompt already shown by generate_gemini_report
        else: 
            st.error("Cleaned data, forecast results, and evaluation metrics required.")
    
    # PDF Download button - enable if report exists
    pdf_disabled = st.session_state.ai_report is None
    if st.button("Download Report (PDF)", key="download_report_btn", use_container_width=True, disabled=pdf_disabled):
        # ... (existing PDF generation logic - no usage limit here) ...
        log_visitor_activity("PDF Report", "download_attempt")
        # ... (rest of PDF generation) ...

    st.header("4. AI Assistant")
    # Check if Chat section should be disabled
    can_use_chat, chat_message = check_usage_limit_and_auth("AI Chat Assistant")
    # Also disable if no forecast context or Gemini not configured
    chat_disabled = not can_use_chat or not gemini_configured or st.session_state.forecast_results is None
    if not can_use_chat and st.session_state.forecast_results is not None:
         st.warning(chat_message)
         
    if st.button("Activate/Deactivate Chat", key="chat_ai_btn", disabled=chat_disabled, use_container_width=True):
        log_visitor_activity("Chat", "toggle_chat")
        st.session_state.chat_active = not st.session_state.chat_active
        if not st.session_state.chat_active: st.session_state.chat_history = []
        else: st.session_state.active_tab = 4 # Switch to chat tab
        st.rerun()
    
    # About Us section (No changes needed)
    # ... (existing About Us markdown) ...
    
    # Admin Analytics Access (Button always visible, access checked in render_admin_analytics)
    st.header("5. Admin")
    if st.button("Analytics Dashboard", key="admin_analytics_btn", use_container_width=True):
        log_visitor_activity("Admin", "access_analytics_attempt")
        st.session_state.active_tab = 5 # Switch to admin tab
        st.rerun()

# --- Main Application Area (MODIFIED for conditional tab access) ---
st.title("DeepHydro AI Forecasting")
log_visitor_activity("Main Page", "view")

# App Introduction (No changes needed)
# ... (existing intro markdown) ...

# Data Loading Logic (No changes needed)
if uploaded_data_file is not None:
    # ... (existing data loading logic) ...
    pass # Keep existing logic

# --- Define Tabs --- 
# REQUIREMENT 2 Part B: Allow Home and XLSX view even if locked
# Tab names
tab_titles = [
    "Data Preview", # Always accessible
    "Forecast Results", 
    "Model Evaluation", 
    "AI Report", 
    "AI Chatbot", 
    "Admin Analytics" # Access controlled internally
]

# Determine which tabs should be accessible based on usage/auth
# Data Preview (Index 0) and Admin (Index 5) have special handling
can_access_forecast, _ = check_usage_limit_and_auth("Predictive Model (Forecasting)")
can_access_report, _ = check_usage_limit_and_auth("AI Report Generator")
can_access_chat, _ = check_usage_limit_and_auth("AI Chat Assistant")

# Create tabs
tabs = st.tabs(tab_titles)

# --- Tab Content --- 

# Data Preview Tab (Index 0) - Always Accessible
with tabs[0]:
    log_visitor_activity("Tab: Data Preview", "view")
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        # ... (existing data preview content) ...
        st.dataframe(st.session_state.cleaned_data)
        # ... (metrics and plot) ...
    else:
        st.info("⬆️ Please upload an XLSX data file using the sidebar.")

# Forecast Results Tab (Index 1)
with tabs[1]:
    log_visitor_activity("Tab: Forecast Results", "view_attempt")
    if not can_access_forecast and st.session_state.cleaned_data is not None:
        st.warning(f"Access restricted. {forecast_message}")
        if not is_user_authenticated(): display_google_login()
    else:
        log_visitor_activity("Tab: Forecast Results", "view_granted")
        st.header("Forecast Results")
        if st.session_state.forecast_results is not None: 
             # ... (existing forecast display content) ...
             if st.session_state.forecast_plot_fig: st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
             st.dataframe(st.session_state.forecast_results, use_container_width=True)
        elif st.session_state.run_forecast_triggered: 
             st.warning("Forecast run, but no results. Check errors or usage limits.")
        else: 
             st.info("Run a forecast using the sidebar.")

# Model Evaluation Tab (Index 2)
with tabs[2]:
    log_visitor_activity("Tab: Model Evaluation", "view_attempt")
    # Access tied to forecast results being available (implicitly requires forecast access)
    if not can_access_forecast and st.session_state.cleaned_data is not None:
         st.warning(f"Access restricted. {forecast_message}")
         if not is_user_authenticated(): display_google_login()
    elif st.session_state.evaluation_metrics is None and st.session_state.run_forecast_triggered:
         st.warning("Forecast run, but no evaluation metrics. Check errors or usage limits.")
    elif st.session_state.evaluation_metrics is None:
         st.info("Run a forecast to see model evaluation metrics.")
    else:
        log_visitor_activity("Tab: Model Evaluation", "view_granted")
        st.header("Model Evaluation")
        # ... (existing evaluation display content) ...
        col1, col2, col3 = st.columns(3)
        # ... (display metrics) ...
        if st.session_state.training_history: # Display loss plot
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: 
            st.info("No training history available.")

# AI Report Tab (Index 3)
with tabs[3]:
    log_visitor_activity("Tab: AI Report", "view_attempt")
    if not can_access_report and st.session_state.forecast_results is not None:
        st.warning(f"Access restricted. {report_message}")
        if not is_user_authenticated(): display_google_login()
    elif not gemini_configured:
        st.warning("AI features disabled. Configure Gemini API Key.")
    else:
        log_visitor_activity("Tab: AI Report", "view_granted")
        st.header("AI-Generated Scientific Report")
        if st.session_state.ai_report: 
            st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
        else: 
            st.info("Click \"Generate AI Report\" in sidebar after a forecast.")

# AI Chatbot Tab (Index 4)
with tabs[4]:
    log_visitor_activity("Tab: AI Chatbot", "view_attempt")
    if not can_access_chat and st.session_state.forecast_results is not None:
        st.warning(f"Access restricted. {chat_message}")
        if not is_user_authenticated(): display_google_login()
    elif not gemini_configured:
        st.warning("AI features disabled. Configure Gemini API Key.")
    elif not st.session_state.chat_active:
        st.info("Click \"Activate/Deactivate Chat\" in sidebar (requires forecast results).")
    elif st.session_state.forecast_results is None:
        st.warning("Run a successful forecast to provide context for the chatbot.")
        st.session_state.chat_active = False # Deactivate if context lost
    else:
        log_visitor_activity("Tab: AI Chatbot", "view_granted")
        st.header("AI Chatbot Assistant")
        st.info("Chat activated. Ask about the results.")
        chat_container = st.container()
        with chat_container:
            for sender, message in st.session_state.chat_history:
                css_class = "user-message" if sender == "User" else "ai-message"
                st.markdown(f'<div class="chat-message {css_class}">{message}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
        
        user_input = st.chat_input("Ask the AI assistant:")
        if user_input:
            st.session_state.chat_history.append(("User", user_input))
            with st.spinner("AI thinking..."):
                # Usage check/increment is inside get_gemini_chat_response
                ai_response = get_gemini_chat_response(
                    user_input, st.session_state.chat_history, st.session_state.cleaned_data,
                    st.session_state.forecast_results, st.session_state.evaluation_metrics, st.session_state.ai_report
                )
            st.session_state.chat_history.append(("AI", ai_response))
            st.rerun()

# Admin Analytics Tab (Index 5)
with tabs[5]:
    log_visitor_activity("Tab: Admin Analytics", "view_attempt")
    # Access control is handled *inside* render_admin_analytics
    render_admin_analytics()
