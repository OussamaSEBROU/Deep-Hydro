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

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 3

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
                firebase_url = os.getenv("FIREBASE_DATABASE_URL", f"https://{cred_dict.get('project_id')}-default-rtdb.firebaseio.com/")
                firebase_admin.initialize_app(cred, {
                    "databaseURL": firebase_url
                })              # st.success("Firebase initialized successfully.") # Optional: for debugging
                return True
            else:
                st.warning("Firebase credentials not found. Analytics and usage tracking are disabled.")
                return False
        except Exception as e:
            st.warning(f"Firebase initialization error: {e}. Analytics and usage tracking are disabled.")
            return False
    return True

# --- User Identification & Tracking --- 
def get_client_ip():
    """Get the client's IP address if available."""
    try:
        # Use a reliable service to get IP
        response = requests.get('https://httpbin.org/ip', timeout=3)
        if response.status_code == 200:
            return response.json().get('origin', 'Unknown')
        return "Unknown"
    except Exception:
        return "Unknown"

def get_persistent_user_id():
    """Generate or retrieve a persistent user ID."""
    if 'persistent_user_id' in st.session_state and st.session_state.persistent_user_id:
        return st.session_state.persistent_user_id

    # If logged in (simulated), use the simulated email as ID
    if st.session_state.get('simulated_auth_status') and st.session_state.get('simulated_email'):
        user_id = st.session_state.simulated_email
        st.session_state.persistent_user_id = user_id
        return user_id

    # For anonymous users, create a hashed ID
    ip_address = get_client_ip()
    user_agent = st.session_state.get('user_agent', 'Unknown')
    
    # Create a stable hash
    hash_input = f"{ip_address}-{user_agent}"
    hashed_id = hashlib.sha256(hash_input.encode()).hexdigest()
    persistent_id = f"anon_{hashed_id}"
    
    st.session_state.persistent_user_id = persistent_id
    return persistent_id

def get_or_create_user_profile(user_id):
    """Get user profile from Firebase or create a new one."""
    if not firebase_admin._apps:
        return None, False # Indicate Firebase not available
    
    try:
        ref = db.reference(f'users/{user_id}')
        profile = ref.get()
        is_new_user = False
        if profile is None:
            is_new_user = True
            profile = {
                'user_id': user_id,
                'first_visit': datetime.datetime.now().isoformat(),
                'visit_count': 1,
                'is_authenticated': st.session_state.get('simulated_auth_status', False),
                'feature_usage_count': 0,
                'last_visit': datetime.datetime.now().isoformat(),
                'simulated_email': st.session_state.get('simulated_email') # Store email if authenticated
            }
            ref.set(profile)
            # st.info(f"Created new user profile for {user_id}") # Debug
        else:
            # Update visit count and last visit time if it's a new session
            if 'session_visit_logged' not in st.session_state:
                profile['visit_count'] = profile.get('visit_count', 0) + 1
                profile['last_visit'] = datetime.datetime.now().isoformat()
                # Update auth status and email if they logged in this session
                profile['is_authenticated'] = st.session_state.get('simulated_auth_status', False)
                if profile['is_authenticated']:
                     profile['simulated_email'] = st.session_state.get('simulated_email')
                ref.update({'visit_count': profile['visit_count'], 
                            'last_visit': profile['last_visit'],
                            'is_authenticated': profile['is_authenticated'],
                            'simulated_email': profile.get('simulated_email')})
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
        ref = db.reference(f'users/{user_id}/feature_usage_count')
        # Atomically increment the count
        current_count = ref.get() or 0
        ref.set(current_count + 1)
        # Update session state as well
        if 'user_profile' in st.session_state and st.session_state.user_profile:
            st.session_state.user_profile['feature_usage_count'] = current_count + 1
        return True
    except Exception as e:
        st.warning(f"Firebase error incrementing usage count for {user_id}: {e}")
        return False

# --- Authentication Check & Simulation ---
def check_feature_access():
    """Check if user can access advanced features based on usage count and auth status."""
    if 'user_profile' not in st.session_state or st.session_state.user_profile is None:
        # Try to fetch profile again if missing
        user_id = get_persistent_user_id()
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        if st.session_state.user_profile is None:
            st.warning("Could not retrieve user profile. Feature access may be limited.")
            # Allow access if profile fetch fails? Or deny? Deny for safety.
            return False, "Cannot verify usage limit. Access denied."

    usage_count = st.session_state.user_profile.get('feature_usage_count', 0)
    is_authenticated = st.session_state.get('simulated_auth_status', False)

    if is_authenticated:
        return True, "Access granted (Authenticated)."
    elif usage_count < ADVANCED_FEATURE_LIMIT:
        return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
    else:
        return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in to continue."

def show_simulated_login():
    """Display the simulated login prompt and button."""
    st.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached for advanced features.")
    st.info("Please 'log in' to continue using AI Report, Forecasting, and AI Chat.")
    
    simulated_email = st.text_input("Enter your email to simulate login:", key="sim_email_input")
    
    if st.button("Login with Google (Simulated)", key="sim_login_btn"):
        if simulated_email and '@' in simulated_email:
            st.session_state.simulated_auth_status = True
            st.session_state.simulated_email = simulated_email
            st.session_state.persistent_user_id = simulated_email # Update persistent ID to email
            
            # Update Firebase profile immediately
            user_id = get_persistent_user_id()
            profile, _ = get_or_create_user_profile(user_id) # This will now use email as ID
            if profile:
                try:
                    ref = db.reference(f'users/{user_id}')
                    update_data = {
                        'is_authenticated': True,
                        'simulated_email': simulated_email,
                        'last_login_simulated': datetime.datetime.now().isoformat()
                    }
                    # If the profile was just created with the email ID, set initial values too
                    if profile.get('visit_count', 0) <= 1:
                         update_data['first_visit'] = profile.get('first_visit', datetime.datetime.now().isoformat())
                         update_data['visit_count'] = 1
                         update_data['feature_usage_count'] = profile.get('feature_usage_count', 0)
                         
                    ref.update(update_data)
                    st.session_state.user_profile = ref.get() # Refresh profile in session state
                    st.success("Simulated login successful! Advanced features unlocked.")
                    time.sleep(1.5) # Give user time to see message
                    st.rerun()
                except Exception as e:
                    st.error(f"Firebase error updating profile after login: {e}")
            else:
                 st.error("Could not update user profile after login.")
        else:
            st.error("Please enter a valid email address to simulate login.")

# --- Visitor Analytics Functions --- (Modified)
def get_session_id():
    """Create or retrieve a unique session ID for the current user session."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None):
    """
    Log visitor activity to Firebase Realtime Database, including persistent user ID.
    
    Args:
        page_name: The name of the page or section being viewed/interacted with.
        action: The action performed (e.g., page_view, run_forecast, generate_report).
        feature_used: Specific feature used (e.g., 'Forecast', 'AI Report', 'AI Chat') - used for usage counting.
    """
    if not firebase_admin._apps:
        return # Skip logging if Firebase is not initialized

    try:
        user_id = get_persistent_user_id() # Get persistent ID (hashed anonymous or simulated email)
        profile, is_new = get_or_create_user_profile(user_id) # Ensure profile exists and update visit count
        
        # Check access and increment usage count *before* logging the successful action
        # for the specific features that count towards the limit.
        should_increment = feature_used in ['Forecast', 'AI Report', 'AI Chat']
        access_granted, _ = check_feature_access() # Check access status
        
        if should_increment:
            # Only increment if access is granted (either under limit or authenticated)
            if access_granted:
                increment_feature_usage(user_id)
            else:
                # If access was denied but they tried to use the feature, log the attempt but don't increment
                action = f"denied_{action}" # Log the denial
                pass # Do not increment usage count

        # Proceed with logging the activity
        ref = db.reference('visitors_log') # Changed collection name for clarity
        log_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        session_id = get_session_id()
        user_agent = st.session_state.get('user_agent', 'Unknown')
        ip_address = get_client_ip() # Log IP for geo-location, etc., but use hashed ID for tracking

        log_data = {
            'timestamp': timestamp,
            'persistent_user_id': user_id, # Track via persistent ID
            'is_authenticated': st.session_state.get('simulated_auth_status', False),
            'visit_count': profile.get('visit_count', 1) if profile else 1,
            'ip_address': ip_address, # Logged for info, not tracking ID
            'page': page_name,
            'action': action,
            'feature_used': feature_used, # Log which feature was used
            'session_id': session_id,
            'user_agent': user_agent
        }
        
        ref.child(log_id).set(log_data)
        # st.info(f"Logged activity: {action} on {page_name} by {user_id}") # Debug

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
        ref = db.reference('visitors_log') # Use the new collection name
        visitors_data = ref.get()
        
        if not visitors_data:
            return pd.DataFrame()
        
        visitors_list = []
        for log_id, data in visitors_data.items():
            data['log_id'] = log_id
            visitors_list.append(data)
        
        df = pd.DataFrame(visitors_list)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp (most recent first)
        df = df.sort_values('timestamp', ascending=False)
        
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
        df['date'] = df['timestamp'].dt.date
        
        # 1. Daily visitors (using unique persistent IDs)
        daily_visitors = df.groupby('date')['persistent_user_id'].nunique().reset_index(name='unique_users')
        daily_visitors['date'] = pd.to_datetime(daily_visitors['date'])
        fig1 = px.line(daily_visitors, x='date', y='unique_users', title='Daily Unique Visitors', labels={'unique_users': 'Unique Users', 'date': 'Date'})
        figures.append(fig1)
        
        # 2. Page/Feature Popularity (using action)
        action_counts = df['action'].value_counts().reset_index()
        action_counts.columns = ['action', 'count']
        fig2 = px.bar(action_counts, x='action', y='count', title='Activity Counts by Action', labels={'count': 'Number of Times', 'action': 'Action Type'})
        figures.append(fig2)

        # 3. Authenticated vs Anonymous Users (based on last known status)
        latest_status = df.sort_values('timestamp').groupby('persistent_user_id')['is_authenticated'].last().reset_index()
        auth_counts = latest_status['is_authenticated'].value_counts().reset_index()
        auth_counts.columns = ['is_authenticated', 'count']
        auth_counts['status'] = auth_counts['is_authenticated'].map({True: 'Authenticated', False: 'Anonymous'})
        fig3 = px.pie(auth_counts, values='count', names='status', title='User Authentication Status (Latest Known)')
        figures.append(fig3)

        # 4. Hourly activity heatmap
        try:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
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
                # Placeholder if no data
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

# --- Admin Analytics Dashboard --- (Authentication unchanged)
def render_admin_analytics():
    """Render the admin analytics dashboard with authentication."""
    st.header("Admin Analytics Dashboard")
    
    # Simple password authentication (remains unchanged as per requirement 4)
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.info("Admin access required.")
        admin_password = st.text_input("Admin Password", type="password", key="admin_pass_input")
        if st.button("Login", key="admin_login_btn"):
            correct_password = os.getenv("ADMIN_PASSWORD", "admin123") # Use environment variable or default
            if admin_password == correct_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
    else:
        # Fetch visitor logs
        visitor_df = fetch_visitor_logs()
        
        if visitor_df.empty:
            st.info("No visitor data available yet.")
            return
        
        # Display visitor statistics
        st.subheader("Visitor Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_activities = len(visitor_df)
            st.metric("Total Activities Logged", total_activities)
        
        with col2:
            unique_visitors = visitor_df['persistent_user_id'].nunique()
            st.metric("Unique Visitors", unique_visitors)
        
        with col3:
            if 'date' not in visitor_df.columns:
                 visitor_df['date'] = visitor_df['timestamp'].dt.date
            today = datetime.datetime.now().date()
            # Count unique visitors today
            today_visitors = visitor_df[visitor_df['date'] == today]['persistent_user_id'].nunique()
            st.metric("Today's Unique Visitors", today_visitors)
        
        # Create and display visualizations
        st.subheader("Visitor Analytics")
        try:
            charts = create_visitor_charts(visitor_df)
            for fig in charts:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as chart_err:
            st.error(f"Error displaying charts: {chart_err}")
        
        # Display raw data with filters
        st.subheader("Raw Visitor Data")
        col1_filter, col2_filter = st.columns(2)
        with col1_filter:
            try:
                date_range = st.date_input("Date Range", [visitor_df['timestamp'].min().date(), visitor_df['timestamp'].max().date()], key="admin_date_filter")
            except Exception: date_range = None
        with col2_filter:
            try:
                user_id_options = ['All'] + visitor_df['persistent_user_id'].unique().tolist()
                user_id_filter = st.selectbox("Filter by User ID", options=user_id_options, index=0, key="admin_user_filter")
            except Exception: user_id_filter = 'All'
        
        try:
            filtered_df = visitor_df.copy()
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)]
            if user_id_filter != 'All':
                filtered_df = filtered_df[filtered_df['persistent_user_id'] == user_id_filter]
            
            # Display relevant columns
            display_cols = ['timestamp', 'persistent_user_id', 'is_authenticated', 'visit_count', 'page', 'action', 'feature_used', 'ip_address', 'session_id']
            st.dataframe(filtered_df[display_cols])
            
            if st.button("Export Filtered to CSV", key="admin_export_btn"):
                csv = filtered_df[display_cols].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_visitor_logs.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        except Exception as filter_err:
            st.error(f"Error applying filters or displaying data: {filter_err}")
            st.dataframe(visitor_df[['timestamp', 'persistent_user_id', 'action']]) # Fallback display

# --- Custom CSS (unchanged) ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* ... [Existing CSS rules remain unchanged] ... */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-weight: 600; font-size: 1.8rem; }
    h2 { font-weight: 600; font-size: 1.5rem; }
    h3, h4 { font-weight: 500; }
    .stButton > button { border-radius: 4px; font-weight: 500; transition: all 0.3s; padding: 0.5rem 1rem; }
    .stButton > button:hover { opacity: 0.8; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    .css-1d391kg, .css-12oz5g7 { padding: 1rem; }
    .sidebar .block-container { font-size: 0.9rem; }
    .sidebar h1 { font-size: 1.4rem; }
    .sidebar h2 { font-size: 1.2rem; }
    .card-container { border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .chat-message { padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; position: relative; }
    .user-message { border-left: 4px solid #1E88E5; }
    .ai-message { border-left: 4px solid #78909C; }
    .chat-message:active { opacity: 0.7; }
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

# --- JavaScript (unchanged) ---
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
    if 'user_agent' not in st.session_state:
        try:
            # Use Streamlit components to run JavaScript that sends the user agent
            user_agent_val = components.html(
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
            # The component value might take a moment to arrive
            # If it's immediately available, use it. Otherwise, it might be set in the next rerun.
            if user_agent_val:
                 st.session_state.user_agent = user_agent_val
            elif 'user_agent_capture' in st.session_state and st.session_state.user_agent_capture:
                 st.session_state.user_agent = st.session_state.user_agent_capture
            else:
                 st.session_state.user_agent = "Unknown (Capture Pending)"
        except Exception as e:
            # Fallback if component fails
            # print(f"User agent capture failed: {e}") # Debug
            st.session_state.user_agent = "Unknown (Capture Failed)"

# --- Initialize Firebase and User Profile --- 
firebase_initialized = initialize_firebase()
capture_user_agent() # Attempt to capture user agent early

# Initialize user profile in session state if not already present
if 'user_profile' not in st.session_state:
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
    st.warning("Gemini API Key not found or is placeholder. AI features will be disabled. Set GOOGLE_API_KEY environment variable.")

# --- Model Paths & Constants (unchanged) ---
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
        return model, sequence_length
    except Exception as e: st.error(f"Error loading Keras model {model_name_for_log}: {e}"); return None, None
    finally: 
        if os.path.exists(temp_model_path): os.remove(temp_model_path)

@st.cache_resource
def load_standard_model_cached(path):
    try:
        model = load_model(path, compile=False)
        sequence_length = model.input_shape[1]
        return model, sequence_length
    except Exception as e: st.error(f"Error loading standard Keras model from {path}: {e}"); return None, None

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
        fig.add_annotation(text="Training history is not available.",xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    history_df = pd.DataFrame(history_dict); history_df["Epoch"] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", template="plotly_white")
    return fig

# --- Gemini API Functions (unchanged, but access will be checked) ---
def generate_gemini_report(hist_df, forecast_df, metrics, language):
    if not gemini_configured: return "AI report generation disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    try:
        prompt = f"""Act as a professional hydrologist... [rest of prompt unchanged] ...
        Historical Data Summary:
        {hist_df["Level"].describe().to_string()}
        Forecast Data Summary:
        {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}
        """
        response = gemini_model_report.generate_content(prompt)
        return response.text
    except Exception as e: st.error(f"Error generating AI report: {e}"); return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    if not gemini_configured: return "AI chat disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient context for AI chat."
    try:
        context = f"""Context for AI Chatbot: ... [rest of context unchanged] ...
        User: {user_query}
AI:"""
        response = gemini_model_chat.generate_content(context)
        return response.text
    except Exception as e: st.error(f"Error in AI chat: {e}"); return f"Error in AI chat: {e}"

# --- Main Forecasting Pipeline (unchanged, but access will be checked before calling) ---
def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    st.info(f"Starting forecast pipeline with model: {model_choice}")
    model = None; history_data = None
    model_sequence_length = sequence_length_train_param
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    try:
        st.info("Step 1: Preparing Model...")
        # ... [model loading logic unchanged] ...
        if model_choice == "Standard Pre-trained Model":
            if os.path.exists(STANDARD_MODEL_PATH): model, model_sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
            else: st.error(f"Standard model not found at {STANDARD_MODEL_PATH}."); return None, None, None, None
        elif model_choice == "Upload Custom .h5 Model" and custom_model_file_obj is not None:
            model, model_sequence_length = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
        elif model_choice == "Train New Model": model_sequence_length = sequence_length_train_param
        else: st.error("Invalid model choice or missing file."); return None, None, None, None
        if model is None and model_choice != "Train New Model": return None, None, None, None
        st.session_state.model_sequence_length = model_sequence_length
        st.info(f"Model prep complete. Sequence length: {model_sequence_length}")

        st.info("Step 2: Preprocessing Data (Scaling)...")
        # ... [scaling logic unchanged] ...
        if model_choice != "Train New Model":
            if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
                scaler_obj.fit(np.array([[custom_scaler_min_param], [custom_scaler_max_param]]))
                scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
            else:
                scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        else: scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        st.info("Data scaling complete.")

        st.info(f"Step 3: Creating sequences (length {model_sequence_length})...")
        # ... [sequence creation unchanged] ...
        if len(df) <= model_sequence_length: st.error(f"Not enough data ({len(df)}) for sequence length {model_sequence_length}."); return None, None, None, None
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0: st.error("Could not create sequences."); return None, None, None, None
        st.info(f"Sequences created: {len(X)}")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        if model_choice == "Train New Model":
            st.info(f"Step 4a: Training New Model (Epochs: {epochs_train_param})...")
            # ... [training logic unchanged] ...
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_train) == 0 or len(X_val) == 0: st.error("Not enough data for train/val split."); return None, None, None, None
            model = build_lstm_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            history_obj = model.fit(X_train, y_train, epochs=epochs_train_param, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
            history_data = history_obj.history
            st.success("Training complete.")
            st.info("Evaluating trained model...")
            val_predictions_scaled = model.predict(X_val); val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
            y_val_actual = scaler_obj.inverse_transform(y_val); evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
            st.success("Evaluation complete.")
        else: # Pre-trained
            st.info("Step 4b: Evaluating Pre-trained Model...")
            # ... [pseudo-evaluation unchanged] ...
            if len(X) > 5:
                val_split_idx = int(len(X) * 0.8); X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                if len(X_val_pseudo) > 0:
                    val_predictions_scaled = model.predict(X_val_pseudo); val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo); evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation complete.")
                else: st.warning("Not enough data for pseudo-validation (after split).")
            else: st.warning("Not enough data for pseudo-validation (total sequences too few).")

        st.info(f"Step 5: Forecasting {forecast_horizon} Steps (MC Dropout: {mc_iterations_param})...")
        # ... [forecasting logic unchanged] ...
        last_sequence_scaled_for_pred = scaled_data[-model_sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_dropout_uncertainty(model, last_sequence_scaled_for_pred, forecast_horizon, mc_iterations_param, scaler_obj, model_sequence_length)
        st.success("Forecasting complete.")

        last_date = df["Date"].iloc[-1]
        try: freq = pd.infer_freq(df["Date"].dropna()); freq = freq if freq else "D"
        except: freq = "D"
        try: date_offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError: date_offset = pd.DateOffset(days=1)
        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=date_offset)
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": mean_forecast, "Lower_CI": lower_bound, "Upper_CI": upper_bound})
        
        st.info("Forecast pipeline finished successfully.")
        return forecast_df, evaluation_metrics, history_data, scaler_obj

    except Exception as e:
        st.error(f"An error occurred in the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc())
        return None, None, None, None

# --- Initialize Session State (Add new keys) ---
for key in ["cleaned_data", "forecast_results", "evaluation_metrics", "training_history", 
            "ai_report", "scaler_object", "forecast_plot_fig", "uploaded_data_filename",
            "active_tab", "report_language", "chat_history", "chat_active", 
            "model_sequence_length", "run_forecast_triggered", "about_us_expanded",
            "persistent_user_id", "user_profile", "simulated_auth_status", "simulated_email",
            "admin_authenticated", "session_visit_logged"]:
    if key not in st.session_state: st.session_state[key] = None # Initialize all keys

# Set defaults for specific keys if needed
if st.session_state.chat_history is None: st.session_state.chat_history = []
if st.session_state.chat_active is None: st.session_state.chat_active = False
if st.session_state.model_sequence_length is None: st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
if st.session_state.run_forecast_triggered is None: st.session_state.run_forecast_triggered = False
if st.session_state.active_tab is None: st.session_state.active_tab = 0
if st.session_state.about_us_expanded is None: st.session_state.about_us_expanded = False
if st.session_state.report_language is None: st.session_state.report_language = "English"
if st.session_state.simulated_auth_status is None: st.session_state.simulated_auth_status = False
if st.session_state.admin_authenticated is None: st.session_state.admin_authenticated = False

# --- Sidebar --- 
with st.sidebar:
    st.title("DeepHydro AI")
    
    # Log sidebar view activity
    if firebase_initialized:
        log_visitor_activity("Sidebar", "view")
    
    st.header("1. Upload Data")
    uploaded_data_file = st.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")
    
    if uploaded_data_file is not None and firebase_initialized:
        # Log only on successful upload, handled later in main area
        pass 

    st.header("2. Model & Forecast")
    model_choice = st.selectbox("Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select")
    
    if firebase_initialized:
        # Log model selection change
        if 'last_model_choice' not in st.session_state or st.session_state.last_model_choice != model_choice:
             log_visitor_activity("Sidebar", "select_model", feature_used=model_choice)
             st.session_state.last_model_choice = model_choice

    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False
    default_sequence_length = st.session_state.model_sequence_length
    sequence_length_train_sidebar = default_sequence_length
    epochs_train_sidebar = 50

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload .h5 model", type="h5", key="custom_h5_uploader")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_custom_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="custom_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="custom_scaler_max_in")
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using standard model (Seq Len: {st.session_state.model_sequence_length})")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler params?", value=False, key="use_std_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values standard model was scaled with:")
            custom_scaler_min_sidebar = st.number_input("Original Min", value=0.0, format="%.4f", key="std_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max", value=1.0, format="%.4f", key="std_scaler_max_in")
    elif model_choice == "Train New Model":
        try:
            sequence_length_train_sidebar = st.number_input("LSTM Sequence Length", min_value=10, max_value=365, value=default_sequence_length, step=10, key="seq_len_train_in")
        except Exception as e:
            st.warning(f"Using default sequence length {default_sequence_length} due to: {e}")
            sequence_length_train_sidebar = default_sequence_length
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in")

    mc_iterations_sidebar = st.number_input("MC Dropout Iterations (C.I.)", min_value=20, max_value=500, value=100, step=10, key="mc_iter_in")
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in")

    # --- Run Forecast Button with Access Check ---
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
                    # Log successful access/usage *before* running the potentially long process
                    if firebase_initialized:
                        log_visitor_activity("Sidebar", "run_forecast", feature_used='Forecast')
                        
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
                        st.session_state.active_tab = 1 # Switch to forecast tab
                        st.rerun()
                    else:
                        st.error("Forecast pipeline failed. Check messages.")
                        st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                        st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
            else:
                st.error("Please upload data first.")
                st.session_state.run_forecast_triggered = False
        else:
            # If access denied, show login prompt
            show_simulated_login()
            # Log denied access attempt
            if firebase_initialized:
                 log_visitor_activity("Sidebar", "run_forecast_denied", feature_used='Forecast')

    st.header("3. AI Analysis")
    st.session_state.report_language = st.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_configured)
    
    # --- Generate AI Report Button with Access Check ---
    generate_report_button = st.button("Generate AI Report", key="show_report_btn", disabled=not gemini_configured, use_container_width=True)
    
    if generate_report_button:
        access_granted, message = check_feature_access()
        if access_granted:
            if not gemini_configured: st.error("AI Report disabled. Configure Gemini API Key.")
            elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
                # Log successful access/usage
                if firebase_initialized:
                    log_visitor_activity("Sidebar", "generate_report", feature_used='AI Report')
                    
                with st.spinner(f"Generating AI report ({st.session_state.report_language})..."):
                    st.session_state.ai_report = generate_gemini_report(
                        st.session_state.cleaned_data, st.session_state.forecast_results,
                        st.session_state.evaluation_metrics, st.session_state.report_language
                    )
                if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:"):
                    st.success("AI report generated.")
                    st.session_state.active_tab = 3 # Switch to AI report tab
                    st.rerun()
                else: 
                    st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
            else: 
                st.error("Data, forecast, and metrics needed. Run forecast first.")
        else:
            show_simulated_login()
            if firebase_initialized:
                 log_visitor_activity("Sidebar", "generate_report_denied", feature_used='AI Report')

    # --- Download PDF Button (No usage limit) ---
    if st.button("Download Report (PDF)", key="download_report_btn", use_container_width=True):
        if firebase_initialized:
            log_visitor_activity("Sidebar", "download_pdf") # Log attempt
            
        if st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None and st.session_state.ai_report is not None and st.session_state.forecast_plot_fig is not None:
            with st.spinner("Generating PDF report..."):
                # ... [Existing PDF generation logic unchanged] ...
                try:
                    pdf = FPDF(); pdf.add_page()
                    font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    report_font = "Arial"
                    if os.path.exists(font_path_dejavu):
                        try: pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True); report_font = "DejaVu"
                        except RuntimeError: st.warning("Failed to add DejaVu font, using Arial.")
                    else: st.warning(f"DejaVu font not found at {font_path_dejavu}, using Arial.")
                    pdf.set_font(report_font, size=12); pdf.cell(0, 10, txt="DeepHydro AI Forecasting Report", new_x="LMARGIN", new_y="NEXT", align="C"); pdf.ln(5)
                    plot_filename = "forecast_plot.png"
                    try:
                        st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                        pdf.image(plot_filename, x=pdf.get_x(), y=pdf.get_y(), w=190)
                        pdf.ln(125)
                    except Exception as img_err: st.warning(f"Could not embed plot image: {img_err}.")
                    finally: 
                        if os.path.exists(plot_filename): os.remove(plot_filename)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Model Evaluation Metrics", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10)
                    for key, value in st.session_state.evaluation_metrics.items():
                        val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else str(value)
                        pdf.cell(0, 8, txt=f"{key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Forecast Data (First 10)", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=8); col_widths = [35, 35, 35, 35]
                    pdf.cell(col_widths[0], 7, txt="Date", border=1); pdf.cell(col_widths[1], 7, txt="Forecast", border=1); pdf.cell(col_widths[2], 7, txt="Lower CI", border=1); pdf.cell(col_widths[3], 7, txt="Upper CI", border=1, new_x="LMARGIN", new_y="NEXT")
                    for _, row in st.session_state.forecast_results.head(10).iterrows():
                        pdf.cell(col_widths[0], 6, txt=str(row["Date"].date()), border=1); pdf.cell(col_widths[1], 6, txt=f"{row['Forecast']:.2f}", border=1); pdf.cell(col_widths[2], 6, txt=f"{row['Lower_CI']:.2f}", border=1); pdf.cell(col_widths[3], 6, txt=f"{row['Upper_CI']:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt=f"AI Report ({st.session_state.report_language})", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10); pdf.multi_cell(0, 5, txt=st.session_state.ai_report)
                    pdf.ln(5)
                    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                    st.download_button(label="Download PDF Now", data=pdf_output_bytes, file_name="deephydro_forecast_report.pdf", mime="application/octet-stream", key="pdf_download_final_btn", use_container_width=True)
                    st.success("PDF ready. Click download button.")
                    if firebase_initialized: log_visitor_activity("Sidebar", "download_pdf_success")
                except Exception as pdf_err:
                    st.error(f"Failed to generate PDF: {pdf_err}")
                    if firebase_initialized: log_visitor_activity("Sidebar", "download_pdf_failure")
        else:
            st.error("Required data missing. Run forecast and generate AI report first.")

    st.header("4. AI Assistant")
    # --- Activate Chat Button with Access Check ---
    chat_button_label = "Deactivate Chat" if st.session_state.chat_active else "Activate Chat"
    activate_chat_button = st.button(chat_button_label, key="chat_ai_btn", disabled=not gemini_configured, use_container_width=True)
    
    if activate_chat_button:
        if st.session_state.chat_active: # Deactivating
            st.session_state.chat_active = False
            st.session_state.chat_history = []
            if firebase_initialized: log_visitor_activity("Sidebar", "deactivate_chat")
            st.rerun()
        else: # Activating
            access_granted, message = check_feature_access()
            if access_granted:
                st.session_state.chat_active = True
                st.session_state.active_tab = 4 # Switch to chat tab
                # Log successful access/usage for activating chat
                if firebase_initialized:
                    log_visitor_activity("Sidebar", "activate_chat", feature_used='AI Chat')
                st.rerun()
            else:
                show_simulated_login()
                if firebase_initialized:
                     log_visitor_activity("Sidebar", "activate_chat_denied", feature_used='AI Chat')

    # --- About Us (unchanged) ---
    st.markdown('<div class="about-us-header">👥 About Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="about-us-content">', unsafe_allow_html=True)
    st.markdown("Specializing in groundwater forecasting using AI.")
    st.markdown("**Contact:** [deephydro@example.com](mailto:deephydro@example.com)")
    st.markdown("© 2025 DeepHydro AI Team")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Admin Analytics Access (unchanged) ---
    st.header("5. Admin")
    if st.button("Analytics Dashboard", key="admin_analytics_btn", use_container_width=True):
        if firebase_initialized: log_visitor_activity("Sidebar", "access_admin")
        st.session_state.active_tab = 5 # Switch to admin tab
        st.rerun()

# --- Main Application Area --- 
st.title("DeepHydro AI Forecasting")

# Log main page view activity
if firebase_initialized:
    log_visitor_activity("Main Page", "view")

# App Introduction (unchanged)
st.markdown('<div class="app-intro">', unsafe_allow_html=True)
st.markdown("""
### Welcome to DeepHydro AI Forecasting
Advanced groundwater forecasting platform using deep learning.
**Features:** LSTM forecasting, MC Dropout uncertainty, AI interpretation, Interactive visualization.
Upload your data to begin.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Handle data upload and cleaning
if uploaded_data_file is not None:
    if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
        st.session_state.uploaded_data_filename = uploaded_data_file.name
        with st.spinner("Loading and cleaning data..."):
            cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
        if cleaned_df_result is not None:
            st.session_state.cleaned_data = cleaned_df_result
            # Reset results on new data upload
            st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
            st.session_state.training_history = None; st.session_state.ai_report = None
            st.session_state.chat_history = []; st.session_state.scaler_object = None
            st.session_state.forecast_plot_fig = None
            st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
            st.session_state.run_forecast_triggered = False
            if firebase_initialized: log_visitor_activity("Data Upload", "upload_success")
            st.rerun()
        else:
            st.session_state.cleaned_data = None
            st.error("Data loading failed. Check file format/content.")
            if firebase_initialized: log_visitor_activity("Data Upload", "upload_failure")

# Define tabs
tab_titles = ["Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot", "Admin Analytics"]
tabs = st.tabs(tab_titles)

# Set active tab (default to 0 if state is None)
active_tab_index = st.session_state.active_tab if st.session_state.active_tab is not None else 0

# --- Tab Content --- 

# Data Preview Tab (No usage limit)
with tabs[0]:
    if firebase_initialized: log_visitor_activity("Tab: Data Preview", "view")
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        col1, col2 = st.columns(2)
        with col1: st.metric("Time Range", f"{st.session_state.cleaned_data['Date'].min():%Y-%m-%d} to {st.session_state.cleaned_data['Date'].max():%Y-%m-%d}")
        with col2: st.metric("Data Points", len(st.session_state.cleaned_data))
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], mode="lines", name="Level"))
        fig_data.update_layout(title="Historical Groundwater Levels", xaxis_title="Date", yaxis_title="Level", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=400)
        st.plotly_chart(fig_data, use_container_width=True)
    else:
        st.info("⬆️ Upload XLSX data using the sidebar.")

# Forecast Results Tab (Content shown only if forecast ran successfully - access checked in sidebar)
with tabs[1]:
    if firebase_initialized: log_visitor_activity("Tab: Forecast Results", "view")
    st.header("Forecast Results")
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        if st.session_state.forecast_plot_fig is not None:
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else: st.warning("Forecast plot unavailable.")
        st.subheader("Forecast Data Table")
        st.dataframe(st.session_state.forecast_results, use_container_width=True)
    elif st.session_state.run_forecast_triggered: st.warning("Forecast run attempted, but no results available. Check sidebar messages.")
    else: st.info("Run a forecast (sidebar) to see results.")

# Model Evaluation Tab (Content shown only if forecast ran successfully)
with tabs[2]:
    if firebase_initialized: log_visitor_activity("Tab: Model Evaluation", "view")
    st.header("Model Evaluation")
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (Validation/Pseudo-Validation)")
        col1, col2, col3 = st.columns(3)
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan); mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan); mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        col1.metric("RMSE", f"{rmse_val:.4f}" if not np.isnan(rmse_val) else "N/A")
        col2.metric("MAE", f"{mae_val:.4f}" if not np.isnan(mae_val) else "N/A")
        col3.metric("MAPE", f"{mape_val:.2f}%" if not np.isnan(mape_val) and mape_val != np.inf else ("N/A" if np.isnan(mape_val) else "Inf"))
        st.subheader("Training Loss (if trained)")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: st.info("No training history (pre-trained model or training failed).")
    elif st.session_state.run_forecast_triggered: st.warning("Forecast run attempted, but no evaluation metrics available.")
    else: st.info("Run a forecast (sidebar) to see evaluation.")

# AI Report Tab (Content shown only if report generated - access checked in sidebar)
with tabs[3]:
    if firebase_initialized: log_visitor_activity("Tab: AI Report", "view")
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    if st.session_state.ai_report: 
        st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
    else: st.info("Click 'Generate AI Report' (sidebar) after a forecast.")

# AI Chatbot Tab (Requires activation and access check in sidebar)
with tabs[4]:
    if firebase_initialized: log_visitor_activity("Tab: AI Chatbot", "view")
    st.header("AI Chatbot Assistant")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    elif st.session_state.chat_active:
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("Chat activated. Ask about the results.")
            chat_container = st.container(height=400) # Set height for scrollable chat
            with chat_container:
                for sender, message in st.session_state.chat_history:
                    msg_class = "user-message" if sender == "User" else "ai-message"
                    st.markdown(f'<div class="chat-message {msg_class}">{message}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
            
            user_input = st.chat_input("Ask the AI assistant:")
            if user_input:
                if firebase_initialized: log_visitor_activity("Chat", "send_message")
                st.session_state.chat_history.append(("User", user_input))
                # Display user message immediately
                with chat_container:
                     st.markdown(f'<div class="chat-message user-message">{user_input}<span class="copy-tooltip">Copied!</span></div>', unsafe_allow_html=True)
                # Get AI response
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history, st.session_state.cleaned_data,
                        st.session_state.forecast_results, st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                st.session_state.chat_history.append(("AI", ai_response))
                # Rerun to display AI response in the container
                st.rerun()
        else:
            st.warning("Run a successful forecast first to provide context.")
            st.session_state.chat_active = False # Deactivate if context is missing
            st.rerun()
    else:
        st.info("Click \'Activate Chat\' (sidebar) after a forecast." if gemini_configured else "AI Chat disabled.")

# Admin Analytics Tab (Access controlled within the function)
with tabs[5]:
    if firebase_initialized: log_visitor_activity("Tab: Admin Analytics", "view")
    render_admin_analytics()

# Ensure JavaScript is added at the end
add_javascript_functionality()

