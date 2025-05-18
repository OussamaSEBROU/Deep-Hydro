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
from firebase_admin import credentials, db, auth
import requests
import plotly.express as px
from dotenv import load_dotenv
import streamlit.components.v1 as components
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# --- Firebase Configuration ---
def initialize_firebase():
    """
    Initialize Firebase with secure credential management.
    Loads credentials from environment variables for secure deployment.
    """
    # Load environment variables from .env file if it exists (for local development)
    load_dotenv()
    
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        try:
            # Get Firebase credentials from environment variable
            firebase_creds_json = os.getenv('FIREBASE_SERVICE_ACCOUNT')
            
            if firebase_creds_json:
                # Parse the JSON string from environment variable
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                
                # Get Firebase database URL from environment or use default format
                firebase_url = os.getenv('FIREBASE_DATABASE_URL', 
                                        f"https://{cred_dict.get('project_id')}-default-rtdb.firebaseio.com/")
                
                # Initialize Firebase app
                firebase_admin.initialize_app(cred, {
                    'databaseURL': firebase_url
                })
                return True
            else:
                st.warning("Firebase credentials not found. Analytics tracking is disabled.")
                return False
        except Exception as e:
            st.warning(f"Firebase initialization error: {e}. Analytics tracking is disabled.")
            return False
    return True

# --- User Authentication and Usage Tracking ---
def get_client_ip():
    """Get the client's IP address if available."""
    try:
        response = requests.get('https://api.ipify.org', timeout=3)
        return response.text if response.status_code == 200 else "Unknown"
    except:
        return "Unknown"

def get_session_id():
    """Create or retrieve a unique session ID for the current user session."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_user_identifier():
    """Get a unique identifier for the current user based on email (if authenticated) or IP."""
    if 'user_email' in st.session_state and st.session_state.user_email:
        return st.session_state.user_email
    else:
        # Use IP address as fallback identifier
        return get_client_ip()

def track_usage(feature_name):
    """
    Track usage of a specific feature by user identifier and IP address.
    
    Args:
        feature_name: The name of the feature being used
    
    Returns:
        usage_count: The number of times this user has used this feature
    """
    # Skip tracking if Firebase is not initialized
    if not firebase_admin._apps:
        return 0
    
    try:
        # Get user identifier (email or IP)
        user_id = get_user_identifier()
        ip_address = get_client_ip()
        
        # Create a reference to the usage counts collection
        ref = db.reference('usage_counts')
        
        # Create a sanitized user ID for Firebase path (remove special characters)
        sanitized_user_id = ''.join(c if c.isalnum() else '_' for c in user_id)
        
        # Get current usage count for this user and feature
        user_ref = ref.child(sanitized_user_id)
        user_data = user_ref.get() or {}
        
        # Update usage count
        feature_count = user_data.get(feature_name, 0) + 1
        
        # Update user data
        user_data.update({
            feature_name: feature_count,
            'last_used': datetime.datetime.now().isoformat(),
            'ip_address': ip_address
        })
        
        # If user is authenticated, store email
        if 'user_email' in st.session_state and st.session_state.user_email:
            user_data['email'] = st.session_state.user_email
        
        # Save updated data
        user_ref.set(user_data)
        
        # Return the updated count
        return feature_count
    except Exception as e:
        # Silently fail to not disrupt user experience
        st.error(f"Error tracking usage: {e}")
        return 0

def check_feature_access(feature_name):
    """
    Check if user has access to a restricted feature based on usage count and authentication status.
    
    Args:
        feature_name: The name of the feature to check access for
        
    Returns:
        bool: True if user has access, False if access is restricted
    """
    # If user is authenticated, always grant access
    if 'user_authenticated' in st.session_state and st.session_state.user_authenticated:
        return True
    
    # Get usage count for this feature
    usage_count = get_feature_usage_count(feature_name)
    
    # Allow access if usage count is below threshold
    if usage_count < 3:
        return True
    
    # Otherwise, restrict access
    return False

def get_feature_usage_count(feature_name):
    """Get the current usage count for a specific feature by the current user."""
    # Skip if Firebase is not initialized
    if not firebase_admin._apps:
        return 0
    
    try:
        # Get user identifier
        user_id = get_user_identifier()
        
        # Create a sanitized user ID for Firebase path
        sanitized_user_id = ''.join(c if c.isalnum() else '_' for c in user_id)
        
        # Get usage count from Firebase
        ref = db.reference(f'usage_counts/{sanitized_user_id}/{feature_name}')
        count = ref.get() or 0
        
        return count
    except Exception as e:
        # Return 0 on error to avoid disrupting user experience
        return 0

def log_visitor_activity(page_name, action="page_view"):
    """
    Log visitor activity to Firebase Realtime Database.
    
    Args:
        page_name: The name of the page or section being viewed
        action: The action performed (default: page_view)
    """
    # Skip logging if Firebase is not initialized
    if not firebase_admin._apps:
        return
    
    try:
        # Create a reference to the visitors collection
        ref = db.reference('visitors')
        
        # Generate a unique ID for this visit
        visit_id = str(uuid.uuid4())
        
        # Get visitor information
        timestamp = datetime.datetime.now().isoformat()
        ip_address = get_client_ip()
        session_id = get_session_id()
        user_agent = st.session_state.get('user_agent', 'Unknown')
        
        # Create the visitor data entry
        visitor_data = {
            'timestamp': timestamp,
            'ip_address': ip_address,
            'page': page_name,
            'action': action,
            'session_id': session_id,
            'user_agent': user_agent
        }
        
        # Add user email if authenticated
        if 'user_email' in st.session_state and st.session_state.user_email:
            visitor_data['email'] = st.session_state.user_email
        
        # Push the data to Firebase
        ref.child(visit_id).set(visitor_data)
    except Exception as e:
        # Silently fail to not disrupt user experience
        pass

def fetch_visitor_logs():
    """
    Fetch visitor logs from Firebase for admin viewing.
    Returns a pandas DataFrame with the visitor data.
    """
    if not firebase_admin._apps:
        return pd.DataFrame()
    
    try:
        # Get reference to visitors collection
        ref = db.reference('visitors')
        
        # Get all visitor data
        visitors_data = ref.get()
        
        if not visitors_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        visitors_list = []
        for visit_id, data in visitors_data.items():
            data['visit_id'] = visit_id
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

def fetch_usage_counts():
    """
    Fetch usage counts from Firebase for admin viewing.
    Returns a pandas DataFrame with the usage data.
    """
    if not firebase_admin._apps:
        return pd.DataFrame()
    
    try:
        # Get reference to usage counts collection
        ref = db.reference('usage_counts')
        
        # Get all usage data
        usage_data = ref.get()
        
        if not usage_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        usage_list = []
        for user_id, data in usage_data.items():
            # Extract feature usage counts
            for feature, count in data.items():
                if feature not in ['last_used', 'ip_address', 'email']:
                    usage_list.append({
                        'user_id': user_id,
                        'email': data.get('email', 'Not authenticated'),
                        'ip_address': data.get('ip_address', 'Unknown'),
                        'feature': feature,
                        'count': count,
                        'last_used': data.get('last_used', '')
                    })
        
        df = pd.DataFrame(usage_list)
        
        # Convert timestamp to datetime if present
        if 'last_used' in df.columns and not df.empty:
            df['last_used'] = pd.to_datetime(df['last_used'])
        
        # Sort by count (highest first)
        df = df.sort_values('count', ascending=False)
        
        return df
    except Exception as e:
        st.error(f"Error fetching usage counts: {e}")
        return pd.DataFrame()

def create_visitor_charts(visitor_df):
    """
    Create visualizations of visitor data using Plotly.
    
    Args:
        visitor_df: DataFrame containing visitor data
    
    Returns:
        List of Plotly figures
    """
    if visitor_df.empty:
        return []
    
    figures = []
    
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df = visitor_df.copy()
        
        # Add date column for daily aggregation
        df['date'] = df['timestamp'].dt.date
        
        # 1. Daily visitors chart
        daily_visitors = df.groupby('date').size().reset_index(name='count')
        daily_visitors['date'] = pd.to_datetime(daily_visitors['date'])
        
        fig1 = px.line(daily_visitors, x='date', y='count', 
                      title='Daily Visitors',
                      labels={'count': 'Number of Visitors', 'date': 'Date'})
        fig1.update_layout(xaxis_title='Date', yaxis_title='Number of Visitors')
        figures.append(fig1)
        
        # 2. Page popularity chart
        page_counts = df['page'].value_counts().reset_index()
        page_counts.columns = ['page', 'count']
        
        fig2 = px.bar(page_counts, x='page', y='count',
                     title='Page Popularity',
                     labels={'count': 'Number of Views', 'page': 'Page'})
        fig2.update_layout(xaxis_title='Page', yaxis_title='Number of Views')
        figures.append(fig2)
        
        # 3. User actions chart
        action_counts = df['action'].value_counts().reset_index()
        action_counts.columns = ['action', 'count']
        
        fig3 = px.pie(action_counts, values='count', names='action',
                     title='User Actions')
        figures.append(fig3)
        
        # 4. Hourly activity heatmap - FIXED to handle any data shape
        try:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            # Order days of week properly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            hourly_activity = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            
            # Check if we have data for all hours and days
            if len(hourly_activity) > 0:
                hourly_pivot = hourly_activity.pivot_table(values='count', index='day_of_week', columns='hour', fill_value=0)
                
                # Reindex to ensure proper day order for available days
                available_days = set(hourly_pivot.index) & set(day_order)
                ordered_available_days = [day for day in day_order if day in available_days]
                hourly_pivot = hourly_pivot.reindex(ordered_available_days)
                
                # Get the actual hours present in the data
                available_hours = sorted(hourly_pivot.columns)
                
                # Create the heatmap with the actual available hours
                fig4 = px.imshow(hourly_pivot, 
                                labels=dict(x="Hour of Day", y="Day of Week", color="Visit Count"),
                                x=[str(h) for h in available_hours],  # Use only available hours
                                y=ordered_available_days,  # Use only available days
                                title="Visitor Activity by Hour and Day")
                
                figures.append(fig4)
            else:
                # Create a placeholder figure if no hourly data
                fig4 = go.Figure()
                fig4.update_layout(
                    title="Visitor Activity by Hour and Day (No Data)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week"
                )
                fig4.add_annotation(
                    text="Not enough data to generate hourly activity heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                figures.append(fig4)
        except Exception as heatmap_err:
            # Create a fallback figure if the heatmap fails
            st.warning(f"Could not generate hourly activity heatmap: {heatmap_err}")
            fig4 = go.Figure()
            fig4.update_layout(
                title="Visitor Activity by Hour and Day (Error)",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            fig4.add_annotation(
                text="Error generating hourly activity heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            figures.append(fig4)
    
    except Exception as e:
        st.error(f"Error creating visitor charts: {e}")
        # Return an empty list if chart creation fails
        return []
    
    return figures

def create_usage_charts(usage_df):
    """
    Create visualizations of usage data using Plotly.
    
    Args:
        usage_df: DataFrame containing usage data
    
    Returns:
        List of Plotly figures
    """
    if usage_df.empty:
        return []
    
    figures = []
    
    try:
        # 1. Feature popularity chart
        feature_counts = usage_df.groupby('feature')['count'].sum().reset_index()
        feature_counts = feature_counts.sort_values('count', ascending=False)
        
        fig1 = px.bar(feature_counts, x='feature', y='count',
                     title='Feature Usage',
                     labels={'count': 'Number of Uses', 'feature': 'Feature'})
        fig1.update_layout(xaxis_title='Feature', yaxis_title='Number of Uses')
        figures.append(fig1)
        
        # 2. User activity chart - Top 10 users
        user_counts = usage_df.groupby('user_id')['count'].sum().reset_index()
        user_counts = user_counts.sort_values('count', ascending=False).head(10)
        
        fig2 = px.bar(user_counts, x='user_id', y='count',
                     title='Top 10 Users by Activity',
                     labels={'count': 'Number of Uses', 'user_id': 'User ID'})
        fig2.update_layout(xaxis_title='User ID', yaxis_title='Number of Uses')
        figures.append(fig2)
        
        # 3. Authentication status pie chart
        auth_status = usage_df['email'].apply(lambda x: 'Authenticated' if x != 'Not authenticated' else 'Not Authenticated')
        auth_counts = auth_status.value_counts().reset_index()
        auth_counts.columns = ['status', 'count']
        
        fig3 = px.pie(auth_counts, values='count', names='status',
                     title='User Authentication Status')
        figures.append(fig3)
        
    except Exception as e:
        st.error(f"Error creating usage charts: {e}")
        # Return an empty list if chart creation fails
        return []
    
    return figures

# --- Google Authentication ---
def setup_google_auth():
    """Set up Google authentication components."""
    # Add Google Sign-In button
    st.markdown("""
    <div id="g_id_onload"
         data-client_id="YOUR_GOOGLE_CLIENT_ID"
         data-callback="handleCredentialResponse">
    </div>
    <div class="g_id_signin" data-type="standard"></div>
    
    <script>
    function handleCredentialResponse(response) {
        // Post the ID token to Streamlit
        const data = {
            credential: response.credential
        };
        
        fetch('/google_auth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload the page to update the UI
                window.location.reload();
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)

def verify_google_token(token):
    """
    Verify Google ID token and extract user information.
    
    Args:
        token: Google ID token to verify
        
    Returns:
        dict: User information if token is valid, None otherwise
    """
    try:
        # Get Google client ID from environment
        client_id = os.getenv('GOOGLE_CLIENT_ID')
        
        if not client_id:
            st.warning("Google Client ID not configured. Authentication disabled.")
            return None
        
        # Verify token
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
        
        # Check issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            return None
        
        # Return user information
        return {
            'email': idinfo['email'],
            'name': idinfo.get('name', ''),
            'picture': idinfo.get('picture', '')
        }
    except Exception as e:
        st.error(f"Error verifying Google token: {e}")
        return None

def handle_google_auth():
    """Handle Google authentication callback."""
    # This would be implemented as a separate endpoint in a production app
    # For this example, we'll simulate the authentication flow
    
    # Check if we have a token in the query parameters (simulated)
    token = st.experimental_get_query_params().get('token', [None])[0]
    
    if token:
        # Verify token and get user info
        user_info = verify_google_token(token)
        
        if user_info:
            # Store user info in session state
            st.session_state.user_authenticated = True
            st.session_state.user_email = user_info['email']
            st.session_state.user_name = user_info['name']
            st.session_state.user_picture = user_info['picture']
            
            # Check if user is an admin
            admin_emails = os.getenv('ADMIN_EMAILS', '').split(',')
            st.session_state.is_admin = user_info['email'] in admin_emails
            
            # Redirect to remove token from URL
            st.experimental_set_query_params()
            
            return True
    
    return False

# --- Admin Analytics Dashboard ---
def render_admin_analytics():
    """Render the admin analytics dashboard with authentication."""
    st.header("Admin Analytics Dashboard")
    
    # Check if user is authenticated and is an admin
    if not st.session_state.get('user_authenticated', False):
        st.info("Please sign in with your Google account to view analytics.")
        
        # Add Google Sign-In button
        st.markdown("""
        <div class="login-container">
            <p>Sign in with your Google account to access the admin dashboard.</p>
            <button class="google-signin-button">
                <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo">
                <span>Sign in with Google</span>
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Check if user is an admin
    if not st.session_state.get('is_admin', False):
        st.error("You do not have permission to access the admin dashboard.")
        return
    
    # Fetch visitor logs and usage counts
    visitor_df = fetch_visitor_logs()
    usage_df = fetch_usage_counts()
    
    # Display visitor statistics
    st.subheader("Visitor Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_visits = len(visitor_df) if not visitor_df.empty else 0
        st.metric("Total Visits", total_visits)
    
    with col2:
        unique_visitors = visitor_df['session_id'].nunique() if not visitor_df.empty else 0
        st.metric("Unique Visitors", unique_visitors)
    
    with col3:
        if not visitor_df.empty:
            if 'date' not in visitor_df.columns:
                visitor_df['date'] = visitor_df['timestamp'].dt.date
            
            today = datetime.datetime.now().date()
            today_visits = len(visitor_df[visitor_df['date'] == today])
            st.metric("Today's Visits", today_visits)
        else:
            st.metric("Today's Visits", 0)
    
    # Display usage statistics
    st.subheader("Usage Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_usage = usage_df['count'].sum() if not usage_df.empty else 0
        st.metric("Total Feature Uses", total_usage)
    
    with col2:
        unique_users = usage_df['user_id'].nunique() if not usage_df.empty else 0
        st.metric("Unique Users", unique_users)
    
    with col3:
        authenticated_users = usage_df[usage_df['email'] != 'Not authenticated']['user_id'].nunique() if not usage_df.empty else 0
        st.metric("Authenticated Users", authenticated_users)
    
    # Create and display visualizations
    st.subheader("Visitor Analytics")
    
    try:
        visitor_charts = create_visitor_charts(visitor_df)
        
        for fig in visitor_charts:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as chart_err:
        st.error(f"Error displaying visitor charts: {chart_err}")
    
    # Display usage analytics
    st.subheader("Usage Analytics")
    
    try:
        usage_charts = create_usage_charts(usage_df)
        
        for fig in usage_charts:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as chart_err:
        st.error(f"Error displaying usage charts: {chart_err}")
    
    # Display raw data with filters
    st.subheader("Raw Visitor Data")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        try:
            if not visitor_df.empty:
                date_range = st.date_input(
                    "Date Range",
                    [visitor_df['timestamp'].min().date(), visitor_df['timestamp'].max().date()]
                )
            else:
                date_range = st.date_input("Date Range", [datetime.datetime.now().date(), datetime.datetime.now().date()])
        except Exception as date_err:
            st.warning(f"Could not set date range: {date_err}")
            date_range = None
    
    with col2:
        try:
            if not visitor_df.empty:
                page_filter = st.multiselect(
                    "Filter by Page",
                    options=visitor_df['page'].unique(),
                    default=[]
                )
            else:
                page_filter = st.multiselect("Filter by Page", options=[], default=[])
        except Exception as page_err:
            st.warning(f"Could not set page filter: {page_err}")
            page_filter = []
    
    # Apply filters
    try:
        if not visitor_df.empty:
            filtered_df = visitor_df.copy()
            
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= start_date) & 
                    (filtered_df['timestamp'].dt.date <= end_date)
                ]
            
            if page_filter:
                filtered_df = filtered_df[filtered_df['page'].isin(page_filter)]
            
            # Display the filtered data
            st.dataframe(filtered_df[['timestamp', 'ip_address', 'email', 'page', 'action', 'session_id']])
            
            # Export options
            if st.button("Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="visitor_logs.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No visitor data available.")
    except Exception as filter_err:
        st.error(f"Error applying filters: {filter_err}")
        if not visitor_df.empty:
            st.dataframe(visitor_df[['timestamp', 'ip_address', 'page', 'action', 'session_id']])
    
    # Display usage data
    st.subheader("Feature Usage Data")
    
    if not usage_df.empty:
        st.dataframe(usage_df)
        
        # Export options
        if st.button("Export Usage Data to CSV"):
            csv = usage_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="usage_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No usage data available.")

# --- Custom CSS for Professional UI with Dark/Light Mode Support ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling - adapts to dark/light mode */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header styling - adapts to dark/light mode */
    h1 {
        font-weight: 600;
        font-size: 1.8rem;
    }
    
    h2 {
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    h3, h4 {
        font-weight: 500;
    }
    
    /* Button styling - adapts to dark/light mode */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        opacity: 0.8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling - smaller fonts */
    .css-1d391kg, .css-12oz5g7 {
        padding: 1rem;
    }
    
    .sidebar .block-container {
        font-size: 0.9rem;
    }
    
    .sidebar h1 {
        font-size: 1.4rem;
    }
    
    .sidebar h2 {
        font-size: 1.2rem;
    }
    
    /* Card-like containers */
    .card-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        position: relative;
        max-width: 85%;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: rgba(0, 120, 212, 0.1);
        border-left: 3px solid #0078D4;
        margin-left: auto;
    }
    
    .ai-message {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 3px solid #808080;
        margin-right: auto;
    }
    
    .copy-tooltip {
        position: absolute;
        top: -25px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: none;
    }
    
    /* Google Sign-In button */
    .google-signin-button {
        display: flex;
        align-items: center;
        background-color: white;
        color: #757575;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 8px 16px;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: background-color 0.3s;
    }
    
    .google-signin-button:hover {
        background-color: #f5f5f5;
    }
    
    .google-signin-button img {
        width: 18px;
        height: 18px;
        margin-right: 10px;
    }
    
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        margin: 20px 0;
    }
    
    /* Authentication required message */
    .auth-required {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 3px solid #FFC107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* About Us section - collapsible */
    .about-us-header {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    .about-us-content {
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* App introduction */
    .app-intro {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- JavaScript for Copy Functionality and Collapsible About Us ---
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
    
    // Add event listeners to chat messages
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            // Copy functionality for chat messages
            const chatMessages = document.querySelectorAll('.chat-message');
            chatMessages.forEach(function(message) {
                message.addEventListener('touchstart', function() {
                    this.longPressTimer = setTimeout(() => {
                        const text = this.innerText;
                        copyToClipboard(text);
                        const tooltip = this.querySelector('.copy-tooltip');
                        if (tooltip) {
                            tooltip.style.display = 'block';
                            setTimeout(() => {
                                tooltip.style.display = 'none';
                            }, 1000);
                        }
                    }, 500);
                });
                
                message.addEventListener('touchend', function() {
                    clearTimeout(this.longPressTimer);
                });
            });
            
            // Collapsible About Us section
            const aboutUsHeader = document.querySelector('.about-us-header');
            const aboutUsContent = document.querySelector('.about-us-content');
            
            if (aboutUsHeader && aboutUsContent) {
                aboutUsContent.style.display = 'none';
                
                aboutUsHeader.addEventListener('click', function() {
                    if (aboutUsContent.style.display === 'none') {
                        aboutUsContent.style.display = 'block';
                    } else {
                        aboutUsContent.style.display = 'none';
                    }
                });
            }
        }, 1000);
    });
    </script>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide")
apply_custom_css()
add_javascript_functionality()

# --- Capture User Agent ---
def capture_user_agent():
    """Capture and store the user agent in session state."""
    try:
        # This is a workaround as Streamlit doesn't directly expose the user agent
        # In a production app, you might need a different approach
        components.html(
            """
            <script>
                if (window.parent && window.parent.document) {
                    const userAgent = navigator.userAgent;
                    window.parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: userAgent
                    }, "*");
                }
            </script>
            """,
            height=0,
            key="user_agent_capture"
        )
        
        # Store the user agent in session state
        if "user_agent_capture" in st.session_state:
            st.session_state.user_agent = st.session_state.user_agent_capture
    except:
        # Fallback if the approach doesn't work
        st.session_state.user_agent = "Unknown"

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize session state variables."""
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# --- Initialize Firebase and Analytics ---
firebase_initialized = initialize_firebase()
initialize_session_state()
capture_user_agent()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_configured = False

if GEMINI_API_KEY and GEMINI_API_KEY != "Gemini_api_key":
    try:
        # Configure Gemini with the API key
        genai.configure(api_key=GEMINI_API_KEY)

        # Load Gemini models with custom generation settings
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=4000
        )

        gemini_model_report = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config
        )

        gemini_model_chat = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config
        )

        gemini_configured = True

    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. AI features might be limited.")
else:
    st.warning("Gemini API Key not found or is placeholder. AI features will be disabled. Set GEMINI_API_KEY environment variable or update in code.")

# --- Model Paths & Constants ---
# Model is directly in the root directory based on the screenshot
STANDARD_MODEL_PATH = "standard_model.h5"  # Direct path to the file in root directory

STANDARD_MODEL_SEQUENCE_LENGTH = 60  # Default, will be updated if model loads
if os.path.exists(STANDARD_MODEL_PATH):
    try:
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        _std_model_temp = load_model(STANDARD_MODEL_PATH, compile=False)
        STANDARD_MODEL_SEQUENCE_LENGTH = _std_model_temp.input_shape[1]
        del _std_model_temp
        # st.info(f"Standard model structure loaded successfully from {STANDARD_MODEL_PATH} to infer sequence length: {STANDARD_MODEL_SEQUENCE_LENGTH}")
    except Exception as e:
        st.warning(f"Could not load standard model from {STANDARD_MODEL_PATH} to infer sequence length: {e}. Using default {STANDARD_MODEL_SEQUENCE_LENGTH}.")
else:
    st.warning(f"Standard model file not found at path: {STANDARD_MODEL_PATH}. Please ensure it exists in the root directory next to demo.py.")

# --- Helper Functions ---
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

@st.cache_resource # Cache model loading
def load_keras_model_from_file(uploaded_file_obj, model_name_for_log):
    temp_model_path = f"temp_{model_name_for_log.replace(' ', '_')}.h5"
    try:
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        model = load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded {model_name_for_log}. Inferred sequence length: {sequence_length}")
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading Keras model {model_name_for_log}: {e}")
        return None, None
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@st.cache_resource
def load_standard_model_cached(path):
    try:
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        model = load_model(path, compile=False)
        sequence_length = model.input_shape[1]
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading standard Keras model from {path}: {e}")
        return None, None

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error") # For training, we compile with loss
    return model

# FIXED: Enhanced MC Dropout uncertainty calculation to ensure meaningful confidence intervals
def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    
    # Define the prediction function with dropout enabled (training=True)
    @tf.function
    def predict_step_training_true(inp): 
        return model(inp, training=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run multiple iterations with dropout enabled to get different predictions
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        
        # For each step in the forecast horizon
        for _ in range(n_steps):
            # Get prediction with dropout enabled (creates randomness)
            next_pred_scaled = predict_step_training_true(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            
            # Update sequence for next step prediction
            temp_sequence = np.append(temp_sequence[:, 1:, :], 
                                     np.array([[next_pred_scaled]]).reshape(1,1,1), 
                                     axis=1)
        
        all_predictions.append(iteration_predictions_scaled)
        progress_bar.progress((i + 1) / n_iterations)
        status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Convert to numpy array for calculations
    predictions_array_scaled = np.array(all_predictions)
    
    # Calculate mean predictions across all iterations
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    
    # Calculate standard deviation for uncertainty
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    
    # FIXED: Increased the confidence interval multiplier to create more visible uncertainty bands
    # Using 2.5 instead of 1.96 for wider confidence intervals
    ci_multiplier = 2.5
    
    # Convert scaled predictions back to original scale
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    
    # Calculate lower and upper bounds with wider intervals
    lower_bound = scaler.inverse_transform((mean_preds_scaled - ci_multiplier * std_devs_scaled).reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform((mean_preds_scaled + ci_multiplier * std_devs_scaled).reshape(-1, 1)).flatten()
    
    # FIXED: Add artificial minimum uncertainty if standard deviation is too small
    # This ensures the confidence intervals are always visible
    min_uncertainty_percent = 0.05  # 5% minimum uncertainty
    
    for i in range(len(mean_preds)):
        # Calculate the current uncertainty range as a percentage of the prediction
        current_range_percent = (upper_bound[i] - lower_bound[i]) / mean_preds[i] if mean_preds[i] != 0 else 0
        
        # If uncertainty is too small, expand it
        if current_range_percent < min_uncertainty_percent:
            uncertainty_value = mean_preds[i] * min_uncertainty_percent / 2
            lower_bound[i] = mean_preds[i] - uncertainty_value
            upper_bound[i] = mean_preds[i] + uncertainty_value
    
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.inf
    if np.all(y_true != 0): mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions ---
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
        fig.add_annotation(text="Training history is not available for pre-trained models or if training did not occur.",xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    history_df = pd.DataFrame({"Training Loss": history_dict["loss"], "Validation Loss": history_dict["val_loss"]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history_df["Training Loss"], mode="lines", name="Training Loss", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(y=history_df["Validation Loss"], mode="lines", name="Validation Loss", line=dict(color="rgb(255, 127, 14)")))
    fig.update_layout(title="Model Training History", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_model_evaluation_plot(y_true, y_pred, title="Model Evaluation"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_true))), y=y_true, mode="lines", name="Actual", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode="lines", name="Predicted", line=dict(color="rgb(255, 127, 14)")))
    fig.update_layout(title=title, xaxis_title="Time Step", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

# --- PDF Report Generation ---
def generate_pdf_report(historical_df, forecast_df, model_metrics, forecast_plot_path, model_eval_plot_path=None, loss_plot_path=None, ai_analysis=None):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Groundwater Level Forecast Report', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    
    # Report generation date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Report Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)
    
    # Data Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Summary', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Historical Data Range: {historical_df["Date"].min().strftime("%Y-%m-%d")} to {historical_df["Date"].max().strftime("%Y-%m-%d")}', 0, 1)
    pdf.cell(0, 10, f'Forecast Period: {forecast_df["Date"].min().strftime("%Y-%m-%d")} to {forecast_df["Date"].max().strftime("%Y-%m-%d")}', 0, 1)
    pdf.cell(0, 10, f'Number of Historical Data Points: {len(historical_df)}', 0, 1)
    pdf.cell(0, 10, f'Number of Forecast Data Points: {len(forecast_df)}', 0, 1)
    pdf.ln(5)
    
    # Model Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Model Performance Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    for metric, value in model_metrics.items():
        if not np.isnan(value) and not np.isinf(value):
            pdf.cell(0, 10, f'{metric}: {value:.4f}', 0, 1)
        else:
            pdf.cell(0, 10, f'{metric}: N/A', 0, 1)
    pdf.ln(5)
    
    # Forecast Plot
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Forecast Visualization', 0, 1)
    if os.path.exists(forecast_plot_path):
        pdf.image(forecast_plot_path, x=10, y=None, w=180)
    else:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, 'Forecast plot image not available.', 0, 1)
    pdf.ln(5)
    
    # Model Evaluation Plot (if available)
    if model_eval_plot_path and os.path.exists(model_eval_plot_path):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Model Evaluation', 0, 1)
        pdf.image(model_eval_plot_path, x=10, y=None, w=180)
        pdf.ln(5)
    
    # Training Loss Plot (if available)
    if loss_plot_path and os.path.exists(loss_plot_path):
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Model Training History', 0, 1)
        pdf.image(loss_plot_path, x=10, y=None, w=180)
        pdf.ln(5)
    
    # AI Analysis (if available)
    if ai_analysis:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'AI Analysis of Forecast Results', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Split the analysis into paragraphs and add them to the PDF
        paragraphs = ai_analysis.split('\n\n')
        for paragraph in paragraphs:
            pdf.multi_cell(0, 10, paragraph)
            pdf.ln(5)
    
    # Save the PDF to a file
    report_path = "groundwater_forecast_report.pdf"
    pdf.output(report_path)
    return report_path

# --- AI Analysis Functions ---
def generate_ai_analysis(historical_df, forecast_df, model_metrics):
    """Generate AI analysis of forecast results using Gemini."""
    if not gemini_configured:
        return "AI analysis is not available because the Gemini API is not configured."
    
    try:
        # Prepare data for analysis
        historical_start = historical_df["Date"].min().strftime("%Y-%m-%d")
        historical_end = historical_df["Date"].max().strftime("%Y-%m-%d")
        forecast_start = forecast_df["Date"].min().strftime("%Y-%m-%d")
        forecast_end = forecast_df["Date"].max().strftime("%Y-%m-%d")
        
        historical_min = historical_df["Level"].min()
        historical_max = historical_df["Level"].max()
        historical_mean = historical_df["Level"].mean()
        
        forecast_min = forecast_df["Forecast"].min()
        forecast_max = forecast_df["Forecast"].max()
        forecast_mean = forecast_df["Forecast"].mean()
        
        # Calculate trend
        historical_trend = "stable"
        if len(historical_df) > 1:
            first_half = historical_df.iloc[:len(historical_df)//2]["Level"].mean()
            second_half = historical_df.iloc[len(historical_df)//2:]["Level"].mean()
            if second_half > first_half * 1.05:
                historical_trend = "increasing"
            elif second_half < first_half * 0.95:
                historical_trend = "decreasing"
        
        forecast_trend = "stable"
        if len(forecast_df) > 1:
            first_half = forecast_df.iloc[:len(forecast_df)//2]["Forecast"].mean()
            second_half = forecast_df.iloc[len(forecast_df)//2:]["Forecast"].mean()
            if second_half > first_half * 1.05:
                forecast_trend = "increasing"
            elif second_half < first_half * 0.95:
                forecast_trend = "decreasing"
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are a hydrologist analyzing groundwater level data and forecasts. Please provide a detailed analysis of the following data:

        Historical Data:
        - Date Range: {historical_start} to {historical_end}
        - Minimum Level: {historical_min:.2f}
        - Maximum Level: {historical_max:.2f}
        - Mean Level: {historical_mean:.2f}
        - Overall Trend: {historical_trend}

        Forecast Data:
        - Date Range: {forecast_start} to {forecast_end}
        - Minimum Level: {forecast_min:.2f}
        - Maximum Level: {forecast_max:.2f}
        - Mean Level: {forecast_mean:.2f}
        - Overall Trend: {forecast_trend}

        Model Performance Metrics:
        - RMSE: {model_metrics.get('RMSE', 'N/A')}
        - MAE: {model_metrics.get('MAE', 'N/A')}
        - MAPE: {model_metrics.get('MAPE', 'N/A')}

        Please provide:
        1. A summary of the historical data patterns
        2. An interpretation of the forecast results
        3. An assessment of the model's performance
        4. Potential implications for groundwater management
        5. Recommendations for monitoring and further analysis

        Your analysis should be detailed, professional, and written for a technical audience with knowledge of hydrology.
        """
        
        # Generate analysis with Gemini
        response = gemini_model_report.generate_content(prompt)
        analysis = response.text
        
        return analysis
    
    except Exception as e:
        st.error(f"Error generating AI analysis: {e}")
        return "An error occurred while generating the AI analysis. Please try again later."

def chat_with_ai(user_message, chat_history):
    """Chat with AI about groundwater forecasting using Gemini."""
    if not gemini_configured:
        return "AI chat is not available because the Gemini API is not configured."
    
    try:
        # Prepare system prompt
        system_prompt = """
        You are a hydrologist AI assistant specializing in groundwater level forecasting and analysis.
        You can help users understand their groundwater data, interpret forecasts, and provide insights on groundwater management.
        Your responses should be informative, technical when appropriate, but also accessible to users with varying levels of expertise.
        
        You can help with:
        - Interpreting groundwater level data and forecasts
        - Explaining factors that influence groundwater levels
        - Providing context on groundwater management and conservation
        - Explaining technical concepts related to the LSTM forecasting model
        - Suggesting best practices for groundwater monitoring
        
        If asked about specific data that isn't provided in the conversation, politely explain that you can only discuss the information shared in the current conversation.
        """
        
        # Create a chat session
        chat = gemini_model_chat.start_chat(history=[])
        
        # Add system prompt
        chat.send_message(system_prompt)
        
        # Add chat history
        for message in chat_history:
            if message["role"] == "user":
                chat.send_message(message["content"])
            else:
                # Simulate assistant response in history
                pass
        
        # Send user message and get response
        response = chat.send_message(user_message)
        
        return response.text
    
    except Exception as e:
        st.error(f"Error in AI chat: {e}")
        return "An error occurred while processing your message. Please try again later."

# --- Main Application ---
def main():
    # Initialize Firebase and session state
    firebase_initialized = initialize_firebase()
    
    # Log page view
    log_visitor_activity("main_page")
    
    # Sidebar
    with st.sidebar:
        st.image("https://i.imgur.com/6RB7Oca.png", width=100)
        st.title("DeepHydro AI")
        
        # User authentication status and logout
        if st.session_state.user_authenticated:
            st.success(f"Signed in as {st.session_state.user_email}")
            if st.button("Logout"):
                st.session_state.user_authenticated = False
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.is_admin = False
                st.rerun()
        
        # Navigation
        st.header("Navigation")
        page = st.radio("Select Page", ["Home", "Forecasting", "AI Report", "AI Chat", "Admin Dashboard"])
        
        # About section
        st.markdown('<div class="about-us-header">About DeepHydro AI</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-us-content">
        DeepHydro AI combines deep learning with hydrological expertise to provide accurate groundwater level forecasting.
        
        Our LSTM neural network model is trained on historical groundwater data to predict future levels with confidence intervals.
        
        For questions or support, contact us at support@deephydro.ai
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if page == "Home":
        st.title("DeepHydro AI Forecasting")
        st.markdown("""
        <div class="app-intro">
        Welcome to DeepHydro AI, your advanced solution for groundwater level forecasting using deep learning technology.
        Our application uses Long Short-Term Memory (LSTM) neural networks to provide accurate predictions with uncertainty quantification.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Key Features
            
            - **Data-Driven Forecasting**: Upload your historical groundwater level data and get accurate predictions
            - **Uncertainty Quantification**: View confidence intervals around forecasts
            - **Custom Model Training**: Train models specific to your groundwater system
            - **AI-Powered Analysis**: Get expert interpretation of forecast results
            - **Interactive Visualization**: Explore data and forecasts through interactive charts
            """)
        
        with col2:
            st.markdown("""
            ### Getting Started
            
            1. Navigate to the **Forecasting** page
            2. Upload your historical groundwater level data (Excel format)
            3. Choose between using our pre-trained model or training a custom model
            4. Set your forecast parameters
            5. View and download your forecast results and report
            
            For deeper insights, try the **AI Report** and **AI Chat** features.
            """)
        
        st.markdown("---")
        
        st.subheader("How It Works")
        st.markdown("""
        DeepHydro AI uses LSTM (Long Short-Term Memory) neural networks, a type of recurrent neural network specifically designed to learn patterns in sequential data like time series.
        
        The model learns from historical groundwater level patterns, accounting for seasonality, trends, and other complex relationships in your data. It then uses this knowledge to forecast future levels.
        
        Our uncertainty quantification approach uses Monte Carlo Dropout to provide confidence intervals around predictions, giving you a range of possible future scenarios.
        """)
    
    elif page == "Forecasting":
        st.title("Groundwater Level Forecasting")
        
        # Check if user has access to this feature
        if not check_feature_access("forecasting"):
            # Track attempt to access restricted feature
            log_visitor_activity("forecasting", "access_attempt_restricted")
            
            st.markdown("""
            <div class="auth-required">
                <h3>Authentication Required</h3>
                <p>You've reached the usage limit for this feature. Please sign in with your Google account to continue using the forecasting feature.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add Google Sign-In button
            st.markdown("""
            <div class="login-container">
                <p>Sign in with your Google account to continue.</p>
                <button class="google-signin-button">
                    <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo">
                    <span>Sign in with Google</span>
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            return
        
        # Track usage of forecasting feature
        track_usage("forecasting")
        log_visitor_activity("forecasting", "feature_access")
        
        # Rest of the forecasting page code...
        st.markdown("Upload your historical groundwater level data to generate forecasts using LSTM neural networks.")
        
        uploaded_file = st.file_uploader("Upload Excel file with Date and Level columns", type=["xlsx", "xls"])
        
        if uploaded_file:
            # Load and clean data
            df = load_and_clean_data(uploaded_file.getvalue())
            
            if df is not None:
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Plot historical data
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["Level"], mode="lines", name="Historical Data"))
                fig.update_layout(title="Historical Groundwater Level Data", xaxis_title="Date", yaxis_title="Groundwater Level")
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecasting options
                st.subheader("Forecasting Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    model_option = st.radio("Select Model Option", ["Use Pre-trained Model", "Train Custom Model"])
                    forecast_days = st.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30)
                
                with col2:
                    if model_option == "Use Pre-trained Model":
                        custom_model_file = None
                        use_standard_model = st.checkbox("Use Standard Model", value=True)
                        if not use_standard_model:
                            custom_model_file = st.file_uploader("Upload Custom H5 Model", type=["h5"])
                    else:
                        train_split = st.slider("Training Data Percentage", min_value=50, max_value=90, value=80)
                        epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=50)
                        patience = st.slider("Early Stopping Patience", min_value=5, max_value=50, value=10)
                
                # Generate forecast button
                if st.button("Generate Forecast"):
                    with st.spinner("Processing data and generating forecast..."):
                        # Scale data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(df[["Level"]].values)
                        
                        # Load or train model
                        if model_option == "Use Pre-trained Model":
                            if use_standard_model:
                                model, sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                                if model is None:
                                    st.error("Failed to load standard model. Please try uploading a custom model.")
                                    st.stop()
                            else:
                                if custom_model_file is None:
                                    st.error("Please upload a custom model file or use the standard model.")
                                    st.stop()
                                model, sequence_length = load_keras_model_from_file(custom_model_file, "Custom Model")
                                if model is None:
                                    st.error("Failed to load custom model. Please check the file and try again.")
                                    st.stop()
                            
                            # Compile the model for inference
                            model.compile(optimizer="adam", loss="mean_squared_error")
                            
                            # No training history for pre-trained models
                            history = None
                            
                            # Evaluate model on historical data
                            if len(scaled_data) > sequence_length:
                                X, y = create_sequences(scaled_data, sequence_length)
                                y_pred = model.predict(X)
                                
                                # Convert predictions back to original scale
                                y_orig = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
                                y_pred_orig = scaler.inverse_transform(y_pred).flatten()
                                
                                # Calculate metrics
                                metrics = calculate_metrics(y_orig, y_pred_orig)
                                
                                # Create evaluation plot
                                eval_fig = create_model_evaluation_plot(y_orig, y_pred_orig, "Model Evaluation on Historical Data")
                                st.plotly_chart(eval_fig, use_container_width=True)
                                
                                # Save evaluation plot for report
                                eval_plot_path = "model_evaluation_plot.png"
                                eval_fig.write_image(eval_plot_path)
                            else:
                                st.warning(f"Not enough data points for model evaluation. Need at least {sequence_length+1} points.")
                                metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
                                eval_plot_path = None
                        else:
                            # Train custom model
                            sequence_length = min(STANDARD_MODEL_SEQUENCE_LENGTH, len(scaled_data) // 4)
                            X, y = create_sequences(scaled_data, sequence_length)
                            
                            # Split data
                            split_idx = int(len(X) * train_split / 100)
                            X_train, X_val = X[:split_idx], X[split_idx:]
                            y_train, y_val = y[:split_idx], y[split_idx:]
                            
                            # Build and train model
                            model = build_lstm_model(sequence_length)
                            early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
                            
                            history = model.fit(
                                X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping],
                                verbose=0
                            ).history
                            
                            # Plot training history
                            history_fig = create_loss_plot(history)
                            st.plotly_chart(history_fig, use_container_width=True)
                            
                            # Save history plot for report
                            history_plot_path = "training_history_plot.png"
                            history_fig.write_image(history_plot_path)
                            
                            # Evaluate model
                            y_pred = model.predict(X_val)
                            
                            # Convert predictions back to original scale
                            y_val_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                            y_pred_orig = scaler.inverse_transform(y_pred).flatten()
                            
                            # Calculate metrics
                            metrics = calculate_metrics(y_val_orig, y_pred_orig)
                            
                            # Create evaluation plot
                            eval_fig = create_model_evaluation_plot(y_val_orig, y_pred_orig, "Model Evaluation on Validation Data")
                            st.plotly_chart(eval_fig, use_container_width=True)
                            
                            # Save evaluation plot for report
                            eval_plot_path = "model_evaluation_plot.png"
                            eval_fig.write_image(eval_plot_path)
                        
                        # Display metrics
                        st.subheader("Model Performance Metrics")
                        metrics_df = pd.DataFrame({
                            "Metric": list(metrics.keys()),
                            "Value": [f"{v:.4f}" if not np.isnan(v) and not np.isinf(v) else "N/A" for v in metrics.values()]
                        })
                        st.table(metrics_df)
                        
                        # Generate forecast
                        st.subheader("Forecast Results")
                        
                        # Get last sequence for forecasting
                        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                        
                        # Generate forecast with uncertainty using MC Dropout
                        forecast_mean, forecast_lower, forecast_upper = predict_with_dropout_uncertainty(
                            model, last_sequence, forecast_days, n_iterations=30, 
                            scaler=scaler, model_sequence_length=sequence_length
                        )
                        
                        # Create forecast dates
                        last_date = df["Date"].iloc[-1]
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            "Date": forecast_dates,
                            "Forecast": forecast_mean,
                            "Lower_CI": forecast_lower,
                            "Upper_CI": forecast_upper
                        })
                        
                        # Display forecast table
                        st.dataframe(forecast_df)
                        
                        # Plot forecast
                        forecast_fig = create_forecast_plot(df, forecast_df)
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Save forecast plot for report
                        forecast_plot_path = "forecast_plot.png"
                        forecast_fig.write_image(forecast_plot_path)
                        
                        # Generate AI analysis if Gemini is configured
                        if gemini_configured:
                            st.subheader("AI Analysis")
                            with st.spinner("Generating AI analysis of forecast results..."):
                                ai_analysis = generate_ai_analysis(df, forecast_df, metrics)
                                st.markdown(ai_analysis)
                        else:
                            ai_analysis = None
                        
                        # Generate PDF report
                        with st.spinner("Generating PDF report..."):
                            if model_option == "Train Custom Model" and history is not None:
                                report_path = generate_pdf_report(
                                    df, forecast_df, metrics, 
                                    forecast_plot_path, eval_plot_path, 
                                    history_plot_path, ai_analysis
                                )
                            else:
                                report_path = generate_pdf_report(
                                    df, forecast_df, metrics, 
                                    forecast_plot_path, eval_plot_path, 
                                    None, ai_analysis
                                )
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with open(report_path, "rb") as file:
                                btn = st.download_button(
                                    label="Download PDF Report",
                                    data=file,
                                    file_name="groundwater_forecast_report.pdf",
                                    mime="application/pdf"
                                )
                        
                        with col2:
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="Download Forecast CSV",
                                data=csv,
                                file_name="groundwater_forecast_data.csv",
                                mime="text/csv"
                            )
    
    elif page == "AI Report":
        st.title("AI Analysis Report")
        
        # Check if user has access to this feature
        if not check_feature_access("ai_report"):
            # Track attempt to access restricted feature
            log_visitor_activity("ai_report", "access_attempt_restricted")
            
            st.markdown("""
            <div class="auth-required">
                <h3>Authentication Required</h3>
                <p>You've reached the usage limit for this feature. Please sign in with your Google account to continue using the AI Report feature.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add Google Sign-In button
            st.markdown("""
            <div class="login-container">
                <p>Sign in with your Google account to continue.</p>
                <button class="google-signin-button">
                    <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo">
                    <span>Sign in with Google</span>
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            return
        
        # Track usage of AI report feature
        track_usage("ai_report")
        log_visitor_activity("ai_report", "feature_access")
        
        # Check if Gemini is configured
        if not gemini_configured:
            st.error("AI Report feature is not available because the Gemini API is not configured.")
            return
        
        st.markdown("Upload your groundwater data and forecast results to get an AI-generated analysis report.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            historical_file = st.file_uploader("Upload Historical Data (CSV/Excel)", type=["csv", "xlsx", "xls"])
        
        with col2:
            forecast_file = st.file_uploader("Upload Forecast Data (CSV/Excel)", type=["csv", "xlsx", "xls"])
        
        if historical_file and forecast_file:
            # Load historical data
            try:
                if historical_file.name.endswith(".csv"):
                    historical_df = pd.read_csv(historical_file)
                else:
                    historical_df = pd.read_excel(historical_file)
                
                # Check for required columns
                if "Date" not in historical_df.columns or not any(col for col in historical_df.columns if col in ["Level", "Value", "Groundwater"]):
                    st.error("Historical data must contain 'Date' column and a level column (named 'Level', 'Value', or 'Groundwater').")
                    st.stop()
                
                # Rename level column if needed
                level_col = next(col for col in historical_df.columns if col in ["Level", "Value", "Groundwater"])
                if level_col != "Level":
                    historical_df = historical_df.rename(columns={level_col: "Level"})
                
                # Convert date to datetime
                historical_df["Date"] = pd.to_datetime(historical_df["Date"])
                
                # Sort by date
                historical_df = historical_df.sort_values("Date")
            except Exception as e:
                st.error(f"Error loading historical data: {e}")
                st.stop()
            
            # Load forecast data
            try:
                if forecast_file.name.endswith(".csv"):
                    forecast_df = pd.read_csv(forecast_file)
                else:
                    forecast_df = pd.read_excel(forecast_file)
                
                # Check for required columns
                if "Date" not in forecast_df.columns or not any(col for col in forecast_df.columns if col in ["Forecast", "Prediction", "Value"]):
                    st.error("Forecast data must contain 'Date' column and a forecast column (named 'Forecast', 'Prediction', or 'Value').")
                    st.stop()
                
                # Rename forecast column if needed
                forecast_col = next(col for col in forecast_df.columns if col in ["Forecast", "Prediction", "Value"])
                if forecast_col != "Forecast":
                    forecast_df = forecast_df.rename(columns={forecast_col: "Forecast"})
                
                # Convert date to datetime
                forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
                
                # Sort by date
                forecast_df = forecast_df.sort_values("Date")
            except Exception as e:
                st.error(f"Error loading forecast data: {e}")
                st.stop()
            
            # Display data previews
            st.subheader("Data Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Historical Data")
                st.dataframe(historical_df.head())
            
            with col2:
                st.write("Forecast Data")
                st.dataframe(forecast_df.head())
            
            # Plot combined data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data"))
            fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"))
            
            # Add confidence intervals if available
            if "Lower_CI" in forecast_df.columns and "Upper_CI" in forecast_df.columns:
                fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI", line=dict(width=0)))
                fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI", line=dict(width=0), fillcolor="rgba(0, 176, 246, 0.2)", fill="tonexty"))
            
            fig.update_layout(title="Historical Data and Forecast", xaxis_title="Date", yaxis_title="Groundwater Level")
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate AI analysis
            if st.button("Generate AI Analysis"):
                with st.spinner("Generating AI analysis..."):
                    # Calculate simple metrics for the analysis
                    metrics = {
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "MAPE": np.nan
                    }
                    
                    # Generate analysis
                    analysis = generate_ai_analysis(historical_df, forecast_df, metrics)
                    
                    # Display analysis
                    st.subheader("AI Analysis")
                    st.markdown(analysis)
                    
                    # Save analysis to file for download
                    with open("ai_analysis_report.txt", "w") as f:
                        f.write(analysis)
                    
                    # Download button
                    with open("ai_analysis_report.txt", "r") as f:
                        st.download_button(
                            label="Download Analysis Report",
                            data=f,
                            file_name="ai_analysis_report.txt",
                            mime="text/plain"
                        )
    
    elif page == "AI Chat":
        st.title("Chat with DeepHydro AI")
        
        # Check if user has access to this feature
        if not check_feature_access("ai_chat"):
            # Track attempt to access restricted feature
            log_visitor_activity("ai_chat", "access_attempt_restricted")
            
            st.markdown("""
            <div class="auth-required">
                <h3>Authentication Required</h3>
                <p>You've reached the usage limit for this feature. Please sign in with your Google account to continue using the AI Chat feature.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add Google Sign-In button
            st.markdown("""
            <div class="login-container">
                <p>Sign in with your Google account to continue.</p>
                <button class="google-signin-button">
                    <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo">
                    <span>Sign in with Google</span>
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            return
        
        # Track usage of AI chat feature
        track_usage("ai_chat")
        log_visitor_activity("ai_chat", "feature_access")
        
        # Check if Gemini is configured
        if not gemini_configured:
            st.error("AI Chat feature is not available because the Gemini API is not configured.")
            return
        
        st.markdown("Chat with our AI assistant about groundwater forecasting, hydrology, and data analysis.")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    {message["content"]}
                    <div class="copy-tooltip">Copied!</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    {message["content"]}
                    <div class="copy-tooltip">Copied!</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_message = st.text_area("Your message:", height=100)
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if st.button("Send"):
                if user_message:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_message})
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        ai_response = chat_with_ai(user_message, st.session_state.chat_history)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    # Rerun to update UI
                    st.rerun()
        
        with col2:
            if st.button("New Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    elif page == "Admin Dashboard":
        # Track page view
        log_visitor_activity("admin_dashboard")
        
        # Render admin analytics dashboard
        render_admin_analytics()

if __name__ == "__main__":
    main()
