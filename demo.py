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
import hashlib

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

# --- Google Authentication Functions ---
def get_google_auth_url():
    """Generate Google authentication URL."""
    # In a real implementation, this would generate a proper OAuth URL
    # For this example, we'll use a placeholder
    auth_url = "https://accounts.google.com/o/oauth2/auth"
    client_id = os.getenv("GOOGLE_CLIENT_ID", "your-client-id")
    redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8501/callback")
    scope = "email profile"
    
    return f"{auth_url}?client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}&response_type=code"

def handle_auth_callback():
    """Handle the authentication callback from Google."""
    # In a real implementation, this would exchange the code for tokens
    # and verify the user's identity
    # For this example, we'll simulate the process
    
    # Get the code from the URL parameters
    code = st.experimental_get_query_params().get("code", [""])[0]
    
    if code:
        # Simulate token exchange and user info retrieval
        # In a real implementation, you would make API calls to Google
        user_email = "user@example.com"  # This would come from Google
        user_name = "Example User"  # This would come from Google
        
        # Store user info in session state
        st.session_state.user_email = user_email
        st.session_state.user_name = user_name
        st.session_state.authenticated = True
        
        # Log the authentication in Firebase
        log_user_authentication(user_email)
        
        return True
    
    return False

def log_user_authentication(email):
    """Log user authentication in Firebase."""
    if not firebase_admin._apps:
        return
    
    try:
        # Create a reference to the users collection
        ref = db.reference('users')
        
        # Get user IP
        ip_address = get_client_ip()
        
        # Create a unique ID for this user based on email and IP
        user_id = hashlib.md5(f"{email}:{ip_address}".encode()).hexdigest()
        
        # Update user data
        user_data = {
            'email': email,
            'ip_address': ip_address,
            'last_login': datetime.datetime.now().isoformat(),
            'authenticated': True
        }
        
        # Update the user data in Firebase
        ref.child(user_id).update(user_data)
    except Exception as e:
        # Silently fail to not disrupt user experience
        pass

# --- Usage Counter Functions ---
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

def get_user_id():
    """Get a unique identifier for the current user based on IP address."""
    ip_address = get_client_ip()
    email = st.session_state.get('user_email', 'anonymous')
    
    # Create a unique ID based on email and IP
    user_id = hashlib.md5(f"{email}:{ip_address}".encode()).hexdigest()
    
    return user_id

def increment_usage_counter():
    """Increment the usage counter for the current user."""
    if not firebase_admin._apps:
        return
    
    try:
        # Get user ID
        user_id = get_user_id()
        
        # Create a reference to the users collection
        ref = db.reference('users')
        
        # Get current user data
        user_data = ref.child(user_id).get() or {}
        
        # Get current usage count
        usage_count = user_data.get('usage_count', 0)
        
        # Increment usage count
        usage_count += 1
        
        # Update user data
        user_data.update({
            'ip_address': get_client_ip(),
            'last_access': datetime.datetime.now().isoformat(),
            'usage_count': usage_count,
            'email': st.session_state.get('user_email', 'anonymous'),
            'authenticated': st.session_state.get('authenticated', False)
        })
        
        # Update the user data in Firebase
        ref.child(user_id).update(user_data)
        
        # Store usage count in session state
        st.session_state.usage_count = usage_count
        
        return usage_count
    except Exception as e:
        # Silently fail to not disrupt user experience
        return st.session_state.get('usage_count', 0)

def get_usage_count():
    """Get the current usage count for the user."""
    if 'usage_count' in st.session_state:
        return st.session_state.usage_count
    
    if not firebase_admin._apps:
        return 0
    
    try:
        # Get user ID
        user_id = get_user_id()
        
        # Create a reference to the users collection
        ref = db.reference('users')
        
        # Get current user data
        user_data = ref.child(user_id).get() or {}
        
        # Get current usage count
        usage_count = user_data.get('usage_count', 0)
        
        # Store in session state
        st.session_state.usage_count = usage_count
        
        return usage_count
    except Exception as e:
        # Silently fail to not disrupt user experience
        return 0

def check_auth_required():
    """Check if authentication is required based on usage count."""
    # Get current usage count
    usage_count = get_usage_count()
    
    # Check if authenticated
    authenticated = st.session_state.get('authenticated', False)
    
    # If usage count is 3 or more and not authenticated, require authentication
    if usage_count >= 3 and not authenticated:
        return True
    
    return False

def is_restricted_feature(feature_name):
    """Check if a feature is restricted after 3 uses."""
    restricted_features = ['ai_report', 'ai_chat', 'forecasting']
    return feature_name in restricted_features

def can_access_feature(feature_name):
    """Check if the user can access a specific feature."""
    # If not a restricted feature, always allow access
    if not is_restricted_feature(feature_name):
        return True
    
    # If authenticated, allow access
    if st.session_state.get('authenticated', False):
        return True
    
    # If usage count is less than 3, allow access
    if get_usage_count() < 3:
        return True
    
    # Otherwise, restrict access
    return False

# --- Visitor Analytics Functions ---
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
        email = st.session_state.get('user_email', 'anonymous')
        authenticated = st.session_state.get('authenticated', False)
        
        # Create the visitor data entry
        visitor_data = {
            'timestamp': timestamp,
            'ip_address': ip_address,
            'page': page_name,
            'action': action,
            'session_id': session_id,
            'user_agent': user_agent,
            'email': email,
            'authenticated': authenticated
        }
        
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

def fetch_user_data():
    """
    Fetch user data from Firebase for admin viewing.
    Returns a pandas DataFrame with the user data.
    """
    if not firebase_admin._apps:
        return pd.DataFrame()
    
    try:
        # Get reference to users collection
        ref = db.reference('users')
        
        # Get all user data
        users_data = ref.get()
        
        if not users_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        users_list = []
        for user_id, data in users_data.items():
            data['user_id'] = user_id
            users_list.append(data)
        
        df = pd.DataFrame(users_list)
        
        # Convert timestamp to datetime if present
        if 'last_access' in df.columns:
            df['last_access'] = pd.to_datetime(df['last_access'])
        if 'last_login' in df.columns:
            df['last_login'] = pd.to_datetime(df['last_login'])
        
        # Sort by last access (most recent first)
        if 'last_access' in df.columns:
            df = df.sort_values('last_access', ascending=False)
        
        return df
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
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
        
        # 4. Authentication status chart
        if 'authenticated' in df.columns:
            auth_counts = df['authenticated'].value_counts().reset_index()
            auth_counts.columns = ['authenticated', 'count']
            
            fig4 = px.pie(auth_counts, values='count', names='authenticated',
                         title='Authentication Status')
            figures.append(fig4)
        
        # 5. Hourly activity heatmap - FIXED to handle any data shape
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
                fig5 = px.imshow(hourly_pivot, 
                                labels=dict(x="Hour of Day", y="Day of Week", color="Visit Count"),
                                x=[str(h) for h in available_hours],  # Use only available hours
                                y=ordered_available_days,  # Use only available days
                                title="Visitor Activity by Hour and Day")
                
                figures.append(fig5)
            else:
                # Create a placeholder figure if no hourly data
                fig5 = go.Figure()
                fig5.update_layout(
                    title="Visitor Activity by Hour and Day (No Data)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week"
                )
                fig5.add_annotation(
                    text="Not enough data to generate hourly activity heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                figures.append(fig5)
        except Exception as heatmap_err:
            # Create a fallback figure if the heatmap fails
            st.warning(f"Could not generate hourly activity heatmap: {heatmap_err}")
            fig5 = go.Figure()
            fig5.update_layout(
                title="Visitor Activity by Hour and Day (Error)",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            fig5.add_annotation(
                text="Error generating hourly activity heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            figures.append(fig5)
    
    except Exception as e:
        st.error(f"Error creating visitor charts: {e}")
        # Return an empty list if chart creation fails
        return []
    
    return figures

def create_user_charts(user_df):
    """
    Create visualizations of user data using Plotly.
    
    Args:
        user_df: DataFrame containing user data
    
    Returns:
        List of Plotly figures
    """
    if user_df.empty:
        return []
    
    figures = []
    
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df = user_df.copy()
        
        # 1. Usage count distribution
        if 'usage_count' in df.columns:
            # Create bins for usage count
            bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
            labels = ['0', '1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51+']
            
            df['usage_bin'] = pd.cut(df['usage_count'], bins=bins, labels=labels)
            usage_counts = df['usage_bin'].value_counts().reset_index()
            usage_counts.columns = ['usage_bin', 'count']
            
            fig1 = px.bar(usage_counts, x='usage_bin', y='count',
                         title='Usage Count Distribution',
                         labels={'count': 'Number of Users', 'usage_bin': 'Usage Count'})
            fig1.update_layout(xaxis_title='Usage Count', yaxis_title='Number of Users')
            figures.append(fig1)
        
        # 2. Authentication status
        if 'authenticated' in df.columns:
            auth_counts = df['authenticated'].value_counts().reset_index()
            auth_counts.columns = ['authenticated', 'count']
            
            fig2 = px.pie(auth_counts, values='count', names='authenticated',
                         title='Authentication Status')
            figures.append(fig2)
        
        # 3. User activity over time
        if 'last_access' in df.columns:
            # Convert to datetime if not already
            df['last_access'] = pd.to_datetime(df['last_access'])
            
            # Add date column for daily aggregation
            df['date'] = df['last_access'].dt.date
            
            # Group by date
            daily_activity = df.groupby('date').size().reset_index(name='count')
            daily_activity['date'] = pd.to_datetime(daily_activity['date'])
            
            fig3 = px.line(daily_activity, x='date', y='count', 
                          title='User Activity Over Time',
                          labels={'count': 'Number of Users', 'date': 'Date'})
            fig3.update_layout(xaxis_title='Date', yaxis_title='Number of Users')
            figures.append(fig3)
    
    except Exception as e:
        st.error(f"Error creating user charts: {e}")
        # Return an empty list if chart creation fails
        return []
    
    return figures

# --- Admin Analytics Dashboard ---
def render_admin_analytics():
    """Render the admin analytics dashboard with authentication."""
    st.header("Admin Analytics Dashboard")
    
    # Check if admin is authenticated
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.info("Please sign in with your Google account to view analytics.")
        
        # Email-only authentication (no password)
        admin_email = st.text_input("Admin Email")
        
        if st.button("Login"):
            # Check if email is in the admin list
            admin_emails = os.getenv("ADMIN_EMAILS", "admin@example.com").split(",")
            
            if admin_email in admin_emails:
                st.session_state.admin_authenticated = True
                st.session_state.admin_email = admin_email
                # Log admin login
                log_visitor_activity("Admin Dashboard", "admin_login")
                st.rerun()
            else:
                st.error("Invalid email")
    else:
        # Create tabs for different analytics views
        tabs = st.tabs(["Visitor Analytics", "User Data", "System Status"])
        
        with tabs[0]:  # Visitor Analytics
            st.subheader("Visitor Analytics")
            
            # Fetch visitor logs
            visitor_df = fetch_visitor_logs()
            
            if visitor_df.empty:
                st.info("No visitor data available yet.")
            else:
                # Display visitor statistics
                st.subheader("Visitor Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_visits = len(visitor_df)
                    st.metric("Total Visits", total_visits)
                
                with col2:
                    unique_visitors = visitor_df['session_id'].nunique()
                    st.metric("Unique Visitors", unique_visitors)
                
                with col3:
                    if 'date' not in visitor_df.columns:
                        visitor_df['date'] = visitor_df['timestamp'].dt.date
                    
                    today = datetime.datetime.now().date()
                    today_visits = len(visitor_df[visitor_df['date'] == today])
                    st.metric("Today's Visits", today_visits)
                
                # Create and display visualizations
                try:
                    charts = create_visitor_charts(visitor_df)
                    
                    for fig in charts:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as chart_err:
                    st.error(f"Error displaying charts: {chart_err}")
                
                # Display raw data with filters
                st.subheader("Raw Visitor Data")
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        date_range = st.date_input(
                            "Date Range",
                            [visitor_df['timestamp'].min().date(), visitor_df['timestamp'].max().date()]
                        )
                    except Exception as date_err:
                        st.warning(f"Could not set date range: {date_err}")
                        date_range = None
                
                with col2:
                    try:
                        page_filter = st.multiselect(
                            "Filter by Page",
                            options=visitor_df['page'].unique(),
                            default=[]
                        )
                    except Exception as page_err:
                        st.warning(f"Could not set page filter: {page_err}")
                        page_filter = []
                
                # Apply filters
                try:
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
                    st.dataframe(filtered_df[['timestamp', 'ip_address', 'email', 'page', 'action', 'session_id', 'authenticated']])
                    
                    # Export options
                    if st.button("Export Visitor Data to CSV"):
                        csv = filtered_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="visitor_logs.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                except Exception as filter_err:
                    st.error(f"Error applying filters: {filter_err}")
                    st.dataframe(visitor_df[['timestamp', 'ip_address', 'email', 'page', 'action', 'session_id', 'authenticated']])
        
        with tabs[1]:  # User Data
            st.subheader("User Data")
            
            # Fetch user data
            user_df = fetch_user_data()
            
            if user_df.empty:
                st.info("No user data available yet.")
            else:
                # Display user statistics
                st.subheader("User Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_users = len(user_df)
                    st.metric("Total Users", total_users)
                
                with col2:
                    authenticated_users = len(user_df[user_df.get('authenticated', False) == True])
                    st.metric("Authenticated Users", authenticated_users)
                
                with col3:
                    if 'usage_count' in user_df.columns:
                        avg_usage = user_df['usage_count'].mean()
                        st.metric("Average Usage Count", f"{avg_usage:.1f}")
                
                # Create and display visualizations
                try:
                    charts = create_user_charts(user_df)
                    
                    for fig in charts:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as chart_err:
                    st.error(f"Error displaying user charts: {chart_err}")
                
                # Display raw data
                st.subheader("Raw User Data")
                
                # Display the data
                st.dataframe(user_df)
                
                # Export options
                if st.button("Export User Data to CSV"):
                    csv = user_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="user_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        with tabs[2]:  # System Status
            st.subheader("System Status")
            
            # Display system information
            st.info("System is running normally.")
            
            # Display Firebase status
            if firebase_admin._apps:
                st.success("Firebase is connected and operational.")
            else:
                st.error("Firebase is not connected.")
            
            # Display Gemini API status
            if 'gemini_configured' in globals() and gemini_configured:
                st.success("Gemini API is configured and operational.")
            else:
                st.error("Gemini API is not configured.")
            
            # Display admin information
            st.subheader("Admin Information")
            st.write(f"Logged in as: {st.session_state.get('admin_email', 'Unknown')}")
            
            # Logout button
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

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
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        position: relative;
    }
    
    .user-message {
        border-left: 4px solid #1E88E5;
    }
    
    .ai-message {
        border-left: 4px solid #78909C;
    }
    
    /* Copy functionality for chat messages */
    .chat-message:active {
        opacity: 0.7;
    }
    
    .copy-tooltip {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        display: none;
    }
    
    .chat-message:active .copy-tooltip {
        display: block;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 4px 4px 0 0;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-weight: 600;
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
    
    /* Authentication required message */
    .auth-required {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #F44336;
        background-color: rgba(244, 67, 54, 0.1);
    }
    
    /* Login form */
    .login-form {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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

# --- Authentication UI Components ---
def render_login_ui():
    """Render the login UI for Google authentication."""
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    st.subheader("Login with Google")
    st.write("Please log in with your Google account to continue using all features.")
    
    # In a real implementation, this would be a proper Google Sign-In button
    if st.button("Sign in with Google"):
        # Redirect to Google authentication URL
        auth_url = get_google_auth_url()
        st.markdown(f'<meta http-equiv="refresh" content="0;URL=\'{auth_url}\'">', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_auth_required_message():
    """Render a message indicating that authentication is required."""
    st.markdown('<div class="auth-required">', unsafe_allow_html=True)
    st.warning("⚠️ Authentication Required")
    st.write("You have used this feature 3 times. Please log in with your Google account to continue using this feature.")
    
    # In a real implementation, this would be a proper Google Sign-In button
    if st.button("Sign in with Google", key="auth_required_button"):
        # Redirect to Google authentication URL
        auth_url = get_google_auth_url()
        st.markdown(f'<meta http-equiv="refresh" content="0;URL=\'{auth_url}\'">', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_user_profile():
    """Render the user profile UI."""
    if st.session_state.get('authenticated', False):
        st.sidebar.success(f"Logged in as: {st.session_state.get('user_email', 'Unknown')}")
        
        # Logout button
        if st.sidebar.button("Logout"):
            # Clear authentication state
            st.session_state.authenticated = False
            if 'user_email' in st.session_state:
                del st.session_state.user_email
            if 'user_name' in st.session_state:
                del st.session_state.user_name
            
            # Log logout action
            log_visitor_activity("User Profile", "logout")
            
            # Refresh the page
            st.rerun()

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

# --- Initialize Firebase and Analytics ---
firebase_initialized = initialize_firebase()

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
    
    return mean_preds, lower_bound, upper_bound

def generate_forecast_dates(last_date, n_steps, freq='D'):
    """Generate future dates for forecasting."""
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq=freq)

def create_forecast_plot(historical_dates, historical_values, forecast_dates, forecast_values, lower_bound=None, upper_bound=None):
    """Create a Plotly figure for the forecast."""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_dates, 
        y=historical_values,
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add uncertainty bounds if provided
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Groundwater Level Forecast',
        xaxis_title='Date',
        yaxis_title='Groundwater Level',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_forecast_table(forecast_dates, forecast_values, lower_bound=None, upper_bound=None):
    """Create a DataFrame for the forecast results."""
    data = {
        'Date': forecast_dates,
        'Forecast': forecast_values
    }
    
    if lower_bound is not None and upper_bound is not None:
        data['Lower Bound'] = lower_bound
        data['Upper Bound'] = upper_bound
    
    return pd.DataFrame(data)

def create_forecast_pdf(df, plot_path, output_path):
    """Create a PDF report of the forecast."""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(0, 10, 'Groundwater Level Forecast Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Add plot
    pdf.image(plot_path, x=10, y=30, w=190)
    pdf.ln(140)  # Move down after the plot
    
    # Add table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Forecast Data', 0, 1, 'L')
    pdf.ln(5)
    
    # Table header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(40, 10, 'Date', 1, 0, 'C')
    pdf.cell(40, 10, 'Forecast', 1, 0, 'C')
    
    if 'Lower Bound' in df.columns and 'Upper Bound' in df.columns:
        pdf.cell(40, 10, 'Lower Bound', 1, 0, 'C')
        pdf.cell(40, 10, 'Upper Bound', 1, 0, 'C')
    
    pdf.ln()
    
    # Table data
    pdf.set_font('Arial', '', 10)
    for i, row in df.iterrows():
        if i < 20:  # Limit to first 20 rows to fit on page
            pdf.cell(40, 10, row['Date'].strftime('%Y-%m-%d'), 1, 0, 'C')
            pdf.cell(40, 10, f"{row['Forecast']:.2f}", 1, 0, 'C')
            
            if 'Lower Bound' in df.columns and 'Upper Bound' in df.columns:
                pdf.cell(40, 10, f"{row['Lower Bound']:.2f}", 1, 0, 'C')
                pdf.cell(40, 10, f"{row['Upper Bound']:.2f}", 1, 0, 'C')
            
            pdf.ln()
    
    # Save the PDF
    pdf.output(output_path)

def generate_ai_report(data_df, forecast_df, model_type):
    """Generate an AI report using Gemini."""
    if not gemini_configured:
        return "AI report generation is not available because the Gemini API is not configured."
    
    try:
        # Prepare data for the AI
        data_summary = f"""
        Historical Data Summary:
        - Start Date: {data_df['Date'].min().strftime('%Y-%m-%d')}
        - End Date: {data_df['Date'].max().strftime('%Y-%m-%d')}
        - Number of Data Points: {len(data_df)}
        - Minimum Level: {data_df['Level'].min():.2f}
        - Maximum Level: {data_df['Level'].max():.2f}
        - Average Level: {data_df['Level'].mean():.2f}
        
        Forecast Summary:
        - Forecast Start: {forecast_df['Date'].min().strftime('%Y-%m-%d')}
        - Forecast End: {forecast_df['Date'].max().strftime('%Y-%m-%d')}
        - Number of Forecast Points: {len(forecast_df)}
        - Minimum Forecast: {forecast_df['Forecast'].min():.2f}
        - Maximum Forecast: {forecast_df['Forecast'].max():.2f}
        - Model Type: {model_type}
        """
        
        # Generate the report
        prompt = f"""
        You are a hydrologist specializing in groundwater analysis. Generate a comprehensive report based on the following groundwater level data and forecast.
        
        {data_summary}
        
        Your report should include:
        1. An executive summary of the groundwater situation
        2. Analysis of historical trends
        3. Interpretation of the forecast results
        4. Potential implications for water resource management
        5. Recommendations for monitoring and management
        
        Write in a professional, scientific tone. Use specific details from the data provided.
        """
        
        response = gemini_model_report.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI report: {e}"

# --- Main Application ---
def main():
    # Capture user agent
    capture_user_agent()
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check for authentication callback
    if 'code' in st.experimental_get_query_params():
        if handle_auth_callback():
            st.success("Authentication successful!")
            st.rerun()
    
    # Display user profile in sidebar if authenticated
    if st.session_state.get('authenticated', False):
        render_user_profile()
    
    # App header
    st.title("DeepHydro AI Forecasting")
    st.markdown('<div class="app-intro">', unsafe_allow_html=True)
    st.write("Welcome to DeepHydro AI Forecasting, an advanced tool for groundwater level prediction using deep learning.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Log page view
    log_visitor_activity("Home Page")
    
    # Increment usage counter
    increment_usage_counter()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Forecasting", "AI Chat", "AI Report", "Admin Dashboard"])
    
    if page == "Forecasting":
        render_forecasting_page()
    elif page == "AI Chat":
        render_ai_chat_page()
    elif page == "AI Report":
        render_ai_report_page()
    elif page == "Admin Dashboard":
        render_admin_analytics()
    
    # About section in sidebar
    st.sidebar.markdown('<div class="about-us-header">About DeepHydro AI</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="about-us-content">DeepHydro AI Forecasting is a state-of-the-art tool for groundwater level prediction using deep learning techniques. Our platform combines LSTM neural networks with uncertainty quantification to provide reliable forecasts for water resource management.</div>', unsafe_allow_html=True)

def render_forecasting_page():
    st.header("Groundwater Level Forecasting")
    
    # Check if authentication is required
    if is_restricted_feature("forecasting") and not can_access_feature("forecasting"):
        render_auth_required_message()
        return
    
    # Log feature access
    log_visitor_activity("Forecasting Page")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your groundwater level data (Excel format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load and clean data
        data_df = load_and_clean_data(uploaded_file.read())
        
        if data_df is not None:
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data_df.head())
            
            # Plot historical data
            st.subheader("Historical Data")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_df['Date'], y=data_df['Level'], mode='lines', name='Historical Data'))
            fig.update_layout(title='Historical Groundwater Level', xaxis_title='Date', yaxis_title='Level')
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecasting options
            st.subheader("Forecasting Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=365, value=30, step=1)
                mc_iterations = st.slider("Uncertainty Iterations", min_value=10, max_value=100, value=30, step=10)
            
            with col2:
                model_option = st.radio("Model Selection", ["Use Standard Model", "Upload Custom Model", "Train New Model"])
            
            # Model handling based on selection
            if model_option == "Use Standard Model":
                model, sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                if model is None:
                    st.error("Standard model could not be loaded. Please try uploading a custom model or training a new one.")
                    return
            
            elif model_option == "Upload Custom Model":
                model_file = st.file_uploader("Upload your custom LSTM model (.h5 format)", type=["h5"])
                if model_file is None:
                    st.info("Please upload your custom model file.")
                    return
                
                model, sequence_length = load_keras_model_from_file(model_file, "Custom Model")
                if model is None:
                    st.error("Custom model could not be loaded. Please check the file format.")
                    return
            
            elif model_option == "Train New Model":
                st.subheader("Model Training Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sequence_length = st.slider("Sequence Length", min_value=10, max_value=120, value=60, step=10)
                    epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=50, step=10)
                
                with col2:
                    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
                    patience = st.slider("Early Stopping Patience", min_value=5, max_value=30, value=10, step=5)
                
                if st.button("Train Model"):
                    with st.spinner("Training model... This may take a few minutes."):
                        # Prepare data for training
                        data_values = data_df['Level'].values.reshape(-1, 1)
                        
                        # Scale the data
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_scaled = scaler.fit_transform(data_values)
                        
                        # Create sequences
                        X, y = create_sequences(data_scaled, sequence_length)
                        
                        # Split into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
                        
                        # Build and train the model
                        model = build_lstm_model(sequence_length)
                        
                        # Early stopping to prevent overfitting
                        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
                        
                        # Train the model
                        history = model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Evaluate the model
                        train_predictions = model.predict(X_train)
                        test_predictions = model.predict(X_test)
                        
                        # Inverse transform the predictions
                        train_predictions = scaler.inverse_transform(train_predictions)
                        test_predictions = scaler.inverse_transform(test_predictions)
                        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
                        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                        
                        # Calculate metrics
                        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions))
                        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
                        train_mae = mean_absolute_error(y_train_inv, train_predictions)
                        test_mae = mean_absolute_error(y_test_inv, test_predictions)
                        
                        # Display metrics
                        st.success("Model training completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Training RMSE", f"{train_rmse:.4f}")
                            st.metric("Training MAE", f"{train_mae:.4f}")
                        
                        with col2:
                            st.metric("Testing RMSE", f"{test_rmse:.4f}")
                            st.metric("Testing MAE", f"{test_mae:.4f}")
                        
                        # Plot training history
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                        fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save the model for download
                        model.save("trained_model.h5")
                        
                        with open("trained_model.h5", "rb") as f:
                            model_bytes = f.read()
                        
                        st.download_button(
                            label="Download Trained Model",
                            data=model_bytes,
                            file_name="trained_model.h5",
                            mime="application/octet-stream"
                        )
                else:
                    st.info("Configure the training options and click 'Train Model' to proceed.")
                    return
            
            # Generate forecast
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast... This may take a moment."):
                    # Prepare data for forecasting
                    data_values = data_df['Level'].values.reshape(-1, 1)
                    
                    # Scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data_scaled = scaler.fit_transform(data_values)
                    
                    # Get the last sequence for forecasting
                    last_sequence = data_scaled[-sequence_length:]
                    
                    # Generate forecast with uncertainty
                    forecast_mean, forecast_lower, forecast_upper = predict_with_dropout_uncertainty(
                        model, last_sequence, forecast_days, mc_iterations, scaler, sequence_length
                    )
                    
                    # Generate forecast dates
                    last_date = data_df['Date'].iloc[-1]
                    forecast_dates = generate_forecast_dates(last_date, forecast_days)
                    
                    # Create forecast plot
                    forecast_fig = create_forecast_plot(
                        data_df['Date'], data_df['Level'],
                        forecast_dates, forecast_mean,
                        forecast_lower, forecast_upper
                    )
                    
                    # Display the forecast plot
                    st.subheader("Forecast Results")
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Create forecast table
                    forecast_table = create_forecast_table(
                        forecast_dates, forecast_mean,
                        forecast_lower, forecast_upper
                    )
                    
                    # Display the forecast table
                    st.subheader("Forecast Data")
                    st.dataframe(forecast_table)
                    
                    # Save the plot as an image for the PDF
                    forecast_fig.write_image("forecast_plot.png")
                    
                    # Create the PDF report
                    create_forecast_pdf(forecast_table, "forecast_plot.png", "forecast_report.pdf")
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open("forecast_report.pdf", "rb") as f:
                            pdf_bytes = f.read()
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name="forecast_report.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        csv = forecast_table.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV Data",
                            data=csv,
                            file_name="forecast_data.csv",
                            mime="text/csv"
                        )
                    
                    # Log forecast generation
                    log_visitor_activity("Forecasting Page", "generate_forecast")

def render_ai_chat_page():
    st.header("AI Chat Assistant")
    
    # Check if authentication is required
    if is_restricted_feature("ai_chat") and not can_access_feature("ai_chat"):
        render_auth_required_message()
        return
    
    # Log feature access
    log_visitor_activity("AI Chat Page")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message ai-message">{message["content"]}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Ask me anything about groundwater or hydrology:", key="chat_input")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate AI response
        if gemini_configured:
            try:
                # Create chat session
                chat = gemini_model_chat.start_chat(history=[])
                
                # Add context about groundwater and hydrology
                system_prompt = """
                You are a hydrologist specializing in groundwater resources. 
                Provide accurate, scientific information about groundwater, hydrology, water resource management, and related topics.
                Base your responses on established scientific knowledge and best practices in the field.
                If you're unsure about something, acknowledge the limitations of your knowledge.
                Keep responses concise but informative, and use a professional tone.
                """
                
                # Add system prompt and chat history to the context
                messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Generate response
                response = chat.send_message(user_input)
                ai_response = response.text
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Log chat interaction
                log_visitor_activity("AI Chat Page", "chat_message")
                
                # Rerun to update the UI
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
                ai_response = "I'm sorry, I encountered an error while processing your request. Please try again later."
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        else:
            ai_response = "I'm sorry, the AI chat feature is currently unavailable because the Gemini API is not configured."
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def render_ai_report_page():
    st.header("AI Groundwater Analysis Report")
    
    # Check if authentication is required
    if is_restricted_feature("ai_report") and not can_access_feature("ai_report"):
        render_auth_required_message()
        return
    
    # Log feature access
    log_visitor_activity("AI Report Page")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your groundwater level data (Excel format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load and clean data
        data_df = load_and_clean_data(uploaded_file.read())
        
        if data_df is not None:
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data_df.head())
            
            # Plot historical data
            st.subheader("Historical Data")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_df['Date'], y=data_df['Level'], mode='lines', name='Historical Data'))
            fig.update_layout(title='Historical Groundwater Level', xaxis_title='Date', yaxis_title='Level')
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate a simple forecast for the report
            st.subheader("Generate AI Report")
            
            forecast_days = st.slider("Forecast Horizon for Report (Days)", min_value=7, max_value=365, value=30, step=1)
            
            if st.button("Generate AI Report"):
                with st.spinner("Generating report... This may take a moment."):
                    # Use standard model for forecasting
                    model, sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                    
                    if model is None:
                        st.error("Standard model could not be loaded. AI report generation is not available.")
                        return
                    
                    # Prepare data for forecasting
                    data_values = data_df['Level'].values.reshape(-1, 1)
                    
                    # Scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data_scaled = scaler.fit_transform(data_values)
                    
                    # Get the last sequence for forecasting
                    last_sequence = data_scaled[-sequence_length:]
                    
                    # Generate forecast with uncertainty
                    forecast_mean, forecast_lower, forecast_upper = predict_with_dropout_uncertainty(
                        model, last_sequence, forecast_days, 30, scaler, sequence_length
                    )
                    
                    # Generate forecast dates
                    last_date = data_df['Date'].iloc[-1]
                    forecast_dates = generate_forecast_dates(last_date, forecast_days)
                    
                    # Create forecast table
                    forecast_table = create_forecast_table(
                        forecast_dates, forecast_mean,
                        forecast_lower, forecast_upper
                    )
                    
                    # Generate AI report
                    report = generate_ai_report(data_df, forecast_table, "LSTM Neural Network")
                    
                    # Display the report
                    st.subheader("AI Analysis Report")
                    st.markdown(report)
                    
                    # Create a downloadable version
                    report_md = f"# Groundwater Analysis Report\n\n{report}"
                    
                    with open("ai_report.md", "w") as f:
                        f.write(report_md)
                    
                    # Convert to PDF
                    os.system("manus-md-to-pdf ai_report.md ai_report.pdf")
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open("ai_report.pdf", "rb") as f:
                            pdf_bytes = f.read()
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name="ai_report.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Markdown Report",
                            data=report_md,
                            file_name="ai_report.md",
                            mime="text/markdown"
                        )
                    
                    # Log report generation
                    log_visitor_activity("AI Report Page", "generate_report")

# --- Run the app ---
if __name__ == "__main__":
    main()
