import streamlit as st
import os
import json
import firebase_admin
from firebase_admin import credentials, db
import uuid
import datetime
import hashlib
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit.components.v1 as components # Import components
from fpdf import FPDF # For PDF generation
import io # For PDF in-memory generation
import base64 # For PDF download
import numpy as np # For data analysis numeric checks

# --- Constants ---
ADVANCED_FEATURE_LIMIT = 5
APP_NAME = "DeepHydro"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") # Updated password

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
                })
                return True
            else:
                return False
        except Exception as e:
            return False
    return True

# --- User Identification & Tracking ---
def get_client_ip():
    """Get the client's IP address if available."""
    try:
        response = requests.get('https://httpbin.org/ip', timeout=3)
        if response.status_code == 200:
            return response.json().get('origin', 'Unknown')
        return "Unknown"
    except Exception:
        return "Unknown"

def get_persistent_user_id():
    """Generate or retrieve a persistent user ID."""
    if st.session_state.get('google_auth_status') and st.session_state.get('google_user_info'):
        user_id = st.session_state.google_user_info.get('id', st.session_state.google_user_info.get('email'))
        if user_id:
            st.session_state.persistent_user_id = user_id
            return user_id

    if 'persistent_user_id' in st.session_state and st.session_state.persistent_user_id and not st.session_state.persistent_user_id.startswith('anon_'):
         if st.session_state.get('google_auth_status'): pass
         else: return st.session_state.persistent_user_id

    ip_address = get_client_ip()
    user_agent = st.session_state.get('user_agent', 'Unknown')

    hash_input = f"{ip_address}-{user_agent}"
    hashed_id = hashlib.sha256(hash_input.encode()).hexdigest()
    persistent_id = f"anon_{hashed_id}"

    if 'persistent_user_id' not in st.session_state or st.session_state.persistent_user_id is None or st.session_state.persistent_user_id.startswith('anon_'):
        st.session_state.persistent_user_id = persistent_id

    return st.session_state.persistent_user_id

def get_or_create_user_profile(user_id):
    """Get user profile from Firebase or create a new one."""
    if not firebase_admin._apps:
        return None, False
    
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
                'is_authenticated': st.session_state.get('google_auth_status', False),
                'feature_usage_count': 0,
                'last_visit': datetime.datetime.now().isoformat(),
                'google_user_info': st.session_state.get('google_user_info')
            }
            ref.set(profile)
        else:
            if 'session_visit_logged' not in st.session_state:
                profile['visit_count'] = profile.get('visit_count', 0) + 1
                profile['last_visit'] = datetime.datetime.now().isoformat()
                profile['is_authenticated'] = st.session_state.get('google_auth_status', False)
                if profile['is_authenticated']:
                     profile['google_user_info'] = st.session_state.get('google_user_info')
                ref.update({'visit_count': profile['visit_count'],
                            'last_visit': profile['last_visit'],
                            'is_authenticated': profile['is_authenticated'],
                            'google_user_info': profile.get('google_user_info')})
                st.session_state.session_visit_logged = True
            
        return profile, is_new_user
    except Exception as e:
        return None, False

def increment_feature_usage(user_id):
    """Increment the feature usage count for the user in Firebase."""
    if not firebase_admin._apps:
        return False
    
    try:
        ref = db.reference(f'users/{user_id}/feature_usage_count')
        current_count = ref.get() or 0
        new_count = current_count + 1
        ref.set(new_count)
        
        if 'user_profile' in st.session_state:
            updated_profile = ref.parent.get()
            if updated_profile:
                st.session_state.user_profile = updated_profile
            else:
                 if st.session_state.user_profile:
                     st.session_state.user_profile['feature_usage_count'] = new_count
        return True
    except Exception as e:
        return False

# --- Authentication Check & Google Login ---
def check_feature_access():
    """Check if user can access advanced features based on usage count and auth status."""
    if 'user_profile' not in st.session_state or st.session_state.user_profile is None:
        user_id = get_persistent_user_id()
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        if st.session_state.user_profile is None:
            return False, "Cannot verify usage limit. Access denied."

    usage_count = st.session_state.user_profile.get('feature_usage_count', 0)
    is_authenticated = st.session_state.get('google_auth_status', False)

    if is_authenticated:
        return True, "Access granted (Authenticated)."
    elif usage_count < ADVANCED_FEATURE_LIMIT:
        return True, f"Access granted (Usage: {usage_count}/{ADVANCED_FEATURE_LIMIT})."
    else:
        return False, f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached. Please log in with Google to continue."

def show_google_login_button():
    """Displays the Google Sign-In button and handles the callback."""
    st.warning(f"Usage limit ({ADVANCED_FEATURE_LIMIT}) reached for advanced features.")
    st.info("Please log in with Google to continue using AI Report, Forecasting, and AI Chat.")

    google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not google_client_id:
        st.error("Google Client ID not configured. Cannot enable Google Sign-In. Set GOOGLE_CLIENT_ID environment variable.")
        return

    components.html(f"""
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        <div id="g_id_onload"
             data-client_id="{google_client_id}"
             data-callback="handleCredentialResponse"
             data-auto_prompt="false">
        </div>
        <div class="g_id_signin"
             data-type="standard"
             data-size="large"
             data-theme="outline"
             data-text="sign_in_with"
             data-shape="rectangular"
             data-logo_alignment="left">
        </div>
        <script>
          function handleCredentialResponse(response) {{
            console.log("Encoded JWT ID token: " + response.credential);
            window.parent.postMessage({{
                'type': 'streamlit:setComponentValue',
                'key': 'google_auth_callback',
                'value': response.credential
            }}, '*');
          }}
        </script>
        """, height=100)

    credential_token = st.session_state.get('google_auth_callback')
    if credential_token:
        try:
            # Placeholder for actual JWT decoding and verification
            # In a real app, you'd use a library like 'google-auth' on a backend.
            # For this demo, we'll use a placeholder/dummy decode.
            # import jwt # pip install pyjwt -- needed for actual JWT parsing
            # decoded_token = jwt.decode(credential_token, options={"verify_signature": False}) # This is insecure for production
            decoded_token = { # Mock for demo
                'email': 'user@example.com',
                'name': 'Demo User',
                'picture': 'https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&s=96',
                'sub': 'mock_google_id_12345'
            }

            st.session_state.google_auth_status = True
            st.session_state.google_user_info = {
                'id': decoded_token.get('sub'),
                'email': decoded_token.get('email'),
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture')
            }
            st.session_state.persistent_user_id = st.session_state.google_user_info.get('id', st.session_state.google_user_info.get('email'))

            user_id = get_persistent_user_id()
            profile, _ = get_or_create_user_profile(user_id)
            if profile:
                try:
                    ref = db.reference(f'users/{user_id}')
                    update_data = {
                        'is_authenticated': True,
                        'google_user_info': st.session_state.google_user_info,
                        'last_login_google': datetime.datetime.now().isoformat()
                    }
                    if profile.get('visit_count', 0) <= 1:
                         update_data['first_visit'] = profile.get('first_visit', datetime.datetime.now().isoformat())
                         update_data['visit_count'] = 1
                         update_data['feature_usage_count'] = profile.get('feature_usage_count', 0)
                    ref.update(update_data)
                    st.session_state.user_profile = ref.get()
                    st.success("Google login successful! Advanced features unlocked.")
                    st.session_state.google_auth_callback = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Firebase error updating profile after login: {e}")
            else:
                 st.error("Could not update user profile after login.")
        except Exception as e:
            st.error(f"Error processing Google Sign-In: {e}")
            st.session_state.google_auth_callback = None

# --- Visitor Analytics Functions ---
def get_session_id():
    """Create or retrieve a unique session ID."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_visitor_activity(page_name, action="page_view", feature_used=None):
    """Log visitor activity to Firebase."""
    if not firebase_admin._apps:
        return
    try:
        user_id = get_persistent_user_id()
        profile, _ = get_or_create_user_profile(user_id)
        
        should_increment = feature_used in ['Forecast', 'AI Report', 'AI Chat']
        access_granted, _ = check_feature_access()
        
        if should_increment:
            if access_granted:
                increment_feature_usage(user_id)
            else:
                action = f"denied_{action}"

        ref = db.reference('visitors_log')
        log_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        session_id = get_session_id()
        user_agent = st.session_state.get('user_agent', 'Unknown')
        ip_address = get_client_ip()
        is_authenticated = st.session_state.get('google_auth_status', False)

        log_data = {
            'timestamp': timestamp,
            'persistent_user_id': user_id,
            'is_authenticated': is_authenticated,
            'visit_count': profile.get('visit_count', 1) if profile else 1,
            'ip_address': ip_address,
            'page': page_name,
            'action': action,
            'feature_used': feature_used,
            'session_id': session_id,
            'user_agent': user_agent,
            'google_email': st.session_state.google_user_info.get('email') if is_authenticated else None
        }
        ref.child(log_id).set(log_data)
    except Exception as e:
        pass # Silently fail logging

def fetch_visitor_logs():
    """Fetch visitor logs from Firebase."""
    if not firebase_admin._apps:
        return pd.DataFrame()
    try:
        ref = db.reference('visitors_log')
        visitors_data = ref.get()
        if not visitors_data: return pd.DataFrame()
        visitors_list = [dict(log_id=log_id, **data) for log_id, data in visitors_data.items()]
        df = pd.DataFrame(visitors_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        return df
    except Exception as e:
        st.error(f"Error fetching visitor logs: {e}")
        return pd.DataFrame()

def fetch_user_profiles():
    """Fetch all user profiles from Firebase."""
    if not firebase_admin._apps:
        return pd.DataFrame()
    try:
        ref = db.reference('users')
        users_data = ref.get()
        if not users_data: return pd.DataFrame()
        
        users_list = []
        for user_id, profile_data in users_data.items():
            profile_data['user_id'] = user_id # Add user_id to the dict
            users_list.append(profile_data)
            
        df = pd.DataFrame(users_list)
        # Convert timestamps if they exist
        for col in ['first_visit', 'last_visit', 'last_login_google']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error fetching user profiles: {e}")
        return pd.DataFrame()

def create_admin_charts(visitor_df, user_profiles_df):
    """Create comprehensive visualizations for admin dashboard."""
    figures = []

    # 1. Daily Unique Visitors (from visitor_df)
    if not visitor_df.empty:
        df_daily_visitors = visitor_df.copy()
        df_daily_visitors['date'] = df_daily_visitors['timestamp'].dt.date
        daily_unique_visitors = df_daily_visitors.groupby('date')['persistent_user_id'].nunique().reset_index(name='unique_users')
        daily_unique_visitors['date'] = pd.to_datetime(daily_unique_visitors['date'])
        fig1 = px.line(daily_unique_visitors, x='date', y='unique_users', title='Daily Unique Visitors', labels={'unique_users': 'Unique Users', 'date': 'Date'},
                       color_discrete_sequence=px.colors.qualitative.Plotly)
        figures.append(fig1)
    else:
        fig1 = go.Figure().update_layout(title="Daily Unique Visitors (No Data)")
        fig1.add_annotation(text="Not enough visitor log data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        figures.append(fig1)

    # 2. Activity Counts by Action (from visitor_df)
    if not visitor_df.empty:
        action_counts = visitor_df['action'].value_counts().reset_index()
        action_counts.columns = ['action', 'count']
        fig2 = px.bar(action_counts, x='action', y='count', title='Activity Counts by Action', labels={'count': 'Number of Times', 'action': 'Action Type'},
                      color_discrete_sequence=px.colors.qualitative.Bold)
        figures.append(fig2)
    else:
        fig2 = go.Figure().update_layout(title="Activity Counts by Action (No Data)")
        fig2.add_annotation(text="Not enough visitor log data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        figures.append(fig2)

    # 3. User Authentication Status (from user_profiles_df)
    if not user_profiles_df.empty:
        auth_counts = user_profiles_df['is_authenticated'].value_counts(normalize=True).reset_index()
        auth_counts.columns = ['is_authenticated', 'percentage']
        auth_counts['status'] = auth_counts['is_authenticated'].map({True: 'Authenticated (Google)', False: 'Anonymous'})
        fig3 = px.pie(auth_counts, values='percentage', names='status', title='Overall User Authentication Status (%)',
                      hole=0.3, # Donut chart
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        figures.append(fig3)
    else:
        fig3 = go.Figure().update_layout(title="Overall User Authentication Status (No Data)")
        fig3.add_annotation(text="No user profile data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        figures.append(fig3)

    # 4. Feature Usage Breakdown (from visitor_df)
    feature_usage_df = visitor_df[visitor_df['feature_used'].notna()]
    if not feature_usage_df.empty:
        feature_counts = feature_usage_df['feature_used'].value_counts().reset_index()
        feature_counts.columns = ['feature', 'count']
        fig4 = px.bar(feature_counts, x='feature', y='count', title='Feature Usage Breakdown', labels={'count': 'Times Used', 'feature': 'Feature Name'},
                      color_discrete_sequence=px.colors.qualitative.Safe)
        figures.append(fig4)
    else:
        fig4 = go.Figure().update_layout(title="Feature Usage Breakdown (No Data)")
        fig4.add_annotation(text="No feature usage data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        figures.append(fig4)

    # 5. Hourly Activity Heatmap (from visitor_df)
    if not visitor_df.empty:
        try:
            df_heatmap = visitor_df.copy()
            df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
            df_heatmap['day_of_week'] = df_heatmap['timestamp'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hourly_activity = df_heatmap.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            
            if len(hourly_activity) > 0:
                hourly_pivot = hourly_activity.pivot_table(values='count', index='day_of_week', columns='hour', fill_value=0)
                available_days = set(hourly_pivot.index) & set(day_order)
                ordered_available_days = [day for day in day_order if day in available_days]
                hourly_pivot = hourly_pivot.reindex(ordered_available_days)
                available_hours = sorted(hourly_pivot.columns)
                fig5 = px.imshow(hourly_pivot, labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
                                x=[str(h) for h in available_hours], y=ordered_available_days, title="Visitor Activity by Hour and Day",
                                color_continuous_scale="Viridis")
                figures.append(fig5)
            else:
                fig5 = go.Figure().update_layout(title="Visitor Activity by Hour and Day (No Data)")
                fig5.add_annotation(text="Not enough data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                figures.append(fig5)
        except Exception as heatmap_err:
            fig5 = go.Figure().update_layout(title="Visitor Activity by Hour and Day (Error)")
            fig5.add_annotation(text=f"Error generating heatmap: {heatmap_err}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            figures.append(fig5)
    else:
        fig5 = go.Figure().update_layout(title="Visitor Activity by Hour and Day (No Data)")
        fig5.add_annotation(text="Not enough visitor log data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        figures.append(fig5)

    return figures

# --- Admin Analytics Dashboard (Protected) ---
def render_admin_analytics():
    """Render the admin analytics dashboard with password protection."""
    st.header("Admin Analytics Dashboard")
    st.markdown("""
    This secure dashboard provides deep insights into user behavior, feature usage, and overall application activity.
    """)

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        password = st.text_input("Enter Admin Password", type="password", key="admin_password")
        if st.button("Unlock Dashboard", key="unlock_admin_btn"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Admin dashboard unlocked!")
                # st.rerun() # No rerun needed, content will appear
            else:
                st.error("Incorrect password.")
    
    if st.session_state.admin_logged_in:
        st.subheader("Overview")
        user_profiles_df = fetch_user_profiles()
        visitor_df = fetch_visitor_logs()

        total_users = user_profiles_df['user_id'].nunique() if not user_profiles_df.empty else 0
        total_visits = visitor_df['session_id'].nunique() if not visitor_df.empty else 0
        total_feature_usages = visitor_df[visitor_df['feature_used'].notna()].shape[0] if not visitor_df.empty else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Total Visits", total_visits)
        with col3:
            st.metric("Total Feature Usages", total_feature_usages)

        st.markdown("---")

        st.subheader("Detailed Analytics Visualizations")
        charts = create_admin_charts(visitor_df, user_profiles_df)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
            st.markdown("---") # Separator between charts

        st.subheader("Raw Data Tables")
        if not user_profiles_df.empty:
            st.write("### User Profiles")
            st.dataframe(user_profiles_df)
        if not visitor_df.empty:
            st.write("### Visitor Log")
            st.dataframe(visitor_df)
        
        if st.button("Lock Dashboard", key="lock_admin_btn_bottom"):
            st.session_state.admin_logged_in = False
            st.rerun() # Rerun to hide content and show password prompt

    log_visitor_activity("Admin Analytics")

# --- PDF Generation Function (Enhanced) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'DeepHydro AI Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f'Generated by DeepHydro - {datetime.date.today().isoformat()}', 0, 0, 'L')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(220, 230, 240) # Light blue background
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body)
        self.ln(6)

def generate_professional_report_pdf(report_content, title="DeepHydro AI Report"):
    """Generates a professional PDF report."""
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15) # Set auto page break

    # Add sections based on a structured report_content (mocked for now)
    sections = {
        "Introduction": "This report provides an AI-driven summary of your hydrological data and/or forecasts, designed to offer actionable insights for water resource management.",
        "Analysis Summary": report_content, # Main content of the AI report
        "Key Findings & Recommendations": "Based on the analysis, key findings indicate [example finding]. We recommend [example recommendation] to optimize resource management and mitigate risks.",
        "Disclaimer": "This report is generated by an AI model and should be used for informational purposes only. Consult with experts for critical decisions."
    }

    for section_title, section_body in sections.items():
        pdf.chapter_title(section_title)
        pdf.chapter_body(section_body)
    
    pdf_output = pdf.output(dest='S').encode('latin1')
    return base64.b64encode(pdf_output).decode('latin1')

# --- Pages Definitions ---
def render_home_page():
    st.markdown(f"""
    <style>
        .montserrat-font {{
            font-family: 'Montserrat', sans-serif;
        }}
        .blue-text {{
            color: #007bff; /* Professional blue */
        }}
    </style>
    <div class="main-header">
        <h1 class="montserrat-font">{APP_NAME}</h1>
        <p class="tagline montserrat-font">DeepHydro revolutionizes water resource management. Leveraging cutting-edge <span class="blue-text">AI</span>, it transforms complex hydrological data into actionable forecasts and insightful analytics, empowering sustainable decisions.</p>
    </div>
    <div class="content-section">
        <h2>Welcome to DeepHydro!</h2>
        <p>Your comprehensive platform for advanced hydrological analysis and forecasting. Upload your data, leverage our deep learning models, and gain unparalleled insights into water resources.</p>
        <p>Explore our features:</p>
        <ul>
            <li><b>📊 Data Analysis:</b> Upload and deeply analyze your numerical datasets.</li>
            <li><b>📈 AI Forecasting:</b> Predict future trends using our advanced AI models.</li>
            <li><b>📝 AI Report Generation:</b> Get intelligent, AI-powered summaries of your analyses.</li>
            <li><b>💬 AI Chat:</b> Interact with our AI for quick insights and support.</li>
        </ul>
        <p>Get started by navigating through the sidebar!</p>
    </div>
    """, unsafe_allow_html=True)
    log_visitor_activity("Home")

def render_data_analysis_page():
    st.header("Data Analysis Dashboard")
    st.markdown("""
    Upload your numerical datasets (CSV, Excel) for deep analysis and interactive visualization.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['uploaded_data_df'] = df # Store in session state for further use
            st.success("File uploaded successfully!")
            
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            st.write(f"Shape of data: {df.shape[0]} rows, {df.shape[1]} columns")

            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe())

            st.subheader("Data Visualization")
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found for plotting.")
                return

            chart_type = st.selectbox("Select Chart Type", ["Line Plot", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])

            if chart_type == "Line Plot":
                x_col = st.selectbox("Select X-axis (Often Date/Time or Index)", df.columns.tolist() + ['(None)'], index=len(df.columns.tolist()))
                y_col = st.selectbox("Select Y-axis (Numeric)", numeric_cols)
                if x_col != '(None)' and y_col:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                elif y_col:
                     st.info("Select an X-axis for a time series plot, or switch to another chart type for other visualizations.")

            elif chart_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis (Numeric)", numeric_cols)
                y_col = st.selectbox("Select Y-axis (Numeric)", numeric_cols)
                color_col = st.selectbox("Select Color (Categorical, Optional)", ['(None)'] + [col for col in df.columns if col not in numeric_cols])
                
                if x_col and y_col:
                    if color_col != '(None)':
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot of {x_col} vs {y_col}")
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram":
                col_hist = st.selectbox("Select Column for Histogram (Numeric)", numeric_cols)
                if col_hist:
                    fig = px.histogram(df, x=col_hist, title=f"Histogram of {col_hist}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot":
                col_box = st.selectbox("Select Column for Box Plot (Numeric)", numeric_cols)
                if col_box:
                    fig = px.box(df, y=col_box, title=f"Box Plot of {col_box}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Correlation Heatmap":
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Correlation Heatmap of Numeric Features",
                                color_continuous_scale=px.colors.sequential.Plasma)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Deep Analysis Insights (Coming Soon!)")
            st.info("In future updates, this section will offer automated insights based on your data, including trend detection, outlier analysis, and predictive summaries using AI.")

        except Exception as e:
            st.error(f"Error processing file: {e}. Please ensure it's a valid CSV or Excel file with numerical data.")
    else:
        st.info("Please upload a file to begin data analysis.")
    log_visitor_activity("Data Analysis")

def render_forecasting_page():
    st.header("AI Forecasting")
    access_granted, message = check_feature_access()
    if access_granted:
        st.success(message)
        st.write("This section will allow you to input data and generate future hydrological forecasts using our advanced AI models (e.g., your `forecasting.h5` Keras model).")
        st.warning("Please provide the full Streamlit code so I can integrate the forecasting model and its specific input/output requirements.")
        
        st.subheader("Input Data for Forecasting")
        st.text_input("Enter input features for forecasting (e.g., comma-separated values):", key="forecast_input_data")
        
        # Placeholder for actual forecast results DataFrame
        if 'forecast_results_df' not in st.session_state:
            st.session_state.forecast_results_df = pd.DataFrame({
                'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
                'Predicted Value': [10.5, 12.1, 11.8]
            })

        if st.button("Generate Forecast", key="generate_forecast_btn"):
            with st.spinner("Generating forecast..."):
                # Placeholder for actual model prediction
                # Load your forecasting.h5 model here and make predictions
                # from tensorflow.keras.models import load_model
                # model = load_model('forecasting.h5')
                # prediction = model.predict(processed_input_data)
                
                # Mock forecast results for demonstration
                mock_data = {
                    'Date': pd.to_datetime(['2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07']),
                    'Predicted Value': np.random.uniform(9.0, 15.0, 4)
                }
                st.session_state.forecast_results_df = pd.DataFrame(mock_data)
                st.success("Forecast generated!")
        
        if not st.session_state.forecast_results_df.empty:
            st.subheader("Forecast Results")
            st.dataframe(st.session_state.forecast_results_df)

            # Download options
            st.subheader("Download Forecast Results")
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                csv_data = st.session_state.forecast_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="DeepHydro_Forecast.csv",
                    mime="text/csv",
                    key="download_forecast_csv"
                )
            with col_xlsx:
                # Requires openpyxl: pip install openpyxl
                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
                    st.session_state.forecast_results_df.to_excel(writer, index=False, sheet_name='Forecast')
                xlsx_buffer.seek(0)
                st.download_button(
                    label="Download as XLSX",
                    data=xlsx_buffer,
                    file_name="DeepHydro_Forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_forecast_xlsx"
                )

        st.subheader("Model Evaluation")
        st.info("The model's performance metrics (e.g., RMSE, MAE, MAPE) will be displayed and saved here, just like in your original code. You can integrate your existing evaluation saving logic here.")
        # Placeholder for where original evaluation saving logic (mean_squared_error, etc.) would go
        # Example:
        # if 'evaluation_results' in st.session_state:
        #     st.write("RMSE:", st.session_state.evaluation_results.get('rmse'))
        #     st.write("MAE:", st.session_state.evaluation_results.get('mae'))
        #     st.write("MAPE:", st.session_state.evaluation_results.get('mape'))
        
        log_visitor_activity("Forecasting", "feature_used", "Forecast")
    else:
        st.warning(message)
        show_google_login_button()
        log_visitor_activity("Forecasting", "access_denied", "Forecast")

def render_ai_report_page():
    st.header("AI Report Generation")
    access_granted, message = check_feature_access()
    if access_granted:
        st.success(message)
        st.write("Generate comprehensive, AI-powered reports based on your data analysis and forecasts.")
        st.warning("Please provide the full Streamlit code so I can integrate the Gemini API for intelligent report generation.")
        
        st.subheader("AI Report Options (Integration Pending Full Code)")
        report_context = st.text_area("Provide context or specific data points for the report (e.g., 'Summarize the rainfall trends for the last 5 years'):", height=150, key="report_context")
        
        if st.button("Generate AI Report", key="generate_ai_report_btn"):
            if report_context:
                with st.spinner("Generating professional AI Report..."):
                    # Placeholder for Gemini API call to generate report content
                    generated_report_content = f"This AI-generated report provides insights based on your request: '{report_context}'.\n\n" \
                                               f"Our advanced Deep Learning models have analyzed the provided context to synthesize a comprehensive summary. \n\n" \
                                               f"### Analysis Highlights:\n" \
                                               f"- Identified trends and patterns.\n" \
                                               f"- Notable observations and potential anomalies.\n\n" \
                                               f"### Predictive Outlook:\n" \
                                               f"- Future projections based on AI models.\n" \
                                               f"- Probable scenarios and their implications.\n\n" \
                                               f"This report aims to empower data-driven decisions in hydrological management. For detailed numerical analysis, please refer to the Data Analysis and AI Forecasting sections of the DeepHydro platform."
                    
                    st.session_state['generated_ai_report'] = generated_report_content
                    st.success("Report generated!")
            else:
                st.warning("Please provide some context for the report.")

        if 'generated_ai_report' in st.session_state and st.session_state['generated_ai_report']:
            st.subheader("Generated Report Preview")
            st.markdown(st.session_state['generated_ai_report'])
            
            # Download PDF Button
            pdf_b64 = generate_professional_report_pdf(st.session_state['generated_ai_report'])
            st.download_button(
                label="Download Professional Report as PDF",
                data=base64.b64decode(pdf_b64),
                file_name="DeepHydro_AI_Report.pdf",
                mime="application/pdf",
                key="download_pdf_btn",
                help="Click to download the generated report as a professionally designed PDF."
            )
        else:
            st.info("AI-generated report will appear here after generation.")
            
        log_visitor_activity("AI Report", "feature_used", "AI Report")
    else:
        st.warning(message)
        show_google_login_button()
        log_visitor_activity("AI Report", "access_denied", "AI Report")

def render_ai_chat_page():
    st.header("AI Chat Assistant")
    access_granted, message = check_feature_access()
    if access_granted:
        st.success(message)
        st.write("Interact with our intelligent AI chat assistant to ask questions about your data, forecasts, or general hydrological topics.")
        st.warning("Please provide the full Streamlit code so I can integrate the Gemini API for interactive chat.")
        
        st.subheader("Chat with AI (Integration Pending Full Code)")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if chat_input := st.chat_input("Your message:", key="ai_chat_input"):
            st.session_state.messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Placeholder for actual Gemini API call
                    # response = model.generate_content(chat_input)
                    # full_response = response.text
                    full_response = f"I am currently a placeholder AI. My full capabilities (powered by Gemini) will be available once the full application code is integrated. You asked: '{chat_input}'"
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        log_visitor_activity("AI Chat", "feature_used", "AI Chat")
    else:
        st.warning(message)
        show_google_login_button()
        log_visitor_activity("AI Chat", "access_denied", "AI Chat")

def render_user_profile_page():
    st.header("Your User Profile")
    st.markdown("""
    View your account details and usage statistics.
    """)
    user_id = get_persistent_user_id()
    profile, _ = get_or_create_user_profile(user_id)
    
    if profile:
        st.subheader("Account Information")
        st.write(f"**User ID:** `{profile.get('user_id', 'N/A')}`")
        if profile.get('is_authenticated'):
            st.success("Status: Logged In with Google")
            st.write(f"**Email:** {profile.get('google_user_info', {}).get('email', 'N/A')}")
            st.write(f"**Name:** {profile.get('google_user_info', {}).get('name', 'N/A')}")
            if profile.get('google_user_info', {}).get('picture'):
                st.image(profile['google_user_info']['picture'], width=50, caption="Profile Picture")
        else:
            st.info("Status: Anonymous User")
            st.markdown("To unlock unlimited advanced features, please login with Google.")
            show_google_login_button()
        
        st.subheader("Usage Statistics")
        st.write(f"**Total Visits:** {profile.get('visit_count', 0)}")
        st.write(f"**Feature Usage (Forecasts/Reports/Chat):** {profile.get('feature_usage_count', 0)} / {ADVANCED_FEATURE_LIMIT}")
        st.write(f"**First Visit:** {profile.get('first_visit', 'N/A').split('T')[0]}")
        st.write(f"**Last Visit:** {profile.get('last_visit', 'N/A').split('T')[0]}")
    else:
        st.warning("Could not load user profile.")
    log_visitor_activity("User Profile")

# --- CSS Styling for Professional Look ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Montserrat:wght@700&display=swap');

    /* Global styles for body and Streamlit elements */
    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-color); /* Use Streamlit's theme variables */
        background-color: var(--background-color);
    }

    /* Define custom colors based on Streamlit theme for adaptability */
    :root {
        --primary-blue: #007bff;
        --dark-blue: #0056b3;
        --accent-blue: #c7e9fb;
        --light-background: #f0f2f6;
        --card-background: #ffffff;
        --border-color: #e0e0e0;

        /* Adapting to Streamlit's dark/light mode */
        --text-color: #333;
        --background-color: #f0f2f6;
        --secondary-background-color: #ffffff;
        --section-shadow: rgba(0,0,0,0.08);
    }

    /* Dark theme specific overrides */
    body[data-theme="dark"] {
        --text-color: #e0e0e0;
        --background-color: #1a1a1a;
        --secondary-background-color: #2c2c2c;
        --card-background: #2c2c2c;
        --border-color: #444;
        --section-shadow: rgba(255,255,255,0.05); /* Lighter shadow for dark theme */
    }

    /* Main Content Area Padding */
    .st-emotion-cache-z5fcl4 { /* Target main content area */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    @media (max-width: 768px) {
        .st-emotion-cache-z5fcl4 {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    /* Sidebar Styling */
    .st-emotion-cache-1ldf020 { /* Target sidebar */
        background-color: var(--secondary-background-color); /* Adapts to theme */
        border-right: 1px solid var(--border-color);
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    .st-emotion-cache-1ldf020 .st-emotion-cache-pkasvj { /* Sidebar title/logo area */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Sidebar Navigation Links */
    .st-emotion-cache-vk3x9t a { /* Links inside sidebar nav */
        color: var(--text-color);
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        display: block;
        transition: all 0.2s ease-in-out;
        text-decoration: none;
    }

    .st-emotion-cache-vk3x9t a:hover {
        background-color: var(--border-color); /* Lighter grey/darker grey on hover */
        color: var(--primary-blue);
        transform: translateX(5px);
    }

    /* Active Page in Sidebar */
    .st-emotion-cache-vk3x9t a.active {
        background-color: var(--primary-blue);
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.2);
    }

    /* Main Content Headers and Marketing Intro */
    .main-header {
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(to right, var(--accent-blue), var(--primary-blue) + 20%); /* Dynamic gradient */
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 15px var(--section-shadow);
        color: white; /* Ensure text is visible on gradient */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    body[data-theme="dark"] .main-header {
        background: linear-gradient(to right, #004080, #002040); /* Darker gradient for dark mode */
        color: #e0e0e0;
    }

    .main-header h1 {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5em;
        margin-bottom: 0.75rem;
        font-weight: 700;
    }
    .main-header .tagline {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.3em;
        max-width: 800px;
        margin: 0 auto;
        font-weight: 400;
        line-height: 1.5;
    }
    .blue-text { /* Specific style for words in intro */
        color: #87CEEB; /* A lighter, distinct blue for emphasis */
        font-weight: 700;
    }
    body[data-theme="dark"] .blue-text {
        color: #ADD8E6; /* Even lighter for dark background */
    }


    .content-section {
        background-color: var(--card-background); /* Adapts to theme */
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px var(--section-shadow);
        margin-bottom: 2rem;
    }
    .content-section h2 {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .content-section ul {
        list-style-type: none;
        padding-left: 0;
    }
    .content-section li {
        margin-bottom: 0.75rem;
        font-size: 1.1em;
        color: var(--text-color);
    }
    .content-section li b {
        color: var(--primary-blue);
    }
    body[data-theme="dark"] .content-section li b {
        color: #90CAF9; /* Lighter blue for dark theme */
    }


    /* Inputs and Buttons */
    .stTextInput>div>div>input,
    .stFileUploader>section>div>div,
    .stSelectbox>div>div,
    .stTextArea>div>div {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        padding: 0.75rem 1rem;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
        transition: all 0.2s;
    }
    .stTextInput>div>div>input:focus,
    .stFileUploader>section>div>div:focus-within,
    .stSelectbox>div>div:focus-within,
    .stTextArea>div>div:focus-within {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    }
    body[data-theme="dark"] .stTextInput>div>div>input,
    body[data-theme="dark"] .stFileUploader>section>div>div,
    body[data-theme="dark"] .stSelectbox>div>div,
    body[data-theme="dark"] .stTextArea>div>div {
        background-color: #3b3b3b;
        border-color: #555;
    }


    /* General Buttons */
    .stButton>button {
        background-color: var(--primary-blue);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1em;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.2);
    }
    .stButton>button:hover {
        background-color: var(--dark-blue);
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3);
    }

    /* Info/Warning/Error boxes */
    .st-emotion-cache-1c7y2kl { /* Info box */
        border-radius: 8px;
        padding: 1rem;
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }

    /* Adjust Streamlit specific elements for rounded corners */
    .stDataFrame, .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px var(--section-shadow);
        margin-top: 1.5rem; /* Add some space above charts/dataframes */
    }
    .stDataFrame {
        border: 1px solid var(--border-color); /* Add border to dataframes */
    }

    /* Table headers for dataframes */
    .stDataFrame table thead th {
        background-color: var(--background-color); /* Adapts to theme */
        font-weight: 600;
        color: var(--text-color);
    }

    /* Metric cards for Admin Dashboard */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px var(--section-shadow);
        border: 1px solid var(--border-color);
    }
    [data-testid="stMetric"] label {
        font-size: 1.1em;
        color: var(--text-color);
        font-weight: 600;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.2em;
        color: var(--primary-blue);
        font-weight: 700;
    }
    body[data-theme="dark"] [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #90CAF9;
    }
    [data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.9em;
        color: var(--text-color);
    }

    </style>
    """, unsafe_allow_html=True)

# --- Main App Logic ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# Initialize Firebase once
firebase_initialized = initialize_firebase()

# Get persistent user ID and profile
if firebase_initialized:
    user_id = get_persistent_user_id()
    if 'user_profile' not in st.session_state: # Only fetch if not already in session
        st.session_state.user_profile, _ = get_or_create_user_profile(user_id)
        if st.session_state.user_profile:
             st.session_state.persistent_user_id = st.session_state.user_profile.get('user_id', user_id)
else:
    st.session_state.user_profile = None # No Firebase, no profile

# Set up page configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/150x150/007bff/ffffff?text=DeepHydro+Logo", use_column_width=True) # Placeholder Logo
    st.title(APP_NAME)
    st.markdown("---")

    # Navigation buttons
    # Apply active class styling via custom CSS for current_page
    def nav_button(label, page_name, icon):
        if st.button(f"{icon} {label}", key=f"nav_{page_name}"):
            st.session_state.current_page = page_name

    nav_button("Home", "Home", "🏠")
    nav_button("Data Analysis", "Data Analysis", "📊")
    nav_button("AI Forecasting", "Forecasting", "📈")
    nav_button("AI Report", "AI Report", "📝")
    nav_button("AI Chat", "AI Chat", "💬")
    
    st.markdown("---")
    
    nav_button("Your Profile", "User Profile", "👤")
    
    # Admin access option - always visible in sidebar
    # The actual access control is handled within render_admin_analytics
    nav_button("Admin Dashboard", "Admin Analytics", "⚙️")
            
    st.markdown("---")
    st.write(f"Logged In: {st.session_state.get('google_auth_status', False)}")
    if st.session_state.get('google_auth_status'):
        st.write(f"Welcome, {st.session_state.google_user_info.get('name', 'User')}!")
        if st.session_state.google_user_info.get('picture'):
            st.image(st.session_state.google_user_info['picture'], width=30)
    else:
        st.write("Current User: Anonymous")
        
    st.write(f"Usage: {st.session_state.user_profile.get('feature_usage_count', 0) if st.session_state.user_profile else 0}/{ADVANCED_FEATURE_LIMIT}")


# --- Main Content Area ---
if st.session_state.current_page == "Home":
    render_home_page()
elif st.session_state.current_page == "Data Analysis":
    render_data_analysis_page()
elif st.session_state.current_page == "Forecasting":
    render_forecasting_page()
elif st.session_state.current_page == "AI Report":
    render_ai_report_page()
elif st.session_state.current_page == "AI Chat":
    render_ai_chat_page()
elif st.session_state.current_page == "User Profile":
    render_user_profile_page()
elif st.session_state.current_page == "Admin Analytics":
    render_admin_analytics()


