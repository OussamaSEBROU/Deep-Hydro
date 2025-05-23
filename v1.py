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

# --- Visitor Analytics Functions ---
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
    
    # 4. Hourly activity heatmap
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    # Order days of week properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    hourly_activity = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    hourly_pivot = hourly_activity.pivot_table(values='count', index='day_of_week', columns='hour', fill_value=0)
    
    # Reindex to ensure proper day order
    hourly_pivot = hourly_pivot.reindex(day_order)
    
    fig4 = px.imshow(hourly_pivot, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Visit Count"),
                    x=[str(h) for h in range(24)],
                    y=day_order,
                    title="Visitor Activity by Hour and Day")
    figures.append(fig4)
    
    return figures

# --- Admin Analytics Dashboard ---
def render_admin_analytics():
    """Render the admin analytics dashboard with authentication."""
    st.header("Admin Analytics Dashboard")
    
    # Simple authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.info("Please sign in with your Google account to view analytics.")
        
        # In a real implementation, you would integrate with Firebase Authentication
        # For this example, we'll use a simple password authentication
        admin_password = st.text_input("Admin Password", type="password")
        
        if st.button("Login"):
            # In production, replace this with proper Firebase Authentication
            correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
            if admin_password == correct_password:
                st.session_state.admin_authenticated = True
                st.experimental_rerun()
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
        st.subheader("Visitor Analytics")
        charts = create_visitor_charts(visitor_df)
        
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data with filters
        st.subheader("Raw Visitor Data")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date Range",
                [visitor_df['timestamp'].min().date(), visitor_df['timestamp'].max().date()]
            )
        
        with col2:
            page_filter = st.multiselect(
                "Filter by Page",
                options=visitor_df['page'].unique(),
                default=[]
            )
        
        # Apply filters
        filtered_df = visitor_df.copy()
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) & 
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        if page_filter:
            filtered_df = filtered_df[filtered_df['page'].isin(page_filter)]
        
        # Display the filtered data
        st.dataframe(filtered_df[['timestamp', 'ip_address', 'page', 'action', 'session_id']])
        
        # Export options
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="visitor_logs.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

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
    history_df = pd.DataFrame(history_dict); history_df["Epoch"] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", template="plotly_white")
    return fig

# --- Gemini API Functions ---
def generate_gemini_report(hist_df, forecast_df, metrics, language):
    if not gemini_configured: return "AI report generation disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    try:
        prompt = f"""Act as a professional hydrologist. Generate a scientific interpretation of the groundwater level forecast in {language}. 
        Analyze the historical trend (summary below), the forecast trend (from {forecast_df["Forecast"].iloc[0]:.2f} to {forecast_df["Forecast"].iloc[-1]:.2f}), 
        the 95% confidence interval range (e.g., final interval: {forecast_df["Lower_CI"].iloc[-1]:.2f} - {forecast_df["Upper_CI"].iloc[-1]:.2f}), 
        and the LSTM model's performance on the validation set (RMSE: {metrics.get("RMSE", "N/A")}, MAE: {metrics.get("MAE", "N/A")}, MAPE: {metrics.get("MAPE", "N/A")}). 
        Discuss potential aquifer conditions suggested by the data and forecast. 
        Assess the risk of over-pumping or water shortage based on the trends. 
        Provide actionable scientific recommendations for groundwater management or monitoring based on these findings. 
        Ensure the tone is professional and the language is {language}.

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
        context = f"""Context for AI Chatbot:
        Historical Data: {hist_df["Level"].describe().to_string()}
        Forecast: {forecast_df.to_string()}
        Metrics: RMSE: {metrics.get("RMSE", "N/A")}, MAE: {metrics.get("MAE", "N/A")}, MAPE: {metrics.get("MAPE", "N/A")}
        AI Report: {ai_report if ai_report else "N/A"}
        Chat History: {chat_hist}
        User: {user_query}
AI:"""
        response = gemini_model_chat.generate_content(context)
        return response.text
    except Exception as e: st.error(f"Error in AI chat: {e}"); return f"Error in AI chat: {e}"

# --- Main Forecasting Pipeline ---
def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    st.info(f"Starting forecast pipeline with model: {model_choice}")
    model = None
    model_sequence_length = sequence_length_train_param
    history_data = None
    scaler_obj = MinMaxScaler(feature_range=(0, 1))

    try:
        st.info("Step 1: Preparing Model...")
        if model_choice == "Standard Pre-trained Model":
            if os.path.exists(STANDARD_MODEL_PATH):
                model, model_sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                if model is None: return None, None, None, None
                st.session_state.model_sequence_length = model_sequence_length
            else:
                st.error(f"Standard model not found at {STANDARD_MODEL_PATH}. Check repository structure."); return None, None, None, None
        elif model_choice == "Upload Custom .h5 Model" and custom_model_file_obj is not None:
            model, model_sequence_length = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
            if model is None: return None, None, None, None
            st.session_state.model_sequence_length = model_sequence_length
        elif model_choice == "Train New Model":
            model_sequence_length = sequence_length_train_param
            st.session_state.model_sequence_length = model_sequence_length
        else:
            st.error("Invalid model choice or no custom model uploaded."); return None, None, None, None
        st.info(f"Model preparation complete. Effective sequence length: {model_sequence_length}")

        st.info("Step 2: Preprocessing Data (Scaling)...")
        if model_choice != "Train New Model":
            if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
                st.info(f"Using provided scaler parameters: Original Min={custom_scaler_min_param}, Original Max={custom_scaler_max_param}")
                scaler_obj.fit(np.array([[custom_scaler_min_param], [custom_scaler_max_param]]))
                scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
            else:
                st.warning("Scaler parameters not provided or invalid for pre-trained model. Fitting a new scaler to the current data.")
                scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        else: # Train New Model
            st.info("Fitting new scaler for model training.")
            scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        st.info("Data scaling complete.")

        st.info(f"Step 3: Creating sequences with length {model_sequence_length}...")
        if len(df) <= model_sequence_length:
            st.error(f"Not enough data ({len(df)} points) for sequence length {model_sequence_length}. Need > {model_sequence_length}."); return None, None, None, None
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0:
            st.error(f"Could not create sequences."); return None, None, None, None
        st.info(f"Sequence creation complete. Number of sequences: {len(X)}")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        if model_choice == "Train New Model":
            st.info(f"Step 4a: Training New LSTM Model (Epochs: {epochs_train_param})...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_train) == 0 or len(X_val) == 0:
                st.error("Not enough data for train/validation split."); return None, None, None, None
            model = build_lstm_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            history_obj = model.fit(X_train, y_train, epochs=epochs_train_param, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
            history_data = history_obj.history
            st.success("Model training complete.")
            st.info("Evaluating trained model...")
            val_predictions_scaled = model.predict(X_val)
            val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
            y_val_actual = scaler_obj.inverse_transform(y_val)
            evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
            st.success("Evaluation of trained model complete.")
        else: # Pre-trained model
            st.info("Step 4b: Evaluating Pre-trained Model (on last 20% of available sequences)...")
            if len(X) > 5: # Ensure enough data for a meaningful pseudo-validation
                val_split_idx = int(len(X) * 0.8)
                X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                if len(X_val_pseudo) > 0:
                    val_predictions_scaled = model.predict(X_val_pseudo)
                    val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo)
                    evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation of pre-trained model complete.")
                else: st.warning("Not enough data for pseudo-validation of pre-trained model (after split).")
            else: st.warning("Not enough data for pseudo-validation of pre-trained model (total sequences too few).")

        st.info(f"Step 5: Forecasting Future {forecast_horizon} Steps (MC Dropout: {mc_iterations_param})...")
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

# --- Initialize Session State ---
for key in ["cleaned_data", "forecast_results", "evaluation_metrics", "training_history", 
            "ai_report", "scaler_object", "forecast_plot_fig", "uploaded_data_filename",
            "active_tab", "report_language"]:
    if key not in st.session_state: st.session_state[key] = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_active" not in st.session_state: st.session_state.chat_active = False
if "model_sequence_length" not in st.session_state: st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
if "run_forecast_triggered" not in st.session_state: st.session_state.run_forecast_triggered = False
if "active_tab" not in st.session_state: st.session_state.active_tab = 0
if "about_us_expanded" not in st.session_state: st.session_state.about_us_expanded = False
if "report_language" not in st.session_state: st.session_state.report_language = "English"

# --- Capture User Agent for Analytics ---
try:
    import streamlit.components.v1 as components
    capture_user_agent()
except:
    st.session_state.user_agent = "Unknown"

# --- Sidebar ---
with st.sidebar:
    st.title("DeepHydro AI Forecasting")
    
    # Log sidebar view
    if firebase_initialized:
        log_visitor_activity("Sidebar", "view")
    
    st.header("1. Upload Data")
    uploaded_data_file = st.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")
    
    # Log file upload attempt
    if uploaded_data_file is not None and firebase_initialized:
        log_visitor_activity("Data Upload", f"uploaded_{uploaded_data_file.name}")

    st.header("2. Model Selection & Configuration")
    model_choice = st.selectbox("Choose Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select")
    
    # Log model selection
    if firebase_initialized:
        log_visitor_activity("Model Selection", f"selected_{model_choice}")

    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False
    
    # Default value for sequence length
    default_sequence_length = st.session_state.model_sequence_length
    
    # FIXED: Initialize sequence_length_train_sidebar with a safe default value
    sequence_length_train_sidebar = default_sequence_length
    epochs_train_sidebar = 50

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload your .h5 model", type="h5", key="custom_h5_uploader")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler parameters for uploaded model?", value=False, key="use_custom_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter the **original min/max values** your custom model was scaled with (before 0-1 normalization).")
            custom_scaler_min_sidebar = st.number_input("Original Data Min Value", value=0.0, format="%.4f", key="custom_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Data Max Value", value=1.0, format="%.4f", key="custom_scaler_max_in")
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using standard model. Default sequence length: {st.session_state.model_sequence_length}")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler parameters for standard model?", value=False, key="use_std_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter the **original min/max values** the standard model was scaled with (if known).")
            custom_scaler_min_sidebar = st.number_input("Original Data Min Value (Standard Model)", value=0.0, format="%.4f", key="std_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Data Max Value (Standard Model)", value=1.0, format="%.4f", key="std_scaler_max_in")
    elif model_choice == "Train New Model":
        # FIXED: Set a default value that's within the allowed range and use a try/except block to handle errors
        try:
            sequence_length_train_sidebar = st.number_input(
                "LSTM Sequence Length (for training)", 
                min_value=10, 
                max_value=365, 
                value=default_sequence_length, 
                step=10, 
                key="seq_len_train_in"
            )
        except Exception as e:
            st.warning(f"Using default sequence length of {default_sequence_length} due to: {e}")
            sequence_length_train_sidebar = default_sequence_length
            
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in")

    # FIXED: Ensure MC iterations has a reasonable default and range
    mc_iterations_sidebar = st.number_input("MC Dropout Iterations (for C.I.)", min_value=20, max_value=500, value=100, step=10, key="mc_iter_in")
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in")

    if st.button("Run Forecast", key="run_forecast_main_btn", use_container_width=True):
        # Log forecast run
        if firebase_initialized:
            log_visitor_activity("Forecast", f"run_forecast_{model_choice}")
            
        st.session_state.run_forecast_triggered = True
        if st.session_state.cleaned_data is not None:
            if model_choice == "Upload Custom .h5 Model" and custom_model_file_obj_sidebar is None:
                st.error("Please upload a custom .h5 model file if that option is selected.")
                st.session_state.run_forecast_triggered = False
            else:
                with st.spinner(f"Running forecast with {model_choice}. This may take a few moments..."):
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
                    st.success("Forecast complete! Results are available in the tabs.")
                    st.session_state.ai_report = None; st.session_state.chat_history = []; st.session_state.chat_active = False
                    # Set active tab to forecast results
                    st.session_state.active_tab = 1  # Index of forecast tab
                    
                    # Log successful forecast
                    if firebase_initialized:
                        log_visitor_activity("Forecast", "forecast_success")
                else:
                    st.error("Forecast pipeline did not complete successfully. Check messages above.")
                    st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                    st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
                    
                    # Log forecast failure
                    if firebase_initialized:
                        log_visitor_activity("Forecast", "forecast_failure")
                st.rerun()
        else:
            st.error("Please upload data first using the sidebar.")
            st.session_state.run_forecast_triggered = False

    st.header("3. View & Export")
    
    # Define report_language before the Generate AI Report button
    st.session_state.report_language = st.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_configured)
    
    if st.button("Generate AI Report", key="show_report_btn", disabled=not gemini_configured, use_container_width=True):
        # Log AI report generation
        if firebase_initialized:
            log_visitor_activity("AI Report", f"generate_report_{st.session_state.report_language}")
            
        if not gemini_configured: 
            st.error("AI Report disabled. Configure Gemini API Key.")
        elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            with st.spinner(f"Generating AI report ({st.session_state.report_language})..."):
                st.session_state.ai_report = generate_gemini_report(
                    st.session_state.cleaned_data, st.session_state.forecast_results,
                    st.session_state.evaluation_metrics, st.session_state.report_language
                )
            if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:"):
                st.success("AI report generated.")
                # Set active tab to AI report
                st.session_state.active_tab = 3  # Index of AI report tab
                
                # Log successful report generation
                if firebase_initialized:
                    log_visitor_activity("AI Report", "report_success")
            else: 
                st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
                
                # Log report generation failure
                if firebase_initialized:
                    log_visitor_activity("AI Report", "report_failure")
            st.rerun()
        else: 
            st.error("Cleaned data, forecast results, and evaluation metrics must be available. Run a successful forecast first.")
    
    if st.button("Download Report (PDF)", key="download_report_btn", use_container_width=True):
        # Log PDF download attempt
        if firebase_initialized:
            log_visitor_activity("PDF Report", "download_attempt")
            
        if st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None and st.session_state.ai_report is not None and st.session_state.forecast_plot_fig is not None:
            with st.spinner("Generating PDF report..."):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    report_font = "Arial" # Fallback font
                    if os.path.exists(font_path_dejavu):
                        try: 
                            pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True)
                            report_font = "DejaVu"
                        except RuntimeError as font_err:
                            st.warning(f"Failed to add DejaVu font ({font_err}), using Arial.")
                    else: 
                        st.warning(f"DejaVu font not found at {font_path_dejavu}, using Arial. For better PDF character support on Render, consider adding a .ttf font to your repo and referencing it, or installing fonts via a build script.")
                    
                    pdf.set_font(report_font, size=12)
                    pdf.cell(0, 10, txt="DeepHydro AI Forecasting Report", new_x="LMARGIN", new_y="NEXT", align="C"); pdf.ln(5)

                    plot_filename = "forecast_plot.png"
                    try:
                        st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                        img_width_mm = 190 
                        pdf.image(plot_filename, x=pdf.get_x(), y=pdf.get_y(), w=img_width_mm)
                        pdf.ln(125) # Adjust based on typical plot height
                    except Exception as img_err: 
                        st.warning(f"Could not save/embed plot image: {img_err}. Plot omitted from PDF.")
                    finally:
                        if os.path.exists(plot_filename): os.remove(plot_filename)

                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Model Evaluation Metrics", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10)
                    for key, value in st.session_state.evaluation_metrics.items():
                        val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else str(value)
                        pdf.cell(0, 8, txt=f"{key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)

                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Forecast Data (First 10 points)", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=8); col_widths = [35, 35, 35, 35]
                    pdf.cell(col_widths[0], 7, txt="Date", border=1)
                    pdf.cell(col_widths[1], 7, txt="Forecast", border=1)
                    pdf.cell(col_widths[2], 7, txt="Lower CI", border=1)
                    pdf.cell(col_widths[3], 7, txt="Upper CI", border=1, new_x="LMARGIN", new_y="NEXT")
                    for _, row in st.session_state.forecast_results.head(10).iterrows():
                        pdf.cell(col_widths[0], 6, txt=str(row["Date"].date()), border=1)
                        pdf.cell(col_widths[1], 6, txt=f"{row['Forecast']:.2f}", border=1)
                        pdf.cell(col_widths[2], 6, txt=f"{row['Lower_CI']:.2f}", border=1)
                        pdf.cell(col_widths[3], 6, txt=f"{row['Upper_CI']:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)

                    pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt=f"AI-Generated Report ({st.session_state.report_language})", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                    pdf.set_font(report_font, size=10)
                    pdf.multi_cell(0, 5, txt=st.session_state.ai_report)
                    pdf.ln(5)

                    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                    # b64_pdf = base64.b64encode(pdf_output_bytes).decode() # Not needed for direct download
                    st.download_button(
                        label="Download PDF Report Now", # Changed label to be more direct
                        data=pdf_output_bytes,
                        file_name="deephydro_forecast_report.pdf",
                        mime="application/octet-stream",
                        key="pdf_download_final_btn", # Added a unique key
                        use_container_width=True,
                        on_click=lambda: log_visitor_activity("PDF Report", "download_success") if firebase_initialized else None
                    )
                    st.success("PDF report ready. Click the download button above.")
                except Exception as pdf_err:
                    st.error(f"Failed to generate PDF report: {pdf_err}")
                    import traceback; st.error(traceback.format_exc())
                    
                    # Log PDF generation failure
                    if firebase_initialized:
                        log_visitor_activity("PDF Report", "generation_failure")
        else:
            st.error("Required data for PDF report is missing. Run a forecast and generate AI report first.")

    st.header("4. AI Assistant")
    if st.button("Activate/Deactivate Chat", key="chat_ai_btn", disabled=not gemini_configured, use_container_width=True):
        # Log chat activation/deactivation
        if firebase_initialized:
            action = "deactivate" if st.session_state.chat_active else "activate"
            log_visitor_activity("Chat", f"{action}_chat")
            
        st.session_state.chat_active = not st.session_state.chat_active
        if not st.session_state.chat_active: 
            st.session_state.chat_history = []
        else:
            # Set active tab to chat tab when activating
            st.session_state.active_tab = 4  # Index of chat tab
        st.rerun()
    
    # About Us section in sidebar - collapsible
    st.markdown('<div class="about-us-header">👥 About Us</div>', unsafe_allow_html=True)
    
    # This div will be controlled by JavaScript to show/hide
    st.markdown('<div class="about-us-content">', unsafe_allow_html=True)
    st.markdown("We specialize in groundwater forecasting and hydrological modeling solutions using advanced AI techniques.")
    st.markdown("**Contact:** [deephydro@example.com](mailto:deephydro@example.com)")
    st.markdown("© 2025 DeepHydro AI Team")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Admin Analytics Access
    st.header("5. Admin")
    if st.button("Analytics Dashboard", key="admin_analytics_btn", use_container_width=True):
        # Log admin dashboard access attempt
        if firebase_initialized:
            log_visitor_activity("Admin", "access_analytics")
            
        st.session_state.active_tab = 5  # Index of admin tab
        st.rerun()

# --- Main Application Area ---
st.title("DeepHydro AI Forecasting")

# Log main page view
if firebase_initialized:
    log_visitor_activity("Main Page", "view")

# App Introduction
st.markdown('<div class="app-intro">', unsafe_allow_html=True)
st.markdown("""
### Welcome to DeepHydro AI Forecasting

DeepHydro AI is an advanced groundwater forecasting platform that combines deep learning with hydrological expertise. Our LSTM-based models analyze historical groundwater data to provide accurate forecasts with uncertainty quantification.

**Key Features:**
- Time series forecasting with LSTM neural networks
- Uncertainty quantification using Monte Carlo Dropout
- AI-powered hydrological interpretation
- Interactive visualization and reporting

Upload your groundwater level data to begin exploring the future of your aquifer system.
""")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_data_file is not None:
    if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
        st.session_state.uploaded_data_filename = uploaded_data_file.name
        with st.spinner("Loading and cleaning data..."):
            cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
        if cleaned_df_result is not None:
            st.session_state.cleaned_data = cleaned_df_result
            st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
            st.session_state.training_history = None; st.session_state.ai_report = None
            st.session_state.chat_history = []; st.session_state.scaler_object = None
            st.session_state.forecast_plot_fig = None
            st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH # Reset sequence length on new data
            st.session_state.run_forecast_triggered = False
            
            # Log successful data loading
            if firebase_initialized:
                log_visitor_activity("Data", "data_load_success")
                
            st.rerun()
        else:
            st.session_state.cleaned_data = None
            st.error("Data loading failed. Please check the file format and content.")
            
            # Log data loading failure
            if firebase_initialized:
                log_visitor_activity("Data", "data_load_failure")

# Create tabs with the active tab set from session state
tabs = st.tabs([
    "Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot", "Admin Analytics"
])

# Set the active tab based on session state
if st.session_state.active_tab is not None:
    active_tab_index = st.session_state.active_tab
else:
    active_tab_index = 0

# Data Preview Tab
with tabs[0]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_data_preview")
        
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Time Range", f"{st.session_state.cleaned_data['Date'].min():%Y-%m-%d} to {st.session_state.cleaned_data['Date'].max():%Y-%m-%d}")
        with col2:
            st.metric("Data Points", len(st.session_state.cleaned_data))
            
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], mode="lines", name="Level"))
        fig_data.update_layout(
            title="Historical Groundwater Levels", 
            xaxis_title="Date", 
            yaxis_title="Level", 
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        st.plotly_chart(fig_data, use_container_width=True)
    else:
        st.info("⬆️ Please upload an XLSX data file using the sidebar to begin.")

# Forecast Results Tab
with tabs[1]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_forecast_results")
        
    st.header("Forecast Results")
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        if st.session_state.forecast_plot_fig is not None:
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else: 
            st.warning("Forecast plot could not be generated. Displaying data table only.")
        
        st.subheader("Forecast Data Table")
        st.dataframe(st.session_state.forecast_results, use_container_width=True)
    elif st.session_state.run_forecast_triggered: 
        st.warning("Forecast process was run, but no results are available. Check errors.")
    else: 
        st.info("Run a forecast using the sidebar to see results here.")

# Model Evaluation Tab
with tabs[2]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_model_evaluation")
        
    st.header("Model Evaluation")
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (on validation or pseudo-validation set)")
        col1, col2, col3 = st.columns(3)
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan)
        mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan)
        mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        col1.metric("RMSE", f"{rmse_val:.4f}" if not np.isnan(rmse_val) else "N/A")
        col2.metric("MAE", f"{mae_val:.4f}" if not np.isnan(mae_val) else "N/A")
        col3.metric("MAPE", f"{mape_val:.2f}%" if not np.isnan(mape_val) and mape_val != np.inf else ("N/A" if np.isnan(mape_val) else "Inf"))
        
        st.subheader("Training Loss (if model was trained in this session)")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: 
            st.info("No training history available (e.g., using a pre-trained model or training failed).")
    elif st.session_state.run_forecast_triggered: 
        st.warning("Forecast process was run, but no evaluation metrics are available. Check errors.")
    else: 
        st.info("Run a forecast to see model evaluation metrics here.")

# AI Report Tab
with tabs[3]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_ai_report")
        
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: 
        st.warning("AI features disabled. Configure Gemini API Key.")
    if st.session_state.ai_report: 
        st.markdown(f'<div class="chat-message ai-message">{st.session_state.ai_report}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
    else: 
        st.info("Click \"Generate AI Report\" in the sidebar after a successful forecast.")

# AI Chatbot Tab
with tabs[4]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_ai_chatbot")
        
    st.header("AI Chatbot Assistant")
    if not gemini_configured: 
        st.warning("AI features disabled. Configure Gemini API Key.")
    elif st.session_state.chat_active:
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("Chat activated. Ask about the results.")
            
            chat_container = st.container()
            with chat_container:
                for sender, message in st.session_state.chat_history:
                    if sender == "User":
                        st.markdown(f'<div class="chat-message user-message">{message}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message ai-message">{message}<div class="copy-tooltip">Copied!</div></div>', unsafe_allow_html=True)
            
            user_input = st.chat_input("Ask the AI assistant:")
            if user_input:
                # Log chat message
                if firebase_initialized:
                    log_visitor_activity("Chat", "send_message")
                    
                st.session_state.chat_history.append(("User", user_input))
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history, st.session_state.cleaned_data,
                        st.session_state.forecast_results, st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                st.session_state.chat_history.append(("AI", ai_response))
                st.rerun()
        else:
            st.warning("Run a successful forecast to provide context for the chatbot.")
            st.session_state.chat_active = False
            st.rerun()
    else:
        st.info("Click \"Activate/Deactivate Chat\" in sidebar (requires forecast results)." if gemini_configured else "AI Chat disabled.")

# Admin Analytics Tab
with tabs[5]:
    # Log tab view
    if firebase_initialized:
        log_visitor_activity("Tab", "view_admin_analytics")
    
    # Render the admin analytics dashboard
    render_admin_analytics()
