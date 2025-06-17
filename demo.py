# -*- coding: utf-8 -*-
"""
DeepHydro AI Streamlit Application - Investor-Ready Version

This application provides groundwater level forecasting using deep learning models,
enhanced with AI-powered analysis and a professional-grade user interface.

Key Features & Enhancements:
1.  **Modular Architecture:** Refactored functions for better organization and maintainability.
2.  **Enhanced UX/UI:**
    * Modern, clean design with custom CSS for professional aesthetics.
    * Responsive layout for various screen sizes.
    * Improved loading indicators and user feedback.
3.  **Core AI Forecasting:**
    * Support for pre-trained, custom-uploaded, or newly-trained deep learning models.
    * Monte Carlo Dropout for robust uncertainty quantification (Confidence Intervals).
4.  **AI-Powered Insights (Gemini API):**
    * Generates comprehensive, non-technical AI reports in multiple languages.
    * Interactive AI Chatbot for deep dive Q&A on data and forecasts.
5.  **Data Analysis Dashboard:** A new tab for generic data visualization and exploration of future datasets.
6.  **Session Activity Log:** A user-facing dashboard to show actions taken within the current session, demonstrating activity tracking capabilities.
7.  **PDF Report Generation:** Creates a downloadable PDF summarizing key findings, plots, and AI analysis.
8.  **Removed Features:** Google Sign-in and Firebase integration have been removed per requirements.

"""

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
import plotly.express as px
from fpdf import FPDF
import google.generativeai as genai
import io
import base64
import time
import os
import json # Used for JSON dumps, not for Firebase config anymore
import datetime
import uuid
import hashlib
import streamlit.components.v1 as components

# --- Constants & Configuration ---
# Set a default sequence length if standard model not found or cannot be loaded
DEFAULT_MODEL_SEQUENCE_LENGTH = 60 

# Path to the pre-trained model (assume it's in the same directory as app.py)
STANDARD_MODEL_PATH = "standard_model.h5"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="DeepHydro AI Forecasting", layout="wide", initial_sidebar_state="expanded")

# --- UI Assets (CSS and JavaScript) ---
def get_custom_css():
    """Returns the custom CSS string for the application."""
    return """
    <style>
    /* General Styles */
    body {
        font-family: 'Inter', sans-serif;
        color: #333;
        background-color: #f0f2f6; /* Light gray background */
    }

    /* Streamlit Overrides */
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .css-1d391kg, .css-12oz5g7 { /* Adjusting main content padding */
        padding: 1rem;
    } 

    /* Headers */
    h1, h2, h3, h4 {
        color: #2c3e50; /* Darker blue-gray for headers */
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.8rem; }
    h3 { font-size: 1.4rem; }
    h4 { font-size: 1.2rem; }

    /* Buttons */
    .stButton > button {
        background-color: #3498db; /* Primary blue */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        width: 100%; /* Make buttons full width in sidebar */
    }
    .stButton > button:hover {
        background-color: #2980b9; /* Darker blue on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar specific styles */
    .sidebar .stButton > button {
        margin-bottom: 0.5rem;
    }
    .sidebar .stFileUploader, .sidebar .stSelectbox, .sidebar .stNumberInput, .sidebar .stCheckbox {
        margin-bottom: 0.8rem;
    }
    .sidebar .stNumberInput input {
        font-size: 0.9rem;
    }
    .sidebar .stExpander {
        border: none;
        box-shadow: none;
        background-color: transparent;
    }
    .sidebar .stExpander header {
        padding: 0.5rem 0;
        font-weight: 500;
        color: #2c3e50;
    }
    .sidebar .stExpander div[data-testid="stExpanderDetails"] {
        padding-left: 0.5rem;
    }

    /* Custom Cards/Panels */
    .card-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
    }
    .app-intro {
        background-color: #e3f2fd; /* Light blue background */
        border-left: 5px solid #2196f3; /* Stronger blue border */
        color: #2196f3;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .app-intro h3 {
        color: #1976d2;
        margin-top: 0;
    }

    /* Chat Messages */
    .chat-message {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        position: relative;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        cursor: pointer;
    }
    .chat-message:hover {
        transform: translateY(-2px);
    }
    .user-message {
        background-color: #e8f5e9; /* Light green */
        border-left: 5px solid #4CAF50; /* Green accent */
        color: #388E3C;
        margin-left: 20%; /* Align right */
    }
    .ai-message {
        background-color: #e3f2fd; /* Light blue */
        border-left: 5px solid #2196f3; /* Blue accent */
        color: #1976d2;
        margin-right: 20%; /* Align left */
    }
    .copy-tooltip {
        position: absolute;
        bottom: 0.5rem;
        right: 0.5rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        z-index: 10;
    }
    .chat-message.copied .copy-tooltip {
        opacity: 1;
        visibility: visible;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; /* Remove gap between tabs */
        border-bottom: 2px solid #e0e0e0; /* Subtle tab line */
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px 8px 0 0;
        background-color: #f8f8f8; /* Light background for inactive tabs */
        color: #777;
        font-weight: 500;
        transition: all 0.3s ease;
        margin-right: 0.25rem; /* Small space between tabs */
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #3498db; /* Active tab color */
        border-top: 2px solid #3498db; /* Blue top border for active tab */
        border-left: 1px solid #e0e0e0;
        border-right: 1px solid #e0e0e0;
        border-bottom: none;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.05);
        z-index: 1; /* Bring active tab to front */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f2f6;
        color: #555;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white; /* Content panel background */
        padding: 1.5rem;
        border-radius: 0 0 12px 12px; /* Rounded bottom corners */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        border-top: none;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #34495e; /* Darker text for values */
        font-size: 1.8rem;
    }
    [data-testid="stMetricLabel"] {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .stAlert {
        border-radius: 8px;
    }

    /* About Us Section */
    .about-us-header {
        cursor: pointer;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-weight: 500;
        background-color: #ecf0f1; /* Light gray for header */
        color: #34495e;
        transition: background-color 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .about-us-header:hover {
        background-color: #dde1e2;
    }
    .about-us-content {
        background-color: #fdfefe; /* Very light background */
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #555;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .about-us-content p {
        margin-bottom: 0.5rem;
    }
    </style>
    """

def get_custom_javascript():
    """Returns the custom JavaScript string for UI interactions."""
    return """
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
    
    // Add event listeners after DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Use a small delay to ensure Streamlit elements are fully rendered
        setTimeout(function() {
            // Copy functionality for chat messages
            const chatMessages = document.querySelectorAll('.chat-message');
            chatMessages.forEach(function(message) {
                // Ensure tooltip exists
                if (!message.querySelector('.copy-tooltip')) {
                    const tooltip = document.createElement('span');
                    tooltip.className = 'copy-tooltip';
                    tooltip.textContent = 'Copied!';
                    message.appendChild(tooltip);
                }

                message.addEventListener('click', function(e) {
                    const textToCopy = this.innerText.replace('Copied!', '').trim(); // Exclude tooltip text
                    copyToClipboard(textToCopy);
                    this.classList.add('copied');
                    const tooltip = this.querySelector('.copy-tooltip');
                    if (tooltip) {
                        tooltip.style.opacity = '1';
                        tooltip.style.visibility = 'visible';
                        setTimeout(() => { 
                            this.classList.remove('copied');
                            tooltip.style.opacity = '0';
                            tooltip.style.visibility = 'hidden';
                        }, 1500);
                    }
                });
            });
            
            // Collapsible About Us section
            const aboutUsHeader = document.querySelector('.about-us-header');
            const aboutUsContent = document.querySelector('.about-us-content');
            if (aboutUsHeader && aboutUsContent) {
                // Initialize state (collapsed by default)
                if (!aboutUsContent.classList.contains('initialized')) {
                     aboutUsContent.style.display = 'none';
                     aboutUsContent.classList.add('initialized');
                }
                aboutUsHeader.addEventListener('click', function() {
                    aboutUsContent.style.display = (aboutUsContent.style.display === 'none') ? 'block' : 'none';
                });
            }
        }, 1000); // Delay helps ensure Streamlit elements are fully rendered
    });
    </script>
    """

def apply_custom_styles_and_scripts():
    """Injects custom CSS and JavaScript into the Streamlit app."""
    # Using components.html for full control over head and body injection
    # This ensures styles and scripts are loaded correctly and persist.
    components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            {get_custom_css()}
        </head>
        <body>
            {get_custom_javascript()}
        </body>
        </html>
    """, height=0, width=0) # height and width 0 means it's not visible

# Call this once at the very beginning of the script
apply_custom_styles_and_scripts()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE": # Check for placeholder
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, top_k=40, max_output_tokens=4000)
        # Use more descriptive model names if available in future Gemini versions
        gemini_model_report = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        gemini_model_chat = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        gemini_configured = True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. AI features will be disabled.")
else:
    st.warning("Gemini API Key not found or is placeholder. AI features will be disabled. "
               "Please set the 'GOOGLE_API_KEY' environment variable.")

# --- Helper Functions (Data, Model, Prediction) ---
@st.cache_data(show_spinner="Loading and Cleaning Data...")
def load_and_clean_data(uploaded_file_content):
    """
    Loads data from an Excel file, cleans it, and infers date and level columns.
    Uses st.cache_data for performance optimization.
    """
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        if df.shape[1] < 2:
            st.error("Uploaded file must have at least two columns: one for Date/Time and one for Level/Value.")
            return None
        
        # Robust column identification
        date_keywords = ["date", "time", "timestamp", "datetime"]
        level_keywords = ["level", "groundwater", "gwl", "value", "measurement"]

        date_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in date_keywords) and pd.api.types.is_datetime64_any_dtype(df[col])), 
                        next((col for col in df.columns if any(kw in str(col).lower() for kw in date_keywords)), None))
        
        level_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in level_keywords) and pd.api.types.is_numeric_dtype(df[col])), 
                         next((col for col in df.columns if any(kw in str(col).lower() for kw in level_keywords)), None))
        
        if not date_col:
            st.error("Could not find a suitable 'Date' or 'Time' column. Please ensure one exists and is recognizable.")
            return None
        if not level_col:
            st.error("Could not find a suitable 'Level' or 'Value' column. Please ensure one exists and is numeric.")
            return None
        
        st.success(f"Identified columns: Date='{date_col}', Level='{level_col}'.")
        df = df.rename(columns={date_col: "Date", level_col: "Level"})[["Date", "Level"]]
        
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
        
        initial_rows = len(df)
        df.dropna(subset=["Date", "Level"], inplace=True)
        if len(df) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df)} rows with invalid Date or Level data.")
        if df.empty:
            st.error("No valid data remaining after cleaning. Please check your file.")
            return None
        
        df = df.sort_values(by="Date").reset_index(drop=True)
        if df.duplicated(subset=["Date"]).any():
            duplicates_count = df.duplicated(subset=["Date"]).sum()
            st.warning(f"Found {duplicates_count} duplicate dates. Keeping the first occurrence for each date.")
            df = df.drop_duplicates(subset=["Date"], keep="first")
            
        if df["Level"].isnull().any():
            missing_before = df["Level"].isnull().sum()
            df["Level"] = df["Level"].interpolate(method="linear", limit_direction="both")
            if df["Level"].isnull().any():
                st.error("Could not fill all missing level values using interpolation. Data may be too sparse.")
                return None
            else:
                st.info(f"Filled {missing_before} missing level values using linear interpolation.")
        
        st.success("Data loaded and cleaned successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading or cleaning data: {e}. Please ensure it's a valid XLSX file.")
        return None

def create_sequences(data, sequence_length):
    """Creates input-output sequences for the deep learning model."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner="Loading Keras Model...")
def load_keras_model_from_file(uploaded_file_obj, model_name_for_log="Custom Model"):
    """
    Loads a Keras model from an uploaded .h5 file.
    Uses st.cache_resource to avoid re-loading the model repeatedly.
    """
    temp_model_path = f"temp_{uuid.uuid4()}.h5" # Use unique temp name
    try:
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        model = tf.keras.models.load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded '{model_name_for_log}' successfully. Input sequence length: {sequence_length}")
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading Keras model '{model_name_for_log}': {e}. Ensure it's a valid .h5 TensorFlow Keras model.")
        return None, None
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@st.cache_resource(show_spinner="Loading Standard Pre-trained Model...")
def load_standard_model_cached(path):
    """
    Loads the standard pre-trained Keras model.
    Uses st.cache_resource for performance.
    """
    try:
        model = tf.keras.models.load_model(path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Standard model loaded (Sequence Length: {sequence_length}).")
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading standard Keras model from {path}: {e}. "
                 "Please ensure 'standard_model.h5' is in the same directory as app.py.")
        return None, None

def build_deep_learning_model(sequence_length, n_features=1):
    """Builds a simple Deep Learning (LSTM) model architecture."""
    model = Sequential([
        LSTM(units=50, activation="relu", input_shape=(sequence_length, n_features), return_sequences=False),
        Dropout(0.3), # Reduced dropout for potentially better training stability
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    """
    Performs multi-step forecasting with Monte Carlo Dropout for uncertainty estimation.
    Applies a minimum uncertainty percentage to prevent overly narrow CIs.
    """
    all_predictions_scaled = []
    
    # Use tf.function for faster graph execution
    @tf.function
    def predict_step_training_true(inp):
        return model(inp, training=True) # Important: inference in training mode for dropout

    progress_bar = st.progress(0, text=f"Running Monte Carlo Dropout (0/{n_iterations} iterations)...")
    
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        # Clone sequence for each iteration to avoid contamination
        temp_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1) 
        
        for _ in range(n_steps):
            # Predict next step
            next_pred_scaled = predict_step_training_true(temp_sequence).numpy()[0, 0]
            iteration_predictions_scaled.append(next_pred_scaled)
            
            # Update sequence for next prediction
            new_step = np.array([[next_pred_scaled]]).reshape(1, 1, 1)
            temp_sequence = np.append(temp_sequence[:, 1:, :], new_step, axis=1) # Shift window
            
        all_predictions_scaled.append(iteration_predictions_scaled)
        progress_bar.progress((i + 1) / n_iterations, text=f"Running Monte Carlo Dropout ({i+1}/{n_iterations} iterations)...")
        
    progress_bar.empty() # Clear progress bar once done

    predictions_array_scaled = np.array(all_predictions_scaled) # Shape: (n_iterations, n_steps)
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    
    # Inverse transform mean predictions
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    
    # Calculate confidence intervals (95% CI -> 1.96 standard deviations)
    ci_multiplier = 1.96
    lower_bound_scaled = mean_preds_scaled - ci_multiplier * std_devs_scaled
    upper_bound_scaled = mean_preds_scaled + ci_multiplier * std_devs_scaled
    
    # Inverse transform confidence intervals
    lower_bound = scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()
    
    # Apply minimum uncertainty percentage to avoid very narrow CIs, especially for stable forecasts
    min_uncertainty_percent = 0.02 # Adjusted to 2%
    for i in range(len(mean_preds)):
        if abs(mean_preds[i]) > 1e-6: # Avoid division by zero if mean_preds is near zero
            min_uncertainty_value = abs(mean_preds[i] * min_uncertainty_percent / 2.0)
            current_half_range = (upper_bound[i] - lower_bound[i]) / 2.0
            if current_half_range < min_uncertainty_value:
                lower_bound[i] = mean_preds[i] - min_uncertainty_value
                upper_bound[i] = mean_preds[i] + min_uncertainty_value
        else:
            # For predictions very close to zero, apply a fixed small absolute uncertainty
            abs_uncertainty = 0.01 
            lower_bound[i] = mean_preds[i] - abs_uncertainty
            upper_bound[i] = mean_preds[i] + abs_uncertainty
             
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, and MAPE between true and predicted values."""
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    
    # Handle NaN values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    
    y_true_clean, y_pred_clean = y_true[mask], y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # Calculate MAPE, handling zero true values
    mape_mask = (y_true_clean != 0)
    mape = np.nan
    if np.any(mape_mask):
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
        
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions ---
def create_forecast_plot(historical_df, forecast_df):
    """Generates an interactive Plotly graph of historical data and forecast with CIs."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", 
                             line=dict(color="#3498db", width=2))) # Primary blue
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="AI Forecast", 
                             line=dict(color="#e67e22", width=2, dash='dash'))) # Orange for forecast
    
    # Add confidence interval shading
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", 
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", 
                             line=dict(width=0), fillcolor="rgba(230, 126, 34, 0.2)", fill="tonexty", showlegend=True))
    
    fig.update_layout(
        title="<b>Groundwater Level: Historical Data & AI Forecast</b>",
        xaxis_title="Date",
        yaxis_title="Groundwater Level",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
        font=dict(family="Inter", size=12),
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    return fig

def create_loss_plot(history_dict):
    """Generates an interactive Plotly graph of training and validation loss."""
    if not history_dict or not isinstance(history_dict, dict) or "loss" not in history_dict or "val_loss" not in history_dict:
        fig = go.Figure().update_layout(
            title="<b>Model Training & Validation Loss</b>",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=400,
            annotations=[dict(text="Training history unavailable.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))]
        )
        return fig

    history_df = pd.DataFrame(history_dict)
    history_df["Epoch"] = history_df.index + 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss", line=dict(color="#27ae60"))) # Green
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss", line=dict(color="#e74c3c"))) # Red
    
    fig.update_layout(
        title="<b>Model Training & Validation Loss</b>",
        xaxis_title="Epoch",
        yaxis_title="Loss (Mean Squared Error)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
        font=dict(family="Inter", size=12),
        title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    return fig

def create_generic_data_plot(df, x_col, y_col, color_col, size_col, chart_type):
    """Generates a dynamic Plotly Express chart for the generic data analyzer."""
    if not df.empty and x_col and y_col:
        try:
            if chart_type == "Line Plot":
                fig = px.line(df, x=x_col, y=y_col, color=color_col, markers=True, title=f"<b>Line Plot of {y_col} vs {x_col}</b>")
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, hover_data=df.columns, title=f"<b>Scatter Plot of {y_col} vs {x_col}</b>")
            elif chart_type == "Bar Chart":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"<b>Bar Chart of {y_col} by {x_col}</b>")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_col, title=f"<b>Histogram of {x_col}</b>")
            elif chart_type == "Box Plot":
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"<b>Box Plot of {y_col} by {x_col}</b>")
            else:
                st.warning("Selected chart type not supported.")
                return go.Figure() # Empty figure
            
            fig.update_layout(
                template="plotly_white",
                font=dict(family="Inter", size=12),
                title_font_size=18,
                margin=dict(l=40, r=40, t=60, b=40),
                height=500
            )
            return fig
        except Exception as e:
            st.error(f"Error generating chart: {e}. Check column selections and data types.")
            return go.Figure() # Return empty figure on error
    return go.Figure() # Return empty figure if no data or columns selected

# --- Gemini API Functions ---
def generate_gemini_report(hist_df, forecast_df, metrics, language):
    """
    Generates a scientific report using Gemini API, focusing on hydrogeological insights.
    Censors specific technical AI terms.
    """
    if not gemini_configured: return "AI report disabled. Gemini API Key is not configured."
    if hist_df is None or forecast_df is None or metrics is None:
        return "Error: Insufficient data (historical, forecast, or metrics) for AI report generation."
    
    prompt = f"""Act as a highly experienced professional hydrologist and data scientist. Provide a concise, clear, and actionable report in {language} based on the provided historical groundwater data, future forecasts, and model performance metrics. Your report should focus on:
    1.  **Executive Summary:** A brief overview of current conditions and future outlook.
    2.  **Historical Data Insights:** Analyze trends, seasonality, and anomalies in the historical groundwater levels.
    3.  **AI Forecast Interpretation:** Explain the predicted future levels, the significance of the confidence intervals (C.I.), and what the uncertainty implies. Discuss potential shifts or continuations of trends.
    4.  **Recommendations/Implications:** Based on the forecast, suggest practical implications for water resource management, environmental planning, or risk assessment.

    **IMPORTANT:**
    - Do NOT discuss the internal technical details of the AI model's architecture, training process, or specific deep learning terms (e.g., LSTM, epochs, layers, dropout, optimizers, sequence length). Use general terms like "AI model" or "deep learning approach."
    - Ensure the language is professional, unbiased, and suitable for a stakeholder presentation.

    ### Historical Data Summary:
    {hist_df["Level"].describe().to_string()}

    ### AI Forecast Summary:
    {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}

    ### Model Accuracy Metrics (Validation/Pseudo-Validation):
    - Root Mean Squared Error (RMSE): {metrics.get('RMSE', 'N/A'):.4f}
    - Mean Absolute Error (MAE): {metrics.get('MAE', 'N/A'):.4f}
    - Mean Absolute Percentage Error (MAPE): {metrics.get('MAPE', 'N/A'):.2f}%

    Please generate the report now.
    """
    try:
        response = gemini_model_report.generate_content(prompt)
        # Post-process to ensure no forbidden terms accidentally slip through (double check)
        forbidden_terms = ["lstm", "long short-term memory", "epoch", "layer", "dropout", "adam optimizer", "sequence length", "relu activation"]
        cleaned_text = response.text
        for term in forbidden_terms:
            cleaned_text = cleaned_text.replace(term, "[deep learning technique]") # Replaced with a generic term
        return cleaned_text
    except Exception as e:
        st.error(f"Error generating AI report: {e}")
        return f"Error: Could not generate AI report due to a technical issue."

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    """
    Provides a conversational AI response based on the provided context.
    Censors specific technical AI terms.
    """
    if not gemini_configured: return "AI chat disabled. Gemini API Key is not configured."
    if hist_df is None or forecast_df is None or metrics is None:
        return "Error: Insufficient context (historical data, forecast, or metrics) for AI chat."
    
    # Construct comprehensive context for the AI
    context_parts = [
        "You are a highly experienced AI and hydrogeology expert with over 20 years of experience in groundwater analysis and predictive modeling.",
        "Your role is to interpret the provided groundwater data, forecasts, and reports, and answer user questions in a professional, engineering-style manner.",
        "**IMPORTANT:** Do NOT discuss internal AI model mechanics, architecture, or specific deep learning terms (e.g., LSTM, epochs, layers). Focus only on data interpretation, patterns, trends, uncertainties, and implications for groundwater behavior.",
        "Be concise, clear, and actionable.",
        "",
        "### Historical Groundwater Data Summary:",
        hist_df["Level"].describe().to_string(),
        "",
        "### AI Forecast Results Summary:",
        forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string(),
        "",
        f"### Forecast Accuracy Metrics:\nRMSE = {metrics.get('RMSE', 'N/A'):.4f}\nMAE = {metrics.get('MAE', 'N/A'):.4f}\nMAPE = {metrics.get('MAPE', 'N/A'):.2f}%",
        "",
        "### Previously Generated AI Report (for reference, if available):",
        ai_report if ai_report else "(No prior AI report has been generated.)",
        "",
        "### Current Conversation History (for context):"
    ]
    
    # Add recent chat history for continuity (last 6 turns to manage token limits)
    for sender, message in chat_hist[-6:]: # Include a few previous turns
        context_parts.append(f"{sender}: {message}")
    
    context_parts.append(f"User: {user_query}")
    context_parts.append("AI:") # Prompt the AI to generate its response

    context_prompt = "\n".join(context_parts)

    try:
        response = gemini_model_chat.generate_content(context_prompt)
        # Post-process to ensure no forbidden terms accidentally slip through
        forbidden_terms = ["lstm", "long short-term memory", "epoch", "layer", "dropout", "adam optimizer", "sequence length", "relu activation"]
        cleaned_text = response.text
        for term in forbidden_terms:
            cleaned_text = cleaned_text.replace(term, "[deep learning technique]")
        return cleaned_text
    except Exception as e:
        st.error(f"Error in AI chat: {e}")
        return f"Error: Could not process your request due to a technical issue."

# --- Main Forecasting Pipeline ---
def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                          sequence_length_train_param, epochs_train_param, 
                          mc_iterations_param, use_custom_scaler_params_flag, 
                          custom_scaler_min_param, custom_scaler_max_param):
    """
    Executes the entire forecasting pipeline: model loading/training, data scaling,
    sequence creation, forecasting with uncertainty, and results compilation.
    """
    st.info(f"Initiating forecast pipeline using: {model_choice}...")
    
    model = None
    history_data = None
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    model_sequence_length = sequence_length_train_param

    try:
        # Step 1: Prepare Model (Load or Build)
        st.subheader("Step 1: Preparing Deep Learning Model")
        if model_choice == "Standard Pre-trained Model":
            if os.path.exists(STANDARD_MODEL_PATH):
                model, model_sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                if model is None: return None, None, None, None # Error occurred during loading
            else:
                st.error(f"Standard model file not found at: '{STANDARD_MODEL_PATH}'. "
                         "Please ensure it's in the application directory. Cannot proceed.")
                return None, None, None, None
        elif model_choice == "Upload Custom .h5 Model":
            if custom_model_file_obj:
                model, model_sequence_length = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
                if model is None: return None, None, None, None
            else:
                st.warning("No custom .h5 model uploaded. Please upload a model file.")
                return None, None, None, None
        elif model_choice == "Train New Model":
            # Model will be built later
            st.info(f"New Deep Learning model will be trained with sequence length: {model_sequence_length}, epochs: {epochs_train_param}.")
        
        st.session_state.model_sequence_length = model_sequence_length

        # Step 2: Preprocess Data (Scaling)
        st.subheader("Step 2: Scaling Data")
        if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
            scaler_obj.fit(np.array([[custom_scaler_min_param], [custom_scaler_max_param]]))
            scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
            st.info(f"Using custom scaler based on provided min={custom_scaler_min_param:.4f}, max={custom_scaler_max_param:.4f}.")
        else:
            scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
            st.info(f"Data scaled using automatic min-max scaling. (Min: {scaler_obj.data_min_[0]:.4f}, Max: {scaler_obj.data_max_[0]:.4f})")
        
        # Step 3: Create Sequences
        st.subheader("Step 3: Generating Input Sequences")
        if len(df) <= model_sequence_length:
            st.error(f"Not enough data points ({len(df)}) to create sequences with length {model_sequence_length}. Please use a shorter sequence length or provide more data.")
            return None, None, None, None
        
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0:
            st.error("Could not create any input-output sequences from the data. This might be due to insufficient data or incorrect sequence length.")
            return None, None, None, None
        st.info(f"Successfully created {len(X)} input sequences for the model.")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

        # Step 4: Model Training or Pseudo-Evaluation
        if model_choice == "Train New Model":
            st.subheader("Step 4: Training New Deep Learning Model")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            if len(X_train) == 0 or len(X_val) == 0:
                st.error("Not enough data to create valid training and validation sets. Reduce test size or provide more data.")
                return None, None, None, None
            
            model = build_deep_learning_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True) # Increased patience
            
            # Custom callback for Streamlit progress bar
            class StreamlitProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, total_epochs, progress_bar_placeholder, status_text_placeholder):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.progress_bar = progress_bar_placeholder
                    self.status_text = status_text_placeholder
                    self.epoch_loss = []
                    self.epoch_val_loss = []

                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.total_epochs
                    self.progress_bar.progress(progress)
                    loss = logs.get('loss', 'N/A')
                    val_loss = logs.get('val_loss', 'N/A')
                    self.epoch_loss.append(loss)
                    self.epoch_val_loss.append(val_loss)
                    self.status_text.text(f"Training Epoch {epoch+1}/{self.total_epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            
            progress_bar_placeholder = st.progress(0, text="Training progress...")
            status_text_placeholder = st.empty()

            history_obj = model.fit(
                X_train, y_train, 
                epochs=epochs_train_param, 
                batch_size=32, 
                validation_data=(X_val, y_val), 
                callbacks=[early_stopping, StreamlitProgressCallback(epochs_train_param, progress_bar_placeholder, status_text_placeholder)], 
                verbose=0 # Suppress verbose output from Keras to keep Streamlit clean
            )
            history_data = history_obj.history
            progress_bar_placeholder.empty()
            status_text_placeholder.empty()
            st.success("Deep Learning model training complete.")
            
            st.info("Evaluating newly trained model on validation set...")
            val_predictions_scaled = model.predict(X_val)
            val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
            y_val_actual = scaler_obj.inverse_transform(y_val)
            evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
            st.success("Model evaluation complete.")
        else: # For pre-trained or custom models, perform a pseudo-validation
            st.subheader("Step 4: Evaluating Model (Pseudo-Validation)")
            if len(X) > DEFAULT_MODEL_SEQUENCE_LENGTH: # Need enough data for a meaningful split
                # Use a small portion of the end of the data for pseudo-validation
                val_split_idx = max(model_sequence_length, int(len(X) * 0.9)) # Use last 10%
                X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                
                if len(X_val_pseudo) > 0:
                    st.info(f"Running pseudo-validation on the last {len(X_val_pseudo)} data points.")
                    val_predictions_scaled = model.predict(X_val_pseudo)
                    val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo)
                    evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation complete.")
                else:
                    st.warning("Not enough sequences available for meaningful pseudo-validation.")
            else:
                st.warning(f"Insufficient data ({len(X)} sequences) for pseudo-validation. Skipping evaluation.")

        # Step 5: Generate Forecast
        st.subheader(f"Step 5: Generating {forecast_horizon}-Step AI Forecast")
        if model is None:
            st.error("AI Model not loaded or trained. Cannot proceed with forecasting.")
            return None, None, None, None

        last_sequence_scaled_for_pred = scaled_data[-model_sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_dropout_uncertainty(
            model, last_sequence_scaled_for_pred, forecast_horizon, mc_iterations_param, scaler_obj, model_sequence_length
        )
        st.success("AI Forecasting complete with uncertainty quantification.")

        # Step 6: Compile Forecast Results
        st.subheader("Step 6: Compiling Results")
        last_date = df["Date"].iloc[-1]
        
        # Try to infer frequency for accurate future dates
        try:
            freq = pd.infer_freq(df["Date"].dropna())
            if freq is None: raise ValueError("Could not infer frequency.")
        except Exception:
            st.warning("Could not automatically infer data frequency. Defaulting to daily ('D').")
            freq = "D"
            
        try:
            date_offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError: 
            st.warning(f"Invalid frequency '{freq}'. Defaulting to daily ('D').")
            date_offset = pd.DateOffset(days=1); freq = 'D' # Fallback to daily if offset conversion fails

        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=freq)
        forecast_df = pd.DataFrame({
            "Date": forecast_dates, 
            "Forecast": mean_forecast, 
            "Lower_CI": lower_bound, 
            "Upper_CI": upper_bound
        })
        
        st.success("Forecast pipeline completed successfully!")
        return forecast_df, evaluation_metrics, history_data, scaler_obj

    except Exception as e:
        st.error(f"An unexpected error occurred during the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc()) # Show full traceback for debugging
        return None, None, None, None

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes all necessary session state variables."""
    defaults = {
        "cleaned_data": None,
        "forecast_results": None,
        "evaluation_metrics": None,
        "training_history": None,
        "ai_report": None,
        "scaler_object": None,
        "forecast_plot_fig": None,
        "uploaded_data_filename": None,
        "active_tab": "Data Preview", # Start on Data Preview tab
        "report_language": "English",
        "chat_history": [],
        "chat_active": False,
        "model_sequence_length": DEFAULT_MODEL_SEQUENCE_LENGTH,
        "run_forecast_triggered": False,
        "generic_uploaded_data": None, # For new generic data analyzer
        "session_activity_log": [], # For user-facing session activity
        "standard_model_loaded_len": DEFAULT_MODEL_SEQUENCE_LENGTH # To store loaded standard model's seq length
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state on app load
initialize_session_state()

# Load standard model's sequence length on startup if available
# This needs to run once per session, outside of the main flow
if 'standard_model_loaded_len' not in st.session_state or st.session_state.standard_model_loaded_len == DEFAULT_MODEL_SEQUENCE_LENGTH:
    if os.path.exists(STANDARD_MODEL_PATH):
        try:
            _temp_model = tf.keras.models.load_model(STANDARD_MODEL_PATH, compile=False)
            st.session_state.standard_model_loaded_len = _temp_model.input_shape[1]
            del _temp_model # Clean up
        except Exception:
            pass # Keep default if loading fails

# --- Session Activity Logging ---
def log_session_activity(activity_type, details=""):
    """Logs an activity to the current session's activity log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.session_activity_log.append({
        "timestamp": timestamp,
        "activity_type": activity_type,
        "details": details
    })

# --- Sidebar Content ---
with st.sidebar:
    st.title("DeepHydro AI")
    st.markdown("---")

    st.header("1. Data Management")
    uploaded_data_file = st.file_uploader("Upload Groundwater Data (XLSX)", type="xlsx", key="gw_data_uploader")
    if uploaded_data_file:
        if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
            st.session_state.uploaded_data_filename = uploaded_data_file.name
            with st.spinner("Loading and cleaning groundwater data..."):
                cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
            if cleaned_df_result is not None:
                st.session_state.cleaned_data = cleaned_df_result
                # Reset previous forecast/report results if new data is loaded
                st.session_state.forecast_results = None
                st.session_state.evaluation_metrics = None
                st.session_state.training_history = None
                st.session_state.ai_report = None
                st.session_state.chat_history = []
                st.session_state.scaler_object = None
                st.session_state.forecast_plot_fig = None
                st.session_state.run_forecast_triggered = False
                st.session_state.active_tab = "Data Preview" # Switch to Data Preview tab
                log_session_activity("Groundwater Data Upload", f"File: {uploaded_data_file.name}, Rows: {len(cleaned_df_result)}")
                st.rerun() # Rerun to update main content area
            else:
                st.session_state.cleaned_data = None
                st.error("Data loading failed. Please check your file and try again.")
                log_session_activity("Groundwater Data Upload Failed", f"File: {uploaded_data_file.name}")

    st.header("2. AI Model & Forecast Settings")
    model_choice = st.selectbox(
        "Choose AI Model Type", 
        ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), 
        key="model_select"
    )
    
    custom_model_file_obj_sidebar = None
    custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
    use_custom_scaler_sidebar = False

    # Adjust default sequence length based on loaded standard model if possible
    current_default_seq_len = st.session_state.get("standard_model_loaded_len", DEFAULT_MODEL_SEQUENCE_LENGTH)

    sequence_length_train_sidebar = st.session_state.get("model_sequence_length", current_default_seq_len)
    epochs_train_sidebar = 50

    if model_choice == "Upload Custom .h5 Model":
        custom_model_file_obj_sidebar = st.file_uploader("Upload your .h5 AI Model", type="h5", key="custom_h5_uploader")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler min/max?", value=False, key="use_custom_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values used for training your model:")
            custom_scaler_min_sidebar = st.number_input("Original Min Value", value=0.0, format="%.4f", key="custom_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max Value", value=1.0, format="%.4f", key="custom_scaler_max_in")
    elif model_choice == "Standard Pre-trained Model":
        st.info(f"Using pre-trained Deep Learning model (Sequence Length: {current_default_seq_len}).")
        use_custom_scaler_sidebar = st.checkbox("Provide custom scaler min/max?", value=False, key="use_std_scaler_cb")
        if use_custom_scaler_sidebar:
            st.markdown("Enter **original min/max** values for scaling:")
            custom_scaler_min_sidebar = st.number_input("Original Min Value", value=0.0, format="%.4f", key="std_scaler_min_in")
            custom_scaler_max_sidebar = st.number_input("Original Max Value", value=1.0, format="%.4f", key="std_scaler_max_in")
    elif model_choice == "Train New Model":
        st.markdown("Configure settings for a new Deep Learning model:")
        try:
            sequence_length_train_sidebar = st.number_input("Input Sequence Length", min_value=10, max_value=365, value=current_default_seq_len, step=5, key="seq_len_train_in", help="Number of past data points to consider for each prediction.")
        except Exception:
            sequence_length_train_sidebar = current_default_seq_len
        epochs_train_sidebar = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in", help="Number of training iterations. More epochs can lead to better learning but also overfitting.")
    
    mc_iterations_sidebar = st.number_input("MC Dropout Iterations (for C.I.)", min_value=50, max_value=500, value=100, step=10, key="mc_iter_in", help="Number of Monte Carlo simulations for calculating confidence intervals. Higher values increase accuracy but also computation time.")
    forecast_horizon_sidebar = st.number_input("Forecast Horizon (Future Steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in", help="Number of future time steps to forecast.")

    # Run Forecast Button
    run_forecast_button = st.button("Run AI Forecast", key="run_forecast_main_btn", use_container_width=True)
    if run_forecast_button:
        if st.session_state.cleaned_data is not None:
            if model_choice == "Upload Custom .h5 Model" and not custom_model_file_obj_sidebar:
                st.error("Please upload a custom .h5 model file to proceed with 'Upload Custom .h5 Model' choice.")
                st.session_state.run_forecast_triggered = False
            else:
                st.session_state.run_forecast_triggered = True
                with st.spinner(f"Running forecast pipeline with {model_choice}... This may take a moment."):
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
                    st.success("AI Forecast complete! Results are ready for review.")
                    st.session_state.ai_report = None; # Clear previous report
                    st.session_state.chat_history = []; st.session_state.chat_active = False # Reset chat
                    st.session_state.active_tab = "Forecast Results" # Switch to results tab
                    log_session_activity("AI Forecast Run", f"Model: {model_choice}, Horizon: {forecast_horizon_sidebar}, Metrics: {json.dumps({k: f'{v:.2f}' for k, v in metrics.items() if not np.isnan(v)})}")
                    st.rerun() # Rerun to update main content area
                else:
                    st.error("AI Forecast pipeline failed. Please check the error messages above.")
                    st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                    st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
                    log_session_activity("AI Forecast Failed", f"Model: {model_choice}, Error: See logs")
        else:
            st.error("Please upload groundwater data first (Step 1) to run a forecast.")

    st.header("3. AI Analysis & Reporting")
    st.session_state.report_language = st.selectbox("AI Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_configured)
    
    # Generate AI Report Button
    generate_report_button = st.button("Generate AI Report", key="show_report_btn", disabled=not gemini_configured or st.session_state.forecast_results is None, use_container_width=True)
    if generate_report_button:
        if not gemini_configured:
            st.error("AI Report generation is disabled. Gemini API Key is not configured.")
        elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            with st.spinner(f"Generating comprehensive AI report in {st.session_state.report_language}..."):
                st.session_state.ai_report = generate_gemini_report(
                    st.session_state.cleaned_data, st.session_state.forecast_results,
                    st.session_state.evaluation_metrics, st.session_state.report_language
                )
            if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:"):
                st.success("AI Report successfully generated.")
                st.session_state.active_tab = "AI Report" # Switch to AI Report tab
                log_session_activity("AI Report Generated", f"Language: {st.session_state.report_language}")
                st.rerun()
            else:
                st.error(f"Failed to generate AI Report. {st.session_state.ai_report}")
                log_session_activity("AI Report Generation Failed", f"Language: {st.session_state.report_language}, Error: See logs")
        else:
            st.error("Data, forecast results, and evaluation metrics are required to generate an AI report. Please run a forecast first.")

    # Download PDF Button
    if st.button("Download Full Report (PDF)", key="download_report_btn", disabled=st.session_state.forecast_results is None or st.session_state.ai_report is None, use_container_width=True):
        if st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None and st.session_state.ai_report is not None and st.session_state.forecast_plot_fig is not None:
            with st.spinner("Compiling PDF report... This may take a moment."):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Try to add a Unicode font for broader language support
                    report_font = "Helvetica" # Default fallback
                    # Path for a common DejaVuSans font in Linux environments (often available on Render.com)
                    font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
                    if os.path.exists(font_path_dejavu):
                        try:
                            pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True)
                            report_font = "DejaVu"
                        except RuntimeError as font_err:
                            st.warning(f"Failed to load DejaVu font ({font_err}), using Helvetica.")
                    else:
                        st.warning(f"DejaVu font not found at {font_path_dejavu}, using Helvetica. Some special characters may not render.")

                    pdf.set_font(report_font, size=18)
                    pdf.cell(0, 15, txt="DeepHydro AI Groundwater Forecasting Report", new_x="LMARGIN", new_y="NEXT", align="C")
                    pdf.ln(10)

                    # --- Embed Plot ---
                    plot_filename = "forecast_plot.png"
                    try:
                        # Save Plotly figure as a static image
                        st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2, width=1000, height=500)
                        pdf.image(plot_filename, x=10, y=pdf.get_y(), w=190) # Adjust width (w) as needed
                        pdf.ln(130) # Move cursor down after image
                    except Exception as img_err:
                        st.warning(f"Could not embed forecast plot image in PDF: {img_err}. Skipping plot.")
                        pdf.set_font(report_font, "I", size=10)
                        pdf.cell(0, 10, txt="[Forecast plot unavailable due to embedding error]", new_x="LMARGIN", new_y="NEXT")
                        pdf.ln(5)
                    finally:
                        if os.path.exists(plot_filename):
                            os.remove(plot_filename) # Clean up temporary image file

                    # --- Model Evaluation Metrics ---
                    pdf.set_font(report_font, "B", size=14)
                    pdf.cell(0, 10, txt="Model Evaluation Metrics (Validation/Pseudo-Validation)", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                    pdf.set_font(report_font, size=11)
                    for key, value in st.session_state.evaluation_metrics.items():
                        val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else "N/A"
                        pdf.cell(0, 8, txt=f"- {key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(8)

                    # --- Forecast Data Table ---
                    pdf.set_font(report_font, "B", size=14)
                    pdf.cell(0, 10, txt="AI Forecast Data (First 10 Rows)", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                    pdf.set_font(report_font, size=9)
                    
                    # Table Headers
                    col_widths = [35, 35, 35, 35] # Date, Forecast, Lower CI, Upper CI
                    pdf.set_fill_color(220, 220, 220) # Light gray for headers
                    pdf.cell(col_widths[0], 8, txt="Date", border=1, fill=True)
                    pdf.cell(col_widths[1], 8, txt="AI Forecast", border=1, fill=True)
                    pdf.cell(col_widths[2], 8, txt="Lower CI", border=1, fill=True)
                    pdf.cell(col_widths[3], 8, txt="Upper CI", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
                    
                    # Table Rows
                    for _, row in st.session_state.forecast_results.head(10).iterrows():
                        pdf.cell(col_widths[0], 7, txt=str(row["Date"].date()), border=1)
                        pdf.cell(col_widths[1], 7, txt=f"{row['Forecast']:.2f}", border=1)
                        pdf.cell(col_widths[2], 7, txt=f"{row['Lower_CI']:.2f}", border=1)
                        pdf.cell(col_widths[3], 7, txt=f"{row['Upper_CI']:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(8)

                    # --- AI Report ---
                    pdf.set_font(report_font, "B", size=14)
                    pdf.cell(0, 10, txt=f"AI-Generated Scientific Report ({st.session_state.report_language})", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                    pdf.set_font(report_font, size=11)
                    # For multi_cell, ensure text is UTF-8 encoded
                    pdf.multi_cell(0, 6, txt=st.session_state.ai_report.encode('latin-1', 'replace').decode('latin-1'))
                    pdf.ln(10)
                    
                    pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_output_bytes,
                        file_name="DeepHydro_AI_Forecast_Report.pdf",
                        mime="application/pdf",
                        key="pdf_download_final_btn",
                        use_container_width=True
                    )
                    st.success("PDF report generated successfully! Click the button above to download.")
                    log_session_activity("PDF Report Downloaded", "Full report with forecast and AI analysis.")
                except Exception as pdf_err:
                    st.error(f"Failed to generate PDF report: {pdf_err}")
                    log_session_activity("PDF Report Generation Failed", f"Error: {pdf_err}")
                    import traceback; st.error(traceback.format_exc()) # For debugging
        else:
            st.error("To download the PDF report, please first upload data, run a forecast, and generate an AI report.")

    st.header("4. AI Assistant")
    # Activate Chat Button
    chat_button_label = "Deactivate AI Chat" if st.session_state.chat_active else "Activate AI Chat"
    activate_chat_button = st.button(chat_button_label, key="chat_ai_btn", disabled=not gemini_configured or st.session_state.forecast_results is None, use_container_width=True)
    if activate_chat_button:
        if st.session_state.chat_active:
            st.session_state.chat_active = False # Deactivate chat
            st.session_state.chat_history = [] # Clear chat history
            st.info("AI Chat deactivated.")
            log_session_activity("AI Chat Deactivated", "Chat history cleared.")
            st.rerun() # Rerun to update button label and chat area
        else:
            st.session_state.chat_active = True # Activate chat
            st.session_state.active_tab = "AI Chatbot" # Switch to AI Chatbot tab
            st.success("AI Chat activated! You can now ask questions about your data and forecast.")
            log_session_activity("AI Chat Activated", "Ready for interaction.")
            st.rerun() # Rerun to update button label and chat area

    # About Us Section
    st.markdown('<div class="about-us-header">👥 About DeepHydro AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="about-us-content">', unsafe_allow_html=True)
    st.markdown("""
        DeepHydro AI specializes in advanced hydrological forecasting solutions using cutting-edge deep learning models.
        Our mission is to empower water resource managers, environmental engineers, and researchers with accurate, interpretable,
        and reliable predictions for critical water parameters like groundwater levels.
        
        **Our Expertise:**
        * Deep Learning for Time Series Analysis
        * Uncertainty Quantification (Monte Carlo Dropout)
        * Hydrogeological Modeling & Interpretation
        * Intuitive Data Visualization & Reporting
        
        **Contact Us:** [info@deephydro.ai](mailto:info@deephydro.ai)
        
        © 2025 DeepHydro AI. All rights reserved.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Application Area ---
st.title("DeepHydro AI: Advanced Groundwater Forecasting")

# App Introduction Card
st.markdown("""
    <div class="app-intro">
        <h3>Welcome to DeepHydro AI Forecasting Platform!</h3>
        <p>
            This powerful application enables you to analyze historical groundwater level data,
            generate accurate forecasts with uncertainty estimates using advanced deep learning models,
            and gain AI-powered insights.
            Upload your data, configure your model, and let our AI assist you in understanding
            future groundwater conditions.
        </p>
    </div>
""", unsafe_allow_html=True)


# Define tabs for main content
tab_titles = ["Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot", "Generic Data Analyzer", "Session Activity Log"]
tabs = st.tabs(tab_titles)

# Programmatic Tab Switching (if a button triggers a tab change)
active_tab_index = tab_titles.index(st.session_state.get("active_tab", "Data Preview"))

# --- Tab Content ---

# Data Preview Tab
with tabs[0]: # Data Preview
    st.header("Historical Groundwater Data Overview")
    if st.session_state.cleaned_data is not None:
        st.subheader("Raw Data Table (First 100 Rows)")
        st.dataframe(st.session_state.cleaned_data.head(100), use_container_width=True) # Limit to 100 for large files
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Data Points", len(st.session_state.cleaned_data))
        with col2:
            min_date = st.session_state.cleaned_data['Date'].min()
            max_date = st.session_state.cleaned_data['Date'].max()
            st.metric("Time Range", f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        st.subheader("Interactive Plot of Historical Levels")
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], 
                                      mode="lines", name="Groundwater Level", line=dict(color="#3498db")))
        fig_data.update_layout(
            title="<b>Historical Groundwater Levels</b>",
            xaxis_title="Date", 
            yaxis_title="Groundwater Level", 
            template="plotly_white", 
            margin=dict(l=40, r=40, t=60, b=40), 
            height=500,
            font=dict(family="Inter", size=12),
            title_font_size=18
        )
        st.plotly_chart(fig_data, use_container_width=True)
    else:
        st.info("⬆️ Please upload your groundwater level data (XLSX format) using the 'Data Management' section in the sidebar to begin.")

# Forecast Results Tab
with tabs[1]: # Forecast Results
    st.header("AI Forecast Results & Visualization")
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        if st.session_state.forecast_plot_fig:
            st.subheader("Interactive AI Forecast Plot")
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else:
            st.warning("Forecast plot is not available. This might happen if the forecast pipeline encountered an issue.")
        
        st.subheader("Detailed AI Forecast Data Table")
        st.dataframe(st.session_state.forecast_results, use_container_width=True)
    elif st.session_state.run_forecast_triggered:
        st.warning("The forecast pipeline was initiated, but no results are available. Please check for any error messages in the sidebar or above.")
    else:
        st.info("To view AI Forecast Results, please upload your data and click 'Run AI Forecast' in the sidebar.")

# Model Evaluation Tab
with tabs[2]: # Model Evaluation
    st.header("Deep Learning Model Performance Evaluation")
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (Validation/Pseudo-Validation)")
        col1, col2, col3 = st.columns(3)
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan)
        mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan)
        mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        
        col1.metric("Root Mean Squared Error (RMSE)", f"{rmse_val:.4f}" if not np.isnan(rmse_val) else "N/A")
        col2.metric("Mean Absolute Error (MAE)", f"{mae_val:.4f}" if not np.isnan(mae_val) else "N/A")
        col3.metric("Mean Absolute Percentage Error (MAPE)", f"{mape_val:.2f}%" if not np.isnan(mape_val) and mape_val != np.inf else ("N/A" if np.isnan(mape_val) else "Inf"))
        
        st.markdown("""
            <p style="font-size:0.9em; color:gray;">
                <b>RMSE</b>: Measures the average magnitude of the errors. Lower is better.<br>
                <b>MAE</b>: Measures the average magnitude of the errors without considering direction. Less sensitive to outliers than RMSE.<br>
                <b>MAPE</b>: Expresses error as a percentage of the actual value. Sensitive to zero or near-zero actual values.
            </p>
        """, unsafe_allow_html=True)

        st.subheader("Training and Validation Loss (for newly trained models)")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
            st.info("This plot shows how the model's performance on training and unseen validation data improved over epochs.")
        else:
            st.info("No training history available. This typically applies to pre-trained or custom uploaded models, or if training encountered an error.")
    elif st.session_state.run_forecast_triggered:
        st.warning("The forecast pipeline was initiated, but no evaluation metrics are available.")
    else:
        st.info("Run an AI forecast (sidebar) to see the model's performance evaluation.")

# AI Report Tab
with tabs[3]: # AI Report
    st.header("AI-Generated Scientific Report")
    if not gemini_configured:
        st.warning("AI features are currently disabled. Please ensure the 'GOOGLE_API_KEY' environment variable is correctly set.")
    if st.session_state.ai_report:
        st.markdown(f"""
            <div class="chat-message ai-message">
                {st.session_state.ai_report}
                <span class="copy-tooltip">Copied!</span>
            </div>
        """, unsafe_allow_html=True)
        st.info("Click on the report to copy its content to your clipboard.")
    else:
        st.info("Click 'Generate AI Report' in the sidebar after running a forecast to get an AI-powered scientific interpretation.")

# AI Chatbot Tab
with tabs[4]: # AI Chatbot
    st.header("DeepHydro AI Assistant Chatbot")
    if not gemini_configured:
        st.warning("AI Chatbot is currently disabled. Please ensure the 'GOOGLE_API_KEY' environment variable is correctly set.")
    elif st.session_state.chat_active:
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("AI Chatbot activated. You can now ask questions about your historical data, the forecast, and the model evaluation results.")
            
            chat_container = st.container(height=500, border=True) # Fixed height for chat area
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown('<p style="text-align: center; color: gray;">Start the conversation by typing your question below.</p>', unsafe_allow_html=True)
                for sender, message in st.session_state.chat_history:
                    msg_class = "user-message" if sender == "User" else "ai-message"
                    st.markdown(f"""
                        <div class="chat-message {msg_class}">
                            <b>{sender}:</b> {message}
                            <span class="copy-tooltip">Copied!</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            user_input = st.chat_input("Ask the AI assistant about your groundwater data or forecast:")
            if user_input:
                log_session_activity("AI Chat Query", user_input)
                st.session_state.chat_history.append(("User", user_input))
                with st.spinner("AI is thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history, st.session_state.cleaned_data,
                        st.session_state.forecast_results, st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                st.session_state.chat_history.append(("AI", ai_response))
                log_session_activity("AI Chat Response", ai_response)
                st.rerun() # Rerun to display new message
        else:
            st.warning("Please upload groundwater data and run an AI forecast first to provide context for the chatbot.")
            st.session_state.chat_active = False # Deactivate if context is missing
            st.rerun()
    else:
        st.info("Click 'Activate AI Chat' in the sidebar to interact with the AI assistant after a successful forecast.")

# Generic Data Analyzer Tab
with tabs[5]: # Generic Data Analyzer
    st.header("Generic Data Analyzer: Visualize Your Datasets")
    st.info("Upload any CSV or Excel file to explore and visualize its contents. This feature is independent of the groundwater forecasting.")
    
    generic_uploaded_file = st.file_uploader("Upload Any Data File (CSV, XLSX)", type=["csv", "xlsx"], key="generic_data_uploader")

    if generic_uploaded_file is not None:
        if st.session_state.get("generic_uploaded_data_filename") != generic_uploaded_file.name:
            st.session_state.generic_uploaded_data_filename = generic_uploaded_file.name
            with st.spinner("Loading generic data..."):
                try:
                    if generic_uploaded_file.type == "text/csv":
                        generic_df = pd.read_csv(generic_uploaded_file)
                    else: # Assuming application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
                        generic_df = pd.read_excel(generic_uploaded_file, engine="openpyxl")
                    st.session_state.generic_uploaded_data = generic_df
                    st.success(f"Successfully loaded '{generic_uploaded_file.name}' with {len(generic_df)} rows and {len(generic_df.columns)} columns.")
                    log_session_activity("Generic Data Upload", f"File: {generic_uploaded_file.name}, Rows: {len(generic_df)}")
                except Exception as e:
                    st.error(f"Error loading generic data: {e}. Please ensure it's a valid CSV or XLSX file.")
                    st.session_state.generic_uploaded_data = None
                    log_session_activity("Generic Data Upload Failed", f"File: {generic_uploaded_file.name}, Error: {e}")
            st.rerun()

    if st.session_state.generic_uploaded_data is not None:
        generic_df = st.session_state.generic_uploaded_data
        st.subheader("Data Preview (First 50 Rows)")
        st.dataframe(generic_df.head(50), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.write(generic_df.describe())

        st.subheader("Interactive Plotting")
        all_columns = generic_df.columns.tolist()
        numeric_columns = generic_df.select_dtypes(include=np.number).columns.tolist()
        date_columns = generic_df.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns.tolist()

        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            x_axis_col = st.selectbox("Select X-axis Column", options=[''] + all_columns, key="x_axis_col")
        with col_plot2:
            y_axis_col = st.selectbox("Select Y-axis Column", options=[''] + (numeric_columns if x_axis_col else all_columns), key="y_axis_col")
        
        col_plot3, col_plot4 = st.columns(2)
        with col_plot3:
            color_col = st.selectbox("Select Color Column (Optional)", options=[''] + all_columns, key="color_col")
        with col_plot4:
            size_col = st.selectbox("Select Size Column (Optional)", options=[''] + numeric_columns, key="size_col")
        
        chart_type = st.selectbox("Select Chart Type", ["Line Plot", "Scatter Plot", "Bar Chart", "Histogram", "Box Plot"], key="chart_type_select")

        if x_axis_col and y_axis_col:
            fig_generic = create_generic_data_plot(generic_df, x_axis_col, y_axis_col, color_col if color_col else None, size_col if size_col else None, chart_type)
            st.plotly_chart(fig_generic, use_container_width=True)
        else:
            st.info("Select X and Y axis columns to generate a plot.")
    else:
        st.info("Upload a CSV or XLSX file above to start analyzing your data.")

# Session Activity Log Tab
with tabs[6]: # Session Activity Log
    st.header("Current Session Activity Log")
    st.info("This dashboard shows the actions you've taken during your current browsing session. This demonstrates the application's ability to track user engagement with key features.")
    
    if st.session_state.session_activity_log:
        activity_df = pd.DataFrame(st.session_state.session_activity_log)
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No activities logged in this session yet. Interact with the application to see activities appear here.")

# --- End of Main Application Area ---

