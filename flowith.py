import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile
from streamlit_oauth import OAuth2Component
import os
import time # Import time for toast messages
import base64
import json

# --- Constants and Configuration ---
PAGE_HOME = 'Home'
PAGE_XLSX_VIZ = 'XLSX Data Visualization'
PAGE_AI_FORECASTING = 'AI Forecasting'
PAGE_AI_REPORT_GENERATION = 'AI Report Generation' # Name updated as per requirement
PAGE_AI_CHATBOT = 'AI Chatbot' # Name updated as per requirement
PAGE_ADMIN_DASHBOARD = 'Analytics Dashboard' # Name updated as per requirement

COUNTED_FEATURES = {PAGE_AI_REPORT_GENERATION, PAGE_AI_CHATBOT, 'Download PDF Report'} # 'Download PDF Report' treated as a feature/page

MAX_ANONYMOUS_USES = 3
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@example.com")

# --- Streamlit OAuth Configuration ---
# You need to set these environment variables or replace the placeholders
# Go to Google Cloud Console -> APIs & Services -> OAuth Consent Screen
# Go to Google Cloud Console -> APIs & Services -> Credentials -> Create Credentials -> OAuth Client ID
# Select "Web application"
# Add the authorized redirect URI: http://localhost:8501/component/streamlit_oauth.login_button
# (Replace localhost:8501 with your deployed app URL if applicable)
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID") # Replace with your Google Client ID
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET") # Replace with your Google Client Secret
REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501/component/streamlit_oauth.login_button") # Replace with your Redirect URI

# Check for placeholder credentials and set a global flag
oauth_configured = not (CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID" or CLIENT_SECRET == "YOUR_GOOGLE_CLIENT_SECRET")

# Instantiate OAuth2Component only if configured
oauth2 = None
if oauth_configured:
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)


# --- Page Configuration ---
st.set_page_config(page_title="AI-Powered Data Insights App", layout="wide")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'page' not in st.session_state:
    st.session_state.page = PAGE_HOME
# Initialize anonymous use counter
if 'anonymous_uses' not in st.session_state:
    st.session_state.anonymous_uses = 0
# Initialize set to track counted anonymous features used
if 'counted_anonymous_features' not in st.session_state:
    st.session_state.counted_anonymous_features = set()
# Initialize authenticated status and user info
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
# Initialize show_page_content flag
if 'show_page_content' not in st.session_state:
    st.session_state.show_page_content = True

# --- Display OAuth Warning if not configured ---
if not oauth_configured:
    st.error("⚠️ Google OAuth is not configured. Authentication is unavailable. Certain features will have limited access.", icon="⛔️")


# --- Authentication Functions ---
def show_login_prompt():
    """
    Displays a message encouraging the user to sign in and the Google login button
    if OAuth is configured and the user is not authenticated.
    """
    # Check if OAuth is configured before attempting to display login
    if oauth_configured and not st.session_state.authenticated:
        st.info("Please sign in to unlock unlimited access.")
        result = oauth2.authorize_button(
            name="Sign in with Google",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_login_prompt" # Use a unique key
        )
        # Handle the login result if available
        if result and result.get("token"):
            handle_oauth_callback(result.get("token"))
    elif not oauth_configured:
        # Display a message that login is unavailable if OAuth is not configured
        st.info("Login unavailable due to missing OAuth configuration.")


def handle_oauth_callback(token):
    """
    Processes the token received from the OAuth provider and handles the login success or failure.
    WARNING: This is a simplified example. In production, always validate the ID token
    with Google's servers.
    """
    try:
         id_token = token.get('id_token')
         if id_token:
             # Decode ID token (simplified - production needs validation)
             payload_base64 = id_token.split('.')[1]
             # Add padding if necessary
             payload_base64 += '=' * (-len(payload_base64) % 4)
             payload = base64.b64decode(payload_base64).decode('utf-8')
             user_info = json.loads(payload)
             handle_login_success(user_info)
         else:
              st.error("Login failed: ID token not received.")
              st.session_state.authenticated = False
              st.session_state.user_info = None
    except Exception as e:
         st.error(f"Error processing login response: {e}")
         st.session_state.authenticated = False
         st.session_state.user_info = None


def handle_login_success(user_info):
    """
    Handles successful login by updating session state variables.
    Resets anonymous counters and redirects to the home page.
    """
    st.session_state.authenticated = True
    st.session_state.user_info = user_info # Store user info
    st.session_state.anonymous_uses = 0 # Reset anonymous counter on successful login
    st.session_state.counted_anonymous_features = set() # Reset counted features
    st.session_state.page = PAGE_HOME # Redirect to home after login success
    st.session_state.show_page_content = True # Ensure content is shown
    st.experimental_rerun() # Rerun to update UI

def handle_logout():
    """
    Handles logout by clearing authentication-related session state variables.
    Resets state and redirects to the home page.
    """
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.page = PAGE_HOME # Redirect to home after logout
    st.session_state.anonymous_uses = 0 # Reset anonymous counter on logout
    st.session_state.counted_anonymous_features = set() # Reset counted features
    st.session_state.show_page_content = True # Ensure content is shown
    st.experimental_rerun() # Rerun to update UI

def check_authentication():
    """
    Checks if the user is currently authenticated.
    Returns True if authenticated, False otherwise.
    Respects the global `oauth_configured` flag.
    """
    # If OAuth is not configured, authentication is effectively always False
    # as there is no mechanism to authenticate.
    if not oauth_configured:
       return False
    return st.session_state.get("authenticated", False)


def increment_anonymous_use_if_needed_and_check_limit(feature_name):
    """
    Increments the anonymous use counter if the user is not authenticated and
    the specific feature hasn't been counted yet in the current session.
    Checks if the anonymous usage limit has been reached.

    Args:
        feature_name (str): The name of the feature being accessed.

    Returns:
        bool: True if access is granted (within limits or authenticated), False otherwise.
    """
    # Authenticated users have unlimited access
    if check_authentication():
        return True

    # Anonymous users:
    if feature_name in COUNTED_FEATURES:
        # Check if this feature hasn't been counted towards the anonymous limit yet
        if feature_name not in st.session_state.counted_anonymous_features:
            if st.session_state.anonymous_uses < MAX_ANONYMOUS_USES:
                st.session_state.anonymous_uses += 1
                st.session_state.counted_anonymous_features.add(feature_name)
                st.toast(f"Anonymous use {st.session_state.anonymous_uses}/{MAX_ANONYMOUS_USES} for restricted feature '{feature_name}' counted.", icon='✅')
                return True # Grant access for this use
            else:
                 # Usage limit reached for this feature
                 st.warning("You have reached your anonymous usage limit for this feature.")
                 show_login_prompt() # Prompt login if limit reached
                 return False # Deny access
        else:
            # Feature already counted in this session. Check if the total anonymous uses
            # is still within the overall limit.
            if st.session_state.anonymous_uses < MAX_ANONYMOUS_USES:
                 return True # Grant access (already counted, still within total limit)
            else:
                 # Usage limit reached overall
                 st.warning("You have reached your anonymous usage limit for this feature.")
                 show_login_prompt() # Prompt login if limit reached
                 return False # Deny access
    else:
        # Feature is not in the COUNTED_FEATURES list, grant access freely to anonymous users.
        return True


def handle_access_control(current_page):
    """
    Determines whether the current user (authenticated or anonymous) has access
    to the requested page. Manages anonymous usage limits and admin access.

    Args:
        current_page (str): The identifier of the page the user is trying to access.

    Returns:
        bool: True if access is granted, False otherwise. Also updates session state.
    """
    st.session_state.show_page_content = False # Assume content is hidden by default

    if check_authentication():
        # Authenticated User Access Logic
        if current_page == PAGE_ADMIN_DASHBOARD:
            # Check if the authenticated user's email matches the admin email
            if st.session_state.user_info and st.session_state.user_info.get('email') == ADMIN_EMAIL:
                st.session_state.show_page_content = True # Grant admin access
                return True
            else:
                # User is authenticated but not the admin
                st.error("Access Denied: This page is for administrators only.")
                st.session_state.show_page_content = False # Deny access
                return False
        else:
            # Authenticated non-admin users have access to all other pages freely
            st.session_state.show_page_content = True # Grant access
            return True

    else: # Anonymous User Access Logic
        # Check if the current page is one of the features with limited anonymous access
        if current_page in COUNTED_FEATURES:
            # Use the helper function to check/increment anonymous usage for counted features
            has_access = increment_anonymous_use_if_needed_and_check_limit(current_page)
            st.session_state.show_page_content = has_access # Update session state based on access grant
            if not has_access:
                # If access is denied by the increment/check function, it has already displayed warnings/login prompt.
                pass # warnings/login prompt handled inside increment_anonymous_use_if_needed_and_check_limit
            return has_access
        elif current_page == PAGE_ADMIN_DASHBOARD:
             # Anonymous users are never allowed access to the admin dashboard
             st.error("Access Denied: This page is for administrators only. Please sign in.")
             show_login_prompt() # Prompt login for admin page
             st.session_state.show_page_content = False
             return False
        else:
            # current_page is NOT a counted feature (e.g., Home, XLSX Viz). Grant access freely.
            st.session_state.show_page_content = True # Grant access
            # If anonymous limit is reached, show a general info message on any page (except restricted ones where specific message is shown)
            if st.session_state.anonymous_uses >= MAX_ANONYMOUS_USES:
                 st.info("You've used the free features limit. Sign in to unlock AI Report Generation, AI Chatbot, and more.")
            return True


# --- Sidebar Content ---
st.sidebar.title("Navigation")

# Display authentication status and options
if check_authentication():
    st.sidebar.success(f"Logged in as: {st.session_state.user_info.get('email', 'User')}") # Assuming email is in user_info
    if st.sidebar.button("Sign out"):
        handle_logout()
else:
    st.sidebar.info(f"Anonymous uses: {st.session_state.anonymous_uses}/{MAX_ANONYMOUS_USES}")
    # Display Sign in with Google button within the sidebar ONLY if OAuth is configured
    if oauth_configured:
        sidebar_auth_result = oauth2.authorize_button(
            name="Sign in with Google",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_login_sidebar" # Use a unique key for sidebar button
        )
        if sidebar_auth_result and sidebar_auth_result.get("token"):
             handle_oauth_callback(sidebar_auth_result.get("token"))
    else:
        st.sidebar.info("Login unavailable.")


# --- Navigation Buttons ---
# Determine which pages to show based on authentication and potential limits (though limits are enforced *after* button click)
# Admin dashboard shown only if authenticated (button appears, access checked on click)
if st.sidebar.button("Home", key="nav_home"):
    st.session_state.page = PAGE_HOME
if st.sidebar.button("XLSX Data Visualization", key="nav_xlsx_viz"):
    st.session_state.page = PAGE_XLSX_VIZ

# Buttons for features potentially requiring sign-in after anonymous uses
if st.sidebar.button(PAGE_AI_REPORT_GENERATION, key="nav_ai_report"):
    st.session_state.page = PAGE_AI_REPORT_GENERATION
if st.sidebar.button(PAGE_AI_CHATBOT, key="nav_ai_chatbot"):
    st.session_state.page = PAGE_AI_CHATBOT

# Button for Admin page - only show button if authenticated because access check is tied to authentication
if check_authentication() and st.session_state.user_info and st.session_state.user_info.get('email') == ADMIN_EMAIL:
     if st.sidebar.button(PAGE_ADMIN_DASHBOARD, key="nav_admin_dashboard"):
          st.session_state.page = PAGE_ADMIN_DASHBOARD


# --- Main Application Logic ---
def main():
    """
    Main function to orchestrate the application flow.
    Handles page navigation and access control.
    """
    current_page = st.session_state.page

    # --- Access Control Logic ---
    # Call the dedicated function to handle access for the current page
    has_access = handle_access_control(current_page)

    # --- Page Content Rendering ---
    # Only show content if access is granted by the handle_access_control function
    if has_access:
        if current_page == PAGE_HOME:
            st.title(":house: Welcome to the AI-Powered Data Insights App")
            st.write("""
            Navigate through the sidebar to access different features:
            - **XLSX Data Visualization:** Upload an Excel report and get basic visualizations (available for anonymous users).
            - **AI Report Generation:** Use AI to generate custom reports (requires sign-in after 3 anonymous uses).
            - **AI Chatbot:** Interact with your data using a conversational AI interface (requires sign-in after 3 anonymous uses).
            - **Analytics Dashboard:** View application analytics (Admin only).
            """)
            # Provide status message about anonymous uses if not authenticated on Home page
            if not check_authentication():
               remaining_uses = MAX_ANONYMOUS_USES - st.session_state.anonymous_uses
               if remaining_uses > 0:
                   st.info(f"You have {remaining_uses} anonymous uses remaining for features like '{PAGE_AI_REPORT_GENERATION}' and '{PAGE_AI_CHATBOT}'. Sign in for unlimited access.")
               else:
                   st.warning("You have used all your anonymous uses for restricted features. Please sign in for unlimited access.")
                   show_login_prompt() # Show login prompt on Home page if limit reached


        elif current_page == PAGE_XLSX_VIZ:
            st.title(":chart_with_upwards_trend: XLSX Data Visualization")
            st.write("Upload your XLSX file below to visualize its contents.")

            uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx", key="xlsx_uploader")

            if uploaded_file is not None:
                try:
                    df = pd.read_excel(uploaded_file)
                    st.write("Data Preview:")
                    st.dataframe(df.head())

                    st.write("---")
                    st.write("Basic Visualizations:")

                    # Attempt to automatically generate some plots
                    numerical_cols = df.select_dtypes(include=['number']).columns
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                    if len(numerical_cols) > 0:
                        st.subheader("Distribution of Numerical Columns")
                        for col in numerical_cols:
                            st.write(f"**{col}**")
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], kde=True, ax=ax)
                            st.pyplot(fig)

                    if len(categorical_cols) > 0:
                        st.subheader("Count of Categories")
                        for col in categorical_cols:
                            st.write(f"**{col}**")
                            try:
                                # Limiting unique values to avoid plotting issues with too many categories
                                if df[col].nunique() < 20:
                                    fig, ax = plt.subplots()
                                    df[col].value_counts().plot(kind='bar', ax=ax)
                                    plt.xticks(rotation=45, ha='right')
                                    st.pyplot(fig)
                                else:
                                    st.write(f"Skipping plot for '{col}' due to too many unique categories ({df[col].nunique()}).")
                            except Exception as e:
                                 st.warning(f"Could not generate plot for column '{col}': {e}")

                except Exception as e:
                    st.error(f"Error reading or processing the XLSX file: {e}")

        elif current_page == PAGE_AI_REPORT_GENERATION:
             st.title(":page_facing_up: Generate Custom Report")
             st.write("Define criteria to generate a custom PDF report from your data.")
             st.warning("Custom report generation functionality is a placeholder.")
             # Real report generation code would go here (using reportlab or similar)

             # Example of a Counted Feature action within the page content
             # We treat the actual generation/download as the counted action
             if st.button("Generate Example PDF (Counts towards anonymous usage limit)"):
                  # Note: The primary increment happens when navigating to this page.
                  # If you wanted _each_ PDF download to count, you'd add increment logic here
                  # AFTER successful generation, but the current logic counts the page view.
                  # The access control function handle_access_control already checked and potentially
                  # incremented/denied access based on the page view. We only reach here if granted.
                  buffer = io.BytesIO()
                  c = canvas.Canvas(buffer, pagesize=letter)
                  c.drawString(100, 750, "Example Report Title")
                  c.drawString(100, 730, "This is a placeholder report content.")
                  c.save()
                  buffer.seek(0)
                  st.download_button(
                      label="Download Example PDF", # This download is considered part of the 'AI Report Generation' feature use
                      data=buffer,
                      file_name="example_report.pdf",
                      mime="application/pdf"
                  )


        elif current_page == PAGE_AI_CHATBOT:
            st.title(":speech_balloon: Ask Data Chatbot")
            st.write("Chat with your data. Ask questions about trends, summaries, etc.")
            st.warning("Chatbot functionality requires data upload (not implemented yet) and an active OpenAI API key (not included).")

            # Placeholder Chatbot logic
            openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", key="openai_api_key_input")

            if openai_api_key:
                if 'chatbot_messages' not in st.session_state:
                     st.session_state.chatbot_messages = [{"role": "assistant", "content": "Upload data and ask me anything!"}]

                for message in st.session_state.chatbot_messages:
                     with st.chat_message(message["role"]):
                         st.markdown(message["content"])

                if prompt := st.chat_input("Ask a question about your data...", key="chatbot_prompt_input"):
                     if not openai_api_key:
                         st.info("Please add your OpenAI API key to continue.")
                         st.stop()

                     try:
                         llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key) # Use a suitable model

                         # Add user message to chat history
                         st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
                         with st.chat_message("user"):
                             st.markdown(prompt)

                         # Generate AI response (this part needs actual data processing context)
                         # This is a simple echo/placeholder response
                         response = {
                             "role": "assistant",
                             # In a real app, prompt would be sent to LLM with data context
                             "content": f"You asked: {prompt}\n\n(Chatbot functionality needs data processing context and full implementation.)"
                         }

                         # Add AI response to chat history
                         st.session_state.chatbot_messages.append(response)
                         with st.chat_message("assistant"):
                             st.markdown(response["content"])
                     except Exception as e:
                         st.error(f"Error interacting with OpenAI API: {e}. Please check your API key or try again later.")
                         st.session_state.chatbot_messages.append({"role": "assistant", "content": f"Error: Could not process your request. {e}"})
                         with st.chat_message("assistant"):
                             st.markdown(f"Error: Could not process your request. {e}")

            else:
                st.info("Please enter your OpenAI API key to use the chatbot.")


        elif current_page == PAGE_ADMIN_DASHBOARD:
             st.title(":bar_chart: Analytics Dashboard (Admin Only)")
             # The access check is done before the content rendering block.
             # If we are here, access is granted (means user is authenticated and is admin).
             st.write("This is a placeholder for the admin analytics dashboard.")
             st.warning("Admin dashboard functionality is a placeholder.")
             st.write(f"Logged in as: {st.session_state.user_info.get('email', 'User')}") # Display user info if authenticated
             # Real admin dashboard content would go here

    # If has_access is False, the handle_access_control function has already displayed
    # an appropriate error, warning, or login prompt. Nothing else is rendered here.


if __name__ == "__main__":
    main()
