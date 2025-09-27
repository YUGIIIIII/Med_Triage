import streamlit as st
from orchestrator import MedicalAgentOrchestrator # Make sure this file exists
import time
import os

# --- Page Configuration ---
st.set_page_config(page_title="Medical Diagnosis Agent", page_icon="🩺", layout="wide")

# --- API Key and State Initialization ---
# For deployment, set your API key in Streamlit's secrets management
# e.g., GOOGLE_API_KEY = "your_key_here"
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    # This is a fallback for local development, not recommended for production
    st.warning("🚨 GOOGLE_API_KEY not found in Streamlit secrets. Please add it for deployed apps.", icon="⚠️")
    # You might want to add a text input for the key for local testing:
    # api_key = st.text_input("Enter your Google API Key:", type="password")
    # if not api_key:
    #     st.stop()
    # os.environ["GOOGLE_API_KEY"] = api_key


# Initialize orchestrator and chat history in session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MedicalAgentOrchestrator()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'case_started' not in st.session_state:
    st.session_state.case_started = False

# --- Sidebar ---
with st.sidebar:
    st.title("🩺 Medical Diagnosis Agent")
    st.markdown("""
    This tool uses a multi-agent system to analyze medical reports.
    
    **How to use:**
    1.  Upload a patient's medical report (.txt).
    2.  The system will perform a full analysis.
    3.  Ask follow-up questions in the chat window.
    """)
    st.divider()
    if st.button("Start New Case"):
        # Reset the session state to start over
        st.session_state.messages = []
        st.session_state.case_started = False
        st.rerun()


# --- Main Content ---
st.title("Medical Analysis & Chat")

# 1. Show the file uploader if the case has not started
if not st.session_state.case_started:
    uploaded_file = st.file_uploader(
        "Upload a .txt medical report to begin the analysis",
        type="txt",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        # Once a file is uploaded, start the analysis
        st.session_state.case_started = True
        report_content = uploaded_file.getvalue().decode("utf-8")
        
        # Display the analysis as it's generated
        with st.chat_message("assistant"):
            with st.spinner("Performing multi-agent analysis... This may take a moment."):
                full_response = ""
                message_placeholder = st.empty()
                # Assuming your orchestrator has a method that yields parts of the response
                for part in st.session_state.orchestrator.process_initial_report(report_content):
                    full_response += part
                    message_placeholder.markdown(full_response + "▌")
                    # A small sleep helps the streaming effect look smoother
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)
        
        # Save the full analysis to the message history and rerun
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# 2. Display the chat interface if the case has started
else:
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle the follow-up chat input
    if prompt := st.chat_input("Ask a follow-up question..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.orchestrator.process_follow_up(prompt)
                st.markdown(response)
        
        # Add AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
