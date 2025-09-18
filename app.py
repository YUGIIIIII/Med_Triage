# File: app.py

import streamlit as st
from orchestrator import MedicalAgentOrchestrator
import time
import os

# --- Page Configuration ---
st.set_page_config(page_title="Medical Diagnosis Agent", page_icon="🩺", layout="wide")

# --- API Key and State Initialization ---
try:
    # For deployment, set this in st.secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("🚨 Google API Key not found. Please set it in your Streamlit secrets.")
    st.stop()
    
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
    1.  Upload a patient's medical report (`.txt`).
    2.  The system will perform a full analysis.
    3.  Ask follow-up questions in the chat window.
    """)

# --- Main Content ---
st.header("1. Upload Medical Report")

# File Uploader - shown only if the case has not started
if not st.session_state.case_started:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        st.session_state.case_started = True
        report_content = uploaded_file.getvalue().decode("utf-8")
        
        # Add a placeholder for the analysis results
        st.session_state.messages.append({"role": "assistant", "content": ""})
        st.rerun()

# Display previous chat messages and run analysis if needed
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If this is the first run after upload, stream the analysis
if st.session_state.case_started and len(st.session_state.messages) == 1:
    with st.chat_message("assistant"):
        with st.spinner('Performing multi-agent analysis...'):
            full_response = ""
            message_placeholder = st.empty()
            
            # The 'report_content' is not in session_state, so we need a temp solution
            # Note: A more robust app might store the report in session_state
            # For this flow, we'll re-read it from the uploader object if available
            # This part is tricky; a simpler way is to just process and then allow chat.
            # Let's simplify the logic to process and then rerun.
            
            # Simplified Logic:
            # We already set case_started, let's process here directly
            # This logic needs to be outside the loop
            pass # We will handle this outside the loop

# This block should be run once after file upload
if st.session_state.case_started and len(st.session_state.messages) == 1:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # The report content is lost on rerun, so let's process it immediately
        # The logic has to be restructured. A better way:
        
        # Corrected Logic for Streamlit
        # The processing must happen right after upload and before the first rerun
        
        # Let's restart this section for clarity
        pass


# ---- CORRECTED MAIN LOGIC ----
st.header("2. Analysis & Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling the upload and initial analysis
if not st.session_state.case_started:
    uploaded_file = st.file_uploader("Choose a .txt file to begin analysis", type="txt")
    if uploaded_file is not None:
        st.session_state.case_started = True
        report_content = uploaded_file.getvalue().decode("utf-8")
        
        # Display the analysis as it's generated
        with st.chat_message("assistant"):
            with st.spinner("Performing multi-agent analysis..."):
                full_response = ""
                message_placeholder = st.empty()
                for part in st.session_state.orchestrator.process_initial_report(report_content):
                    full_response += part
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.1)
                message_placeholder.markdown(full_response)
        
        # Save the full analysis to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Rerun to clear the file uploader and move to chat input

# Handling the follow-up chat
if st.session_state.case_started:
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