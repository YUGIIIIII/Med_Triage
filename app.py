import streamlit as st
# from orchestrator import MedicalAgentOrchestrator # Assuming you have this file
import time
import os

# --- Dummy Orchestrator for Demonstration ---
# Replace this with your actual orchestrator.py import
class MedicalAgentOrchestrator:
    def process_initial_report(self, report_content):
        # This is a generator, yielding parts of the response
        response = f"### Analysis of Report\n\n- **Patient Symptoms:** Analysis of symptoms mentioned in the report...\n- **Initial Findings:** Based on the content: '{report_content[:100]}...'\n- **Specialist Consultation Summary:**\n  - **Cardiologist:** Notes on cardiac health.\n  - **Neurologist:** Notes on neurological signs.\n\n**Final Assessment:** The diagnosis is pending further tests. Please ask any follow-up questions."
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    def process_follow_up(self, prompt):
        return f"Regarding your question about '{prompt}', the multi-agent team suggests further consultation. The confidence score is currently 0.75."

# --- Page Configuration ---
st.set_page_config(page_title="Medical Diagnosis Agent", page_icon="🩺", layout="wide")


# --- API Key and State Initialization ---
# This part is for deployment. For local testing, you can comment it out
# and set the key directly if needed.
# try:
#     os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# except:
#     st.error("🚨 Google API Key not found. Please set it in your Streamlit secrets.")
#     st.stop()

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
    1. Upload a patient's medical report (.txt).
    2. The system will perform a full analysis.
    3. Ask follow-up questions in the chat window.
    """)
    if st.button("Start New Case"):
        st.session_state.messages = []
        st.session_state.case_started = False
        st.rerun()


# --- Main App Logic ---
st.header("Case Analysis & Chat")

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main logic controller
if not st.session_state.case_started:
    # --- STATE 1: WAITING FOR FILE UPLOAD ---
    uploaded_file = st.file_uploader(
        "Upload a patient's .txt medical report to begin analysis.",
        type="txt"
    )
    if uploaded_file is not None:
        # File has been uploaded, start the analysis
        st.session_state.case_started = True
        report_content = uploaded_file.getvalue().decode("utf-8")

        # Display the streaming analysis as the first assistant message
        with st.chat_message("assistant"):
            with st.spinner("Performing multi-agent analysis... This may take a moment."):
                full_response = st.write_stream(
                    st.session_state.orchestrator.process_initial_report(report_content)
                )

        # Save the full analysis to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

else:
    # --- STATE 2: CASE STARTED, WAITING FOR FOLLOW-UP ---
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
