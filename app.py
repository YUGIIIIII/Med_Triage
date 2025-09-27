# File: app.py

import streamlit as st
from orchestrator import MedicalAgentOrchestrator

# Initialize orchestrator in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MedicalAgentOrchestrator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store Q&A for chat UI

orchestrator = st.session_state.orchestrator

st.title("🩺 Medical Triage Assistant")

# --- File Upload ---
st.subheader("Upload Medical Report")
uploaded_file = st.file_uploader("Upload a .txt medical report", type=["txt"])

if uploaded_file is not None and st.button("Run Triage"):
    medical_report = uploaded_file.read().decode("utf-8")
    st.session_state.chat_history.append(("system", "📄 Report uploaded and triage started."))

    st.write("### Triage Process")
    triage_output = ""
    for step in orchestrator.process_initial_report(medical_report):
        triage_output += step + "\n"
        st.markdown(step)
    
    # Save final triage result to history
    st.session_state.chat_history.append(("assistant", triage_output))


# --- Chat Feature ---
st.subheader("Follow-up Chat")

user_message = st.chat_input("Ask a follow-up question...")

if user_message:
    st.session_state.chat_history.append(("user", user_message))
    response = orchestrator.process_follow_up(user_message)
    st.session_state.chat_history.append(("assistant", response))

# Render chat messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    elif role == "assistant":
        st.chat_message("assistant").markdown(message)
    else:
        st.chat_message("system").markdown(message)
