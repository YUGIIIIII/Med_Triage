# File: app.py

import streamlit as st
from orchestrator import MedicalAgentOrchestrator  # ✅ fixed import

# Initialize orchestrator
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MedicalAgentOrchestrator()

orchestrator = st.session_state.orchestrator

st.title("🩺 Medical Triage Assistant")

# Input for initial medical report
st.subheader("Initial Medical Report")
medical_report = st.text_area("Paste the patient's medical report here:")

if st.button("Run Triage") and medical_report:
    st.write("### Triage Process")
    for step in orchestrator.process_initial_report(medical_report):
        st.markdown(step)

# Section for follow-up questions
st.subheader("Follow-up Questions")
follow_up = st.text_input("Ask a follow-up question:")

if st.button("Submit Question") and follow_up:
    response = orchestrator.process_follow_up(follow_up)
    st.markdown("### Answer")
    st.markdown(response)
