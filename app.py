import streamlit as st
from orchestrator import Orchestrator

st.set_page_config(page_title="Med Triage", page_icon="🩺", layout="wide")

# --- Initialize session state ---
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "case_started" not in st.session_state:
    st.session_state.case_started = False

if "report_content" not in st.session_state:
    st.session_state.report_content = None


# --- Sidebar ---
with st.sidebar:
    st.header("🩺 Med Triage")
    st.write("Upload a medical report to begin the triage process.")

    if st.button("🔄 Reset Case"):
        st.session_state.messages = []
        st.session_state.case_started = False
        st.session_state.report_content = None
        st.experimental_rerun()


# --- Main Area ---
st.title("Medical Report Triage Assistant")

if not st.session_state.case_started:
    # First step: upload the report
    uploaded_file = st.file_uploader("Upload initial medical report", type=["txt"])

    if uploaded_file is not None:
        # Save case state and file content
        st.session_state.case_started = True
        st.session_state.report_content = uploaded_file.getvalue().decode("utf-8")

        # Process the initial report once
        with st.spinner("Analyzing initial report..."):
            for part in st.session_state.orchestrator.process_initial_report(
                st.session_state.report_content
            ):
                st.session_state.messages.append({"role": "assistant", "content": part})

        st.success("Initial report processed. Chat is ready!")

else:
    # --- Chat Interface ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the patient..."):
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.orchestrator.handle_user_message(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
