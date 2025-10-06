import streamlit as st
import os
import json
import datetime
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from chromadb.utils import embedding_functions

# --- CORE ORCHESTRATOR LOGIC (Adapted from your Notebook) ---

class MedicalAgentOrchestrator:
    """ Manages the workflow of memory-enabled medical agents. """

    def __init__(self, llm):
        self.llm = llm
        self.main_conversation_history = ""
        self.log_dir = "diagnosis_logs"
        self.json_log_dir = "diagnosis_logs_json"
        self.vector_db_path = "vector_db"
        self._setup_logging()
        self.agents = self._create_agents()
        self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_diagnoses", embedding_function=self.embedding_function
        )
        self.case_started = False
        self.referred_specialists_for_retry = []


    def _setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.json_log_dir, exist_ok=True)

    def _create_prompt_template(self, role_prompt):
        template = f"""{role_prompt}

Current Conversation:
{{history}}

Human: {{input}}
AI:"""
        return PromptTemplate(input_variables=["history", "input"], template=template)

    def _create_agents(self):
        agents = {}
        prompts = {
            "GeneralPhysician": """
                Act as an expert General Physician.
                Your first task is to triage a patient based on their medical report. First, determine if the issue is a general, non-severe condition (like a common cold or simple flu) that you can handle directly.
                - If it IS a general condition, provide a brief diagnosis and recommendation.
                - If it requires a specialist, determine which specialists are needed from the available list: Cardiologist, Psychologist, Pulmonologist, Neurologist.
                You MUST return a JSON object with two keys: "referral" and "diagnosis".
                - For specialist referral: {"referral": ["Cardiologist", "Psychologist"], "diagnosis": null}
                - For direct diagnosis: {"referral": [], "diagnosis": "The patient appears to have a common cold. Recommend rest and hydration."}
                Return *only* the JSON object.

                Your second task is to review the specialist reports and check for consistency.
                - If the reports are consistent, you should confirm this.
                - If there are inconsistencies, you must clearly state the conflict.
                You MUST return a JSON object with a single key: "conflict".
                - If there is a conflict: {"conflict": "The Cardiologist suggests the issue is stress-related, while the Pulmonologist points to a possible respiratory infection. This is a conflict."}
                - If there is no conflict: {"conflict": null}
                Return *only* the JSON object.
            """,
            "Cardiologist": "Act as an expert Cardiologist. Analyze the conversation and medical history, providing a focused analysis on cardiovascular health. State your possible causes and recommended next steps.",
            "Psychologist": "Act as an expert Psychologist. Analyze the conversation and medical history, providing a psychological assessment. State your possible diagnoses and recommended next steps.",
            "Pulmonologist": "Act as an expert Pulmonologist. Analyze the conversation and medical history, providing a pulmonary assessment. State your possible diagnoses and recommended next steps.",
            "Neurologist": "Act as an expert Neurologist. Analyze the conversation and medical history, providing a neurological assessment. State your possible diagnoses and recommended next steps.",
            "ConflictResolver": """
                Act as an expert conflict resolver. You are given a series of specialist reports that have been flagged as potentially conflicting.
                Your task is to analyze the reports, identify the specific points of disagreement, and provide a reasoned resolution.
                You MUST return a JSON object with a single key: "resolution".
                - The value of "resolution" should be a string that explains your reasoning and provides a final, synthesized conclusion.
                Return *only* the JSON object.
            """,
            "MultidisciplinaryTeam": "Act as a multidisciplinary team. Review the entire conversation history, including the GP's notes, all specialist reports, and any conflict resolutions. Synthesize this information to provide a final list of the 3 most likely health issues, each with a concise justification. When asked a follow-up question, use the full conversation context to provide a helpful and accurate answer."
        }
        for role, prompt_text in prompts.items():
            prompt_template = self._create_prompt_template(prompt_text)
            memory = ConversationBufferWindowMemory(k=15)
            agents[role] = ConversationChain(llm=self.llm, prompt=prompt_template, memory=memory)
        return agents

    def _log_event(self, event_name, data):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_name,
            "data": data,
        }
        base_filename = os.path.splitext(self.report_filename)[0]
        log_filename = f"{self.json_log_dir}/{base_filename}_{self.run_id}.jsonl"
        with open(log_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _save_conversation_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(self.report_filename)[0]
        log_filename = f"{self.log_dir}/{base_filename}_{timestamp}.txt"
        try:
            with open(log_filename, "w") as f:
                f.write(self.main_conversation_history)
            
            # Add final conversation to ChromaDB
            self.collection.add(documents=[self.main_conversation_history], ids=[str(self.run_id)])
            return f"✅ Diagnosis log saved to: `{log_filename}`"
        except Exception as e:
            return f"❌ Error saving log file: {e}"
            
    def _get_similar_cases(self, query):
        try:
            results = self.collection.query(query_texts=[query], n_results=3)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            # Handle cases where the collection might be empty or other DB errors
            st.error(f"Could not retrieve similar cases from Vector DB: {e}")
            return []

    def process_report(self, medical_report, report_filename="uploaded_report.txt"):
        """Main generator function to process a medical report and yield updates."""
        self.case_started = True
        self.report_filename = report_filename
        self.run_id = uuid.uuid4()
        self.main_conversation_history = "" # Reset for each new report

        # Step 0: Retrieve Similar Cases
        yield "### Step 0: Retrieving Similar Cases\n---\n"
        similar_cases = self._get_similar_cases(medical_report)
        if similar_cases:
            self.main_conversation_history += "\n--- Similar Past Cases ---\n"
            case_str = "\n\n".join(similar_cases)
            self.main_conversation_history += f"{case_str}\n\n"
            yield f"Found {len(similar_cases)} similar past cases to provide context.\n"
        else:
            yield "No similar past cases found.\n"
        self._log_event("similar_cases_retrieved", {"cases": similar_cases})

        # Step 1: GP Triage
        yield "\n### Step 1: General Physician Triage\n---\n"
        self.main_conversation_history += f"Initial Medical Report:\n\n{medical_report}\n"
        gp_chain = self.agents["GeneralPhysician"]
        gp_response_str = gp_chain.predict(input=self.main_conversation_history)
        self.main_conversation_history += f"\n--- General Physician Triage Notes ---\n{gp_response_str}\n"
        self._log_event("gp_triage", {"input": medical_report, "output": gp_response_str})
        yield f"**GP Triage Output:**\n```json\n{gp_response_str}\n```\n"

        try:
            cleaned_gp_response_str = gp_response_str.strip().replace("`", "").replace("json", "")
            gp_decision = json.loads(cleaned_gp_response_str)
        except (json.JSONDecodeError, TypeError) as e:
            yield f"❌ Error decoding GP response: {e}. Cannot proceed."
            self._log_event("error", {"message": "Fatal: Could not decode GP JSON response."})
            return

        # Step 2: Specialist Consultation (or Final GP Diagnosis)
        if gp_decision.get("referral") and len(gp_decision["referral"]) > 0:
            referred_specialists = gp_decision["referral"]
            yield f"\n### Step 2: Specialist Consultations for `{referred_specialists}`\n---\n"
            
            # This runs the full specialist -> consistency -> conflict -> MDT flow
            for update in self._run_specialist_flow(referred_specialists):
                yield update
        else:
            yield "\n### Final Diagnosis from General Physician\n---\n"
            final_diagnosis = gp_decision.get('diagnosis', 'No diagnosis provided.')
            yield f"**{final_diagnosis}**\n"
            self._log_event("final_diagnosis", {"diagnosis": final_diagnosis})
            
        log_status = self._save_conversation_log()
        yield f"\n---\n{log_status}\n\n"
        yield "**Analysis complete. You can now ask follow-up questions below.**"

    def _run_specialist_flow(self, referred_specialists, is_retry=False):
        """A generator to handle the specialist consultation and subsequent steps."""
        self.referred_specialists_for_retry = referred_specialists
        
        for i, specialist_name in enumerate(referred_specialists):
            yield f"\n*Consulting {specialist_name} ({i+1}/{len(referred_specialists)})...*\n"
            if specialist_name in self.agents:
                specialist_chain = self.agents[specialist_name]
                specialist_response = specialist_chain.predict(input=self.main_conversation_history)
                self.main_conversation_history += f"\n--- {specialist_name} Report ---\n{specialist_response}\n"
                yield f"**Report from {specialist_name}:**\n\n{specialist_response}\n"
                self._log_event("specialist_consultation", {"specialist": specialist_name, "report": specialist_response})
            else:
                yield f"⚠️ Warning: Specialist '{specialist_name}' not found.\n"
                self._log_event("error", {"message": f"Specialist agent '{specialist_name}' not found."})

        # Step 3: GP Consistency Check
        yield "\n### Step 3: GP Consistency Check\n---\n"
        gp_chain = self.agents["GeneralPhysician"]
        consistency_check_input = self.main_conversation_history + "\n\nPlease review the specialist reports above for consistency."
        gp_consistency_response_str = gp_chain.predict(input=consistency_check_input)
        self.main_conversation_history += f"\n--- GP Consistency Check ---\n{gp_consistency_response_str}\n"
        yield f"**GP Consistency Check Output:**\n```json\n{gp_consistency_response_str}\n```\n"
        self._log_event("gp_consistency_check", {"output": gp_consistency_response_str})
        
        try:
            cleaned_str = gp_consistency_response_str.strip().replace("`", "").replace("json", "")
            gp_consistency_decision = json.loads(cleaned_str)
        except (json.JSONDecodeError, TypeError):
            yield "⚠️ Could not decode GP consistency check. Bypassing conflict resolution and proceeding to MDT assessment."
            gp_consistency_decision = {"conflict": None} # Assume no conflict if JSON fails

        # Step 4 & 5: Conflict Resolution and MDT Assessment
        if gp_consistency_decision.get("conflict"):
            yield f"\n### Step 4: Conflict Resolution\n---\n"
            yield f"Conflict identified: {gp_consistency_decision.get('conflict')}"
            resolver_chain = self.agents["ConflictResolver"]
            resolution_input = self.main_conversation_history + f"\n\nA conflict has been identified: {gp_consistency_decision.get('conflict')}. Please resolve this conflict."
            resolution_response_str = resolver_chain.predict(input=resolution_input)
            self.main_conversation_history += f"\n--- Conflict Resolution ---\n{resolution_response_str}\n"
            yield f"**Conflict Resolution Output:**\n{resolution_response_str}\n"
            self._log_event("conflict_resolution", {"conflict": gp_consistency_decision.get("conflict"), "resolution": resolution_response_str})

        yield "\n### Step 5: Final Multidisciplinary Team Assessment\n---\n"
        mdt_chain = self.agents["MultidisciplinaryTeam"]
        mdt_response = mdt_chain.predict(input=self.main_conversation_history)
        self.main_conversation_history += f"\n--- Final MDT Assessment ---\n{mdt_response}\n"
        yield f"**Final Assessment:**\n\n{mdt_response}\n"
        self._log_event("final_diagnosis", {"diagnosis": mdt_response})

    def process_follow_up(self, question):
        """Processes a follow-up question using the MDT agent."""
        if not self.case_started:
            return "Please start by providing an initial medical report."
        
        self.main_conversation_history += f"\n--- Follow-up Question ---\n{question}\n"
        team_chain = self.agents["MultidisciplinaryTeam"]
        response = team_chain.predict(input=self.main_conversation_history)
        self.main_conversation_history += f"\n--- Follow-up Answer ---\n{response}\n"
        
        log_status = self._save_conversation_log()
        st.info(log_status) # Show log status for follow-ups as well
        return response

# --- STREAMLIT UI ---

st.set_page_config(page_title="🩺 MedAgent Collaborative Diagnosis", layout="wide")

st.title("🩺 MedAgent: A Collaborative AI Diagnostic System")
st.markdown("""
This application uses a multi-agent AI system to analyze medical reports. 
A General Physician agent performs initial triage, referring to specialist agents (Cardiologist, Psychologist, etc.) as needed. 
The system includes consistency checks and conflict resolution before a final assessment by a Multidisciplinary Team agent.
""")

# --- Sidebar for API Key and Configuration ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter your Google API Key", type="password")
    
    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.warning("""
    This software is for educational and demonstration purposes only. It does not provide professional medical advice, diagnosis, or treatment. 
    Always seek the guidance of a licensed physician for any medical concerns.
    """)

# --- Main App Logic ---

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# Function to initialize the orchestrator
def initialize_orchestrator(api_key):
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        return MedicalAgentOrchestrator(llm=llm)
    except Exception as e:
        st.error(f"Failed to initialize the language model. Please check your API key. Error: {e}")
        return None

# Check for API key and initialize orchestrator
if google_api_key:
    if st.session_state.orchestrator is None:
        with st.spinner("Initializing agents..."):
            st.session_state.orchestrator = initialize_orchestrator(google_api_key)
else:
    st.warning("Please enter your Google API Key in the sidebar to begin.")
    st.stop()

if st.session_state.orchestrator:
    # --- Report Submission Section ---
    st.header("1. Submit Medical Report")
    
    # Example report for user convenience
    example_report = """**Patient:** Michael Johnson, 35-year-old male, Software Engineer.
**Symptoms:** Recurrent episodes of intense fear, heart palpitations, shortness of breath, chest tightness, dizziness, and trembling. Episodes last 10-15 minutes.
**History:** Two emergency room visits in the past six months; all cardiac workups were negative. Patient is worried he is having a heart attack."""

    medical_report = st.text_area("Paste the medical report below:", height=200, value=example_report)

    if st.button("Analyze Report", type="primary"):
        if not medical_report.strip():
            st.error("Please paste a medical report to analyze.")
        else:
            st.session_state.messages = [] # Clear previous messages
            st.session_state.analysis_complete = False
            with st.status("Running full diagnostic workflow...", expanded=True) as status:
                st.session_state.messages.append({"role": "user", "content": medical_report})
                
                # Use st.write_stream for real-time output
                full_response = st.write_stream(st.session_state.orchestrator.process_report(medical_report))
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                status.update(label="Analysis Complete!", state="complete")
            st.session_state.analysis_complete = True
            st.rerun() # Rerun to show the follow-up input box

    # --- Display conversation ---
    if st.session_state.messages:
        st.header("Diagnostic Process Log")
        for message in st.session_state.messages:
            if message["role"] == "user":
                 with st.chat_message("user", avatar="👤"):
                    st.markdown("**Initial Report:**")
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])

    # --- Follow-up Question Section ---
    if st.session_state.analysis_complete:
        st.header("2. Ask Follow-up Questions")
        
        if prompt := st.chat_input("Ask a follow-up question to the Multidisciplinary Team..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("MDT is thinking..."):
                    response = st.session_state.orchestrator.process_follow_up(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
