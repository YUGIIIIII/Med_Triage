import streamlit as st
import os
import json
import datetime
import uuid
import re
import time
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from chromadb.utils import embedding_functions

# --- UTILITY FUNCTIONS ---

def validate_google_api_key(api_key):
    """Validates Google API key and checks for available generation models."""
    if not api_key:
        return False, "API key cannot be empty."
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        generation_models = [
            m.name for m in models
            if 'generateContent' in getattr(m, 'supported_generation_methods', [])
        ]
        
        if not generation_models:
            return False, "No generation models found for this API key. Ensure Generative Language API is enabled."
        
        return True, f"API key valid. Found {len(generation_models)} generation models."
    except Exception as e:
        return False, f"API key validation failed: {e}. Check key format and permissions."

def initialize_orchestrator(api_key):
    """Initialize the orchestrator with Gemini 2.5 Flash model."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Use Gemini 2.5 Flash model specifically
        model_name = "gemini-2.5-flash"
        
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, max_retries=3)
            llm.invoke("Connection test")
            st.success(f"✅ Successfully connected using model: {model_name}")
            return MedicalAgentOrchestrator(llm=llm)
        except Exception as model_error:
            st.error(f"Failed to initialize {model_name}: {model_error}")
            raise Exception(f"Model initialization failed: {model_error}")
        
    except Exception as e:
        st.error(f"Failed to initialize language model: {e}")
        return None

# --- CORE ORCHESTRATOR LOGIC ---

class MedicalAgentOrchestrator:
    """Manages the workflow of memory-enabled medical agents."""

    def __init__(self, llm):
        self.llm = llm
        self.main_conversation_history = ""
        self.log_dir = "diagnosis_logs"
        self.json_log_dir = "diagnosis_logs_json"
        self.vector_db_path = "vector_db"
        self._setup_logging()
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_diagnoses", embedding_function=self.embedding_function
            )
        except Exception:
            self.chroma_client = None
            self.collection = None

        self.agents = self._create_agents()
        self.case_started = False
        self.referred_specialists_for_retry = []

    def _setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.json_log_dir, exist_ok=True)

    def _create_prompt_template(self, role_prompt):
        template = role_prompt.strip() + "\n\nCurrent Conversation:\n{history}\n\nHuman: {input}\nAI:"
        return PromptTemplate(input_variables=["history", "input"], template=template)

    def _create_agents(self):
        agents = {}
        prompts = {
            "GeneralPhysician_Triage": """Act as an expert General Physician performing initial triage. You can refer to one or more of these specialists: Cardiologist, Psychologist, Pulmonologist, Neurologist. Analyze the patient's symptoms and medical history. You MUST return ONLY a JSON object with two keys: "referral" (a list of specialist names from the available specialists, or an empty list if you can diagnose directly) and "diagnosis" (a string with your findings if not referring, or null if referring to specialists). Example: {{"referral": ["Cardiologist", "Psychologist"], "diagnosis": null}}. Return ONLY the JSON object, nothing else.""",
            "GeneralPhysician_Check": """Act as an expert General Physician reviewing specialist reports for consistency. Analyze all specialist reports provided in the conversation. You MUST return ONLY a JSON object with one key: "conflict" (a string describing any conflict between specialist reports, or null if reports are consistent). Example: {{"conflict": "Cardiologist suggests cardiac issue while Psychologist indicates anxiety disorder"}} or {{"conflict": null}}. Return ONLY the JSON object, nothing else.""",
            "Cardiologist": "Act as an expert Cardiologist. Analyze the conversation and medical history, providing a focused analysis on cardiovascular health. Consider symptoms like chest pain, palpitations, shortness of breath, and other cardiac-related concerns.",
            "Psychologist": "Act as an expert Psychologist. Analyze the conversation and medical history, providing a psychological assessment. Consider mental health conditions like anxiety, depression, panic disorders, and stress-related symptoms.",
            "Pulmonologist": "Act as an expert Pulmonologist. Analyze the conversation and medical history, providing a pulmonary assessment. Consider respiratory symptoms like shortness of breath, coughing, wheezing, and other lung-related concerns.",
            "Neurologist": "Act as an expert Neurologist. Analyze the conversation and medical history, providing a neurological assessment. Consider symptoms like headaches, dizziness, trembling, numbness, and other neurological concerns.",
            "ConflictResolver": """Act as an expert medical conflict resolver. You are given specialist reports that are conflicting. Your task is to analyze them and provide a reasoned resolution. You MUST return a JSON object with a single key: "resolution". Example: {{"resolution": "The symptoms align more with stress-induced tachycardia, so the Psychologist's assessment should be prioritized."}}.""",
            "MultidisciplinaryTeam": "Act as a multidisciplinary team consisting of a Cardiologist, Psychologist, Pulmonologist, and Neurologist. Review the entire conversation history and all specialist reports. Synthesize this information to provide a final list of the 3 most likely health issues, each with a concise justification based on the specialist assessments. When asked a follow-up question, use the full conversation context to provide an accurate answer."
        }
        for role, prompt_text in prompts.items():
            prompt_template = self._create_prompt_template(prompt_text)
            memory = ConversationBufferWindowMemory(k=15)
            agents[role] = ConversationChain(llm=self.llm, prompt=prompt_template, memory=memory)
        
        # Create separate instances for GP's two roles
        agents["GeneralPhysician"] = agents["GeneralPhysician_Triage"]
        
        return agents

    def safe_json_parse(self, json_string):
        try:
            if json_string is None: return None
            cleaned = str(json_string).strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            json_match = re.search(r'\{[\s\S]*\}', cleaned.strip())
            return json.loads(json_match.group(0)) if json_match else None
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

    def _log_event(self, event_name, data):
        try:
            log_entry = {"timestamp": datetime.datetime.now().isoformat(), "event": event_name, "data": data}
            base_filename = os.path.splitext(self.report_filename)[0]
            log_filename = f"{self.json_log_dir}/{base_filename}_{self.run_id}.jsonl"
            with open(log_filename, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception: pass

    def _save_conversation_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(self.report_filename)[0]
        log_filename = f"{self.log_dir}/{base_filename}_{timestamp}.txt"
        try:
            with open(log_filename, "w") as f: f.write(self.main_conversation_history)
            if self.collection is not None:
                self.collection.upsert(documents=[self.main_conversation_history], ids=[str(self.run_id)])
            self.last_log_filename = log_filename  # Store the filename
            return f"✅ Diagnosis log saved to: {log_filename}"
        except Exception as e:
            return f"❌ Error saving log file: {e}"
    
    def get_last_log_content(self):
        """Get the content of the last saved log file."""
        if hasattr(self, 'last_log_filename'):
            try:
                with open(self.last_log_filename, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading log file: {e}"
        return "No log file available yet."

    def get_agent_memories(self):
        """Retrieve the conversation memory from all agents."""
        memories = {}
        for agent_name, agent_chain in self.agents.items():
            try:
                # Get the memory buffer content
                memory_vars = agent_chain.memory.load_memory_variables({})
                memories[agent_name] = memory_vars.get('history', 'No conversation history')
            except Exception as e:
                memories[agent_name] = f"Error retrieving memory: {e}"
        return memories

    def export_all_memories(self):
        """Export all agent memories to a formatted string."""
        memories = self.get_agent_memories()
        output = "=" * 80 + "\n"
        output += "AGENT MEMORY LOGS\n"
        output += "=" * 80 + "\n\n"
        
        for agent_name, memory_content in memories.items():
            output += f"\n{'='*80}\n"
            output += f"AGENT: {agent_name}\n"
            output += f"{'='*80}\n"
            output += f"{memory_content}\n"
        
        return output

    def _get_similar_cases(self, query):
        if self.collection is None: return []
        try:
            results = self.collection.query(query_texts=[query], n_results=3)
            return results["documents"][0] if results.get("documents") else []
        except Exception:
            return []

    def process_report(self, medical_report, report_filename="uploaded_report.txt"):
        self.case_started = True
        self.report_filename = report_filename
        self.run_id = uuid.uuid4()
        self.main_conversation_history = ""
        yield "### Step 0: Retrieving Similar Cases\n---\n"
        similar_cases = self._get_similar_cases(medical_report)
        if similar_cases:
            self.main_conversation_history += "\n--- Similar Past Cases ---\n" + "\n\n".join(similar_cases) + "\n\n"
            yield f"Found {len(similar_cases)} similar past cases.\n"
        else:
            yield "No similar past cases found.\n"
        yield "\n### Step 1: General Physician Triage\n---\n"
        self.main_conversation_history += f"Initial Medical Report:\n\n{medical_report}\n"
        gp_chain = self.agents["GeneralPhysician"]
        gp_response_str = gp_chain.predict(input=self.main_conversation_history)
        time.sleep(1)
        self.main_conversation_history += f"\n--- General Physician Triage Notes ---\n{gp_response_str}\n"
        yield f"*GP Triage Output:*\n```json\n{gp_response_str}\n```\n\n"
        gp_decision = self.safe_json_parse(gp_response_str)
        if gp_decision is None:
            yield "❌ Error: GP response was not valid JSON. Cannot proceed."
            return
        if gp_decision.get("referral"):
            yield f"\n### Step 2: Specialist Consultations for {gp_decision['referral']}\n---\n"
            for update in self._run_specialist_flow(gp_decision["referral"]): yield update
        else:
            yield "\n### Final Diagnosis from General Physician\n---\n"
            yield f"{gp_decision.get('diagnosis', 'No diagnosis provided.')}\n"
        yield f"\n---\n{self._save_conversation_log()}\n\n"
        yield "*Analysis complete. You can now ask follow-up questions.*"

    def _run_specialist_flow(self, referred_specialists):
        for i, specialist_name in enumerate(referred_specialists):
            yield f"\n*Consulting {specialist_name} ({i+1}/{len(referred_specialists)})...*\n"
            if specialist_name in self.agents:
                specialist_response = self.agents[specialist_name].predict(input=self.main_conversation_history)
                time.sleep(1)
                self.main_conversation_history += f"\n--- {specialist_name} Report ---\n{specialist_response}\n"
                yield f"*Report from {specialist_name}:*\n\n{specialist_response}\n"
        yield "\n### Step 3: GP Consistency Check\n---\n"
        gp_consistency_response_str = self.agents["GeneralPhysician_Check"].predict(input=self.main_conversation_history)
        time.sleep(1)
        self.main_conversation_history += f"\n--- GP Consistency Check ---\n{gp_consistency_response_str}\n"
        yield f"*GP Consistency Check Output:*\n```json\n{gp_consistency_response_str}\n```\n\n"
        gp_consistency_decision = self.safe_json_parse(gp_consistency_response_str)
        if gp_consistency_decision and gp_consistency_decision.get("conflict"):
            yield f"\n### Step 4: Conflict Resolution\n---\n"
            resolution_response_str = self.agents["ConflictResolver"].predict(input=self.main_conversation_history)
            time.sleep(1)
            self.main_conversation_history += f"\n--- Conflict Resolution ---\n{resolution_response_str}\n"
            yield f"*Conflict Resolution Output:*\n{resolution_response_str}\n"
        yield "\n### Step 5: Final Multidisciplinary Team Assessment\n---\n"
        mdt_response = self.agents["MultidisciplinaryTeam"].predict(input=self.main_conversation_history)
        time.sleep(1)
        self.main_conversation_history += f"\n--- Final MDT Assessment ---\n{mdt_response}\n"
        yield f"*Final Assessment:*\n\n{mdt_response}\n"

    def process_follow_up(self, question):
        if not self.case_started: return "Please start with an initial report."
        self.main_conversation_history += f"\n--- Follow-up Question ---\nHuman: {question}\n"
        response = self.agents["MultidisciplinaryTeam"].predict(input=self.main_conversation_history)
        time.sleep(1)
        self.main_conversation_history += f"AI: {response}\n"
        st.info(self._save_conversation_log())
        return response

# --- STREAMLIT UI ---

st.set_page_config(page_title="🩺 MedAgent Collaborative Diagnosis", layout="wide")
st.title("🩺 MedAgent: A Collaborative AI Diagnostic System")
st.markdown("This application uses a multi-agent AI system to analyze medical reports.")

# Get API key from Streamlit secrets
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("❌ Google API Key not found in Streamlit secrets. Please add GOOGLE_API_KEY to your secrets.toml file.")
    st.stop()

# Validate API key
is_key_valid, validation_message = validate_google_api_key(google_api_key)
if not is_key_valid:
    st.error(f"❌ {validation_message}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.header("⚠ Disclaimer")
    st.warning("This software is for educational purposes only and is not a substitute for professional medical advice.")
    
    # Agent Memory Inspector in Sidebar
    st.markdown("---")
    st.header("🧠 Agent Memory Inspector")
    st.markdown("View the conversational buffer memory of individual agents.")
    
    if "orchestrator" in st.session_state and st.session_state.orchestrator and st.session_state.get("analysis_complete", False):
        selected_agent = st.selectbox(
            "Select Agent to Inspect",
            list(st.session_state.orchestrator.agents.keys())
        )
        
        if st.button("🔍 View Memory", key="view_memory_btn"):
            with st.spinner("Retrieving memory..."):
                memories = st.session_state.orchestrator.get_agent_memories()
                st.text_area(
                    f"{selected_agent} Memory Buffer",
                    memories.get(selected_agent, "No memory found"),
                    height=300,
                    key=f"memory_display_{selected_agent}"
                )
        
        st.markdown("---")
        all_memories = st.session_state.orchestrator.export_all_memories()
        st.download_button(
            label="⬇️ Download All Memories",
            data=all_memories,
            file_name=f"agent_memories_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_all_memories"
        )
    else:
        st.info("Complete an analysis to inspect agent memories.")

if "orchestrator" not in st.session_state: st.session_state.orchestrator = None
if "messages" not in st.session_state: st.session_state.messages = []
if "analysis_complete" not in st.session_state: st.session_state.analysis_complete = False

if st.session_state.orchestrator is None:
    with st.spinner("Initializing agents with Gemini 2.5 Flash..."):
        st.session_state.orchestrator = initialize_orchestrator(google_api_key)

if st.session_state.orchestrator:
    st.header("1. Submit Medical Report")
    example_report = """*Patient:* Michael Johnson, 35-year-old male, Software Engineer.
*Symptoms:* Recurrent episodes of intense fear, heart palpitations, shortness of breath, chest tightness, dizziness, and trembling.
*History:* Two ER visits in the past six months; all cardiac workups were negative. Patient is worried he is having a heart attack."""
    medical_report = st.text_area("Paste the medical report below:", height=200, value=example_report)

    if st.button("Analyze Report", type="primary"):
        if medical_report.strip():
            st.session_state.messages = [{"role": "user", "content": medical_report}]
            st.session_state.analysis_complete = False
            with st.status("Running full diagnostic workflow...", expanded=True) as status:
                full_response_content = "".join(st.session_state.orchestrator.process_report(medical_report))
                st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                status.update(label="Analysis Complete!", state="complete")
            st.session_state.analysis_complete = True
            st.rerun()

    if st.session_state.messages:
        st.header("Diagnostic Process Log")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.analysis_complete:
        st.header("2. Ask Follow-up Questions")
        if prompt := st.chat_input("Ask a follow-up question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("MDT is thinking..."):
                    response = st.session_state.orchestrator.process_follow_up(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add section to view and download the diagnosis log
        st.header("3. View & Download Diagnosis Log")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📄 View Full Diagnosis Log"):
                log_content = st.session_state.orchestrator.get_last_log_content()
                st.text_area("Complete Diagnosis Log", log_content, height=400, key="full_log_display")
        
        with col2:
            log_content = st.session_state.orchestrator.get_last_log_content()
            st.download_button(
                label="⬇️ Download Diagnosis Log",
                data=log_content,
                file_name=f"diagnosis_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_diagnosis_log"
            )
else:
    st.warning("Please resolve API key/model issues to proceed.")
