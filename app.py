{"conflict": "Cardiologist points to stress, Pulmonologist points to infection."}
```""",
            "Cardiologist": "Act as an expert Cardiologist. Analyze the conversation and medical history, providing a focused analysis on cardiovascular health.",
            "Psychologist": "Act as an expert Psychologist. Analyze the conversation and medical history, providing a psychological assessment.",
            "Pulmonologist": "Act as an expert Pulmonologist. Analyze the conversation and medical history, providing a pulmonary assessment.",
            "Neurologist": "Act as an expert Neurologist. Analyze the conversation and medical history, providing a neurological assessment.",
            "ConflictResolver": """Act as an expert conflict resolver. You are given specialist reports that are conflicting. Your task is to analyze them and provide a reasoned resolution. You MUST return a JSON object with a single key: "resolution". Example: {"resolution": "The symptoms align more with stress-induced tachycardia, so the Psychologist's assessment should be prioritized."}.""",
            "MultidisciplinaryTeam": "Act as a multidisciplinary team. Review the entire conversation history. Synthesize this information to provide a final list of the 3 most likely health issues, each with a concise justification. When asked a follow-up question, use the full conversation context to provide an accurate answer."
        }
        for role, prompt_text in prompts.items():
            prompt_template = self._create_prompt_template(prompt_text)
            memory = ConversationBufferWindowMemory(k=15)
            agents[role] = ConversationChain(llm=self.llm, prompt=prompt_template, memory=memory)
        return agents

    def safe_json_parse(self, json_string):
        try:
            if json_string is None: return None
            # Find the first '{' and the last '}' to extract the JSON object
            json_match = re.search(r'\{.*\}', str(json_string), re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None
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
            return f"✅ Diagnosis log saved to: {log_filename}"
        except Exception as e:
            return f"❌ Error saving log file: {e}"

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
        gp_consistency_response_str = self.agents["GeneralPhysician"].predict(input=self.main_conversation_history)
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

with st.sidebar:
    st.header("⚠ Disclaimer")
    st.warning("This software is for educational purposes only and is not a substitute for professional medical advice.")
    st.markdown("---")
    st.info("API Key is securely loaded from Streamlit Secrets.")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("🔴 GOOGLE_API_KEY not found in Streamlit Secrets.")
    st.info("Please create a .streamlit/secrets.toml file and add your key.")
    st.stop()


if "orchestrator" not in st.session_state: st.session_state.orchestrator = None
if "messages" not in st.session_state: st.session_state.messages = []
if "analysis_complete" not in st.session_state: st.session_state.analysis_complete = False

if google_api_key and st.session_state.orchestrator is None:
    with st.spinner("Initializing agents... This may take a moment."):
        st.session_state.orchestrator = initialize_orchestrator(google_api_key)

if st.session_state.orchestrator:
    st.header("1. Submit Medical Report")
    example_report = """*Patient:* Michael Johnson, 35-year-old male, Software Engineer.
*Symptoms:* Recurrent episodes of intense fear, heart palpitations, shortness of breath, chest tightness, dizziness, and trembling.
*History:* Two ER visits in the past six months; all cardiac workups were negative. Patient is worried he is having a heart attack."""
    medical_report = st.text_area("Paste the medical report below:", height=200, value=example_report)

    if st.button("Analyze Report", type="primary"):
        if medical_report.strip():
            st.session_state.messages = [{"role": "user", "content": f"**Initial Report:**\n\n---\n\n{medical_report}"}]
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
            with st.chat_message("user"): 
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("MDT is thinking..."):
                    response = st.session_state.orchestrator.process_follow_up(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Could not initialize the diagnostic agents. Please check the error messages above.")
