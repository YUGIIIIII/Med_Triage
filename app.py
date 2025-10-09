# app.py
import streamlit as st
import os
import json
import uuid
import re
import time
from typing import Dict, Any, List, Optional

# LangChain + Google generative adapter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Helper: parse JSON safely
# ---------------------------
def safe_json_parse(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    # remove triple backticks and language hints
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # find first JSON object
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # try a more lenient cleanup (replace single quotes)
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None

# ---------------------------
# Agent prompt templates
# ---------------------------
GP_PROMPT = """
You are a General Physician triage agent.

Task 1 (TRIAGE):
Read the patient information and decide whether the case can be handled as a general primary-care issue or requires referral to one or more specialists.
You MUST return a JSON object ONLY (no extra commentary) with these keys:
- "referral": a list of specialist names (choose from "Cardiologist", "Psychologist", "Pulmonologist", "Neurologist") OR an empty list if no referral is needed.
- "diagnosis": a concise diagnosis string if it's a general issue, otherwise null.

Example (general issue):
{"referral": [], "diagnosis": "Acute upper respiratory infection likely; symptomatic care advised."}

Example (needs specialists):
{"referral": ["Cardiologist","Psychologist"], "diagnosis": null}

Task 2 (CONSISTENCY CHECK):
When provided with the full conversation and specialist reports, you may be asked to check for conflicts. In that mode you will be given specialist reports and should return JSON:
{"conflict": null}  or {"conflict": "Short description of conflict"}.

Respond with only the requested JSON object when performing either task.
"""

SPECIALIST_PROMPT = """
You are a {role} specialist.

Read the patient information below and produce a JSON object ONLY with these keys:
- "diagnosis": short diagnosis or differential (string)
- "recommendation": concise next steps or tests (string)
- "confidence": one of "Low", "Medium", "High"

Patient information:
{input}

Return a single JSON object and nothing else.
"""

CONFLICT_PROMPT = """
You are a Conflict Resolver agent.
You will be provided a JSON object that contains specialist reports (diagnosis, recommendation, confidence) for each specialist.
Analyze those specialist reports and determine if there are any conflicts in diagnoses or recommended next steps.
Return a JSON object ONLY with:
- "conflict": null if no meaningful conflict, else a short explanation string.
- "priority": if conflict exists, recommend which specialist opinion to prioritize (one of the specialists' names) or null.
Example:
{"conflict": "Cardiologist suspects ischemia; Pulmonologist attributes symptoms to asthma exacerbation.", "priority": "Cardiologist"}
"""

MDT_PROMPT = """
You are a Multidisciplinary Team (MDT) synthesizer.

Input: full conversation, the General Physician triage, and all specialist reports.
Produce a JSON object ONLY with:
- "top_issues": a list of up to 3 objects, each with keys:
    - "issue": brief issue/diagnosis
    - "justification": 1-2 sentence justification combining available reports
    - "recommended_next_steps": short next steps
Return clean JSON only.
"""

# ---------------------------
# Orchestrator
# ---------------------------
class MedicalAgentOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        # create agents with their own memory
        self.agents = self._create_agents()
        self.run_id = None
        self.conversation_history = ""

    def _create_agents(self) -> Dict[str, LLMChain]:
        agents = {}

        # General Physician - special triage prompt
        gp_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        gp_template = PromptTemplate(input_variables=["input"], template=GP_PROMPT)
        agents["GeneralPhysician"] = LLMChain(llm=self.llm, prompt=gp_template, memory=gp_memory, verbose=False)

        # Specialists
        specialist_roles = {
            "Cardiologist": "Analyzes cardiac-related symptoms such as chest pain, palpitations, or syncope.",
            "Psychologist": "Analyzes psychological symptoms such as anxiety, panic attacks, depression, or behavioral issues.",
            "Pulmonologist": "Analyzes respiratory symptoms such as cough, shortness of breath, wheeze, or chronic lung disease.",
            "Neurologist": "Analyzes neurological symptoms such as dizziness, numbness, weakness, headaches, or seizures.",
        }

        for role, _desc in specialist_roles.items():
            mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            prompt = PromptTemplate(input_variables=["input", "role"], template=SPECIALIST_PROMPT)
            # wrap with a small lambda-style prompt replacement: we will pass role via format
            agents[role] = LLMChain(llm=self.llm, prompt=prompt, memory=mem, verbose=False)

        # Conflict Resolver
        conflict_mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conflict_template = PromptTemplate(input_variables=["input"], template=CONFLICT_PROMPT)
        agents["ConflictResolver"] = LLMChain(llm=self.llm, prompt=conflict_template, memory=conflict_mem, verbose=False)

        # Multidisciplinary team
        mdt_mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        mdt_template = PromptTemplate(input_variables=["input"], template=MDT_PROMPT)
        agents["Multidisciplinary"] = LLMChain(llm=self.llm, prompt=mdt_template, memory=mdt_mem, verbose=False)

        return agents

    def _log(self, entry: str):
        # keep a lightweight conversation history for logs
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history += f"\n[{ts}] {entry}\n"

    def process_report(self, medical_report: str):
        """
        Generator that yields progress messages (string). It:
        1. Runs GP triage
        2. If GP returns referral -> run specialists requested
        3. Run conflict resolver
        4. Run MDT synthesizer
        """
        self.run_id = uuid.uuid4().hex[:8]
        self.conversation_history = ""
        self._log("Received new medical report")
        self._log(medical_report)
        yield "### Step 1 — General Physician triage\n---\n"
        # 1. GP triage
        gp_chain: LLMChain = self.agents["GeneralPhysician"]
        gp_out = gp_chain.run(input=medical_report)
        self._log("GP output: " + str(gp_out))
        yield f"**General Physician output:**\n```\n{gp_out}\n```\n"
        gp_json = safe_json_parse(gp_out)

        if not gp_json:
            yield "❌ Error: General Physician did not return valid JSON. Aborting.\n"
            return

        # If GP decides no referral -> final diagnosis
        referrals: List[str] = gp_json.get("referral", []) or []
        diagnosis = gp_json.get("diagnosis")
        if not referrals:
            yield "\n### Final Diagnosis (handled by General Physician)\n---\n"
            yield f"**Diagnosis:** {diagnosis or 'No diagnosis provided.'}\n"
            self._log("Case closed by GP with diagnosis.")
            return

        # Otherwise consult specialists
        yield f"\n### Step 2 — Specialist consultations: {referrals}\n---\n"
        specialist_reports = {}
        for spec in referrals:
            spec = spec.strip()
            if spec not in self.agents:
                yield f"⚠️ Specialist `{spec}` not found. Skipping.\n"
                continue
            # construct input: include original report + GP notes so specialists have context
            specialist_input = f"Original Report:\n{medical_report}\n\nGP Triage:\n{json.dumps(gp_json)}"
            # For specialist prompt template we need to pass role variable too
            chain: LLMChain = self.agents[spec]
            # run and get text
            spec_out = chain.run(input=specialist_input, role=spec)
            self._log(f"{spec} output: {spec_out}")
            yield f"**Report from {spec}:**\n```\n{spec_out}\n```\n"
            # try to parse JSON from specialist
            parsed = safe_json_parse(spec_out)
            specialist_reports[spec] = {"raw": spec_out, "parsed": parsed}
            time.sleep(0.8)

        # Step 3: GP consistency check — ask GP to check specialist reports for conflict
        yield "\n### Step 3 — GP consistency check\n---\n"
        gp_check_input = (
            "Full conversation:\n"
            f"Initial report:\n{medical_report}\n\nGP triage:\n{json.dumps(gp_json)}\n\n"
            "Specialist reports:\n"
            + "\n".join([f"{k}: {specialist_reports[k]['raw']}" for k in specialist_reports])
        )
        gp_check_out = self.agents["GeneralPhysician"].run(input=gp_check_input)
        self._log("GP consistency check: " + str(gp_check_out))
        yield f"**GP consistency check:**\n```\n{gp_check_out}\n```\n"
        gp_check_json = safe_json_parse(gp_check_out)  # could contain "conflict" key

        # Step 4: Conflict Resolver - analyze specialist outputs
        yield "\n### Step 4 — Conflict Resolution\n---\n"
        # Build structured input for conflict resolver
        conflict_input_obj = {
            "specialist_reports": {
                k: (specialist_reports[k]["parsed"] if specialist_reports[k]["parsed"] else {"text": specialist_reports[k]["raw"]})
                for k in specialist_reports
            },
            "gp_triage": gp_json,
            "gp_check": gp_check_json,
        }
        conflict_input_text = json.dumps(conflict_input_obj, indent=2)
        conflict_out = self.agents["ConflictResolver"].run(input=conflict_input_text)
        self._log("ConflictResolver output: " + str(conflict_out))
        yield f"**Conflict Resolver output:**\n```\n{conflict_out}\n```\n"
        conflict_json = safe_json_parse(conflict_out)

        # Step 5: Multidisciplinary synthesis
        yield "\n### Step 5 — Multidisciplinary Team (MDT) synthesis\n---\n"
        mdt_input_obj = {
            "initial_report": medical_report,
            "gp_triage": gp_json,
            "specialist_reports": {k: specialist_reports[k]["parsed"] or specialist_reports[k]["raw"] for k in specialist_reports},
            "conflict": conflict_json,
        }
        mdt_input_text = json.dumps(mdt_input_obj, indent=2)
        mdt_out = self.agents["Multidisciplinary"].run(input=mdt_input_text)
        self._log("MDT output: " + str(mdt_out))
        yield f"**MDT Final Assessment:**\n```\n{mdt_out}\n```\n"

        yield "\n---\nAnalysis complete. You may ask follow-up questions.\n"

    def process_followup(self, question: str) -> str:
        """
        Append follow-up to MDT memory and get a response from MDT.
        """
        # feed last conversation + follow-up to MDT
        input_text = f"Previous conversation and logs:\n{self.conversation_history}\n\nFollow-up question:\n{question}"
        out = self.agents["Multidisciplinary"].run(input=input_text)
        self._log("Follow-up -> MDT: " + question)
        self._log("MDT follow-up response: " + out)
        return out

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🩺 MedAgent Triage", layout="wide")
st.title("🩺 MedAgent — Multi-Agent Medical Triage")
st.markdown("General Physician triage → specialists → conflict resolver → multidisciplinary synthesis.\n\nAPI key is loaded from Streamlit secrets.")

with st.sidebar:
    st.header("Configuration")
    st.info("Using Gemini 2.5 Flash via Streamlit secrets.")
    st.markdown("---")
    st.header("Disclaimer")
    st.warning("Educational/demo only. Not a substitute for professional medical care.")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.3)
except Exception as e:
    st.error("Could not initialize LLM. Check GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

# Initialize orchestrator in session state
if "orchestrator" not in st.session_state:
    with st.spinner("Initializing agents..."):
        st.session_state.orchestrator = MedicalAgentOrchestrator(llm=llm)

# Input area
st.header("1) Submit Medical Report")
example = (
    "**Patient:** Michael Johnson, 35-year-old male.\n"
    "**Symptoms:** Recurrent episodes of intense fear, palpitations, shortness of breath, chest tightness, dizziness.\n"
    "**History:** Two ER visits; cardiac workups negative."
)
medical_report = st.text_area("Paste the medical report:", value=example, height=180)

if st.button("Analyze Report"):
    if not medical_report.strip():
        st.warning("Please provide a medical report.")
    else:
        orchestrator: MedicalAgentOrchestrator = st.session_state.orchestrator
        output_lines = []
        with st.spinner("Running multi-agent triage..."):
            for part in orchestrator.process_report(medical_report):
                output_lines.append(part)
                # Stream the progress to the app
                st.markdown(part)
                time.sleep(0.2)
        # Save last conversation history for follow-ups
        st.session_state.last_run_history = orchestrator.conversation_history
        st.success("Analysis finished. You can ask follow-up questions below.")

# Show follow-up chat if we have run
if "last_run_history" in st.session_state:
    st.header("2) Ask follow-up questions (MDT)")
    follow = st.text_input("Ask a follow-up question based on the analysis above:")
    if follow:
        orchestrator = st.session_state.orchestrator
        with st.spinner("MDT thinking..."):
            reply = orchestrator.process_followup(follow)
            st.markdown("**MDT Reply:**")
            st.write(reply)
            # persist
            st.session_state.last_run_history = orchestrator.conversation_history

# Footer
st.markdown("---")
st.caption("© MedAgent Triage — Demo (Gemini 2.5 Flash). Keep patient data private.")
