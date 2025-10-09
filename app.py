# app.py
import streamlit as st
import os
import json
import uuid
import re
import time
from typing import Dict, Any, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Safe JSON parser
# ---------------------------
def safe_json_parse(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None

# ---------------------------
# Prompt templates
# ---------------------------
GP_PROMPT = """
You are a General Physician triage agent.

Task 1 (TRIAGE):
Read the patient information and decide whether the case can be handled as a general primary-care issue or requires referral to one or more specialists.
Return **strict JSON ONLY** with:
- "referral": list of specialists ("Cardiologist", "Psychologist", "Pulmonologist", "Neurologist") OR []
- "diagnosis": concise diagnosis string if general issue, else null
"""

SPECIALIST_PROMPT = """
You are a {role} specialist.
Patient information:
{input}

Return **strict JSON ONLY**:
- "diagnosis": short string
- "recommendation": short next steps
- "confidence": "Low", "Medium", "High"
"""

CONFLICT_PROMPT = """
You are a Conflict Resolver.
Input: specialist reports and GP triage.
Return **strict JSON ONLY**:
- "conflict": null or short explanation
- "priority": name of specialist to prioritize if conflict exists
"""

MDT_PROMPT = """
You are a Multidisciplinary Team synthesizer.
Input: GP triage, specialist reports, conflict resolver output.
Return **strict JSON ONLY**:
- "top_issues": list of up to 3 issues, each with "issue", "justification", "recommended_next_steps"
"""

# ---------------------------
# Orchestrator
# ---------------------------
class MedicalAgentOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.agents = self._create_agents()
        self.run_id = None
        self.conversation_history = ""

    def _create_agents(self) -> Dict[str, LLMChain]:
        agents = {}

        # General Physician
        gp_mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        gp_template = PromptTemplate(input_variables=["input"], template=GP_PROMPT)
        agents["GeneralPhysician"] = LLMChain(llm=self.llm, prompt=gp_template, memory=gp_mem, verbose=True)

        # Specialists
        for role in ["Cardiologist", "Psychologist", "Pulmonologist", "Neurologist"]:
            mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            prompt = PromptTemplate(input_variables=["input", "role"], template=SPECIALIST_PROMPT)
            agents[role] = LLMChain(llm=self.llm, prompt=prompt, memory=mem, verbose=True)

        # Conflict Resolver
        conflict_mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conflict_template = PromptTemplate(input_variables=["input"], template=CONFLICT_PROMPT)
        agents["ConflictResolver"] = LLMChain(llm=self.llm, prompt=conflict_template, memory=conflict_mem, verbose=True)

        # MDT
        mdt_mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        mdt_template = PromptTemplate(input_variables=["input"], template=MDT_PROMPT)
        agents["Multidisciplinary"] = LLMChain(llm=self.llm, prompt=mdt_template, memory=mdt_mem, verbose=True)

        return agents

    def _log(self, entry: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history += f"\n[{ts}] {entry}\n"

    def process_report(self, medical_report: str):
        self.run_id = uuid.uuid4().hex[:8]
        self.conversation_history = ""
        self._log("Received new medical report")
        self._log(medical_report)

        # Step 1: General Physician
        yield "### Step 1 — General Physician triage\n---\n"
        gp_chain = self.agents["GeneralPhysician"]
        try:
            gp_out = gp_chain.run({"input": medical_report})
        except Exception as e:
            yield f"❌ GP failed: {e}"
            return
        self._log("GP output: " + gp_out)
        yield f"**GP output:**\n```\n{gp_out}\n```\n"
        gp_json = safe_json_parse(gp_out)
        if not gp_json:
            yield "❌ Error: GP did not return valid JSON."
            return

        referrals = gp_json.get("referral", []) or []
        diagnosis = gp_json.get("diagnosis")
        if not referrals:
            yield "\n### Final Diagnosis (GP)\n---\n"
            yield f"**Diagnosis:** {diagnosis or 'No diagnosis'}\n"
            self._log("Case closed by GP")
            return

        # Step 2: Specialist consultations
        yield f"\n### Step 2 — Specialist consultations: {referrals}\n---\n"
        specialist_reports = {}
        for spec in referrals:
            chain = self.agents[spec]
            try:
                spec_out = chain.run({"input": medical_report, "role": spec})
            except Exception as e:
                spec_out = f"❌ {spec} failed: {e}"
            self._log(f"{spec} output: {spec_out}")
            yield f"**Report {spec}:**\n```\n{spec_out}\n```\n"
            specialist_reports[spec] = {"raw": spec_out, "parsed": safe_json_parse(spec_out)}
            time.sleep(0.5)

        # Step 3: Conflict Resolver
        yield "\n### Step 3 — Conflict Resolution\n---\n"
        conflict_input_text = json.dumps({"specialist_reports": specialist_reports, "gp_triage": gp_json})
        try:
            conflict_out = self.agents["ConflictResolver"].run({"input": conflict_input_text})
        except Exception as e:
            conflict_out = f"❌ ConflictResolver failed: {e}"
        self._log("ConflictResolver output: " + conflict_out)
        yield f"**Conflict Resolver:**\n```\n{conflict_out}\n```\n"

        # Step 4: MDT synthesis
        yield "\n### Step 4 — MDT synthesis\n---\n"
        mdt_input_text = json.dumps({"gp_triage": gp_json, "specialist_reports": specialist_reports, "conflict": conflict_out})
        try:
            mdt_out = self.agents["Multidisciplinary"].run({"input": mdt_input_text})
        except Exception as e:
            mdt_out = f"❌ MDT failed: {e}"
        self._log("MDT output: " + mdt_out)
        yield f"**MDT Final Assessment:**\n```\n{mdt_out}\n```\n"
        yield "\nAnalysis complete. Follow-up questions can be asked."

    def process_followup(self, question: str) -> str:
        input_text = f"Previous conversation:\n{self.conversation_history}\nFollow-up question:\n{question}"
        try:
            out = self.agents["Multidisciplinary"].run({"input": input_text})
        except Exception as e:
            out = f"❌ MDT follow-up failed: {e}"
        self._log("Follow-up -> MDT: " + question)
        self._log("MDT response: " + out)
        return out

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🩺 MedAgent Triage", layout="wide")
st.title("🩺 MedAgent — Multi-Agent Medical Triage")
st.markdown("Workflow: General Physician → Specialists → Conflict Resolver → MDT")

with st.sidebar:
    st.header("Disclaimer")
    st.warning("Educational/demo only. Not for real medical use.")

# Initialize Gemini 2.5 Flash
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3,
    )
except Exception as e:
    st.error("Failed to initialize Gemini 2.5 Flash. Check your API key in Streamlit secrets.")
    st.stop()

# Orchestrator
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MedicalAgentOrchestrator(llm)

# Input
st.header("1) Submit Medical Report")
example = (
    "**Patient:** John Doe, 40M\n"
    "**Symptoms:** Chest pain, palpitations, shortness of breath.\n"
    "**History:** ER visits, cardiac workups normal."
)
medical_report = st.text_area("Paste the medical report:", value=example, height=180)

if st.button("Analyze Report"):
    if not medical_report.strip():
        st.warning("Enter a report first")
    else:
        orchestrator = st.session_state.orchestrator
        gen = orchestrator.process_report(medical_report)
        try:
            while True:
                part = next(gen)
                st.markdown(part)
                time.sleep(0.2)
        except StopIteration:
            pass
        except Exception as e:
            st.error(f"Error during workflow: {e}")
        st.session_state.last_history = orchestrator.conversation_history
        st.success("Analysis complete!")

# Follow-up questions
if "last_history" in st.session_state:
    st.header("2) Ask follow-up questions")
    follow = st.text_input("Ask MDT:")
    if follow:
        orchestrator = st.session_state.orchestrator
        with st.spinner("MDT thinking..."):
            reply = orchestrator.process_followup(follow)
            st.markdown("**MDT Reply:**")
            st.write(reply)
