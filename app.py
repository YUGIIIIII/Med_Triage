import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------------------------------------
# 1️⃣ PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="MedTriage AI",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 MedTriage – AI Medical Triage Assistant")
st.markdown(
    """
    MedTriage helps analyze patient symptoms using multiple medical specialists powered by **Gemini 2.5 Flash**.
    Enter the patient's symptoms to get detailed assessments from each specialist.
    """
)

# -----------------------------------------------------------
# 2️⃣ INITIALIZE LLM USING STREAMLIT SECRETS
# -----------------------------------------------------------
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ Using Gemini 2.5 Flash
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.5,
    )
except Exception as e:
    st.error("❌ Could not initialize Gemini 2.5 Flash. Please check Streamlit secrets.")
    st.stop()

# -----------------------------------------------------------
# 3️⃣ MEDICAL AGENT ORCHESTRATOR CLASS
# -----------------------------------------------------------
class MedicalAgentOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.agents = self._create_agents()

    def _create_agents(self):
        agents = {}

        agent_roles = {
            "Cardiologist": "Analyzes cardiac-related symptoms such as chest pain, palpitations, or shortness of breath.",
            "Psychologist": "Analyzes psychological symptoms including anxiety, depression, stress, or behavioral changes.",
            "Pulmonologist": "Focuses on respiratory symptoms such as cough, wheezing, asthma, or lung disorders.",
            "Neurologist": "Analyzes neurological symptoms like dizziness, headaches, numbness, or weakness.",
        }

        for role, desc in agent_roles.items():
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            prompt_template = PromptTemplate.from_template(
                f"""
                You are a {role}. {desc}
                Respond in a clear, empathetic, and medically appropriate tone.

                Patient information:
                {{input}}

                Provide:
                1. A possible cause or diagnosis (based on symptoms)
                2. Recommended next steps or tests
                3. A confidence level (Low / Medium / High)
                """
            )

            agents[role] = ConversationChain(llm=self.llm, prompt=prompt_template, memory=memory)

        return agents

    def diagnose(self, patient_input):
        responses = {}
        for role, agent in self.agents.items():
            try:
                response = agent.run(input=patient_input)
                responses[role] = response
            except Exception as e:
                responses[role] = f"⚠️ Error: {e}"
        return responses

# -----------------------------------------------------------
# 4️⃣ INITIALIZE SESSION STATE
# -----------------------------------------------------------
if "orchestrator" not in st.session_state:
    with st.spinner("Initializing specialist agents..."):
        st.session_state.orchestrator = MedicalAgentOrchestrator(llm=llm)

# -----------------------------------------------------------
# 5️⃣ SIDEBAR CONFIGURATION
# -----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Gemini 2.5 Flash is connected automatically using Streamlit Secrets.")
    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.warning(
        "This tool is for **educational and informational purposes only**. "
        "It should not replace professional medical advice or diagnosis."
    )

# -----------------------------------------------------------
# 6️⃣ MAIN USER INPUT
# -----------------------------------------------------------
st.markdown("### 🧾 Enter Patient Symptoms")
patient_input = st.text_area(
    "Describe the patient's symptoms, duration, and relevant history:",
    placeholder=(
        "Example: 35-year-old female experiencing chest tightness, shortness of breath, "
        "and mild anxiety for the past 2 days."
    ),
    height=150,
)

# -----------------------------------------------------------
# 7️⃣ RUN DIAGNOSIS
# -----------------------------------------------------------
if st.button("🧩 Analyze Symptoms"):
    if not patient_input.strip():
        st.warning("Please enter patient symptoms before proceeding.")
    else:
        orchestrator = st.session_state.orchestrator
        with st.spinner("Analyzing symptoms with specialist agents..."):
            results = orchestrator.diagnose(patient_input)

        st.success("✅ Analysis complete! See the results below:")

        for specialist, result in results.items():
            with st.expander(f"🔹 {specialist}"):
                st.write(result)

# -----------------------------------------------------------
# 8️⃣ FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("© 2025 MedTriage AI | Powered by Gemini 2.5 Flash")
