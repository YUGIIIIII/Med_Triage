# File: orchestrator.py

import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class MedicalAgentOrchestrator:
    """Manages the workflow of memory-enabled medical agents."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-1.5-flash-latest")
        self.main_conversation_history = ""
        self.agents = self._create_agents()
        self.case_started = False

    def _create_prompt_template(self, role_prompt):
        template = f"""{role_prompt}\n\nCurrent Conversation:\n{{history}}\nHuman: {{input}}\nAI:"""
        return PromptTemplate(input_variables=["history", "input"], template=template)

    def _create_agents(self):
        agents = {}
        prompts = {
            "GeneralPhysician": """
                Act as an expert General Physician. Your task is to triage a patient based on their medical report. First, determine if the issue is a general, non-severe condition (like a common cold or simple flu) that you can handle directly.
                - If it IS a general condition, provide a brief diagnosis and recommendation.
                - If it requires a specialist, determine which specialists are needed from the available list: Cardiologist, Psychologist, Pulmonologist, Neurologist.
                You MUST return a JSON object with two keys: "referral" and "diagnosis".
                - For specialist referral: {{"referral": ["Cardiologist", "Psychologist"], "diagnosis": null}}
                - For direct diagnosis: {{"referral": [], "diagnosis": "The patient appears to have a common cold. Recommend rest and hydration."}}
                Return *only* the JSON object.
            """,
            "Cardiologist": "Act as an expert Cardiologist. Analyze the conversation and medical history, providing a focused analysis on cardiovascular health.",
            "Psychologist": "Act as an expert Psychologist. Analyze the conversation and medical history, providing a psychological assessment.",
            "Pulmonologist": "Act as an expert Pulmonologist. Analyze the conversation and medical history, providing a pulmonary assessment.",
            "Neurologist": "Act as an expert Neurologist. Analyze the conversation and medical history, providing a neurological assessment.",
            "MultidisciplinaryTeam": "Act as a multidisciplinary team. Review the entire conversation history. Synthesize this information to provide a final list of the 3 most likely health issues. When asked a follow-up question, use the full conversation context to provide a helpful and accurate answer."
        }
        for role, prompt_text in prompts.items():
            prompt_template = self._create_prompt_template(prompt_text)
            memory = ConversationBufferWindowMemory(k=15)
            agents[role] = ConversationChain(llm=self.llm, prompt=prompt_template, memory=memory)
        return agents

    def process_initial_report(self, medical_report):
        self.case_started = True
        self.main_conversation_history = f"Initial Medical Report:\n\n{medical_report}\n"
        yield "**Step 1: General Physician Triage**\n---\n"
        gp_chain = self.agents["GeneralPhysician"]
        gp_response_str = gp_chain.predict(input=self.main_conversation_history)
        self.main_conversation_history += f"\n--- General Physician Triage Notes ---\n{gp_response_str}\n"
        yield f"**GP Output:**\n```json\n{gp_response_str}\n```\n"
        try:
            cleaned_gp_response_str = gp_response_str.strip().replace("`", "").replace("json", "")
            gp_decision = json.loads(cleaned_gp_response_str)
        except (json.JSONDecodeError, TypeError) as e:
            yield f"Error decoding GP response: {e}\n"
            return
            
        if gp_decision.get("referral") and len(gp_decision["referral"]) > 0:
            referred_specialists = gp_decision["referral"]
            yield f"\n**Step 2: Specialist Consultations for {referred_specialists}**\n---\n"
            for i, specialist_name in enumerate(referred_specialists):
                yield f"\n*Consulting {specialist_name}...*\n"
                if specialist_name in self.agents:
                    specialist_chain = self.agents[specialist_name]
                    specialist_response = specialist_chain.predict(input=self.main_conversation_history)
                    self.main_conversation_history += f"\n--- {specialist_name} Report ---\n{specialist_response}\n"
                    yield f"**Report from {specialist_name}:**\n\n{specialist_response}\n"
                else:
                    yield f"Warning: Specialist '{specialist_name}' not recognized.\n"
            yield "\n**Step 3: Final Multidisciplinary Team Assessment**\n---\n"
            team_chain = self.agents["MultidisciplinaryTeam"]
            final_diagnosis = team_chain.predict(input=self.main_conversation_history)
            self.main_conversation_history += f"\n--- Final Assessment ---\n{final_diagnosis}\n"
            yield f"**Final Assessment:**\n\n{final_diagnosis}\n"
        else:
            yield "\n**Final Diagnosis from General Physician**\n---\n"
            yield f"{gp_decision.get('diagnosis', 'No diagnosis provided.')}\n"
        
        yield "\n---\n*Initial analysis complete. You can now ask follow-up questions.*"

    def process_follow_up(self, question):
        if not self.case_started:
            return "Please start by providing the initial medical report."
        self.main_conversation_history += f"\n--- Follow-up Question ---\n{question}\n"
        team_chain = self.agents["MultidisciplinaryTeam"]
        response = team_chain.predict(input=self.main_conversation_history)
        self.main_conversation_history += f"\n--- Follow-up Answer ---\n{response}\n"
        return response