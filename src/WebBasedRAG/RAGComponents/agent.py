import re
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models.chat_models import BaseChatModel

import os
if os.getenv("IS_LANGFUSE_ENABLED"):
    from analytics.langfuse_dashboard import init_langfuse
    langfuse_handler = init_langfuse()

class AnswerWithSources(BaseModel):
    answer: str = Field(description="La risposta discorsiva, chiara e completa.")
    sources_used: List[int] = Field(description="Lista degli ID numerici dei documenti usati. Vuoto [] se non c'è risposta.")


class SupervisorAgent:
    """L'unico scopo di questo Agente è eseguire i tool e popolare il GlobalSessionState."""
    def __init__(self, base_llm: BaseChatModel, tools: list, session_state, max_retries: int = 3):
        self.base_llm = base_llm
        self.tools = tools
        self.session_state = session_state 

        now = datetime.now()
        days_in_english = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = days_in_english[now.weekday()]
        
        _datetime = (
            f"- Day: {day_of_week}\n"
            f"- Date (YYYY-MM-DD): {now.strftime('%Y-%m-%d')}\n"
            f"- Time (HH:MM:SS): {now.strftime('%H:%M:%S')}\n"
            f"- Current Year: {now.year}\n"
        )
        # 1. Definiamo il prompt per l'Agente
        system_prompt = f"""<role>
You are the official AI Assistant for the DIEM (Department of Information, Electrical, and Applied Mathematics Engineering) at UNISA (University of Salerno).
Your primary goal is to retrieve information and assist users strictly regarding this department.
</role>

<temporal_context>
Do NOT use your internal training date. We are currently in this datetime: 
{_datetime}
</temporal_context>

<constraints>
- SCOPE: You are authorized to provide information ONLY about the DIEM department, its faculties, professors, and rules. 
- OUT OF SCOPE: Other engineering departments or unrelated topics are out of scope. If asked, politely decline and clarify your role.
- TRUTH: Rely entirely on the information retrieved via your tools. Do not hallucinate data.
</constraints>

<tool_instructions>
- You are a BLIND EXECUTOR. You have absolutely no strategic freedom.
- You MUST follow the Execution Plan step-by-step, exactly as written. Do not skip steps, do not reorder them, do not invent new tools to use and do not stop if you didn't execute all the given steps.
- PARAMETER EXTRACTION (YOUR ONLY JOB): The Planner will tell you to use a tool. You must look at the 'AVAILABLE KNOWLEDGE IN MEMORY' to find the exact numbers, rules, or names required by that tool's schema. 
- NEVER hallucinate or guess parameter values. If the memory contains the rule, copy it strictly.
- MULTI-TURN TOOLS: Always start with empty fields, wait for choices, then select based on the user's query.
</tool_instructions>
"""

        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            # Questo placeholder è obbligatorio: qui LangChain inserirà in automatico 
            # la "memoria" temporanea dei tool chiamati durante i vari tentativi
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 2. Creiamo l'Agente (colui che pensa e decide i tool)
        agent = create_tool_calling_agent(self.base_llm, self.tools, agent_prompt)

        # 3. Creiamo l'Executor (il motore che esegue i cicli for al posto tuo)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            max_iterations=max_retries, # Sostituisce il tuo range(max_retries)
            return_intermediate_steps=True, # FONDAMENTALE: ci restituisce l'output dei tool!
            verbose=True, # Stampa i log delle azioni e dei tool nel terminale
            handle_parsing_errors = True
        )


    def run(self, user_query: str, context: str = "", plan: str = "", previous_feedback: str = "") -> tuple[str, bool]:
        """Esegue i tool. Restituisce (testo_del_contesto, True/False se ha trovato qualcosa)."""
        print(" 🤖 [AGENTE] Avvio esecuzione del piano...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
        # Inseriamo il contesto nella memoria dell'Agente!
        base_input = f"AVAILABLE KNOWLEDGE IN MEMORY:\n{context}\n\nUser Query: {user_query}\n\n"
        if previous_feedback:
            agent_input = (
                f"{base_input}"
                f"ATTENTION: Your previous context was rejected by the auditor.\n"
                f"Reason for rejection: '{previous_feedback}'.\n\n"
                f"Original User Query: {user_query}\n\n"
                f"Revised Execution Plan (Follow this step-by-step to fix the issue):\n{plan}"
            )
        else:
            agent_input = (
                f"{base_input}"
                f"User Query: {user_query}\n\n"
                f"Execution Plan (Execute these steps sequentially using your tools):\n{plan}"
            )

        # FASE 1: L'agente esegue i tool (che riempiono autonomamente il DB)
        self.agent_executor.invoke({"input": agent_input}, config={"callbacks": callbacks})
        
        # FASE 2: Estrazione del contesto grezzo dal DB
        final_context = self.session_state.get_all_context()
        
        if not final_context.strip():
            return "", False

        return final_context, True