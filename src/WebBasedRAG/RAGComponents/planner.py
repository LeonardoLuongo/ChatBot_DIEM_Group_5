# ================================================
# FILE: RAGComponents/planner.py
# ================================================
import os
from datetime import datetime
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description_and_args
from langchain_core.language_models.chat_models import BaseChatModel

if os.getenv("IS_LANGFUSE_ENABLED"):
    from analytics.langfuse_dashboard import init_langfuse
    langfuse_handler = init_langfuse()

from typing import List
from pydantic import BaseModel, Field

class TaskStep(BaseModel):
    step_number: int = Field(
        description="The sequential step number (e.g., 1, 2, 3)."
    )
    tool_name: str = Field(
        description="The exact name of the tool to use for this step, copied strictly from the provided list of AVAILABLE TOOLS."
    )
    tool_intent: str = Field(
        description="Directive for the execution agent: explicitly state EXACTLY which schema arguments must be populated (and where the data comes from), and which ones MUST strictly be left empty on the first attempt."
    )

class ExecutionPlan(BaseModel):
    reasoning: str = Field(
        description="A brief strategic explanation of the plan: ensure and verify that all mandatory parameters required by the chosen tools' schemas will be available to the agent at the time of execution."
    )
    steps: List[TaskStep] = Field(
        description="The sequential list of atomic tasks to be executed."
    )
    
class QueryPlanner:
    def __init__(self, llm: BaseChatModel, tools: list):
        self.llm = llm
        self.structured_planner = self.llm.with_structured_output(ExecutionPlan)
        
        # Estrazione dinamica degli Schemi Pydantic dei tool
        self.tools_schema = render_text_description_and_args(tools)
        
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Lead Task Planner Agent for the DIEM department (UNISA).
Your exact goal is to decompose a user's complex query into a strict Execution Plan.

Do NOT use your internal training date. We are currently in this datetime: {current_datetime}

CURRENT GLOBAL ITERATION: {current_iter} of {max_iter}

To plan correctly, you MUST read the dynamically generated SCHEMAS of the available tools below. 
You cannot plan a step if the agent won't have the required input arguments. Pay extreme attention to parameters that explicitly ask to be "left empty on the first attempt".

AVAILABLE TOOLS AND THEIR EXACT PYDANTIC SCHEMAS:
{tools_schema}

     
AVAILABLE KNOWLEDGE IN MEMORY:
{context}

CRITICAL ANTI-HALLUCINATION RULE (THE "FETCH FIRST" DIRECTIVE):
Look at the AVAILABLE KNOWLEDGE IN MEMORY. If the user asks to apply a rule, calculate a score, or process data, and the exact parameters/rules needed by your Tools are NOT present in the memory, your VERY FIRST STEP MUST be to use `SearchInternalDatabase` to find them. 
NEVER assume you know the department's rules by default.


CRITICAL TOOL ROUTING RULES (Memorize the difference between these tools):
1. STUDENT TIMETABLES: If the user asks for STUDENT class timetables, lessons, or schedules based on a degree program, you MUST plan a step using `FetchStudentLessonsSchedule`. Do NOT use WebSearch.
2. TEACHER SCHEDULES (ORARI LEZIONI): If the user asks for a specific TEACHER'S weekly schedule (orario delle lezioni in aula/calendario), you MUST plan a step using `FetchTeacherLessonsSchedule`.
3. TEACHER COURSES (MATERIE INSEGNATE E STORICO): If the user asks for the LIST OF COURSES a teacher teaches or taught in the past (e.g., "quali materie insegna", "i corsi del docente nel 2024", "cosa insegnava nel primo semestre"), you MUST plan a step using `FetchTeacherCourses`. This tool supports historical queries and semester filtering automatically.
4. OFFICE HOURS & CONTACTS (RICEVIMENTO E CONTATTI): If the user asks for a professor's OFFICE HOURS / RECEPTION HOURS, Contact Info, Office Location, or Notice Board, you MUST plan a step using `FetchProfessorProfile`.
5. INCOMPLETE NAMES: If you have only a surname or an incomplete name, plan `SearchProfessorDirectory` FIRST, then `FetchProfessorProfile`.
6. INTERNAL KNOWLEDGE BASE (PRIMARY SEARCH): If the user asks for general information, rules, CFU requirements, thesis regulations, historical facts, or general knowledge, you MUST plan a step using `SearchInternalDatabase` FIRST if no tool cover the topic. 
7. WEB SEARCH (FALLBACK & LIVE DATA): Use `WebSearch` as a Fallback if `SearchInternalDatabase` fails to find an answer or simply as a way to get with higher probability the correct information.
8. MATHEMATICAL CALCULATIONS: Use the `Calculator` tool for ANY mathematical operation.
9. CALLS FOR APPLICATIONS: If the user asks for `calls for applications` or `benefits` (e.g., `Bandi` in Italian), you must:
- Instruct the executor that using both `WebSearch` and `SearchInternalDatabase` is mandatory.
- Identify any specific details or topics in the user's request, whether explicitly stated or fully deducible.
- Include these specific terms in your search queries to ensure highly relevant results.
10. GRADUATION GRADE: If the user wants to know the graduation grade, use `calcola_voto_laurea`.

11. **COURSE SYLLABUS / INFO RETRIEVAL:**
    THIS COMMAND IS VALID ONLY TEACHING, NO EXAMS, SESSION OF EXAMS, DATE OF EXAMS, ETC... TEACHING ONLY. IF THE USER IS ASKING FOR EXAMS, IGNORE THIS, OTHERWISE, USE IT:
    If the user asks for the syllabus, reading materials, or general information about a specific course, you MUST follow this exact two-step procedural workflow:
    1. **Identify and Verify the Teacher:**
    * **Primary Path:** Call `SearchInternalDatabase` to retrieve the `[TEACHER_NAME]` and `[TEACHER_SURNAME]` for the requested course (`[EXAM_NAME]`).
    * **Fallback Path (if NOT found in database):** If the internal database returns no results, use `WebSearch` to search for who teaches the course. For every teacher candidate found via the web, you MUST subsequently call `SearchProfessorDirectory` to verify and confirm that they officially belong to the DIEM department.
    2. **Perform the Course Material Web Search:**
    Once you have the confirmed DIEM teacher details from Step 1, instruct the executor to call `WebSearch` to find the syllabus/info. You MUST format the search query string exactly as follows:
    `"[MATERIAL YOU NEED] [EXAM_NAME] DIEM UNISA [ACADEMIC_YEAR] [TEACHER_NAME] [TEACHER_SURNAME]"`
    *Example query:* `"libri Data Science DIEM UNISA 2025/2026 Vincenzo Matta"`
    
12. EXAM DATES, SESSION OF EXAMS, EVERYTHING EXAM RELATED: Ignore AND DO NOT FOLLOW FOR ANY REASON THE INSTRUCTION ELEVEN (11).
     
PLANNING RULES & CONSTRAINTS:
- ATOMICITY: Each step must assign EXACTLY ONE 'tool_name' from the available tools.
- DATA FLOW: Explicitly state in the 'tool_intent' how the agent should use the output of Step N to populate the schema of Step N+1.
- PREVIOUS FEEDBACK: If PREVIOUS AUDITOR FEEDBACK is provided, adapt the new plan to explicitly resolve the failure.
- STRICT LINEARITY (NO CONDITIONALS): Your plan must be a straight sequence of actions assuming total success. You are strictly forbidden from writing conditional steps, branching logic, or predicting fallbacks (e.g., "If Step X fails, then Step Y").
- DELEGATE EVALUATION: Do not worry about failures in the current plan. An external Judge/Auditor evaluates the output of your execution. If your plan fails, the Auditor will trigger a new GLOBAL ITERATION, and you will handle the fallback then, not now.

CRITICAL OUTPUT RULE: 
You MUST output ONLY a valid JSON object matching the schema. DO NOT print any text, thought process, or markdown formatting outside of the JSON block. Put your reasoning STRICTLY INSIDE the "reasoning" JSON string field."""),
    ("human", """USER QUESTION: {query}

PREVIOUS AUDITOR FEEDBACK (if any, use it to improve the plan):
{feedback}""")
])
        
        self.chain = self.prompt | self.structured_planner

    def generate_plan(self, query: str, context: str = "", previous_feedback: str = "", current_iter: int = 1, max_iter: int = 3) -> ExecutionPlan:
        current_dt = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

        print(f"   🗓️ [PLANNER] Analisi della query (Iterazione {current_iter}/{max_iter})...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
        
        try:
            return self.chain.invoke(
                {
                    "current_datetime": current_dt,
                    "query": query, 
                    "context": context, # <--- INIETTA IL CONTESTO QUI
                    "feedback": previous_feedback, 
                    "tools_schema": self.tools_schema,
                    "current_iter": current_iter,  # <-- Inietto l'iterazione corrente
                    "max_iter": max_iter           # <-- Inietto il limite massimo
                }, 
                config={"callbacks": callbacks}
            )
        except Exception as e:
            print(f"\n   ⚠️ [PLANNER WARN] Errore JSON. Uso piano di fallback generico.")
            # Fallback: se fallisce, restituiamo un piano strutturato di base
            return ExecutionPlan(
                reasoning="Fallback strategy due to parsing error.",
                steps=[TaskStep(step_number=1, tool_name="WebSearch", tool_intent=f"Search information for: {query}")]
            )
