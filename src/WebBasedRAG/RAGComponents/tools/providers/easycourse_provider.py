import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider
from WebBasedRAG.RAGComponents.tools.services.easycourse_service import EasyCourseService
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState

# =====================================================================
# SCHEMI INPUT SEMPLIFICATI (Thick Tool / Thin Agent)
# =====================================================================
class TeacherCoursesInput(BaseModel):
    query: str = Field(
        ..., 
        description=(
            "The natural language query from the user containing the professor's name and optionally a date, year, or semester. "
            "Example: 'quali materie insegnava il professore Mario Rossi nel primo semestre 2024?' "
            "RULE: If a past/future semester or year is mentioned, keep it in the query so the internal agent can calculate the exact date."
        )
    )



class StudentScheduleInput(BaseModel):
    query: str = Field(
        ..., 
        description=(
            "The natural language query from the user containing the degree course, the year/curriculum, and the date. "
            "Example: 'orario di ingegneria informatica primo anno per 12-01-2025'."
            "RULE: Do not insert 'oggi', 'domani', etc... But, the precises date in number format."
        )
    )

class TeacherScheduleInput(BaseModel):
    query: str = Field(
        ..., 
        description=(
            "The natural language query from the user containing the professor's name and the date. "
            "Example: 'orario del professore Mario Rossi per il 15-04-2023'."
            "RULE: Do not insert 'oggi', 'domani', etc... But, the precises date in number format."
        )
    )

class EasyCourseToolProvider(ToolProvider):
    def __init__(self, service: EasyCourseService, session_state: GlobalSessionState):
        self.service = service
        self.session_state = session_state

    def get_tools(self) -> List[StructuredTool]:
        def fetch_teacher_courses_action(query: str) -> str:
            # Deleghiamo il lavoro al Service
            result_json = self.service.fetch_teacher_courses(query)
            
            try:
                data = json.loads(result_json)
                if data.get("status") == "SUCCESS":
                    url = data.get("direct_url", "https://easycourse.unisa.it")
                    md_list = data.get("markdown_list", "")
                    
                    doc_id = self.session_state._doc_counter + 1
                    title = f"Insegnamenti Docente - Query: '{query}'"
                    
                    testo_da_salvare = (
                        f"Titolo del Documento: {title}\n"
                        f"Contesto: Lista degli insegnamenti estratti dal sistema EasyCourse per il docente indicato.\n\n"
                        f"{md_list}"
                    )
                    
                    self.session_state.add_source(
                        url=url, 
                        title=title, 
                        testo_estratto=f"[Doc {doc_id}]\n{testo_da_salvare}",
                        smart_url=url, 
                        score=1.0
                    )
                    
                    return f"✅ Insegnamenti del docente recuperati e salvati nel DB come [Doc {doc_id}]. Contenuto:\n{testo_da_salvare}\n"
                else:
                    return f"❌ Fallimento del tool interno: {data.get('message')}"
            except Exception as e: 
                return f"❌ Errore parsing risposta EasyCourse: {str(e)}"

        # Configurazione dello strumento
        teacher_courses_tool = StructuredTool.from_function(
            func=fetch_teacher_courses_action,
            name="FetchTeacherCourses",
            description=(
                "Fetches the list of courses taught by a specific Teacher (Docente). "
                "It supports historical data: pass the user's natural language request directly to this tool, "
                "including any specific year, semester, or date requested by the user."
                "CRITICAL RULE: This tool can return only a semester per call. So if you must retrieve the list of courses taught by a specific Teacher for a whole year, you should call this tool two times: one for the first semester and one for the second semester."
                "If a semester is not explicitly requested, you MUST include the current semester directly in the query string."
                "You must give name and surname of the teacher. If you don't have it, search for it."
            ),
            args_schema=TeacherCoursesInput
        )



        
        def fetch_students_schedule_action(query: str) -> str:
            # Deleghiamo TUTTO il lavoro sporco al Service
            result_json = self.service.fetch_students_schedule(query)
            
            try:
                data = json.loads(result_json)
                if data.get("status") == "SUCCESS":
                    url = data.get("direct_url", "https://easycourse.unisa.it")
                    md_table = data.get("markdown_table", "")
                    
                    doc_id = self.session_state._doc_counter + 1
                    title = f"Orario Studenti - Query: '{query}'"
                    
                    testo_da_salvare = (
                        f"Titolo del Documento: {title}\n"
                        f"Contesto: Questo è l'orario ufficiale estratto dal sistema EasyCourse.\n\n"
                        f"{md_table}"
                    )
                    
                    self.session_state.add_source(
                        url=url, 
                        title=title, 
                        testo_estratto=f"[Doc {doc_id}]\n{testo_da_salvare}", 
                        smart_url=url, 
                        score=1.0
                    )
                    
                    return f"✅ Orario studenti recuperato con successo e salvato come [Doc {doc_id}]. Contenuto: {testo_da_salvare}\n"
                else:
                    return f"❌ Fallimento del tool interno: {data.get('message')}"
            except Exception as e: 
                return f"❌ Errore parsing risposta EasyCourse: {str(e)}"

        
        def fetch_teacher_schedule_action(query: str) -> str:
            # Deleghiamo TUTTO il lavoro sporco al Service
            result_json = self.service.fetch_teacher_schedule(query)
            
            try:
                data = json.loads(result_json)
                if data.get("status") == "SUCCESS":
                    url = data.get("direct_url", "https://easycourse.unisa.it")
                    md_table = data.get("markdown_table", "")
                    
                    doc_id = self.session_state._doc_counter + 1
                    title = f"Orario Docente - Query: '{query}'"
                    
                    testo_da_salvare = (
                        f"Titolo del Documento: {title}\n"
                        f"Contesto: Orario lezioni ed impegni estratti per il docente dal sistema EasyCourse.\n\n"
                        f"{md_table}"
                    )
                    
                    self.session_state.add_source(
                        url=url, 
                        title=title, 
                        testo_estratto=f"[Doc {doc_id}]\n{testo_da_salvare}",
                        smart_url=url, 
                        score=1.0
                    )
                    
                    return f"✅ Orario docente recuperato e salvato nel DB come [Doc {doc_id}]. Contenuto: {testo_da_salvare}\n"
                else:
                    return f"❌ Fallimento del tool interno: {data.get('message')}"
            except Exception as e: 
                return f"❌ Errore parsing risposta EasyCourse: {str(e)}"

        student_tool = StructuredTool.from_function(
            func=fetch_students_schedule_action,
            name="FetchStudentLessonsSchedule",
            description=(
                "Fetches the university class schedule for students based on their degree program. "
                "Pass the user's natural language request directly to this tool. "
                "The tool has an internal AI agent that will navigate the website automatically."
            ),
            args_schema=StudentScheduleInput
        )

        teacher_tool = StructuredTool.from_function(
            func=fetch_teacher_schedule_action,
            name="FetchTeacherLessonsSchedule",
            description=(
                "Fetches the weekly schedule and courses taught by a specific Teacher (Docente). "
                "Pass the user's natural language request directly to this tool. "
                "The tool has an internal AI agent that will navigate the website automatically."
            ),
            args_schema=TeacherScheduleInput
        )
        
        return [student_tool, teacher_tool, teacher_courses_tool]
