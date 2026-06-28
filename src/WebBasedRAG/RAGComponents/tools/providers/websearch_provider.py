import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider

from WebBasedRAG.RAGComponents.web_explorer_components.explorer import WebExplorer
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState

class AskDIEMKnowledgeBase(BaseModel):
    """Input schema for the DIEM Web Search tool."""
    natural_language_question: str = Field(
        ..., 
        description=(
            "CRITICAL: The exact question formulated in natural language. "
            "DO NOT extract keywords. DO NOT use boolean logic. "
            "Formulate the input as if you were speaking directly to a human secretary, but keep it short from 3 to 6 words maximum. "
            "So, in order to achive better seach be concise and make the query shorter."
            "INCORRECT EXAMPLE: 'Chi è il primo laureato in Electrical Engineering?' "
            "CORRECT EXAMPLE: 'primo laureato Electrical Engineering DIEM'."
        )
    )

class WebSearchToolProvider(ToolProvider):
    """
    Gestisce l'interfaccia tool per l'esploratore web e registra
    le fonti trovate nel GlobalSessionState condiviso.
    """
    def __init__(self, web_explorer: WebExplorer, session_state: GlobalSessionState, debug_mode: bool = False):
        self.web_explorer = web_explorer
        self.session_state = session_state
        self.debug_mode = debug_mode
        self.last_mapping = {}
        self._doc_counter = 0

    def reset_session(self):
        self.last_mapping = {}
        self._doc_counter = 0

    def get_tools(self) -> List[StructuredTool]:
        def web_search_action(natural_language_question: str) -> str:
            # L'explorer restituisce il contesto e un mapping temporaneo con ID da 1 a 5
            context, raw_mapping = self.web_explorer.explore_and_retrieve(
                search_query=natural_language_question,
                debug=self.debug_mode
            )
            
            if not context:
                return "NESSUN RISULTATO TROVATO"

            # Ricostruiamo i chunk
            import re
            chunks = re.split(r'(\[Doc \d+\]\n)', context)
            
            remapped_context = context
            for old_id in sorted(raw_mapping.keys()):
                doc_data = raw_mapping[old_id]
                
                # 🔴 FIX: Calcoliamo il NUOVO ID reale prima di incollarlo nel testo!
                new_id = self.session_state._doc_counter + 1
                
                doc_text = context # fallback
                for i in range(1, len(chunks)):
                    if chunks[i] == f"[Doc {old_id}]\n":
                        # Sostituiamo il vecchio [Doc vecchi_id] con il [Doc nuovo_id]
                        doc_text = f"[Doc {new_id}]\n" + chunks[i+1]
                        break
                        
                # Salviamo il documento, il DB userà automaticamente new_id
                added_id = self.session_state.add_source(
                    url=doc_data["url"],
                    title=doc_data["titolo"],
                    testo_estratto=doc_text,
                    smart_url=doc_data.get("smart_url"),
                    score=doc_data.get("score", 1.0),
                    external_links=doc_data.get("links", []) 
                )

                # Aggiorniamo la preview che verrà mostrata all'Agente
                remapped_context = remapped_context.replace(f"[Doc {old_id}]", f"[Doc {added_id}]", 1)

            # Restituiamo all'agente i chunk estratti così li può leggere!
            return f"✅ Risultati Web salvati nel DB. Anteprima dati:\n{remapped_context}"

        web_search_tool = StructuredTool.from_function(
            func=web_search_action,
            name="WebSearch",
            description="Search the department's knowledge base. Use natural language.",
            args_schema=AskDIEMKnowledgeBase
        )
        return [web_search_tool]
