"""
textbook_finder_provider.py
==============================
Tool Provider per il recupero dei testi consigliati (bibliografia) di un
insegnamento UniSA/DIEM, a partire dal nome dell'esame.

Integra TextbookFinderService nel ToolRegistry dell'AgenticRAG.
Gestisce il dialogo multi-turno per raccogliere:
  1. Il nome dell'insegnamento
  2. (Solo se non risolvibile automaticamente) nome e cognome del docente

Segue esattamente il pattern architetturale di GraduationGradeToolProvider.
"""

import json
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from WebBasedRAG.RAGComponents.tools.core.registry import ToolProvider
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.tools.services.textbook_finder_service import (
    TextbookFinderService,
)


# =========================================================================
# SCHEMA PYDANTIC PER L'INVOCAZIONE DEL TOOL
# =========================================================================

class TextbookFinderInput(BaseModel):
    """
    Input schema per il tool di ricerca testi consigliati.
    L'agente LLM popola questo schema estraendo le informazioni dal messaggio
    utente. Se il docente non è presente nel messaggio, lasciare i campi
    vuoti: il tool proverà a risolverlo da solo e, se non ci riesce, lo
    chiederà all'utente.
    """

    exam_name: str = Field(
        default="",
        description=(
            "Nome dell'insegnamento/esame di cui recuperare i testi consigliati. "
            "Esempi: 'Artificial Vision', 'Ingegneria del Software', 'Basi di Dati'. "
            "Lasciare '' se non specificato."
        )
    )
    professor_nome: str = Field(
        default="",
        description=(
            "Nome di battesimo del docente titolare dell'insegnamento, SOLO se "
            "esplicitamente indicato dall'utente (es. 'Mario' in 'Mario Vento'). "
            "Lasciare '' se non fornito: il tool proverà a risolverlo automaticamente."
        )
    )
    professor_cognome: str = Field(
        default="",
        description=(
            "Cognome del docente titolare dell'insegnamento, SOLO se esplicitamente "
            "indicato dall'utente (es. 'Vento' in 'Mario Vento'). "
            "Lasciare '' se non fornito: il tool proverà a risolverlo automaticamente."
        )
    )
    academic_year: str = Field(
        default="",
        description="L'anno accademico di riferimento (es. '2025/2026'). Se non specificato, il tool cercherà nell'anno corrente."
    )


# =========================================================================
# PROVIDER
# =========================================================================

class TextbookFinderToolProvider(ToolProvider):
    """
    Provider che espone il tool 'TextbookFinder' all'agente.
    Risolve automaticamente il docente titolare dell'esame quando possibile
    (mappa locale esame->docente); se non ci riesce, chiede all'utente nome
    e cognome del docente e riprova.
    """

    # Chiavi usate nel session_state per mantenere i dati tra i turni
    _SK_EXAM        = "textbook_exam_name"
    _SK_PROF_NOME   = "textbook_prof_nome"
    _SK_PROF_COGN   = "textbook_prof_cognome"

    def __init__(self, session_state: GlobalSessionState, headless: bool = True):
        self.service = TextbookFinderService(headless=headless)
        self.session_state = session_state

    # ------------------------------------------------------------------
    # INTERFACCIA BaseToolProvider
    # ------------------------------------------------------------------

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool(
                name="TextbookFinder",
                description=(
                    "Retrieve the official recommended textbooks / bibliography for a "
                    "university course, by scraping the professor's teaching page on "
                    "docenti.unisa.it. "
                    "USE THIS TOOL when the user asks things like: 'what are the textbooks "
                    "for Artificial Vision?', 'tell me the recommended books for exam X', "
                    "'what should I study for the [course name] exam?', or similar questions "
                    "about official course bibliography/reading material. "
                    "If the professor cannot be resolved automatically from the course name, "
                    "the tool will ask the user for the professor's name and surname. "
                    "CRITICAL RULE: DO NOT USE THIS TOOL for questions about course content, "
                    "syllabus objectives, exam modality, or CFU — use it ONLY for requests "
                    "about textbooks/bibliography/reading material."
                ),
                args_schema=TextbookFinderInput,
                func=self._run,
            )
        ]

    def _return_and_save(self, data_dict: dict) -> str:
        """Converte il dizionario in JSON e lo inietta nel Session State come documento."""
        json_str = json.dumps(data_dict, ensure_ascii=False, default=str)
        self.session_state.add_source(
            url=data_dict.get("url_scheda", "tool://TextbookFinder"),
            title=f"Testi consigliati — {data_dict.get('esame', '')}",
            testo_estratto=json_str
        )
        return json_str

    # ------------------------------------------------------------------
    # LOGICA PRINCIPALE DEL TOOL
    # ------------------------------------------------------------------

    def _run(
        self,
        exam_name: str = "",
        professor_nome: str = "",
        professor_cognome: str = "",
        academic_year: str = "",
    ) -> str:
        """Funzione principale del tool. Raccoglie i dati mancanti, risolve
        il docente e recupera i testi consigliati."""

        ss = self.session_state
        stored_exam      = ss.get(_ctx(self._SK_EXAM))
        stored_prof_nome = ss.get(_ctx(self._SK_PROF_NOME))
        stored_prof_cogn = ss.get(_ctx(self._SK_PROF_COGN))

        # --- Aggiorna stato con i nuovi valori forniti ---
        exam = stored_exam
        if exam_name.strip():
            exam = exam_name.strip()
            ss.set(_ctx(self._SK_EXAM), exam)

        prof_nome = stored_prof_nome
        if professor_nome.strip():
            prof_nome = professor_nome.strip()
            ss.set(_ctx(self._SK_PROF_NOME), prof_nome)

        prof_cogn = stored_prof_cogn
        if professor_cognome.strip():
            prof_cogn = professor_cognome.strip()
            ss.set(_ctx(self._SK_PROF_COGN), prof_cogn)

        # --- 1. Nome esame obbligatorio ---
        if not exam:
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": "Per quale insegnamento vuoi conoscere i testi consigliati?",
            })

        # --- 2. Chiama il service. Se non riesce a risolvere il docente,
        #        il service stesso restituisce NEED_INFO con la domanda
        #        giusta, che propaghiamo così com'è. ---
        result = self.service.get_textbooks(
            exam_name=exam,
            professor_nome=prof_nome or None,
            professor_cognome=prof_cogn or None,
            academic_year=academic_year or None
        )

        # --- 3. Se il service chiede il docente, salviamo lo stato per
        #        recuperarlo al turno successivo e restituiamo la domanda. ---
        if result.get("status") == "NEED_INFO":
            return self._return_and_save(result)

        # --- 4. Calcolo/scraping completato (successo o errore): pulizia stato ---
        self._clear_session()
        return self._return_and_save(result)

    # ------------------------------------------------------------------
    # HELPERS INTERNI
    # ------------------------------------------------------------------

    def _clear_session(self):
        """Rimuove le chiavi temporanee dal session state dopo il recupero."""
        keys = [self._SK_EXAM, self._SK_PROF_NOME, self._SK_PROF_COGN]
        for k in keys:
            self.session_state.remove(_ctx(k))


# =========================================================================
# UTILITY FUNCTIONS (modulo-level, non dipendenti dalla classe)
# =========================================================================

def _ctx(key: str) -> str:
    """Aggiunge un namespace alle chiavi del session state per evitare conflitti."""
    return f"textbook_tool::{key}"
