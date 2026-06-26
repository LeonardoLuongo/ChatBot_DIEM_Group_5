"""
graduation_grade_provider.py
==============================
Tool Provider per il calcolo del voto di laurea dell'Università degli Studi di Salerno.

Integra GraduationGradeService nel ToolRegistry dell'AgenticRAG.
Gestisce il dialogo multi-turno per raccogliere:
  1. Il corso di laurea dell'utente
  2. La coorte di appartenenza
  3. I dati numerici (media, punteggio tesi, bonus carriera)

Segue esattamente il pattern architetturale di EasyCourseToolProvider.
"""

import json
import re
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from WebBasedRAG.RAGComponents.tools.core.registry import ToolProvider
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.tools.services.graduation_grade_service import (
    GraduationGradeService,
    DISPLAY_NAMES,
    TRIENNALE_COURSES,
    MAGISTRALE_COURSES,
    COURSE_L8_INFO, COURSE_L8_MED,
    COURSE_LM32_INFO, COURSE_LM32_MED, COURSE_LM28_ENERGY,
)


# =========================================================================
# SCHEMA PYDANTIC PER L'INVOCAZIONE DEL TOOL
# =========================================================================

class GraduationGradeInput(BaseModel):
    """
    Input schema per il tool di calcolo del voto di laurea.
    L'agente LLM popola questo schema estraendo le informazioni dal messaggio utente.
    Se una delle informazioni non è presente nel messaggio, lasciare il campo al
    valore di default ('') e il tool chiederà all'utente di specificarla.
    """

    # ── Identificazione corso e coorte ──────────────────────────────────────
    course_name: str = Field(
        default="",
        description=(
            "Nome o sigla del corso di laurea. Esempi accettati: "
            "'Ingegneria Informatica L-8', 'Ingegneria Informatica magistrale', "
            "'Information Engineering for Digital Medicine', "
            "'Electrical Engineering for Digital Energy', 'LM-28', 'LM-32', 'L-8'. "
            "Lasciare '' se non specificato."
        )
    )
    cohort_year: str = Field(
        default="",
        description=(
            "Coorte di immatricolazione dell'utente. "
            "Usare '25/26' per la coorte 2025/2026, "
            "'24/25' (o qualsiasi valore precedente come '23/24', '22/23') per tutte le coorti più vecchie. "
            "Lasciare '' se non specificato."
        )
    )

    # ── Dati numerici ────────────────────────────────────────────────────────
    media_30: float = Field(
        default=-1.0,
        description=(
            "Media ponderata degli esami in trentesimi (es. 27.5). "
            "Usare -1.0 se non fornita dall'utente."
        )
    )
    pt: float = Field(
        default=-1.0,
        description=(
            "Punteggio tesi proposto dal relatore e assegnato dalla Commissione. "
            "Intervallo [0-2] per corsi triennali, [0-4] per corsi magistrali. "
            "Usare -1.0 se non fornito."
        )
    )

    # ── Bonus carriera ───────────────────────────────────────────────────────
    laurea_in_corso: Optional[bool] = Field(
        default=None,
        description=(
            "True se lo studente si laurea senza uscire fuori corso entro il limite "
            "temporale previsto dalla propria coorte. None se non specificato."
        )
    )
    cfu_estero: float = Field(
        default=0.0,
        description="Numero di CFU svolti all'estero. Default 0."
    )
    programma_internazionale: Optional[bool] = Field(
        default=None,
        description=(
            "True se lo studente partecipa a un programma di doppio titolo internazionale "
            "(rilevante solo per magistrali). None se non specificato."
        )
    )
    percorso_eccellenza: Optional[bool] = Field(
        default=None,
        description="True se lo studente ha completato il percorso di eccellenza. None se non specificato."
    )


# =========================================================================
# PROVIDER
# =========================================================================

class GraduationGradeToolProvider(ToolProvider):
    """
    Provider che espone il tool 'calcola_voto_laurea' all'agente.
    Gestisce la raccolta progressiva dei dati necessari al calcolo
    attraverso domande mirate all'utente.
    """

    # Chiavi usate nel session_state per mantenere i dati tra i turni
    _SK_COURSE   = "grad_grade_course_id"
    _SK_COHORT   = "grad_grade_cohort"
    _SK_MEDIA    = "grad_grade_media"
    _SK_PT       = "grad_grade_pt"
    _SK_IN_CORSO = "grad_grade_laurea_in_corso"
    _SK_CFU      = "grad_grade_cfu_estero"
    _SK_INTL     = "grad_grade_programma_internazionale"
    _SK_EXCEL    = "grad_grade_percorso_eccellenza"

    def __init__(self, session_state: GlobalSessionState):
        self.service = GraduationGradeService()
        self.session_state = session_state

    # ------------------------------------------------------------------
    # INTERFACCIA BaseToolProvider
    # ------------------------------------------------------------------

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool(
                name="calcola_voto_laurea",
                description=(
                    "Calcola il voto di laurea stimato di uno studente dell'Università degli Studi di Salerno (DIEM) "
                    "applicando la formula ufficiale corretta in base al corso di laurea e alla coorte di immatricolazione. "
                    "Supporta: Ingegneria Informatica (L-8 e LM-32), Ingegneria dell'Informazione per la Medicina Digitale (L-8 e LM-32), "
                    "Information Engineering for Digital Medicine (LM-32), Electrical Engineering for Digital Energy (LM-28). "
                    "USARE QUESTO TOOL quando l'utente chiede: 'che voto prendo?', 'quanto sarà il mio voto di laurea?', "
                    "'ho questa media, che voto posso prendere alla laurea?', o domande simili sul voto finale. "
                    "Se il corso o la coorte non sono specificati, il tool chiederà all'utente le informazioni mancanti."
                ),
                args_schema=GraduationGradeInput,
                func=self._run,
            )
        ]
    
    def _return_and_save(self, data_dict: dict) -> str:
        """Converte il dizionario in JSON e lo inietta nel Session State come documento."""
        json_str = json.dumps(data_dict, ensure_ascii=False, default=str)
        self.session_state.add_source(
            url="tool://calcolo_voto_laurea",
            title="Richiesta Dati Tool",
            testo_estratto=json_str
        )
        return json_str

    # ------------------------------------------------------------------
    # LOGICA PRINCIPALE DEL TOOL
    # ------------------------------------------------------------------

    def _run(
        self,
        course_name: str = "",
        cohort_year: str = "",
        media_30: float = -1.0,
        pt: float = -1.0,
        laurea_in_corso: Optional[bool] = None,
        cfu_estero: float = 0.0,
        programma_internazionale: Optional[bool] = None,
        percorso_eccellenza: Optional[bool] = None,
    ) -> str:
        """Funzione principale del tool. Raccoglie i dati, li valida e calcola il voto."""

        # 1. RECUPERA STATO SESSIONE (dati da turni precedenti)
        ss = self.session_state
        stored_course    = ss.get(_ctx(self._SK_COURSE))
        stored_cohort    = ss.get(_ctx(self._SK_COHORT))
        stored_media     = ss.get(_ctx(self._SK_MEDIA))
        stored_pt        = ss.get(_ctx(self._SK_PT))
        stored_in_corso  = ss.get(_ctx(self._SK_IN_CORSO))
        stored_cfu       = ss.get(_ctx(self._SK_CFU))
        stored_intl      = ss.get(_ctx(self._SK_INTL))
        stored_excel     = ss.get(_ctx(self._SK_EXCEL))

        # 2. AGGIORNA STATO CON I NUOVI VALORI FORNITI
        # --- Corso ---
        course_id = stored_course
        if course_name.strip():
            resolved = GraduationGradeService.resolve_course_id(course_name)
            if resolved:
                course_id = resolved
                ss.set(_ctx(self._SK_COURSE), course_id)

        # --- Coorte ---
        cohort = stored_cohort
        if cohort_year.strip():
            cohort = _normalize_cohort(cohort_year)
            ss.set(_ctx(self._SK_COHORT), cohort)

        # --- Media ---
        media = stored_media
        if media_30 >= 0:
            media = media_30
            ss.set(_ctx(self._SK_MEDIA), media)

        # --- PT ---
        pt_val = stored_pt
        if pt >= 0:
            pt_val = pt
            ss.set(_ctx(self._SK_PT), pt_val)

        # --- Bonus opzionali ---
        in_corso = stored_in_corso
        if laurea_in_corso is not None:
            in_corso = laurea_in_corso
            ss.set(_ctx(self._SK_IN_CORSO), in_corso)

        cfu = stored_cfu if stored_cfu is not None else cfu_estero
        if cfu_estero > 0:
            cfu = cfu_estero
            ss.set(_ctx(self._SK_CFU), cfu)

        intl = stored_intl
        if programma_internazionale is not None:
            intl = programma_internazionale
            ss.set(_ctx(self._SK_INTL), intl)

        excel = stored_excel
        if percorso_eccellenza is not None:
            excel = percorso_eccellenza
            ss.set(_ctx(self._SK_EXCEL), excel)

        # 3. RACCOLTA PROGRESSIVA — chiede le informazioni mancanti

        # 3a. Corso di laurea
        if not course_id:
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    "Per calcolare il tuo voto di laurea, ho bisogno di sapere il tuo **corso di laurea**.\n\n"
                    + GraduationGradeService.get_course_menu()
                    + "\n\nA quale corso fai riferimento?"
                ),
            })

        # 3b. Coorte
        if not cohort:
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    f"Ottimo! Hai selezionato: **{DISPLAY_NAMES.get(course_id, course_id)}**.\n\n"
                    "A quale **coorte di immatricolazione** appartieni?\n"
                    "  • `25/26` — immatricolato nell'a.a. 2025/2026\n"
                    "  • `24/25 o precedenti` — immatricolato nell'a.a. 2024/2025 o prima\n\n"
                    "Rispondi con la tua coorte (es. *25/26* oppure *24/25*)."
                ),
            })

        # 3c. Media in trentesimi
        if media is None or media < 0:
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    "Perfetto! Ora dimmi la tua **media ponderata degli esami di profitto in trentesimi** "
                    "(escluse le idoneità).\n"
                    "Esempio: `27.5`"
                ),
            })

        # 3d. Punteggio tesi
        if pt_val is None or pt_val < 0:
            is_triennale = course_id in TRIENNALE_COURSES
            pt_range = "[0-2]" if is_triennale else "[0-4]"
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    f"Qual è il **punteggio tesi (PT)** che ti aspetti di ricevere dalla Commissione? "
                    f"Per il tuo corso l'intervallo è **{pt_range}**.\n"
                    "Puoi indicare anche un valore ipotetico (es. `1.5`)."
                ),
            })

        # 4. RACCOLTA BONUS CARRIERA (opzionale ma utile per un calcolo preciso)
        # Chiediamo solo se non già memorizzati

        if in_corso is None:
            corso_label = _get_in_corso_label(course_id, cohort)
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    f"Ti laurei **senza essere uscito fuori corso** (cioè {corso_label})?\n"
                    "Rispondi con **sì** o **no**."
                ),
            })

        # Per magistrali: programma internazionale
        if course_id in MAGISTRALE_COURSES and intl is None:
            return self._return_and_save({
                "status": "NEED_INFO",
                "message": (
                    "Hai svolto un **programma internazionale di doppio titolo**?\n"
                    "Rispondi con **sì** o **no**.\n"
                    "*(Se no, ti chiederò se hai svolto CFU all'estero)*"
                ),
            })

        # CFU estero — solo se non doppio titolo
        if cfu == 0.0 and not intl:
            # Non chiediamo esplicitamente: usiamo il valore fornito (default 0)
            # ma informiamo l'utente che può specificarlo
            pass  # Procedi col calcolo con cfu=0; l'utente avrà fornito il valore se necessario

        # 5. ESEGUI IL CALCOLO
        result = self.service.calculate(
            course_id=course_id,
            cohort_year=cohort,
            media_30=float(media),
            pt=float(pt_val),
            laurea_in_corso=bool(in_corso),
            cfu_estero=float(cfu or 0.0),
            programma_internazionale=bool(intl) if intl is not None else False,
            percorso_eccellenza=bool(excel) if excel is not None else False,
        )

        # 6. PULIZIA SESSION STATE (calcolo completato)
        self._clear_session()

        return self._return_and_save(result)

    # ------------------------------------------------------------------
    # HELPERS INTERNI
    # ------------------------------------------------------------------

    def _clear_session(self):
        """Rimuove le chiavi temporanee dal session state dopo il calcolo."""
        keys = [
            self._SK_COURSE, self._SK_COHORT, self._SK_MEDIA, self._SK_PT,
            self._SK_IN_CORSO, self._SK_CFU, self._SK_INTL, self._SK_EXCEL,
        ]
        for k in keys:
            self.session_state.remove(_ctx(k))


# =========================================================================
# UTILITY FUNCTIONS (modulo-level, non dipendenti dalla classe)
# =========================================================================

def _ctx(key: str) -> str:
    """Aggiunge un namespace alle chiavi del session state per evitare conflitti."""
    return f"grad_tool::{key}"


def _normalize_cohort(raw: str) -> str:
    """
    Normalizza l'input utente sulla coorte:
      '25/26', '2025/26', '2025/2026'  → '25/26'
      '24/25', '23/24', '22/23' ecc.  → '24/25' (formula vecchia)
    """
    s = raw.strip().replace(" ", "")
    # Cerca l'anno iniziale con 2 o 4 cifre
    match = re.search(r"(\d{2,4})[/\-](\d{2,4})", s)
    if match:
        start = match.group(1)[-2:]  # Ultime 2 cifre
        end   = match.group(2)[-2:]
        label = f"{start}/{end}"
        if label == "25/26":
            return "25/26"
        else:
            return "24/25"  # Tutte le coorti precedenti usano la stessa formula
    # Fallback: se c'è solo "25" o "2025" nell'input
    if "25" in s or "2025" in s:
        return "25/26"
    return "24/25"


def _get_in_corso_label(course_id: str, cohort: str) -> str:
    """Restituisce una descrizione leggibile del limite temporale per 'in corso'."""
    is_new = (cohort == "25/26")
    if course_id in TRIENNALE_COURSES:
        if is_new:
            return "entro il terzo anno accademico senza fuori corso"
        else:
            return "entro il 31 maggio del quarto anno solare dall'immatricolazione"
    else:  # Magistrale
        if is_new:
            return "entro il secondo anno accademico senza fuori corso"
        else:
            return "entro il 31 maggio del quarto anno solare dall'immatricolazione"
