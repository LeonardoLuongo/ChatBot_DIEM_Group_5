from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider
from WebBasedRAG.RAGComponents.tools.services.professors_service import ProfessorsService

# IMPORTA IL SESSION STATE
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.web_explorer_components.explorer import WebExplorer

# 1. Definiamo gli Schemi
class GetProfessorProfileInput(BaseModel):
    nome: Optional[str] = Field(default="", description="""First name of the professor (e.g., "Pasquale", "Maria pia"). Leave empty if unknown.""")
    cognome: Optional[str] = Field(default="", description="""Last name of the professor (e.g., "Foggia", "D'arienzo"). Leave empty if unknown.""")
    url: Optional[str] = Field(default=None, description="""Optional. Direct URL of the professor's profile.""")


class SearchProfessorInput(BaseModel):
    query_ricerca: str = Field(..., description="""Search query for the directory. 
CRITICAL FORMATTING RULE: The underlying search system strictly requires the format to be "Lastname Firstname" if both are provided.
- If you have both, you MUST put the last name first (e.g., DO: "Capuano Nicola"). Writing "Nicola Capuano" will return NO results.
- You can search by first name only (e.g., "Nicola") or last name only (e.g., "Capuano").
- Partial combinations of both names (e.g., "Pas Fog") will fail. Use either the full "Lastname Firstname" format or just one of the names in full.""")

# 2. Definiamo il Provider
class ProfessorsToolProvider(ToolProvider):
    """
    Fa da Adapter tra il LangChain e il ProfessorsService.
    """
    # MODIFICA: Aggiunto session_state al costruttore
    def __init__(self, service: ProfessorsService, session_state: GlobalSessionState, web_explorer: WebExplorer):
        self.service = service
        self.session_state = session_state
        self.web_explorer = web_explorer # <-- Riceviamo l'explorer con il chunker e il reranker

    def get_tools(self) -> List[StructuredTool]:
        
        # WRAPPER 1: Profilo Docente
        # Aggiunti i default e l'argomento 'url'
        def fetch_professor_profile_action(nome: str = "", cognome: str = "", url: str = None) -> str:
            
            # --- LOGICA AIKIDO: Se il modello passa l'URL, estraiamo nome e cognome da lì ---
            if url and (not nome or not cognome):
                try:
                    # Esempio URL: https://docenti.unisa.it/nicola.capuano
                    # Rimuoviamo eventuali slash finali e prendiamo l'ultimo blocco
                    parti = url.rstrip('/').split('/')[-1].split('.')
                    if len(parti) >= 2:
                        nome = parti[0]
                        cognome = parti[1]
                        print(f"   🩹 [AUTO-FIX] Estratto dal URL: Nome='{nome}', Cognome='{cognome}'")
                except Exception as e:
                    print(f"   ⚠️ [AUTO-FIX WARN] Fallita estrazione nome/cognome dall'URL: {e}")

            # Controllo standard di sicurezza
            if not nome or not cognome:
                return (
                    f"❌ ERROR: Missing required parameters. You provided nome='{nome}' and cognome='{cognome}'. "
                    f"You MUST provide BOTH exact first name and last name. "
                    f"If you only have a partial name (like just the surname), you MUST stop and use the `SearchProfessorDirectory` tool FIRST to find the full exact name."
                )
            raw_markdown = self.service.get_professor_profile(nome, cognome)
            
            # Se il risultato è valido, lo salviamo nel contesto globale
            if "non è stato trovato" not in raw_markdown and "Errore" not in raw_markdown:
                doc_id = self.session_state._doc_counter + 1
                
                # Normalizzazioni per l'URL e per il Titolo
                nome_norm_url = nome.strip().lower().replace(" ", "")
                cognome_norm_url = cognome.strip().lower().replace(" ", "").replace("'", "")
                url = f"https://docenti.unisa.it/{nome_norm_url}.{cognome_norm_url}"
                
                print(f"   [PULIZIA] Rimozione intestazioni inutili dal profilo di {nome} {cognome}...")
                
                import re
                
                # --- LOGICA DI ESTRAZIONE ROBUSTA ---
                
                # 1. TROVA L'INIZIO (Ignora tutto il menu superiore)
                # Cerca l'intestazione che contiene "Home" (es. "## ANTONIO GRECO Home" o "# ANTONIO GRECO | Home")
                start_idx = 0
                match_start = re.search(r'##?\s+.*?Home', raw_markdown, re.IGNORECASE)
                if match_start:
                    start_idx = match_start.start()
                
                # 2. TROVA LA FINE (Ignora tutto il footer e i menu laterali)
                end_idx = len(raw_markdown)
                
                # Lista di "parole di stop" che indicano la fine del profilo utile
                stop_markers = [
                    "![Home]",             # Inizio banner immagine
                    "* Docenti",           # Inizio menu laterale
                    "### [CAMPUS MAP",     # Inizio footer
                    "## Pubblicazioni",    # Sezione troppo lunga
                    "## Progetti"          # Sezione non pertinente per il bot
                ]
                
                for marker in stop_markers:
                    idx = raw_markdown.find(marker, start_idx)
                    # Se trova il marcatore e viene PRIMA dell'attuale end_idx, aggiorna end_idx
                    if idx != -1 and idx < end_idx:
                        end_idx = idx
                
                # 3. RITAGLIO IL TESTO
                # Prende ESATTAMENTE dal titolo fino a un attimo prima del menu/footer
                testo_finale = raw_markdown[start_idx:end_idx].strip()
                
                # Fallback di sicurezza: se per qualche assurdo motivo estrae troppo poco (es < 50 char), 
                # mandiamo i primi 3000 caratteri del markdown grezzo.
                if len(testo_finale) < 50:
                    testo_finale = raw_markdown[:3000]
                
                # --- NOVITÀ: COSTRUZIONE TITOLO DINAMICO ---
                nome_pulito = nome.strip().title()
                cognome_pulito = cognome.strip().title()
                title = f"Profilo Ufficiale Docente - {nome_pulito} {cognome_pulito}"
                
                testo_da_salvare = (
                    f"Titolo del Documento: {title}\n"
                    f"Contesto: Questo documento contiene i dati ufficiali, contatti, ufficio e orario di ricevimento del docente.\n\n"
                    f"{testo_finale}"
                )
                
                testo = f"[Doc {doc_id}]\n{testo_da_salvare}"

                # Salviamo nel DB
                self.session_state.add_source(
                    url=url,
                    title=title,
                    testo_estratto=testo,
                    score=1.0
                )
                
                return (f"SUCCESS: Il profilo di {nome_pulito} {cognome_pulito} è stato scaricato. Informazioni contenute:\n{testo}\n")
            
            return raw_markdown

        # WRAPPER 2: Ricerca Docente
        def search_professor_directory_action(query_ricerca: str) -> str:
            result = self.service.search_professor_directory(query_ricerca)
            
            if "Nessun risultato trovato" not in result and "errore" not in result.lower():
                doc_id = self.session_state._doc_counter + 1
                
                # --- NOVITÀ: COSTRUZIONE TITOLO DINAMICO ---
                title = f"Risultati Ricerca Rubrica Docenti - Query: '{query_ricerca}'"
                
                testo_da_salvare = (
                    f"Titolo del Documento: {title}\n"
                    f"Contesto: Lista dei docenti trovati nella rubrica di ateneo per la query indicata.\n\n"
                    f"{result}"
                )
                
                testo = f"[Doc {doc_id}]\n{testo_da_salvare}"

                self.session_state.add_source(
                    url="https://docenti.unisa.it/",
                    title=title,
                    testo_estratto=testo,
                    score=1.0
                )
                
                # IL FIX È QUI: L'agente DEVE vedere il risultato in chiaro, 
                # altrimenti non saprà quale NOME ESATTO passare a FetchProfessorProfile nello Step 2.
                return (f"SUCCESS: Ricerca salvata nel DB come [Doc {doc_id}]. Informazioni contenute: {testo}\n")
                
            return result


        profile_tool = StructuredTool.from_function(
            func=fetch_professor_profile_action,  # <-- Usiamo il wrapper locale
            name="FetchProfessorProfile",
            description="Use this to download the complete profile (bio, contacts, info) of a professor given exact first and last name. CRITICAL: NEVER pass URLs as arguments. Use ONLY 'nome' and 'cognome'.",
            args_schema=GetProfessorProfileInput,
            handle_tool_error=True
        )

        search_tool = StructuredTool.from_function(
            func=search_professor_directory_action, # <-- Usiamo il wrapper locale
            name="SearchProfessorDirectory",
            description="CRITICAL: You MUST use this tool FIRST if you only have a single word (like just the surname) or some chars for a professor. It searches the directory and returns the full first and last name needed for FetchProfessorProfile.",
            args_schema=SearchProfessorInput
        )

        return [profile_tool, search_tool]