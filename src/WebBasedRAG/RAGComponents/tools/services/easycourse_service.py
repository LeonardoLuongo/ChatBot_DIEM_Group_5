import json
import re
import concurrent.futures 
from typing import Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from rapidfuzz import fuzz
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from WebBasedRAG.RAGComponents.relevance_evaluator import RelevanceEvaluator
from langchain_core.language_models.chat_models import BaseChatModel

# =====================================================================
# MODELLI PYDANTIC PER L'ESTRAZIONE LLM INTERNA
# =====================================================================
class InfoInsegnamenti(BaseModel):
    docente: str = Field(
        default="", 
        description="Professor's name strictly as 'LASTNAME FIRSTNAME' (e.g., 'GRECO ANTONIO')."
    )
    semestre: str = Field(
        default="", 
        description="The specific semester requested (e.g., 'primo semestre', 'secondo semestre', 'annuale'). Leave empty if not specified."
    )
    data: str = Field(default="", description="The exact date or academic year as written by the user (e.g. '2025/2026', '2026', '24-05-2026'). Empty if absent.")

class InfoDocente(BaseModel):
    docente: str = Field(
        default="", 
        description="Professor's name. You MUST format it strictly as 'LASTNAME FIRSTNAME' uppercase (e.g., if query says 'Mario Rossi', output 'ROSSI MARIO')."
    )
    data: str = Field(default="", description="The exact date or academic year as written by the user (e.g. '2025/2026', '2026', '24-05-2026'). Empty if absent.")

class InfoStudente(BaseModel):
    corso_di_laurea: str = Field(default="", description="E.g., 'Computer Engineering'. Omit words like 'lessons' or 'bachelor's'.")
    curriculum_anno: str = Field(default="", description="E.g., 'Third year', 'Common track'.")
    data: str = Field(default="", description="The exact date or academic year as written by the user (e.g. '2025/2026', '2026', '24-05-2026'). Empty if absent.")

class SceltaMenu(BaseModel):
    scelta_html_value: str = Field(description="The EXACT HTML value extracted from the provided dictionary.")
    motivo: str = Field(description="Brief reason for the choice.")

DEBUG = False
TIMEOUT = 1000
class EasyCourseService:
    """Service dedicato all'estrazione orari da EasyCourse Unisa tramite Playwright e LLM (Thick Tool)."""

    # INIETTIAMO L'EVALUATOR!
    def __init__(self, llm: BaseChatModel, evaluator: RelevanceEvaluator = None):
        self.llm = llm 
        self.evaluator = evaluator 
        
        self.studente_parser = self.llm.with_structured_output(InfoStudente)
        self.docente_parser = self.llm.with_structured_output(InfoDocente)
        self.menu_parser = self.llm.with_structured_output(SceltaMenu)
        self._pw_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # =====================================================================
    # LOGICA DI FILTRAGGIO SEMANTICO (Sostituisce il Fuzzy Matching)
    # =====================================================================
    def _filtra_opzioni_semantiche(self, termine_ricerca: str, opzioni_dict: Dict[str, str], top_k: int = 5) -> Dict[str, str]:
        if not opzioni_dict or not termine_ricerca.strip():
            return opzioni_dict 
            
        # Fallback al fuzzy matching se per qualche motivo il reranker non viene passato (es. in test isolati)
        if self.evaluator is None:
            print("   ⚠️ [EASYCOURSE] Reranker non trovato, fallback su Fuzzy Matching.")
            return self._filtra_opzioni_fuzzy_fallback(termine_ricerca, opzioni_dict, top_k)

        # 1. Prepariamo la lista delle stringhe estratte dal menu a tendina
        testi_opzioni = list(opzioni_dict.keys())
        
        # 2. Chiediamo al Cross-Encoder di valutare l'affinità semantica (Batch Mode = Velocissimo)
        scores = self.evaluator.calculate_rewards_batch(termine_ricerca, testi_opzioni)
        
        # 3. Zippiamo i punteggi con i testi originali e ordiniamo in modo decrescente
        scored_options = list(zip(scores, testi_opzioni))
        scored_options.sort(key=lambda x: x[0], reverse=True)
        risultato_filtrato = {}
        print(f"\n🔍 [FILTRO SEMANTICO] Top candidati per '{termine_ricerca}':")
        
        best_score = scored_options[0][0] if scored_options else 0
        if best_score < 0.1: # I punteggi del reranker BGE sono in genere tra 0 e 1 (sigmoide)
            print(f"   ⚠️ [EASYCOURSE] Score semantico basso ({best_score:.2f}). L'opzione potrebbe mancare.")
            
        # 4. Ricostruiamo il dizionario solo per i Top K candidati
        for score, text in scored_options[:top_k]:
            val = opzioni_dict[text]
            print(f"   - {text} (Score semantico: {score:.3f})")
            risultato_filtrato[text] = val
            
        return risultato_filtrato

    def _filtra_opzioni_fuzzy_fallback(self, termine_ricerca: str, opzioni_dict: Dict[str, str], top_k: int = 5) -> Dict[str, str]:
        print(f"\n--- Inizio Fuzzy Search Fallback ---")
        print(f"Termine di ricerca: '{termine_ricerca}' | Opzioni totali da analizzare: {len(opzioni_dict)}")
        
        scored_options = []
        for text, val in opzioni_dict.items():
            score = fuzz.WRatio(termine_ricerca.lower(), text.lower())
            scored_options.append((score, text, val))
            
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        top_results = scored_options[:top_k]
        
        print(f"Top {top_k} risultati trovati:")
        for i, (score, text, val) in enumerate(top_results, 1):
            print(f"  {i}. Score: {score:>3} | Testo: '{text}' -> Valore HTML: '{val}'")
        print(f"------------------------------------\n")
            
        return {text: val for _, text, val in top_results}

    def _ask_llm_to_choose(self, query_utente: str, options_dict: Dict[str, str], contesto: str) -> Optional[str]:
        if not options_dict:
            return None
            
        print(f"   🤖 [TOOL-LLM] Scelta da menu: {contesto}...")
        
        regola_anno = ""
        if "Academic Year" in contesto or "Anno Accademico" in contesto:
            regola_anno = (
                "\nVITAL RULE FOR THE ACADEMIC YEAR:\n"
                "An Academic Year begins in September of Year X and ends in August of Year X+1. "
                "If the date requested is between January and August (e.g., May 2026), the correct academic year is the previous year (e.g., 2025/2026). "
                "If the date is between September and December, the academic year is the same year as the date."
            )
            
        regole_generali = (
            "\nGENERAL LOGIC RULES:\n"
            "- If the user asks for a specific year (e.g., 'secondo anno', 'primo anno'), you MUST select the option that explicitly contains that number (e.g., '2 -', '1 -').\n"
            "- If the user asks for a Degree Course but DOES NOT specify 'Magistrale' or 'Master', you MUST prefer the base 'CORSO DI LAUREA' over 'CORSO DI LAUREA MAGISTRALE'."
            "- If the user asks for a Degree Course but specify 'Magistrale' or 'Master', you MUST prefer 'CORSO DI LAUREA MAGISTRALE' over 'CORSO DI LAUREA'."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert web scraping agent. Your current task is: {contesto}.\n"
                       f"Read the user's request and find the best option among those provided in the JSON Array.\n"
                       f"You MUST return EXACTLY and ONLY the string from the 'html_value' field. Do not concatenate or modify it."
                       f"{regola_anno}{regole_generali}"),
            ("human", "User's original query: '{query_utente}'\n\n"
                      "Available options:\n"
                      "{options_json}\n\n"
                      "Extract the correct 'html_value'.")
        ])
        
        try:
            chain = prompt | self.menu_parser
            
            options_list = [{"text": k, "html_value": v} for k, v in options_dict.items()]
            options_str = json.dumps(options_list, indent=2, ensure_ascii=False)
            
            risultato: SceltaMenu = chain.invoke({
                "query_utente": query_utente, 
                "options_json": options_str
            })
            
            scelta_sporca = risultato.scelta_html_value
            valori_validi = list(options_dict.values())
            
            # -------------------------------------------------------------
            # SAFETY NET: Sanitizzazione dell'output sporco dell'LLM
            # -------------------------------------------------------------
            scelta_pulita = None
            if scelta_sporca in valori_validi:
                scelta_pulita = scelta_sporca
            else:
                # Se l'LLM ha aggiunto spazzatura (es: 'PDS0|1'}<tool_call>), 
                # cerchiamo quale valore valido è contenuto nella stringa sporca.
                for val in valori_validi:
                    if val in scelta_sporca:
                        scelta_pulita = val
                        print(f"      🩹 [AUTO-FIX] Stringa sporca corretta in: '{scelta_pulita}'")
                        break
            
            if not scelta_pulita:
                print(f"      ❌ [ERRORE] Il valore '{scelta_sporca}' non esiste tra le opzioni HTML.")
                return None
                
            print(f"      ✅ Scelto (Pulito): '{scelta_pulita}' ({risultato.motivo})")
            return scelta_pulita
            
        except Exception as e:
            print(f"      ❌ Errore Tool-LLM Scelta Menu: {e}")
            return None


    def _attendi_ajax(self, page):
        """Attende in modo sicuro che EasyCourse finisca di caricare la tendina successiva."""
        try:
            page.wait_for_response(lambda response: "index.php" in response.url, timeout=3000)
        except Exception:
            pass
        page.wait_for_timeout(TIMEOUT)

    def _normalizza_data(self, data_str: str) -> str:
        """Forza qualunque data (anche ISO 8601 o stringhe sporche) estratta dall'LLM nel formato robusto %d-%m-%Y."""
        if not data_str:
            return ""
        
        data_str = data_str.strip()
        
        # 1. Preserviamo se è un Anno Accademico (es. "2025/2026")
        if re.search(r'20\d{2}[/\\-]20\d{2}', data_str):
            return data_str
            
        # 2. Preserviamo se è solo un anno (es. "2026")
        if re.fullmatch(r'20\d{2}', data_str):
            return data_str

        # 3. Logica originale per estrazione data
        numeri = re.findall(r'\d+', data_str)
        
        # Se non ci sono almeno 3 blocchi numerici, non è una data
        if len(numeri) < 3:
            return data_str  # Restituisce ciò che ha trovato invece di schiantarsi

        # Prendiamo solo i primi tre elementi (anno, mese, giorno... o viceversa)
        blocco1, blocco2, blocco3 = numeri[0], numeri[1], numeri[2]
        
        try:
            # Caso 1: L'anno è all'inizio (YYYY-MM-DD)
            if len(blocco1) == 4:
                dt = datetime(int(blocco1), int(blocco2), int(blocco3))
            # Caso 2: L'anno è alla fine (DD-MM-YYYY)
            elif len(blocco3) == 4:
                dt = datetime(int(blocco3), int(blocco2), int(blocco1))
            else:
                # Se l'anno è a 2 cifre (DD-MM-YY), per default datetime usa l'anno corrente
                dt = datetime.strptime(f"{blocco1}-{blocco2}-{blocco3}", "%d-%m-%y")
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            print(f"   ⚠️ Impossibile costruire una data logica da: '{data_str}'.")
            return data_str

    def _calcola_anno_accademico(self, data_str: str, opzioni_anno: Dict[str, str]) -> str:
        """Calcola deterministicamente l'anno accademico in Python, bypassando l'LLM."""
        if not opzioni_anno: 
            return ""

        print(f"   ⚙️ [PYTHON] Calcolo matematico dell'Anno Accademico (Input LLM: '{data_str}')...")
        try:
            target_year = None
            
            # CASO 1: Anno Accademico esplicito (es. "2025/2026")
            match_aa = re.search(r'(20\d{2})\s*[/\\-]\s*(20\d{2})', data_str)
            if match_aa:
                target_year = int(match_aa.group(1))
            
            # CASO 2: Nessun anno esplicito, usiamo data esatta o anno singolo
            else:
                try:
                    dt = datetime.strptime(data_str, "%d-%m-%Y") if data_str else datetime.now()
                except ValueError:
                    # Se ha scritto "2026", simuliamo il 1 Gennaio 2026
                    match_anno = re.search(r'(20\d{2})', data_str)
                    if match_anno:
                        dt = datetime(int(match_anno.group(1)), 1, 1)
                    else:
                        dt = datetime.now()

                # Regola aurea: mesi < Settembre -> anno precedente
                target_year = dt.year if dt.month >= 9 else dt.year - 1

            target_val = str(target_year)

            # Verifichiamo che l'anno calcolato esista effettivamente nella tendina HTML
            valori_html = list(opzioni_anno.values())
            if target_val in valori_html:
                print(f"      ✅ Anno Accademico calcolato: {target_year}/{target_year+1} (HTML Value: '{target_val}')")
                return target_val

            # Fallback: se l'anno futuro non è ancora nel database, prendiamo il più recente disponibile (il primo della lista)
            fallback = valori_html[0]
            print(f"      ⚠️ L'anno {target_year} non è ancora nel DB. Uso il fallback più recente: '{fallback}'")
            return fallback
            
        except Exception as e:
            print(f"      ❌ Errore calcolo anno: {e}")
            return list(opzioni_anno.values())[0] if opzioni_anno else ""


    def _estrai_opzioni_select(self, page, selettore: str) -> Dict[str, str]:
        return page.evaluate(f"""() => {{
            let safeSelector = '{selettore}'.split(',').map(s => s.trim() + ' option').join(', ');
            let opts = Array.from(document.querySelectorAll(safeSelector));
            let result = {{}};
            opts.forEach(o => {{
                let text = (o.textContent || o.text || "").trim();
                let val = o.value || "";
                // Rimuoviamo placeholder vuoti (--) e bug del DB Unisa (-1)
                if (text && !text.includes('--') && !text.startsWith('-1')) {{ 
                    result[text] = val; 
                }}
            }});
            return result;
        }}""")

    # =====================================================================
    # FLUSSO 1: STUDENTI
    # =====================================================================
    def fetch_students_schedule(self, query: str) -> str:
        """Esegue il workflow in un thread isolato (Safe per Gradio)."""
        return self._pw_executor.submit(self._fetch_students_schedule_isolato, query).result()

    def _fetch_students_schedule_isolato(self, query: str) -> str:
        """Esegue il workflow completo di scraping guidato dall'LLM per gli studenti."""
        print(f"   🤖 [TOOL-LLM] Analisi query studente: '{query}'")
        
        # 1. Estrazione Entità Iniziali
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the 'corso_di_laurea', the 'curriculum_anno' and the 'data' from the user query."),
                ("human", "{query}")
            ])
            info: InfoStudente = (prompt | self.studente_parser).invoke({"query": query})
        except Exception:
            info = InfoStudente(corso_di_laurea=query, curriculum_anno=query, data="")

        # AGGIUNGI QUESTA RIGA QUI:
        info.data = self._normalizza_data(info.data)
        print(f"      📅 Data normalizzata per Playwright: '{info.data}'")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not(DEBUG))
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            try:
                page.goto("https://easycourse.unisa.it/AgendaStudenti/index.php?view=easycourse&_lang=it&include=corso")
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(TIMEOUT)

                # -- STEP 1: ANNO ACCADEMICO --
                opzioni_anno = self._estrai_opzioni_select(page, 'select[name="anno"]')
                
                # Bypasso l'LLM e calcolo la data matematicamente
                id_anno_scelto = self._calcola_anno_accademico(info.data, opzioni_anno)
                
                if id_anno_scelto:
                    page.locator('select[name="anno"]').select_option(value=id_anno_scelto, force=True)
                    self._attendi_ajax(page)



                # -- STEP 2: CORSO DI STUDIO --
                opzioni_corso_totali = self._estrai_opzioni_select(page, 'select[name="corso"]')
                opzioni_corso_papabili = self._filtra_opzioni_semantiche(info.corso_di_laurea, opzioni_corso_totali, top_k=5)
                id_corso_scelto = self._ask_llm_to_choose(query, opzioni_corso_papabili, "Select the Degree Course")
                
                if not id_corso_scelto:
                    return json.dumps({"status": "ERROR", "message": "Impossibile determinare il corso richiesto."})
                
                page.locator('select[name="corso"]').select_option(value=id_corso_scelto, force=True)
                page.wait_for_timeout(TIMEOUT + 200)

                # -- STEP 3: CURRICULUM --
                opzioni_curriculum_totali = self._estrai_opzioni_select(page, 'select[name="anno2[]"], select#anno2')
                if opzioni_curriculum_totali:
                    
                    # NESSUN FILTRO! Passiamo tutte le opzioni direttamente all'LLM (sono poche, max 5-10).
                    # L'LLM userà le sue "GENERAL LOGIC RULES" per mappare intelligentemente "primo" a "1".
                    id_curr_scelto = self._ask_llm_to_choose(query, opzioni_curriculum_totali, "Select the Year of Study / Curriculum")
                    
                    if id_curr_scelto:
                        page.locator('select[name="anno2[]"], select#anno2').select_option(value=id_curr_scelto, force=True)
                        self._attendi_ajax(page)

                # -- STEP 4: DATA E SUBMIT --
                if info.data:
                    js_code = f"""
                    (() => {{
                        // 1. Cerca il campo con id="date" o name="date" (come da HTML)
                        let dateInput = document.getElementById('date') || document.querySelector('input[name="date"]');
                        if (dateInput) {{
                            // 2. Rimuove il blocco readonly per permettere la sovrascrittura
                            dateInput.removeAttribute('readonly');
                            
                            // 3. Forza la data
                            dateInput.value = '{info.data}';
                            
                            // 4. Avvisa il form che il valore è cambiato
                            dateInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            dateInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }})();
                    """
                    page.evaluate(js_code)
                    page.wait_for_timeout(TIMEOUT - 200)

                btn_locator = page.locator('button:has-text("Mostra"), input[value="Mostra Orario"]').first
                if btn_locator.count() > 0:
                    btn_locator.click()
                else:
                    page.evaluate("document.forms[0].submit()")
                
                # -- STEP 5: ESTRAZIONE DATI --
                return self._estrai_tabella_markdown(page, info.data)

            except Exception as e:
                return json.dumps({"status": "ERROR", "message": f"Errore Playwright: {str(e)}"})
            finally:
                browser.close()

    # =====================================================================
    # FLUSSO 2: DOCENTI
    # =====================================================================
    def fetch_teacher_schedule(self, query: str) -> str:
        """Esegue il workflow in un thread isolato (Safe per Gradio)."""
        return self._pw_executor.submit(self.fetch_teacher_schedule_isolato, query).result()

    def fetch_teacher_schedule_isolato(self, query: str) -> str:
        """Esegue il workflow completo di scraping guidato dall'LLM per i docenti."""
        print(f"   🤖 [TOOL-LLM] Analisi query docente: '{query}'")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the professor's name ('docente') and the target date ('data') from the user query. "
                           "CRITICAL: Format the 'docente' field strictly as 'SURNAME NAME' (e.g. 'GRECO ANTONIO')."),
                ("human", "{query}")
            ])
            info: InfoDocente = (prompt | self.docente_parser).invoke({"query": query})
        except Exception:
            info = InfoDocente(docente=query, data="")


        # AGGIUNGI QUESTA RIGA QUI:
        info.data = self._normalizza_data(info.data)
        print(f"      📅 Data normalizzata per Playwright: '{info.data}'")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not(DEBUG))
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            try:
                page.goto("https://easycourse.unisa.it/AgendaStudenti/index.php?view=easycourse&_lang=it&include=docente")
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(TIMEOUT)

                # -- STEP 1: ANNO ACCADEMICO --
                opzioni_anno = self._estrai_opzioni_select(page, 'select[name="anno"]')
                
                # Bypasso l'LLM e calcolo la data matematicamente in Python
                id_anno_scelto = self._calcola_anno_accademico(info.data, opzioni_anno)
                
                if id_anno_scelto:
                    page.locator('select[name="anno"]').select_option(value=id_anno_scelto, force=True)
                    self._attendi_ajax(page)


                # -- STEP 2: DOCENTE --
                print(f"      🔎 Nome estratto dall'LLM: '{info.docente}'")
                
                opzioni_docenti_totali = self._estrai_opzioni_select(page, 'select[name="docente"]')
                
                if not opzioni_docenti_totali:
                    return json.dumps({"status": "ERROR", "message": "Nessun docente presente nel database per l'Anno Accademico selezionato."})

                # Se l'LLM ha fallito l'estrazione, usiamo la query intera. 
                # Il Fuzzy (WRatio) è immune al rumore extra delle frasi lunghe.
                termine_da_cercare = info.docente if info.docente.strip() else query
                
                # USA DIRETTAMENTE IL FUZZY MATCHING
                opzioni_docenti_papabili = self._filtra_opzioni_fuzzy_fallback(termine_da_cercare, opzioni_docenti_totali, top_k=5)

                id_docente_scelto = self._ask_llm_to_choose(query, opzioni_docenti_papabili, "Select the Professor full name")
                
                if not id_docente_scelto:
                    return json.dumps({"status": "ERROR", "message": f"Impossibile determinare il docente '{termine_da_cercare}' tra le opzioni disponibili."})
                
                page.locator('select[name="docente"]').select_option(value=id_docente_scelto, force=True)
                self._attendi_ajax(page)



                # -- STEP 3: DATA E SUBMIT --
                if info.data:
                    js_code = f"""
                    (() => {{
                        // 1. Cerca il campo con id="date" o name="date" (come da HTML)
                        let dateInput = document.getElementById('date') || document.querySelector('input[name="date"]');
                        if (dateInput) {{
                            // 2. Rimuove il blocco readonly per permettere la sovrascrittura
                            dateInput.removeAttribute('readonly');
                            
                            // 3. Forza la data
                            dateInput.value = '{info.data}';
                            
                            // 4. Avvisa il form che il valore è cambiato
                            dateInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            dateInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }})();
                    """
                    page.evaluate(js_code)
                    page.wait_for_timeout(TIMEOUT - 200)

                btn_locator = page.locator('button:has-text("Mostra"), input[value="Mostra Orario"]').first
                if btn_locator.count() > 0:
                    btn_locator.click()
                else:
                    page.evaluate("document.forms[0].submit()")
                
                # -- STEP 4: ESTRAZIONE DATI --
                return self._estrai_tabella_markdown(page, info.data)

            except Exception as e:
                return json.dumps({"status": "ERROR", "message": f"Errore Playwright: {str(e)}"})
            finally:
                browser.close()

    # =====================================================================
    # FLUSSO 3: INSEGNAMENTI DEL DOCENTE (Materie)
    # =====================================================================
    def fetch_teacher_courses(self, query: str) -> str:
        """Esegue il workflow in un thread isolato (Safe per Gradio)."""
        return self._pw_executor.submit(self.fetch_teacher_courses_isolato, query).result()

    def fetch_teacher_courses_isolato(self, query: str) -> str:
        print(f"   🤖 [TOOL-LLM] Analisi query insegnamenti docente: '{query}'")
        
        insegnamenti_parser = self.llm.with_structured_output(InfoInsegnamenti)
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the professor's name ('docente'), the target semester ('semestre'), and the date ('data') from the user query. "
                           "CRITICAL: Format 'docente' strictly as 'SURNAME NAME' (e.g. 'GRECO ANTONIO'). "
                           "For 'semestre', extract words like 'primo semestre' or 'secondo semestre'. If none, leave empty. "
                           "For 'data', extract the exact string typed by the user (e.g. '2025/2026', '2026', '12-05-2026'). Do not force DD-MM-YYYY if it's an academic year."),
                ("human", "{query}")
            ])
            info: InfoInsegnamenti = (prompt | insegnamenti_parser).invoke({"query": query})
        except Exception:
            info = InfoInsegnamenti(docente=query, semestre="", data="")

        info.data = self._normalizza_data(info.data)
        
        # --- LOGICA SEMESTRE CORRENTE ---
        is_default_semester = False
        target_semestre = info.semestre.strip()
        
        if not target_semestre:
            is_default_semester = True
            dt = datetime.strptime(info.data, "%d-%m-%Y") if info.data else datetime.now()
            # Calendario Didattico:
            # Settembre (9) a Febbraio (2) -> Primo semestre
            # Marzo (3) ad Agosto (8) -> Secondo semestre
            if dt.month >= 9 or dt.month <= 2:
                target_semestre = "Primo semestre"
            else:
                target_semestre = "Secondo semestre"
            print(f"      📚 Nessun semestre specificato. Calcolato default: '{target_semestre}'")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not(DEBUG))
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            try:
                page.goto("https://easycourse.unisa.it/AgendaStudenti/index.php?view=easycourse&_lang=it&include=docente")
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(TIMEOUT)

                # -- STEP 1: ANNO ACCADEMICO --
                opzioni_anno = self._estrai_opzioni_select(page, 'select[name="anno"]')
                id_anno_scelto = self._calcola_anno_accademico(info.data, opzioni_anno)
                
                if id_anno_scelto:
                    page.locator('select[name="anno"]').select_option(value=id_anno_scelto, force=True)
                    self._attendi_ajax(page)

                # -- STEP 2: DOCENTE --
                opzioni_docenti_totali = self._estrai_opzioni_select(page, 'select[name="docente"]')
                if not opzioni_docenti_totali:
                    return json.dumps({"status": "ERROR", "message": "Nessun docente presente nel DB per l'anno selezionato."})

                termine_da_cercare = info.docente if info.docente.strip() else query
                opzioni_docenti_papabili = self._filtra_opzioni_fuzzy_fallback(termine_da_cercare, opzioni_docenti_totali, top_k=5)
                id_docente_scelto = self._ask_llm_to_choose(query, opzioni_docenti_papabili, "Select the Professor full name")
                
                if not id_docente_scelto:
                    return json.dumps({"status": "ERROR", "message": f"Impossibile determinare il docente '{termine_da_cercare}'."})
                
                page.locator('select[name="docente"]').select_option(value=id_docente_scelto, force=True)
                self._attendi_ajax(page)

                # -- STEP 3: PASSA A "ORARIO BASE" (via click reale, no jQuery.trigger) --
                try:
                    opzioni_tipo = self._select2_open_and_get_options(page, "doc_vo")
                    print(f"      📋 Opzioni tipo orario: {opzioni_tipo}")

                    val_orario_base = next(
                        (v for t, v in opzioni_tipo.items() if "orario base" in t.lower()), None
                    )
                    if val_orario_base:
                        self._select2_click_value(page, "doc_vo", val_orario_base)
                        print("      ✅ Cliccato 'Orario base', attendo caricamento semestri via AJAX...")

                        # Aspetta che il <select> nativo di doc_pd riceva le opzioni
                        try:
                            page.wait_for_function(
                                """() => {
                                    const sel = document.querySelector('select[name="doc_pd"]')
                                            || document.querySelector('select[name="periodo"]');
                                    return sel && sel.options.length > 1;
                                }""",
                                timeout=5000
                            )
                            print("      ✅ Semestri caricati nel native select.")
                        except Exception:
                            # EasyCourse potrebbe popolare solo la UI Select2, non il <select> nativo
                            print("      ⚠️ Timeout: opzioni non nel native select, si userà Select2 UI.")

                        page.wait_for_timeout(150)
                    else:
                        print("      ⚠️ Opzione 'Orario base' non trovata nel menu.")

                except Exception as e:
                    print(f"      ⚠️ STEP 3 fallito: {e}")

                # -- STEP 4: SELEZIONE SEMESTRE --
                # Prova prima il <select> nativo (dovrebbe essere popolato dall'AJAX)
                opzioni_semestre = self._estrai_opzioni_select(page, 'select[name="doc_pd"]')
                sem_name = "doc_pd"
                if not opzioni_semestre:
                    opzioni_semestre = self._estrai_opzioni_select(page, 'select[name="periodo"]')
                    if opzioni_semestre:
                        sem_name = "periodo"

                # Fallback: native select vuoto → leggi direttamente dai <li> Select2
                if not opzioni_semestre:
                    print("      ℹ️ Native select vuoto, leggo opzioni dal dropdown Select2...")
                    for candidate_name in ("doc_pd", "periodo"):
                        try:
                            opzioni_semestre = self._select2_open_and_get_options(page, candidate_name)
                            if opzioni_semestre:
                                sem_name = candidate_name
                                break
                        except Exception as e:
                            print(f"         ⚠️ Select2 '{candidate_name}' non disponibile: {e}")

                print(f"      📋 Opzioni semestre trovate ({sem_name}): {opzioni_semestre}")

                if opzioni_semestre:
                    id_sem_scelto = self._ask_llm_to_choose(
                        target_semestre, opzioni_semestre, "Select the Teaching Period / Semester"
                    )
                    if id_sem_scelto:
                        # _select2_click_value apre il dropdown se chiuso, clicca l'opzione
                        if self._select2_click_value(page, sem_name, id_sem_scelto):
                            self._attendi_ajax(page)
                            print(f"      ✅ Semestre selezionato: '{id_sem_scelto}'")
                        else:
                            print(f"      ⚠️ Value '{id_sem_scelto}' non trovato tra i <li>.")
                    else:
                        print("      ⚠️ LLM non ha scelto alcun semestre.")
                else:
                    print("      ❌ Nessuna opzione semestre disponibile — si procede senza filtro periodo.")

                # -- STEP 5: SUBMIT E ESTRAZIONE --
                btn_locator = page.locator('button:has-text("Mostra"), input[value="Mostra Orario"]').first
                if btn_locator.count() > 0:
                    btn_locator.click()
                else:
                    page.evaluate("document.forms[0].submit()")
                    
                from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
                try:
                    page.wait_for_selector('a.custom_color_text_legend', timeout=TIMEOUT + 1500)
                    page.wait_for_timeout(500)
                except PlaywrightTimeoutError:
                    pass
                    
                insegnamenti_correnti = page.evaluate("""() => {
                    let risultati = [];
                    let items = document.querySelectorAll('a.custom_color_text_legend');
                    items.forEach(a_tag => {
                        let testo = a_tag.innerText || a_tag.textContent || "";
                        if (testo.trim()) risultati.push(testo.trim());
                    });
                    return risultati;
                }""")
                
                # -- COSTRUZIONE RISPOSTA FINALE --
                if is_default_semester:
                    avviso = "\n*⚠️ Nota: Poiché non è stato specificato alcun periodo nella richiesta, è stato selezionato automaticamente il semestre corrente.*\n"
                else:
                    avviso = ""
                    
                rif_display = target_semestre.capitalize()

                if not insegnamenti_correnti:
                    return json.dumps({
                        "status": "SUCCESS", 
                        "markdown_list": f"{avviso}\nNessun insegnamento trovato per questo docente nel periodo richiesto ({rif_display}).",
                        "direct_url": page.url
                    })
                
                md_final = f"### Materie Insegnate ({rif_display})\n{avviso}\n"
                md_final += f"**Link Fonte Ufficiale:** [Consulta EasyCourse]({page.url})\n\n"
                
                for ins in sorted(insegnamenti_correnti):
                    md_final += f"- {ins}\n"
                    
                return json.dumps({
                    "status": "SUCCESS",
                    "markdown_list": md_final,
                    "direct_url": page.url
                })

            except Exception as e:
                return json.dumps({"status": "ERROR", "message": f"Errore Playwright: {str(e)}"})
            finally:
                browser.close()

    def _select2_open_and_get_options(self, page, select_name: str) -> Dict[str, str]:
        """
        Apre un dropdown Select2 via click reale e restituisce {testo_visibile: html_value}
        leggendo i <li> (NON il <select> nativo, che può rimanere vuoto in modalità AJAX).
        LASCIA IL DROPDOWN APERTO — chiama _select2_click_value subito dopo.
        """
        container_css = f"#select2-{select_name}-container"
        results_css   = f"#select2-{select_name}-results li.select2-results__option--selectable"
        prefix        = f"select2-{select_name}-result-"

        page.locator(container_css).click(timeout=3000)
        page.wait_for_timeout(400)
        page.wait_for_selector(results_css, state="visible", timeout=5000)

        # Estrae {testo: value} dai <li>
        # ID pattern: select2-{name}-result-XXXX-{value}  →  slice prefix + skip "XXXX-"
        return page.evaluate(f"""() => {{
            const result = {{}};
            const prefix = "{prefix}";
            document.querySelectorAll("{results_css}").forEach(li => {{
                const text = (li.innerText || "").trim();
                const rest = (li.id || "").slice(prefix.length);   // "XXXX-<value>"
                const dash = rest.indexOf("-");
                const val  = dash !== -1 ? rest.slice(dash + 1) : "";
                if (text && val) result[text] = val;
            }});
            return result;
        }}""")


    def _select2_click_value(self, page, select_name: str, option_value: str) -> bool:
        """
        Clicca l'opzione via click REALE Playwright (non JS .click()).
        Così Select2 riceve mousedown/mouseup e aggiorna la selezione + triggera l'AJAX.
        """
        container_css = f"#select2-{select_name}-container"
        results_css   = f"#select2-{select_name}-results li.select2-results__option--selectable"
        prefix        = f"select2-{select_name}-result-"

        # Apre il dropdown solo se non è già visibile
        is_open = (
            page.locator(results_css).count() > 0
            and page.locator(results_css).first.is_visible()
        )
        if not is_open:
            page.locator(container_css).click(timeout=3000)
            page.wait_for_timeout(400)
            page.wait_for_selector(results_css, state="visible", timeout=5000)

        # Trova l'ID esatto del <li> via JS
        # (gestisce valori con caratteri speciali come S1|ING_INF senza problemi)
        li_id = page.evaluate(f"""(target) => {{
            const prefix = "{prefix}";
            for (const li of document.querySelectorAll("{results_css}")) {{
                const rest = (li.id || "").slice(prefix.length);  // "XXXX-<value>"
                const dash = rest.indexOf("-");
                if (dash !== -1 && rest.slice(dash + 1) === target) {{
                    return li.id;
                }}
            }}
            return null;
        }}""", option_value)

        if not li_id:
            print(f"         ⚠️ Nessun <li> trovato per value='{option_value}' in '{select_name}'")
            return False

        # ✅ Click REALE tramite Playwright — genera tutti gli eventi mouse che Select2 aspetta
        # XPath evita problemi di escaping CSS con | e altri caratteri speciali negli ID
        page.locator(f"xpath=//li[@id='{li_id}']").click(timeout=3000)
        return True


    def _estrai_lista_insegnamenti(self, page, data_target: str = "") -> str:
        """Estrae l'elenco dei corsi insegnati dal docente dalla legenda HTML"""
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            # Attendiamo il caricamento degli elementi. Se va in timeout, non crashiamo: 
            # significa semplicemente che la lista è vuota per quel semestre/anno.
            try:
                page.wait_for_selector('a.custom_color_text_legend', timeout=TIMEOUT + 1500)
                page.wait_for_timeout(500)
            except PlaywrightTimeoutError:
                pass 
            
            dati_insegnamenti = page.evaluate("""() => {
                let risultati = [];
                // Selezioniamo direttamente i link con la classe specifica dei corsi!
                let items = document.querySelectorAll('a.custom_color_text_legend');
                items.forEach(a_tag => {
                    let testo = a_tag.innerText || a_tag.textContent || "";
                    testo = testo.trim();
                    if (testo) {
                        risultati.push(testo);
                    }
                });
                return risultati;
            }""")
            
            if not dati_insegnamenti:
                rif = data_target if data_target else "l'anno corrente"
                return json.dumps({
                    "status": "SUCCESS", 
                    "markdown_list": f"Nessun insegnamento trovato per questo docente nel periodo richiesto ({rif}).",
                    "direct_url": page.url
                })
            
            rif = f" (Riferimento temporale: {data_target})" if data_target else ""
            
            # --- AGGIUNTO LINK ESPLICITO ALLA FONTE UFFICIALE ---
            md_final = f"### Materie Insegnate{rif}\n"
            md_final += f"**Link Fonte Ufficiale:** [Consulta EasyCourse]({page.url})\n\n"
            
            for ins in dati_insegnamenti:
                md_final += f"- {ins}\n"
                
            return json.dumps({
                "status": "SUCCESS",
                "markdown_list": md_final,
                "direct_url": page.url
            })
            
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Errore estrazione legenda: {str(e)}"})



    # =====================================================================
    # ESTRAZIONE FINALE DELLA TABELLA (Comune a Studenti e Docenti)
    # =====================================================================
    def _estrai_tabella_markdown(self, page, data_target: str) -> str:
        try:
            page.wait_for_function("""() => {
                let cells = document.querySelectorAll('.rTableCell.rTableClass').length > 0;
                let rows = document.querySelectorAll('.calendar-table-row').length > 0;
                let alerts = Array.from(document.querySelectorAll('.alert-warning')).some(el => el.offsetParent !== null);
                return cells || rows || alerts;
            }""", timeout=TIMEOUT + 200)
            page.wait_for_timeout(TIMEOUT)
            
            # ELIMINATO il calcolo matematico in Python. 
            # Ora leggiamo le intestazioni reali direttamente dall'HTML!
            dati_lezioni = page.evaluate("""() => {
                let risultati = [];
                let lezioni = document.querySelectorAll('.rTableCell.rTableClass');
                
                // 1. Costruiamo la mappa dei giorni leggendo le INTESTAZIONI REALI DELLA TABELLA
                let giorni_map = {};
                for(let i=1; i<=6; i++) {
                    // Cerca il tag <th> o <div> dell'intestazione che contiene la data reale
                    let header = document.querySelector(`th.day-${i}, .rTableHead.day-${i}, .calendar-table-header .day-${i}`);
                    if (header) {
                        // Prende "lunedì 25/05" e lo pulisce dai ritorni a capo
                        giorni_map[`day-${i}`] = header.innerText.replace(/\\n/g, ' ').trim();
                    } else {
                        giorni_map[`day-${i}`] = "Giorno " + i;
                    }
                }
                
                // 2. Estraiamo i blocchi delle lezioni
                lezioni.forEach(lezione => {
                    let classi = Array.from(lezione.classList);
                    let id_giorno = classi.find(c => c.startsWith('day-'));
                    let nome_giorno = id_giorno && giorni_map[id_giorno] ? giorni_map[id_giorno] : '-';
                    
                    let contentDiv = lezione.querySelector('.rClassContent') || lezione;
                    let rawText = contentDiv.innerText || "";
                    let lines = rawText.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
                    
                    let orarioIdx = lines.findIndex(l => l.match(/\\d{1,2}:\\d{2}/));

                    if (orarioIdx !== -1) { 
                        let orario = lines[orarioIdx];
                        let corso = lines[0] || ""; 
                        let docente = lines.length >= 4 ? lines[1] : "";
                        let aula = lines.length >= 4 ? lines[2] : (lines.length === 3 ? lines[1] : "");

                        risultati.push({
                            giorno: nome_giorno, corso: corso, docenti: docente,
                            aula: aula, orario: orario.replace(/\\s+/g, ' ')
                        });
                    } else { 
                        let testolower = rawText.toLowerCase();
                        if (testolower.includes("festivit") || testolower.includes("sospens") || testolower.includes("vacanz") || testolower.includes("chiusur")) {
                            risultati.push({
                                giorno: nome_giorno, corso: "🔴 " + lines.join(" - ").toUpperCase(),
                                docenti: "-", aula: "-", orario: "Tutto il giorno"
                            });
                        }
                    }
                });
                return risultati;
            }""")

            dati_lezioni = [l for l in dati_lezioni if l.get("giorno") != "-"]
            
            # Usiamo la data estratta, così se la pagina è andata su Maggio, l'LLM lo sa.
            md_final = f"### Orario Universitario Estratto dal Sistema\n\n"
            if not dati_lezioni:
                md_final += "Nessuna lezione programmata o festività rilevata in questa vista."
            else:
                md_final += "| Data/Giorno | Orario | Insegnamento | Docente | Aula |\n|---|---|---|---|---|\n"
                for l in dati_lezioni:
                    # Titoli (Uppercase) per chiarezza es: "LUNEDÌ 25/05"
                    g = l["giorno"].replace("|", "/").upper()
                    o = l["orario"].replace("|", "/")
                    c = l["corso"].replace("|", "/")
                    d = l["docenti"].replace("|", "/")
                    a = l["aula"].replace("|", "/")
                    md_final += f"| **{g}** | {o} | {c} | {d} | {a} |\n"

            return json.dumps({
                "status": "SUCCESS", 
                "markdown_table": md_final, 
                "direct_url": page.url
            })

        except Exception as e:
            # Rimuovo PlaywrightTimeoutError perché catturo tutto in Exception e non serve importarlo
            return json.dumps({
                "status": "SUCCESS", 
                "markdown_table": f"### Orario Universitario (Settimana del {data_target})\n\n⚠️ Nessun dato presente o settimana vuota.", 
                "direct_url": page.url
            })
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": f"Errore tabella: {str(e)}"})




if __name__ == "__main__":
    import json
    import time
    from langchain_ollama.chat_models import ChatOllama

    print("="*60)
    print("🛠️ TEST ESTRAZIONE DATE HTML (EASYCOURSE)")
    print("="*60)

    # 1. Inizializzazione
    print("-> Caricamento LLM (Ministral)...")
    test_llm = ChatOllama(model="ministral-3:8b", temperature=0.0)

    print("-> Setup EasyCourseService (senza Reranker per test rapido)...")
    easycourse_service = EasyCourseService(llm=test_llm, evaluator=None)
    
    def stampa_risultato(json_str: str):
        try:
            parsed = json.loads(json_str)
            if parsed.get("status") == "SUCCESS":
                print("\n✅ ESTRAZIONE COMPLETATA. URL:", parsed.get("direct_url"))
                print("\n--- RESA MARKDOWN (Controlla le date nella prima colonna!) ---")
                print(parsed.get("markdown_table"))
                print("------------------------------------------------------------\n")
            else:
                print("❌ ERRORE:", parsed.get("message"))
        except Exception as e:
            print("Output non valido:", json_str)

    # =====================================================================
    # TEST 1: FLUSSO STUDENTI
    # =====================================================================
    query_studente = "Orario lezioni Magistrale Ingegneria Informatica per la settimana del 25-05-2026"
    print("\n" + "*"*50)
    print(f"🚀 TEST 1 (STUDENTI): {query_studente}")
    print("*"*50)
    
    res_stud = easycourse_service.fetch_students_schedule(query_studente)
    stampa_risultato(res_stud)

    # =====================================================================
    # TEST 2: FLUSSO DOCENTI
    # =====================================================================
    query_docente = "Orario del professore Mario Vento per la settimana del 25-05-2026"
    print("\n" + "*"*50)
    print(f"🚀 TEST 2 (DOCENTI): {query_docente}")
    print("*"*50)
    
    res_doc = easycourse_service.fetch_teacher_schedule(query_docente)
    stampa_risultato(res_doc)
    
    print("✅ Test completati. Verifica se i giorni stampati combaciano con quelli reali!")
