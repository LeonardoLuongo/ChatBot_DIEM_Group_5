import os
import sys
import time
import gc
import re
import torch
import ssl
import gradio as gr
from datetime import datetime
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
import os 
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util


# ==========================================
# 1. SETUP PATHS
# ==========================================
CURRENT_DIR = Path(os.getcwd())
SRC_PATH = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv

# SRC_PATH punta a "src", quindi suo "padre" è la cartella principale del progetto
env_path = SRC_PATH.parent / ".env"
load_dotenv(dotenv_path=env_path)

#QDRANTDB_PATH = SRC_PATH / "VectorDB" / "qdrant_trafilatura_1800_d6"
SESSION_DIR = SRC_PATH / "session"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

HISTORY_FILE = SESSION_DIR / "session_history.jsonl"
GUARDRAILS_LOG_FILE = SESSION_DIR / "guardrails_log.jsonl"

#Svuota la history fisica all'avvio del programma
try:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            pass # Aprire in modalità 'w' e non scrivere nulla svuota il file
        print("🧹 [SYSTEM] File session_history.jsonl azzerato all'avvio.")
except Exception as e:
    print(f"⚠️ [SYSTEM] Impossibile svuotare la history all'avvio: {e}")

# ==========================================
# 2. IMPORT COMPONENTI
# ==========================================
from langchain_ollama import ChatOllama
from langchain_community.utilities import SearxSearchWrapper
from WebBasedRAG.RAGComponents.relevance_evaluator import RelevanceEvaluator
from WebBasedRAG.RAGComponents.agent import SupervisorAgent
from WebBasedRAG.RAGComponents.judge import AnswerJudge
from WebBasedRAG.RAGComponents.planner import QueryPlanner
from WebBasedRAG.RAGComponents.generator import AnswerGenerator
from WebBasedRAG.RAGComponents.router import IntentRouter # <--- AGGIUNTO ROUTER
from WebBasedRAG.RAGComponents.embeddings.nomic_embeddings import NomicEmbeddings
from WebBasedRAG.RAGComponents.embeddings.jina_embeddings import JinaEmbeddings
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.embeddings.gte_reranker import BGE8KReranker

# Utils
from Utils.history_rewriter import load_history, rewrite_with_history, append_history_turn
from Utils.guardrails import validate_input, validate_output, refusal_message, GuardrailLogger

# Web Explorer Components
from WebBasedRAG.RAGComponents.web_explorer_components.browser import BrowserEngine
from WebBasedRAG.RAGComponents.web_explorer_components.scraping import HTMLCrawler, PDFExtractor
from WebBasedRAG.RAGComponents.web_explorer_components.utils import DocumentChunker
from WebBasedRAG.RAGComponents.web_explorer_components.retrieval import RetrievalEngine
from WebBasedRAG.RAGComponents.web_explorer_components.links import SmartLinkGenerator, PDFHighlightServer
from WebBasedRAG.RAGComponents.web_explorer_components.explorer import WebExplorer
from WebBasedRAG.RAGComponents.web_explorer_components.search import (
    QueryOptimizer, GoogleSearch, BingSearch, SearxSearch, HybridSearchOrchestrator
)

# Tools
from WebBasedRAG.RAGComponents.context_assessor import ContextAssessor
from WebBasedRAG.RAGComponents.tools.core.registry import ToolRegistry
from WebBasedRAG.RAGComponents.tools.services.easycourse_service import EasyCourseService
from WebBasedRAG.RAGComponents.tools.services.professors_service import ProfessorsService
from WebBasedRAG.RAGComponents.tools.providers.datetime_provider import DateTimeToolProvider
from WebBasedRAG.RAGComponents.tools.providers.websearch_provider import WebSearchToolProvider
from WebBasedRAG.RAGComponents.tools.providers.easycourse_provider import EasyCourseToolProvider
from WebBasedRAG.RAGComponents.tools.providers.professors_provider import ProfessorsToolProvider
from WebBasedRAG.RAGComponents.tools.providers.math_provider import MathToolProvider
from WebBasedRAG.RAGComponents.tools.services.qdrant_service import QdrantService
from WebBasedRAG.RAGComponents.tools.providers.qdrant_provider import QdrantToolProvider
from WebBasedRAG.RAGComponents.tools.services.graduation_grade_service import GraduationGradeService
from WebBasedRAG.RAGComponents.tools.providers.graduation_grade_provider import GraduationGradeToolProvider

import threading
import requests

# ==========================================
# 3. AGENTIC RAG SYSTEM
# ==========================================
class AgenticRag:
    def __init__(self):
        print("-- Inizializzazione Sistema: AGENT + ROUTER + VECTOR DB + HISTORY...")
        ssl._create_default_https_context = ssl._create_unverified_context
        self.logs = []
        self.guardrail_logger = GuardrailLogger(GUARDRAILS_LOG_FILE)
        
        # --- Configurazione Modelli Google Gemini ---
        # Assicurati di avere la variabile d'ambiente GOOGLE_API_KEY impostata.
        
        # Consiglio flash per ruoli veloci/economici e pro per la generazione finale complessa
        self.explorer_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        self.agent_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0) 
        self.judge_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        #self.final_rewriter_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
        
        # Per il generatore finale potresti voler usare il modello 'pro' se serve più ragionamento
        self.gen_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0) 
        
        
        #self.embeddings_model = NomicEmbeddings()
        print("-- Caricamento Jina Embeddings...")
        self.embeddings_model = JinaEmbeddings()
        self.reranker = BGE8KReranker()
        self.evaluator = RelevanceEvaluator(self.reranker)
        
        # Percorso DB Qdrant
        #self.qdrant_store_dir = QDRANTDB_PATH
        try:
            # Assicurati che QdrantService accetti url e collection_name al posto di store_dir_path
            self.qdrant_service = QdrantService(url="http://localhost:6333", collection_name="diem_chunks")
            print(" ✅ Qdrant Vector DB su Docker connesso.")
        except Exception as e:
            print(f" ⚠️ [QDRANT WARN] Impossibile avviare il Vector DB su Docker: {e}")
            self.qdrant_service = None

        self.session_state = GlobalSessionState()
        self.browser_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        self.initialize_system_with_retry()

    def initialize_system_with_retry(self):
        try:
            self.browser_engine = BrowserEngine(self.browser_path)
            self.query_opt = QueryOptimizer(self.explorer_llm)
            
            # Orchestrazione Ricerca Ibrida
            google = GoogleSearch(self.browser_engine)
            bing = BingSearch(self.browser_engine)
            searx = SearxSearch(SearxSearchWrapper(searx_host="http://127.0.0.1:8080", categories=["general"]))
            search_orch = HybridSearchOrchestrator([google, bing, searx])
            
            self.explorer = WebExplorer(
                query_optimizer=self.query_opt, 
                search_orchestrator=search_orch, 
                html_crawler=HTMLCrawler(), 
                pdf_extractor=PDFExtractor(), 
                chunker=DocumentChunker(embedding_model=self.embeddings_model), 
                retrieval_engine=RetrievalEngine(self.embeddings_model, self.evaluator), 
                smart_link_gen=SmartLinkGenerator(), 
                pdf_server=PDFHighlightServer()
            )

            
            # Registrazione Tools
            registry = ToolRegistry()
            registry.register_provider(DateTimeToolProvider())
            
            # Iniezione dipendenze nei service (LLM + Evaluator)
            easycourse_service = EasyCourseService(llm=self.agent_llm, evaluator=self.evaluator)
            registry.register_provider(EasyCourseToolProvider(easycourse_service, self.session_state))
            
            if self.qdrant_service: 
                registry.register_provider(QdrantToolProvider(self.qdrant_service, self.session_state, self.evaluator))
            
            registry.register_provider(WebSearchToolProvider(self.explorer, self.session_state, debug_mode=True))
            registry.register_provider(ProfessorsToolProvider(ProfessorsService(), self.session_state, self.explorer))
            registry.register_provider(MathToolProvider(session_state=self.session_state))
            registry.register_provider(GraduationGradeToolProvider(self.session_state))
            
            tools = registry.get_all_tools()
            
            # Componenti Core
            self.router = IntentRouter(llm=self.agent_llm)
            self.agent = SupervisorAgent(base_llm=self.agent_llm, tools=tools, session_state=self.session_state, max_retries=10)
            self.judge = AnswerJudge(llm=self.judge_llm)
            self.planner = QueryPlanner(llm=self.judge_llm, tools=tools) 
            self.generator = AnswerGenerator(llm=self.gen_llm)
            self.assessor = ContextAssessor(llm=self.judge_llm)

        except Exception as e: 
            print(f"Errore inizializzazione sistema: {e}")
            raise e

    def log(self, module, msg, color="#fb923c"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"<div style='margin-bottom:8px; font-family: monospace; font-size: 13px; border-left: 3px solid {color}; padding-left: 10px;'><span style='color: {color}; font-weight: bold;'>[{timestamp}] {module}:</span><br>{msg}</div>"
        self.logs.append(entry)
        return "".join(self.logs[::-1])

    def run(self, raw_query):
        self.logs = []
        self.session_state.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # =========================================================
        # GUSCIO DI SICUREZZA GLOBALE (Gestione Crash e Timeout API)
        # =========================================================
        try:

            yield self.log("SISTEMA", f"Inizio elaborazione query: <i>'{raw_query}'</i>", color="#94a3b8"), "⏳ *Verifica sicurezza...*"
            
            # 1. GUARDRAILS (Input)
            input_check = validate_input(raw_query, logger=self.guardrail_logger)
            if not input_check.allowed:
                yield self.log("GUARDRAILS", "🚫 Richiesta bloccata per violazione policy.", color="#ef4444"), refusal_message(input_check.reasons)
                return 

            # 2. HISTORY & REWRITING
            history = load_history(HISTORY_FILE)
            yield self.log("MEMORIA", f"Cronologia caricata: {len(history)} turni.", color="#a855f7"), "⏳ *Analisi contesto...*"
            
            rewrite_result = rewrite_with_history(query=raw_query, history=history, temperature=0.0)
            clean_query = rewrite_result.final_query
            
            history_log = f"Query contestualizzata: <b>'{clean_query}'</b>" if rewrite_result.uses_history else "Query considerata autonoma."
            yield self.log("HISTORY", history_log, color="#3b82f6"), "⏳ *Routing query...*"

            # 3. INTENT ROUTING (Gateway)
            decision = self.router.route(clean_query)
            yield self.log("ROUTER", f"🧭 <b>Intento:</b> {decision.intent}<br><i>{decision.reasoning}</i>", color="#8b5cf6"), "⏳ *Pianificazione...*"

            if decision.intent == "DIRECT_CHAT":
                yield self.log("SISTEMA", "Generazione risposta diretta (No RAG).", color="#2dd4bf"), "⏳ *Generazione...*"
                # 1. Recupera lo storico effettivo della conversazione
                context = self.session_state.get_all_context()
                
                # 2. Costruisci un prompt dinamico che include il ragionamento del router
                system_prompt = f"""You are the final conversational agent. The user's input has been routed to you 
without querying the database or external tools for the following reason:
'{decision.reasoning}'

Recent conversation history:
{context}

INSTRUCTIONS:
- Use the routing reason as internal context to understand how to handle the user (e.g., if the question is adversarial/misleading, handle it diplomatically; if it is small talk, be polite).
- DO NOT repeat the routing reason to the user; use it only to calibrate your tone and response.
- If the user asks follow-up questions (e.g., "are you sure?"), use the conversation history to understand the context of their query."""
                
                # 3. Passa il prompt dinamico al generatore
                response = self.generator.generate(clean_query, system_prompt)
                
                append_history_turn(HISTORY_FILE, user=raw_query, final_query=clean_query, assistant=response.answer)
                yield self.log("SISTEMA", "Completato.", color="#22c55e"), response.answer
                return

            # ---------------------------------------------------------
            # NUOVO BLOCCO: 4. ACTIVE CONTEXT PRUNING (Il Mietitore)
            # ---------------------------------------------------------
            pruned_count = self.session_state.prune_context(clean_query, self.evaluator, threshold=0.20)
            if pruned_count > 0:
                yield self.log("MEMORIA", f"🧹 Potatura: eliminati {pruned_count} documenti obsoleti.", color="#fb923c"), "⏳ *Valutazione cache...*"

            feedback_for_search = ""
            is_resolved = False
            context = self.session_state.get_all_context()
            is_success = False
            final_result = None

            # ---------------------------------------------------------
            # NUOVO BLOCCO: 5. PRE-RETRIEVAL ASSESSMENT (Il Casello)
            # ---------------------------------------------------------
            if context.strip():
                yield self.log("CASELLO", "Valutazione sufficienza del contesto in memoria...", color="#eab308"), "⏳ *Analisi cache...*"
                assessment = self.assessor.evaluate_sufficiency(clean_query, context)
                
                if assessment.is_sufficient:
                    yield self.log("CASELLO", f"⚡ CACHE HIT! Contesto già sufficiente per rispondere.<br><i>{assessment.reasoning}</i>", color="#22c55e"), "⏳ *Generazione risposta...*"
                    is_success = True
                    is_resolved = True # Questo salterà il Loop Agentico!
                    final_result = self.generator.generate(clean_query, context)
                else:
                    yield self.log("CASELLO", f"🔍 Dati in cache parziali. Procedo con la ricerca.<br><i>{assessment.reasoning}</i>", color="#facc15"), "⏳ *Pianificazione...*"
                    # Passiamo al planner l'indizio su cosa manca!
                    feedback_for_search = f"We have some existing context, but it's incomplete: {assessment.reasoning}. Focus your plan on finding the missing information."

            # ---------------------------------------------------------
            # 6. LOOP AGENTICO RAG (Eseguito SOLO se is_resolved è False)
            # ---------------------------------------------------------

            feedback_for_search = ""; is_resolved = False; context = ""; is_success = False; final_result = None

            for global_iter in range(3):
                if is_resolved: break
                
                # --- PIANIFICAZIONE ---
                # Passiamo 'context' al planner
                plan_obj = self.planner.generate_plan(clean_query, context, previous_feedback=feedback_for_search, current_iter=global_iter+1)
                
                plan_log = f"🎯 <b>Strategia (Giro {global_iter+1}):</b> {plan_obj.reasoning}<br>"
                for s in plan_obj.steps: plan_log += f"&nbsp;&nbsp;📍 Step {s.step_number} [{s.tool_name}]: {s.tool_intent}<br>"
                yield self.log("PLANNER", plan_log, color="#60a5fa"), "⏳ *Esecuzione Agent...*"
                
                # --- AGENTE (Tool Execution) ---
                yield self.log("AGENTE", "🤖 Analisi documenti e utilizzo tool...", color="#fb923c"), "⏳ *Recupero dati in corso...*"
                plan_text = "\n".join([f"Step {s.step_number}: use {s.tool_name} - {s.tool_intent}" for s in plan_obj.steps])
                
                # Passiamo 'context' all'agente
                current_context, found_something = self.agent.run(user_query=clean_query, context=context, plan=plan_text, previous_feedback=feedback_for_search)
                
                context = current_context; is_success = found_something
                print("Context:" + "="*50 + f"\n{context}\n" + "="*50 + "\n")
                if not found_something:
                    yield self.log("AGENTE", "⚠️ Nessun dato utile trovato.", color="#facc15"), "Mi dispiace, non ho trovato informazioni ufficiali per rispondere a questa domanda."
                    break

                # --- GENERAZIONE ---
                yield self.log("GENERATOR", "Creazione risposta basata sul contesto...", color="#2dd4bf"), "⏳ *Validazione risposta...*"
                final_result = self.generator.generate(clean_query, context)
                print("Bozza:" + "="*50 + f"\n{final_result}\n" + "="*50 + "\n")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # --- GIUDICE (Evaluation & Loop) ---
                for rewrite_iter in range(3):
                    evaluation = self.judge.evaluate(
                        clean_query, 
                        context, 
                        final_result.answer, 
                        current_iter=rewrite_iter + 1, 
                        max_iter=3
                    )

                    eval_color = "#22c55e" if evaluation.decision == "PASS" else "#ef4444"
                    yield self.log("GIUDICE", f"⚖️ <b>Decisione: {evaluation.decision}</b><br>{evaluation.reasoning}", color=eval_color), "⏳ *Revisione...*"
                    

                    if evaluation.decision == "PASS":
                        is_resolved = True; break
                    elif evaluation.decision == "AMBIGUOUS":
                            print(f"   [GIUDICE] AMBIGUOUS. Necessario chiarimento utente. ({evaluation.reasoning})")
                            # Chiamiamo il metodo speciale per fare la domanda all'utente
                            final_result = self.generator.ask_clarification(clean_query, context, evaluation.reasoning)
                            is_resolved = True # Dichiariamo risolto il loop interno, restituiamo la palla all'utente!
                            break
                    elif evaluation.decision == "REWRITE":
                        final_result = self.generator.revise(clean_query, context, evaluation.reasoning)
                        print("Bozza:" + "="*50 + f"\n{final_result}\n" + "="*50 + "\n")
                    else:
                        # SEARCH_AGAIN: feedback al planner e ricomincia loop globale
                        feedback_for_search = evaluation.reasoning; 
                        self.session_state.clear()
                        break

            # =========================================================
            # 5. FINALIZZAZIONE & LINK INJECTION
            # =========================================================
            if final_result:
                # Prendiamo la risposta formattata direttamente dall'LLM
                final_text = final_result.answer
                
                # --- FIX ELENCHI PUNTATI ---
                # Aggiungiamo i doppi a capo prima degli asterischi
                final_text = final_text.replace(" * ", "\n\n* ")

                # --- B. TRASFORMAZIONE IN LINK HTML (CON FILTRO GOOGLE AI) ---
                source_mapping = self.session_state.source_mapping
                cited_docs_ordered = [] 
                doc_to_new_index = {}   

                if is_success and hasattr(final_result, 'sources_used') and final_result.sources_used:
                    
                    def replace_citation(match):
                        doc_id_str = match.group(1)
                        real_doc_id = doc_id_str if doc_id_str in source_mapping else int(doc_id_str) if int(doc_id_str) in source_mapping else None
                            
                        if real_doc_id is not None:
                            dati = source_mapping[real_doc_id]
                            
                            # --- FILTRO FONTI INDESIDERATE ---
                            titolo_lower = dati.get('titolo', '').lower()
                            if "google ai" in titolo_lower or "risposta rapida" in titolo_lower or "overview" in titolo_lower:
                                return "" 
                            # ---------------------------------
                            
                            if real_doc_id not in doc_to_new_index:
                                cited_docs_ordered.append(real_doc_id)
                                doc_to_new_index[real_doc_id] = len(cited_docs_ordered)
                            
                            new_idx = doc_to_new_index[real_doc_id]
                            url = dati.get('smart_url', dati.get('url', '#'))
                            
                            return f"<sup style='margin-left: 2px;'><a href='{url}' target='_blank' style='text-decoration: none; color: #fb923c; font-weight: bold;'>[{new_idx}]</a></sup>"
                        
                        return match.group(0)

                    # Intercetta i [ X ] generati dall'LLM
                    pattern = re.compile(r'\[\s*(\d+)\s*\]')
                    final_text = pattern.sub(replace_citation, final_text)
                    
                    # 🔴 FIX: Usiamo [ \t]+ invece di \s+
                    # In questo modo puliamo i doppi spazi orizzontali ma SALVIAMO gli a capo (\n)!
                    final_text = re.sub(r'[ \t]+', ' ', final_text).strip()


                # --- COSTRUZIONE UI E BIBLIOGRAFIA ---
                testo_ui = f"{final_text}\n\n"
                
                if cited_docs_ordered:
                    testo_ui += "---\n### 📚 Fonti Utilizzate\n"
                    for idx, doc_id in enumerate(cited_docs_ordered, start=1):
                        dati = source_mapping[doc_id]
                        url = dati.get('smart_url', dati.get('url', '#'))
                        titolo = dati.get('titolo', 'Documento senza titolo').strip()
                        
                        if len(titolo) > 90:
                            titolo = titolo[:87] + "..."
                            
                        testo_ui += f"{idx}. **[{titolo}]({url})**\n"

                append_history_turn(HISTORY_FILE, user=raw_query, final_query=clean_query, assistant=final_result.answer)
                yield self.log("SISTEMA", "Generazione e formattazione completate.", color="#22c55e"), testo_ui
                
        # =========================================================
        # GESTIONE ECCEZIONI: Cattura l'errore 503 o qualsiasi altro crash
        # =========================================================
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n🚨 [CRASH PREVENUTO] Errore critico nel metodo run:\n{error_details}")
            
            error_msg = str(e).lower()
            if "503" in error_msg or "unavailable" in error_msg or "429" in error_msg:
                # Errore API Google
                yield self.log("SERVER OVERLOAD", "I server di Google sono momentaneamente congestionati.", color="#ef4444"), "⚠️ **Server Sovraccarichi:** Il modello AI sta registrando un picco di richieste da parte degli utenti (Errore 503). Per favore, attendi qualche secondo e riprova."
            else:
                # Qualsiasi altro errore Python imprevisto
                yield self.log("ERRORE CRITICO", f"Eccezione: {str(e)}", color="#ef4444"), "🚨 **Errore di sistema inaspettato.** Per favore, riprova o riformula la domanda."


# Istanza per Gradio
agent_system = AgenticRag()

# ==========================================
# 4. INTERFACCIA GRADIO 
# ==========================================
pro_dark_theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate").set(
    body_background_fill="#0f172a", block_background_fill="#1e293b",
    block_border_width="1px", block_border_color="#334155", input_background_fill="#0f172a"
)

minimal_css = """
#log_panel { height: 550px !important; overflow-y: auto !important; padding: 15px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; }
.gradio-container { max-width: 100% !important; padding: 0 20px !important; }

/* Giustifica i paragrafi normali */
.message.bot .prose p {
    text-align: justify !important;
}

/* Stile per gli elenchi puntati */
.message.bot .prose ul {
    list-style-type: disc !important; /* Pallino pieno */
    list-style-position: outside !important; /* Mantiene la seconda riga allineata con la prima! */
    padding-left: 2rem !important; /* Rientro totale dell'elenco verso destra */
    margin-top: 0.8rem !important;
    margin-bottom: 0.8rem !important;
}

.message.bot .prose li {
    text-align: justify !important; /* FORZA IL GIUSTIFICATO ANCHE NELLE LISTE */
    padding-left: 0.3rem !important; /* Distanza tra pallino e testo */
    margin-bottom: 0.6rem !important; /* Spazio tra un punto e l'altro */
    line-height: 1.6 !important;
}

/* Colore e dimensione del pallino */
.message.bot .prose li::marker {
    color: #fb923c !important; /* Arancione DIEM */
    font-size: 1.1em !important;
}
"""

logo_path = os.path.join(SRC_PATH.parent , "logo_ChatBot_DIEM_Gruop5.png")

with gr.Blocks(fill_width=True, css=minimal_css) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=80):
            if os.path.exists(logo_path): gr.Image(logo_path, show_label=False, width=80, container=False)
            else: gr.Markdown("# 🤖")
        with gr.Column(scale=11):
            gr.Markdown("## DIEM Agentic Helper\n*Università degli Studi di Salerno - A.A. 2025/2026*")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Agente Conversazionale", height=520, avatar_images=(None, logo_path if os.path.exists(logo_path) else None), autoscroll=False)
            with gr.Row():
                # Definiamo i componenti
                msg = gr.Textbox(placeholder="Fai una domanda...", container=False, scale=5, autofocus=True,  html_attributes=gr.InputHTMLAttributes(autocorrect="off", spellcheck=False))
                submit_btn = gr.Button("Invia", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Svuota Chat", variant="stop", size="sm")

        with gr.Column(scale=3):
            gr.Markdown("### 🧠 Agentic Reasoning Logs")
            log_display = gr.HTML(value="<div style='color:#64748b; font-style: italic;'>In attesa di input...</div>", elem_id="log_panel")

    # --- LOGICA DI BLOCCO/SBLOCCO ---
    
    def user_action(user_message, history):
        if not user_message.strip(): return gr.update(), history, gr.update(), gr.update()
        history = history or []
        history.append({"role": "user", "content": user_message})
        return gr.update(value="", interactive=False), history, gr.update(interactive=False), gr.update(interactive=False)

    def bot_action(history):
        if not history or history[-1]["role"] != "user": 
            yield history, gr.update(); return
        
        # --- FIX ESTRAZIONE QUERY ---
        raw_content = history[-1]["content"]
        user_query = user_query = raw_content if isinstance(raw_content, str) else " ".join([p["text"] for p in raw_content if "text" in p])

        if isinstance(raw_content, str):
            user_query = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            user_query = " ".join(parts)
        else:
            user_query = str(raw_content)
        
        start_time = time.time()
        history.append({"role": "assistant", "content": "..."})
        
        last_log = ""
        for log_html, partial_answer in agent_system.run(user_query):
            history[-1]["content"] = partial_answer
            last_log = log_html
            yield history, log_html

        total_time = time.time() - start_time
        mins, secs = divmod(int(total_time), 60)
        history[-1]["content"] += f"\n\n---\n*⏱️ Tempo di elaborazione: {mins}m {secs}s*"
        yield history, last_log

    def reenable(): 
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def clear_session():
        agent_system.logs = []
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'w'): pass
                print(" ✅ History fisica svuotata.")
        except Exception as e: print(f"Errore clear: {e}")
        return [], "<div style='color:#64748b;'>Sessione riavviata.</div>"

   # ---------------------------------------------------------
    # GESTIONE EVENTI (Senza JS esterno)
    # ---------------------------------------------------------
    
    # --- JAVASCRIPT PER AUTOSCROLL INTELLIGENTE ---
    smart_scroll_js = """
    function() {
        setTimeout(function() {
            let user_messages = document.querySelectorAll('.message.user, .user');
            if (user_messages.length > 0) {
                let last_user_message = user_messages[user_messages.length - 1];
                last_user_message.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 200); 
    }
    """
    # --- EVENTS ---
    
    # 1. Listener frontend indipendente: fa SOLO lo scroll senza toccare i dati
    submit_btn.click(fn=None, inputs=None, outputs=None, js=smart_scroll_js)
    msg.submit(fn=None, inputs=None, outputs=None, js=smart_scroll_js)

    # 2. Pipeline dati originale intatta: il Python lavora in modo sicuro
    for trigger in [submit_btn.click, msg.submit]:
        trigger(user_action, [msg, chatbot], [msg, chatbot, submit_btn, clear_btn], queue=False)\
            .then(bot_action, [chatbot], [chatbot, log_display])\
            .then(reenable, None, [msg, submit_btn, clear_btn])
    # # --- EVENTS ---
    # for trigger in [submit_btn.click, msg.submit]:
    #     trigger(user_action, [msg, chatbot], [msg, chatbot, submit_btn, clear_btn], queue=False)\
    #         .then(bot_action, [chatbot], [chatbot, log_display])\
    #         .then(reenable, None, [msg, submit_btn, clear_btn])

    clear_btn.click(lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)), None, [msg, submit_btn, clear_btn])\
             .then(clear_session, None, [chatbot, log_display])\
             .then(lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)), None, [msg, submit_btn, clear_btn])

if __name__ == "__main__":
    # Assicurati di non avere altre istanze attive prima di rilanciare
    demo.launch(theme=pro_dark_theme, css=minimal_css)