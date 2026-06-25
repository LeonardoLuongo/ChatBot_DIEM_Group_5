import os
import gc
import ssl
import time
import torch
import traceback
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SearxSearchWrapper

# --- RAG Components ---
from WebBasedRAG.RAGComponents.judge import AnswerJudge
from WebBasedRAG.RAGComponents.planner import QueryPlanner
from WebBasedRAG.RAGComponents.agent import SupervisorAgent
from WebBasedRAG.RAGComponents.generator import AnswerGenerator
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.embeddings.gte_reranker import BGE8KReranker
from WebBasedRAG.RAGComponents.relevance_evaluator import RelevanceEvaluator
from WebBasedRAG.RAGComponents.embeddings.nomic_embeddings import NomicEmbeddings
from WebBasedRAG.RAGComponents.embeddings.emdebbings_interface import BaseEmbeddings
from WebBasedRAG.RAGComponents.tools.providers.math_provider import MathToolProvider

# --- Modular WebExplorer ---
from WebBasedRAG.RAGComponents.router import IntentRouter
from WebBasedRAG.RAGComponents.web_explorer_components.explorer import WebExplorer
from WebBasedRAG.RAGComponents.web_explorer_components.browser import BrowserEngine
from WebBasedRAG.RAGComponents.web_explorer_components.utils import DocumentChunker
from WebBasedRAG.RAGComponents.web_explorer_components.retrieval import RetrievalEngine
from WebBasedRAG.RAGComponents.web_explorer_components.scraping import HTMLCrawler, PDFExtractor
from WebBasedRAG.RAGComponents.web_explorer_components.links import SmartLinkGenerator, PDFHighlightServer
from WebBasedRAG.RAGComponents.web_explorer_components.search import QueryOptimizer, GoogleSearch, BingSearch, SearxSearch, HybridSearchOrchestrator

# --- Tools --- 
from WebBasedRAG.RAGComponents.tools.core.registry import ToolRegistry
from WebBasedRAG.RAGComponents.tools.services.qdrant_service import QdrantService
from WebBasedRAG.RAGComponents.tools.providers.qdrant_provider import QdrantToolProvider
from WebBasedRAG.RAGComponents.tools.services.easycourse_service import EasyCourseService
from WebBasedRAG.RAGComponents.tools.services.professors_service import ProfessorsService
from WebBasedRAG.RAGComponents.tools.providers.datetime_provider import DateTimeToolProvider
from WebBasedRAG.RAGComponents.tools.providers.websearch_provider import WebSearchToolProvider
from WebBasedRAG.RAGComponents.tools.providers.easycourse_provider import EasyCourseToolProvider
from WebBasedRAG.RAGComponents.tools.providers.professors_provider import ProfessorsToolProvider

if os.getenv("IS_LANGFUSE_ENABLED"):
    print("LANGFUSE: ENABLED")
    from analytics.langfuse_dashboard import init_langfuse
    from langfuse import observe
    langfuse_handler = init_langfuse()


def get_basic_explorer(llm: ChatOllama, embeddings: BaseEmbeddings, evaluator: RelevanceEvaluator):
    # Fix SSL Globale
    ssl._create_default_https_context = ssl._create_unverified_context
    BROWSER_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    # Inizializzazione Sotto-Moduli
    browser = BrowserEngine(BROWSER_PATH)
    query_opt = QueryOptimizer(llm)
    
    # Pipeline di Ricerca
    google = GoogleSearch(browser)
    bing = BingSearch(browser)
    searx = SearxSearch(SearxSearchWrapper(searx_host="http://127.0.0.1:8080", categories=["general"]))
    search_orch = HybridSearchOrchestrator([google, bing, searx])

    return WebExplorer(
        query_optimizer=query_opt,
        search_orchestrator=search_orch,
        html_crawler=HTMLCrawler(),
        pdf_extractor=PDFExtractor(),
        chunker=DocumentChunker(),
        retrieval_engine=RetrievalEngine(embeddings, evaluator),
        smart_link_gen=SmartLinkGenerator(),
        pdf_server=PDFHighlightServer()
    ), browser  # <-- MODIFICA: Restituisco anche il browser per poterlo chiudere in caso di errore

# Aggiungiamo 'evaluator' ai parametri
def get_all_tools(web_explorer: WebExplorer, session_state: GlobalSessionState, qdrant_service: QdrantService, evaluator: RelevanceEvaluator, tool_llm: ChatOllama):
    datetime_provider = DateTimeToolProvider()
    
    # INIEZIONE DEL RERANKER (evaluator) E DELL'LLM NEL SERVICE!
    easycourse_service = EasyCourseService(llm=tool_llm, evaluator=evaluator)
    easycourse_provider = EasyCourseToolProvider(service=easycourse_service, session_state=session_state)

    websearch_provider = WebSearchToolProvider(web_explorer=web_explorer, debug_mode=True, session_state=session_state)
    professors_provider = ProfessorsToolProvider(service=ProfessorsService(), session_state=session_state, web_explorer=web_explorer)
    math_provider = MathToolProvider(session_state=session_state)
    
    registry = ToolRegistry()
    registry.register_provider(datetime_provider)
    registry.register_provider(easycourse_provider)
    
    if qdrant_service:
        qdrant_provider = QdrantToolProvider(service=qdrant_service, session_state=session_state, evaluator=evaluator)
        registry.register_provider(qdrant_provider)
    
    registry.register_provider(websearch_provider)
    registry.register_provider(professors_provider)
    registry.register_provider(math_provider)

    return registry, registry.get_all_tools()



def initialize_system(gen_llm, agent_llm, tools_llm, judge_llm, planner_llm, explorer_llm, embeddings_model, evaluator, session_state, qdrant_service, agent_max_retries=3):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"   -> Setup Ambiente Web (Tentativo {attempt + 1}/{max_retries})...")
            explorer, browser_engine = get_basic_explorer(llm=explorer_llm, embeddings=embeddings_model, evaluator=evaluator)
            
            # Passiamo 'evaluator' a get_all_tools
            registry, tools = get_all_tools(explorer, session_state, qdrant_service, evaluator, tools_llm)
            
            agent = SupervisorAgent(base_llm=agent_llm, tools=tools, session_state=session_state, max_retries=agent_max_retries)
            judge = AnswerJudge(llm=judge_llm)
            planner = QueryPlanner(llm=planner_llm, tools=tools) # Ricordati i tools come concordato prima!
            generator = AnswerGenerator(llm=gen_llm) # <-- NUOVO: inizializziamo il Generator
            
            # ---> NUOVO: Inizializziamo il Router con l'agent_llm (temperature=0.0) per massima precisione deterministica
            router = IntentRouter(llm=agent_llm) 
            
            return explorer, browser_engine, agent, judge, planner, generator, router # <--- Ritorna anche il router
        except Exception as e:
            print(f"   ⚠️ Errore di inizializzazione: {e}")
            time.sleep(2)


def run_qa_bot():
    print("\n" + "*"*60)
    print(" AVVIO MODALITÀ SOTA RAG: AGENT + ROUTING JUDGE")
    print("*"*60)

    # 1. Caricamento Modelli Base (avviene una sola volta)
    print("   -> Caricamento Modelli AI (GPU/CPU)...")
    reranker_model = BGE8KReranker()
    explorer_llm = ChatOllama(
        model="qwen2.5:7b-instruct",
        keep_alive="15m"
    )
    agent_llm = ChatOllama(
        model="qwen2.5:7b-instruct",
        keep_alive="15m",
        temperature=0.0
    )
    judge_llm = ChatOllama(
        model="granite4.1:8b",
        temperature=0.0,
        # top_p=0.9,
        # top_k=64,
        keep_alive="0s"
    )
    gen_llm = ChatOllama(
        model="granite4.1:8b",
        temperature=0.5,
        keep_alive="15m"
    )

    embeddings_model = NomicEmbeddings()
    evaluator = RelevanceEvaluator(reranker_model)
    
    # =========================================================
    # INIZIALIZZAZIONE QDRANT VECTOR STORE
    # =========================================================
    # ATTENZIONE: Questo deve essere il path alla cartella che CONTIENE il manifest.json
    # In base al tuo path, dovrebbe essere la root del DB
    QDRANT_STORE_DIR = os.getenv("DIEM_QDRANT_STORE_DIR", r"qdrant_trafilatura_1800_d6") 
    
    try:
        qdrant_service = QdrantService(store_dir_path=QDRANT_STORE_DIR)
    except Exception as e:
        print(f" ⚠️ [QDRANT WARN] Impossibile avviare il Vector DB: {e}. Il tool non sarà disponibile.")
        qdrant_service = None
    # =========================================================
    
    session_state = GlobalSessionState()
    # ---> NUOVO: spacchettiamo anche il 'router'
    explorer, browser_engine, agent, judge, planner, generator, router = initialize_system( 
        gen_llm=gen_llm, agent_llm=agent_llm, tools_llm=agent_llm, judge_llm=judge_llm, planner_llm=judge_llm, 
        explorer_llm=explorer_llm, embeddings_model=embeddings_model, evaluator=evaluator,
        session_state=session_state, qdrant_service=qdrant_service, agent_max_retries=10
    )



    # 3. Loop di Chat
    while True:
        try:
            print("[SISTEMA ONLINE]:")
            query = input("\n> ")
            if query.lower() in ['exit', 'quit', 'esci']:
                break
            if not query.strip(): continue

            session_state.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.time()
            
            # =========================================================
            # GATEWAY: QUERY ROUTING
            # =========================================================
            decision = router.route(query)
            print(f"   🧭 [GATEWAY] Decisione: {decision.intent} ({decision.reasoning})")

            if decision.intent == "DIRECT_CHAT":
                print("   💬 [ORCHESTRATORE] Flusso deviato: Generazione risposta diretta...")
                
                # Usiamo il generator. Passiamo un contesto vuoto o generico.
                response = generator.generate(query, "Contesto vuoto. Rispondi cortesemente all'utente ma ricordagli il tuo scopo se necessario.")
                
                elapsed_time = time.time() - start_time
                print("\n" + "*"*30)
                print(f"✅ BOT (Risposta Diretta in {elapsed_time:.1f}s):\n")
                
                # MODIFICA QUI: usiamo .answer invece di .content
                print(response.answer) 
                print("*"*30)
                
                continue


            # =========================================================
            # FLUSSO NORMALE RAG (DOMAIN_SEARCH)
            # =========================================================

            
            # VARIABILI DI STATO
            max_global_iterations = 3
            max_rewrite_iterations = 3
            current_global_iter = 0
            
            feedback_for_search = ""
            final_result = None
            context = ""
            is_success = False
            is_resolved = False 
            last_evaluation = None 

            # LOOP GLOBALE (Ricerca)
            while current_global_iter < max_global_iterations and not is_resolved:
                current_global_iter += 1
                if current_global_iter > 1:
                    print(f"\n [ORCHESTRATORE] Avvio iterazione globale {current_global_iter}/{max_global_iterations} con nuova strategia...")
                
                # --- 1. PIANIFICAZIONE ---
                plan_obj = planner.generate_plan(
                    query, 
                    previous_feedback=feedback_for_search,
                    current_iter=current_global_iter,
                    max_iter=max_global_iterations
                )

                
                # Stampiamo la strategia a schermo tipizzata
                print(f"\n   🧠 [STRATEGIA D'AZIONE]: {plan_obj.reasoning}")
                for step in plan_obj.steps:
                    print(f"     Step {step.step_number} [{step.tool_name}]: {step.tool_intent}")
                
                # Trasformiamo la To-Do list in direttive imperative per l'Agente
                plan_text = "\n".join([f"STEP {step.step_number} - REQUIRED TOOL: '{step.tool_name}'\nINSTRUCTIONS: {step.tool_intent}\n" for step in plan_obj.steps])

                # --- 2. ESECUZIONE (Data Ingestion) ---
                current_context, found_something = agent.run(
                    user_query=query, 
                    plan=plan_text,  # <- Passiamo il piano!
                    previous_feedback=feedback_for_search
                )
                context = current_context
                is_success = found_something
                print(f"Final Context:\n{context}\n")
                
                # Se non trova contesto al primo giro, è inutile farlo sbattere contro il giudice
                if not found_something:
                    print("   ⚠️ Nessun contesto utile trovato dai tool. Mi fermo per evitare allucinazioni.")
                    break 
                
                # --- 3. GENERAZIONE INIZIALE (RAG) ---
                final_result = generator.generate(query, context)

                print("*"*50 + f"DRAFT:\n\n{final_result}\n\n" + "*"*50 )

                # --- 4. VALUTAZIONE E CORREZIONE (Giudice) ---
                current_rewrite_iter = 0
                while current_rewrite_iter < max_rewrite_iterations:
                    current_rewrite_iter += 1
                    
                    evaluation = judge.evaluate(
                        query, 
                        context, 
                        final_result.answer, 
                        current_iter=current_rewrite_iter, 
                        max_iter=max_rewrite_iterations
                    )

                    last_evaluation = evaluation
                    
                    if evaluation.decision == "PASS":
                        print(f"   [GIUDICE] PASS. Risposta validata. ({evaluation.reasoning})")
                        is_resolved = True 
                        break 
                        
                    elif evaluation.decision == "REWRITE":
                        print(f"  [GIUDICE] REWRITE (Tentativo {current_rewrite_iter}/{max_rewrite_iterations}). ({evaluation.reasoning})")
                        if current_rewrite_iter < max_rewrite_iterations:
                            # La riscrittura la fa il GENERATOR
                            final_result = generator.revise(query, context, evaluation.reasoning)
                        else:
                            print("  [ORCHESTRATORE] Finiti i tentativi di riscrittura per questo contesto.")
                            # --- MODIFICA CRITICA: AGGIORNIAMO IL FEEDBACK PER IL PLANNER ---
                            feedback_for_search = (
                                f"I already tried to answer with the previous context but failed. "
                                f"The Auditor said: '{evaluation.reasoning}'. "
                                f"Since the context was useless, you MUST change your search strategy. "
                                f"If you used 'SearchInternalDatabase' before, you MUST skip it now and directly use 'WebSearch'."
                            )
                            break 
                            
                    elif evaluation.decision == "SEARCH_AGAIN":
                        print(f"   [GIUDICE] SEARCH AGAIN. Contesto inutile. ({evaluation.reasoning})")
                        feedback_for_search = evaluation.reasoning
                        break 

            elapsed_time = time.time() - start_time

            # GESTIONE FALLIMENTI (Giudice)
            if not is_resolved and final_result is not None and is_success:
                print("\n" + "-"*15)
                print(" [ORCHESTRATORE] Limite tentativi raggiunto. La risposta potrebbe essere imprecisa.")
                print(f"   Ultimo Giudizio: {last_evaluation.decision if last_evaluation else 'N/D'}")
                print(f"   Motivazione: {last_evaluation.reasoning if last_evaluation else 'N/D'}")
                print("-"*15)

            # STAMPA FINALE
            if final_result is not None:
                print("\n" + "*"*30)
                status_icon = "✅" if is_resolved else "⚠️"
                print(f"{status_icon} BOT (Tempo: {elapsed_time:.1f}s):\n")
                
                # =========================================================
                # ALGORITMO DI POST-PROCESSING (Link Injection)
                # =========================================================
                final_text = final_result.answer
                links_found = []
                
                if is_success and hasattr(final_result, 'sources_used') and final_result.sources_used:
                    source_mapping = session_state.source_mapping
                    
                    # 1. Raccogliamo tutti i link dai documenti EFFETTIVAMENTE usati
                    all_possible_links = []
                    for doc_id in set(final_result.sources_used):
                        if doc_id in source_mapping:
                            all_possible_links.extend(source_mapping[doc_id].get('external_links', []))
                    
                    # 2. Filtriamo link inutili (es. "clicca qui") e ordiniamo dal più lungo al più corto
                    stop_words = {"clicca", "qui", "link", "scarica", "download", "home", "sito", "http", "https"}
                    valid_links = []
                    for link in all_possible_links: # Rimuovi duplicati basati sull'URL
                        anchor = link['text'].strip()
                        if len(anchor) > 4 and anchor.lower() not in stop_words:
                            if not any(v['url'] == link['url'] for v in valid_links):
                                valid_links.append(link)
                    
                    valid_links.sort(key=lambda x: len(x['text']), reverse=True)
                    
                    # 3. Cerchiamo la parola nel testo e la sostituiamo iniettando [X]
                    import re
                    link_counter = 1
                    for link in valid_links:
                        anchor = link['text']
                        # Regex per trovare l'anchor ignorando Maiuscole/Minuscole
                        pattern = re.compile(re.escape(anchor), re.IGNORECASE)
                        
                        # Se l'anchor è presente nel testo generato
                        if pattern.search(final_text):
                            # Sostituisci solo la PRIMA occorrenza per evitare spam visivo
                            final_text = pattern.sub(f"**{anchor}** [{link_counter}]", final_text, count=1)
                            links_found.append((link_counter, anchor, link['url']))
                            link_counter += 1
                
                # --- STAMPA DEL TESTO ARRICCHITO ---
                print(final_text)
                
                print("\n FONTI UTILIZZATE:")
                if is_success and hasattr(final_result, 'sources_used') and final_result.sources_used:
                    
                    # Stampa documenti
                    for doc_id in sorted(set(final_result.sources_used)):
                        if doc_id in source_mapping:
                            dati = source_mapping[doc_id]
                            link_da_stampare = dati.get('smart_url', dati['url'])
                            print(f"   [Doc {doc_id}] {dati['titolo'][:60]}...")
                            print(f"         <{link_da_stampare}>")
                    
                    # Stampa Link Esterni recuperati
                    if links_found:
                        print("\n   🔗 Link esterni citati nel testo:")
                        for idx, anchor, url in links_found:
                            print(f"      [{idx}] {anchor}: <{url}>")
                            
                else:
                    print("   Nessuna fonte specifica utilizzata.")
                print("*"*30)

        except KeyboardInterrupt:
            print("\nChiusura forzata dall'utente...")
            break
        except Exception as e:
            errore_completo = traceback.format_exc()
            print(f"\n 🚨 Si è verificato un errore critico: {str(e)}")
            print(f"Dettagli tecnici del Crash:\n{errore_completo}")
            
            print(" 🔄 Tentativo di ripristino del motore web in corso...")
            try:
                if browser_engine:
                    browser_engine.close_browser()
                explorer, browser_engine, agent, judge, planner, generator = initialize_system( 
                    gen_llm=gen_llm, agent_llm=agent_llm, tools_llm=agent_llm, judge_llm=judge_llm, planner_llm=judge_llm, 
                    explorer_llm=explorer_llm, embeddings_model=embeddings_model, evaluator=evaluator,
                    session_state=session_state, qdrant_service=qdrant_service, agent_max_retries=10
                )
                print(" ✅ Sistema ripristinato con successo! Prova a riformulare la domanda.")
            except Exception as reset_error:
                print(f" 🛑 Fallimento critico nel ripristino: {reset_error}")
                print(" Ti consiglio di chiudere e riavviare lo script.")
                break

def optional_observe():
    def decorator(func):
        if os.getenv("IS_LANGFUSE_ENABLED"):
            return observe()(func)
        return func
    return decorator

@optional_observe()
def main():
    run_qa_bot()

if __name__ == "__main__":
    main()
