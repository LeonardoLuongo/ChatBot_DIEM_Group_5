import abc
import time
import random
import urllib.parse
import base64
from typing import List, Tuple, Dict
from langchain_community.utilities import SearxSearchWrapper
from langchain_ollama.chat_models import ChatOllama
from .utils import LRUCache
from .browser import BrowserEngine
from langchain_core.language_models.chat_models import BaseChatModel


class QueryOptimizer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.cache = LRUCache(maxsize=50)

    def optimize(self, query: str) -> str:
        words = query.split()
        operators = [w for w in words if "site:" in w or w == "OR"]
        natural_text = " ".join([w for w in words if "site:" not in w and w != "OR"])
        
        if not natural_text: return query
        if natural_text in self.cache:
            cleaned_text = self.cache[natural_text]
            print(f"   [CACHE] Trovata query ottimizzata: '{cleaned_text}'")
        else:
            print("   [LLM] Ottimizzazione della query in corso...")
            prompt = f"""You are an SEO expert and a search query analyst. 
Your task is to take a user's request (written in natural language for an AI) and convert it into a clean search query that works perfectly for search engines.

RULES:
1. Extract only names, main entities, places, and key concepts.
2. Remove all polite phrases (e.g., "Hello," "Please," "Can you explain").
3. Remove unnecessary stop words (the, a, of, like, why).
4. Do NOT use quotation marks, hyphens, or advanced operators. Just keywords.
5. Return ONLY the final search query string, nothing else.
6. Prioritize Italian language keywords.

User Prompt:
{natural_text}"""
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                if isinstance(content, list):
                    content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
                elif not isinstance(content, str):
                    content = str(content)
                cleaned_text = content.strip().lower()
                self.cache[natural_text] = cleaned_text
            except Exception as e:
                print(f"  ️ Errore LLM: {e}.")
                cleaned_text = natural_text
        return f"{cleaned_text} {' '.join(operators)}".strip()

class BaseSearchEngine(abc.ABC):
    def __init__(self):
        self._last_request_time = 0.0

    def _enforce_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        min_delay = random.uniform(5.0, 8.5)
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        else:
            if 0.5 > random.uniform(0.0, 1.0): time.sleep(random.uniform(0.0, 1.0))
        self._last_request_time = time.time()

    @abc.abstractmethod
    def search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, str]]:
        pass



class GoogleSearch(BaseSearchEngine):
    def __init__(self, browser: BrowserEngine):
        super().__init__()
        self.browser = browser

    def search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, str]]:
        print(f"\n 🕵️‍♂️ [GOOGLE SNIPER] Avvio ricerca per: '{query}' (Target: {max_results})")
        urls = []
        hash_map = {}
        try:
            self._enforce_rate_limit()
            page, sb = self.browser.get_page()
            
            search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&hl=it&gl=IT&filter=0"
            
            # --- GESTIONE ERRORI NAVIGAZIONE ---
            try: 
                page.goto(search_url)
            except Exception as e:
                # Ignoriamo l'errore se abbiamo interrotto noi un caricamento precedente
                if "interrupted" not in str(e).lower() and "aborted" not in str(e).lower():
                    print(f" ⚠️ Errore di navigazione Google: {e}")

            # --- GESTIONE COOKIE BANNER EUROPEO ---
            try:
                page.wait_for_selector("button:has-text('Accetta tutto'), button:has-text('Rifiuta tutto')", timeout=2500)
                print(" 🍪 [COOKIE] Rilevato banner di Google, accetto/rifiuto per proseguire...")
                page.locator("button:has-text('Accetta tutto'), button:has-text('Rifiuta tutto')").first.click()
                page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass 

            # --- SMART WAIT E RENDER GARANTITO ---
            try:
                # Usiamo selettori molto più larghi (h3 per i titoli, a[data-ved] per i link) e state="attached"
                page.wait_for_selector("div#search, div#rso, form#captcha-form, #recaptcha, div#topstuff, h3, a[data-ved]", state="attached", timeout=5000)
            except Exception:
                # Se va in timeout, controlliamo se ci sono comunque link. Se ci sono, non stampiamo l'allarme falso.
                if page.locator("a[href]").count() == 0:
                    print(" ⚠️ [TIMEOUT Google] Selettori non trovati e nessun link presente.")
            
            # --- CONTROLLO NESSUN RISULTATO (FAST FAIL) ---
            if page.locator("div#topstuff").count() > 0 and page.locator("div#search a[href]").count() == 0:
                print(" ∅ [GOOGLE] Nessun risultato trovato per questa query. Interrompo ricerca.")
                return [], {}
            
            # --- GESTIONE INTELLIGENTE CAPTCHA ---
            
            # --- GESTIONE INTELLIGENTE CAPTCHA ---
            if "sorry" in page.url or page.locator("form#captcha-form").count() > 0 or page.locator("#recaptcha").count() > 0:
                print(" 🤖 [ANTI-BOT] Rilevato CAPTCHA di Google! Avvio risoluzione...")
                try:
                    # Usiamo 'sb' locale estratto da self.browser.get_page()
                    sb.solve_captcha()
                    print(" 🔓 [ANTI-BOT] Captcha risolto. Attendo il redirect di Google alla pagina risultati...")
                    page.wait_for_selector("div#search, div#rso", state="visible", timeout=25000)
                    page.wait_for_timeout(1500)
                    print(" ✅ CAPTCHA bypassato e risultati caricati con successo!")
                except Exception as ce:
                    print(f" ❌ Fallita risoluzione CAPTCHA: {ce}")

            # =========================================================
            # STRATEGIA ESTRAZIONE: "VACUUM & FILTER"
            # =========================================================
            print(" 🔍 Ricerca dei link nel DOM (Strategia Vacuum & Filter)...")
            links_locators = page.locator("div#search a[href], div#rso a[href]").all()
            if not links_locators: 
                print(" ⚠️ Fallback locator globale attivato per Google...")
                links_locators = page.locator("a[href]").all()

            # BLACKLIST ESPANSA
            forbidden_domains = [
                "google.", "youtube.com", "blogger.com", 
                "support.google", "accounts.google", "translate.google", 
                "webcache.googleusercontent", "policies.google"
            ]

            for link in links_locators:
                try: 
                    href = link.get_attribute("href")
                except Exception: 
                    continue
                
                if not href: 
                    continue

                # --- DECODIFICA REDIRECT INTERNI DI GOOGLE ---
                if "/url?" in href:
                    try:
                        parsed = urllib.parse.urlparse(href)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if 'q' in qs:
                            href = qs['q'][0]
                        elif 'url' in qs:
                            href = qs['url'][0]
                    except Exception:
                        pass
                
                if "#" in href:
                    print("debug print")
                    
                original_href = href
                clean_href = original_href.split('#')[0]

                if not clean_href.startswith("http"): 
                    continue
                
                if any(domain in clean_href.lower() for domain in forbidden_domains): 
                    continue
                
                # --- DEDUPLICAZIONE E POPOLAMENTO DIZIONARIO ---
                if clean_href not in urls:
                    urls.append(clean_href)
                    # Manteniamo la logica originaria: assegniamo a hash_map locale
                    hash_map[clean_href] = original_href
                    if len(urls) >= max_results: 
                        break
                        
        except Exception as e:
            print(f" ❌ Errore critico in Google Sniper: {e}")
            
        finally:
            if 'page' in locals():
                try: 
                    page.evaluate("window.stop()")
                except Exception: 
                    pass
                    
        print(f" └── ✅ Trovati {len(urls)} risultati organici validi su Google.")
        return urls, hash_map

class BingSearch(BaseSearchEngine):
    def __init__(self, browser: BrowserEngine):
        super().__init__()
        self.browser = browser

    def search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, str]]:
        print(f"\n   [BING SNIPER] Avvio ricerca per: '{query}' (Target: {max_results})")
        urls = []
        hash_map = {}
        try:
            self._enforce_rate_limit()
            page, sb = self.browser.get_page()
            
            search_url = f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
            try:
                page.goto(search_url)
            except Exception as e:
                if "interrupted" not in str(e).lower() and "aborted" not in str(e).lower():
                    print(f"  ⚠️ Errore di navigazione Bing: {e}")

            # --- SMART WAIT E RENDER GARANTITO ---
            try:
                # Aggiunti selettori universali (main, h2 per i titoli) e timeout ridotto
                page.wait_for_selector("li.b_algo, div#b_captcha, iframe[src*='captcha'], .b_no, main, h2", state="attached", timeout=5000)
            except Exception:
                if page.locator("a[href]").count() == 0:
                    print("  ⚠️ [TIMEOUT Bing] Selettori non trovati e nessun link presente.")

            # --- CONTROLLO NESSUN RISULTATO (FAST FAIL) ---
            if page.locator(".b_no").count() > 0:
                print(" ∅ [BING] Nessun risultato trovato per questa query. Interrompo ricerca.")
                return [], {}
            
            # --- GESTIONE INTELLIGENTE CAPTCHA ---
            if "captcha" in page.url or page.locator("div#b_captcha").count() > 0 or page.locator("iframe[src*='captcha']").count() > 0:
                print("  ⚠️ [ANTI-BOT] Rilevato CAPTCHA di Bing! Avvio risoluzione...")
                try:
                    if "captcha/challenge" not in page.url:
                        page.goto("https://www.bing.com/turing/captcha/challenge")
                        page.wait_for_load_state("networkidle", timeout=3000)
                    
                    sb.solve_captcha()
                    sb.wait_for_element_absent("input[disabled]", timeout=15)
                    
                    print("   [ANTI-BOT] Captcha risolto. Ricarico la ricerca originale...")
                    page.goto(search_url)
                    
                    page.wait_for_selector("li.b_algo", state="visible", timeout=25000)
                    page.wait_for_timeout(1500)
                    
                    print("   CAPTCHA bypassato e risultati caricati con successo!")
                except Exception as ce:
                    print(f"   Fallita risoluzione CAPTCHA: {ce}")

            # --- ESTRAZIONE LINK ---
            links_locators = page.locator("li.b_algo h2 a").all()
            for link in links_locators:
                href = link.get_attribute("href")
                
                if not href:
                    continue

                # --- DECODER BING TRACKING URL ---
                if "bing.com/ck" in href:
                    try:
                        parsed = urllib.parse.urlparse(href)
                        query_params = urllib.parse.parse_qs(parsed.query)
                        
                        if 'u' in query_params:
                            u_param = query_params['u'][0]
                            if u_param.startswith('a1'):
                                u_param = u_param[2:]
                            
                            u_param += "=" * ((4 - len(u_param) % 4) % 4)
                            decoded_bytes = base64.urlsafe_b64decode(u_param)
                            decoded_url = decoded_bytes.decode('utf-8', errors='ignore')
                            
                            if decoded_url.startswith("http"):
                                href = decoded_url
                    except Exception as e:
                        print(f"  ⚠️ [BING DECODER] Impossibile decodificare: {e}")

                # --- SALVATAGGIO URL ---
                if "#" in href:
                    print("debug print")
                
                original_href = href
                clean_href = original_href.split('#')[0]

                if clean_href.startswith("http") and clean_href not in urls:
                    urls.append(clean_href)
                    hash_map[clean_href] = original_href
                    if len(urls) >= max_results:
                        break
        
        except Exception as e:
            print(f"   Errore critico in Browser Sniper: {e}")
            
        finally:
            if 'page' in locals():
                try:
                    page.evaluate("window.stop()")
                except Exception:
                    pass
            
            print(f"  ├── ️ Trovati {len(urls)} risultati organici validi.")
            
        return urls, hash_map

from typing import List, Tuple, Dict

class SearxSearch(BaseSearchEngine):
    def __init__(self, wrapper: SearxSearchWrapper): # Sostituisci SearxSearchWrapper con il tipo corretto se serve
        super().__init__()
        self.wrapper = wrapper

    def search(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, str]]:
        urls = []
        hash_map = {}
        try:
            fetch_count = max(15, max_results) 
            self._enforce_rate_limit()
            results = self.wrapper.results(query, num_results=fetch_count)
            
            # --- INIZIO SEZIONE DEBUG MOTORI REINTEGRATA ---
            print(f"\n  [ DEBUG SEARXNG] Analisi fornitura tramite LangChain per la query: '{query}'")
            
            # Contiamo quanti risultati ha portato ogni motore
            engine_stats = {}
            for r in results:
                engines_used = r.get("engines", ["sconosciuto"])
                for eng in engines_used:
                    engine_stats[eng] = engine_stats.get(eng, 0) + 1
            
            if engine_stats:
                print("  ├── Risultati forniti da:")
                for eng, count in sorted(engine_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"  │   ⚙️ {eng.capitalize()}: {count} risultati")
            else:
                print("  ├── ⚠️ Nessun motore ha restituito risultati validi.")

            print("  └── ✅ Interrogazione gestita con successo da LangChain.")
            # --- FINE SEZIONE DEBUG MOTORI ---

            # Estraiamo gli URL e li deduplichiamo
            for r in results:
                link = r.get("link")
                if link:
                    # Reintegrato il debug per i frammenti
                    if "#" in link:
                        print("debug print")

                    original_link = link
                    clean_link = original_link.split('#')[0]
                    
                    if clean_link not in urls:
                        urls.append(clean_link)
                        hash_map[clean_link] = original_link
                        
        except Exception as e:
            print(f"   ⚠️ Errore API SearxNG tramite LangChain: {e}")
            
        return urls[:max_results], hash_map



class HybridSearchOrchestrator:
    def __init__(self, engines: List['BaseSearchEngine']):
        # Riceve solo la lista dei motori (es. Google, Bing, Searx)
        self.engines = engines

    def search_all(self, query: str, max_results: int) -> Tuple[List[str], Dict[str, str]]:
        """Cascata intelligente dinamica con logica a riempimento (Bucket Fill)."""
        urls = []
        hash_map = {}

        for i, engine in enumerate(self.engines):
            engine_name = engine.__class__.__name__.replace('Search', '')
            
            if len(urls) >= max_results:
                break
                
            needed = max_results - len(urls)
            
            # --- LOGICA DI LOGGING DELLA CASCATA ---
            if i > 0:
                if len(urls) == 0:
                    print(f"\n   [CASCATA] Motori precedenti bloccati (0/{max_results}). Attivo {engine_name} per tutti i {needed}...")
                else:
                    print(f"\n   [CASCATA] I motori precedenti hanno fornito {len(urls)}/{max_results} link. Passo a {engine_name} per i restanti {needed}...")

            # --- ESECUZIONE RICERCA (usa la query grezza coi Dorks passata dal WebExplorer) ---
            engine_urls, engine_map = engine.search(query, needed)
            
            for u in engine_urls:
                if u not in urls:
                    urls.append(u)
                    if u in engine_map:
                        hash_map[u] = engine_map[u]
        
        # --- LOG DI SUFFICIENZA/SALTO ---
        if len(urls) >= max_results and i < len(self.engines) - 1:
            motori_usati = "+".join([e.__class__.__name__.replace('Search', '') for e in self.engines[:i+1]])
            print(f"   [CASCATA] {motori_usati} sufficienti ({len(urls)} link). Salto i motori successivi.")
        
        return urls[:max_results], hash_map