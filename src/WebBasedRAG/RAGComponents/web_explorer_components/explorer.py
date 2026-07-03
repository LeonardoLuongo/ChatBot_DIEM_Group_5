import concurrent.futures
import re
from typing import List, Dict, Tuple
from urllib.parse import urljoin
import trafilatura
from lxml import html

import concurrent.futures
from rapidfuzz import fuzz
from typing import List, Dict, Tuple
from .utils import LRUCache, DocumentChunker
from .search import QueryOptimizer, HybridSearchOrchestrator
from .scraping import HTMLCrawler, PDFExtractor
from .retrieval import RetrievalEngine
from .links import SmartLinkGenerator, PDFHighlightServer

from playwright.sync_api import sync_playwright

import time

def pre_clean_unisa_html(raw_html, base_url):
    """La tua funzione di pulizia HTML per domini Unisa"""
    if not raw_html: return None
    
    try:
        # STRATEGIA ANTI-CRASH: Rimuoviamo eventuali intestazioni XML (<?xml...?>)
        # che confondono lxml e Trafilatura portando all'errore "Document object"
        if isinstance(raw_html, bytes):
            raw_html_str = raw_html.decode('utf-8', errors='ignore')
        else:
            raw_html_str = raw_html
            
        raw_html_str = re.sub(r'^\s*<\?xml[^>]*\?>', '', raw_html_str, flags=re.IGNORECASE)
        tree = html.fromstring(raw_html_str)
        
        # 🚀 FIX DEFINITIVO: Trasformiamo tutti i link relativi della pagina in assoluti!
        if base_url:
            tree.make_links_absolute(base_url)
            
    except Exception as e:
        print(f"   [ERRORE HTML] Parsing fallito per {base_url}: {e}")
        return None
    
    # =========================================================================
    # --- STRATEGIA ANTI-CRASH (LXML / TRAFILATURA) ---
    # Rimuoviamo preventivamente tutti i commenti HTML e Processing Instructions
    # perché l'iteratore interno di Trafilatura tenta di modificare la proprietà '.tag' 
    # che nei Commenti di lxml è di sola lettura (è una funzione), causando crash.
    # =========================================================================
    for bad_node in tree.xpath('//comment() | //processing-instruction()'):
        if bad_node.getparent() is not None:
            bad_node.getparent().remove(bad_node)

    # 1. Pulizia Strutturale
    xpaths_to_remove = [
        '//div[@id="skiptocontent"]', '//div[@id="shortcuts"]', 
        '//div[contains(@class, "hidden") and @role="navigation"]', 
        '//nav', '//footer', '//aside', '//div[@id="unisa-left-menu"]', 
        '//div[@id="unisa-right-menu"]', '//div[contains(@class, "menu-laterale")]', 
        '//div[contains(@class, "b-calendar")]', '//table[contains(@class, "ui-datepicker")]', 
        '//a[contains(@class, "carousel-control")]', '//div[contains(@class, "modal")]', 
        '//ul[contains(@class, "pagination")]', '//*[contains(@class, "breadcrumb")]',
        '//*[contains(@id, "cookie")]', '//*[contains(@class, "cookie")]', 
        '//div[contains(@class, "cc-window")]', '//div[contains(@class, "cc-banner")]',
        '//div[@id="app-disclaimer"]'
    ]
    for xp in xpaths_to_remove:
        for bad_el in tree.xpath(xp):
            if bad_el.getparent() is not None:
                bad_el.getparent().remove(bad_el)

    # 2. Accordion e Tabelle (FIX SCRAMBLED CONTEXT)
    # Copiamo il titolo dell'Accordion e lo incolliamo FISICAMENTE dentro il contenuto
    for panel in tree.xpath('//div[contains(@class, "panel")]'):
        heading = panel.xpath('.//div[contains(@class, "panel-heading")]')
        # Cerca il corpo, che sia collapse o normale
        content_container = panel.xpath('.//div[contains(@class, "panel-collapse")] | .//div[contains(@class, "panel-body")]')
        
        if heading and content_container:
            title_text = heading[0].text_content().strip()
            if title_text:
                # Creiamo un finto titolo <h3> da ficcare DENTRO il box nascosto
                new_title = html.Element("h3")
                new_title.text = f"RIGUARDO {title_text.upper()}:"
                # Incolla il titolo fisicamente dentro il box di testo
                content_container[0].insert(0, new_title)

    for hidden_el in tree.xpath('//div[contains(@class, "collapse")]'):
        classes = hidden_el.get("class", "")
        hidden_el.set("class", classes.replace("collapse", "").replace("in", ""))
        for attr in ["style", "aria-expanded"]:
            if attr in hidden_el.attrib: 
                del hidden_el.attrib[attr]

    for title_link in tree.xpath('//h3[contains(@class, "panel-title")]/a | //h4[contains(@class, "panel-title")]/a'):
        title_link.tag = "span"       
        title_link.attrib.clear()

    # 3. TABELLE: TRASFORMA IN PARAGRAFI PER SALVARE LINK E DATI (1, 2 o 3 colonne)
    for table in tree.xpath('//table'):
        div = html.Element("div")
        for row in table.xpath('.//tr'):
            cells = row.xpath('.//td | .//th')
            
            if len(cells) == 0:
                continue
                
            p = html.Element("p")
            
            # Assegnazione dinamica sicura
            if len(cells) >= 3:
                chiave = cells[1].text_content().strip()
                cella_valore = cells[2]
            elif len(cells) == 2:
                chiave = cells[0].text_content().strip()
                cella_valore = cells[1]
            else:
                chiave = ""
                cella_valore = cells[0]
            
            valore_pulito = " ".join(cella_valore.text_content().strip().split())
            
            if chiave or valore_pulito:
                if chiave:
                    b = html.Element("b")
                    b.text = f"{chiave}: "
                    p.append(b)
                
                if valore_pulito:
                    testo_nodo = html.Element("span")
                    testo_nodo.text = valore_pulito + " "
                    p.append(testo_nodo)
                
                # Salvataggio dei link
                for a in cella_valore.xpath('.//a'):
                    href = a.get('href')
                    if href:
                        nuovo_a = html.Element("a", href=href)
                        nuovo_a.text = a.text_content().strip() or "Link"
                        nuovo_a.tail = " "
                        p.append(nuovo_a)
                        
                div.append(p)
                
        if len(div) > 0: 
            table.getparent().replace(table, div)

    # 4. Pulizia UI Testuale
    ui_garbage = ["torna su", "condividi", "previous", "next", "×", "‹", "›", "tutte le news", "agenda", "ultime news", "area riservata personale", "area riservata personale diem", "dettaglio", "filtro", "cerca"]
    for el in tree.xpath('//a | //button | //span | //i | //h1 | //h2 | //h3 | //h4 | //h5 | //h6 | //p'):
        if el.getparent() is None: continue
        if el.text_content().strip().lower().strip('-').strip() in ui_garbage:
            if el.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']: el.getparent().remove(el)
            else: el.drop_tree()

    # (Passaggio 4 rimosso: Trafilatura estrarrà nativamente i link markdown)

    return tree

def extract_and_clean_links(md_text: str) -> tuple[str, list]:
    """Rimuove la sintassi markdown dai link, lasciando il testo pulito, ed estrae i metadati."""
    extracted_links = []
    
    def replacer(match):
        anchor = match.group(1).strip()
        url = match.group(2).strip()
        
        # Evitiamo link interni, javascript o ancore vuote
        if not url.startswith('#') and not url.startswith('javascript'):
            # Salviamo il link se non è già presente
            if not any(item['url'] == url for item in extracted_links):
                extracted_links.append({'text': anchor, 'url': url})
                
        # Restituiamo SOLO il testo pulito al LLM
        return anchor

    # Rimuoviamo eventuali vecchi residui di [URL: ...]
    md_text = re.sub(r'\s*\[URL:\s*https?://[^\]]+\]', '', md_text)
    
    # Sostituiamo i link e raccogliamo i dati
    md_text = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', replacer, md_text)
            
    return md_text, extracted_links

class WebExplorer:
    def __init__(self, 
                 query_optimizer: QueryOptimizer,
                 search_orchestrator: HybridSearchOrchestrator,
                 html_crawler: HTMLCrawler,
                 pdf_extractor: PDFExtractor,
                 chunker: DocumentChunker,
                 retrieval_engine: RetrievalEngine,
                 smart_link_gen: SmartLinkGenerator,
                 pdf_server: PDFHighlightServer):
        
        self.query_optimizer = query_optimizer
        self.search_orchestrator = search_orchestrator
        self.html_crawler = html_crawler
        self.pdf_extractor = pdf_extractor
        self.chunker = chunker
        self.retrieval_engine = retrieval_engine
        self.smart_link_gen = smart_link_gen
        self.pdf_server = pdf_server
        
        self.domains = ["diem.unisa.it", "docenti.unisa.it", "corsi.unisa.it", "easycourse.unisa.it"]
        self.url_hash_map = LRUCache(maxsize=200)


    def _build_strategies(self, query: str) -> List[Tuple[str, str, int]]:
        # Rimuoviamo l'operatore 'site:' che rompe spesso le query automatizzate.
        # Aggiungiamo contesto testuale ("unisa", "diem") e chiediamo 25 risultati (max_res).
        # Il filtro in locale penserà a scartare i siti non ufficiali.
        strategies = [
            ("Ricerca Organica DIEM", f"{query} dipartimento DIEM unisa", 25),
            ("Ricerca Organica UNISA", f"{query} università degli studi di salerno", 25)
        ]
        return strategies

    def explore_and_retrieve(self, search_query: str, target_score: float = 5.0, debug: bool = False, max_timeout_sec: float = 120.0, min_score: float = 0.6, k: int = 15):
        print(f"\n[WEB EXPLORER] Avvio ricerca per: '{search_query}'")
        start_time = time.time()
        
        optimized_query = self.query_optimizer.optimize(search_query)
        strategies = self._build_strategies(optimized_query)
        
        collected_chunks = []
        page_cache = {}

        for strategy_name, query, max_res in strategies:
            # Controllo preventivo prima di lanciare la strategia
            elapsed = time.time() - start_time
            if elapsed >= max_timeout_sec:
                print(f"  ⏰ [TIMEOUT PREVENTIVO] Tempo limite ({max_timeout_sec}s) raggiunto. Termino ricerca.")
                break
                
            print(f"\n  [STRATEGIA] Tento: {strategy_name}")
            urls, hash_map = self.search_orchestrator.search_all(query, max_res)
            self.url_hash_map.update(hash_map)
            
            valid_urls = [u for u in urls if u not in page_cache and self.html_crawler.scraper.is_url_allowed(u)]
            if not valid_urls: continue

            # MODIFICA: Passiamo search_query al fetcher per attivare i filtri intelligenti
            pages_content = self._fetch_all_contents(valid_urls, page_cache, search_query)
            
            strategy_chunks = []
            for url, (title, md_text, page_date) in pages_content.items(): 
                
                # --- FORMATTAZIONE LINK STILE WIKIPEDIA ---
                clean_md_text, page_links = extract_and_clean_links(md_text)
                
                page_cache[url] = {"testo": clean_md_text, "titolo": title, "data": page_date, "links": page_links}
                chonkie_chunks = self.chunker.process_markdown(clean_md_text, title=title, date_str=page_date)
                
                for chunk in chonkie_chunks:
                    strategy_chunks.append({
                        "url": url, 
                        "text": chunk["text"],
                        "embedding": chunk["embedding"]
                    })


            if strategy_chunks:
                ranked_results = self.retrieval_engine.retrieve_and_rank(search_query, strategy_chunks)
                collected_chunks.extend(ranked_results)
                collected_chunks.sort(key=lambda x: x[0], reverse=True)

            # --- CALCOLO SOGLIA DINAMICA ---
            elapsed = time.time() - start_time
            
            if elapsed >= max_timeout_sec:
                current_target = min_score
            else:
                # Decadimento lineare: la soglia scende proporzionalmente al tempo trascorso
                decay_factor = elapsed / max_timeout_sec
                current_target = target_score - ((target_score - min_score) * decay_factor)
                
            cumulative = sum(score for score, _, _ in collected_chunks[:k])
            print(f"  📊 Tempo: {elapsed:.1f}s/{max_timeout_sec}s | Soglia Dinamica: {current_target:.2f} | Score Accumulato: {cumulative:.2f}")
            
            if cumulative >= current_target:
                print(f"  🎯 [SUCCESSO] Obiettivo dinamico raggiunto! Termino la ricerca.")
                break
                
            if elapsed >= max_timeout_sec:
                print(f"  ⏰ [TIMEOUT] Tempo esaurito post-elaborazione. Mi accontento dei risultati trovati finora.")
                break

        # Se il timer scatta ma non abbiamo nulla, restituiamo stringa vuota
        if not collected_chunks: 
            return "", {}
            
        # Altrimenti restituiamo fieramente i risultati parziali/finali!
        return self._format_output(collected_chunks[:k], page_cache, debug)

    # MODIFICA: Aggiunto parametro search_query
    def _fetch_all_contents(self, valid_urls: List[str], page_cache: dict, search_query: str) -> Dict[str, Tuple[str, str]]:
        pdf_urls = [u for u in valid_urls if u.lower().endswith(".pdf")]
        html_urls = [u for u in valid_urls if not u.lower().endswith(".pdf")]
        
        pages_content = {}
        pending_pdfs = {} 

        # I PDF che arrivano direttamente dalla SERP di Google/Bing non hanno un HTML circostante, 
        # quindi il loro "html_context" sarà basato solo sul nome del file.
        for url in pdf_urls:
            pending_pdfs[url] = {
                "title": url.split('/')[-1], 
                "html_context": f"Documento PDF diretto da motore di ricerca. File: {url.split('/')[-1]}"
            } 

        if html_urls:
            print(f"   [MULTI-THREAD] Scrape & Clean Trafilatura per {len(html_urls)} pagine HTML...")

            def fetch_with_js_fallback(url: str):
                """Prova a scaricare la pagina. Se sembra generata via JS, usa Playwright."""
                # 1. Tentativo Veloce (Statico)
                html_content = trafilatura.fetch_url(url)
                
                # Criterio euristico: se l'HTML è troppo piccolo o manca di body/testo sostanzioso,
                # probabilmente richiede JavaScript per renderizzare il contenuto.
                if not html_content or len(html_content) < 2000 or 'id="app"' in html_content or 'ng-app' in html_content:
                    print(f"   [JS RENDER] Sospetta pagina JS, avvio browser headless per: {url}")
                    try:
                        with sync_playwright() as p:
                            browser = p.chromium.launch(headless=True)
                            page = browser.new_page()
                            # wait_until="networkidle" aspetta che le chiamate di rete AJAX si plachino
                            page.goto(url, wait_until="networkidle", timeout=15000) 
                            html_content = page.content()
                            browser.close()
                    except Exception as e:
                        print(f"   [ERRORE JS] Fallito rendering con Playwright per {url}: {e}")
                
                return html_content

            def smart_fetch_html_task(url: str):
                # Download diretto (statico o con fallback Playwright)
                downloaded_html = fetch_with_js_fallback(url)
                if not downloaded_html:
                    return url, False, None, None, None

                # ========================================================
                # 🔥 FIX TITOLO: Estraiamo il titolo dal RAW HTML prima 
                # che la conversione in lxml faccia perdere i metadati
                # ========================================================
                extracted_title = None
                try:
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', downloaded_html, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        # Rimuoviamo ritorni a capo o tag sporchi dentro il titolo
                        extracted_title = re.sub(r'<[^>]+>', '', title_match.group(1)).replace('\n', ' ').strip()
                except Exception:
                    pass

                # Ora pre_clean_unisa_html ci restituisce un OGGETTO DOM (tree), non una stringa
                cleaned_tree = pre_clean_unisa_html(downloaded_html, url)
                
                # Se il parsing è andato in errore
                if cleaned_tree is None:
                    return url, False, None, None, None
                
                # Passiamo l'oggetto `cleaned_tree` direttamente a Trafilatura.
                # Questo azzera i re-parsing buggati ed evita i crash 'Document' object!
                data = trafilatura.bare_extraction(
                    cleaned_tree,
                    url=url,
                    include_formatting=True,
                    include_tables=True,
                    favor_recall=True,
                    include_links=True,
                    include_images=False,
                    include_comments=False
                )
                
                if not data:
                    return url, False, None, None, None

                # GESTIONE COMPATIBILITA': Trafilatura >= 1.9.0 restituisce un oggetto Document, non un dict
                if hasattr(data, 'as_dict'):
                    data = data.as_dict()
                elif not isinstance(data, dict):
                    data = {
                        'text': getattr(data, 'text', None),
                        'title': getattr(data, 'title', None),
                        'date': getattr(data, 'date', None)
                    }

                if not data.get('text'):
                    return url, False, None, None, None

                # ========================================================
                # Usiamo il titolo che abbiamo estratto a mano. 
                # Se per qualche motivo fallisce, usa quello di trafilatura o "Pagina Web"
                # ========================================================
                title = extracted_title or data.get('title') or 'Pagina Web'
                
                md_text = data.get('text')
                page_date = data.get('date') or "Sconosciuta (considerare informazione atemporale)"

                
                return url, True, title, md_text, page_date

            # Processamento parallelo veloce delle pagine web
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(smart_fetch_html_task, u): u for u in html_urls}
                
                for future in concurrent.futures.as_completed(futures):
                    url = futures[future]
                    try:
                        url, success, title, md, page_date = future.result()
                        
                        if success and md:
                            if page_date == "Sconosciuta (considerare informazione atemporale)":
                                print(f"Data sconosciuta per {url}")
                                
                            self.url_hash_map[url] = url
                            pages_content[url] = (title, md, page_date)

                            # --- FASE 1: RACCOLTA DEL CONTESTO ESTERNO DAI LINK PDF TROVATI NEL MARKDOWN ---
                            lines = md.split('\n')
                            for i, line in enumerate(lines):
                                # Questo continua a funzionare perfettamente perché i link ora hanno 
                                # la formattazione pulita [Testo](url) generata da Trafilatura 
                                # combinata con il tuo [URL: ...] esplicito
                                pdf_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+\.pdf[^\)]*)\)', line, flags=re.IGNORECASE)
                                for anchor_text, pdf_url in pdf_links:
                                    clean_pdf_url = pdf_url.split(' ')[0].split('#')[0].strip()
                                    
                                    if (any(domain in clean_pdf_url for domain in self.domains) and 
                                        clean_pdf_url not in pages_content and 
                                        clean_pdf_url not in page_cache and 
                                        clean_pdf_url not in valid_urls and 
                                        clean_pdf_url not in pending_pdfs):
                                        
                                        # Alziamo il limite per evitare che i link morti di Google intasino la coda
                                        if len(pending_pdfs) < 20:
                                            local_text = f"{lines[i-1]} {line}" if i > 0 else line
                                            local_text_clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', local_text)
                                            external_context = f"Sito: {title}. Link: {anchor_text}. Testo nella pagina: {local_text_clean}"
                                            
                                            pending_pdfs[clean_pdf_url] = {
                                                "title": f"Allegato PDF: {anchor_text}",
                                                "html_context": external_context
                                            }
                                            self.url_hash_map[clean_pdf_url] = clean_pdf_url
                                            
                    except Exception as e:
                        print(f"Errore Trafilatura su {url}: {e}")

        # --- FASE 2: PDF PEEKING + CONTEXTUAL FUSION ---
        if pending_pdfs:
            print(f"   [MULTI-THREAD] Fusione contestuale e verifica in corso per {len(pending_pdfs)} PDF...")

            # ==========================================
            # ⚙️ MANOPOLE DI TUNING PER I PDF
            # Impostati a 'None' = Accuratezza Massima
            # ==========================================
            PDF_MAX_PAGES = None   # Es: 3 (Estrae solo le prime 3 pagine)
            PDF_PEEK_CHARS = 3_000  # Es: 1500 (Valuta solo i primi 1500 caratteri per la stima)

            def smart_fetch_pdf_task(pdf_url, meta):
                # 1. Download Unico In-Memory (regolato da PDF_MAX_PAGES)
                text = self.pdf_extractor.fetch_raw_text(pdf_url, max_pages=PDF_MAX_PAGES)
                
                # FIX DIFENSIVO: Se text è None (per via di un errore catturato) o vuoto, scartiamo senza fare crash
                if not text or not text.strip():
                    return pdf_url, meta["title"], "", ""
                
                # 2. Porzione per la stima (regolata da PDF_PEEK_CHARS)
                peek_text = text if PDF_PEEK_CHARS is None else text[:PDF_PEEK_CHARS]
                
                # 3. FUSIONE SOTA: Contesto HTML + Testo Puro Estratto
                fused_context = f"{meta['html_context']}\nContenuto file interno: {peek_text}"
                
                # 4. GIUDIZIO UNICO: Il Reranker valuta testo vero e proprio
                fused_score = self.retrieval_engine.evaluator.calculate_reward(search_query, fused_context)
                
                if fused_score < 0.10: 
                    print(f"      [BOUNCER 🚫] Scartato (Score fuso: {fused_score:.2f}): {meta['title']}\n")
                    # MODIFICA QUI: Aggiunto un quarto parametro vuoto per pareggiare le variabili
                    return pdf_url, meta["title"], "", "" 
                
                print(f"      [ACCETTATO ✅] PDF rilevante (Score fuso: {fused_score:.2f}): {meta['title']}\n")
                
                # 5. Restituiamo il testo già scaricato, senza fare una seconda chiamata HTTP!
                return pdf_url, meta["title"], text, "Sconosciuta (File PDF)"


            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(smart_fetch_pdf_task, u, meta) for u, meta in pending_pdfs.items()]
                for future in concurrent.futures.as_completed(futures):
                    # Aggiorna qui per spacchettare 4 variabili
                    u, t, text, pdf_date = future.result()
                    if text.strip(): pages_content[u] = (t, text, pdf_date)


        return pages_content

    def _format_output(self, top_chunks, page_cache, debug_mode):
        formatted_chunks = []
        source_mapping = {}
        
        # --- TRUE PARENT-CHILD DEDUPLICATION ---
        seen_parent_ids = set()
        url_chunk_counts = {} # Traccia quanti chunk abbiamo preso per ogni URL
        unique_chunks = []
        
        for score, url, chunk in top_chunks:
            # 1. Pulizia richiesta: riduce gli a capo multipli salvando token
            clean_chunk = re.sub(r'\n{3,}', '\n\n', chunk).strip()
            
            # Calcolo budget dinamico basato sul punteggio del Cross-Encoder (BGE)
            if score >= 0.95:
                max_chunks_for_url = 20
            elif score >= 0.85:
                max_chunks_for_url = 12  # Rilevanza altissima
            elif score >= 0.60:
                max_chunks_for_url = 5   # Rilevanza media
            else:
                max_chunks_for_url = 2   # Rilevanza bassa (evita inquinamento)
                
            current_count = url_chunk_counts.get(url, 0)

            # 2. Deduplicazione infallibile O(1):
            # Siccome ogni "Figlio" ha esattamente lo stesso testo del "Padre", 
            # se due figli fanno match, la stringa pulita sarà identica.
            if clean_chunk not in seen_parent_ids and current_count < max_chunks_for_url:
                seen_parent_ids.add(clean_chunk)
                unique_chunks.append((score, url, clean_chunk))
                url_chunk_counts[url] = current_count + 1

        # 3. Costruzione output finale pulitissimo
        for i, (score, url, clean_chunk) in enumerate(unique_chunks, 1):
            url_pulito = url.strip()
            base_url = self.url_hash_map.get(url_pulito, url_pulito).split('#')[0]
            
            if base_url.lower().endswith('.pdf'):
                smart_url = self.pdf_server.create_highlighted_pdf(base_url, clean_chunk)
            else:
                smart_url = self.smart_link_gen.generate(base_url, clean_chunk, debug_mode)
                
            formatted_chunks.append(f"[Doc {i}]\n{clean_chunk}")
            source_mapping[i] = {
                "url": url, "smart_url": smart_url,
                "titolo": page_cache.get(url_pulito, {}).get("titolo", "Pagina Web"),
                "score": score,
                "links": page_cache.get(url_pulito, {}).get("links", []) 
            }
        return "\n\n".join(formatted_chunks), source_mapping

