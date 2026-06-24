import os
import asyncio
import concurrent.futures
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from playwright.sync_api import sync_playwright

class ProfessorsService:
    """
    Gestisce la logica di scraping per le informazioni dei docenti.
    Esegue le operazioni in un Thread dedicato per evitare conflitti con i worker di Gradio.
    """

    def _run_in_isolated_thread(self, func, *args):
        """
        Esegue una funzione in un thread isolato.
        Questo protegge Playwright dai loop asyncio già attivi (o mancanti) in Gradio.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args)
            return future.result()

    def _scarica_profilo_docente_isolato(self, nome: str, cognome: str) -> str:
        """Viene eseguito nel thread isolato. Crea un loop pulito per Crawl4AI."""
        async def _job():
            indirizzo_base = "https://docenti.unisa.it"
            nome_norm = nome.strip().lower().replace(" ", "")
            cognome_norm = cognome.strip().lower().replace(" ", "").replace("'", "")
            url_target = f"{indirizzo_base}/{nome_norm}.{cognome_norm}"
            
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                            url=url_target,
                            excluded_tags=['nav', 'footer', 'header', 'aside', 'menu', 'form', 'script', 'style', 'noscript'],
                            word_count_threshold=10,
                            remove_overlay_elements=True,
                            config=CrawlerRunConfig(page_timeout=60000),
                            magic=True
                        )
                
                if result.success:
                    if "Docente non trovato!" in result.html:
                        return f"Il docente {nome} {cognome} non è stato trovato (URL: {url_target})."
                    return result.markdown 
                else:
                    return f"Errore durante l'estrazione: {result.error_message}"

        # Creiamo e gestiamo l'event loop specificamente per questo thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_job())
        finally:
            loop.close()

    def _cerca_docente_playwright_isolato(self, query_ricerca: str) -> str:
        """Viene eseguito nel thread isolato usando sync_playwright."""
        indirizzo_lista = "https://docenti.unisa.it/"
        xpath_risultati = "/html/body/div[4]/div/div[4]/div/div/div/div/div/div/div[2]"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                # Timeout aumentato e wait_until="domcontentloaded" per aggirare asset lenti/bloccanti
                page.goto(indirizzo_lista, timeout=60000, wait_until="domcontentloaded")
                
                page.wait_for_selector("#filter", timeout=30000)
                page.type("#filter", query_ricerca, delay=50)
                page.wait_for_timeout(1000) 
                
                risultati_locator = page.locator(f"xpath={xpath_risultati}")
                testo_risultati = risultati_locator.inner_text(timeout=15000)
                
                if not testo_risultati or testo_risultati.strip() == "":
                    return f"Nessun risultato trovato per la query: '{query_ricerca}'"
                return testo_risultati
            except Exception as e:
                return f"Si è verificato un errore durante la ricerca: {str(e)}"
            finally:
                browser.close()

    # --- WRAPPER PER LANGCHAIN TOOL ---
    
    def get_professor_profile(self, nome: str, cognome: str) -> str:
        return self._run_in_isolated_thread(self._scarica_profilo_docente_isolato, nome, cognome)

    def search_professor_directory(self, query_ricerca: str) -> str:
        return self._run_in_isolated_thread(self._cerca_docente_playwright_isolato, query_ricerca)