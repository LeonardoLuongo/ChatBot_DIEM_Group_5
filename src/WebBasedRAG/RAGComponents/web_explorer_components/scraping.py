import os
import asyncio
import concurrent.futures
import requests
import fitz
import re
import string
from typing import List
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from WebBasedRAG.RAGComponents.live_web_scraper import LiveWebScraper

fitz.TOOLS.mupdf_display_errors(False) 

class HTMLCrawler:
    def __init__(self):
        self.scraper = LiveWebScraper()

    def crawl_sync(self, urls: List[str]) -> List:
        """Esegue Crawl4AI in un thread isolato con un event loop pulito (Safe per Gradio)."""
        def _isolated_job():
            async def _crawl():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    return await crawler.arun_many(
                        urls=urls,
                        excluded_tags=['nav', 'footer', 'header', 'aside', 'menu', 'form', 'script', 'style', 'noscript'],
                        word_count_threshold=10,
                        remove_overlay_elements=True,
                        config=CrawlerRunConfig(page_timeout=60000),
                        magic=True
                    )
            
            # Creazione di un loop vergine solo per questo thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_crawl())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_isolated_job).result()

class PDFExtractor:
    def fetch_raw_text(self, url: str, max_pages: int = None) -> str:
        """
        Scarica il PDF e ne estrae il testo.
        :param max_pages: Se None, estrae tutte le pagine. Se intero (es. 3), estrae solo le prime N.
        """
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            head_resp = requests.head(url, timeout=3, verify=False)
            if head_resp.status_code == 200:
                content_length = int(head_resp.headers.get('Content-Length', 0))
                # Limite di sicurezza hardcoded: scarta PDF più grandi di 5MB
                if content_length > 5 * 1024 * 1024:
                    print(f"      [BOUNCER] Scartato PDF > 5MB: {url}")
                    return ""

            response = requests.get(url, timeout=5, verify=False)
            response.raise_for_status()

            if 'application/pdf' not in response.headers.get('Content-Type', '').lower() and not url.lower().endswith('.pdf'):
                return ""

            
            doc = fitz.open(stream=response.content, filetype="pdf")
            raw_text = []
            
            for i, page in enumerate(doc):
                if max_pages is not None and i >= max_pages:
                    break  # Si ferma al limite di pagine richiesto
                raw_text.append(page.get_text("text"))
                
            doc.close()
            return "\n\n".join(raw_text)
        # REINTEGRATO: Gestione esplicita del timeout
        except requests.Timeout:
            print(f"      [TIMEOUT] ⏳ Tempo superato per: {url.split('/')[-1]}")
            return ""
        except Exception as e:
            print(f"      [ERRORE PDF] Impossibile leggere {url}: {str(e)}")
            return ""
