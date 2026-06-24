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
    def peek_pdf(self, url: str, peek_bytes: int = 65536) -> str:
        """SOTA: Scarica solo i primi 64KB del PDF per valutarne il contenuto."""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            # 1. Richiesta parziale
            headers = {"Range": f"bytes=0-{peek_bytes}"}
            response = requests.get(url, headers=headers, timeout=4, verify=False)
            
            if response.status_code not in [200, 206]:
                return ""
                
            # 2. Tentativo primario: Parsing di PDF Linearizzati (Web Fast View)
            try:
                doc = fitz.open(stream=response.content, filetype="pdf")
                text = doc[0].get_text("text") if len(doc) > 0 else ""
                doc.close()
                if text.strip(): return text
            except Exception:
                pass # Fallirà per i PDF standard perché manca la tabella XREF alla fine
                
            # 3. Fallback SOTA: Estrazione grezza dei metadati in chiaro dall'intestazione binaria!
            # Molti PDF hanno Titolo, Autore e Subject in chiaro nei primi byte prima dello stream zippato.
            raw_strings = re.findall(b'[ -~]{5,}', response.content) # Trova stringhe ASCII stampabili
            fallback_text = " ".join([s.decode('ascii', errors='ignore') for s in raw_strings])
            
            # Puliamo il rumore di fondo dei metadati PDF interni
            fallback_text = re.sub(r'/(?:Type|Pages|Catalog|Font|Length|Filter|ColorSpace|ProcSet|MediaBox|CropBox)', '', fallback_text)
            return fallback_text[:1500] # Restituiamo un assaggio al Reranker
            
        except Exception as e:
            return ""

    def fetch_raw_text(self, url: str) -> str:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            head_resp = requests.head(url, timeout=3, verify=False)
            if head_resp.status_code == 200:
                content_length = int(head_resp.headers.get('Content-Length', 0))
                if content_length > 5 * 1024 * 1024:
                    print(f"      [BOUNCER] Scartato PDF > 5MB: {url}")
                    return ""

            response = requests.get(url, timeout=5, verify=False)
            response.raise_for_status()

            if 'application/pdf' not in response.headers.get('Content-Type', '').lower() and not url.lower().endswith('.pdf'):
                return ""

            
            doc = fitz.open(stream=response.content, filetype="pdf")
            raw_text = [page.get_text("text") for page in doc]
            doc.close()
            return "\n\n".join(raw_text)
        # REINTEGRATO: Gestione esplicita del timeout
        except requests.Timeout:
            print(f"      [TIMEOUT] ⏳ Tempo superato per: {url.split('/')[-1]}")
            return ""
        except Exception as e:
            print(f"      [ERRORE PDF] Impossibile leggere {url}: {str(e)[:50]}")
            return ""
