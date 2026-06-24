# ================================================
# FILE: utils/live_web_scraper.py
# ================================================
import requests
import tempfile
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from unstructured.partition.auto import partition

class LiveWebScraper:
    def __init__(self, allowed_domains: list = None, timeout: int = 10):
        self.allowed_domains = allowed_domains or [
            "www.diem.unisa.it", "diem.unisa.it", "docenti.unisa.it",
            "corsi.unisa.it", "easycourse.unisa.it"
        ]
        self.forbidden_urls = []
        self.timeout = timeout
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    def _get_expanded_context(self, a_tag) -> str:
        block_tags = ['p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']
        curr = a_tag.parent
        while curr and curr.name not in block_tags and curr.name != 'body':
            curr = curr.parent
        if curr:
            testo_esteso = curr.get_text(separator=" ", strip=True)
            anchor_text = a_tag.get_text(strip=True)
            if len(testo_esteso) > 300:
                try:
                    idx = testo_esteso.index(anchor_text)
                    start, end = max(0, idx - 100), min(len(testo_esteso), idx + len(anchor_text) + 100)
                    testo_esteso = f"...{testo_esteso[start:end]}..."
                except ValueError:
                    testo_esteso = testo_esteso[:300] + "..."
            return testo_esteso
        return a_tag.get_text(strip=True)

    def is_url_allowed(self, url: str) -> bool:
        try:
            for forbidden_url in self.forbidden_urls:
                if url.startswith(forbidden_url): return False
            return urlparse(url).netloc in self.allowed_domains
        except Exception:
            return False

    def extract_clean_text(self, soup: BeautifulSoup) -> str:
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        return " ".join(soup.stripped_strings)

    def download_page(self, current_url: str) -> dict:
        print(f"  [SCRAPER] Richiesta a: {current_url}...")
        try:
            response = requests.get(current_url, headers=self.headers, timeout=self.timeout, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                valid_links_with_context = {}
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    full_url = urljoin(current_url, href).split('#')[0]
                    
                    if self.is_url_allowed(full_url) and full_url not in valid_links_with_context:
                        anchor_text = a_tag.get_text(strip=True) or (a_tag.find('img')['alt'] if a_tag.find('img') and a_tag.find('img').has_attr('alt') else "Link")
                        contesto_esteso = self._get_expanded_context(a_tag)
                        contesto_base = contesto_esteso.replace(anchor_text, f"[{anchor_text}]") if anchor_text in contesto_esteso else f"[{anchor_text}]"
                        
                        is_structural = a_tag.find_parent(['nav', 'footer', 'header', 'aside']) is not None
                        tag_strutturale = "[NAV]" if is_structural else "[CONTENT]"
                        
                        valid_links_with_context[full_url] = f"{tag_strutturale} {contesto_base}"

                titolo_pagina = soup.title.string.strip() if soup.title and soup.title.string else "Pagina Web"
                clean_text = self.extract_clean_text(soup)
                
                if len(clean_text.strip()) < 200:
                    return {"text": "Errore: Contenuto insufficiente (< 200 char).", "links": {}, "title": "Scartata"}
                
                return {"text": clean_text, "links": valid_links_with_context, "title": titolo_pagina}

            elif "application/pdf" in content_type:
                print(f"   [SCRAPER] Trovato documento PDF. Elaborazione con Unstructured...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_filepath = temp_file.name
                try:
                    elements = partition(filename=temp_filepath, languages=["ita", "eng"])
                    clean_text = "\n\n".join([str(el) for el in elements if str(el).strip()])
                finally:
                    os.remove(temp_filepath)
                return {"text": clean_text, "links": {}, "title": f"Documento: {current_url.split('/')[-1]}"}
            else:
                return {"text": f"Contenuto non supportato ({content_type}).", "links": {}, "title": "Ignorato"}
        except Exception as e:
            return {"text": f"Errore: {str(e)}", "links": {}, "title": "Errore"}