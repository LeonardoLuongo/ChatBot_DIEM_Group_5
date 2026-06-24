import os
import re
import urllib.parse
import tempfile
import random
import atexit
import http.server
import socketserver
import threading
import functools
import requests
import fitz
from rapidfuzz import fuzz

class SmartLinkGenerator:
    def generate(self, base_url: str, chunk_text: str, debug_mode: bool = False) -> str:
        clean = re.sub(r'^CONTESTO:.*?\]\n', '', chunk_text).strip()
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean) 
        clean = re.sub(r'![\[].*?[\]]\(.*?\)', '', clean)       
        clean = re.sub(r'[*#_`~]', '', clean)
        clean = re.sub(r'<[^>]+>', '', clean)

        lines = clean.split('\n')
        valid_blocks = []
        for line in lines:
            l_clean = re.sub(r'[ \t]+', ' ', line)
            l_clean = re.sub(r' ([.,!?;:])', r'\1', l_clean) 
            l_clean = re.sub(r'\(\s+', '(', l_clean)         
            l_clean = re.sub(r'\s+\)', ')', l_clean)         
            l_clean = re.sub(r'^[^a-zA-Z0-9À-ÿ]+', '', l_clean)
            l_clean = re.sub(r'[^a-zA-Z0-9À-ÿ]+$', '', l_clean)
            words = l_clean.strip().split()
            if len(words) >= 4: valid_blocks.append(words)

        if not valid_blocks: return base_url 
        if not debug_mode: valid_blocks.sort(key=len, reverse=True)
        target_blocks = valid_blocks if debug_mode else valid_blocks[:1]

        fragments = []
        for words in target_blocks:
            if len(words) <= 8:
                fragments.append(f"text={urllib.parse.quote(' '.join(words))}")
            else:
                start = urllib.parse.quote(' '.join(words[:4]))
                end = urllib.parse.quote(' '.join(words[-4:]))
                fragments.append(f"text={start},{end}")

        return f"{base_url}#:~:{'&'.join(fragments)}"

class PDFHighlightServer:
    def __init__(self):
        self.port = random.randint(8081, 8999)
        self.host_dir = tempfile.mkdtemp(prefix="rag_pdfs_")
        self._start_server()
        atexit.register(self.cleanup)

    def _start_server(self):
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, directory=None, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)
            def log_message(self, format, *args): pass 
            def end_headers(self):
                if self.path.lower().endswith('.pdf'):
                    self.send_header('Content-Type', 'application/pdf')
                    self.send_header('Content-Disposition', 'inline')
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                super().end_headers()

        handler = functools.partial(Handler, directory=self.host_dir)
        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
        
        try:
            self.httpd = ThreadedHTTPServer(("", self.port), handler)
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
            print(f"   [SERVER] Host PDF locale (Threaded) avviato su http://localhost:{self.port}")
        except Exception as e:
            print(f"   [SERVER] Impossibile avviare il server PDF: {e}")


    def cleanup(self):
        import shutil
        try: shutil.rmtree(self.host_dir)
        except: pass

    def create_highlighted_pdf(self, pdf_url: str, chunk_text: str) -> str:
        import urllib3
        try:
            safe_name = "doc_" + str(abs(hash(pdf_url)))[:8] + ".pdf"
            out_path = os.path.join(self.host_dir, safe_name)

            if os.path.exists(out_path):
                # REINTEGRATO: Uso della cache locale
                print("      [CACHE] PDF già presente localmente. Aggiungo nuova evidenziazione...")
                with open(out_path, "rb") as f:
                    doc = fitz.open(stream=f.read(), filetype="pdf")
            else:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(pdf_url, timeout=15, verify=False)
                doc = fitz.open(stream=response.content, filetype="pdf")

            clean_chunk = re.sub(r'^CONTESTO:\s*\[.*?\]\s*', '', chunk_text.strip()).strip()
            cleaned_words = [re.sub(r'[^a-zA-Z0-9]', '', w.lower()) for w in clean_chunk.split()[:14]]
            target_str = "".join([w for w in cleaned_words if w])
            base_window_size = len([w for w in cleaned_words if w])

            if not target_str:
                # REINTEGRATO: Controllo parole target
                print("      ⚠️ Impossibile estrarre parole target dal chunk.")
                return pdf_url

            target_page = 1
            best_score = 0
            best_rects = []

            if target_str:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pdf_words = page.get_text("words")
                    clean_pdf_words = [re.sub(r'[^a-zA-Z0-9]', '', w[4].lower()) for w in pdf_words]
                    
                    for window_offset in [0, 1, 2, 3]:
                        window_size = base_window_size + window_offset
                        if window_size > len(clean_pdf_words): continue
                        for i in range(len(clean_pdf_words) - window_size + 1):
                            window_str = "".join([w for w in clean_pdf_words[i : i + window_size] if w])
                            score = fuzz.ratio(target_str, window_str)
                            if score > best_score:
                                best_score = score
                                target_page = page_num + 1
                                best_rects = [fitz.Rect(pdf_words[k][:4]) for k in range(i, i + window_size)]
                            if best_score >= 95: break 
                        if best_score >= 95: break
                    if best_score >= 95: break

            if best_score > 70:
                print(f"      ✅ Trovato (Match: {best_score:.1f}%)! Evidenziato a Pagina {target_page}.")
                page_to_highlight = doc[target_page - 1]
                for rect in best_rects:
                    page_to_highlight.add_highlight_annot(rect).update()
            else:
                print(f"      ⚠️ Frase non trovata (Match: {best_score:.1f}%). Aperto a pag 1.")
                target_page = 1

            doc.save(out_path)
            doc.close()
            return f"http://localhost:{self.port}/{safe_name}#page={target_page}"
            
        except Exception as e:
            # REINTEGRATO: Log di errore
            print(f"   [PDF MAGIC] Errore: {e}")
            return pdf_url
