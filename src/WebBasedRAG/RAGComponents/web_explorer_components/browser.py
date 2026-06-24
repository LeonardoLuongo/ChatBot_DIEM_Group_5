import time
import atexit
import concurrent.futures
from playwright.sync_api import sync_playwright
from seleniumbase import sb_cdp
from pyvda import VirtualDesktop, get_virtual_desktops, AppView

class ThreadedPlaywrightProxy:
    """
    Proxy che assicura che TUTTE le interazioni con Playwright avvengano
    in un unico thread dedicato, isolandolo dal loop asyncio di LangChain.
    """
    def __init__(self, target, executor):
        self._target = target
        self._executor = executor

    def __getattr__(self, name):
        # 1. RECUPERO ATTRIBUTO: Fatto nel thread isolato per evitare il crash immediato
        attr = self._executor.submit(getattr, self._target, name).result()
        
        # 2. SE E' UNA FUNZIONE (es. page.goto(), page.content(), locator.all()):
        if callable(attr):
            def wrapper(*args, **kwargs):
                # Esegue la funzione nel thread isolato
                res = self._executor.submit(attr, *args, **kwargs).result()
                
                # A. Se restituisce un singolo oggetto Playwright (es. Locator), lo avvolgiamo
                if "playwright.sync_api" in str(type(res)):
                    return ThreadedPlaywrightProxy(res, self._executor)
                
                # B. NUOVO: Se restituisce una lista (es. res da locator.all()), avvolgiamo gli elementi interni
                elif isinstance(res, list):
                    return [
                        ThreadedPlaywrightProxy(item, self._executor) 
                        if "playwright.sync_api" in str(type(item)) else item
                        for item in res
                    ]
                    
                return res
            return wrapper
        
        # 3. SE E' UNA PROPRIETA' (es. page.url o un sotto-oggetto):
        if "playwright.sync_api" in str(type(attr)):
            return ThreadedPlaywrightProxy(attr, self._executor)
            
        # NUOVO: Se la proprietà restituisce direttamente una lista
        elif isinstance(attr, list):
            return [
                ThreadedPlaywrightProxy(item, self._executor) 
                if "playwright.sync_api" in str(type(item)) else item
                for item in attr
            ]
            
        return attr



class BrowserEngine:
    def __init__(self, browser_path: str):
        self.browser_path = browser_path
        self.sb = None
        self.playwright = None
        self.browser_instance = None
        
        # CRUCIALE: Un UNICO executor permanente con 1 solo worker.
        # Playwright vivrà interamente dentro questo singolo thread, sempre.
        self.pw_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        print("   [SYSTEM] Avvio istanza Browser e configurazione Desktop...")
        
        # 1. Gestione Desktop Virtuale
        if len(get_virtual_desktops()) < 2:
            print("   ⚙️ Creazione Desktop virtuale 2 in corso...")
            VirtualDesktop.create()
        target_desktop = VirtualDesktop(2)
        
        # 2. Avvio del browser
        self.sb = sb_cdp.Chrome(browser_executable_path=self.browser_path)
        time.sleep(1.5) 
        
        # 3. Spostamento della finestra nel Desktop 2
        try:
            edge_window = AppView.current() 
            edge_window.move(target_desktop)
            print("   [MAGIC] Finestra teletrasportata nel Desktop 2!")
        except Exception as e:
            print(f"   ⚠️ Impossibile spostare la finestra. Motivo: {e}")

        # 4. Inizializzazione Playwright: ESEGUITA NEL THREAD ISOLATO
        def _init_pw():
            pw = sync_playwright().start()
            browser = pw.chromium.connect_over_cdp(self.sb.get_endpoint_url())
            return pw, browser

        self.playwright, self.browser_instance = self.pw_executor.submit(_init_pw).result()
        
        atexit.register(self.close_browser)

    def get_page(self):
        # 5. Recupero della pagina: ESEGUITO NEL THREAD ISOLATO
        def _get_page():
            context = self.browser_instance.contexts[0]
            return context.pages[0] if context.pages else context.new_page()
            
        page = self.pw_executor.submit(_get_page).result()
        
        # Restituiamo il PROXY protetto e il driver SeleniumBase
        return ThreadedPlaywrightProxy(page, self.pw_executor), self.sb

    def close_browser(self):
        print("\n   [SYSTEM] Pulizia risorse: Chiusura Browser in corso...")
        if self.playwright:
            try: 
                self.pw_executor.submit(self.playwright.stop).result()
            except: pass
            self.playwright = None
            
        if self.sb:
            try: self.sb.quit()
            except: pass
            self.sb = None
            
        # Chiudiamo anche il pool di thread
        self.pw_executor.shutdown(wait=False)
        print("   Browser chiuso con successo.")