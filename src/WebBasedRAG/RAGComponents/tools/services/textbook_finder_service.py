from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# =========================================================================
# ECCEZIONI
# =========================================================================

class TextbookFinderError(Exception):
    """Errore generico del tool."""


class ProfessorNotFoundError(TextbookFinderError):
    """Non è stato possibile risolvere il docente per l'esame richiesto."""


class CourseNotFoundOnPageError(TextbookFinderError):
    """La pagina del docente è stata raggiunta ma l'esame non compare tra
    gli insegnamenti elencati (nome diverso, anno accademico cambiato, ecc.)."""


class TestiSectionNotFoundError(TextbookFinderError):
    """La scheda dell'insegnamento è stata aperta ma la sezione "Testi"
    non è stata trovata all'interno."""


# =========================================================================
# MAPPA ESAME -> DOCENTE
# =========================================================================
# Popola questa mappa mano a mano che aggiungi esami. Chiave = nome esame
# normalizzato (lowercase, senza accenti), valore = (nome, cognome) del
# docente titolare così come compaiono nell'URL di docenti.unisa.it
# (es. "mario.vento" -> ("mario", "vento")).
#
# In alternativa, se il tuo crawler DIEM ha già una tabella
# esame -> docente (probabile, dato che la crawli comunque per il RAG),
# puoi popolare/estendere questa mappa a runtime chiamando
# TextbookFinderService.register_exam(...) invece di editare qui a mano.

EXAM_PROFESSOR_MAP: dict[str, tuple[str, str]] = {
    "artificial vision": ("mario", "vento"),
    "visione artificiale": ("mario", "vento"),
}


@dataclass
class TextbookResult:
    """Risultato del recupero testi per un insegnamento."""
    esame: str                     # Nome esame come richiesto dall'utente
    docente: str                   # Nome completo del docente risolto
    url_scheda: str                # URL della pagina didattica utilizzata
    testi_raw: str                 # Testo grezzo estratto dalla sezione "Testi"
    testi_lista: list[str] = field(default_factory=list)  # Testi splittati per riga/voce


# =========================================================================
# HELPER: SLUG / NORMALIZZAZIONE
# =========================================================================

def _strip_accents(s: str) -> str:
    """Rimuove accenti/diacritici (es. 'à' -> 'a') per costruire slug URL-safe."""
    normalized = unicodedata.normalize("NFKD", s)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _normalize_key(s: str) -> str:
    """Normalizza un testo per confronti case/accent-insensitive."""
    return _strip_accents(s).lower().strip()


def build_docente_slug(nome: str, cognome: str) -> str:
    """Costruisce lo slug docenti.unisa.it a partire da nome e cognome.

    Esempio: ('Mario', 'Vento') -> 'mario.vento'
    Gestisce nomi/cognomi composti sostituendo gli spazi con '-'
    (pattern osservato comunemente sui portali basati su questo CMS).
    """
    n = _strip_accents(nome).lower().strip().replace(" ", "-")
    c = _strip_accents(cognome).lower().strip().replace(" ", "-")
    n = re.sub(r"[^a-z\-]", "", n)
    c = re.sub(r"[^a-z\-]", "", c)
    return f"{n}.{c}"


def build_didattica_url(nome: str, cognome: str) -> str:
    """URL diretto della sezione didattica del docente."""
    return f"https://docenti.unisa.it/{build_docente_slug(nome, cognome)}/didattica"


# =========================================================================
# SERVICE
# =========================================================================

class TextbookFinderService:
    """
    Service per il recupero automatico dei testi consigliati di un
    insegnamento DIEM (o comunque UniSA) tramite scraping mirato di
    docenti.unisa.it con Playwright.
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 60000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        # cache in-memory: chiave (slug_docente, esame_normalizzato) -> TextbookResult
        self._cache: dict[tuple[str, str], TextbookResult] = {}

    # ------------------------------------------------------------------
    # RISOLUZIONE ESAME -> DOCENTE
    # ------------------------------------------------------------------

    @staticmethod
    def register_exam(exam_name: str, nome: str, cognome: str) -> None:
        """Aggiunge/aggiorna a runtime una voce nella mappa esame->docente."""
        EXAM_PROFESSOR_MAP[_normalize_key(exam_name)] = (nome.strip(), cognome.strip())

    @staticmethod
    def resolve_professor(exam_name: str) -> Optional[tuple[str, str]]:
        """Cerca il docente titolare di un esame nella mappa locale.

        Match esatto prima, poi substring match (tollerante a piccole
        variazioni tipo 'Artificial Vision' vs 'artificial vision course').
        Restituisce None se non trovato: in tal caso il chiamante (provider)
        dovrà chiedere il nome del docente all'utente.
        """
        key = _normalize_key(exam_name)
        if key in EXAM_PROFESSOR_MAP:
            return EXAM_PROFESSOR_MAP[key]
        for mapped_key, prof in EXAM_PROFESSOR_MAP.items():
            if mapped_key in key or key in mapped_key:
                return prof
        return None

    # ------------------------------------------------------------------
    # ENTRY POINT SINCRONO (usato dal provider / dal tool LangChain)
    # ------------------------------------------------------------------

    def get_textbooks(
        self,
        exam_name: str,
        professor_nome: Optional[str] = None,
        professor_cognome: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Recupera i testi consigliati per un esame.

        Args:
            exam_name:          Nome dell'insegnamento (es. "Artificial Vision").
            professor_nome:     Nome del docente, se già noto/fornito dall'utente.
            professor_cognome:  Cognome del docente, se già noto/fornito dall'utente.
            force_refresh:      Se True, ignora la cache e rifà lo scraping.

        Returns:
            dict JSON-friendly con status SUCCESS / NEED_INFO / ERROR.
        """
        try:
            # 1. Risoluzione docente
            if professor_nome and professor_cognome:
                nome, cognome = professor_nome.strip(), professor_cognome.strip()
            else:
                resolved = self.resolve_professor(exam_name)
                if resolved is None:
                    return {
                        "status": "NEED_INFO",
                        "message": (
                            f"Non ho trovato in memoria il docente titolare dell'esame "
                            f"'{exam_name}'. Puoi indicarmi nome e cognome del docente?"
                        ),
                    }
                nome, cognome = resolved

            docente_label = f"{nome.capitalize()} {cognome.capitalize()}"
            slug = build_docente_slug(nome, cognome)
            cache_key = (slug, _normalize_key(exam_name))

            if not force_refresh and cache_key in self._cache:
                cached = self._cache[cache_key]
                return self._format_success(cached)

            testi_raw, url_finale = self._run_scrape(nome=nome, cognome=cognome, exam_name=exam_name)

            result = TextbookResult(
                esame=exam_name,
                docente=docente_label,
                url_scheda=url_finale, # Salviamo l'URL numerico reale
                testi_raw=testi_raw,
                testi_lista=_split_testi(testi_raw),
            )
            self._cache[cache_key] = result

            return self._format_success(result)

        except ProfessorNotFoundError as e:
            return {"status": "ERROR", "message": str(e)}
        except CourseNotFoundOnPageError as e:
            return {
                "status": "ERROR",
                "message": (
                    f"{e} Verifica che il nome dell'insegnamento corrisponda esattamente "
                    "a quello indicato sulla pagina del docente (potrebbe differire per "
                    "anno accademico o lingua del corso)."
                ),
            }
        except TestiSectionNotFoundError as e:
            return {"status": "ERROR", "message": str(e)}
        except Exception as e:
            return {"status": "ERROR", "message": f"Errore imprevisto nel recupero testi: {e}"}

    # ------------------------------------------------------------------
    # FORMATTAZIONE OUTPUT
    # ------------------------------------------------------------------

    @staticmethod
    def _format_success(result: TextbookResult) -> dict:
        if result.testi_lista:
            testi_fmt = "\n".join(f"  • {t}" for t in result.testi_lista)
        else:
            testi_fmt = result.testi_raw or "(nessun testo indicato dal docente)"

        explanation = (
            f"**📚 Testi consigliati — {result.esame}**\n"
            f"Docente: {result.docente}\n\n"
            f"{testi_fmt}\n\n"
            f"Fonte: {result.url_scheda}\n\n"
            "⚠️ ISTRUZIONE INTERNA PER L'AI: riporta fedelmente l'elenco dei testi "
            "sopra riportato, senza inventare o omettere voci."
        )

        return {
            "status": "SUCCESS",
            "esame": result.esame,
            "docente": result.docente,
            "url_scheda": result.url_scheda,
            "testi": result.testi_lista or [result.testi_raw],
            "spiegazione": explanation,
        }

    # ------------------------------------------------------------------
    # SCRAPING (Playwright)
    # ------------------------------------------------------------------

    def _run_scrape(self, nome: str, cognome: str, exam_name: str) -> tuple[str, str]:
        """Wrapper sincrono attorno allo scraping async, gestisce il caso
        in cui venga chiamato da dentro un event loop già attivo (es. se
        l'AgenticRAG gira già dentro asyncio) usando un loop dedicato in
        thread separato."""
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(self._scrape_async(nome, cognome, exam_name))

        
        def _runner():
            return asyncio.run(self._scrape_async(nome, cognome, exam_name))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_runner)
            return future.result()
    
    import re
    async def _scrape_async(self, nome: str, cognome: str, exam_name: str) -> tuple[str, str]:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page()
                page.set_default_timeout(self.timeout_ms)
                
                # 1. Atterraggio sulla directory principale
                await page.goto("https://docenti.unisa.it/", wait_until="domcontentloaded")

                # 2. Ricerca nel filtro
                search_box = page.locator("#filter")
                await search_box.fill(f"{cognome} {nome}")
                await search_box.press("Enter")
                
                # Aspettiamo il caricamento dei risultati
                await page.wait_for_timeout(2500)

                # 3. Cerchiamo l'URL esatto usando NOME.COGNOME (es. mario.vento)
                # Questo rende impossibile confondere "Vento" con "Benevento"
                slug_esatto = f"{nome.lower()}.{cognome.lower()}"
                prof_link = page.locator(f"a[href*='{slug_esatto}']")
                
                if await prof_link.count() == 0:
                     raise ProfessorNotFoundError(f"Docente {nome} {cognome} non trovato nella rubrica.")
                
                url_vanity = await prof_link.first.get_attribute("href")
                if not url_vanity.startswith("http"):
                    url_vanity = "https://docenti.unisa.it" + url_vanity
                
                # 4. Navighiamo alla home del professore (questo innesca il redirect di UniSa)
                await page.goto(url_vanity, wait_until="domcontentloaded")
                await page.wait_for_load_state("networkidle")
                
                # --- LA MAGIA: NESSUN CLICK SU "DIDATTICA" ---
                # A questo punto l'URL nella barra in alto sarà diventato qualcosa come:
                # "https://docenti.unisa.it/005501/home"
                url_reale = page.url
                
                # Estraiamo matematicamente l'ID numerico (es. 005501) con una Regex
                match_id = re.search(r'/(\d{5,})', url_reale)
                
                if match_id:
                    id_numerico = match_id.group(1)
                    url_didattica = f"https://docenti.unisa.it/{id_numerico}/didattica"
                else:
                    # Fallback di sicurezza se per caso non trova numeri
                    url_didattica = url_reale.replace("/home", "/didattica").replace("/en/home", "/en/didattica")

                # Navighiamo direttamente in Didattica inserendo il link!
                await page.goto(url_didattica, wait_until="domcontentloaded")
                await page.wait_for_load_state("networkidle")

                # 5. Trova ed espandi il box dell'insegnamento
                course_locator = page.get_by_text(exam_name, exact=False)
                if await course_locator.count() == 0:
                    first_word = exam_name.split()[0]
                    course_locator = page.get_by_text(first_word, exact=False)

                if await course_locator.count() == 0:
                    raise CourseNotFoundOnPageError(f"Insegnamento '{exam_name}' non trovato sulla pagina didattica.")

                await course_locator.first.click()
                await page.wait_for_load_state("networkidle")

                # 6. Apri la tab "SCHEDA" (se esiste)
                try:
                    scheda_tab = page.get_by_text("SCHEDA", exact=True)
                    if await scheda_tab.count() > 0:
                        await scheda_tab.first.click(timeout=3000)
                        await page.wait_for_timeout(500)
                except PWTimeoutError:
                    pass

                # 7. Estrai la sezione "Testi"
                testi_text = await self._extract_testi_section(page)
                if not testi_text:
                    raise TestiSectionNotFoundError(f"Sezione 'Testi' non trovata nella scheda di '{exam_name}'.")
                    
                return testi_text.strip(), url_reale
            finally:
                await browser.close()

    @staticmethod
    async def _extract_testi_section(page) -> str:
        """Estrae il contenuto della sezione 'Testi', con 3 strategie in cascata.

        Il layout osservato (vedi screenshot di riferimento) è tabellare:
        una riga con l'etichetta 'Testi' seguita dalla riga/cella col
        contenuto vero e proprio, e termina prima della sezione successiva
        ('Altre Informazioni' o fine tabella).
        """
        # --- Strategia 1: riga/cella immediatamente successiva alla label "Testi" ---
        try:
            label = page.get_by_text("Testi", exact=True)
            if await label.count() > 0:
                row = label.first.locator(
                    "xpath=ancestor::tr[1]/following-sibling::tr[1]"
                )
                if await row.count() > 0:
                    text = (await row.first.inner_text()).strip()
                    if text:
                        return text
        except Exception:
            pass

        # --- Strategia 2: contenitore genitore della label, prendi tutto il testo
        #     del blocco genitore ed elimina la parola "Testi" iniziale ---
        try:
            label = page.get_by_text("Testi", exact=True)
            if await label.count() > 0:
                container = label.first.locator("xpath=ancestor::*[self::div or self::section][1]")
                if await container.count() > 0:
                    text = (await container.first.inner_text()).strip()
                    text = re.sub(r"^Testi\s*", "", text, flags=re.IGNORECASE)
                    if text:
                        return text
        except Exception:
            pass

        # --- Strategia 3: fallback regex sull'intero testo della pagina,
        #     tra "Testi" e la sezione successiva nota ---
        try:
            full_text = await page.inner_text("body")
            match = re.search(
                r"Testi\s*(.+?)(?:Altre Informazioni|BETA VERSION|$)",
                full_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        except Exception:
            pass

        return ""


def _split_testi(raw: str) -> list[str]:
    """Divide il blob di testo dei testi consigliati in voci separate,
    provando a riconoscere righe/voci bibliografiche distinte."""
    if not raw:
        return []
    # Split su newline; scarta righe vuote o troppo corte per essere un titolo
    lines = [l.strip(" -•\t") for l in raw.splitlines()]
    lines = [l for l in lines if len(l) > 3]
    return lines

if __name__ == "__main__":
    service = TextbookFinderService(headless=False) # Metti False per vedere il browser che si muove!
    # Testiamo con i dati degli screenshot
    risultato = service.get_textbooks(exam_name="Artificial Intelligence for Cybersecurity", professor_nome="Antonio", professor_cognome="Greco")
    print(risultato)