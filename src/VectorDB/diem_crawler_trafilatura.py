from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from tqdm import tqdm

try:
    from trafilatura import extract, bare_extraction
except ImportError:
    raise ImportError("Trafilatura non installata.")

from VectorDB.diem_tools.chunking import clean_markdown
from VectorDB.diem_tools.dedupe import PageDeduplicator
from VectorDB.diem_tools.pdf_io import (
    extract_pdf_pages,
    fetch_url_bytes,
    markdown_pdf_links,
    pdf_page_context,
    title_from_url,
)
from VectorDB.diem_tools.url_policy import (
    DEFAULT_START_URLS,
    canonicalize_url,
    collect_result_links,
    has_diem_affiliation,
    is_allowed_crawl_url,
    is_clearly_non_diem_bandi_text,
    is_docenti_url,
    is_institutional_pdf_url,
    is_probably_pdf,
    is_rubrica_url,
)


@dataclass(slots=True)
class SimpleChunk:
    """Record JSONL salvato dal crawler: testo pulito + metadati minimi."""

    chunk_id: str
    page_id: str
    url: str
    page_title: str
    text: str
    chunk_index: int
    token_estimate: int
    depth: int
    content_type: str
    hash_code: str
    source_type: str
    references: list[dict[str, str | int]]
    page_context: str = ""  # Riassunto della pagina per contesto retrieval
    parent_url: str | None = None
    pdf_page: int | None = None
    headings: dict[str, str] | None = None


@dataclass(slots=True)
class SimplePage:
    """Record JSONL di pagina: serve per audit e ripresa manuale del crawl."""

    page_id: str
    url: str
    final_url: str
    title: str
    status_code: int | None
    content_type: str
    depth: int
    hash_code: str
    crawled_at: str
    chunks_count: int
    text_chars: int
    source_type: str


class LinkReferenceCleaner:
    """Converte link markdown/URL nudi in riferimenti stile Wikipedia: [1], [2], ..."""

    markdown_link_pattern = re.compile(
        r"!?\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*([^\s)]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
    )
    bare_url_pattern = re.compile(r"(?<!\]\()<?https?://[^\s<>\]\)\"']+>?")

    def clean(self, text: str) -> tuple[str, list[dict[str, str | int]]]:
        references: list[dict[str, str | int]] = []
        url_to_id: dict[str, int] = {}

        def add_reference(url: str, label: str = "") -> int:
            normalized = canonicalize_url(url) or url
            if normalized not in url_to_id:
                ref_id = len(references) + 1
                url_to_id[normalized] = ref_id
                references.append({"id": ref_id, "text": clean_markdown(label), "url": normalized})
            return url_to_id[normalized]

        def replace_markdown(match: re.Match[str]) -> str:
            label, url = match.group(1).strip(), match.group(2).strip().strip("<>")
            label = clean_markdown(re.sub(r"\[|\]", "", label))
            if not url.lower().startswith(("http://", "https://")):
                return label
            ref_id = add_reference(url, label)
            return f"{label} [{ref_id}]" if label else f"[{ref_id}]"

        previous = None
        while previous != text:
            previous = text
            text = self.markdown_link_pattern.sub(replace_markdown, text)

        def replace_bare(match: re.Match[str]) -> str:
            raw_url = match.group(0).strip("<>")
            trailing = ""
            while raw_url and raw_url[-1] in ".,;:":
                trailing = raw_url[-1] + trailing
                raw_url = raw_url[:-1]
            ref_id = add_reference(raw_url)
            return f"[{ref_id}]{trailing}"

        text = self.bare_url_pattern.sub(replace_bare, text)
        text = re.sub(r"\[[^\]]*\]\(\s*(?:javascript|mailto|tel):[^)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return clean_markdown(text), references


def split_markdown_table_row(line: str) -> list[str]:
    """Divide una riga tabellare markdown in celle pulite."""
    line = line.strip().strip("|")
    return [clean_markdown(cell) for cell in line.split("|")]


def is_markdown_table_separator(line: str) -> bool:
    """Riconosce la riga --- | :---: tipica delle tabelle markdown."""
    cells = split_markdown_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells if cell)


def looks_like_real_table_header(cells: list[str]) -> bool:
    """Distingue intestazioni vere da prime righe dati usate come falso header."""
    joined = " ".join(cells).lower()
    header_markers = {
        "#", "nome", "cognome", "componente", "ruolo", "funzione", "email", "telefono",
        "insegnamento", "corso", "docente", "cfu", "ssd", "semestre", "anno",
        "aula", "orario", "data", "titolo", "struttura", "scadenza",
    }
    return any(re.search(rf"\b{re.escape(marker)}\b", joined) for marker in header_markers if marker != "#") or "#" in cells


def looks_like_person_name(cell: str) -> bool:
    """Riconosce celle che sembrano nomi di persone, senza usare regole specifiche DIEM."""
    clean = re.sub(r"[*_`]", "", cell).strip()
    return bool(
        re.fullmatch(
            r"[A-Z?-?][A-Za-z?-?']+\s+[A-Z?-?][A-Za-z?-?']+(?:\s+[A-Z?-?][A-Za-z?-?']+)?",
            clean,
        )
    )


def looks_like_role(cell: str) -> bool:
    """Riconosce celle che sembrano ruoli/funzioni universitarie."""
    lowered = cell.lower()
    role_words = (
        "direttore", "membro", "membri", "presidente", "coordinatore", "delegato", "rappresentante",
        "vicario", "responsabile", "componente", "segretario", "referente",
    )
    return any(word in lowered for word in role_words)


def infer_headers_from_first_data_row(row: list[str]) -> list[str]:
    """Crea header artificiali quando la tabella markdown non espone intestazioni reali."""
    headers: list[str] = []
    for index, cell in enumerate(row):
        lowered = cell.lower()
        stripped = cell.strip()
        if index == 0 and re.fullmatch(r"\d+", stripped):
            headers.append("#")
        elif looks_like_person_name(cell):
            headers.append("Nome")
        elif "semestre" in lowered:
            headers.append("Semestre")
        elif "moodle" in lowered:
            headers.append("Moodle")
        elif "teams" in lowered:
            headers.append("Teams")
        elif looks_like_role(cell):
            headers.append("Ruolo")
        elif re.fullmatch(r"\[\s*\d+\s*\]", stripped):
            headers.append("Profilo")
        elif "@" in cell:
            headers.append("Email")
        elif re.fullmatch(r"\d{3,}", stripped):
            headers.append("Codice")
        elif index == 1 and len(cell) > 8:
            headers.append("Insegnamento")
        else:
            headers.append(f"Colonna {index + 1}")
    return headers


def table_block_to_semantic_rows(lines: list[str], table_index: int) -> str:
    """Converte una tabella markdown in righe header-valore per non perdere le colonne."""
    if len(lines) < 3 or not is_markdown_table_separator(lines[1]):
        return "\n".join(lines)

    first_row = split_markdown_table_row(lines[0])
    remaining_rows = [split_markdown_table_row(line) for line in lines[2:]]
    if looks_like_real_table_header(first_row):
        headers = first_row
        data_rows = remaining_rows
    else:
        headers = infer_headers_from_first_data_row(first_row)
        data_rows = [first_row, *remaining_rows]

    output = [f"[TABELLA {table_index}]"]
    for row in data_rows:
        pairs = [
            f"{header}: {cell}"
            for header, cell in zip(headers, row)
            if header and cell
        ]
        if pairs:
            output.append("Riga tabella: " + "; ".join(pairs))
    output.append(f"[/TABELLA {table_index}]")
    return "\n".join(output)


def semanticize_loose_pipe_rows(text: str) -> str:
    """Converte righe tabellari pipe rimaste senza blocco markdown completo."""
    output: list[str] = []
    loose_table_index = 1

    for line in text.splitlines():
        if line.count("|") < 4:
            output.append(line)
            continue

        cells = [clean_markdown(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            output.append(line)
            continue

        headers = infer_headers_from_first_data_row(cells)
        pairs = [f"{header}: {cell}" for header, cell in zip(headers, cells) if cell]
        if pairs:
            output.append(f"[RIGA TABELLA {loose_table_index}] " + "; ".join(pairs))
            loose_table_index += 1
        else:
            output.append(line)

    return "\n".join(output)


def convert_markdown_tables_to_semantic_rows(text: str) -> str:
    """Trasforma tabelle markdown in righe autoesplicative prima del chunking."""
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    table_index = 1

    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" in line and is_markdown_table_separator(next_line):
            table_lines = [line, next_line]
            index += 2
            while index < len(lines) and "|" in lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            output.append(table_block_to_semantic_rows(table_lines, table_index))
            table_index += 1
            continue
        output.append(line)
        index += 1

    return clean_markdown(semanticize_loose_pipe_rows("\n".join(output)))


def infer_table_label(previous_lines: list[str]) -> str:
    """Trova l'etichetta della tabella dal titolo contenitore precedente.

    Nelle pagine UNISA capita spesso questa struttura:

    Giunta
    Componenti
    | Nome | Ruolo |

    In quel caso "Componenti" non è il nome reale della tabella: è un
    sottotitolo generico. La label corretta è il titolo contenitore ("Giunta").
    Per questo saltiamo i titoli generici e continuiamo a cercare verso l'alto.
    """
    generic_labels = {
        "componenti",
        "tabella",
        "tabelle",
        "tabelle estratte da crawl4ai",
    }
    def is_probably_plain_title(label: str) -> bool:
        """Accetta solo righe che sembrano titoli, non frasi del contenuto."""
        normalized = label.strip()
        lowered = normalized.lower()
        body_starters = (
            "da ",
            "dai ",
            "dalle ",
            "dagli ",
            "di ",
            "del ",
            "della ",
            "degli ",
            "dello ",
            "ai ",
            "il ",
            "la ",
            "lo ",
            "le ",
            "gli ",
            "un ",
            "una ",
            "nella ",
            "nello ",
            "partecipa ",
            "riga tabella",
        )
        if lowered in generic_labels:
            return False
        if lowered.startswith(body_starters):
            return False
        if any(char in normalized for char in ".;:"):
            return False
        if len(normalized.split()) > 6:
            return False
        return bool(normalized and normalized[0].isupper())

    generic_fallback = ""

    for raw_line in reversed(previous_lines[-12:]):
        line = clean_markdown(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            label = re.sub(r"^#+\s*", "", line)
            label = re.sub(r"\s*\[\d+\]", "", label).strip()
            if label:
                if label.lower() in generic_labels:
                    generic_fallback = generic_fallback or label
                    continue
                return label
        if len(line) <= 80 and not line.startswith("|") and not is_markdown_table_separator(line):
            label = re.sub(r"[*_`]", "", line).strip()
            if label and is_probably_plain_title(label):
                return label
            if label.lower() in generic_labels:
                generic_fallback = generic_fallback or label
    return generic_fallback


def extract_markdown_table_blocks(markdown_text: str) -> str:
    """Estrae solo tabelle markdown dal testo prodotto da Crawl4AI."""
    markdown_text = clean_markdown(markdown_text or "")
    if not markdown_text:
        return ""

    lines = markdown_text.splitlines()
    output: list[str] = []
    previous_lines: list[str] = []
    index = 0
    table_index = 1

    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" in line and is_markdown_table_separator(next_line):
            table_lines = [line, next_line]
            index += 2
            while index < len(lines) and "|" in lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            label = infer_table_label(previous_lines)
            table_text = table_block_to_semantic_rows(table_lines, table_index)
            if label:
                table_text = table_text.replace(f"[TABELLA {table_index}]", f"[TABELLA {table_index} - {label}]", 1)
            output.append(table_text)
            table_index += 1
            continue
        previous_lines.append(line)
        index += 1

    return clean_markdown("\n\n".join(output))


def merge_trafilatura_text_with_crawl4ai_tables(trafilatura_text: str, crawl4ai_markdown: str) -> str:
    """Unisce testo pulito Trafilatura e tabelle preservate da Crawl4AI."""
    parts = [clean_markdown(trafilatura_text)]
    table_text = extract_markdown_table_blocks(crawl4ai_markdown)
    if table_text:
        parts.append("## Tabelle estratte da Crawl4AI\n" + table_text)
    return clean_markdown("\n\n".join(part for part in parts if part))


def split_atomic_table_blocks(text: str) -> list[tuple[str, str]]:
    """Separa testo normale e blocchi tabellari semanticizzati."""
    blocks: list[tuple[str, str]] = []
    table_pattern = re.compile(r"(\[TABELLA[^\n]*\].*?\[/TABELLA[^\n]*\])", re.DOTALL)
    position = 0
    for match in table_pattern.finditer(text):
        before = clean_markdown(text[position:match.start()])
        if before:
            blocks.append(("text", before))
        table = clean_markdown(match.group(1))
        if table:
            blocks.append(("table", table))
        position = match.end()
    after = clean_markdown(text[position:])
    if after:
        blocks.append(("text", after))
    return blocks


def split_large_table_by_rows(table_text: str, max_chars: int) -> list[str]:
    """Divide una tabella grande solo tra righe, mai a metà riga."""
    lines = [line for line in table_text.splitlines() if line.strip()]
    if len(table_text) <= max_chars or len(lines) <= 3:
        return [table_text]

    header = lines[0]
    footer = lines[-1] if lines[-1].startswith("[/TABELLA") else ""
    rows = lines[1:-1] if footer else lines[1:]
    chunks: list[str] = []
    current: list[str] = [header]

    for row in rows:
        candidate = "\n".join([*current, row, footer]).strip()
        if len(candidate) > max_chars and len(current) > 1:
            if footer:
                current.append(footer)
            chunks.append(clean_markdown("\n".join(current)))
            current = [header, row]
        else:
            current.append(row)

    if footer:
        current.append(footer)
    if len(current) > 1:
        chunks.append(clean_markdown("\n".join(current)))
    return chunks


class TrafilaturaExtractor:
    """Estrae testo pulito da HTML usando trafilatura, mantenendo tabelle e struttura."""

    def extract_markdown(self, html: str, url: str) -> str:
        """
        Estrae testo da HTML con trafilatura.
        Rimuove: banners, menu, footer, script, cookie dialogs.
        Mantiene: tabelle, liste, heading structure.
        """
        try:
            # Estrazione con trafilatura (include_comments=False per rimuovere commenti HTML)
            extracted = extract(
                html,
                include_comments=False,
                favor_precision=True, 
                output_format="markdown",
            )
            
            if not extracted:
                return ""
            
            # Pulizia base (mantiene markdown structure)
            text = re.sub(r"\n{4,}", "\n\n\n", extracted)
            text = re.sub(r"[ \t]+", " ", text)
            return text.strip()
            
        except Exception as e:
            # Fallback: se trafilatura fallisce, empty string
            print(f"Trafilatura extraction fallita per {url}: {e}")
            return ""

    def extract_metadata(self, html: str) -> dict[str, str]:
        """Estrae metadata basic (title, description) con trafilatura."""
        try:
            meta = bare_extraction(html)
            return {
                "title": meta.get("title") or "",
                "author": meta.get("author") or "",
                "description": meta.get("description") or "",
            }
        except Exception:
            return {"title": "", "author": "", "description": ""}


def extract_page_summary(text: str, title: str, url: str, max_length: int = 300) -> str:
    """
    Genera riassunto della pagina per contesto retrieval.
    Estrae: titolo + primi heading + prime frasi.
    """
    lines = text.split("\n")
    summary_parts = [f"Pagina: {title}"]
    
    # Estrai heading principali (primi 3)
    heading_count = 0
    for line in lines[:20]:
        if line.startswith("#"):
            summary_parts.append(line.replace("#", "").strip())
            heading_count += 1
            if heading_count >= 3:
                break
    
    # Aggiungi prime frasi significative
    for line in lines:
        if line and not line.startswith("#") and len(line) > 20:
            summary_parts.append(line[:100] + "...")
            break
    
    summary = " | ".join(summary_parts)
    return summary[:max_length]


def sha256_text(text: str) -> str:
    """Hash SHA256 stabile per deduplication."""
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def short_id(seed: str) -> str:
    """ID breve stabile da seed URL/path."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


class SimpleMarkdownChunker:
    """MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter, senza small-to-big."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 190, max_table_chars: int = 3000) -> None:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Titolo_1"),
                ("##", "Titolo_2"),
                ("###", "Titolo_3"),
                ("####", "Titolo_4"),
            ],
            strip_headers=False,
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        self.link_cleaner = LinkReferenceCleaner()
        self.max_table_chars = max_table_chars

    def split(self, markdown_text: str) -> list[tuple[str, dict[str, str], list[dict[str, str | int]]]]:
        """Restituisce chunk testuali con headings e references del singolo chunk."""
        markdown_text = convert_markdown_tables_to_semantic_rows(clean_markdown(markdown_text))
        if not markdown_text:
            return []
        markdown_docs = self.markdown_splitter.split_text(markdown_text)

        chunks: list[tuple[str, dict[str, str], list[dict[str, str | int]]]] = []
        for doc in markdown_docs:
            headings = {key: value for key, value in doc.metadata.items() if value}
            splits: list[str] = []
            for block_type, block_text in split_atomic_table_blocks(doc.page_content):
                if block_type == "table":
                    splits.extend(split_large_table_by_rows(block_text, self.max_table_chars))
                else:
                    splits.extend(self.recursive_splitter.split_text(block_text))

            for split in splits:
                cleaned_text, references = self.link_cleaner.clean(split)
                if cleaned_text:
                    chunks.append((cleaned_text, headings, references))

        return chunks


class SimpleDiemCrawler:
    """Crawler DIEM: Trafilatura per HTML, BM25 dedup, chunking semantico."""

    def __init__(
        self,
        output_dir: Path,
        start_urls: list[str],
        max_pages: int = 1500,
        max_depth: int = 5,
        concurrency: int = 8,
        chunk_size: int = 1500,
        chunk_overlap: int = 190,
    ) -> None:
        self.output_dir = output_dir
        self.start_urls = start_urls
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency

        self.queue: asyncio.Queue[tuple[str, int, str | None]] = asyncio.Queue()
        self.queued: set[str] = set()
        self.lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

        self.expanded_pages = 0
        self.pages_written = 0
        self.chunks_written = 0
        self.duplicates = 0
        self.near_duplicates = 0
        self.pdf_with_text = 0
        self.pdf_without_text = 0
        self.pdf_errors = 0
        self.errors: list[str] = []

        self.seen_page_hashes: set[str] = set()
        self.seen_chunk_hashes: set[str] = set()
        self.deduplicator = PageDeduplicator()
        self.chunker = SimpleMarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.extractor = TrafilaturaExtractor()
        self.progress = None

    async def crawl(self) -> None:
        """Entry point crawling asincrono."""
        self.loop = asyncio.get_event_loop()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        chunks_file_path = self.output_dir / "diem_chunks.jsonl"
        pages_file_path = self.output_dir / "diem_pages.jsonl"

        with chunks_file_path.open("w", encoding="utf-8") as chunks_file, pages_file_path.open(
            "w", encoding="utf-8"
        ) as pages_file:
            self.chunks_file = chunks_file
            self.pages_file = pages_file

            for start_url in self.start_urls:
                normalized = canonicalize_url(start_url)
                if normalized and is_allowed_crawl_url(normalized):
                    await self.queue.put((normalized, 0, None))
                    self.queued.add(normalized)

            self.progress = tqdm(total=self.max_pages, desc="Crawl DIEM", unit="pages")
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

            browser_config = BrowserConfig(headless=True, verbose=False)
            run_config = CrawlerRunConfig(word_count_threshold=5, verbose=False)
            async with AsyncWebCrawler(config=browser_config) as crawler:
                workers = [
                    asyncio.create_task(self._worker(crawler, run_config))
                    for _ in range(self.concurrency)
                ]
                await self.queue.join()

                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

            if self.progress:
                self.progress.close()

        self.print_report()

    async def _worker(self, crawler, run_config) -> None:
        """Worker asincrono: processa URL dalla coda."""
        while True:
            try:
                url, depth, parent_url = await self.queue.get()
            except asyncio.CancelledError:
                break

            try:
                if depth > self.max_depth:
                    continue

                url = canonicalize_url(url) or url
                if not is_allowed_crawl_url(url):
                    continue

                if is_institutional_pdf_url(url):
                    await asyncio.to_thread(self.process_pdf_url, url, depth, parent_url)
                else:
                    result = await crawler.arun(url=url, config=run_config)
                    await self.process_html_result(url, depth, result)
            except Exception as exc:
                self.errors.append(f"{url}: {exc}")
            finally:
                self.queue.task_done()

    async def process_html_result(self, url: str, depth: int, result) -> None:
        """Gestisce una pagina HTML: estrae con trafilatura, link, PDF."""
        if not getattr(result, "success", False):
            self.errors.append(f"{url}: crawl non riuscito")
            return

        final_url = canonicalize_url(getattr(result, "url", None) or url) or url
        if not is_allowed_crawl_url(final_url):
            return

        # Testo principale: Trafilatura. Tabelle: markdown prodotto da Crawl4AI.
        html_content = getattr(result, "content", None) or getattr(result, "html", "") or ""
        trafilatura_markdown = self.extractor.extract_markdown(html_content, final_url) if html_content else ""
        crawl4ai_markdown = clean_markdown(getattr(result, "fit_markdown", None) or getattr(result, "markdown", "") or "")
        markdown = merge_trafilatura_text_with_crawl4ai_tables(trafilatura_markdown, crawl4ai_markdown)
        
        if not markdown:
            return

        if (is_rubrica_url(final_url) or is_docenti_url(final_url)) and not has_diem_affiliation(markdown):
            return
        if is_clearly_non_diem_bandi_text(markdown):
            return

        content_type = getattr(result, "response_headers", {}).get("content-type", "text/markdown")
        if is_probably_pdf(final_url, content_type):
            await asyncio.to_thread(self.process_pdf_url, final_url, depth, url)
            return

        should_expand = False
        async with self.lock:
            if self.expanded_pages < self.max_pages:
                self.expanded_pages += 1
                should_expand = True
                if self.progress:
                    self.progress.update(1)
                    self.progress.set_postfix(
                        salvate=self.pages_written,
                        chunk=self.chunks_written,
                        coda=self.queue.qsize(),
                        errori=len(self.errors),
                    )

        if should_expand and depth + 1 <= self.max_depth:
            for link in collect_result_links(result, final_url):
                if is_allowed_crawl_url(link) and link not in self.queued:
                    self.queued.add(link)
                    await self.queue.put((link, depth + 1, final_url))
            for pdf_url in markdown_pdf_links(markdown):
                if is_allowed_crawl_url(pdf_url) and pdf_url not in self.queued:
                    self.queued.add(pdf_url)
                    await self.queue.put((pdf_url, depth + 1, final_url))

        title = clean_markdown(getattr(result, "metadata", {}).get("title", "") or getattr(result, "title", "") or final_url)
        await self.write_page_chunks(final_url, final_url, title, markdown, depth, "html", "text/markdown", None)

    def process_pdf_url(self, url: str, depth: int, parent_url: str | None) -> None:
        """Scarica un PDF istituzionale, estrae testo pagina per pagina e crea chunk."""
        try:
            pdf_bytes, content_type, final_url = fetch_url_bytes(url)
            pages = extract_pdf_pages(pdf_bytes, clean_markdown)
        except Exception as exc:
            self.pdf_errors += 1
            self.errors.append(f"{url}: PDF non raggiungibile/leggibile: {exc}")
            return

        if not pages:
            self.pdf_without_text += 1
            return

        self.pdf_with_text += 1
        title = title_from_url(final_url)
        for page_number, page_text in pages:
            context = pdf_page_context(title, final_url, page_number, page_text)
            markdown = f"{context}\n\n{page_text}"
            asyncio.run_coroutine_threadsafe(
                self.write_page_chunks(final_url, final_url, title, markdown, depth, "pdf", content_type, parent_url, page_number),
                self.loop,
            ).result()

    async def write_page_chunks(
        self,
        url: str,
        final_url: str,
        title: str,
        markdown: str,
        depth: int,
        source_type: str,
        content_type: str,
        parent_url: str | None,
        pdf_page: int | None = None,
    ) -> None:
        """Deduplica pagina/chunk e scrive record JSONL con page_context."""
        async with self.write_lock:
            page_hash = sha256_text(markdown)
            if page_hash in self.seen_page_hashes:
                self.duplicates += 1
                return
            near_duplicate_of, _similarity = self.deduplicator.find_near_duplicate(markdown)
            if near_duplicate_of:
                self.near_duplicates += 1
                return
            self.seen_page_hashes.add(page_hash)
            self.deduplicator.register_unique_page(final_url, markdown)

            # Calcola page_context una volta per la pagina
            page_context = extract_page_summary(markdown, title, final_url)

            page_id_seed = f"{final_url}#page={pdf_page}" if pdf_page else final_url
            page_id = short_id(page_id_seed)
            chunks = self.chunker.split(markdown)
            written_for_page = 0

            for index, (text, headings, references) in enumerate(chunks):
                chunk_hash = sha256_text(text)
                if chunk_hash in self.seen_chunk_hashes:
                    continue
                self.seen_chunk_hashes.add(chunk_hash)
                chunk = SimpleChunk(
                    chunk_id=f"{page_id}:{index:04d}",
                    page_id=page_id,
                    url=final_url,
                    page_title=title,
                    text=text,
                    chunk_index=index,
                    token_estimate=max(1, len(text) // 4),
                    depth=depth,
                    content_type=content_type,
                    hash_code=chunk_hash,
                    source_type=source_type,
                    references=references,
                    page_context=page_context,  
                    parent_url=parent_url,
                    pdf_page=pdf_page,
                    headings=headings,
                )
                self.chunks_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                written_for_page += 1
                self.chunks_written += 1

            if written_for_page:
                page = SimplePage(
                    page_id=page_id,
                    url=url,
                    final_url=final_url,
                    title=title,
                    status_code=200,
                    content_type=content_type,
                    depth=depth,
                    hash_code=page_hash,
                    crawled_at=datetime.now(timezone.utc).isoformat(),
                    chunks_count=written_for_page,
                    text_chars=len(markdown),
                    source_type=source_type,
                )
                self.pages_file.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                self.pages_written += 1
                if self.progress:
                    self.progress.set_postfix(
                        salvate=self.pages_written,
                        chunk=self.chunks_written,
                        coda=self.queue.qsize(),
                        errori=len(self.errors),
                    )

    def print_report(self) -> None:
        """Stampa un riepilogo finale leggibile."""
        print("\nCrawl con trafilatura completato")
        print(f"Pagine usate per espandere la coda: {self.expanded_pages}")
        print(f"Pagine salvate: {self.pages_written}")
        print(f"Chunk creati: {self.chunks_written}")
        print(f"PDF analizzati con testo: {self.pdf_with_text}")
        print(f"PDF senza testo estraibile: {self.pdf_without_text}")
        print(f"PDF non raggiungibili / errore download: {self.pdf_errors}")
        print(f"Pagine duplicate saltate: {self.duplicates}")
        print(f"Pagine quasi duplicate TF-IDF saltate: {self.near_duplicates}")
        print(f"Errori: {len(self.errors)}")
        if self.errors:
            print("Primi errori:")
            for error in self.errors[:10]:
                print(f"- {error}")


def parse_args() -> argparse.Namespace:
    """Parametri CLI del crawler."""
    parser = argparse.ArgumentParser(description="Crawler DIEM con trafilatura: pulizia HTML + page_context.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pages", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1100)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--start-url", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    start_urls = args.start_url or DEFAULT_START_URLS
    crawler = SimpleDiemCrawler(
        output_dir=args.output_dir,
        start_urls=start_urls,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    asyncio.run(crawler.crawl())


if __name__ == "__main__":
    main()
