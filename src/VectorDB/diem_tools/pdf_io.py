from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from VectorDB.diem_tools.url_policy import canonicalize_url, is_institutional_pdf_url

USER_AGENT = "DIEM-Chatbot-Research/1.0 (+university project)"


PDF_URL_PATTERN = re.compile(
    r"https?://[^\s\]\)\}\"']+?\.pdf(?:\?[^\s\]\)\}\"']*)?",
    flags=re.IGNORECASE,
)


def require_pdf_reader():
    """Carica pypdf/PyPDF2 solo quando serve."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError("Manca una libreria PDF. Installa con: pip install pypdf") from exc
    return PdfReader


def require_fitz():
    """Carica PyMuPDF solo quando serve; se manca useremo il fallback pypdf."""
    try:
        import fitz  # type: ignore
        return fitz
    except ImportError:
        return None


def fetch_url_bytes(url: str) -> tuple[bytes, str, str]:
    """Scarica un URL e restituisce bytes, content-type e URL finale."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        content = response.read()
        content_type = response.headers.get("content-type") or "application/pdf"
        final_url = canonicalize_url(response.geturl()) or url
    return content, content_type, final_url


def clean_table_cell(value: object) -> str:
    """Normalizza una cella PDF evitando newline e spazi spurii."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def looks_like_header(row: list[str]) -> bool:
    """Riconosce intestazioni tabellari didattiche in modo conservativo."""
    joined = " ".join(row).lower()
    markers = ("insegnamento", "modulo", "cfu", "ssd", "taf", "tipologia", "semestre")
    return sum(1 for marker in markers if marker in joined) >= 2


def table_to_semantic_rows(rows: list[list[object]], table_index: int) -> str:
    """Converte una tabella PDF in righe chiave-valore leggibili dal RAG."""
    cleaned_rows = [
        [clean_table_cell(cell) for cell in row]
        for row in rows
        if any(clean_table_cell(cell) for cell in row)
    ]
    if not cleaned_rows:
        return ""

    headers: list[str] = []
    data_rows = cleaned_rows
    if looks_like_header(cleaned_rows[0]):
        headers = cleaned_rows[0]
        data_rows = cleaned_rows[1:]

    lines = [f"[TABELLA PDF {table_index}]"]
    for row in data_rows:
        if headers and len(headers) == len(row):
            pairs = [
                f"{header}: {cell}"
                for header, cell in zip(headers, row)
                if header and cell
            ]
            if pairs:
                lines.append("Riga tabella: " + "; ".join(pairs))
                continue
        compact = " | ".join(cell for cell in row if cell)
        if compact:
            lines.append(f"Riga tabella: {compact}")
    lines.append(f"[/TABELLA PDF {table_index}]")
    return "\n".join(lines)


def extract_pdf_pages_with_fitz(pdf_bytes: bytes, clean_markdown) -> list[tuple[int, str]]:
    """Estrae testo e tabelle pagina per pagina usando PyMuPDF quando disponibile."""
    fitz = require_fitz()
    if fitz is None or not hasattr(fitz.Page, "find_tables"):
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []
    try:
        for page_index, page in enumerate(doc, 1):
            blocks: list[str] = []
            table_blocks: list[str] = []
            try:
                found_tables = page.find_tables()
                tables = getattr(found_tables, "tables", []) or []
                for table_index, table in enumerate(tables, 1):
                    table_text = table_to_semantic_rows(table.extract(), table_index)
                    if table_text:
                        table_blocks.append(table_text)
            except Exception:
                table_blocks = []

            if table_blocks:
                blocks.append("## Tabelle estratte con colonne preservate\n" + "\n\n".join(table_blocks))

            page_text = clean_markdown(page.get_text("text") or "")
            if page_text:
                blocks.append("## Testo continuo della pagina\n" + page_text)

            text = clean_markdown("\n\n".join(blocks))
            if text:
                pages.append((page_index, text))
    finally:
        doc.close()
    return pages


def extract_pdf_pages(pdf_bytes: bytes, clean_markdown) -> list[tuple[int, str]]:
    """Estrae testo pagina per pagina, preferendo PyMuPDF table-aware."""
    fitz_pages = extract_pdf_pages_with_fitz(pdf_bytes, clean_markdown)
    if fitz_pages:
        return fitz_pages

    reader = require_pdf_reader()(io.BytesIO(pdf_bytes))
    pages = []
    for page_index, page in enumerate(reader.pages, 1):
        text = clean_markdown(page.extract_text() or "")
        if text:
            pages.append((page_index, text))
    return pages


def pdf_page_context(title: str, url: str, page_number: int, page_text: str) -> str:
    """Crea una testata pagina-aware per non perdere contesto nei PDF tabellari."""
    context = [f"# Documento PDF: {title}", f"Pagina PDF: {page_number}"]
    lowered_url = url.lower()
    if "__piano-studi-cds" in lowered_url:
        context.append("Tipo documento: Piano di Studi ufficiale")
    if "__regolamenti-cds" in lowered_url:
        context.append("Tipo documento: Regolamento didattico ufficiale")

    academic_year = re.search(r"(?:a\.a\.\s*)?(20\d{2}\s*/\s*20\d{2}|20\d{2}\s*/\s*\d{2})", page_text, re.IGNORECASE)
    study_year = re.search(r"\b([123])\s*[°o]\s*anno\b", page_text, re.IGNORECASE)
    semester = re.search(r"\b(primo|secondo)\s+semestre\b", page_text, re.IGNORECASE)
    if academic_year:
        context.append(f"Anno accademico: {academic_year.group(1).replace(' ', '')}")
    if study_year:
        context.append(f"Anno di corso: {study_year.group(1)}")
    if semester:
        context.append(f"Semestre: {semester.group(1).lower()}")
    return "\n".join(context)


def title_from_url(url: str) -> str:
    """Ricava un titolo leggibile dal nome del file PDF."""
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        return url
    return re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE).replace("_", " ").replace("-", " ").strip() or name


def markdown_pdf_links(markdown_text: str) -> list[str]:
    """Estrae URL PDF istituzionali citati nel markdown."""
    urls = []
    seen = set()
    for raw_url in PDF_URL_PATTERN.findall(markdown_text or ""):
        absolute = canonicalize_url(raw_url)
        if not absolute or absolute in seen or not is_institutional_pdf_url(absolute):
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls
