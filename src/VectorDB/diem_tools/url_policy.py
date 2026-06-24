from __future__ import annotations

import re
from urllib.parse import parse_qs, parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

ALLOWED_DOMAINS = {
    "www.diem.unisa.it",
    "diem.unisa.it",
    "docenti.unisa.it",
    "rubrica.unisa.it",
    "corsi.unisa.it",
}
ALLOWED_INSTITUTIONAL_PDF_DOMAINS = {
    "www.diem.unisa.it",
    "diem.unisa.it",
    "corsi.unisa.it",
    "docenti.unisa.it",
    "www.unisa.it",
    "web.unisa.it",
}
ALLOWED_CORSI_PREFIXES = (
    "/ingegneria-dell-informazione-per-la-medicina-digitale",
    "/ingegneria-informatica",
    "/electrical-engineering-for-digital-energy",
    "/information-engineering-for-digital-medicine",
    "/ingegneria-informatica-magistrale",
    "/ingegneria-dell-informazione",
    "/photovoltaics",
)
DEFAULT_START_URLS = [
    "https://www.diem.unisa.it/",
    "https://www.diem.unisa.it/dipartimento/personale",
    "https://www.diem.unisa.it/didattica/offerta-formativa",
    "https://corsi.unisa.it/ingegneria-informatica",
    "https://corsi.unisa.it/ingegneria-informatica/didattica/piano-di-studi",
    "https://corsi.unisa.it/ingegneria-informatica-magistrale",
    "https://corsi.unisa.it/ingegneria-informatica-magistrale/didattica/piano-di-studi",
    "https://corsi.unisa.it/ingegneria-dell-informazione",
    "https://corsi.unisa.it/ingegneria-dell-informazione/didattica/piano-di-studi",
    "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
    "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale/didattica/piano-di-studi",
    "https://corsi.unisa.it/information-engineering-for-digital-medicine",
    "https://corsi.unisa.it/information-engineering-for-digital-medicine/didattica/piano-di-studi",
    "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
    "https://corsi.unisa.it/electrical-engineering-for-digital-energy/didattica/piano-di-studi",
    "https://corsi.unisa.it/photovoltaics",
    "https://corsi.unisa.it/photovoltaics/didattica/piano-di-studi",
]
DROP_QUERY_PREFIXES = ("utm_",)
DROP_QUERY_KEYS = {"fbclid", "gclid", "yclid", "sitemap"}
SKIP_PATH_PARTS = {"sitemap", "mappa-del-sito", "search", "cerca", "print", "stampa"}
PDF_TYPES = ("application/pdf",)
ALLOWED_DIEM_BANDI_STRUCTURES = {"300638"}
BANDI_STRUCTURE_QUERY_KEYS = ("struttura", "cdsStruttura")
DIEM_STRUCTURE_MARKERS = (
    "dipartimento di ingegneria dell'informazione ed elettrica e matematica applicata/diem",
    "dipartimento di ingegneria dell informazione ed elettrica e matematica applicata/diem",
    "dipartimento di ingegneria dell'informazione ed elettrica e matematica applicata",
    "dipartimento di ingegneria dell informazione ed elettrica e matematica applicata",
    "dipartimento di ingegneria dell'informazione ed elettrica e matematica",
    "dipartimento di ingegneria dell informazione ed elettrica e matematica",
    "matematica applicata/diem",
)


def canonicalize_url(url: str) -> str | None:
    """Normalizza URL e rimuove frammenti/query inutili."""
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in DROP_QUERY_KEYS and not any(key.startswith(prefix) for prefix in DROP_QUERY_PREFIXES)
    ]
    path = parsed.path or "/"
    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=path,
        query=urlencode(query_pairs, doseq=True),
    )
    return urlunparse(normalized)


def is_url_allowed(url: str) -> bool:
    """Verifica dominio e path ammessi per il crawl."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in ALLOWED_DOMAINS:
        return False
    if any(part in parsed.path.lower() for part in SKIP_PATH_PARTS):
        return False
    if is_bandi_url(url):
        if not is_allowed_bandi_url(url):
            return False
    if host == "corsi.unisa.it":
        return any(parsed.path.startswith(prefix) for prefix in ALLOWED_CORSI_PREFIXES)
    return True


def is_bandi_url(url: str) -> bool:
    """Riconosce le pagine /home/bandi del portale DIEM."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return host in {"www.diem.unisa.it", "diem.unisa.it"} and parsed.path.rstrip("/") == "/home/bandi"


def is_allowed_bandi_url(url: str) -> bool:
    """Filtra i bandi per evitare strutture non DIEM già note dalla query string."""
    if not is_bandi_url(url):
        return True
    query = parse_qs(urlparse(url).query)
    structures = [
        value
        for key in BANDI_STRUCTURE_QUERY_KEYS
        for value in query.get(key, [])
    ]
    if any(value == "more" for value in structures):
        return False
    if structures:
        return any(value in ALLOWED_DIEM_BANDI_STRUCTURES for value in structures)
    return True


def is_institutional_pdf_url(url: str) -> bool:
    """Accetta PDF solo da domini istituzionali autorizzati."""
    parsed = urlparse(url)
    return parsed.netloc.lower() in ALLOWED_INSTITUTIONAL_PDF_DOMAINS and parsed.path.lower().endswith(".pdf")


def is_probably_pdf(url: str, content_type: str = "") -> bool:
    """Riconosce PDF da estensione o content-type."""
    return url.lower().split("?", 1)[0].endswith(".pdf") or any(kind in content_type for kind in PDF_TYPES)


def is_rubrica_url(url: str) -> bool:
    """Restituisce True per URL del dominio rubrica.unisa.it."""
    return urlparse(url).netloc.lower() == "rubrica.unisa.it"


def is_docenti_url(url: str) -> bool:
    """Restituisce True per URL del dominio docenti.unisa.it."""
    return urlparse(url).netloc.lower() == "docenti.unisa.it"


def docenti_profile_key(url: str) -> str | None:
    """Estrae l'identificatore del profilo da un URL docenti.unisa.it."""
    if not is_docenti_url(url):
        return None
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[0].lower() if parts else None


def same_docenti_profile(left_url: str, right_url: str) -> bool:
    """Verifica se due URL appartengono allo stesso profilo docente."""
    left_key = docenti_profile_key(left_url)
    right_key = docenti_profile_key(right_url)
    return bool(left_key and right_key and left_key == right_key)


def is_rubrica_person_url(url: str) -> bool:
    """Accetta solo pagine persona della rubrica con matricola esplicita."""
    parsed = urlparse(url)
    if parsed.netloc.lower() != "rubrica.unisa.it":
        return False
    path = (parsed.path or "/").rstrip("/") or "/"
    query = parse_qs(parsed.query)
    return path in {"/", "/persone"} and bool(query.get("matricola"))


def is_allowed_crawl_url(url: str) -> bool:
    """Applica tutte le regole di ammissibilità degli URL del progetto."""
    if is_rubrica_url(url):
        return is_rubrica_person_url(url)
    if is_institutional_pdf_url(url):
        return True
    return is_url_allowed(url)


def has_diem_affiliation(text: str) -> bool:
    """Riconosce l'affiliazione DIEM nei profili rubrica/docenti."""
    normalized = text.lower()
    markers = (
        "dipartimento di ingegneria dell'informazione ed elettrica e matematica applicata/diem",
        "dipartimento di ingegneria dell'informazione ed elettrica e matematica",
        "dipartimento di ingegneria dell informazione ed elettrica e matematica",
        "ingegneria dell'informazione ed elettrica e matematica applicata",
        "ingegneria dell informazione ed elettrica e matematica applicata",
        "matematica applicata/diem",
        "matematica applicata / diem",
        "/diem",
        " diem ",
    )
    padded = f" {normalized} "
    return any(marker in padded for marker in markers)


def has_diem_structure_text(text: str) -> bool:
    """Riconosce la struttura DIEM nel contenuto di una pagina bando."""
    normalized = re.sub(r"\s+", " ", (text or "").lower().replace("\\'", "'"))
    padded = f" {normalized} "
    return any(marker in padded for marker in DIEM_STRUCTURE_MARKERS)


def is_clearly_non_diem_bandi_text(text: str) -> bool:
    """
    Identifica pagine bando sicuramente non DIEM.

    Alcune pagine /home/bandi non espongono la struttura nella query string:
    in quel caso il crawler guarda il testo della pagina. La regola è
    conservativa: scarta solo quando compare una struttura esplicita non DIEM
    (ad esempio ATENEO o un altro Dipartimento), così non buttiamo via liste
    generiche che potrebbero contenere link DIEM utili.
    """
    normalized = re.sub(r"\s+", " ", (text or "").lower().replace("\\'", "'"))
    if has_diem_structure_text(normalized):
        return False
    if re.search(r"\bstruttura\s+ateneo\b", normalized):
        return True
    if "struttura dipartimento di " in normalized:
        return True
    return False


def collect_result_links(result: object, base_url: str) -> list[str]:
    """Estrae e filtra i link crawlable da un risultato Crawl4AI."""
    links = getattr(result, "links", None) or {}
    urls = []
    seen = set()
    base_is_rubrica = is_rubrica_url(base_url)
    base_is_docenti = is_docenti_url(base_url)
    for group in ("internal", "external"):
        for link in links.get(group, []) or []:
            href = link.get("href") if isinstance(link, dict) else None
            if not href:
                continue
            absolute = canonicalize_url(urljoin(base_url, href))
            if not absolute or not is_allowed_crawl_url(absolute) or absolute in seen:
                continue
            if base_is_rubrica and is_rubrica_url(absolute):
                continue
            if base_is_docenti and is_docenti_url(absolute) and not same_docenti_profile(base_url, absolute):
                continue
            seen.add(absolute)
            urls.append(absolute)
    return urls
