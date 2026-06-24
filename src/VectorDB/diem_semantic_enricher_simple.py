from __future__ import annotations
 
import argparse
import hashlib
import json
import re
from json import JSONDecodeError
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm


ITALIAN_STOPWORDS = {
    "alla", "allo", "alle", "agli", "dalla", "dallo", "delle", "degli", "della", "dell", "dell'",
    "del", "dei", "con", "per", "tra", "fra", "che", "non", "sono", "come", "anche", "nel", "nella",
    "nei", "nelle", "sul", "sulla", "sui", "sulle", "una", "uno", "gli", "le", "la", "il", "lo",
    "di", "a", "da", "in", "su", "e", "o", "un", "al", "ai", "ad", "ed", "dei", "più", "può",
    "università", "salerno", "diem", "dipartimento", "pagina", "testo", "passage", "contesto",
}


def normalize_text(text: str) -> str:
    """Normalizza il testo per hash, keyword e confronti semplici."""
    text = re.sub(r"^passage:\s*", "", text or "", flags=re.IGNORECASE).strip()
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(text: str) -> str:
    """Hash stabile del chunk normalizzato, usato per deduplicare prima degli embedding."""
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_keywords(text: str, max_keywords: int = 18) -> list[str]:
    """Estrae keyword semplici e spiegabili con frequenza su parole significative."""
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]{2,}", normalize_text(text).lower())
    candidates = [
        word.strip("-'")
        for word in words
        if word not in ITALIAN_STOPWORDS and not word.isdigit()
    ]
    counts = Counter(candidates)
    return [word for word, _count in counts.most_common(max_keywords)]


def extract_academic_years(text: str) -> list[str]:
    """Trova anni accademici nel formato 2025/2026 o 2025/26."""
    found = re.findall(r"\b(20\d{2}\s*/\s*(?:20)?\d{2})\b", text or "")
    normalized = []
    for item in found:
        left, right = [part.strip() for part in item.split("/", 1)]
        if len(right) == 2:
            right = left[:2] + right
        value = f"{left}/{right}"
        if value not in normalized:
            normalized.append(value)
    return normalized


def extract_years(text: str) -> list[str]:
    """Trova anni civili nel testo."""
    years = []
    for year in re.findall(r"\b(20\d{2})\b", text or ""):
        if year not in years:
            years.append(year)
    return years


def extract_people(text: str, max_people: int = 12) -> list[str]:
    """Estrae nomi probabili senza pretendere NER perfetto: solo pattern Nome Cognome / COGNOME Nome."""
    clean = normalize_text(text)
    patterns = [
        r"\b(?:prof\.?|professore|professoressa|dott\.?|dottore|dottoressa)\s+([A-ZÀ-Ý][a-zà-ÿ']+\s+[A-ZÀ-Ý][a-zà-ÿ']+)\b",
        r"\b([A-ZÀ-Ý]{2,}\s+[A-ZÀ-Ý][a-zà-ÿ']+)\b",
        r"\b([A-ZÀ-Ý][a-zà-ÿ']+\s+[A-ZÀ-Ý][a-zà-ÿ']+)\b",
    ]
    people: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, clean):
            value = re.sub(r"\s+", " ", match).strip()
            if value.lower() in ITALIAN_STOPWORDS or value in people:
                continue
            if len(value.split()) == 2:
                people.append(value)
            if len(people) >= max_people:
                return people
    return people


def extract_course_names(text: str, max_courses: int = 12) -> list[str]:
    """Estrae possibili nomi di insegnamenti da righe in maiuscolo o campi tabellari."""
    clean = normalize_text(text)
    courses: list[str] = []
    for line in re.split(r"[\n;]", text or ""):
        line = re.sub(r"^passage:\s*", "", line).strip()
        if not line:
            continue
        if re.search(r"\b(insegnamento|corso|modulo)\s*:", line, flags=re.IGNORECASE):
            value = re.sub(r".*?:", "", line, count=1).strip()
        elif line.isupper() and 8 <= len(line) <= 90:
            value = line.title()
        else:
            continue
        value = re.sub(r"\s+", " ", value).strip(" -|")
        if value and value not in courses:
            courses.append(value)
        if len(courses) >= max_courses:
            break
    if not courses:
        for match in re.findall(r"\b[A-Z][A-Z ]{8,80}\b", clean):
            value = re.sub(r"\s+", " ", match).strip().title()
            if value and value not in courses:
                courses.append(value)
            if len(courses) >= max_courses:
                break
    return courses


def classify_page_type(url: str, title: str, text: str, source_type: str) -> str:
    """Classifica larga e verificabile del tipo di contenuto."""
    joined = f"{url} {title} {text[:1200]}".lower()
    path = urlparse(url).path.lower()
    if source_type == "pdf":
        if "piano" in joined and "studi" in joined:
            return "study_plan_pdf"
        if "regolamento" in joined:
            return "regulation_pdf"
        if "bando" in joined or "/bandi" in joined:
            return "call_pdf"
        return "pdf"
    if "docenti.unisa.it" in url:
        if "didattica" in path:
            return "teacher_teaching_page"
        return "teacher_profile"
    if "rubrica.unisa.it" in url:
        return "directory_profile"
    if "corsi.unisa.it" in url:
        if "piano-di-studi" in path:
            return "study_plan_page"
        return "course_site_page"
    if "/home/bandi" in path or "bando" in joined:
        return "call_page"
    if "/dipartimento" in path:
        return "department_page"
    if "/didattica" in path:
        return "teaching_page"
    return "web_page"


def build_retrieval_text(record: dict, metadata: dict) -> str:
    """Costruisce il testo mandato agli embedding: metadati utili + chunk pulito."""
    chunk_text = record.get("text") or ""
    lines = ["passage:"]
    if record.get("page_title"):
        lines.append(f"TITOLO: {record['page_title']}")
    lines.append(f"TIPO: {metadata['page_type']}")
    if metadata["academic_years"]:
        lines.append("ANNI_ACCADEMICI: " + ", ".join(metadata["academic_years"]))
    if metadata["keywords"]:
        lines.append("PAROLE_CHIAVE: " + ", ".join(metadata["keywords"]))
    lines.append("")
    lines.append("TESTO:")
    lines.append(chunk_text)
    return "\n".join(lines)


def enrich_record(record: dict) -> dict:
    """Arricchisce un chunk con metadati leggeri e tracciabili."""
    text = record.get("text", "")
    metadata = {
        "domain": urlparse(record.get("url", "")).netloc.lower(),
        "page_type": classify_page_type(
            record.get("url", ""),
            record.get("page_title", ""),
            text,
            record.get("source_type", "html"),
        ),
        "keywords": extract_keywords(text),
        "academic_years": extract_academic_years(text),
        "years": extract_years(text),
        "people": extract_people(text),
        "course_names": extract_course_names(text),
    }
    enriched = dict(record)
    enriched.update(metadata)
    enriched["semantic_dedupe_hash"] = stable_hash(text)
    enriched["retrieval_text"] = build_retrieval_text(record, metadata)
    return enriched


def parse_args() -> argparse.Namespace:
    """Parametri CLI dell'enricher."""
    parser = argparse.ArgumentParser(description="Arricchimento semantico semplice e verificabile dei chunk DIEM.")
    parser.add_argument("--crawl-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dedupe", action="store_true", help="Salta chunk testualmente duplicati prima degli embedding.")
    return parser.parse_args()


def main() -> None:
    """Legge diem_chunks.jsonl e produce semantic_chunks.jsonl."""
    args = parse_args()
    input_file = args.crawl_dir / "diem_chunks.jsonl"
    output_file = args.output_dir / "semantic_chunks.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    total = 0
    written = 0
    duplicates = 0
    decode_or_json_errors = 0
    incomplete_records = 0

    with input_file.open("r", encoding="utf-8", errors="replace") as source, output_file.open("w", encoding="utf-8") as target:
        for line in tqdm(source, desc="Arricchimento semplice", unit="chunk"):
            total += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except JSONDecodeError:
                decode_or_json_errors += 1
                continue
            if not isinstance(record, dict) or not record.get("chunk_id") or not record.get("text"):
                incomplete_records += 1
                continue
            dedupe_hash = stable_hash(record.get("text", ""))
            if args.dedupe and dedupe_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(dedupe_hash)
            enriched = enrich_record(record)
            target.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            written += 1

    report = {
        "input_chunks": total,
        "output_chunks": written,
        "duplicates_skipped": duplicates,
        "decode_or_json_errors": decode_or_json_errors,
        "incomplete_records_skipped": incomplete_records,
        "dedupe_enabled": bool(args.dedupe),
        "output_file": str(output_file),
    }
    (args.output_dir / "metadata_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Esportati {written} chunk arricchiti in: {output_file}")
    if args.dedupe:
        print(f"Deduplica enricher: {duplicates} chunk duplicati saltati su {total}")
    if decode_or_json_errors:
        print(f"Righe corrotte/non leggibili saltate: {decode_or_json_errors}")
    if incomplete_records:
        print(f"Record incompleti saltati: {incomplete_records}")


if __name__ == "__main__":
    main()
