from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
import torch
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


COLLECTION_NAME = "diem_chunks"
DEFAULT_DENSE_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

DENSE_VECTOR_NAME = "dense"
CONTEXT_VECTOR_NAME = "context_dense"
QUESTION_VECTOR_NAME = "question_dense"
SPARSE_VECTOR_NAME = "sparse"


CONTEXT_METADATA_KEYS = [
    "source",
    "section",
    "subsection",
    "title",
    "titolo",
    "course_name",
    "course_type",
    "course_class",
    "academic_year",
    "year",
    "anno",
    "keywords",
    "questions",
    "membri",
    "prof_name",
    "prof_surname",
    "ruolo",
    "struttura",
    "commission",
    "responsabile",
    "responsible_person",
    "nome_progetto",
    "research_area",
    "research_category",
    "universita_partner",
    "paese",
    "programma",
    "scadenza",
    "data_scadenza",
    "data",
    "target",
    "tipologia",
    "doc_type",
]

PAYLOAD_KEYS = [
    "chunk_id",
    "page_id",
    "url",
    "page_title",
    "text",
    "page_context",
    "retrieval_text",
    "question_text",
    "source_type",
    "category",
    "domain",
    "section",
    "subsection",
    "keywords",
    "questions",
    "node_type",
    "raptor_level",
    "children_ids",
    "descendant_leaf_ids",
    "source_urls",
    "source_file",
    "source_index",
    "semantic_dedupe_hash",
    "duplicate_count",
    "duplicate_sources",
]


@dataclass(slots=True)
class BuildStats:
    input_files: int
    input_chunks: int
    leaf_chunks: int
    duplicates_skipped: int
    raptor_nodes: int
    replacement_char_chunks: int


class JinaEmbedder:
    """Wrapper per i modelli Jina che supportano context window lunghe."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL, device: str | None = None, batch_size: int = 32, max_length: int = 8192) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Carica il modello
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})
        
        # IMPOSTA LA GRANDEZZA MASSIMA DEI TOKEN QUI:
        self.model.max_seq_length = max_length

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


def point_id(chunk_id: str) -> str:
    """Stable UUID used as Qdrant point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def text_hash(text: str) -> str:
    normalized = normalize_space(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def metadata_to_text(value: Any, max_items: int = 20) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [metadata_to_text(item) for item in value[:max_items]]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:max_items]:
            item_text = metadata_to_text(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "; ".join(parts)
    return normalize_space(str(value))


def safe_list_of_strings(value: Any, limit: int = 40) -> list[str]:
    items = []
    for item in as_list(value):
        text = metadata_to_text(item)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def truncate(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def slugify(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def category_from_path(relative_path: Path) -> str:
    parts = [part.lower() for part in relative_path.parts]
    if not parts:
        return "unknown"
    first = parts[0]
    if first == "home" and len(parts) > 1:
        first = parts[1]
    aliases = {
        "terzamissione": "terza_missione",
        "terza_missione": "terza_missione",
        "bandi": "bandi",
        "contatti": "contatti",
        "news": "news",
        "eventi": "eventi",
        "offerta_formativa": "offerta_formativa",
        "docenti": "docenti",
        "international": "international",
        "ricerca": "ricerca",
        "dipartimento": "dipartimento",
        "focus_didattica": "focus_didattica",
        "consiglio_didattico": "consiglio_didattico",
        "alternanza_scuola_lavoro": "alternanza_scuola_lavoro",
    }
    return aliases.get(first, slugify(first))


def infer_title(metadata: dict[str, Any], source_file: Path) -> str:
    title_keys = [
        "title",
        "titolo",
        "course_name",
        "nome_progetto",
        "section",
        "subsection",
        "struttura",
        "commission",
    ]
    for key in title_keys:
        value = metadata_to_text(metadata.get(key))
        if value:
            return truncate(value, 180)

    prof_name = metadata_to_text(metadata.get("prof_name"))
    prof_surname = metadata_to_text(metadata.get("prof_surname"))
    if prof_name or prof_surname:
        return normalize_space(f"{prof_name} {prof_surname}")

    source = metadata_to_text(metadata.get("source"))
    if source:
        parsed = urlparse(source)
        path_title = parsed.path.strip("/").split("/")[-1]
        if path_title:
            return truncate(path_title.replace("-", " ").replace("_", " "), 180)

    return source_file.stem.replace("_rag_metadata", "").replace("_", " ")


def build_page_context(metadata: dict[str, Any], category: str, page_title: str) -> str:
    lines = [f"AREA: {category}"]
    if page_title:
        lines.append(f"TITOLO: {page_title}")

    seen = {"title", "titolo"}
    for key in CONTEXT_METADATA_KEYS:
        if key in seen:
            continue
        value = metadata_to_text(metadata.get(key))
        if value:
            lines.append(f"{key.upper()}: {truncate(value, 700)}")
            seen.add(key)

    return "\n".join(lines)


def build_question_text(metadata: dict[str, Any], page_context: str, text: str) -> str:
    questions = safe_list_of_strings(metadata.get("questions"), limit=20)
    keywords = safe_list_of_strings(metadata.get("keywords"), limit=30)
    lines = []
    if questions:
        lines.append("DOMANDE METADATA:")
        lines.extend(f"- {question}" for question in questions)
    if keywords:
        lines.append("PAROLE CHIAVE:")
        lines.append(", ".join(keywords))
    section = metadata_to_text(metadata.get("section"))
    subsection = metadata_to_text(metadata.get("subsection"))
    if section or subsection:
        lines.append(f"SEZIONE: {normalize_space(section + ' ' + subsection)}")
    if not lines:
        lines.append(truncate(page_context or text, 1200))
    return "\n".join(lines)


def build_retrieval_text(page_context: str, question_text: str, text: str) -> str:
    parts = [
        "CONTESTO METADATI:",
        page_context,
        "",
        "QUERY ASSOCIATE AI METADATI:",
        question_text,
        "",
        "TESTO CHUNK:",
        text,
    ]
    return "\n".join(part for part in parts if part is not None)


def iter_chunk_objects(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(data, dict):
        return

    page_content = data.get("page_content")
    metadata = data.get("metadata")
    if isinstance(page_content, list):
        metadata_items = metadata if isinstance(metadata, list) else None
        for index, text in enumerate(page_content):
            item_metadata = metadata_items[index] if metadata_items and index < len(metadata_items) else metadata
            yield {"page_content": text, "metadata": item_metadata or {}}
    else:
        yield data


def build_leaf_record(item: dict[str, Any], source_file: Path, relative_path: Path, source_index: int) -> dict[str, Any] | None:
    text = item.get("page_content")
    if text is None:
        text = item.get("text")
    text = str(text or "").strip()
    if not text:
        return None

    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    category = category_from_path(relative_path)
    url = metadata_to_text(metadata.get("source"))
    page_title = infer_title(metadata, relative_path)
    source_type = metadata_to_text(metadata.get("doc_type")) or "html"
    domain = urlparse(url).netloc.lower() if url else ""
    section = metadata_to_text(metadata.get("section"))
    subsection = metadata_to_text(metadata.get("subsection"))
    dedupe_hash = text_hash(text)
    page_seed = url or relative_path.as_posix()
    page_id = f"page:{stable_hash(page_seed)[:16]}"
    chunk_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative_path.with_suffix("").as_posix()).strip("_")
    chunk_id = f"chunks_pro:{chunk_slug}:{source_index:05d}:{dedupe_hash[:12]}"

    page_context = build_page_context(metadata, category, page_title)
    question_text = build_question_text(metadata, page_context, text)
    retrieval = build_retrieval_text(page_context, question_text, text)

    return {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "url": url,
        "page_title": page_title,
        "text": text,
        "page_context": page_context,
        "retrieval_text": retrieval,
        "question_text": question_text,
        "source_type": source_type,
        "category": category,
        "domain": domain,
        "section": section,
        "subsection": subsection,
        "keywords": safe_list_of_strings(metadata.get("keywords"), limit=50),
        "questions": safe_list_of_strings(metadata.get("questions"), limit=30),
        "node_type": "leaf",
        "raptor_level": 0,
        "children_ids": [],
        "descendant_leaf_ids": [],
        "source_urls": [url] if url else [],
        "source_file": relative_path.as_posix(),
        "source_index": source_index,
        "semantic_dedupe_hash": dedupe_hash,
        "duplicate_count": 0,
        "duplicate_sources": [],
        "metadata": metadata,
    }


def load_chunks_pro(chunks_dir: Path) -> tuple[list[dict[str, Any]], int, int, int]:
    records: list[dict[str, Any]] = []
    input_files = 0
    input_chunks = 0
    replacement_char_chunks = 0

    for path in sorted(chunks_dir.rglob("*.json")):
        if "metadata_manifest" in path.parts or "metadata_manifest_ridotto" in path.parts:
            continue
        input_files += 1
        relative_path = path.relative_to(chunks_dir)
        data = json.loads(path.read_text(encoding="utf-8"))
        for source_index, item in enumerate(iter_chunk_objects(data)):
            input_chunks += 1
            record = build_leaf_record(item, path, relative_path, source_index)
            if record is None:
                continue
            if "\ufffd" in record["text"]:
                replacement_char_chunks += 1
            records.append(record)

    return records, input_files, input_chunks, replacement_char_chunks


def deduplicate_records(records: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], int]:
    if mode == "none":
        return records, 0
    if mode not in {"text", "text_url"}:
        raise ValueError(f"Modalita dedupe non supportata: {mode}")

    unique: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicates = 0

    for record in records:
        if mode == "text":
            key = record["semantic_dedupe_hash"]
        else:
            key = f"{record['semantic_dedupe_hash']}|{record.get('url') or record.get('source_file')}"

        existing = seen.get(key)
        if existing is None:
            seen[key] = record
            unique.append(record)
            continue

        duplicates += 1
        existing["duplicate_count"] = int(existing.get("duplicate_count") or 0) + 1
        duplicate_source = {
            "url": record.get("url"),
            "source_file": record.get("source_file"),
            "source_index": record.get("source_index"),
        }
        duplicate_sources = existing.setdefault("duplicate_sources", [])
        if len(duplicate_sources) < 20:
            duplicate_sources.append(duplicate_source)

    return unique, duplicates


def unique_preserve_order(values: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def split_batches(items: list[dict[str, Any]], max_children: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), max_children):
        yield items[start : start + max_children]


def extract_sentences(text: str, max_sentences: int = 3) -> list[str]:
    compact = normalize_space(text)
    if not compact:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", compact)
    sentences = []
    for piece in pieces:
        piece = truncate(piece, 450)
        if len(piece) >= 40 and piece not in sentences:
            sentences.append(piece)
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        sentences.append(truncate(compact, 450))
    return sentences


def summarize_children(children: list[dict[str, Any]], level: int) -> dict[str, Any]:
    categories = unique_preserve_order(child.get("category", "") for child in children)
    sections = unique_preserve_order((child.get("section", "") for child in children if child.get("section")), limit=6)
    urls = unique_preserve_order(
        (url for child in children for url in as_list(child.get("source_urls") or child.get("url")) if url),
        limit=30,
    )
    child_ids = [child["chunk_id"] for child in children]
    descendant_ids = unique_preserve_order(
        (
            descendant
            for child in children
            for descendant in (child.get("descendant_leaf_ids") or [child["chunk_id"]])
        )
    )

    keyword_counter: Counter[str] = Counter()
    question_list: list[str] = []
    for child in children:
        keyword_counter.update(safe_list_of_strings(child.get("keywords"), limit=30))
        for question in safe_list_of_strings(child.get("questions"), limit=20):
            if question not in question_list:
                question_list.append(question)

    top_keywords = [keyword for keyword, _count in keyword_counter.most_common(24)]
    snippets: list[str] = []
    for child in children:
        for sentence in extract_sentences(child.get("text", ""), max_sentences=1):
            if sentence not in snippets:
                snippets.append(sentence)
        if len(snippets) >= 8:
            break

    category_text = ", ".join(categories) if categories else "diem"
    section_text = "; ".join(sections)
    title = f"RAPTOR L{level} - {category_text}"
    if section_text:
        title = f"{title} - {truncate(section_text, 90)}"

    lines = [
        title,
        f"Numero frammenti coperti: {len(descendant_ids)}",
    ]
    if urls:
        lines.append("Fonti principali: " + "; ".join(urls[:5]))
    if section_text:
        lines.append("Sezioni: " + section_text)
    if top_keywords:
        lines.append("Parole chiave: " + ", ".join(top_keywords))
    if question_list:
        lines.append("Domande rappresentative:")
        lines.extend(f"- {question}" for question in question_list[:8])
    if snippets:
        lines.append("Estratti rappresentativi:")
        lines.extend(f"- {snippet}" for snippet in snippets)

    summary_text = "\n".join(lines)
    node_seed = "|".join(child_ids)
    chunk_id = f"raptor:l{level}:{stable_hash(node_seed)[:18]}"
    page_id = f"raptor_page:{stable_hash(category_text + section_text)[:16]}"
    category = categories[0] if len(categories) == 1 else "mixed"
    url = urls[0] if len(urls) == 1 else f"raptor://level-{level}/{stable_hash(node_seed)[:16]}"
    context = "\n".join(
        [
            f"AREA: {category_text}",
            f"RAPTOR_LEVEL: {level}",
            f"SEZIONI: {section_text}" if section_text else "",
            "PAROLE_CHIAVE: " + ", ".join(top_keywords) if top_keywords else "",
        ]
    ).strip()
    question_text = "\n".join(question_list[:12]) if question_list else " ".join(top_keywords)

    return {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "url": url,
        "page_title": title,
        "text": summary_text,
        "page_context": context,
        "retrieval_text": build_retrieval_text(context, question_text, summary_text),
        "question_text": question_text or summary_text,
        "source_type": "raptor_summary",
        "category": category,
        "domain": "raptor",
        "section": section_text,
        "subsection": "",
        "keywords": top_keywords,
        "questions": question_list[:20],
        "node_type": "raptor",
        "raptor_level": level,
        "children_ids": child_ids,
        "descendant_leaf_ids": descendant_ids,
        "source_urls": urls,
        "source_file": "",
        "source_index": -1,
        "semantic_dedupe_hash": text_hash(summary_text),
        "duplicate_count": 0,
        "duplicate_sources": [],
        "metadata": {},
    }


def build_raptor_nodes(
    leaf_records: list[dict[str, Any]],
    max_levels: int = 2,
    max_children: int = 8,
    min_children: int = 3,
) -> list[dict[str, Any]]:
    """Builds RAPTOR-style summary nodes from the already curated chunks_pro records."""
    all_nodes: list[dict[str, Any]] = []

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in leaf_records:
        groups[(record.get("category", ""), record.get("url", "") or record.get("page_id", ""), record.get("section", ""))].append(record)

    current_level_nodes: list[dict[str, Any]] = []
    for group_records in groups.values():
        if len(group_records) < min_children:
            continue
        group_records = sorted(group_records, key=lambda item: (item.get("source_file", ""), int(item.get("source_index") or 0)))
        for batch in split_batches(group_records, max_children):
            if len(batch) >= min_children:
                current_level_nodes.append(summarize_children(batch, level=1))

    all_nodes.extend(current_level_nodes)

    for level in range(2, max_levels + 1):
        if not current_level_nodes:
            break
        higher_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in current_level_nodes:
            higher_groups[node.get("category") or "mixed"].append(node)

        next_level_nodes: list[dict[str, Any]] = []
        for group_records in higher_groups.values():
            if len(group_records) < min_children:
                continue
            for batch in split_batches(group_records, max_children):
                if len(batch) >= min_children:
                    next_level_nodes.append(summarize_children(batch, level=level))

        all_nodes.extend(next_level_nodes)
        current_level_nodes = next_level_nodes

    return all_nodes


def payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = {key: record.get(key) for key in PAYLOAD_KEYS if key in record}
    metadata = record.get("metadata") or {}
    if metadata:
        payload["metadata_fields"] = {
            key: metadata.get(key)
            for key in CONTEXT_METADATA_KEYS
            if key in metadata and metadata.get(key) is not None
        }
    return payload


def ensure_fastembed_available() -> None:
    try:
        import fastembed  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "fastembed non e installato. Serve per creare i vettori sparse/BM25 di Qdrant. "
            "Installa le dipendenze aggiornate oppure esegui: pip install fastembed"
        ) from exc


class ChunksProQdrantBuilder:
    """Builds a Qdrant store directly from chunks_pro JSON files."""

    def __init__(
        self,
        chunks_pro_dir: Path,
        output_dir: Path,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        upsert_batch_size: int = 256,
        max_length: int = 8192,
        dedupe_mode: str = "text_url",
        use_raptor: bool = True,
        raptor_max_levels: int = 2,
        raptor_max_children: int = 8,
        raptor_min_children: int = 3,
    ) -> None:
        self.chunks_pro_dir = chunks_pro_dir
        self.output_dir = output_dir
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.device = device
        self.batch_size = batch_size
        self.upsert_batch_size = upsert_batch_size
        self.max_length = max_length
        self.dedupe_mode = dedupe_mode
        self.use_raptor = use_raptor
        self.raptor_max_levels = raptor_max_levels
        self.raptor_max_children = raptor_max_children
        self.raptor_min_children = raptor_min_children

    def build(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_fastembed_available()

        records, input_files, input_chunks, replacement_char_chunks = load_chunks_pro(self.chunks_pro_dir)
        if not records:
            raise RuntimeError(f"Nessun chunk valido trovato in {self.chunks_pro_dir}")

        records, duplicates = deduplicate_records(records, self.dedupe_mode)
        raptor_nodes = (
            build_raptor_nodes(
                records,
                max_levels=self.raptor_max_levels,
                max_children=self.raptor_max_children,
                min_children=self.raptor_min_children,
            )
            if self.use_raptor
            else []
        )
        all_records = records + raptor_nodes

        print(f"File chunks_pro letti: {input_files}")
        print(f"Chunk input: {input_chunks}")
        print(f"Leaf chunk indicizzabili: {len(records)}")
        print(f"Duplicati saltati ({self.dedupe_mode}): {duplicates}")
        print(f"Nodi RAPTOR: {len(raptor_nodes)}")
        if replacement_char_chunks:
            print(f"Warning: {replacement_char_chunks} chunk contengono il carattere replacement U+FFFD")

        embedder = JinaEmbedder(self.dense_model, device=self.device, batch_size=self.batch_size, max_length=self.max_length)

        dense_texts = [record.get("text") or "" for record in all_records]
        context_texts = [record.get("retrieval_text") or record.get("text") or "" for record in all_records]
        question_texts = [record.get("question_text") or record.get("page_context") or record.get("text") or "" for record in all_records]
        sparse_texts = [f"{question_texts[index]}\n{context_texts[index]}" for index in range(len(all_records))]

        print("Embedding testo chunk...")
        dense_vectors = embedder.embed_passages(dense_texts).astype(np.float32)
        print("Embedding testo arricchito con metadati...")
        context_vectors = embedder.embed_passages(context_texts).astype(np.float32)
        print("Embedding domande/keyword metadata...")
        question_vectors = embedder.embed_passages(question_texts).astype(np.float32)

        for name, vectors in [
            ("dense", dense_vectors),
            ("context_dense", context_vectors),
            ("question_dense", question_vectors),
        ]:
            if not np.isfinite(vectors).all():
                bad_indexes = np.where(~np.isfinite(vectors).all(axis=1))[0].tolist()
                raise RuntimeError(f"Embedding {name} non finiti agli indici: {bad_indexes[:20]}")

        client = QdrantClient(url="http://localhost:6333")
        vector_size = int(dense_vectors.shape[1])
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                CONTEXT_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                QUESTION_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )

        for start in tqdm(range(0, len(all_records), self.upsert_batch_size), desc="Indicizzazione Qdrant", unit="batch"):
            end = min(start + self.upsert_batch_size, len(all_records))
            points = []
            for index in range(start, end):
                record = all_records[index]
                points.append(
                    models.PointStruct(
                        id=point_id(record["chunk_id"]),
                        vector={
                            DENSE_VECTOR_NAME: dense_vectors[index].tolist(),
                            CONTEXT_VECTOR_NAME: context_vectors[index].tolist(),
                            QUESTION_VECTOR_NAME: question_vectors[index].tolist(),
                            SPARSE_VECTOR_NAME: models.Document(text=sparse_texts[index], model=self.sparse_model),
                        },
                        payload=payload_from_record(record),
                    )
                )
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        stats = BuildStats(
            input_files=input_files,
            input_chunks=input_chunks,
            leaf_chunks=len(records),
            duplicates_skipped=duplicates,
            raptor_nodes=len(raptor_nodes),
            replacement_char_chunks=replacement_char_chunks,
        )
        manifest = {
            "collection_name": COLLECTION_NAME,
            "qdrant_path": "qdrant",
            "builder": "chunks_pro_qdrant_vector_store",
            "chunks_pro_dir": str(self.chunks_pro_dir),
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "dense_vector_name": DENSE_VECTOR_NAME,
            "context_vector_name": CONTEXT_VECTOR_NAME,
            "question_vector_name": QUESTION_VECTOR_NAME,
            "sparse_vector_name": SPARSE_VECTOR_NAME,
            "vector_size": vector_size,
            "dedupe_mode": self.dedupe_mode,
            "raptor_enabled": self.use_raptor,
            "raptor_max_levels": self.raptor_max_levels,
            "raptor_max_children": self.raptor_max_children,
            "raptor_min_children": self.raptor_min_children,
            "input_files": stats.input_files,
            "input_chunks": stats.input_chunks,
            "leaf_chunks": stats.leaf_chunks,
            "raptor_nodes": stats.raptor_nodes,
            "indexed_points": len(all_records),
            "duplicates_skipped": stats.duplicates_skipped,
            "replacement_char_chunks": stats.replacement_char_chunks,
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.output_dir / "build_report.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        client.close()

        print(f"OK Qdrant store creato in: {self.output_dir}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Punti indicizzati: {len(all_records)}")
        print(f"Dense model: {self.dense_model}")
        print(f"Sparse model: {self.sparse_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea un Qdrant store direttamente da chunks_pro con metadata matching e RAPTOR.")
    default_chunks_dir = Path("ChatBot_DIEM_Group_5/chunks_pro") 
    default_output_dir = Path("ChatBot_DIEM_Group_5/src/VectorDB/qdrant_chunks_pro_raptor_jina")
    parser.add_argument("--chunks-pro-dir", type=Path, default=default_chunks_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--device", default=None, help="Esempio: cuda oppure cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--dedupe-mode", choices=["none", "text", "text_url"], default="text_url")
    parser.add_argument("--no-raptor", action="store_true", help="Disattiva la creazione dei nodi RAPTOR.")
    parser.add_argument("--raptor-max-levels", type=int, default=2)
    parser.add_argument("--raptor-max-children", type=int, default=8)
    parser.add_argument("--raptor-min-children", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096, help="Numero massimo di token per chunk (es. 1024, 2048, 8192)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = ChunksProQdrantBuilder(
        chunks_pro_dir=args.chunks_pro_dir,
        output_dir=args.output_dir,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        device=args.device,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
        dedupe_mode=args.dedupe_mode,
        use_raptor=not args.no_raptor,
        raptor_max_levels=args.raptor_max_levels,
        raptor_max_children=args.raptor_max_children,
        raptor_min_children=args.raptor_min_children,
        max_length=args.max_length
    )
    builder.build()


if __name__ == "__main__":
    main()
