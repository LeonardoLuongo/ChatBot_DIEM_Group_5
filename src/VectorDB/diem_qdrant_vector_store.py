from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


COLLECTION_NAME = "diem_chunks"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
CONTEXT_VECTOR_NAME = "context_dense"  # NUOVO: vettore per page_context
RRF_K = 60


@dataclass(slots=True)
class SearchResult:
    """Risultato del retrieval ibrido pronto per il reranker."""

    score: float
    dense_score: float | None
    sparse_score: float | None
    context_score: float | None  # NUOVO: score dalla ricerca contestuale
    text: str
    url: str
    page_title: str
    page_context: str  # NUOVO: contesto per debugging
    payload: dict[str, Any]


class E5Embedder:
    """Wrapper minimo per multilingual-e5: passage per documenti, query per domande."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL, device: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Crea embedding dense normalizzati dei chunk."""
        prefixed = [text if text.lower().startswith("passage:") else f"passage: {text}" for text in texts]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Crea embedding dense normalizzato della query."""
        vector = self.model.encode(
            [f"query: {query}"],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)


def point_id(chunk_id: str) -> str:
    """Trasforma un chunk_id in UUID stabile compatibile con Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_semantic_chunks(path: Path) -> list[dict[str, Any]]:
    """Carica semantic_chunks.jsonl in memoria."""
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def text_hash(text: str) -> str:
    """Hash fallback stabile quando il chunk non ha già un hash semantico."""
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Rimuove chunk duplicati prima di calcolare embedding costosi."""
    seen_hashes: set[str] = set()
    unique_chunks: list[dict[str, Any]] = []
    duplicates = 0

    for chunk in chunks:
        dedupe_hash = (
            chunk.get("semantic_dedupe_hash")
            or chunk.get("hash_code")
            or text_hash(chunk.get("retrieval_text") or chunk.get("text") or "")
        )
        if dedupe_hash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(dedupe_hash)
        unique_chunks.append(chunk)

    return unique_chunks, duplicates


def payload_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Tiene nel payload solo campi utili a risposta, debug e fonti."""
    keys = [
        "chunk_id",
        "page_id",
        "url",
        "page_title",
        "text",
        "page_context",  
        "source_type",
        "pdf_page",
        "parent_url",
        "references",
        "page_type",
        "keywords",
        "academic_years",
        "years",
        "people",
        "course_names",
        "domain",
    ]
    return {key: chunk.get(key) for key in keys if key in chunk}


def retrieval_text(chunk: dict[str, Any]) -> str:
    """
    Crea testo composito per dense embedding.
    NUOVO: Combina chunk text + page_context per contesto semantico.
    """
    text = chunk.get("retrieval_text") or chunk.get("text") or ""
    context = chunk.get("page_context") or ""
    
    if context:
        return f"{context}\n\n{text}"
    return text


class SimpleQdrantBuilder:
    """Costruisce collection Qdrant con vettore dense E5, sparse BM25 e context dense."""

    def __init__(
        self,
        semantic_chunks: Path,
        output_dir: Path,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        upsert_batch_size: int = 256,
    ) -> None:
        self.semantic_chunks = semantic_chunks
        self.output_dir = output_dir
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.device = device
        self.batch_size = batch_size
        self.upsert_batch_size = upsert_batch_size

    def build(self) -> None:
        """Calcola gli embedding dense (chunk + context) e sparse BM25."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chunks = load_semantic_chunks(self.semantic_chunks)
        if not chunks:
            raise RuntimeError("Nessun chunk trovato nel file semantic_chunks.jsonl")
        chunks, duplicates = deduplicate_chunks(chunks)
        print(f"Deduplica vector store: {len(chunks)} unici, {duplicates} duplicati rimossi")

        # Creiamo due testi: uno per dense (chunk semplice) e uno per context_dense
        texts_chunk = [chunk.get("retrieval_text") or chunk.get("text") or "" for chunk in chunks]
        texts_context = [retrieval_text(chunk) for chunk in chunks]
        
        embedder = E5Embedder(self.dense_model, device=self.device, batch_size=self.batch_size)
        
        print("Embedding chunk semplici...")
        dense_vectors = embedder.embed_passages(texts_chunk).astype(np.float32)
        
        print("Embedding con page_context...")
        context_vectors = embedder.embed_passages(texts_context).astype(np.float32)

        if not np.isfinite(dense_vectors).all():
            bad_indexes = np.where(~np.isfinite(dense_vectors).all(axis=1))[0].tolist()
            raise RuntimeError(f"Embedding dense non finiti trovati agli indici: {bad_indexes[:20]}")

        if not np.isfinite(context_vectors).all():
            bad_indexes = np.where(~np.isfinite(context_vectors).all(axis=1))[0].tolist()
            raise RuntimeError(f"Embedding context non finiti trovati agli indici: {bad_indexes[:20]}")

        client = QdrantClient(path=str(self.output_dir / "qdrant"))
        vector_size = int(dense_vectors.shape[1])
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                CONTEXT_VECTOR_NAME: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),  # NUOVO
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )

        for start in tqdm(range(0, len(chunks), self.upsert_batch_size), desc="Indicizzazione Qdrant", unit="batch"):
            end = min(start + self.upsert_batch_size, len(chunks))
            points = [
                models.PointStruct(
                    id=point_id(chunks[i]["chunk_id"]),
                    vector={
                        DENSE_VECTOR_NAME: dense_vectors[i].tolist(),
                        CONTEXT_VECTOR_NAME: context_vectors[i].tolist(),  # NUOVO
                        SPARSE_VECTOR_NAME: models.Document(text=texts_chunk[i], model=self.sparse_model),
                    },
                    payload=payload_from_chunk(chunks[i]),
                )
                for i in range(start, end)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        manifest = {
            "collection_name": COLLECTION_NAME,
            "qdrant_path": "qdrant",
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "dense_vector_name": DENSE_VECTOR_NAME,
            "context_vector_name": CONTEXT_VECTOR_NAME,  
            "sparse_vector_name": SPARSE_VECTOR_NAME,
            "vector_size": vector_size,
            "chunks": len(chunks),
            "semantic_chunks": str(self.semantic_chunks),
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        client.close()

        print(f"✓ Qdrant store ibrido creato in: {self.output_dir}")
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Chunk indicizzati: {len(chunks)}")
        print(f"  Dense: {self.dense_model}")
        print(f"  Context Dense: {self.dense_model} (con page_context)")
        print(f"  Sparse: {self.sparse_model}")


class SimpleQdrantStore:
    """Retriever ibrido: dense + context_dense + sparse BM25 + RRF pesato."""

    def __init__(self, store_dir: Path, device: str | None = None, batch_size: int = 32) -> None:
        manifest_path = store_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.collection_name = self.manifest["collection_name"]
        self.dense_vector_name = self.manifest.get("dense_vector_name", DENSE_VECTOR_NAME)
        self.context_vector_name = self.manifest.get("context_vector_name", CONTEXT_VECTOR_NAME)  # NUOVO
        self.sparse_vector_name = self.manifest.get("sparse_vector_name", SPARSE_VECTOR_NAME)
        self.sparse_model = self.manifest.get("sparse_model", DEFAULT_SPARSE_MODEL)
        self.client = QdrantClient(path=str(store_dir / self.manifest["qdrant_path"]))
        self.embedder = E5Embedder(self.manifest["dense_model"], device=device, batch_size=batch_size)

    def search(self, query: str, limit: int = 20, boost_keyword: bool = False) -> list[SearchResult]:
        """
        Cerca candidati con dense, context_dense e BM25, poi li fonde con RRF pesato.
        
        Args:
            query: Query testuale
            limit: Numero massimo di risultati
            boost_keyword: Se True, aumenta peso sparse search (utile per query con nomi/codici)
        """
        dense_points = self._dense_search(query, limit)
        context_points = self._context_search(query, limit)  
        sparse_points = self._sparse_search(query, limit)
        return self._rrf_fusion(dense_points, context_points, sparse_points, limit, boost_keyword)

    def _dense_search(self, query: str, limit: int) -> list[Any]:
        """Ricerca semantica con multilingual-e5 su chunk puro."""
        query_vector = self.embedder.embed_query(query).tolist()
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.dense_vector_name,
                limit=limit,
                with_payload=True,
            )
            return list(response.points)
        except AttributeError:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(self.dense_vector_name, query_vector),
                    limit=limit,
                    with_payload=True,
                )
            )

    def _context_search(self, query: str, limit: int) -> list[Any]:
        """NUOVO: Ricerca semantica con context (chunk + page_context)."""
        query_vector = self.embedder.embed_query(query).tolist()
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.context_vector_name,
                limit=limit,
                with_payload=True,
            )
            return list(response.points)
        except AttributeError:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(self.context_vector_name, query_vector),
                    limit=limit,
                    with_payload=True,
                )
            )

    def _sparse_search(self, query: str, limit: int) -> list[Any]:
        """Ricerca lessicale BM25 per parole esatte, nomi, sigle, anni e codici."""
        sparse_query = models.Document(text=query, model=self.sparse_model)
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using=self.sparse_vector_name,
                limit=limit,
                with_payload=True,
            )
            return list(response.points)
        except AttributeError:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(self.sparse_vector_name, sparse_query),
                    limit=limit,
                    with_payload=True,
                )
            )

    def _rrf_fusion(
        self,
        dense_points: list[Any],
        context_points: list[Any],
        sparse_points: list[Any],
        limit: int,
        boost_keyword: bool = False,
    ) -> list[SearchResult]:
        """
        Unisce le classifiche con RRF pesato intelligente.
        NUOVO: Context search boost se query semantica, sparse boost se keyword-heavy.
        """
        fused: dict[str, dict[str, Any]] = {}

        def add_points(points: list[Any], channel: str, weight: float = 1.0) -> None:
            """Aggiunge punti con peso personalizzato."""
            for rank, point in enumerate(points, 1):
                payload = dict(point.payload or {})
                chunk_id = payload.get("chunk_id") or str(point.id)
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "payload": payload,
                        "rrf_score": 0.0,
                        "dense_score": None,
                        "context_score": None,  
                        "sparse_score": None,
                    },
                )
                # RRF con peso: weight * (1 / (K + rank))
                rrf_component = weight / (RRF_K + rank)
                entry["rrf_score"] += rrf_component
                
                raw_score = float(getattr(point, "score", 0.0) or 0.0)
                if math.isfinite(raw_score):
                    entry[f"{channel}_score"] = raw_score

        # Pesi adattivi
        weight_sparse = 2.0 if boost_keyword else 1.0
        add_points(dense_points, "dense", weight=1.0)
        add_points(context_points, "context", weight=0.8)  
        add_points(sparse_points, "sparse", weight=weight_sparse)

        ordered = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        results: list[SearchResult] = []
        for item in ordered[:limit]:
            payload = item["payload"]
            score = float(item["rrf_score"])
            if not math.isfinite(score):
                continue
            results.append(
                SearchResult(
                    score=score,
                    dense_score=item["dense_score"],
                    context_score=item["context_score"], 
                    sparse_score=item["sparse_score"],
                    text=payload.get("text") or "",
                    url=payload.get("url") or "",
                    page_title=payload.get("page_title") or "",
                    page_context=payload.get("page_context") or "",
                    payload=payload,
                )
            )
        return results

    def close(self) -> None:
        """Chiude il client Qdrant."""
        self.client.close()


def detect_keyword_query(query: str) -> bool:
    """
    Detecta se una query è keyword-heavy (nomi, sigle, codici, anni).
    Usato per boost sparse search automatico.
    """
    # Pattern: anno (2020-2024), codici (MAT101), sigle (CFU, SSD)
    year_pattern = r"\b(20\d{2}|19\d{2})\b"
    code_pattern = r"\b[A-Z]{2,}[0-9]{2,}\b|\b[A-Z]{3,}\b"
    
    has_year = bool(re.search(year_pattern, query))
    has_code = bool(re.search(code_pattern, query))
    has_email = "@" in query
    
    # Query corta con molte maiuscole = keyword-heavy
    short_with_caps = len(query.split()) <= 4 and sum(1 for c in query if c.isupper()) > 3
    
    return has_year or has_code or has_email or short_with_caps


def parse_args() -> argparse.Namespace:
    """Parametri CLI per creare il vector store."""
    parser = argparse.ArgumentParser(description="Crea un Qdrant vector store ibrido con page_context e RRF pesato per DIEM.")
    parser.add_argument("--semantic-chunks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--device", default=None, help="Esempio: cuda oppure cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    builder = SimpleQdrantBuilder(
        semantic_chunks=args.semantic_chunks,
        output_dir=args.output_dir,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        device=args.device,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
    )
    builder.build()


if __name__ == "__main__":
    main()