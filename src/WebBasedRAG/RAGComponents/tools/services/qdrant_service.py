import json
import math
import re
from pathlib import Path
from typing import Any, List
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "diem_chunks"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
CONTEXT_VECTOR_NAME = "context_dense"
RRF_K = 60

@dataclass(slots=True)
class SearchResult:
    score: float
    text: str
    url: str
    page_title: str
    payload: dict[str, Any]

class E5Embedder:
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL, device: str = None) -> None:
        print(f"   -> Caricamento E5 Embedder ({model_name}) per Qdrant DB...")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [f"query: {query}"],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)

def detect_keyword_query(query: str) -> bool:
    year_pattern = r"\b(20\d{2}|19\d{2})\b"
    code_pattern = r"\b[A-Z]{2,}[0-9]{2,}\b|\b[A-Z]{3,}\b"
    return bool(re.search(year_pattern, query)) or bool(re.search(code_pattern, query)) or "@" in query

class QdrantService:
    def __init__(self, store_dir_path: str, device: str = None):
        store_dir = Path(store_dir_path)
        manifest_path = store_dir / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest non trovato in: {manifest_path}")
            
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.collection_name = self.manifest["collection_name"]
        
        qdrant_db_path = str(store_dir / self.manifest["qdrant_path"])
        print(f"   -> Connessione a Qdrant Local DB: {qdrant_db_path}")
        self.client = QdrantClient(path=qdrant_db_path)
        
        self.embedder = E5Embedder(self.manifest["dense_model"], device=device)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        boost_keyword = detect_keyword_query(query)
        # Cerchiamo il doppio dei candidati in fase di recupero per la fusione RRF
        fetch_limit = limit * 2 
        
        dense_points = self._dense_search(query, fetch_limit)
        context_points = self._context_search(query, fetch_limit)
        sparse_points = self._sparse_search(query, fetch_limit)
        
        return self._rrf_fusion(dense_points, context_points, sparse_points, limit, boost_keyword)

    def _dense_search(self, query: str, limit: int) -> list:
        vector = self.embedder.embed_query(query).tolist()
        try:
            return list(self.client.query_points(collection_name=self.collection_name, query=vector, using=DENSE_VECTOR_NAME, limit=limit, with_payload=True).points)
        except AttributeError:
            return list(self.client.search(collection_name=self.collection_name, query_vector=(DENSE_VECTOR_NAME, vector), limit=limit, with_payload=True))

    def _context_search(self, query: str, limit: int) -> list:
        vector = self.embedder.embed_query(query).tolist()
        try:
            return list(self.client.query_points(collection_name=self.collection_name, query=vector, using=CONTEXT_VECTOR_NAME, limit=limit, with_payload=True).points)
        except AttributeError:
            return list(self.client.search(collection_name=self.collection_name, query_vector=(CONTEXT_VECTOR_NAME, vector), limit=limit, with_payload=True))

    def _sparse_search(self, query: str, limit: int) -> list:
        sparse_query = models.Document(text=query, model=DEFAULT_SPARSE_MODEL)
        try:
            return list(self.client.query_points(collection_name=self.collection_name, query=sparse_query, using=SPARSE_VECTOR_NAME, limit=limit, with_payload=True).points)
        except AttributeError:
            return list(self.client.search(collection_name=self.collection_name, query_vector=(SPARSE_VECTOR_NAME, sparse_query), limit=limit, with_payload=True))

    def _rrf_fusion(self, dense, context, sparse, limit: int, boost_keyword: bool) -> List[SearchResult]:
        fused = {}

        def add_points(points, weight=1.0):
            for rank, point in enumerate(points, 1):
                payload = dict(point.payload or {})
                chunk_id = payload.get("chunk_id") or str(point.id)
                entry = fused.setdefault(chunk_id, {"payload": payload, "score": 0.0})
                entry["score"] += weight / (RRF_K + rank)

        add_points(dense, weight=1.0)
        add_points(context, weight=0.8)
        add_points(sparse, weight=2.0 if boost_keyword else 1.0)

        ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        results = []
        for item in ordered[:limit]:
            p = item["payload"]
            results.append(SearchResult(
                score=item["score"], text=p.get("text", ""), url=p.get("url", ""), 
                page_title=p.get("page_title", ""), payload=p
            ))
        return results