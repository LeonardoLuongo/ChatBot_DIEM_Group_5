import json
import math
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "diem_chunks"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
CONTEXT_VECTOR_NAME = "context_dense"
QUESTION_VECTOR_NAME = "question_dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 60


@dataclass(slots=True)
class SearchResult:
    score: float
    text: str
    url: str
    page_title: str
    payload: dict[str, Any]
    page_context: str = ""


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


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def detect_keyword_query(query: str) -> bool:
    year_pattern = r"\b(20\d{2}|19\d{2})\b"
    code_pattern = r"\b[A-Z]{2,}[0-9]{2,}\b|\b[A-Z]{3,}\b"
    return bool(re.search(year_pattern, query)) or bool(re.search(code_pattern, query)) or "@" in query


def detect_category_filters(query: str) -> list[str]:
    """Soft domain detector used for metadata filtering before vector matching."""
    q = fold_text(query)
    rules = [
        (
            "docenti",
            [
                "docente",
                "docenti",
                "professore",
                "professoressa",
                "prof ",
                "ricevimento",
                "orario di ricevimento",
                "curriculum docente",
            ],
        ),
        (
            "bandi",
            ["bando", "bandi", "borsa", "borse", "scadenza", "graduatoria", "selezione pubblica"],
        ),
        (
            "international",
            ["erasmus", "mobilita", "international", "internazionale", "traineeship", "doppio titolo", "accordi"],
        ),
        (
            "offerta_formativa",
            [
                "offerta formativa",
                "laurea",
                "magistrale",
                "triennale",
                "dottorato",
                "corso di studio",
                "cfu",
                "piano di studi",
                "insegnamento",
                "syllabus",
            ],
        ),
        (
            "ricerca",
            ["ricerca", "progetto", "progetti", "premio", "premi", "iris", "area di ricerca", "focus di ricerca"],
        ),
        (
            "dipartimento",
            ["dipartimento", "laboratorio", "laboratori", "aula", "aule", "struttura", "commissione", "delegato"],
        ),
        (
            "contatti",
            ["contatto", "contatti", "email", "mail", "telefono", "ufficio", "segreteria", "indirizzo"],
        ),
        (
            "terza_missione",
            ["terza missione", "brevetto", "brevetti", "spin off", "spin-off", "conto terzi"],
        ),
        (
            "consiglio_didattico",
            ["consiglio didattico", "cpds", "commissione paritetica", "studenti docenti"],
        ),
    ]

    categories: list[str] = []
    for category, terms in rules:
        if any(term in q for term in terms):
            categories.append(category)
    return categories[:3]


def build_category_filter(categories: list[str]) -> models.Filter | None:
    if not categories:
        return None
    return models.Filter(
        should=[
            models.FieldCondition(
                key="category",
                match=models.MatchAny(any=categories),
            )
        ]
    )


class QdrantService:
    def __init__(self, store_dir_path: str, device: str = None):
        store_dir = Path(store_dir_path)
        manifest_path = store_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest non trovato in: {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.collection_name = self.manifest["collection_name"]
        self.dense_vector_name = self.manifest.get("dense_vector_name", DENSE_VECTOR_NAME)
        self.context_vector_name = self.manifest.get("context_vector_name", CONTEXT_VECTOR_NAME)
        self.question_vector_name = self.manifest.get("question_vector_name")
        self.sparse_vector_name = self.manifest.get("sparse_vector_name", SPARSE_VECTOR_NAME)
        self.sparse_model = self.manifest.get("sparse_model", DEFAULT_SPARSE_MODEL)

        qdrant_db_path = str(store_dir / self.manifest["qdrant_path"])
        print(f"   -> Connessione a Qdrant Local DB: {qdrant_db_path}")
        self.client = QdrantClient(path=qdrant_db_path)

        self.embedder = E5Embedder(self.manifest["dense_model"], device=device)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        boost_keyword = detect_keyword_query(query)
        categories = detect_category_filters(query)
        query_filter = build_category_filter(categories)
        fetch_limit = max(limit * 3, 20)
        query_vector = self.embedder.embed_query(query).tolist()

        results = self._search_with_filter(query, query_vector, fetch_limit, limit, boost_keyword, categories, query_filter)

        if query_filter is not None and len(results) < min(limit, 5):
            # Category filters are useful, but they must never hide the answer.
            results = self._search_with_filter(query, query_vector, fetch_limit, limit, boost_keyword, [], None)

        return self._expand_raptor_results(results, limit)

    def _search_with_filter(
        self,
        query: str,
        query_vector: list[float],
        fetch_limit: int,
        limit: int,
        boost_keyword: bool,
        categories: list[str],
        query_filter: models.Filter | None,
    ) -> list[SearchResult]:
        dense_points = self._vector_search(query_vector, self.dense_vector_name, fetch_limit, query_filter)
        context_points = self._vector_search(query_vector, self.context_vector_name, fetch_limit, query_filter)
        question_points = (
            self._vector_search(query_vector, self.question_vector_name, fetch_limit, query_filter)
            if self.question_vector_name
            else []
        )
        sparse_points = self._sparse_search(query, fetch_limit, query_filter)
        return self._rrf_fusion(
            dense_points=dense_points,
            context_points=context_points,
            question_points=question_points,
            sparse_points=sparse_points,
            limit=limit,
            boost_keyword=boost_keyword,
            preferred_categories=categories,
        )

    def _vector_search(
        self,
        vector: list[float],
        vector_name: str,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[Any]:
        kwargs = {
            "collection_name": self.collection_name,
            "query": vector,
            "using": vector_name,
            "limit": limit,
            "with_payload": True,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        try:
            return list(self.client.query_points(**kwargs).points)
        except TypeError:
            kwargs.pop("query_filter", None)
            return list(self.client.query_points(**kwargs).points)
        except AttributeError:
            search_kwargs = {
                "collection_name": self.collection_name,
                "query_vector": (vector_name, vector),
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                search_kwargs["query_filter"] = query_filter
            try:
                return list(self.client.search(**search_kwargs))
            except TypeError:
                search_kwargs.pop("query_filter", None)
                return list(self.client.search(**search_kwargs))

    def _sparse_search(self, query: str, limit: int, query_filter: models.Filter | None = None) -> list[Any]:
        sparse_query = models.Document(text=query, model=self.sparse_model)
        kwargs = {
            "collection_name": self.collection_name,
            "query": sparse_query,
            "using": self.sparse_vector_name,
            "limit": limit,
            "with_payload": True,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        try:
            return list(self.client.query_points(**kwargs).points)
        except ImportError as exc:
            print(f"   [QDRANT WARN] Ricerca sparse/BM25 non disponibile: {exc}")
            return []
        except TypeError:
            kwargs.pop("query_filter", None)
            try:
                return list(self.client.query_points(**kwargs).points)
            except ImportError as exc:
                print(f"   [QDRANT WARN] Ricerca sparse/BM25 non disponibile: {exc}")
                return []
        except AttributeError:
            search_kwargs = {
                "collection_name": self.collection_name,
                "query_vector": (self.sparse_vector_name, sparse_query),
                "limit": limit,
                "with_payload": True,
            }
            if query_filter is not None:
                search_kwargs["query_filter"] = query_filter
            try:
                return list(self.client.search(**search_kwargs))
            except ImportError as exc:
                print(f"   [QDRANT WARN] Ricerca sparse/BM25 non disponibile: {exc}")
                return []
            except TypeError:
                search_kwargs.pop("query_filter", None)
                try:
                    return list(self.client.search(**search_kwargs))
                except ImportError as exc:
                    print(f"   [QDRANT WARN] Ricerca sparse/BM25 non disponibile: {exc}")
                    return []

    def _rrf_fusion(
        self,
        dense_points: list[Any],
        context_points: list[Any],
        question_points: list[Any],
        sparse_points: list[Any],
        limit: int,
        boost_keyword: bool,
        preferred_categories: list[str],
    ) -> list[SearchResult]:
        fused: dict[str, dict[str, Any]] = {}

        def add_points(points: list[Any], channel: str, weight: float = 1.0) -> None:
            for rank, point in enumerate(points, 1):
                payload = dict(point.payload or {})
                chunk_id = payload.get("chunk_id") or str(point.id)
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "payload": payload,
                        "score": 0.0,
                        "channels": set(),
                        "raw_scores": {},
                    },
                )
                entry["score"] += weight / (RRF_K + rank)
                entry["channels"].add(channel)
                raw_score = float(getattr(point, "score", 0.0) or 0.0)
                if math.isfinite(raw_score):
                    entry["raw_scores"][channel] = raw_score

        add_points(dense_points, "dense", weight=1.0)
        add_points(context_points, "context", weight=0.9)
        add_points(question_points, "question", weight=1.15)
        add_points(sparse_points, "sparse", weight=2.0 if boost_keyword else 1.1)

        for entry in fused.values():
            payload = entry["payload"]
            if preferred_categories and payload.get("category") in preferred_categories:
                entry["score"] *= 1.15
            if payload.get("node_type") == "raptor" or int(payload.get("raptor_level") or 0) > 0:
                entry["score"] *= 1.05
            payload["retrieval_channels"] = sorted(entry["channels"])
            payload["retrieval_raw_scores"] = entry["raw_scores"]

        ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        results: list[SearchResult] = []
        for item in ordered[:limit]:
            p = item["payload"]
            score = float(item["score"])
            if not math.isfinite(score):
                continue
            results.append(
                SearchResult(
                    score=score,
                    text=p.get("text", ""),
                    url=p.get("url", ""),
                    page_title=p.get("page_title", ""),
                    page_context=p.get("page_context", ""),
                    payload=p,
                )
            )
        return results

    def _expand_raptor_results(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        expanded: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()

        def add_result(result: SearchResult) -> None:
            chunk_id = result.payload.get("chunk_id") or f"{result.url}:{len(expanded)}"
            if chunk_id in seen_chunk_ids or not result.text:
                return
            seen_chunk_ids.add(chunk_id)
            expanded.append(result)

        for result in results:
            payload = result.payload
            is_raptor = payload.get("node_type") == "raptor" or int(payload.get("raptor_level") or 0) > 0
            if not is_raptor:
                add_result(result)
            else:
                child_ids = payload.get("descendant_leaf_ids") or payload.get("children_ids") or []
                child_points = self._retrieve_points_by_chunk_ids(child_ids[:12])
                if not child_points:
                    add_result(result)
                for rank, point in enumerate(child_points, 1):
                    child_payload = dict(point.payload or {})
                    if child_payload.get("node_type") == "raptor":
                        continue
                    child_score = result.score * max(0.65, 1.0 - rank * 0.03)
                    add_result(
                        SearchResult(
                            score=child_score,
                            text=child_payload.get("text", ""),
                            url=child_payload.get("url", ""),
                            page_title=child_payload.get("page_title", ""),
                            page_context=child_payload.get("page_context", ""),
                            payload=child_payload,
                        )
                    )
            if len(expanded) >= limit:
                break

        return expanded[:limit]

    def _retrieve_points_by_chunk_ids(self, chunk_ids: list[str]) -> list[Any]:
        ids = [point_id(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return []
        try:
            return list(
                self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=False,
                )
            )
        except Exception as exc:
            print(f"   [QDRANT WARN] Impossibile espandere nodo RAPTOR: {exc}")
            return []
