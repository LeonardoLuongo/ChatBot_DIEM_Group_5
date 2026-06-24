from __future__ import annotations

import math
import re
from collections import Counter


class PageDeduplicator:
    """Gestisce deduplica esatta e near-duplicate via TF-IDF online."""

    def __init__(self, near_duplicate_threshold: float = 0.96, max_candidates: int = 250) -> None:
        self.near_duplicate_threshold = near_duplicate_threshold
        self.max_candidates = max_candidates
        self.text_hash_to_url: dict[str, str] = {}
        self.accepted_page_term_counts: list[tuple[str, Counter[str], set[str], int]] = []
        self.document_frequency: Counter[str] = Counter()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Tokenizza il testo per il confronto TF-IDF."""
        return [token for token in re.findall(r"\w{3,}", text.lower()) if not token.isdigit()]

    def _tfidf_vector(self, term_counts: Counter[str], corpus_size: int) -> dict[str, float]:
        """Costruisce un vettore TF-IDF semplice sul corpus già accettato."""
        return {
            term: count * (math.log((1 + corpus_size) / (1 + self.document_frequency[term])) + 1.0)
            for term, count in term_counts.items()
        }

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        """Calcola la cosine similarity tra due vettori sparsi."""
        if not left or not right:
            return 0.0
        shared_terms = left.keys() & right.keys()
        numerator = sum(left[term] * right[term] for term in shared_terms)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def find_near_duplicate(self, text: str) -> tuple[str | None, float]:
        """Trova il documento piu simile usando TF-IDF su candidati prefiltrati."""
        if not self.accepted_page_term_counts:
            return None, 0.0
        candidate_counts = Counter(self._tokens(text))
        candidate_terms = set(candidate_counts)
        candidate_len = max(1, sum(candidate_counts.values()))
        if not candidate_terms:
            return None, 0.0

        # una pagina quasi identica deve condividere molti
        # termini e avere una lunghezza confrontabile. Cosi evitiamo confronti
        # TF-IDF contro migliaia di pagine sicuramente diverse.
        rough_candidates: list[tuple[float, str, Counter[str]]] = []
        for accepted_url, accepted_counts, accepted_terms, accepted_len in self.accepted_page_term_counts:
            length_ratio = min(candidate_len, accepted_len) / max(candidate_len, accepted_len)
            if length_ratio < 0.55:
                continue
            overlap = len(candidate_terms & accepted_terms)
            if overlap == 0:
                continue
            jaccard = overlap / len(candidate_terms | accepted_terms)
            if jaccard < 0.18:
                continue
            rough_candidates.append((jaccard * length_ratio, accepted_url, accepted_counts))

        if not rough_candidates:
            return None, 0.0

        rough_candidates.sort(reverse=True, key=lambda item: item[0])
        rough_candidates = rough_candidates[: self.max_candidates]

        corpus_size = len(self.accepted_page_term_counts) + 1
        candidate_vector = self._tfidf_vector(candidate_counts, corpus_size)
        best_url = None
        best_similarity = 0.0
        for _rough_score, accepted_url, accepted_counts in rough_candidates:
            similarity = self._cosine_similarity(
                candidate_vector,
                self._tfidf_vector(accepted_counts, corpus_size),
            )
            if similarity > best_similarity:
                best_url = accepted_url
                best_similarity = similarity
        if best_similarity >= self.near_duplicate_threshold:
            return best_url, best_similarity
        return None, best_similarity

    def register_unique_page(self, url: str, text: str) -> None:
        """Registra una pagina già accettata nel corpus di confronto."""
        term_counts = Counter(self._tokens(text))
        self.accepted_page_term_counts.append((url, term_counts, set(term_counts), max(1, sum(term_counts.values()))))
        self.document_frequency.update(term_counts.keys())
