from typing import Dict, Any, List, Optional

class GlobalSessionState:
    def __init__(self, max_docs: int = 10):
        # 1. Memoria per i Documenti RAG
        self.source_mapping: Dict[int, Dict[str, Any]] = {}
        self._doc_counter: int = 0
        self.max_docs: int = max_docs
        
        # 2. Memoria per i Tool Multi-Turno
        self._tool_memory: Dict[str, Any] = {}

    # =========================================================
    # GESTIONE CONTESTO E DOCUMENTI (RAG)
    # =========================================================

    def prune_context(self, query: str, evaluator, threshold: float = 0.20) -> int:
        """
        Active Context Pruning: Valuta i documenti in memoria rispetto alla nuova query 
        e scarta quelli con score semantico inferiore alla soglia.
        Restituisce il numero di documenti rimossi.
        """
        if not self.source_mapping:
            return 0
            
        docs = list(self.source_mapping.items())
        texts = [data["testo"] for _, data in docs]
        
        # Calcolo score in batch tramite il Cross-Encoder SOTA (Velocissimo)
        scores = evaluator.calculate_rewards_batch(query, texts)
        
        keys_to_remove = []
        for (doc_id, _), score in zip(docs, scores):
            if score < threshold:
                keys_to_remove.append(doc_id)
                
        for doc_id in keys_to_remove:
            del self.source_mapping[doc_id]
            
        return len(keys_to_remove)

    def add_source(self, url: str, title: str, testo_estratto: str, smart_url: Optional[str] = None, score: float = 1.0, external_links: Optional[List] = None) -> int:
        self._doc_counter += 1
        doc_id = self._doc_counter
        
        self.source_mapping[doc_id] = {
            "url": url,
            "smart_url": smart_url or url,
            "titolo": title,
            "score": score,
            "testo": testo_estratto,
            "external_links": external_links or [] 
        }

        # --- POLIZIA DEL CONTESTO (Eviction Policy) ---
        if len(self.source_mapping) > self.max_docs:
            # Ordiniamo per Score (crescente). A parità di score, usiamo il doc_id 
            # (così a parità di punteggio eliminiamo il documento più vecchio!)
            id_da_rimuovere = min(self.source_mapping.keys(), key=lambda k: (self.source_mapping[k]["score"], k))
            del self.source_mapping[id_da_rimuovere]
            
        return doc_id

    def get_all_context(self) -> str:
        """Restituisce tutto il contesto accumulato per la risposta finale"""
        if not self.source_mapping:
            return ""
        context_blocks = [data["testo"] for doc_id, data in self.source_mapping.items()]
        return "\n\n".join(context_blocks)

    def clear(self):
        """Pulisce l'intera sessione (chiamato ad ogni nuova query o svuota chat)"""
        self.source_mapping.clear()
        self._doc_counter = 0
        self._tool_memory.clear()

    # =========================================================
    # GESTIONE MEMORIA DEI TOOL (Multi-Turno)
    # =========================================================

    def get(self, key: str, default: Any = None) -> Any:
        return self._tool_memory.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._tool_memory[key] = value

    def remove(self, key: str) -> None:
        if key in self._tool_memory:
            del self._tool_memory[key]