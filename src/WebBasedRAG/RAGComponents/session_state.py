# FILE: RAGComponents/session_state.py
from typing import Dict, Any

class GlobalSessionState:
    def __init__(self, max_docs: int = 10):
        self.source_mapping: Dict[int, Dict[str, Any]] = {}
        self._doc_counter: int = 0
        self.max_docs = max_docs

    def add_source(self, url: str, title: str, testo_estratto: str, smart_url: str = None, score: float = 1.0, external_links: list = None) -> int:
        self._doc_counter += 1
        doc_id = self._doc_counter
        
        self.source_mapping[doc_id] = {
            "url": url,
            "smart_url": smart_url or url,
            "titolo": title,
            "score": score,
            "testo": testo_estratto,
            "external_links": external_links or [] # <-- Salvato qui!
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
        context_blocks = []
        for doc_id, data in self.source_mapping.items():
            context_blocks.append(data["testo"])
        return "\n\n".join(context_blocks)

    def clear(self):
        self.source_mapping.clear()
        self._doc_counter = 0