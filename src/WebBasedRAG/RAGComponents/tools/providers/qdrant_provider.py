from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider
from WebBasedRAG.RAGComponents.tools.services.qdrant_service import QdrantService
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState
from WebBasedRAG.RAGComponents.relevance_evaluator import RelevanceEvaluator
import re
class SearchDatabaseInput(BaseModel):
    query: str = Field(
        ..., 
        description="The natural language question or keywords to search in the internal database."
    )

def normalize_scores(values: list[float]) -> list[float]:
    """Normalizza una lista di score in [0, 1] (Portato dal tuo script SOTA)"""
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]

class QdrantToolProvider(ToolProvider):
    def __init__(self, service: QdrantService, session_state: GlobalSessionState, evaluator: RelevanceEvaluator):
        self.service = service
        self.session_state = session_state
        self.evaluator = evaluator # <-- Il nostro CrossEncoder condiviso!

    def get_tools(self) -> List[StructuredTool]:
        
        def search_internal_db(query: str) -> str:
            print(f"   [QDRANT] Ricerca Fase 1 (120 candidati) per: '{query}'")
            
            # 1. RETRIEVAL (Alta Recall): Peschiamo 300 candidati come nel tuo script
            candidates = self.service.search(query=query, limit=120)
            
            if not candidates:
                return "NESSUN RISULTATO TROVATO NEL DATABASE INTERNO. Procedi utilizzando il tool 'WebSearch'."
                
            # 2. RERANKING (Alta Precisione)
            print(f"   [QDRANT] Ricerca Fase 2: Reranking profondo (BGE) su {len(candidates)} chunk...")
            texts = [c.text for c in candidates]
            rerank_raw = self.evaluator.calculate_rewards_batch(query, texts)
            
            # 3. FUSIONE DEI PUNTEGGI (La tua logica 70/30)
            retrieval_scores = [float(c.score) for c in candidates]
            retrieval_norm = normalize_scores(retrieval_scores)
            rerank_norm = normalize_scores(rerank_raw)
            
            scored_candidates = []
            for idx, candidate in enumerate(candidates):
                final_score = 0.70 * rerank_norm[idx] + 0.30 * retrieval_norm[idx]
                scored_candidates.append((final_score, candidate))
                
            # Ordiniamo per il punteggio finale tutti i frammenti
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Riduciamo il pool iniziale per velocità
            frammenti_estesi = scored_candidates[:20]
            
            # --- LIMITE CHUNK: Massimo 3 frammenti per ogni singola pagina web ---
            MAX_CHUNKS_PER_URL = 3 
            
            documenti_raggruppati = {}
            for score, candidate in frammenti_estesi:
                url = candidate.url
                testo_nuovo = candidate.text.strip()
                
                if url not in documenti_raggruppati:
                    documenti_raggruppati[url] = {
                        "titolo": candidate.page_title,
                        "testo_unito": testo_nuovo,
                        "best_score": score,
                        "chunk_count": 1
                    }
                else:
                    # 1. CONTROLLO LIMITE: Se abbiamo già i 3 pezzi migliori di questa pagina, ignoriamo il resto!
                    if documenti_raggruppati[url]["chunk_count"] >= MAX_CHUNKS_PER_URL:
                        continue
                        
                    testo_attuale = documenti_raggruppati[url]["testo_unito"]
                    
                    # 2. FIX ANTI-RIPETIZIONE
                    if testo_nuovo in testo_attuale:
                        continue
                    elif testo_attuale in testo_nuovo:
                        documenti_raggruppati[url]["testo_unito"] = testo_nuovo
                    else:
                        documenti_raggruppati[url]["testo_unito"] += f"\n[...] {testo_nuovo}"
                        documenti_raggruppati[url]["chunk_count"] += 1

            # Ordiniamo i documenti raggruppati per il loro score migliore
            top_grouped_docs = sorted(documenti_raggruppati.items(), key=lambda x: x[1]["best_score"], reverse=True)
            
            # Passiamo i 6 documenti migliori (Snelli e iper-focalizzati)
            top_urls = top_grouped_docs[:6]
            
            # 4. SALVATAGGIO
            preview_blocks = []
            for url, dati in top_urls:
                doc_id = self.session_state._doc_counter + 1
                
                testo_pulito = re.sub(r'\n{3,}', '\n\n', dati['testo_unito']).strip()
                
                testo_da_salvare = (
                    f"Titolo del Documento: {dati['titolo']} (Costruito da {dati['chunk_count']} frammenti principali)\n"
                    f"Contesto Esteso:\n{testo_pulito}"
                )
                
                # Inserisce nel DB Globale
                new_id = self.session_state.add_source(
                    url=url,
                    title=dati['titolo'],
                    testo_estratto=f"[Doc {doc_id}]\n{testo_da_salvare}",
                    smart_url=url,
                    score=dati['best_score']
                )
                
                preview_blocks.append(testo_da_salvare)

            return f"✅ Trovati {len(top_urls)} documenti rilevanti (deduplicati e ottimizzati in lunghezza). Anteprima:\n\n" + "\n\n---\n\n".join(preview_blocks)

        qdrant_tool = StructuredTool.from_function(
            func=search_internal_db,
            name="SearchInternalDatabase",
            description=(
                "Searches the verified Internal Vector Database of the DIEM department. "
                "USE THIS TOOL FIRST for regulations, thesis rules, history, CFU requirements, and general knowledge. "
            ),
            args_schema=SearchDatabaseInput
        )
        
        return [qdrant_tool]