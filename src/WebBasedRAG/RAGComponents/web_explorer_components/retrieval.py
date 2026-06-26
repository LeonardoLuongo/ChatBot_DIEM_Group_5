import torch
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from .utils import MultilingualProcessor
from WebBasedRAG.RAGComponents.relevance_evaluator import RelevanceEvaluator
from WebBasedRAG.RAGComponents.embeddings.jina_embeddings import JinaEmbeddings

class RetrievalEngine:
    def __init__(self, embeddings: JinaEmbeddings, evaluator: RelevanceEvaluator):
        self.embeddings = embeddings
        self.evaluator = evaluator
        # Soglia per decidere l'algoritmo (300 chunk = ~15 pagine PDF)
        self.HYBRID_THRESHOLD = 300 

    def retrieve_and_rank(self, query: str, chunks: List[Dict]) -> List[Tuple[float, str, str]]:
        if not chunks: 
            return []
        
        combined_candidates = []
        
        # =========================================================
        # ADAPTIVE STAGE 1: Dynamic Retrieval Routing
        # =========================================================
        if len(chunks) <= self.HYBRID_THRESHOLD:
            # ---------------------------------------------------------
            # BRANCH A: TRUE HYBRID RRF (Alta Precisione)
            # ---------------------------------------------------------
            print(f"   [STAGE 1 - ADAPTIVE] Documento compatto ({len(chunks)} chunk). Attivo True Hybrid RRF...")
            
            tokenized_corpus = [MultilingualProcessor.preprocess(c["text"]) for c in chunks]
            bm25_index = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25_index.get_scores(MultilingualProcessor.preprocess(query))
            bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}

            texts_corpus = [c["text"] for c in chunks]
            # ESTREMA VELOCITÀ: I chunk hanno già l'embedding dal Late Chunking.
            # Li impiliamo direttamente e li carichiamo sulla GPU/CPU.
            embeddings_list = [c["embedding"] for c in chunks]
            if len(embeddings_list) > 0 and isinstance(embeddings_list[0], torch.Tensor):
                corpus_embeddings = torch.stack(embeddings_list).to(dtype=torch.float32, device=self.embeddings.device)
            else:
                corpus_embeddings = torch.tensor(
                    np.array(embeddings_list), dtype=torch.float32
                ).to(self.embeddings.device)
            
            # Codifichiamo solo la domanda dell'utente
            query_embedding = self.embeddings.encode([query], task="retrieval.query")
            
            similarities = self.embeddings.compute_similarity(query_embedding, corpus_embeddings)[0]
            similarities_np = similarities.cpu().numpy()
            dense_ranks = {idx: rank for rank, idx in enumerate(np.argsort(similarities_np)[::-1])}

            K = 60
            rrf_scores = []
            for i in range(len(chunks)):
                b_rank = bm25_ranks[i] if bm25_scores[i] > 0 else 1000 
                d_rank = dense_ranks[i]
                rrf_scores.append(((1 / (K + b_rank)) + (1 / (K + d_rank)), chunks[i]))
                
            rrf_scores.sort(key=lambda x: x[0], reverse=True)
            combined_candidates = [item[1] for item in rrf_scores[:30]]
            
            # --- PROTEZIONE VRAM REINTEGRATA ---
            del corpus_embeddings, query_embedding, similarities, similarities_np

        else:
            # ---------------------------------------------------------
            # BRANCH B: LEXICAL + DENSE FALLBACK (Alta Velocità)
            # ---------------------------------------------------------
            print(f"   [STAGE 1 - ADAPTIVE] Documento massivo ({len(chunks)} chunk). Attivo Lexical Filter + Dense Fallback...")
            
            tokenized_corpus = [MultilingualProcessor.preprocess(c["text"]) for c in chunks]
            bm25_index = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25_index.get_scores(MultilingualProcessor.preprocess(query))
            top_n_indices_bm25 = bm25_index.get_top_n(MultilingualProcessor.preprocess(query), range(len(chunks)), n=60)
            
            lexical_candidates = [chunks[i] for i in top_n_indices_bm25[:25] if bm25_scores[i] > 0]
            lexical_texts = {c['text'] for c in lexical_candidates}
            
            # Protezione VRAM: analizziamo max 128 orfani
            orphan_chunks = [c for c in chunks if c['text'] not in lexical_texts][:128]
            dense_candidates = []
            
            if orphan_chunks:
                # Usiamo gli embedding pre-calcolati anche qui
                orphan_embs = [c["embedding"] for c in orphan_chunks]
                if len(orphan_embs) > 0 and isinstance(orphan_embs[0], torch.Tensor):
                    corpus_embeddings = torch.stack(orphan_embs).to(dtype=torch.float32, device=self.embeddings.device)
                else:
                    corpus_embeddings = torch.tensor(
                        np.array(orphan_embs), dtype=torch.float32
                    ).to(self.embeddings.device)
                
                query_embedding = self.embeddings.encode([query], task="retrieval.query")
                similarities = self.embeddings.compute_similarity(query_embedding, corpus_embeddings)[0]

                
                top_indices = torch.topk(similarities, min(15, len(orphan_chunks))).indices.tolist()
                dense_candidates = [orphan_chunks[i] for i in top_indices]
                
                # --- PROTEZIONE VRAM REINTEGRATA ---
                del corpus_embeddings, query_embedding, similarities

            combined_candidates = lexical_candidates.copy()
            seen_texts = {c['text'] for c in combined_candidates}
            for dc in dense_candidates:
                if dc['text'] not in seen_texts:
                    combined_candidates.append(dc)
                    seen_texts.add(dc['text'])

        # =========================================================
        # STAGE 2: Deep Reranking (Cross-Encoder)
        # =========================================================
        print(f"   [STAGE 2] Reranking profondo su {len(combined_candidates)} chunk finali...")
        collected = []
        
        if combined_candidates:
            texts_to_rerank = [c["text"] for c in combined_candidates]
            scores = self.evaluator.calculate_rewards_batch(query, texts_to_rerank)

            for chunk_data, score in zip(combined_candidates, scores):
                if score >= 0.525: 
                    # --- DEBUG SNIPPET REINTEGRATO ---
                    snippet = chunk_data["text"][:100].replace('\n', ' ')
                    print(f"        ✅ Trovato utile (Score {score:.2f}): {snippet}...")
                    
                    collected.append((score, chunk_data["url"], chunk_data["text"]))

        # Non serve sortare qui, lo fai già nel WebExplorer dopo l'extend
        return collected