# ================================================
# FILE: utils/relevance_evaluator.py
# ================================================
import numpy as np

class RelevanceEvaluator:
    def __init__(self, reranker_model):
        self.reranker = reranker_model

    def calculate_rewards_batch(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        
        # Il modello restituisce le probabilità o i logit grezzi
        scores = self.reranker.predict(pairs)
        
        # Normalizziamo l'array (a volte predict restituisce array annidati)
        if isinstance(scores, np.ndarray) and scores.ndim > 1:
            scores = scores.flatten()
        elif isinstance(scores, list) and isinstance(scores[0], list):
            scores = np.array([x[0] for x in scores])
        else:
            scores = np.array(scores)
            
        return scores.tolist()
    
    def calculate_reward(self, query: str, document: str) -> float:
        return self.calculate_rewards_batch(query, [document])[0]