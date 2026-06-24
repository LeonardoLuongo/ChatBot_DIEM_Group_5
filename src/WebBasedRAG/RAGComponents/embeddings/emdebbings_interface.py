from abc import ABC, abstractmethod
from typing import List, Optional, Literal
import torch

# 1. Definizione dell'Interfaccia Base
class BaseEmbeddings(ABC):
    @abstractmethod
    def encode(
        self, 
        texts: List[str], 
        task: Optional[Literal[
            'retrieval.query', 
            'retrieval.passage', 
            'separation', 
            'classification', 
            'text-matching'
        ]] = None,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Genera gli embeddings per i testi forniti."""
        pass

    @abstractmethod
    def compute_similarity(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """Calcola la similarità tra due set di embeddings."""
        pass