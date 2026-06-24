import torch
from transformers import AutoModel, PreTrainedModel
from typing import List, Optional, Literal

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# Patch per Jina V3
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

class JinaEmbeddings:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        print(f"jinaai/jina-embeddings-v3 loaded on device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

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
        """
        Genera gli embeddings per i testi forniti usando i LoRA adapters specifici per il task.
        Se task è None, usa i pesi base senza adapter.
        """
        with torch.no_grad():
            # Passiamo il batch_size al modello sottostante
            embeddings = self.model.encode(texts, task=task, batch_size=batch_size)
            # Converte in tensore PyTorch se non lo è già, e lo sposta sul device corretto
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)
            return embeddings.to(self.device)

    def compute_similarity(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """Calcola la similarità (dot product) tra due set di embeddings."""
        return embeddings_a @ embeddings_b.T