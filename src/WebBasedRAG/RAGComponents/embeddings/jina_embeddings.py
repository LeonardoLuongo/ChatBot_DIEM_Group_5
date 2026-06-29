import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, PreTrainedModel
from typing import List, Optional, Literal

# Patch per Jina V3
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}


class JinaEmbeddings:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        print(f"Caricamento Jina Embeddings ({model_name}) su device: {self.device}...")
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
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
        Genera gli embeddings per i testi forniti.
        """
        embeddings = self.model.encode(
            texts, 
            task=task,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            prompt_name=task 
        )
        # Convertiamo sempre l'output in float32 per evitare crash nel dot product
        return embeddings.to(device=self.device, dtype=torch.float32)

    def compute_similarity(self, query_embedding, corpus_embeddings) -> torch.Tensor:
        """Calcola la similarità coseno tra due set di embeddings."""
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding)
            
        if isinstance(corpus_embeddings, list):
            corpus_embeddings = torch.tensor(np.array(corpus_embeddings))
        elif not isinstance(corpus_embeddings, torch.Tensor):
            corpus_embeddings = torch.tensor(corpus_embeddings)
            
        # Forziamo la precisione float32 su entrambi gli operandi
        return util.cos_sim(
            query_embedding.to(device=self.device, dtype=torch.float32), 
            corpus_embeddings.to(device=self.device, dtype=torch.float32)
        )

