from abc import ABC, abstractmethod
from typing import List, Optional, Literal
import torch
import torch.nn.functional as F

# Per Nomic è raccomandato usare sentence-transformers per gestire il pooling nativo
from sentence_transformers import SentenceTransformer
from WebBasedRAG.RAGComponents.embeddings.emdebbings_interface import BaseEmbeddings

# 3. La nuova classe Nomic
class NomicEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on device: {self.device}")
        # SentenceTransformer è il modo più pulito per caricare Nomic e gestisce in automatico il pooling
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.model.eval()

        # Mappiamo i task di Jina ai prefissi richiesti da Nomic
        self.task_prefixes = {
            'retrieval.query': 'search_query: ',
            'retrieval.passage': 'search_document: ',
            'classification': 'classification: ',
            'separation': 'clustering: ',
            'text-matching': 'search_document: ' 
        }

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
        
        # Aggiunta dinamica del prefisso (Nomic richiede questa formattazione)
        if task and task in self.task_prefixes:
            prefix = self.task_prefixes[task]
            processed_texts = [f"{prefix}{text}" for text in texts]
        else:
            processed_texts = texts

        with torch.no_grad():
            # convert_to_tensor=True restituisce direttamente un tensore PyTorch
            embeddings = self.model.encode(
                processed_texts, 
                batch_size=batch_size, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Normalizzazione L2: raccomandata per Nomic per fare in modo che 
            # il dot product coincida matematicamente con la cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.to(self.device)

    def compute_similarity(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        # Se normalizzati, embeddings_a @ embeddings_b.T equivale alla Cosine Similarity
        return embeddings_a @ embeddings_b.T
    





from transformers import AutoTokenizer

def test_nomic_context_window():
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    print("--- INIZIO TEST CONTEXT WINDOW ---")
    
    # 1. Carichiamo il tokenizer per contare i token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Creiamo un documento gigante (ripetendo una stringa)
    frase = "Questo è un test per verificare la context window del modello Nomic al DIEM. "
    testo_gigante = frase * 1000  # Circa 12.000 parole
    
    # 3. Contiamo i token reali
    # add_special_tokens=False ci dà il numero esatto di token di testo
    token_count = len(tokenizer.encode(testo_gigante, add_special_tokens=False))
    print(f"Lunghezza del documento generato: {token_count} token")
    
    # 4. Inizializziamo il nostro embedder
    embedder = NomicEmbeddings(model_name=model_name)
    
    # 5. Verifichiamo il limite impostato internamente da SentenceTransformer
    max_seq_len = embedder.model.max_seq_length
    print(f"Finestra di contesto massima del modello: {max_seq_len} token")
    
    # 6. Generiamo l'embedding (il modello troncherà automaticamente a 8192)
    print("Generazione embedding in corso...")
    embedding = embedder.encode([testo_gigante], task='retrieval.passage')
    
    print(f"Successo! Shape dell'embedding: {embedding.shape}")
    print("Il modello ha elaborato i primi 8192 token senza errori di 'Index out of range'.")

if __name__ == "__main__":
    test_nomic_context_window()