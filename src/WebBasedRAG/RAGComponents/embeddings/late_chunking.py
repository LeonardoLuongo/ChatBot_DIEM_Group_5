import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Literal
from .emdebbings_interface import BaseEmbeddings

class NomicLateChunkingEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} (Late Chunking enabled) on {self.device}")

        # Usiamo i modelli base di HF per avere accesso ai token nascosti (hidden states)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        self.task_prefixes = {
            'retrieval.passage': 'search_document: ',
            'retrieval.query': 'search_query: '
        }

    def encode_late_chunking(self, full_document: str, chunk_texts: List[str]) -> torch.Tensor:
        """
        Il cuore del Late Chunking. Calcola gli embedding passando il documento
        INTERO una sola volta, e poi ritaglia i vettori dei chunk a posteriori.
        I chunk fuori dalla Context Window vengono processati parallelamente in BATCH.
        """
        prefix = self.task_prefixes['retrieval.passage']
        doc_with_prefix = prefix + full_document

        # 1. Tokenizziamo l'intero documento mantenendo la mappa dei caratteri
        inputs = self.tokenizer(
            doc_with_prefix,
            return_tensors="pt",
            return_offsets_mapping=True, 
            truncation=True,
            max_length=8192, 
            padding=False
        ).to(self.device)

        offset_mapping = inputs.pop("offset_mapping")[0] 

        # 2. Passaggio SINGOLO nel modello (O(1) invece di O(N) chunk!)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]

        chunk_embeddings = []
        search_start = 0 
        
        # Variabili di appoggio per il salvataggio in BATCH
        fallback_indices = []
        fallback_texts = []

        # 3. Pooling a posteriori sui token ritagliati
        for idx, chunk in enumerate(chunk_texts):
            start_char = full_document.find(chunk, search_start)
            
            if start_char == -1:
                # Se il chunk non fa match esatto, segnamolo per il fallback
                fallback_indices.append(idx)
                fallback_texts.append(prefix + chunk)
                chunk_embeddings.append(None) # Segnaposto
                continue
                
            search_start = start_char + len(chunk)
            
            start_char += len(prefix)
            end_char = start_char + len(chunk)

            token_start, token_end = -1, -1
            for offset_idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start == t_end == 0: continue 
                if token_start == -1 and t_start >= start_char:
                    token_start = offset_idx
                if t_end <= end_char:
                    token_end = offset_idx

            if token_start != -1 and token_end != -1 and token_end >= token_start:
                # MEAN POOLING per i chunk dentro gli 8192 token
                chunk_emb = token_embeddings[token_start:token_end+1].mean(dim=0)
                chunk_embeddings.append(chunk_emb)
            else:
                # Se il chunk supera il limite degli 8192 token, lo mandiamo in fallback
                fallback_indices.append(idx)
                fallback_texts.append(prefix + chunk)
                chunk_embeddings.append(None) # Segnaposto

        # 4. ESECUZIONE BATCH DEL FALLBACK (Fix per il Freeze critico)
        if fallback_texts:
            batch_size = 32
            all_fallback_embs = []
            
            for i in range(0, len(fallback_texts), batch_size):
                batch = fallback_texts[i:i+batch_size]
                # Processiamo 32 chunk in un colpo solo invece di 1 alla volta
                emb_batch = self._early_chunking_fallback(batch) 
                all_fallback_embs.append(emb_batch)
            
            # Uniamo i risultati del batch
            all_fallback_embs = torch.cat(all_fallback_embs, dim=0)
            
            # Sostituiamo i 'None' (segnaposto) con i tensori appena calcolati
            for i, original_idx in enumerate(fallback_indices):
                chunk_embeddings[original_idx] = all_fallback_embs[i]

        # 5. Impiliamo e Normalizziamo
        final_embeddings = torch.stack(chunk_embeddings)
        return F.normalize(final_embeddings, p=2, dim=1)


    def encode(self, texts: List[str], task: str = None, batch_size: int = 32) -> torch.Tensor:
        """Metodo standard per le Query o per compatibilità con la vecchia interfaccia."""
        prefix = self.task_prefixes.get(task, '') if task else ''
        processed_texts = [prefix + t for t in texts]
        
        all_embeddings = []
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i+batch_size]
            emb = self._early_chunking_fallback(batch) # Può accettare batch
            all_embeddings.append(emb)
            
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return F.normalize(final_embeddings, p=2, dim=1)

    def compute_similarity(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        return embeddings_a @ embeddings_b.T

    def _early_chunking_fallback(self, texts) -> torch.Tensor:
        if isinstance(texts, str): texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=8192, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_emb = torch.sum(outputs.last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask