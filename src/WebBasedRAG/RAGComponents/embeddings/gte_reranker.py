# ================================================
# FILE: RAGComponents/embeddings/bge_8k_reranker.py
# ================================================
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

class BGE8KReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        # Salviamo il device "target" per la GPU
        self.gpu_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.gpu_device # Device attuale del modello
        
        print(f"   -> Caricamento BGE-m3 Reranker (8K Context) in FP16...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # FP16 dimezza la memoria e raddoppia la velocità sulla tua 5070 Ti
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 
        )
        
        # Inizializziamo il modello sulla GPU (oppure puoi forzarlo su CPU chiamando subito to_cpu)
        self.model.to(self.device)
        self.model.eval()

    def to_cpu(self):
        """Sposta il modello in RAM e libera la VRAM della GPU."""
        if self.device == "cpu":
            return
            
        self.model.to("cpu")
        self.device = "cpu"
        
        # CRITICO: PyTorch trattiene la memoria cache anche dopo aver spostato il modello.
        # Questa riga forza il rilascio effettivo della VRAM al sistema.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # print("   -> Reranker spostato su CPU (VRAM liberata).") # Decommenta per debug

    def to_gpu(self):
        """Sposta il modello sulla GPU per l'inferenza."""
        if self.device == self.gpu_device:
            return
            
        self.model.to(self.gpu_device)
        self.device = self.gpu_device
        
        # print(f"   -> Reranker spostato su {self.gpu_device}.") # Decommenta per debug

    def predict(self, pairs: List[List[str]], batch_size: int = 16, auto_manage_vram: bool = False) -> List[float]:
        # 1. Se richiesto, sposta il modello su GPU prima di iniziare
        if auto_manage_vram:
            self.to_gpu()

        all_scores = []
        
        # Usiamo try...finally per garantire che, in caso di crash sui tensori, 
        # il modello non rimanga bloccato nella GPU consumando memoria.
        try:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(self.device) # I tensori seguono il device attuale del modello
                
                # Fix per il crash XLM-RoBERTa
                if "token_type_ids" in inputs:
                    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
                    
                with torch.inference_mode():
                    logits = self.model(**inputs).logits.squeeze(-1)
                    
                    if logits.ndim == 0:
                        logits = logits.unsqueeze(0)
                        
                    scores = torch.sigmoid(logits).cpu().tolist()
                    
                    if isinstance(scores, float):
                        all_scores.append(scores)
                    else:
                        all_scores.extend(scores)
                        
        finally:
            # 2. Riporta immediatamente il modello su CPU a fine ciclo
            if auto_manage_vram:
                self.to_cpu()
                
        return all_scores