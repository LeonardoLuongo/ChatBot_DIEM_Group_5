from llmlingua import PromptCompressor

class ContextCompressor:
    def __init__(self, device: str = "cuda"):
        print("Loading LLMLingua-2 for context compression...")
        # Il modello xlm-roberta è SOTA per la compressione cross-lingual
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )

    def compress(self, context: str, target_ratio: float = 0.5) -> str:
        """
        Comprime il contesto mantenendo le informazioni semantiche.
        target_ratio=0.5 riduce il numero di token di circa la metà.
        """
        if not context or not context.strip():
            return context
            
        # force_tokens protegge i metadati essenziali per la mappa delle fonti
        results = self.compressor.compress_prompt_llmlingua2(
            context,
            rate=target_ratio,
            force_tokens=["[Doc", "]", "link:", "http", "https"] 
        )
        
        return results['compressed_prompt']