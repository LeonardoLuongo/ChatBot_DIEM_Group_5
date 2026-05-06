import time
import numpy as np
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision, ContextRecall, 
    ResponseRelevancy, Faithfulness, FactualCorrectness
)

class RAGEvaluator:
    def __init__(self, rag_chain, chat_model, embedding_model, 
                 llm_model_name, embedding_model_name, logger=None):
        """
        Classe parametrizzata per la valutazione del sistema RAG.
        
        :param rag_chain: La catena RAG da testare.
        :param chat_model: L'oggetto modello LLM.
        :param embedding_model: L'oggetto modello di embedding.
        :param llm_model_name: Stringa col nome del modello (es. "Qwen2.5-7B").
        :param embedding_model_name: Stringa col nome dell'embedding (es. "bge-small-en").
        :param logger: Istanza di GoogleSheetLogger.
        """
        self.rag_chain = rag_chain
        self.llm_wrapper = LangchainLLMWrapper(chat_model)
        self.embedding_model = embedding_model
        
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.logger = logger
        
        self.metrics = [
            ContextPrecision(), 
            ContextRecall(),    
            ResponseRelevancy(),
            Faithfulness(),     
            FactualCorrectness(mode="f1") 
        ]

    def run_evaluation(self, test_data, tester_name="Francesco", strategy_name="Vector Search"):
        """
        Esegue la valutazione e logga i risultati usando i nomi dei modelli parametrizzati.
        """
        print(f"\n[EVAL] Avvio valutazione per strategia: {strategy_name}")
        print(f"[INFO] LLM: {self.llm_model_name} | Embed: {self.embedding_model_name}")
        
        dataset_records = []
        inference_times = []

        # 1. Fase di Inferenza
        for i, item in enumerate(test_data):
            query = item["query"]
            gold = item["reference"]
            
            start_time = time.perf_counter()
            result = self.rag_chain.invoke(query)
            inf_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            inference_times.append(inf_time_ms)
            dataset_records.append({
                "user_input": query,
                "retrieved_contexts": [doc.page_content for doc in result["sources"]],
                "response": result["answer"],
                "reference": gold
            })
            print(f"  [{i+1}/{len(test_data)}] Completata in {inf_time_ms}ms")

        # 2. Calcolo RAGAS
        eval_dataset = EvaluationDataset.from_list(dataset_records)
        print("[EVAL] Calcolo metriche RAGAS con Judge LLM...")
        
        results = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
            llm=self.llm_wrapper,
            embeddings=self.embedding_model
        )
        
        # Estrazione punteggi (gestione dizionario dei risultati)
        res_dict = results.scores[0] if isinstance(results, list) else results
        scores = {
            "cp": round(res_dict.get("context_precision", 0), 3),
            "cr": round(res_dict.get("context_recall", 0), 3),
            "ar": round(res_dict.get("answer_relevancy", 0), 3),
            "f": round(res_dict.get("faithfulness", 0), 3),
            "fc": round(res_dict.get("factual_correctness", 0), 3)
        }

        # 3. Logging parametrizzato
        if self.logger:
            avg_inf = int(np.mean(inference_times))
            
            self.logger.log_ragas_evaluation(
                tester=tester_name,
                llm_model=self.llm_model_name,       
                embedding_model=self.embedding_model_name,
                retrieval_strategy=strategy_name,
                context_prec=str(scores["cp"]).replace('.', ','),
                context_rec=str(scores["cr"]).replace('.', ','),
                ans_rel=str(scores["ar"]).replace('.', ','),
                faith=str(scores["f"]).replace('.', ','),
                fact_f1=str(scores["fc"]).replace('.', ','),
                prec=str(scores["cp"]).replace('.', ','), 
                rec=str(scores["cr"]).replace('.', ','), 
                inf_time=str(avg_inf),
                notes=f"Test Set DIEM - Queries: {len(test_data)}"
            )
        
        print(f"[SUCCESS] Valutazione completata e loggata.")
        return results.to_pandas()