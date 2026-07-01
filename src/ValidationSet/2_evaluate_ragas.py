import os
import sys
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from typing import List
from ragas.run_config import RunConfig

# ==========================================
# 0. IMPORTA LA CLASSE JINA
# ==========================================
from WebBasedRAG.RAGComponents.embeddings.jina_embeddings import JinaEmbeddings 
from langchain_core.embeddings import Embeddings

# ==========================================
# 1. IL WRAPPER PER RAGAS
# ==========================================
class RagasJinaWrapper(Embeddings):
    def __init__(self, jina_instance: JinaEmbeddings):
        self.jina = jina_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Metodo chiamato da Ragas per calcolare gli embeddings dei contesti recuperati"""
        # Sfruttiamo il task specifico di Jina V3 per i documenti
        tensors = self.jina.encode(texts, task="retrieval.passage")
        # Convertiamo il tensore Pytorch in lista Python
        return tensors.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Metodo chiamato da Ragas per calcolare l'embedding della domanda (es. per Answer Relevancy)"""
        # Sfruttiamo il task specifico di Jina V3 per le query
        tensors = self.jina.encode([text], task="retrieval.query")
        return tensors.cpu().numpy().tolist()[0]

# ==========================================
# 2. SETUP MODELLI GIUDICI E RAGAS
# ==========================================
from ragas import evaluate

# 1. IMPORTA LE CLASSI CON L'INIZIALE MAIUSCOLA
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_google_genai import ChatGoogleGenerativeAI

# 2. SETUP MODELLO (Senza safety_settings come richiesto)
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.0           
) 

# 3. INIZIALIZZA GLI OGGETTI DELLE METRICHE (Aggiungi le parentesi!)
faith_metric = Faithfulness()
ans_rel_metric = AnswerRelevancy()
ans_rel_metric.strictness = 1  # <-- Si imposta QUI, per risolvere il limite di Gemini 3.1
c_prec_metric = ContextPrecision()
c_rec_metric = ContextRecall()

print("⏳ Caricamento modello Jina in memoria...")
jina_model = JinaEmbeddings(model_name="jinaai/jina-embeddings-v3")
judge_embeddings = RagasJinaWrapper(jina_model)

# ==========================================
# 3. SETUP PATHS, CREDENZIALI E BATCHING
# ==========================================
CURRENT_DIR = Path(os.getcwd())
SRC_PATH = Path(__file__).resolve().parent.parent
SESSION_DIR = SRC_PATH / "session"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

ROOT_DIR = SRC_PATH.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_IN_DOMAIN = SRC_PATH / "ValidationSet" / "results_diem_ragas.json"
TXT_LOG_FILE = SRC_PATH / "ValidationSet" / "ragas_evaluation_log.txt"

EVALUATED_CACHE_FILE = SRC_PATH / "ValidationSet" / "ragas_evaluated_cache.json"

FILE_CREDENZIALI = CURRENT_DIR / "credentials.json"
NOME_FOGLIO = "ChatBot_DIEM_Gruop5" 

BATCH_SIZE = 5 

# ==========================================
# 4. IMPORT DEL LOGGER GOOGLE
# ==========================================
try:
    import config as cfg
    cfg.CREDENTIALS_PATH = FILE_CREDENZIALI
    cfg.SHEET_NAME = NOME_FOGLIO
    from Utils.google_logger import GoogleSheetLogger
except ImportError as e:
    print(f"❌ ERRORE DI IMPORT LOGGER: {e}")
    sys.exit(1)


def run_evaluation():
    print(f"🧪 Avvio Valutazione RAGAS (Batch Size: {BATCH_SIZE})...")
    
    # 1. Carica il Dataset Generato (tutte le domande prodotte dallo step 1)
    if not os.path.exists(RESULTS_IN_DOMAIN):
        print(f"❌ Errore: File {RESULTS_IN_DOMAIN} non trovato. Esegui prima lo step 1.")
        return
        
    with open(RESULTS_IN_DOMAIN, 'r', encoding='utf-8') as f:
        all_generated_data = json.load(f)
    
    # 2. Carica lo storico delle valutazioni (se esiste)
    storico_valutazioni = []
    se_esiste = os.path.exists(EVALUATED_CACHE_FILE)
    if se_esiste:
        with open(EVALUATED_CACHE_FILE, 'r', encoding='utf-8') as f:
            storico_valutazioni = json.load(f)
            
    # Crea un set delle domande già valutate per ricerca veloce
    domande_gia_valutate = {item.get('question', item.get('user_input', '')) for item in storico_valutazioni}
    print(f"📚 Storico: Trovate {len(domande_gia_valutate)} domande già valutate in precedenza.")

    # 3. Filtra le domande che non sono ancora state valutate
    dati_da_valutare = [item for item in all_generated_data if item['question'] not in domande_gia_valutate]
    
    # Prendi solo il batch desiderato (es. le prime 5 non valutate)
    batch_attuale = dati_da_valutare[:BATCH_SIZE]
    
    if len(batch_attuale) == 0:
        print("✅ Tutte le domande nel file generato sono già state valutate! Nessun token sprecato.")
        return

    print(f"⏳ Invio al giudice LLM di {len(batch_attuale)} nuove domande...")
    
    # 4. Esecuzione RAGAS SOLO SULLE NUOVE DOMANDE
    dataset_hf = Dataset.from_list(batch_attuale)
    run_config = RunConfig(max_workers=2, max_wait=60)
    
    # Passa la lista con gli oggetti inizializzati (_metric)
    result = evaluate(
        dataset=dataset_hf,
        metrics=[faith_metric, ans_rel_metric, c_prec_metric, c_rec_metric],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config
    )
    
    # Converti i nuovi risultati in dizionari
    df_nuovi_risultati = result.to_pandas()
    nuovi_risultati_dict = df_nuovi_risultati.to_dict(orient='records')
    
    # 5. Aggiorna lo storico e salva su file JSON
    storico_valutazioni.extend(nuovi_risultati_dict)
    with open(EVALUATED_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(storico_valutazioni, f, indent=4, ensure_ascii=False)
        
    print(f"💾 File di cache aggiornato. Totale valutazioni salvate: {len(storico_valutazioni)}")
    
    # 6. Calcolo Medie GLOBALI (su tutto lo storico, non solo sulle ultime 5)
    df_totale = pd.DataFrame(storico_valutazioni)
    avg_faithfulness = df_totale['faithfulness'].mean()
    avg_relevancy = df_totale['answer_relevancy'].mean()
    avg_precision = df_totale['context_precision'].mean()
    avg_recall = df_totale['context_recall'].mean()
    
    print("\n📊 --- RISULTATI FINALI CUMULATIVI ---")
    print(f"Domande totali valutate finora: {len(df_totale)}")
    print(f"Faithfulness: {avg_faithfulness:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Context Precision: {avg_precision:.2f}")
    print(f"Context Recall: {avg_recall:.2f}")

    # 7. Salvataggio su TXT 
    print("\n💾 Salvataggio log locale su TXT...")
    try:
        with open(TXT_LOG_FILE, 'a', encoding='utf-8') as f:
            from datetime import datetime
            ora_corrente = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Valutazione del {ora_corrente} (Tot: {len(df_totale)} domande) ---\n")
            f.write(f"Faithfulness: {avg_faithfulness:.4f}\n")
            f.write(f"Answer Relevancy: {avg_relevancy:.4f}\n")
            f.write(f"Context Precision: {avg_precision:.4f}\n")
            f.write(f"Context Recall: {avg_recall:.4f}\n")
            f.write("Dettagli del nuovo Batch:\n")
            
            # 👇👇 MODIFICA QUI IL CICLO FOR 👇👇
            for index, row in df_nuovi_risultati.iterrows():
                # RAGAS 0.2.x rinomina 'question' in 'user_input'
                q_text = row.get('question', row.get('user_input', 'Domanda Sconosciuta'))
                faith = row.get('faithfulness', 'N/A')
                rel = row.get('answer_relevancy', 'N/A')
                c_prec = row.get('context_precision', 'N/A')
                c_rec = row.get('context_recall', 'N/A')
                
                f.write(f"Q: {q_text}\n  Faith: {faith} | Rel: {rel} | C.Prec: {c_prec} | C.Rec: {c_rec}\n")
            # 👆👆-------------------------👆👆
        print("✅ TXT salvato.")
    except Exception as e:
        print(f"⚠️ Errore salvataggio TXT: {e}")

    # 8. Invio a Google Sheets 
    print("\n🌐 Invio dati a Google Sheets...")
    try:
        logger = GoogleSheetLogger()
        logger.log_ragas_evaluation(
            tester="Francesco",
            llm_model="gemini-3.1-flash-lite", 
            embedding_model="JinaEmbeddings_V3",
            retrieval_strategy="Agentic_RAG",
            context_prec=str(round(avg_precision, 4)), 
            context_rec=str(round(avg_recall, 4)),     
            ans_rel=str(round(avg_relevancy, 4)),
            faith=str(round(avg_faithfulness, 4)),
            fact_f1="N/A",
            prec="N/A",   
            rec="N/A",    
            inf_time="N/A",
            notes=f"Media cumulativa basata su {len(df_totale)} domande. Valutazione_Automatica_Batch_{len(df_totale)}"
        )
        print("✅ Invio a Google Sheets completato con successo!")
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO GOOGLE SHEETS: {e}")
        print("I dati sono comunque salvi nel file TXT locale e nel file di cache.")

if __name__ == "__main__":
    run_evaluation()