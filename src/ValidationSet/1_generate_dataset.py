import os
import sys
import json
from pathlib import Path
from EntryPoint.AgenticRAG import AgenticRag

# ==========================================
# 1. SETUP PATHS E BATCHING
# ==========================================
CURRENT_DIR = Path(os.getcwd())
SRC_PATH = Path(__file__).resolve().parent.parent 
SESSION_DIR = SRC_PATH / "session"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

HISTORY_FILE = SESSION_DIR / "session_history.jsonl"

DATASET_IN_DOMAIN = SRC_PATH / "ValidationSet" / "dataset_diem_NUOVO.json"
RESULTS_IN_DOMAIN = SRC_PATH / "ValidationSet" / "results_diem_ragas.json"

# ----> IMPOSTA QUI QUANTE DOMANDE GENERARE AD OGNI AVVIO <----
BATCH_SIZE = 5

# ==========================================
# 2. IMPORT 
# ==========================================
from src.EntryPoint.AgenticRAG import AgenticRag 

def clear_bot_memory():
    """Svuota la memoria per garantire valutazioni indipendenti per ogni domanda."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                pass # Aprire in modalità 'w' e non scrivere nulla svuota il file
            print("🧹 [SYSTEM] File session_history.jsonl azzerato all'avvio.")
    except Exception as e:
        print(f"⚠️ [SYSTEM] Impossibile svuotare la history all'avvio: {e}")


def run_generation():
    print(f"📂 Caricamento dataset originale da: {DATASET_IN_DOMAIN}")
    with open(DATASET_IN_DOMAIN, 'r', encoding='utf-8') as f:
        dataset_completo = json.load(f)

    # 1. Carica lo storico delle risposte già generate (se esiste)
    ragas_dataset = []
    if os.path.exists(RESULTS_IN_DOMAIN):
        with open(RESULTS_IN_DOMAIN, 'r', encoding='utf-8') as f:
            try:
                ragas_dataset = json.load(f)
            except json.JSONDecodeError:
                ragas_dataset = [] # In caso il file esista ma sia vuoto/corrotto

    # Crea un set delle domande già fatte per filtrarle velocemente
    domande_gia_generate = {item["question"] for item in ragas_dataset}
    print(f"📚 Storico: Trovate {len(domande_gia_generate)} risposte già generate in precedenza.")

    # 2. Filtra le domande che NON hanno ancora una risposta
    dati_da_generare = [item for item in dataset_completo if item["domanda"] not in domande_gia_generate]
    
    # 3. Prendi solo il batch desiderato
    batch_attuale = dati_da_generare[:BATCH_SIZE]

    if len(batch_attuale) == 0:
        print("✅ Tutte le domande del dataset originale hanno già una risposta generata! Generazione completata.")
        return

    print(f"🚀 Avvio generazione su un batch di {len(batch_attuale)} nuove domande (su {len(dati_da_generare)} rimanenti)...")
    
    # 4. Esegui il bot solo sulle nuove domande
    agente = AgenticRag()
    for i, item in enumerate(batch_attuale):
        print(f"\n--- Elaborazione {i+1}/{len(batch_attuale)} ---")
        domanda = item["domanda"]
        risposta_ideale = item["risposta"]
        
        # 4a. Svuota la memoria prima di ogni domanda
        clear_bot_memory()
        
        # 4b. Esegui il bot in modo "silenzioso"
        try:
            for log_html, ui_text in agente.run(raw_query=domanda):
                pass
        except Exception as e:
            print(f"⚠️ Errore generazione per la domanda '{domanda[:20]}': {e}")
            continue # Salta questa domanda e passa alla prossima del batch
        
        # 4c. Cattura i dati che servono a RAGAS
        risposta_generata = getattr(agente, 'last_pure_answer', "Errore generazione") 
    
        contesto_testuale = getattr(agente, 'last_retrieved_context', "")
        contesti_recuperati = [contesto_testuale] if contesto_testuale.strip() else []

        # 4d. Appendi al dataset complessivo
        ragas_dataset.append({
            "question": domanda,
            "answer": risposta_generata,
            "contexts": contesti_recuperati,
            "ground_truth": risposta_ideale
        })
        
        print(f"✅ Fatto. Risposta generata per: '{domanda[:30]}...'")

        # 5. SALVATAGGIO INCREMENTALE 
        # Salviamo il file json DOPO OGNI DOMANDA. Così, se il bot va in crash 
        # a metà del batch, non perdi nemmeno la domanda precedente!
        with open(RESULTS_IN_DOMAIN, 'w', encoding='utf-8') as f:
            json.dump(ragas_dataset, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Batch completato! Risultati aggiornati pronti per RAGAS in: {RESULTS_IN_DOMAIN}")
    print(f"📊 Progresso totale: {len(ragas_dataset)} / {len(dataset_completo)}")

if __name__ == "__main__":
    run_generation()