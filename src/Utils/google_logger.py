import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from pathlib import Path
import sys

def find_project_root():
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config.py").exists():
            return current
        current = current.parent
    return None

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    print("[ERROR] Impossibile trovare la root!")
    sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

class GoogleSheetLogger:
    def __init__(self):
        self.scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        self.sheet = None

        try:
            if not cfg.CREDENTIALS_PATH.exists():
                print(f"[ERROR] File credenziali non trovato in: {cfg.CREDENTIALS_PATH}")
                return

            creds = Credentials.from_service_account_file(str(cfg.CREDENTIALS_PATH), scopes=self.scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(cfg.SHEET_NAME).sheet1
            print(f"[SUCCESS] Connesso al foglio Google: '{cfg.SHEET_NAME}'")
            
        except Exception as e:
            print(f"[ERROR] Connessione a Google Sheets fallita: {e}")

    def log_ragas_evaluation(self, tester, llm_model, embedding_model, retrieval_strategy, 
                             context_prec, context_rec, ans_rel, faith, fact_f1, 
                             prec, rec, inf_time="N/A", notes="", extra_metrics=None):
        """
        Invia i dati al foglio seguendo l'ordine:
        A: Timestamp, B: Tester, C: LLM, D: Embedding, E: Retrieval, F: Context Prec, 
        G: Context Rec, H: Ans Rel, I: Faith, J: F1 Score, K: Precision, L: Recall, 
        M: Inf Time, N: Notes, O: Extra
        """
        if self.sheet is None:
            print("[WARNING] Logger non connesso.")
            return

        try:
            # Costruzione riga basata sull'ordine dello screenshot
            row = [
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), # A: Data e ora
                tester,               # B: Tester
                llm_model,            # C: LLM_model
                embedding_model,      # D: embedding_model
                retrieval_strategy,   # E: Retrieval_strategy
                context_prec,         # F: Context_precision
                context_rec,          # G: Context_recall
                ans_rel,              # H: Answer_Relevancy
                faith,                # I: Faithfulness
                fact_f1,              # J: F1 Score (Factual Correctness)
                prec,                 # K: Precision
                rec,                  # L: Recall
                inf_time,             # M: Inference Time (ms)
                notes                 # N: Notes
            ]
            
            if extra_metrics:
                row.extend(extra_metrics) # O: Extra
            
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"[INFO] Riga aggiunta correttamente per il tester: {tester}")
            
        except Exception as e:
            print(f"[ERROR] Errore durante l'invio al foglio: {e}")

# ==========================================
# TEST DI FUNZIONAMENTO
# ==========================================
if __name__ == "__main__":
    logger = GoogleSheetLogger()
    
    # Esempio 
    logger.log_ragas_evaluation(
        tester="Francesco",
        llm_model="Qwen2.5-7B-Instruct",
        embedding_model="bge-small-en-v1.5",
        retrieval_strategy="Vector Search (k=3)",
        context_prec="0,88",
        context_rec="0,91",
        ans_rel="0,85",
        faith="0,98",
        fact_f1="0,82",
        prec="0,84",   
        rec="0,80",   
        inf_time="1200",
        notes="Test",
        extra_metrics=["N/A"]
    )