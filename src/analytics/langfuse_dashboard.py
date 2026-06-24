import os
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

# --- SETUP PATH ---
import os
import sys
from pathlib import Path
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(SRC_PATH)


_langfuse_handler = None
def init_langfuse() -> CallbackHandler:
    global _langfuse_handler
    if _langfuse_handler:
        return _langfuse_handler
    load_dotenv()

    # Inserisci le chiavi generate prima
    _langfuse_handler = CallbackHandler(
        # public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        # secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        # host=os.getenv("LANGFUSE_BASE_URL")
    )
    return _langfuse_handler
