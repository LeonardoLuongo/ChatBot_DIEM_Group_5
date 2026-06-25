import string
import re
import jieba
from typing import List
import concurrent.futures
from collections import OrderedDict
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect, LangDetectException
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class LRUCache(OrderedDict):
    def __init__(self, maxsize=50, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

class MultilingualProcessor:
    stemmers = {
        'it': SnowballStemmer("italian"),
        'en': SnowballStemmer("english"),
        'es': SnowballStemmer("spanish"),
        'fr': SnowballStemmer("french")
    }

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            lang = detect(text)
            if lang.startswith('zh'): return 'zh'
            return lang
        except LangDetectException:
            return 'en'

    @classmethod
    def preprocess(cls, text: str) -> List[str]:
        lang = cls.detect_language(text)
        text_clean = text.translate(str.maketrans('', '', string.punctuation)).lower()
        if lang == 'zh':
            return list(jieba.cut(text_clean, cut_all=False))
        elif lang in cls.stemmers:
            return [cls.stemmers[lang].stem(word) for word in text_clean.split()]
        else:
            return text_clean.split()

class DocumentChunkerOld:
    def __init__(self):
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Titolo_1"), ("##", "Titolo_2"), ("###", "Titolo_3"), ("####", "Titolo_4")],
            strip_headers=False 
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=400, separators=["\n\n", "\n", ".", " "]
        )

    def process_markdown(self, markdown_text: str, timeout: float = 0.5) -> List[str]:
        def _process():
            md_splits = self.markdown_splitter.split_text(markdown_text)
            final_splits = self.fallback_splitter.split_documents(md_splits)
            return [
                f"CONTESTO: [{' > '.join(doc.metadata.values())}]\n{doc.page_content}" 
                if doc.metadata.values() else doc.page_content 
                for doc in final_splits
            ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_process)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"Timeout chunker superato ({timeout}s).")
                return []


from dataclasses import dataclass
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import LateChunker
import torch
import re

# Semplice wrapper per simulare il comportamento di Chonkie ma con il testo espanso
@dataclass
class ExpandedChunk:
    text: str
    embedding: Any

class DocumentChunker:
    def __init__(self):
        print("Caricamento Chonkie (Modalità: Late Chunking + Small-to-Big) in corso...")
        
        # 1. CHILD SPLITTER: Genera i vettori ultra-precisi (Small)
        self.child_chunker = LateChunker(
            embedding_model="nomic-ai/nomic-embed-text-v1.5", 
            chunk_size=128,
            trust_remote_code=True,
            min_characters_per_chunk=50
        )
        
        # 2. PARENT SPLITTER: Genera i blocchi di testo da passare all'LLM (Big)
        # 1500 caratteri sono perfetti per l'LLM: abbastanza grandi per il contesto, non troppo da saturare la VRAM
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,   
            chunk_overlap=200,  
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_markdown(self, markdown_text: str, title: str = "", date_str: str = "") -> list:
        # Pulizia base
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()
        
        # Header semantico
        header_date = f"Data di pubblicazione/riferimento: {date_str}\n" if date_str else ""
        header = f"Titolo Pagina/Documento: {title}\n{header_date}\n"
        
        # 1. Creiamo i PARENT CHUNKS
        parent_docs = self.parent_splitter.split_text(markdown_text)
        
        all_expanded_chunks = []
        seen_texts = set() # Per deduplicazione immediata
        
        for p_idx, parent_text in enumerate(parent_docs):
            full_parent_text = header + parent_text
            full_parent_text = re.sub(r'\n{3,}', '\n\n', full_parent_text).strip()
            
            # Generiamo un ID univoco per questo Padre (es. Titolo_0, Titolo_1)
            safe_title = re.sub(r'[^a-zA-Z0-9]', '', title)[:15]
            parent_id = f"{safe_title}_{p_idx}"
            
            # 2. Creiamo i CHILD CHUNKS a partire da questo specifico Padre
            small_chunks = self.child_chunker.chunk(full_parent_text)
            
            for child in small_chunks:
                # MAGIA: Abbiniamo l'embedding del Figlio al testo del Padre, e salviamo il parent_id!
                all_expanded_chunks.append({
                    "text": full_parent_text, # TESTO BIG
                    "embedding": child.embedding, # VETTORE SMALL
                    "parent_id": parent_id # ID UNIVOCO PER DEDUPLICAZIONE
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_expanded_chunks