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


import re
import torch
from dataclasses import dataclass
from typing import Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Wrapper per Jina
class JinaLangChainWrapper(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, task="retrieval.passage", batch_size=16).tolist()
    def embed_query(self, text):
        return self.model.encode([text], task="retrieval.query", batch_size=1)[0].tolist()

class DocumentChunker:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        print("Caricamento Chunker (Modalità: Two-Stage Recursive + Semantic)...")
        
        # STADIO 1: Il "Garante della Memoria". Assicura che nessun blocco superi i 4000 caratteri.
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        if self.embedding_model:
            self.lc_embedder = JinaLangChainWrapper(self.embedding_model)
            # STADIO 2: Il "Rifinitore Logico". Trova il senso compiuto dentro i blocchi di 4000 char.
            self.semantic_splitter = SemanticChunker(
                self.lc_embedder, 
                breakpoint_threshold_type="percentile" 
            )
        else:
            self.semantic_splitter = self.base_splitter

    def process_markdown(self, markdown_text: str, title: str = "", date_str: str = "") -> list:
        # Pulizia base
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()
        
        header_date = f"Data di pubblicazione/riferimento: {date_str}\n" if date_str else ""
        header = f"Titolo Pagina/Documento: {title}\n{header_date}\n"
        
        all_expanded_chunks = []
        text_chunks = []
        
        # --- ADAPTIVE CHUNKING IN AZIONE ---
        
        # 1. Taglio strutturale sicuro
        safe_blocks = self.base_splitter.split_text(markdown_text)
        
        # 2. Fast-Track: Se la pagina è immensa, evitiamo la lentezza del SemanticChunker
        is_massive_page = len(markdown_text) > 15000
        
        if is_massive_page:
            # print(f"      ⚡ [FAST TRACK] Pagina enorme rilevata ({len(markdown_text)} char). Chunking semantico saltato per velocità.")
            for block in safe_blocks:
                if block.strip():
                    text_chunks.append(f"{header}{block.strip()}")
        else:
            # Taglio semantico di precisione per le pagine normali
            for block in safe_blocks:
                try:
                    semantic_docs = self.semantic_splitter.create_documents([block])
                    for doc in semantic_docs:
                        if doc.page_content.strip():
                            text_chunks.append(f"{header}{doc.page_content.strip()}")
                except Exception as e:
                    # Fallback di emergenza
                    if block.strip():
                        text_chunks.append(f"{header}{block.strip()}")
        
        if not text_chunks:
            return []
            
        # 3. Calcolo Embedding dei chunk finali in batch
        if self.embedding_model:
            embeddings_tensors = self.embedding_model.encode(text_chunks, task="retrieval.passage", batch_size=16)
        else:
            embeddings_tensors = [None] * len(text_chunks)
            
        # 4. Salvataggio
        safe_title = re.sub(r'[^a-zA-Z0-9]', '', title)[:15]
        
        for idx, (text_chunk, emb) in enumerate(zip(text_chunks, embeddings_tensors)):
            all_expanded_chunks.append({
                "text": text_chunk,             
                "embedding": emb,               
                "parent_id": f"{safe_title}_{idx}"
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_expanded_chunks