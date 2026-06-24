from __future__ import annotations

import re
from dataclasses import dataclass


def clean_markdown(text: str) -> str:
    """Normalizza spazi e ritorni a capo mantenendo la struttura markdown."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


@dataclass(slots=True)
class ChunkRecord:
    """Rappresenta un chunk pronto per essere esportato."""

    chunk_id: str
    page_id: str
    url: str
    page_title: str
    text: str
    chunk_index: int
    token_estimate: int
    depth: int
    content_type: str
    hash_code: str
    source_type: str = "html"
    parent_url: str | None = None
    pdf_page: int | None = None
    text_format: str | None = None
    headings: dict[str, str] | None = None


class MarkdownChunker:
    """Divide markdown in parent chunk ampi; i child vengono creati nel vector store."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 250) -> None:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Titolo_1"),
                ("##", "Titolo_2"),
                ("###", "Titolo_3"),
                ("####", "Titolo_4"),
            ],
            strip_headers=False,
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, markdown_text: str) -> list[tuple[str, dict[str, str]]]:
        """Restituisce coppie testo+headings già pronte per il retrieval."""
        markdown_text = clean_markdown(markdown_text)
        if not markdown_text:
            return []
        markdown_documents = self.markdown_splitter.split_text(markdown_text)
        final_documents = self.fallback_splitter.split_documents(markdown_documents)
        chunks = []
        for document in final_documents:
            text = clean_markdown(document.page_content)
            metadata = {str(key): str(value) for key, value in (document.metadata or {}).items() if value}
            if not text:
                continue
            context = " > ".join(metadata.values())
            enriched_text = f"CONTESTO: [{context}]\n{text}" if context else text
            if len(enriched_text.strip()) > 50:
                chunks.append((f"passage: {enriched_text}", metadata))
        return chunks
