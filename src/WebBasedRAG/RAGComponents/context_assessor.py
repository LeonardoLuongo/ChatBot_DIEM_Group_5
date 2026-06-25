import os
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from enum import Enum

# IMPORTIAMO IL PARSER ROBUSTO
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

import os
if os.getenv("IS_LANGFUSE_ENABLED"):
    from analytics.langfuse_dashboard import init_langfuse
    langfuse_handler = init_langfuse()


class AssessorDecision(BaseModel):
    is_sufficient: bool = Field(description="True se il contesto in memoria ha GIA' TUTTE le informazioni per rispondere, False altrimenti.")
    reasoning: str = Field(description="Spiega perché il contesto basta, oppure specifica cosa manca per indirizzare la ricerca.")

class ContextAssessor:
    """Il 'Casello': un valutatore super veloce per il Bypass del Retrieval."""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(AssessorDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fast Pre-Retrieval Assessor for an Agentic RAG system.
Read the user's NEW QUERY and the AVAILABLE CONTEXT saved from previous turns. 
Does the context ALREADY contain sufficient, explicit, and accurate information to answer the query exhaustively?

RULES:
- Return True ONLY if you are 100% sure you can generate a complete answer without searching anything else.
- If the context is only partially related, outdated, or missing the needed data, return False."""),
            ("human", "NEW QUERY: {query}\n\nAVAILABLE CONTEXT:\n{context}")
        ])
        self.chain = self.prompt | self.structured_llm

    def evaluate_sufficiency(self, query: str, context: str) -> AssessorDecision:
        print("   🔍 [CASELLO] Valutazione pre-retrieval in corso...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
        try:
            return self.chain.invoke({"query": query, "context": context}, config={"callbacks": callbacks})
        except Exception as e:
            print(f"   ⚠️ [CASELLO WARN] Errore di parsing: {e}")
            return AssessorDecision(is_sufficient=False, reasoning="Fallimento parsing pre-retrieval.")
