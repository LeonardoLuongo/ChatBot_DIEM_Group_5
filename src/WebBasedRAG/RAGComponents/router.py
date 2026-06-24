# ================================================
# FILE: RAGComponents/router.py
# ================================================
import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


if os.getenv("IS_LANGFUSE_ENABLED"):
    from analytics.langfuse_dashboard import init_langfuse
    langfuse_handler = init_langfuse()

class RouteDecision(BaseModel):
    intent: Literal["DOMAIN_SEARCH", "DIRECT_CHAT"] = Field(
        description="Choose DOMAIN_SEARCH if the query needs department/university info. Choose DIRECT_CHAT for greetings, general code, or out-of-domain topics."
    )
    reasoning: str = Field(description="Brief reason for the routing decision.")

class IntentRouter:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(RouteDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the primary Query Router for the DIEM (Department of Information Engineering, Electrical Engineering and Applied Mathematics) chatbot at UNISA.
Your job is to analyze the user's input and decide if it requires searching the university's knowledge base or if it can be answered directly.

ROUTING RULES:
- Choose 'DIRECT_CHAT' if the query is:
  1. A simple greeting, pleasantry, or gratitude (e.g., "Ciao", "Come stai?", "Grazie").
  2. A generic request unrelated to the university (e.g., "Scrivimi uno script in Python", "Traduci questa frase", "Parlami di sport").
  3. Clearly out of scope for the DIEM department.
- Choose 'DOMAIN_SEARCH' if the query is:
  1. About university schedules, professors, office hours, or contacts.
  2. About exams, thesis, CFU, rules, or degree programs at DIEM.
  3. Anything that might reasonably require looking up institutional or dynamic data."""),
            ("human", "USER QUERY: {query}")
        ])
        self.chain = self.prompt | self.structured_llm

    def route(self, query: str) -> RouteDecision:
        print("   🚦 [ROUTER] Analisi dell'intento in corso...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
        
        try:
            return self.chain.invoke({"query": query}, config={"callbacks": callbacks})
        except Exception as e:
            print(f"   ⚠️ [ROUTER WARN] Fallimento strutturato, bypasso e avvio RAG di default. Errore: {e}")
            return RouteDecision(intent="DOMAIN_SEARCH", reasoning="Fallback due to parsing error")