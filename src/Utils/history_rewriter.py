from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_HISTORY_LIMIT = 6

@dataclass
class ConversationTurn:
    """Turno salvato in memoria."""
    user: str
    assistant: str = ""
    source: str = ""
    final_query: str = ""
    timestamp: str = ""


@dataclass
class HistoryRewriteResult:
    """Output stabile del rewriter (mantiene totale compatibilità con AgenticRAG)."""
    original_query: str
    final_query: str
    needs_history: bool
    uses_history: bool
    confidence: float
    rationale: str = ""


class RewriteOutput(BaseModel):
    reasoning: str = Field(
        description="Analisi logica: spiega brevemente se la query attuale contiene pronomi (lui, quello), riferimenti impliciti, o se è una continuazione del discorso precedente."
    )
    needs_rewrite: bool = Field(
        description="True se la query dipende dalla cronologia e deve essere riscritta per avere senso da sola. False se è già autonoma."
    )
    final_query: str = Field(
        description="La query di ricerca completa e indipendente dal contesto (in Italiano). Se 'needs_rewrite' è False, questo campo DEVE essere identico alla query originale."
    )


def load_history(path: Path | None, limit: int = DEFAULT_HISTORY_LIMIT) -> list[ConversationTurn]:
    """Carica gli ultimi turni da JSONL."""
    if not path or not path.exists():
        return []
    
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
        
    turns: list[ConversationTurn] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            turns.append(ConversationTurn(
                user=str(row.get("user") or ""),
                assistant=str(row.get("assistant") or ""),
                source=str(row.get("source") or ""),
                final_query=str(row.get("final_query") or ""),
                timestamp=str(row.get("timestamp") or ""),
            ))
        except json.JSONDecodeError:
            continue
            
    return [turn for turn in turns if turn.user.strip()][-limit:]


def append_history_turn(
    path: Path,
    *,
    user: str,
    final_query: str,
    assistant: str = "",
    source: str = "",
    max_turns: int = DEFAULT_HISTORY_LIMIT,
    # Questi parametri sono mantenuti per non rompere il codice legacy in main.py, ma ignorati.
    focus_entities: Any = None, 
    focus_entity_type: Any = None,
    nlp: Any = None
) -> None:
    """Salva un turno in JSONL e mantiene solo gli ultimi max_turns."""
    existing_turns = load_history(path, limit=max_turns)
    
    existing_turns.append(
        ConversationTurn(
            user=user,
            final_query=final_query,
            assistant=assistant,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )
    
    with path.open("w", encoding="utf-8") as file:
        for turn in existing_turns[-max_turns:]:
            file.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")


def format_history_for_llm(history: list[ConversationTurn]) -> str:
    """Formatta la history in un formato leggibile per Gemini."""
    if not history:
        return "Nessuna cronologia precedente."
        
    blocks = []
    for i, turn in enumerate(history, 1):
        # Troncamento della risposta dell'assistente per risparmiare token (prendiamo i primi 500 caratteri)
        assistant_excerpt = turn.assistant[:500] + "..." if len(turn.assistant) > 500 else turn.assistant
        blocks.append(f"[Turno {i}]\nUtente: {turn.user}\nAssistente: {assistant_excerpt}")
    return "\n\n".join(blocks)


def rewrite_with_history(
    query: str,
    history: list[ConversationTurn],
    *,
    model: str = "gemini-2.5-flash-lite", 
    temperature: float = 0.0,
    use_spacy: bool = False # Ignorato, mantenuto per compatibilità
) -> HistoryRewriteResult:
    """
    Riscrive la query utente usando la SOTA per la 
    Conversational Query Rewriting (CQR) basata su LLM.
    """
    if not history:
        return HistoryRewriteResult(
            original_query=query, final_query=query, 
            needs_history=False, uses_history=False, 
            confidence=1.0, rationale="Nessuna cronologia disponibile."
        )

    system_prompt = """You are a Conversational Query Rewriting (CQR) AI for a university RAG system.
Your only task is to analyze the user's latest query in the context of the previous conversation history, and output a "Standalone Query".

CRITICAL RULES:
1. CORE OBJECTIVE: If the current user query contains implicit references, pronouns (he, she, it, this, that), or is a follow-up question, you MUST rewrite it so that a search engine can understand it completely without needing the conversation history.
2. PRESERVE INTENT: Do not answer the question. Only reformulate it. Keep the original language (Italian).
3. EXPLICIT RESOLUTION: Resolve entities exactly as they appeared in the history. If the assistant previously talked about "Prof. Antonio Greco" and the user asks "Where is his office?", rewrite to "Dov'è l'ufficio del professore Antonio Greco?".
4. ACTIVE CLARIFICATION HANDLING: If the assistant previously asked the user to choose an option (e.g. "Do you mean Bachelor or Master?"), and the user responds with just the option (e.g. "Bachelor"), you MUST combine the previous topic with the user's choice to form a complete query (e.g. "Regulations for the Bachelor degree").
5. STANDALONE CHECK: If the user's current query is ALREADY self-contained and does not refer to previous turns, you MUST set `needs_rewrite` to False and return the query EXACTLY as the user typed it.
6. NO HALLUCINATION: Do not add external facts or constraints that the user didn't mention."""

    human_prompt = """CONVERSATION HISTORY:
{formatted_history}

LATEST USER QUERY:
{query}

Analyze the history and provide the Standalone Query."""

    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    structured_llm = llm.with_structured_output(RewriteOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | structured_llm
    
    try:
        payload: RewriteOutput = chain.invoke({
            "formatted_history": format_history_for_llm(history),
            "query": query
        })
        
        final_query = payload.final_query.strip() if payload.needs_rewrite else query
        
        return HistoryRewriteResult(
            original_query=query,
            final_query=final_query,
            needs_history=True,
            uses_history=payload.needs_rewrite,
            confidence=1.0,
            rationale=payload.reasoning
        )
        
    except Exception as e:
        print(f"⚠️ [History Rewriter] API Error: {e}. Fallback to original query.")
        return HistoryRewriteResult(
            original_query=query, final_query=query, 
            needs_history=False, uses_history=False, 
            confidence=0.0, rationale=f"Error: {e}"
        )