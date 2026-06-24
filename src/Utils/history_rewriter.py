from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.1"
SPACY_MODEL_CANDIDATES = ("it_core_news_lg", "it_core_news_md", "it_core_news_sm")
DEFAULT_HISTORY_LIMIT = 6

FOLLOWUP_MARKERS = {
    "lui", "lei", "loro", "questo", "questa", "questi", "queste", "quello", "quella",
    "quelli", "quelle", "suo", "sua", "suoi", "sue", "tale", "suddetto", "suddetta",
    "precedente", "prima", "stesso", "stessa", "altro", "altra", "altri", "altre",
}
ELLIPTIC_PREFIXES = ("e ", "e invece", "invece", "anche", "poi", "allora")
PERSONAL_ATTRIBUTE_TERMS = {
    "orario", "ricevimento", "ufficio", "email", "telefono", "lezioni", "corsi",
    "insegnamenti", "materie", "insegna", "insegnare",
}
QUESTION_WORDS = {"Chi", "Che", "Cosa", "Come", "Quando", "Dove", "Qual", "Quale", "Quali", "Quanto", "Quanti"}
LIGHT_STOPWORDS = {
    "anche", "cosa", "come", "con", "del", "della", "delle", "dello", "degli", "dei",
    "dove", "invece", "prof", "professor", "professore", "professoressa", "quale", "quali",
    "quello", "quella", "quelli", "quelle", "questo", "questa", "questi", "queste", "suo",
    "sua", "suoi", "sue", "tiene", "trova",
}


# Classe Pydantic per forzare il JSON Output perfetto da Gemini
class RewriteOutput(BaseModel):
    final_query: str = Field(description="The complete natural-language question in Italian.")
    uses_history: bool = Field(description="True if history was used to resolve the query, False otherwise.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    rationale: str = Field(description="Brief explanation of the reasoning.")

@dataclass
class ConversationTurn:
    """Turno salvato in memoria.

    focus_entities rappresenta l'entit? centrale emersa nel turno, anche se compare
    nella risposta del chatbot e non nella domanda dell'utente.
    """

    user: str
    assistant: str = ""
    source: str = ""
    final_query: str = ""
    timestamp: str = ""
    focus_entities: list[str] | None = None
    focus_entity_type: str = ""

    def __post_init__(self) -> None:
        self.focus_entities = self.focus_entities or []


@dataclass
class HistoryReferenceAnalysis:
    """Segnali linguistici passati a Ollama per guidare la riscrittura."""

    selected_turn_index: int | None
    selected_turn_user: str = ""
    latest_turn_index: int | None = None
    latest_turn_user: str = ""
    preferred_subject_entities: list[str] | None = None
    attribute_turn_index: int | None = None
    attribute_turn_user: str = ""
    current_entities: list[str] | None = None
    current_keywords: list[str] | None = None
    has_new_explicit_entity: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        self.preferred_subject_entities = self.preferred_subject_entities or []
        self.current_entities = self.current_entities or []
        self.current_keywords = self.current_keywords or []


@dataclass
class HistoryRewriteResult:
    """Output stabile del rewriter."""

    original_query: str
    final_query: str
    needs_history: bool
    uses_history: bool
    confidence: float
    rationale: str = ""


def load_history(path: Path | None, limit: int = DEFAULT_HISTORY_LIMIT) -> list[ConversationTurn]:
    """Carica gli ultimi turni da JSON, {turns: [...]}, singolo JSON o JSONL."""
    if not path or not path.exists():
        return []
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    try:
        turns = _turns_from_json_payload(json.loads(raw))
    except json.JSONDecodeError:
        turns = _load_jsonl_history(raw)
    return [turn for turn in turns if turn.user.strip()][-limit:]


def _turns_from_json_payload(payload: Any) -> list[ConversationTurn]:
    """Normalizza i formati JSON supportati in ConversationTurn."""
    if isinstance(payload, dict) and "turns" in payload:
        payload = payload["turns"]
    if isinstance(payload, dict):
        return [_turn_from_dict(payload)]
    if isinstance(payload, list):
        return [_turn_from_dict(row) for row in payload if isinstance(row, dict)]
    return []


def _turn_from_dict(row: dict[str, Any]) -> ConversationTurn:
    """Crea un turno leggendo sia i campi nuovi sia alias legacy."""
    focus = row.get("focus_entities", [])
    return ConversationTurn(
        user=str(row.get("user") or row.get("query") or row.get("human") or row.get("content") or ""),
        assistant=str(row.get("assistant") or row.get("answer") or row.get("ai") or ""),
        source=str(row.get("source") or row.get("primary_source_url") or ""),
        final_query=str(row.get("final_query") or row.get("rewritten_query") or ""),
        timestamp=str(row.get("timestamp") or ""),
        focus_entities=[str(item) for item in focus if str(item).strip()] if isinstance(focus, list) else [],
        focus_entity_type=str(row.get("focus_entity_type") or ""),
    )


def _load_jsonl_history(raw: str) -> list[ConversationTurn]:
    """Legge JSONL. Supporta sia righe ConversationTurn sia righe role/content."""
    turns: list[ConversationTurn] = []
    pending_user = ""
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        role = str(row.get("role") or "").lower()
        content = str(row.get("content") or "")
        if role == "user":
            if pending_user:
                turns.append(ConversationTurn(user=pending_user))
            pending_user = content
        elif role in {"assistant", "ai"} and pending_user:
            turns.append(ConversationTurn(user=pending_user, assistant=content))
            pending_user = ""
        else:
            turns.append(_turn_from_dict(row))
    if pending_user:
        turns.append(ConversationTurn(user=pending_user))
    return turns


def load_spacy_model(enabled: bool = True) -> Any | None:
    """Carica spaCy se possibile; se manca, il modulo usa fallback regex."""
    if not enabled:
        return None
    try:
        import spacy
    except ImportError:
        return None
    for model_name in SPACY_MODEL_CANDIDATES:
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    try:
        return spacy.blank("it")
    except Exception:
        return None


def linguistic_features(text: str, nlp: Any | None = None) -> dict[str, list[str]]:
    """Estrae entit?/nomi propri e parole informative da testo libero."""
    if nlp is None:
        return _regex_linguistic_features(text)
    doc = nlp(text)
    entities = [ent.text.strip() for ent in getattr(doc, "ents", []) if _is_valid_entity(ent.text.strip())]
    proper_nouns = [token.text for token in doc if getattr(token, "pos_", "") == "PROPN"]
    keywords = []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            value = (token.lemma_ or token.text).lower()
            if len(value) >= 3 and value not in LIGHT_STOPWORDS:
                keywords.append(value)
    fallback = _regex_linguistic_features(text)
    return {
        "entities": _unique(entities or fallback["entities"]),
        "proper_nouns": _unique(proper_nouns or fallback["proper_nouns"]),
        "keywords": _unique(keywords or fallback["keywords"]),
    }


def _regex_linguistic_features(text: str) -> dict[str, list[str]]:
    """Fallback leggero quando spaCy o il modello italiano non sono disponibili."""
    words = re.findall(r"[A-Z?-?][a-z?-?]+|[a-z?-?]{3,}", text)
    entities = [word for word in words if _is_valid_entity(word)]
    keywords = [word.lower() for word in words if word.lower() not in LIGHT_STOPWORDS]
    return {"entities": _unique(entities), "proper_nouns": _unique(entities), "keywords": _unique(keywords)}


def _is_valid_entity(value: str) -> bool:
    """Evita falsi positivi banali come 'Qual' o parole minuscole."""
    return bool(value) and value not in QUESTION_WORDS and not value[:1].islower() and len(value) >= 3


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def has_new_entity_pattern(query: str, features: dict[str, list[str]]) -> bool:
    """True se la query corrente nomina esplicitamente una nuova entit?."""
    if features["entities"] or features["proper_nouns"]:
        return True
    return bool(re.search(r"\b(?:di|del|della|prof(?:essor)?\.?)\s+[A-Z?-?][a-z?-?]+", query))


def likely_needs_history(query: str, history: list[ConversationTurn], nlp: Any | None = None) -> bool:
    """Splitter prudente: decide se serve history prima di chiamare Ollama."""
    if not history:
        return False
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    tokens = set(re.findall(r"\w+", normalized))
    if tokens.intersection(FOLLOWUP_MARKERS):
        return True
    if any(normalized.startswith(prefix) for prefix in ELLIPTIC_PREFIXES):
        return True
    if len(tokens) <= 4 and re.search(r"\b(orari?|lezioni|corsi|insegnamenti|ricevimento|email|telefono|dove|quando)\b", normalized):
        return True
    features = linguistic_features(query, nlp)
    return bool(tokens.intersection(PERSONAL_ATTRIBUTE_TERMS)) and not has_new_entity_pattern(query, features)


def extract_focus_entities(*, user: str, final_query: str, assistant: str, nlp: Any | None = None, max_entities: int = 3) -> list[str]:
    """Estrae il focus del turno, privilegiando la risposta del chatbot."""
    candidates: list[str] = []
    for text in (assistant, final_query, user):
        features = linguistic_features(text, nlp)
        for entity in features["entities"] + features["proper_nouns"]:
            clean = re.sub(r"\s+", " ", entity).strip(" .,;:()[]")
            if _is_valid_entity(clean) and clean not in candidates:
                candidates.append(clean)
        if candidates:
            break
    return candidates[:max_entities]


def append_history_turn(
    path: Path,
    *,
    user: str,
    final_query: str,
    assistant: str = "",
    source: str = "",
    focus_entities: list[str] | None = None,
    focus_entity_type: str = "person",
    max_turns: int = DEFAULT_HISTORY_LIMIT,
    nlp: Any | None = None,
) -> None:
    """Salva un turno in JSONL e mantiene solo gli ultimi max_turns."""
    existing_turns = load_history(path, limit=max_turns)
    focus = focus_entities or extract_focus_entities(user=user, final_query=final_query, assistant=assistant, nlp=nlp)
    existing_turns.append(
        ConversationTurn(
            user=user,
            final_query=final_query,
            assistant=assistant,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            focus_entities=focus,
            focus_entity_type=focus_entity_type if focus else "",
        )
    )
    with path.open("w", encoding="utf-8") as file:
        for turn in existing_turns[-max_turns:]:
            file.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")


def best_attribute_turn(query: str, history: list[ConversationTurn], nlp: Any | None = None) -> int:
    """Seleziona il turno che meglio suggerisce il tipo di informazione richiesta."""
    current_keywords = set(linguistic_features(query, nlp)["keywords"])
    if not current_keywords:
        return len(history) - 1
    best_index = len(history) - 1
    best_score = -1.0
    for index, turn in enumerate(history):
        turn_text = " ".join(part for part in (turn.user, turn.final_query, turn.assistant) if part)
        turn_keywords = set(linguistic_features(turn_text, nlp)["keywords"])
        recency = index / max(1, len(history) - 1)
        score = len(current_keywords.intersection(turn_keywords)) + (0.5 * recency)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def analyze_history_reference(query: str, history: list[ConversationTurn], nlp: Any | None = None) -> HistoryReferenceAnalysis:
    """Costruisce una mini-analisi per guidare il prompt di riscrittura."""
    if not history:
        return HistoryReferenceAnalysis(selected_turn_index=None, reason="Nessuna history disponibile.")
    latest_index = len(history) - 1
    attribute_index = best_attribute_turn(query, history, nlp)
    current = linguistic_features(query, nlp)
    return HistoryReferenceAnalysis(
        selected_turn_index=latest_index,
        selected_turn_user=history[latest_index].user,
        latest_turn_index=latest_index,
        latest_turn_user=history[latest_index].user,
        preferred_subject_entities=history[latest_index].focus_entities or [],
        attribute_turn_index=attribute_index,
        attribute_turn_user=history[attribute_index].user,
        current_entities=current["entities"] or current["proper_nouns"],
        current_keywords=current["keywords"],
        has_new_explicit_entity=has_new_entity_pattern(query, current),
        reason="Priorit? al turno pi? recente: il focus dell'ultimo turno ? il soggetto preferito del follow-up.",
    )


def compact_history(history: list[ConversationTurn], analysis: HistoryReferenceAnalysis | None = None) -> str:
    """Prepara una history breve e leggibile per il modello."""
    if not history:
        return "Nessuna history."
    blocks = []
    for index, turn in enumerate(history, 1):
        marker = ""
        if analysis and analysis.latest_turn_index == index - 1:
            marker += " [TURNO PI? RECENTE / SOGGETTO PROBABILE]"
        if analysis and analysis.attribute_turn_index == index - 1 and analysis.attribute_turn_index != analysis.latest_turn_index:
            marker += " [TIPO INFORMAZIONE PROBABILE]"
        final_query = f"\nQuery finale usata: {turn.final_query}" if turn.final_query else ""
        focus = f"\nFocus entities ({turn.focus_entity_type}): {', '.join(turn.focus_entities)}" if turn.focus_entities else ""
        source = f"\nFonte: {turn.source}" if turn.source else ""
        assistant_excerpt = re.sub(r"\s+", " ", turn.assistant).strip()[:500]
        blocks.append(f"[{index}]{marker} Utente: {turn.user}{final_query}{focus}\nAssistente: {assistant_excerpt}{source}")
    return "\n\n".join(blocks)


def reference_analysis_block(analysis: HistoryReferenceAnalysis | None) -> str:
    """Serializza l'analisi in testo semplice per Ollama."""
    if not analysis or analysis.selected_turn_index is None:
        return "Nessun turno specifico selezionato."
    return (
        f"Most recent turn, preferred for subject resolution: {analysis.latest_turn_index + 1 if analysis.latest_turn_index is not None else 'unknown'}\n"
        f"Most recent user question: {analysis.latest_turn_user or 'unknown'}\n"
        f"Preferred subject entities from most recent turn: {', '.join(analysis.preferred_subject_entities) or 'none'}\n"
        f"Probable attribute/information-type turn: {analysis.attribute_turn_index + 1 if analysis.attribute_turn_index is not None else 'unknown'}\n"
        f"Attribute source question: {analysis.attribute_turn_user or 'unknown'}\n"
        f"Current explicit entities: {', '.join(analysis.current_entities) or 'none'}\n"
        f"Current informative words: {', '.join(analysis.current_keywords) or 'none'}\n"
        f"Current query has a new explicit entity: {analysis.has_new_explicit_entity}\n"
        f"Reason: {analysis.reason}"
    )


def build_rewrite_prompts(query: str, history: list[ConversationTurn], analysis: HistoryReferenceAnalysis) -> tuple[str, str]:
    """Crea prompt system/user per il modello di riscrittura."""
    system_prompt = """You are a history-aware query rewriting module for a university RAG chatbot.
Your only task is to transform a follow-up user question into a standalone query.

Rules:
- Reply ONLY with valid JSON.
- Do not answer the user's question.
- Use conversation history only to resolve implicit references.
- Recency is very important: if the current query is ambiguous, resolve the subject from the most recent relevant history turn.
- If the most recent turn has Focus entities, use them as the preferred subject, even if the entity appeared only in the assistant answer.
- Older turns may help infer the requested attribute or information type, but they must not override the subject introduced in the latest turn.
- If the current query contains a new explicit entity/person/course, keep that entity.
- A new explicit entity may be an object/course, not necessarily the subject.
- Preserve the logical form of the user question. If the user asks a yes/no question, keep it as a yes/no question.
- The final_query MUST be a complete natural-language question in Italian, not a keyword list.
- Do not invent names, dates, courses, departments, or details that are not present in the user query or history.
- If the reference cannot be resolved safely, keep the original query and set uses_history=false.

Examples:
- User: "e che altre materie insegna?"; latest focus entity: "Luca Greco"
  final_query: "Quali altre materie insegna il professor Luca Greco?"
- User: "Embedded ? insegnato da lui?"; latest focus entity: "Gennaro Percannella"
  final_query: "Il corso Embedded ? insegnato dal professor Gennaro Percannella?"

Schema:
{
  "final_query": "string",
  "uses_history": true,
  "confidence": 0.0,
  "rationale": "breve spiegazione"
}
"""
    user_prompt = f"""HISTORY RECENTE
{compact_history(history, analysis)}

LIGHTWEIGHT LINGUISTIC ANALYSIS
{reference_analysis_block(analysis)}

QUERY UTENTE
{query}

Produci il JSON."""
    return system_prompt, user_prompt


def call_ollama_json(*, ollama_url: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    """Chiama Ollama chiedendo un JSON valido."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
        "format": "json",
        "keep_alive": 0
    }
    request = Request(ollama_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Impossibile contattare Ollama su {ollama_url}.") from exc
    content = ((body.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Ollama non ha restituito testo utile: {body}")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Risposta non JSON dal rewriter: {content}")
        return json.loads(match.group(0))


def rewrite_with_history(
    query: str,
    history: list[ConversationTurn],
    *,
    model: str = "gemini-2.5-flash-lite", # <--- Default cambiato a Gemini
    temperature: float = 0.0,
    use_spacy: bool = True,
) -> HistoryRewriteResult:
    """Riscrive solo query dipendenti dalla history usando Google Gemini tramite LangChain."""
    
    nlp = load_spacy_model(use_spacy)
    if not likely_needs_history(query, history, nlp):
        return HistoryRewriteResult(query, query, False, False, 1.0, "Query considerata autonoma: nessuna riscrittura applicata.")
    
    analysis = analyze_history_reference(query, history, nlp)
    system_prompt_text, user_prompt_text = build_rewrite_prompts(query, history, analysis)
    
    # 1. Inizializza Gemini
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    # 2. Forza l'output JSON strutturato
    structured_llm = llm.with_structured_output(RewriteOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", user_prompt_text)
    ])
    
    chain = prompt | structured_llm
    
    try:
        # 3. Invoca la chain (Gestisce autonomamente connessione, retry e parsing JSON)
        payload: RewriteOutput = chain.invoke({})
        
        final_query = payload.final_query.strip() if payload.final_query else query
        confidence = max(0.0, min(1.0, float(payload.confidence)))
        
        return HistoryRewriteResult(
            original_query=query,
            final_query=final_query,
            needs_history=True,
            uses_history=bool(payload.uses_history or final_query != query),
            confidence=confidence,
            rationale=payload.rationale.strip()
        )
        
    except Exception as e:
        print(f"⚠️ [History Rewriter] Errore API Gemini: {e}. Fallback alla query originale.")
        return HistoryRewriteResult(query, query, False, False, 0.0, f"Errore API: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test isolato: riscrive solo query dipendenti dalla history.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--save-turn", action="store_true", help="Salva il turno nel file history JSONL dopo la riscrittura.")
    parser.add_argument("--assistant", default="", help="Risposta assistant da salvare nel turno, se disponibile.")
    parser.add_argument("--source", default="", help="Fonte principale da salvare nel turno, se disponibile.")
    parser.add_argument("--focus-entity", action="append", default=[], help="Entit? centrale da salvare nel turno; ripetibile.")
    parser.add_argument("--focus-entity-type", default="person")
    parser.add_argument("--max-history-turns", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-spacy", action="store_true", help="Disattiva spaCy/fallback linguistico.")
    parser.add_argument("--show-history-analysis", action="store_true", help="Stampa anche l'analisi del riferimento alla history.")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history_file, limit=args.history_limit)
    nlp = load_spacy_model(not args.no_spacy)
    result = rewrite_with_history(args.query, history, ollama_url=args.ollama_url, model=args.ollama_model, temperature=args.temperature, use_spacy=not args.no_spacy)
    if args.show_history_analysis:
        print(json.dumps(asdict(analyze_history_reference(args.query, history, nlp)), ensure_ascii=False, indent=2 if args.pretty else None))
    if args.save_turn:
        if not args.history_file:
            raise RuntimeError("--save-turn richiede --history-file")
        append_history_turn(
            args.history_file,
            user=args.query,
            final_query=result.final_query,
            assistant=args.assistant,
            source=args.source,
            focus_entities=args.focus_entity or None,
            focus_entity_type=args.focus_entity_type,
            max_turns=args.max_history_turns,
            nlp=nlp,
        )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
