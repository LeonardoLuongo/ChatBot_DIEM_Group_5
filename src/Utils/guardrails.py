from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_SOURCE_DOMAINS = {
    "www.diem.unisa.it",
    "diem.unisa.it",
    "corsi.unisa.it",
    "docenti.unisa.it",
    "rubrica.unisa.it",
    "www.unisa.it",
    "web.unisa.it",
}

OFFENSIVE_TERMS = {
    # Lista volutamente piccola e trasparente: blocca insulti espliciti senza
    # trasformare il guardrail in un classificatore opaco.
    "coglione",
    "porco dio",
    "porca madonna",
    "dio cane",
    "minchione",
    "ricchione",
    "frocio",
    "coglioni",
    "stronzo",
    "stronza",
    "stronzi",
    "merda",
    "vaffanculo",
    "fanculo",
    "troia",
    "puttana",
    "idiot",
    "goddamn",
    "holy shit",
    "goddamn",
    "idiot",
    "faggot",
    "asshole",
    "bitch",
    "assholes",
    "shit",
    "fuck you",
    "fuck",
    "whore",
    "whore"
}

SENSITIVE_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE),
    "password": re.compile(r"\b(?:password|passwd|pwd|token|api[_ -]?key|secret)\s*[:=]\s*\S+", re.IGNORECASE),
}

OFFENSIVE_REFUSALS = [
    "Posso aiutarti, ma riformula la richiesta evitando linguaggio offensivo.\n",
    "Riformula la domanda in modo rispettoso e ti aiuto volentieri.\n",
    "Con quella bocca baci tua madre? sei pregato di rivolgerti in maniera rispettosa.\n"
    "Per continuare, ho bisogno che la richiesta sia espressa senza insulti o linguaggio aggressivo.\n",
    "Ti aiuto senza problemi, ma riscrivi la richiesta usando un tono rispettoso.\n",
    "La richiesta è stata bloccata per linguaggio offensivo: riformulala in modo neutro e riproviamo.\n",
]

SENSITIVE_REFUSALS = [
    "Non posso trattare password, token, carte di pagamento o IBAN. Rimuovi questi dati e riprova.\n",
    "Per sicurezza non posso elaborare password, token, carte o IBAN. Toglili dalla richiesta e riproviamo.\n",
    "La richiesta contiene dati sensibili come credenziali o dati finanziari: rimuovili e posso continuare.\n",
]



@dataclass(slots=True)
class GuardrailResult:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)


class GuardrailLogger:
    """Scrive eventi guardrail in JSONL senza salvare testo sensibile in chiaro."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path

    def log(self, *, stage: str, action: str, reasons: list[str], text: str) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "action": action,
            "reasons": reasons,
            "text_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def detect_offensive_text(text: str) -> list[str]:
    """Rileva linguaggio offensivo esplicito con regole deterministiche."""
    normalized = text.lower()
    found = []
    for term in OFFENSIVE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            found.append(term)
    return sorted(found)


def detect_sensitive_private_data(text: str) -> list[str]:
    """Rileva solo dati altamente sensibili: password/token, carte, IBAN."""
    reasons = []
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text or ""):
            reasons.append(name)
    return reasons


def validate_input(query: str, logger: GuardrailLogger | None = None) -> GuardrailResult:
    """Blocca input offensivi o contenenti segreti/dati finanziari privati."""
    reasons = []
    offensive_terms = detect_offensive_text(query)
    if offensive_terms:
        reasons.append("offensive_language")
    sensitive = detect_sensitive_private_data(query)
    reasons.extend(f"sensitive_{item}" for item in sensitive)

    allowed = not reasons
    if logger:
        logger.log(stage="input", action="allow" if allowed else "block", reasons=reasons, text=query)
    return GuardrailResult(allowed=allowed, text=query, reasons=reasons)


def validate_sources(results: list[object], logger: GuardrailLogger | None = None) -> GuardrailResult:
    """Verifica che i risultati recuperati provengano solo da domini istituzionali ammessi."""
    blocked = []
    for result in results:
        url = getattr(result, "url", "") or ""
        parent_url = getattr(result, "parent_url", "") or ""
        for candidate in (url, parent_url):
            if not candidate or candidate.startswith("file://"):
                continue
            host = urlparse(candidate).netloc.lower()
            if host and host not in ALLOWED_SOURCE_DOMAINS:
                blocked.append(candidate)
    allowed = not blocked
    reasons = ["source_domain_not_allowed"] if blocked else []
    if logger:
        logger.log(stage="retrieval", action="allow" if allowed else "block", reasons=reasons, text="\n".join(blocked))
    return GuardrailResult(allowed=allowed, text="", reasons=reasons)


def validate_output(answer: str, has_sources: bool, logger: GuardrailLogger | None = None) -> GuardrailResult:
    """Controlla che l'output finale non violi i vincoli minimi del progetto."""
    reasons = []
    offensive_terms = detect_offensive_text(answer)
    if offensive_terms:
        reasons.append("offensive_language")
    sensitive = detect_sensitive_private_data(answer)
    reasons.extend(f"sensitive_{item}" for item in sensitive)
    if not has_sources:
        reasons.append("missing_source")

    allowed = not reasons
    if logger:
        logger.log(stage="output", action="allow" if allowed else "block", reasons=reasons, text=answer)
    return GuardrailResult(allowed=allowed, text=answer, reasons=reasons)


def refusal_message(reasons: list[str]) -> str:
    """Messaggio utente quando un guardrail blocca la richiesta o la risposta."""
    if any(reason.startswith("sensitive_") for reason in reasons):
        return random.choice(SENSITIVE_REFUSALS)
    if "offensive_language" in reasons:
        return random.choice(OFFENSIVE_REFUSALS)
    if "source_domain_not_allowed" in reasons:
        return "Ho recuperato fonti fuori dai domini DIEM/UNISA ammessi, quindi non posso usare quei risultati."
    if "missing_source" in reasons:
        return "Non ho trovato informazioni sufficienti nelle fonti DIEM/UNISA disponibili."
    return "Non posso completare la risposta perché viola i guardrails configurati."
