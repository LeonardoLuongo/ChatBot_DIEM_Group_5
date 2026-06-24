import math
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider
from WebBasedRAG.RAGComponents.session_state import GlobalSessionState

class MathInput(BaseModel):
    expression: str = Field(
        ..., 
        description=(
            "The mathematical expression to evaluate. "
            "Examples: '(28 * 6 + 25 * 9) / 15', '100 * 11 / 3 + 2', 'min(28, 30)', 'max(voti)'. "
            "Use standard Python operators (+, -, *, /, **)."
        )
    )

class MathToolProvider(ToolProvider):
    """
    Fornisce un tool per calcoli matematici sicuri.
    Risolve il problema delle allucinazioni numeriche dei LLM e salva il risultato nel contesto.
    """
    def __init__(self, session_state: GlobalSessionState):
        self.session_state = session_state

    def get_tools(self) -> List[StructuredTool]:
        
        def evaluate_math(expression: str) -> str:
            # 1. FIX SINTASSI: I LLM spesso usano ^ per le potenze invece di **
            safe_expr = expression.replace("^", "**")

            # 2. AMBIENTE SICURO MA COMPLETO
            # Creiamo un ambiente vuoto per bloccare import os, sys, ecc.
            safe_env = {"__builtins__": {}}
            
            # Concediamo l'accesso a tutta la libreria math (es. math.sqrt, math.pi)
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            
            # AGGIUNTA FONDAMENTALE: Reintegriamo le funzioni native matematiche
            allowed_names.update({
                "min": min,
                "max": max,
                "abs": abs,
                "sum": sum,
                "round": round
            })
            
            try:
                # Eseguiamo l'espressione in isolamento
                result = eval(safe_expr, safe_env, allowed_names)
                
                # Arrotondiamo a 4 cifre decimali per pulizia
                if isinstance(result, float):
                    result = round(result, 4)
                    
                # 3. SALVATAGGIO NEL GLOBAL SESSION STATE
                doc_id = self.session_state._doc_counter + 1
                title = "Calcolatrice Matematica di Sistema"
                
                testo_da_salvare = (
                    f"Titolo del Documento: {title}\n"
                    f"Contesto: Questo è il risultato esatto del calcolo matematico richiesto.\n\n"
                    f"Espressione Originale: {expression}\n"
                    f"Risultato Esatto: {result}"
                )
                
                self.session_state.add_source(
                    url="local://calculator",
                    title=title,
                    testo_estratto=f"[Doc {doc_id}]\n{testo_da_salvare}",
                    score=1.0
                )
                    
                return f"✅ Calcolo eseguito con successo. Risultato: {result}. Dati salvati nel DB come [Doc {doc_id}]."
                
            except Exception as e:
                return (
                    f"⚠️ Errore di sintassi matematica: {e}. "
                    "Assicurati di passare solo numeri e operatori validi Python (es. usa ** per le potenze, e min/max/abs)."
                )

        math_tool = StructuredTool.from_function(
            func=evaluate_math,
            name="Calculator",
            description=(
                "Evaluates a mathematical expression and returns the exact numerical result. "
                "Use this tool EVERY TIME you need to perform math, calculate averages, percentages, "
                "or compute a graduation grade (voto di laurea). DO NOT do math in your head."
            ),
            args_schema=MathInput
        )
        
        return [math_tool]