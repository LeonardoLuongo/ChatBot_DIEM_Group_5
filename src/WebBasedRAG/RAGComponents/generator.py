# ================================================
# FILE: RAGComponents/generator.py
# ================================================
import re
import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException

if os.getenv("IS_LANGFUSE_ENABLED"):
    from analytics.langfuse_dashboard import init_langfuse
    langfuse_handler = init_langfuse()

class AnswerWithSources(BaseModel):
    thought_process: str = Field(
        description="CRITICAL: Use this field as a scratchpad. Before writing the answer, read the context and explicitly list ALL the items/courses/rules requested by the user. Do not skip any."
    )
    answer: str = Field(
        description="La risposta discorsiva, chiara, ESTREMAMENTE COMPLETA ed ESAUSTIVA per l'utente."
    )
    sources_used: List[int] = Field(
        description="Lista degli ID numerici dei documenti usati. Vuoto [] se non c'è risposta."
    )

class AnswerGenerator:
    """Gestisce ESCLUSIVAMENTE la fase di generazione RAG e la correzione degli errori."""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # Inizializziamo il parser strutturato nativo
        self.structured_llm = self.llm.with_structured_output(AnswerWithSources)
        
        self.system_prompt = """You are an agent for the DIEM (Department of Information, Electrical, and Applied Mathematics Engineering) 
departement of the university of UNISA (UNiversity of Salerno).
You must answer and provide information about this department and its faculties, nothing else, even the other Engineering Department are out of your scope.

DOMAIN KNOWLEDGE - ACADEMIC CALENDAR:
- University classes are strictly held from Monday to Friday.
- There are NO classes scheduled on Saturdays and Sundays. If a user asks for a schedule on a weekend, immediately inform them that no classes are held on weekends.

ANTI-LAZINESS & EXHAUSTIVENESS RULES (CRITICAL):
1. DO NOT SUMMARIZE. If the context contains a list of courses, requirements, or schedules, you MUST extract and list EVERY SINGLE ITEM.
2. NEVER stop halfway through a list. Be absolutely exhaustive.
3. NEVER write "there are no other items" or "eccetera" if the context contains more data.

CRITICAL MAPPING & CITATION RULES:
1. Every document in the context starts strictly with a tag like '[Doc X]'. 
2. Whenever you extract a fact from a document, you MUST insert the literal tag '[Doc X]' inside your text answer at the end of the relevant sentence (e.g., "...as scheduled on Monday [Doc 6].").
3. For EVERY '[Doc X]' tag you write in your text, you MUST append the exact integer X into the `sources_used` array.
4. The IDs in your text and the integers in the `sources_used` list MUST match perfectly.
5. If a document ID is not mentioned in your text, DO NOT include it in `sources_used`. DO NOT hallucinate or invent document IDs.
6. Always respond in the exact same language as the user's `query`.
7. EXTERNAL LINKS: If the context text contains external link markers like '[L1]', '[L2]', etc., you MUST preserve them. If you extract information attached to a link, copy the exact '[Lx]' tag in your text answer.
8. If the context does not contain enough data to answer, state clearly what is missing and leave the `sources_used` list empty [].

CRITICAL OUTPUT RULE: 
You MUST output ONLY a valid JSON object. 
DO NOT print any text, preamble, or markdown (like ```json) outside of the JSON block. 
If you need to think or use a scratchpad, you MUST do it STRICTLY INSIDE the "thought_process" JSON field."""

    def _defensive_parsing(self, error: Exception) -> AnswerWithSources:
        """Metodo di emergenza per estrarre il JSON se il modello sbava testo (Preamble Leakage)"""
        print("   ⚠️ [GENERATOR WARN] Errore di parsing nativo. Attivazione Regex Fallback...")
        
        # Recupera il testo grezzo generato dall'LLM dall'eccezione LangChain
        raw_output = getattr(error, 'llm_output', str(error))
        
        try:
            # Cerca tutto ciò che è compreso tra { e } includendo ritorni a capo
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed_dict = json.loads(json_str)
                print("   🩹 [GENERATOR FIX] JSON estratto e recuperato con successo via Regex!")
                return AnswerWithSources(**parsed_dict)
            else:
                raise ValueError("Nessun pattern JSON identificato nel testo grezzo.")
                
        except Exception as e:
            print(f"   🚨 [GENERATOR CRITICAL] Fallito anche il Regex Fallback: {e}")
            # Ritorna un oggetto sicuro per evitare il crash dell'app Gradio
            return AnswerWithSources(
                thought_process="[CRASH PREVENTED] LLM failed to format output.",
                answer="Si è verificato un errore tecnico nella generazione della risposta. Riprova.",
                sources_used=[]
            )

    def generate(self, query: str, context: str) -> AnswerWithSources:
        print(" ✍️ [GENERATOR] Contesto ricevuto. Generazione risposta formattata...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "CONTEXT:\n{context}\n\nQUERY: {query}\n\nRemember: Be 100% exhaustive. Extract all items.")
        ])
        
        rag_chain = prompt | self.structured_llm
        
        try:
            result = rag_chain.invoke({"context": context, "query": query}, config={"callbacks": callbacks})
        except Exception as e:
            # Catturiamo l'OutputParserException (e derivati) e attiviamo la difesa
            result = self._defensive_parsing(e)
        
        # Stampiamo solo una porzione del thought_process per non inquinare i log
        if result and hasattr(result, 'thought_process'):
            thought_preview = result.thought_process.replace('\n', ' ')[:150]
            print(f"   🧠 [GENERATOR THOUGHTS]: {thought_preview}...\n")
            
        return self._post_parse_sources(result)

    def revise(self, query: str, context: str, judge_feedback: str) -> AnswerWithSources:
        print(" ✍️ [GENERATOR] Ricevuta critica dal giudice. Riscrittura in corso...")
        callbacks = [langfuse_handler] if os.getenv("IS_LANGFUSE_ENABLED") else []
    
        revise_system_prompt = f"""{self.system_prompt}
        
In this setting, you are revising a previously rejected answer. The auditor gave you feedback in the 'feedback' section. 
Use the feedback as a strict compass to fix the answer. Do NOT violate tone rules. Ensure strict citation matching."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", revise_system_prompt),
            ("human", "CONTEXT:\n{context}\n\nQUERY: {query}\n\n"
                      "AUDITOR'S FEEDBACK:{feedback}\n\n"
                      "Generate the REVISED answer. Be exhaustive.")
        ])
        
        rag_chain = prompt | self.structured_llm
        
        try:
            result = rag_chain.invoke({"context": context, "query": query, "feedback": judge_feedback}, config={"callbacks": callbacks})
        except Exception as e:
            # Difesa anche per la fase di revisione
            result = self._defensive_parsing(e)
        
        return self._post_parse_sources(result)

    def _post_parse_sources(self, result: AnswerWithSources) -> AnswerWithSources:
        """Post-Parser Difensivo per allineare gli ID nel testo con l'array sources_used."""
        if result and result.answer:
            extracted_ids = [int(num) for num in re.findall(r"\[Doc\s*(\d+)\]", result.answer)]
            if extracted_ids:
                updated_sources = list(set(result.sources_used + extracted_ids))
                result = result.model_copy(update={"sources_used": updated_sources})
                print(f"   [POST-PARSER] Allineamento fonti completato. ID rilevati nel testo: {extracted_ids}")
        return result