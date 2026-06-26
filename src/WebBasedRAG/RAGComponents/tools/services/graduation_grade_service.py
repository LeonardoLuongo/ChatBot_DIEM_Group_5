"""
graduation_grade_service.py
============================
Servizio per il calcolo del voto di laurea dell'Università degli Studi di Salerno,
Dipartimento DIEM. Supporta le formule ufficiali per le seguenti combinazioni
di corso di laurea e coorte di immatricolazione.

Corsi supportati:
  TRIENNALI (L-8):
    - Ingegneria Informatica (L-8)
    - Ingegneria dell'Informazione per la Medicina Digitale (L-8)
    Coorte 25/26   → Formula nuova x*100/30
    Coorte ≤24/25  → Formula vecchia

  MAGISTRALI:
    - Ingegneria Informatica (LM-32)
    - Information Engineering for Digital Medicine (LM-32)
    Coorte 25/26   → Formula nuova x*100/30
    Coorte ≤24/25  → Formula vecchia

    - Electrical Engineering for Digital Energy (LM-28)
    Coorte 25/26   → Formula nuova x*100/30
    Coorte ≤24/25  → Formula vecchia
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional


# =========================================================================
# COSTANTI E MAPPE DI LOOKUP
# =========================================================================

# Identificatori interni dei corsi di laurea
COURSE_L8_INFO    = "L8_INFO"          # Ingegneria Informatica (L-8)
COURSE_L8_MED     = "L8_MED"           # Ing. Info per Medicina Digitale (L-8)
COURSE_LM32_INFO  = "LM32_INFO"        # Ingegneria Informatica (LM-32)
COURSE_LM32_MED   = "LM32_MED"        # Information Engineering Digital Medicine (LM-32)
COURSE_LM28_ENERGY= "LM28_ENERGY"     # Electrical Engineering for Digital Energy (LM-28)

# Mappa: nome leggibile → ID interno (case-insensitive matching in provider)
COURSE_NAME_MAP: dict[str, str] = {
    # --- L-8 ---
    "ingegneria informatica triennale":          COURSE_L8_INFO,
    "ingegneria informatica l-8":                COURSE_L8_INFO,
    "ingegneria informatica (l-8)":              COURSE_L8_INFO,
    "l8 informatica":                            COURSE_L8_INFO,
    "ingegneria dell'informazione per la medicina digitale": COURSE_L8_MED,
    "ingegneria informazione medicina digitale": COURSE_L8_MED,
    "l8 medicina":                               COURSE_L8_MED,
    "l-8 medicina":                              COURSE_L8_MED,
    # --- LM-32 ---
    "ingegneria informatica magistrale":         COURSE_LM32_INFO,
    "ingegneria informatica lm-32":              COURSE_LM32_INFO,
    "ingegneria informatica (lm-32)":            COURSE_LM32_INFO,
    "lm32 informatica":                          COURSE_LM32_INFO,
    "information engineering for digital medicine": COURSE_LM32_MED,
    "information engineering digital medicine":  COURSE_LM32_MED,
    "lm32 medicina":                             COURSE_LM32_MED,
    # --- LM-28 ---
    "electrical engineering for digital energy": COURSE_LM28_ENERGY,
    "electrical engineering digital energy":     COURSE_LM28_ENERGY,
    "lm28":                                      COURSE_LM28_ENERGY,
    "lm-28":                                     COURSE_LM28_ENERGY,
}

# Lista normalizzata dei nomi "display" dei corsi per l'utente
DISPLAY_NAMES: dict[str, str] = {
    COURSE_L8_INFO:    "Ingegneria Informatica (L-8) — Triennale",
    COURSE_L8_MED:     "Ingegneria dell'Informazione per la Medicina Digitale (L-8) — Triennale",
    COURSE_LM32_INFO:  "Ingegneria Informatica (LM-32) — Magistrale",
    COURSE_LM32_MED:   "Information Engineering for Digital Medicine (LM-32) — Magistrale",
    COURSE_LM28_ENERGY:"Electrical Engineering for Digital Energy (LM-28) — Magistrale",
}

# Identifica se un corso è triennale o magistrale
TRIENNALE_COURSES = {COURSE_L8_INFO, COURSE_L8_MED}
MAGISTRALE_COURSES = {COURSE_LM32_INFO, COURSE_LM32_MED, COURSE_LM28_ENERGY}


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class GradeInputTriennale:
    """Dati di input per il calcolo del voto di laurea triennale."""
    media_30:          float             # Media ponderata degli esami in trentesimi
    pt:                float             # Punteggio tesi assegnato dalla Commissione [0-2]
    # Bonus carriera
    laurea_in_corso:   bool = False      # Laureato senza fuori corso entro il limite temporale
    cfu_estero:        float = 0.0       # CFU svolti all'estero


@dataclass
class GradeInputMagistrale:
    """Dati di input per il calcolo del voto di laurea magistrale."""
    media_30:          float             # Media ponderata degli esami in trentesimi
    pt:                float             # Punteggio tesi assegnato dalla Commissione [0-4]
    # Bonus carriera
    laurea_in_corso:   bool = False      # Laureato senza fuori corso entro il limite temporale
    programma_internazionale: bool = False  # Doppio titolo / programma internazionale
    cfu_estero:        float = 0.0       # CFU svolti all'estero (rilevante se non doppio titolo)
    percorso_eccellenza: bool = False    # Percorso di eccellenza completato


@dataclass
class GradeResult:
    """Risultato del calcolo del voto di laurea."""
    voto_laurea:    float           # Voto finale (centodecimi)
    voto_min:       float           # V_MIN in centodecimi
    pt_score:       float           # Punteggio tesi (PT)
    career_bonus:   float           # Punteggio carriera grezzo
    pt_career_cap:  float           # min(PT + PCp, cap) effettivo
    pcp:            float           # Punteggio carriera pesato (PCp)
    fc:             float           # Fattore correttivo pesato (FCp)
    lode_possible:  bool            # True se il voto è ≥ 112 (requisito per la lode)
    formula_used:   str             # Stringa descrittiva della formula applicata
    breakdown:      dict = field(default_factory=dict)  # Dettaglio completo dei passaggi


# =========================================================================
# LOGICA DI CALCOLO — FORMULE PER COORTE E CORSO
# =========================================================================

class GraduationGradeService:
    """
    Service per il calcolo del voto di laurea dell'Università degli Studi di Salerno (DIEM).
    Implementa le quattro varianti di formula previste dal regolamento:

      - Formula A  → Triennali L-8, coorte 25/26            (Immagine 1)
      - Formula B  → Triennali L-8, coorte ≤ 24/25          (Immagine 2) [Voto_Base]
      - Formula C  → Magistrali LM-32, coorte 25/26         (Immagine 3)
      - Formula D  → Magistrali LM-32/LM-28, coorte ≤ 24/25 (Immagine 4) [Voto_Base]
      - Formula E  → Magistrale LM-28, coorte 25/26         (Immagine 5)
    """

    # ------------------------------------------------------------------
    # HELPER COMUNI
    # ------------------------------------------------------------------

    @staticmethod
    def _voto_min(media_30: float) -> float:
        """Converte la media in trentesimi nel voto minimo in centodecimi."""
        return round((media_30 * 110) / 30, 2)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _standard_round(value: float) -> int:
        """Arrotonda per eccesso se >= .5, altrimenti per difetto."""
        return math.floor(value + 0.5)
    
    # ------------------------------------------------------------------
    # FORMULA A — Triennali L-8, coorte 25/26  (Immagine 1)
    # Voto = V_MIN + min(PT + PCp, 5) + FCp
    # PT ∈ [0,2], PCp = career_bonus × (PT/2)
    # FC = (4.1 × media_30 − 7.8) − V_MIN,  FCp = FC × (PT/2)
    # cap = 5
    # ------------------------------------------------------------------
    def _formula_a_triennale_2526(self, inp: GradeInputTriennale) -> GradeResult:
        voto_base = self._voto_min(inp.media_30)

        # --- Punteggio carriera grezzo ---
        career_bonus = 0.0
        if inp.laurea_in_corso:
            career_bonus += 2.0          # 2 punti se si laurea entro il terzo anno acc.
        if inp.cfu_estero >= 12:
            career_bonus += 1.0          # 1 punto se ha svolto ≥ 12 CFU all'estero
        # (percorso di eccellenza non previsto in questa formula)

        # --- Punteggio carriera pesato ---
        pcp = career_bonus * (inp.pt / 2)

        # --- Fattore correttivo ---
        fc_raw = (4.1 * inp.media_30 - 7.8) - voto_base
        fcp = fc_raw * (inp.pt / 2)

        # --- Applicazione cap ---
        sum_pt_pcp = inp.pt + pcp
        capped = min(sum_pt_pcp, 5)

        voto = voto_base + capped + fcp
        voto = round(self._clamp(voto, voto_base, 110), 2)  # Non può scendere sotto V_MIN

        return GradeResult(
            voto_laurea=voto,
            voto_min=round(voto_base, 2),
            pt_score=inp.pt,
            career_bonus=career_bonus,
            pt_career_cap=round(capped, 4),
            pcp=round(pcp, 4),
            fc=round(fcp, 4),
            lode_possible=(voto >= 112),
            formula_used="Formula A — Triennale L-8, coorte 25/26",
            breakdown={
                "V_MIN (centodecimi)": round(voto_base, 2),
                "PT (punteggio tesi)": inp.pt,
                "PCp = career × (PT/2)": round(pcp, 4),
                "min(PT+PCp, 5)": round(capped, 4),
                "FC_raw = (4.1×media30 − 7.8) − V_MIN": round(fc_raw, 4),
                "FCp = FC_raw × (PT/2)": round(fcp, 4),
                "Voto di Laurea": voto,
            }
        )

    # ------------------------------------------------------------------
    # FORMULA B — Triennali L-8, coorte ≤ 24/25  (Immagine 2)
    # Voto_Base = (4.1×media_30 − 7.8) / 110 → arrotondato all'intero più vicino
    # Commissione assegna: V_Base - 5  ≤ voto ≤ Voto_Base + 5 (max bonus = 5)
    # La formula calcola il Voto_Base + bonus
    # ------------------------------------------------------------------
    def _formula_b_triennale_old(self, inp: GradeInputTriennale) -> GradeResult:
        """
        Nota: questa formula non usa PT/PCp/FCp ma opera con Voto_Base +
        punti aggiuntivi assegnati dalla Commissione (0-2-0-2 centodecimi).
        Il tool restituisce il Voto_Base e i massimi punti aggiuntivi raggiungibili.
        """
        # Voto_Base (Immagine 2)
        voto_base_raw = (4.1 * inp.media_30 - 7.8)
        voto_base = self._standard_round(voto_base_raw)   # Arrotondato all'intero più vicino

        # Punti aggiuntivi (bonus carriera, non pesato da PT in questa formula)
        bonus_tempo = 0.0
        if inp.laurea_in_corso:
            bonus_tempo = 3.0  # 3 centodecimi se entro il 31/05 del 4° anno solare
            # (il tool usa il dato fornito senza distinguere 4° e 5° anno,
            #  per semplicità l'utente indica se è in corso o no)

        bonus_estero = 1.0 if inp.cfu_estero >= 12 else 0.0

        # Maturità culturale (0-2) e contenuto applicativo (0-1) → inseriti come PT
        # In questa formula PT viene riutilizzato come "punti commissione" 0-2
        punti_commissione = self._clamp(inp.pt, 0, 2)   # Re-mappa PT → commissione max 2

        # Totale punti aggiuntivi (cap 5)
        total_bonus = bonus_tempo + bonus_estero + punti_commissione
        total_bonus_capped = min(total_bonus, 5)

        voto = voto_base + total_bonus_capped
        voto = round(self._clamp(voto, voto_base, 110), 2)

        return GradeResult(
            voto_laurea=voto,
            voto_min=voto_base,
            pt_score=inp.pt,
            career_bonus=bonus_tempo + bonus_estero,
            pt_career_cap=round(total_bonus_capped, 4),
            pcp=0.0,
            fc=0.0,
            lode_possible=(voto_base + total_bonus_capped >= 112),
            formula_used="Formula B — Triennale L-8, coorte ≤ 24/25 (Voto_Base)",
            breakdown={
                "Voto di Partenza = round((4.1×media30−7.8)/110)": voto_base,
                "Bonus tempo (in corso)": bonus_tempo,
                "Bonus estero (≥12 CFU)": bonus_estero,
                "Punti commissione (maturità+contenuto, max 2)": round(punti_commissione, 2),
                "Totale bonus (post-cap)": round(total_bonus_capped, 4),
                "Voto di Laurea": voto,
            }
        )

    # ------------------------------------------------------------------
    # FORMULA C — Magistrali LM-32, coorte 25/26  (Immagine 3)
    # Voto = V_MIN + min(PT + PCp, 6) + FCp
    # PT ∈ [0,4], PCp = career_bonus × (PT/4)
    # FC = (4.1 × media_30 − 8.8) − V_MIN,  FCp = FC × (PT/4)
    # cap = 6
    # ------------------------------------------------------------------
    def _formula_c_magistrale_lm32_2526(self, inp: GradeInputMagistrale) -> GradeResult:
        voto_base = self._voto_min(inp.media_30)

        # --- Punteggio carriera grezzo ---
        career_bonus = 0.0
        if inp.laurea_in_corso:
            career_bonus += 1.0   # 1 punto se entro il secondo anno acc. senza fuori corso
        if inp.programma_internazionale:
            career_bonus += 1.0   # 1 punto doppio titolo
        elif inp.cfu_estero >= 15:
            career_bonus += 0.5   # 0.5 punti se ≥ 15 CFU all'estero
        if inp.percorso_eccellenza:
            career_bonus += 1.0   # 1 punto percorso di eccellenza

        # --- Punteggio carriera pesato ---
        pcp = career_bonus * (inp.pt / 4)

        # --- Fattore correttivo ---
        fc_raw = (4.1 * inp.media_30 - 8.8) - voto_base
        fcp = fc_raw * (inp.pt / 4)

        # --- Applicazione cap ---
        sum_pt_pcp = inp.pt + pcp
        capped = min(sum_pt_pcp, 6)

        voto = voto_base + capped + fcp
        voto = round(self._clamp(voto, voto_base, 110), 2)

        return GradeResult(
            voto_laurea=voto,
            voto_min=round(voto_base, 2),
            pt_score=inp.pt,
            career_bonus=career_bonus,
            pt_career_cap=round(capped, 4),
            pcp=round(pcp, 4),
            fc=round(fcp, 4),
            lode_possible=(voto >= 112),
            formula_used="Formula C — Magistrale LM-32, coorte 25/26",
            breakdown={
                "V_MIN (centodecimi)": round(voto_base, 2),
                "PT (punteggio tesi)": inp.pt,
                "Punteggio carriera grezzo": career_bonus,
                "PCp = career × (PT/4)": round(pcp, 4),
                "min(PT+PCp, 6)": round(capped, 4),
                "FC_raw = (4.1×media30 − 8.8) − V_MIN": round(fc_raw, 4),
                "FCp = FC_raw × (PT/4)": round(fcp, 4),
                "Voto di Laurea": voto,
            }
        )

    # ------------------------------------------------------------------
    # FORMULA D — Magistrali LM-32 e LM-28, coorte ≤ 24/25  (Immagine 4)
    # Voto_Base = (4.1×media_30 − 8.8) / 110  → arrotondato all'intero più vicino
    # Cap bonus = 6 centodecimi
    # ------------------------------------------------------------------
    def _formula_d_magistrale_old(self, inp: GradeInputMagistrale) -> GradeResult:
        voto_base_raw = (4.1 * inp.media_30 - 8.8) 
        voto_base = self._standard_round(voto_base_raw)

        # Bonus tempo (centodecimi)
        bonus_tempo = 0.0
        if inp.laurea_in_corso:
            bonus_tempo = 2.0   # 2 centodecimi entro il 31/05 del 4° anno

        # Bonus estero
        bonus_estero = 0.0
        if inp.programma_internazionale:
            bonus_estero = 2.0
        elif inp.cfu_estero >= 15:
            bonus_estero = 1.0

        # Maturità culturale (0-2) + contenuto ingegneristico (0-2) → riutilizza PT [0-4]
        punti_commissione = self._clamp(inp.pt, 0, 4)

        total_bonus = bonus_tempo + bonus_estero + punti_commissione
        total_bonus_capped = min(total_bonus, 6)

        
        voto = voto_base + total_bonus_capped
        voto = round(self._clamp(voto, voto_base, 110), 2)

        return GradeResult(
            voto_laurea=voto,
            voto_min=voto_base,
            pt_score=inp.pt,
            career_bonus=bonus_tempo + bonus_estero,
            pt_career_cap=round(total_bonus_capped, 4),
            pcp=0.0,
            fc=0.0,
            lode_possible=(voto_base + total_bonus_capped >= 112),
            formula_used="Formula D — Magistrale LM-32/LM-28, coorte ≤ 24/25 (Voto_Base)",
            breakdown={
                "Voto di Partenza = round((4.1×media30−8.8)/110)": voto_base,
                "Bonus tempo (in corso)": bonus_tempo,
                "Bonus estero": bonus_estero,
                "Punti commissione (maturità+contenuto, max 4)": round(punti_commissione, 2),
                "Totale bonus (post-cap)": round(total_bonus_capped, 4),
                "Voto di Laurea": voto,
            }
        )

    # ------------------------------------------------------------------
    # FORMULA E — Magistrale LM-28, coorte 25/26  (Immagine 5)
    # Identica alla Formula C ma per LM-28
    # Voto = V_MIN + min(PT + PCp, 6) + FCp
    # ------------------------------------------------------------------
    def _formula_e_magistrale_lm28_2526(self, inp: GradeInputMagistrale) -> GradeResult:
        result = self._formula_c_magistrale_lm32_2526(inp)
        result.formula_used = "Formula E — Magistrale LM-28, coorte 25/26"
        return result

    # ------------------------------------------------------------------
    # DISPATCHER PRINCIPALE
    # ------------------------------------------------------------------

    def calculate(
        self,
        course_id: str,
        cohort_year: str,
        media_30: float,
        pt: float,
        # Triennale
        laurea_in_corso: bool = False,
        cfu_estero: float = 0.0,
        # Magistrale only
        programma_internazionale: bool = False,
        percorso_eccellenza: bool = False,
    ) -> dict:
        """
        Calcola il voto di laurea e restituisce un dizionario JSON-friendly.

        Args:
            course_id:              ID interno del corso (usa COURSE_* costanti).
            cohort_year:            '25/26' per coorte 25/26, qualsiasi altra stringa per vecchie coorti.
            media_30:               Media ponderata in trentesimi (es. 27.5).
            pt:                     Punteggio tesi (0-2 triennale / 0-4 magistrale).
            laurea_in_corso:        True se laureato senza fuori corso entro il limite.
            cfu_estero:             CFU svolti all'estero.
            programma_internazionale: True se ha fatto doppio titolo (magistrale).
            percorso_eccellenza:    True se ha completato percorso di eccellenza.

        Returns:
            dict con 'status' (SUCCESS/ERROR), 'result' e 'explanation'.
        """
        try:
            is_new_cohort = (str(cohort_year).strip() == "25/26")

            # ── TRIENNALI ──────────────────────────────────────────────
            if course_id in TRIENNALE_COURSES:
                inp = GradeInputTriennale(
                    media_30=media_30,
                    pt=self._clamp(pt, 0, 2),
                    laurea_in_corso=laurea_in_corso,
                    cfu_estero=cfu_estero,
                )
                result = (
                    self._formula_a_triennale_2526(inp)
                    if is_new_cohort
                    else self._formula_b_triennale_old(inp)
                )

            # ── MAGISTRALI ─────────────────────────────────────────────
            elif course_id in MAGISTRALE_COURSES:
                inp = GradeInputMagistrale(
                    media_30=media_30,
                    pt=self._clamp(pt, 0, 4),
                    laurea_in_corso=laurea_in_corso,
                    programma_internazionale=programma_internazionale,
                    cfu_estero=cfu_estero,
                    percorso_eccellenza=percorso_eccellenza,
                )
                if is_new_cohort:
                    if course_id == COURSE_LM28_ENERGY:
                        result = self._formula_e_magistrale_lm28_2526(inp)
                    else:  # LM-32
                        result = self._formula_c_magistrale_lm32_2526(inp)
                else:
                    result = self._formula_d_magistrale_old(inp)

            else:
                return {
                    "status": "ERROR",
                    "message": f"Corso '{course_id}' non riconosciuto. Corsi supportati: {list(DISPLAY_NAMES.keys())}"
                }

            # ── FORMATTAZIONE RISPOSTA ─────────────────────────────────
            voto_display = math.floor(result.voto_laurea)  # Intero in centodecimi
            lode_note = (
                "✅ Requisito voto (≥112/110) soddisfatto per la lode con votazione unanime."
                if result.lode_possible
                else "ℹ️ Voto inferiore a 112: la lode non è assegnabile."
            )

            explanation = self._build_explanation(
                course_id=course_id,
                cohort_year=cohort_year,
                result=result,
                voto_display=voto_display,
                lode_note=lode_note,
            )

            return {
                "status": "SUCCESS",
                "corso": DISPLAY_NAMES.get(course_id, course_id),
                "coorte": cohort_year,
                "formula_utilizzata": result.formula_used,
                "voto_di_laurea": voto_display,
                "lode_possibile": result.lode_possible,
                "dettaglio_calcolo": result.breakdown,
                "spiegazione": explanation,
            }

        except Exception as e:
            return {"status": "ERROR", "message": f"Errore nel calcolo: {str(e)}"}

    # ------------------------------------------------------------------
    # HELPER: COSTRUZIONE TESTO ESPLICATIVO
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        course_id: str,
        cohort_year: str,
        result: GradeResult,
        voto_display: int,
        lode_note: str,
    ) -> str:
        lines = [
            f"**📐 Formula applicata:** {result.formula_used}",
            "",
            "**🔢 Dettaglio passaggi:**",
        ]
        for step, value in result.breakdown.items():
            lines.append(f"  - {step}: **{value}**")
        lines += [
            f"Ecco la stima del tuo voto di laurea per **{DISPLAY_NAMES.get(course_id, course_id)}** (Coorte {cohort_year}):",
            "",
            f"• **Voto di partenza:** {result.breakdown.get('Voto di partenza', result.voto_min)}",
            f"• **Possibili punti bonus:** {result.breakdown.get('Possibili punti bonus (Tesi + Carriera)', result.pt_career_cap)}",
            f"• **Voto finale stimato:** {voto_display}/110",
            "",
            lode_note,
            "",
            "> *Il voto finale è determinato dalla Commissione e potrebbe variare entro i limiti regolamentari.*",
            "",
            "⚠️ ISTRUZIONE INTERNA PER L'AI: Riproduci FEDELMENTE il testo puntato qui sopra per l'utente, mantenendo i grassetti. Non riassumere e non modificare i numeri."
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # UTILITY: RISOLUZIONE CORSO DA TESTO LIBERO
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_course_id(raw_text: str) -> Optional[str]:
        """
        Tenta di mappare un testo libero in un course_id.
        Restituisce None se non trovato.
        """
        normalized = raw_text.lower().strip()
        # Match esatto
        if normalized in COURSE_NAME_MAP:
            return COURSE_NAME_MAP[normalized]
        # Match parziale (substring)
        for key, cid in COURSE_NAME_MAP.items():
            if key in normalized or normalized in key:
                return cid
        return None

    @staticmethod
    def get_course_menu() -> str:
        """Restituisce un menu leggibile dei corsi supportati."""
        lines = ["Corsi di laurea supportati:"]
        for cid, name in DISPLAY_NAMES.items():
            lines.append(f"  • {name}")
        return "\n".join(lines)

    @staticmethod
    def get_cohort_options_for_course(course_id: str) -> list[str]:
        """Restituisce le coorti disponibili per il corso."""
        return ["25/26", "24/25 o precedenti"]
