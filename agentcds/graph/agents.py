"""
Specialist Agents — Layer 2
============================
Three domain-specific agents that run in parallel after the initial
differential is formed.  Each interprets a different slice of the
patient record and returns structured clinical signals that are merged
by the Orchestrator before retrieval begins.

Lab Agent
  Reads all lab values as a *pattern* (not individually) and names
  known clinical syndromes such as pancytopenia, macrocytosis, or
  hypoproliferative anaemia.

Radiology Agent
  Converts free-text imaging reports into present/absent finding lists
  and maps the combination to diagnostic implications.

Pharmacology Agent
  Checks medications for drug-induced disease causes, drug–drug
  interactions (via RxNorm MCP), and renal dosing concerns.
"""
from __future__ import annotations

import json
import re

from agentcds import llm, config
from agentcds.schemas import LabResult, Patient

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LAB_AGENT_PROMPT = r"""You are a haematology/biochemistry specialist.

Patient lab results:
{labs}

Reference vitals context:
{vitals}

Instructions:
1. Identify ALL named clinical syndromes visible in this panel (e.g.
   pancytopenia, macrocytosis, microcytic anaemia, hypoproliferative
   pattern, elevated LDH with low retics, neutrophilia, etc.).
2. Note which cell lines / axes are involved and the severity of deviation.
3. State the diagnostic category implications (e.g. bone-marrow failure,
   haemolysis, nutritional deficiency, infection).

Respond with ONLY a JSON object:
{{
  "patterns": ["pattern1", "pattern2"],
  "severity": "mild|moderate|severe",
  "implications": ["implication1", "implication2"],
  "summary": "<2-sentence plain-text summary>"
}}"""


RADIOLOGY_AGENT_PROMPT = r"""You are a radiology specialist interpreting clinical imaging reports.

Imaging reports:
{imaging}

Instructions:
1. Extract ALL findings that are PRESENT (explicitly stated as present or found).
2. Extract ALL findings that are ABSENT (explicitly negated or stated as absent/clear).
3. Map the present+absent combination to diagnostic implications.
   Use standard radiological reasoning (e.g. effusion + no consolidation → PE/CHF;
   widened mediastinum → aortic dissection; subsegmental atelectasis → PE).

Respond with ONLY a JSON object:
{{
  "present": ["finding1", "finding2"],
  "absent": ["finding3"],
  "implications": ["implication1"],
  "summary": "<2-sentence plain-text summary>"
}}"""


PHARMA_DRUG_CAUSE_PROMPT = r"""You are a clinical pharmacologist.

Patient medications:
{medications}

Patient labs (selected):
{labs}

Known drug-symptom / drug-lab relationships to screen for:
- Metformin → B12 deficiency (macrocytosis)
- Statins → myopathy, elevated CK
- NSAIDs → renal impairment, GI bleeding
- Diuretics → electrolyte abnormalities
- ACE inhibitors / ARBs → hyperkalaemia, cough
- Chemotherapy agents → cytopenias
- PPIs → B12, magnesium deficiency
- Anticoagulants → bleeding risk
- Antibiotics (long-course) → C. diff, dysbiosis

Which drug-induced causes are plausible for THIS patient given the
medications and labs above?

Respond with ONLY a JSON array of strings (empty array if none):
["Drug X may cause Y because ..."]"""


PHARMA_RENAL_PROMPT = r"""You are a clinical pharmacologist reviewing renal dosing safety.

Patient creatinine: {creatinine} mg/dL  (estimated eGFR: {egfr:.0f} mL/min/1.73m²)
Patient age: {age}  Sex: {sex}

Medications:
{medications}

Which of these medications require dose adjustment or should be
used with caution at this eGFR level?

Respond with ONLY a JSON array of strings (empty array if none):
["[RENAL] Drug X: <specific concern and recommended action>"]"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_json_list(raw: str) -> list:
    """Extract a JSON list from an LLM response string."""
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def _safe_json_obj(raw: str) -> dict:
    """Extract a JSON object from an LLM response string."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


def _egfr(creatinine: float, age: int, sex: str) -> float:
    """CKD-EPI approximation (simplified Cockcroft-Gault for safety screening)."""
    if creatinine <= 0:
        return 90.0
    egfr = (140 - age) * (0.85 if sex.upper() == "F" else 1.0) / (72 * creatinine)
    return max(5.0, egfr * 100)   # crude scaling; sufficient for flag-level decisions


# ---------------------------------------------------------------------------
# Lab Agent
# ---------------------------------------------------------------------------

def run_lab_agent(patient: Patient) -> str:
    """
    Interpret the patient's laboratory panel as a clinical pattern.

    Returns a plain-text summary of named syndromes and their diagnostic
    implications, intended to be injected into the orchestrator's
    enriched context before RAG begins.
    """
    if not patient.labs:
        return "No laboratory data available."

    labs_text = "\n".join(
        f"  {l.name}: {l.value} {l.unit}"
        f"  (ref {l.ref_low}–{l.ref_high})"
        f"{'  ← ABNORMAL' if l.abnormal else ''}"
        for l in patient.labs
    )
    vitals_text = "  ".join(f"{k}: {v}" for k, v in patient.vitals.items()) or "Not recorded"

    prompt = LAB_AGENT_PROMPT.format(labs=labs_text, vitals=vitals_text)
    raw = llm.ask_json(prompt, max_tokens=400)
    obj = _safe_json_obj(raw)

    if not obj:
        return f"Lab Agent: unable to parse patterns from {len(patient.labs)} results."

    summary = obj.get("summary", "")
    patterns = obj.get("patterns", [])
    implications = obj.get("implications", [])
    severity = obj.get("severity", "unknown")

    print(f"  [Lab Agent]   severity={severity}  patterns={patterns}")

    lines = [f"LAB PATTERNS [{severity.upper()}]:"]
    for p in patterns:
        lines.append(f"  • {p}")
    if implications:
        lines.append("LAB IMPLICATIONS:")
        for i in implications:
            lines.append(f"  → {i}")
    if summary:
        lines.append(f"SUMMARY: {summary}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Radiology Agent
# ---------------------------------------------------------------------------

def run_radiology_agent(patient: Patient) -> str:
    """
    Parse free-text imaging reports into structured present/absent findings.

    Returns structured text ready to be merged into enriched clinical context.
    """
    if not patient.imaging:
        return "No imaging data available."

    imaging_text = "\n".join(f"  [{i+1}] {r}" for i, r in enumerate(patient.imaging))
    prompt = RADIOLOGY_AGENT_PROMPT.format(imaging=imaging_text)
    raw = llm.ask_json(prompt, max_tokens=400)
    obj = _safe_json_obj(raw)

    if not obj:
        return f"Radiology Agent: unable to parse {len(patient.imaging)} imaging report(s)."

    present = obj.get("present", [])
    absent  = obj.get("absent", [])
    implications = obj.get("implications", [])
    summary = obj.get("summary", "")

    print(f"  [Radiology]   present={len(present)}  absent={len(absent)}  implications={len(implications)}")

    lines = ["IMAGING FINDINGS:"]
    if present:
        lines.append("  PRESENT: " + ", ".join(present))
    if absent:
        lines.append("  ABSENT: " + ", ".join(absent))
    if implications:
        lines.append("IMAGING IMPLICATIONS:")
        for imp in implications:
            lines.append(f"  → {imp}")
    if summary:
        lines.append(f"SUMMARY: {summary}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pharmacology Agent
# ---------------------------------------------------------------------------

def run_pharmacology_agent(patient: Patient) -> tuple[str, list[str]]:
    """
    Three-stage drug safety analysis:
      1. Drug-induced cause screening (LLM, prompt-based)
      2. Drug–drug interaction checking (RxNorm MCP, hard API call)
      3. Renal dose checking (LLM, if impaired creatinine)

    Returns:
        pharma_context (str): plain-text summary for enriched context
        drug_warnings  (list[str]): ranked warning strings for output layer
    """
    if not patient.medications:
        return "No medications recorded.", []

    warnings: list[str] = []
    context_lines: list[str] = ["PHARMACOLOGY ANALYSIS:"]

    # ── Step 1: Drug-induced cause screening ──────────────────────
    abnormal_labs_text = "\n".join(
        f"  {l.name}: {l.value} {l.unit} ← ABNORMAL"
        for l in patient.labs if l.abnormal
    ) or "  None"
    meds_text = "\n".join(f"  • {m}" for m in patient.medications)

    cause_prompt = PHARMA_DRUG_CAUSE_PROMPT.format(
        medications=meds_text,
        labs=abnormal_labs_text,
    )
    raw_causes = llm.ask_json(cause_prompt, max_tokens=300)
    drug_causes = _safe_json_list(raw_causes)

    if drug_causes:
        context_lines.append("DRUG-INDUCED CAUSES:")
        for c in drug_causes:
            context_lines.append(f"  ⚕ {c}")
            warnings.append(f"[DRUG-CAUSE] {c}")
        print(f"  [Pharma]      drug-induced causes: {len(drug_causes)}")
    else:
        context_lines.append("DRUG-INDUCED CAUSES: None identified")
        print("  [Pharma]      no drug-induced causes identified")

    # ── Step 2: Drug–drug interactions via RxNorm MCP ─────────────
    if len(patient.medications) >= 2:
        try:
            import asyncio
            from fastmcp import Client
            from agentcds.mcp.rxnorm import mcp as rxnorm_mcp

            drug_names = [m.split()[0] for m in patient.medications]

            async def _check():
                async with Client(rxnorm_mcp) as c:
                    result = await c.call_tool("check_interactions", {"drug_names": drug_names})
                if not result:
                    return []
                content = getattr(result, "content", None)
                if content:
                    text = "".join(
                        ch.text if hasattr(ch, "text") else str(ch)
                        for ch in content
                    )
                else:
                    text = result.text if hasattr(result, "text") else str(result)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return []

            interactions = asyncio.run(_check()) or []
            severity_order = {"major": 0, "moderate": 1, "minor": 2}
            interactions = sorted(
                interactions,
                key=lambda x: severity_order.get(
                    x.get("severity", "").lower() if isinstance(x, dict) else "", 3
                ),
            )

            if interactions:
                context_lines.append("DRUG INTERACTIONS (RxNorm):")
                for ix in interactions:
                    if isinstance(ix, dict):
                        sev  = ix.get("severity", "unknown").upper()
                        d1   = ix.get("drug_1", "?")
                        d2   = ix.get("drug_2", "?")
                        desc = ix.get("description", "")[:120]
                        msg  = f"[{sev}] {d1} + {d2}: {desc}"
                    else:
                        msg = str(ix)
                    context_lines.append(f"  ⚠ {msg}")
                    warnings.append(f"[INTERACTION] {msg}")
                print(f"  [Pharma]      interactions found: {len(interactions)}")
            else:
                context_lines.append("DRUG INTERACTIONS: None significant via RxNorm")
                print("  [Pharma]      no significant interactions")

        except Exception as exc:
            print(f"  [Pharma]      RxNorm check failed: {exc}")
            context_lines.append(f"DRUG INTERACTIONS: Check failed ({exc})")

    # ── Step 3: Renal dose checking ───────────────────────────────
    creatinine_lab = next(
        (l for l in patient.labs if "creatinine" in l.name.lower()), None
    )
    if creatinine_lab and creatinine_lab.value > 1.3:
        egfr = _egfr(creatinine_lab.value, patient.age, patient.sex)
        renal_prompt = PHARMA_RENAL_PROMPT.format(
            creatinine=creatinine_lab.value,
            egfr=egfr,
            age=patient.age,
            sex=patient.sex,
            medications=meds_text,
        )
        raw_renal = llm.ask_json(renal_prompt, max_tokens=200)
        renal_flags = _safe_json_list(raw_renal)
        if renal_flags:
            context_lines.append(f"RENAL DOSING (eGFR ≈ {egfr:.0f}):")
            for flag in renal_flags:
                context_lines.append(f"  ⚠ {flag}")
                warnings.append(flag)
            print(f"  [Pharma]      renal flags: {len(renal_flags)}")
    else:
        print("  [Pharma]      renal function acceptable — no dose flags")

    return "\n".join(context_lines), warnings
