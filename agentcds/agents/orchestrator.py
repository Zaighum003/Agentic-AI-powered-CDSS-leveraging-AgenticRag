"""
Orchestrator Agent
===================
The main reasoning loop. Coordinates LLM calls, RAG pipeline, and MCP tools.

Session flow:
  1. Form initial differential from patient presentation
  2. Run Agentic RAG loop (HyDE → Self-RAG → CRAG)
  3. Drug safety check via RxNorm MCP
  4. Generate clarification questions if still uncertain
  5. Build and return DiagnosticResult
"""
import asyncio
import json
import re

from fastmcp import Client

from agentcds import llm, config
from agentcds.schemas import Patient, Hypothesis, DiagnosticResult
from agentcds.rag import pipeline
from agentcds.mcp.pubmed  import mcp as pubmed_mcp
from agentcds.mcp.rxnorm  import mcp as rxnorm_mcp


DIFFERENTIAL_PROMPT = r"""You are a clinical attending physician.

Patient:
{patient}

Generate a differential diagnosis of up to {n} conditions.
Assign an initial probability (0.0–1.0) based solely on the presentation.

Respond with ONLY a JSON array:
[
  {{"label": "Condition Name", "confidence": 0.XX, "icd11": "X.X",
    "urgency": "routine|urgent|emergent", "workup": ["test1", "test2"]}},
  ...
]"""


SYNTHESIS_PROMPT = r"""You are a clinical attending physician summarizing a diagnostic workup.

Patient: {patient}

Final differential after evidence review:
{differential}

Provide:
1. 3–5 specific recommended next steps
2. Any safety concerns

Respond with ONLY a JSON object:
{{"next_steps": ["step1", "step2"], "safety": ["concern1"]}}"""


CLARIFICATION_PROMPT = r"""The diagnostic confidence is low (top: {conf:.0%}).

Patient: {patient}
Differential: {differential}

Write 1–2 targeted follow-up questions that would most resolve the uncertainty.
Focus on pertinent positives/negatives the clinician could quickly check.

Respond with ONLY a JSON array of question strings:
["Question 1?", "Question 2?"]"""


async def _call_tool(mcp_server, tool_name: str, params: dict):
    """Call a FastMCP tool in-process and return parsed result."""
    async with Client(mcp_server) as client:
        result = await client.call_tool(tool_name, params)
    if result:
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _call(mcp_server, tool_name: str, params: dict):
    """Sync wrapper for _call_tool."""
    return asyncio.run(_call_tool(mcp_server, tool_name, params))


def _form_differential(patient: Patient) -> list[Hypothesis]:
    """Ask the LLM to generate an initial differential diagnosis."""
    prompt = DIFFERENTIAL_PROMPT.format(
        patient=patient.to_text(),
        n=config.MAX_DIFFERENTIAL,
    )
    raw = llm.ask_json(prompt, max_tokens=600)

    hypotheses = []
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            for item in json.loads(match.group()):
                hypotheses.append(Hypothesis(
                    label=item["label"],
                    confidence=float(item.get("confidence", 0.2)),
                    icd11=item.get("icd11"),
                    urgency=item.get("urgency", "routine"),
                    workup=item.get("workup", []),
                ))
    except (json.JSONDecodeError, KeyError):
        hypotheses = [Hypothesis(label="Undetermined — workup required", confidence=0.1)]

    return hypotheses


def _augment_with_pubmed(hypotheses: list[Hypothesis]) -> None:
    """
    Fetch PubMed articles for top hypotheses and add to the RAG cache.
    This seeds the vector store before the main RAG loop.
    """
    from agentcds import vector_store
    top = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:2]
    for h in top:
        articles = _call(pubmed_mcp, "pubmed_search", {
            "query": f"{h.label} diagnosis clinical criteria",
            "n": 3,
        }) or []
        for art in articles:
            text = f"[PubMed {art.get('pmid','')}] {art.get('title','')}\n{art.get('abstract','')}"
            meta = {"source": f"pubmed:{art.get('pmid','')}", "study_type": art.get("study_type","")}
            vector_store.add("cache", [text], [meta])


def _drug_safety_check(patient: Patient) -> list[str]:
    """Check drug interactions via RxNorm MCP. Returns list of warning strings."""
    if len(patient.medications) < 2:
        return []
    drug_names = [m.split()[0] for m in patient.medications]  # first word = drug name
    summary = _call(rxnorm_mcp, "interaction_summary", {"drug_names": drug_names})
    if isinstance(summary, str) and "No significant" not in summary:
        return [summary]
    return []


def _synthesis(patient: Patient, hypotheses: list[Hypothesis]) -> tuple[list[str], list[str]]:
    """Get recommended next steps and safety flags from the LLM."""
    diff_str = "\n".join(
        f"  {h.label} ({h.confidence:.0%}) [{h.urgency}]"
        for h in sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
    )
    prompt = SYNTHESIS_PROMPT.format(patient=patient.to_text()[:500], differential=diff_str)
    raw = llm.ask_json(prompt, max_tokens=300)

    next_steps, safety = [], []
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            next_steps = data.get("next_steps", [])
            safety     = data.get("safety", [])
    except (json.JSONDecodeError, AttributeError):
        pass

    return next_steps, safety


def _clarifications(patient: Patient, hypotheses: list[Hypothesis]) -> list[str]:
    """Generate targeted follow-up questions when confidence is below threshold."""
    top_conf = max(h.confidence for h in hypotheses)
    if top_conf >= config.CONFIDENCE_LOW:
        return []

    diff_str = "\n".join(
        f"  {h.label} ({h.confidence:.0%})"
        for h in sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
    )
    prompt = CLARIFICATION_PROMPT.format(
        conf=top_conf,
        patient=patient.to_text()[:400],
        differential=diff_str,
    )
    raw = llm.ask_json(prompt, max_tokens=200)
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def diagnose(patient: Patient) -> DiagnosticResult:
    """
    Run a complete clinical diagnostic session for a patient.

    Args:
        patient: Patient object (from schemas.py or FHIR MCP)

    Returns:
        DiagnosticResult with differential, next steps, and warnings
    """
    # 1. Initial differential
    hypotheses = _form_differential(patient)

    # 2. Seed vector store with PubMed articles for top hypotheses
    _augment_with_pubmed(hypotheses)

    # 3. Agentic RAG loop (HyDE → Self-RAG → CRAG → confidence update)
    iterations = pipeline.run(patient, hypotheses)

    # 4. Drug safety check
    drug_warnings = _drug_safety_check(patient)

    # 5. Synthesis — next steps + safety
    next_steps, safety_flags = _synthesis(patient, hypotheses)
    drug_warnings.extend(safety_flags)

    # 6. Clarification questions if still uncertain
    clarifications = _clarifications(patient, hypotheses)

    return DiagnosticResult(
        patient_id=patient.patient_id,
        differential=sorted(hypotheses, key=lambda h: h.confidence, reverse=True),
        next_steps=next_steps,
        clarifications=clarifications,
        drug_warnings=drug_warnings,
        rag_iterations=iterations,
    )
