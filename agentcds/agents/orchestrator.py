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
from agentcds.mcp.pubmed      import mcp as pubmed_mcp
from agentcds.mcp.rxnorm      import mcp as rxnorm_mcp
from agentcds.mcp.web_search  import mcp as websearch_mcp


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


def _tool_result_text(result) -> str:
    """Extract text payload from FastMCP call_tool result across API versions."""
    if result is None:
        return ""

    content = getattr(result, "content", None)
    if content:
        # Aggregate ALL content chunks (large MCP results may be split)
        parts = []
        for chunk in content:
            parts.append(chunk.text if hasattr(chunk, "text") else str(chunk))
        return "".join(parts)

    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return first.text if hasattr(first, "text") else str(first)

    return result.text if hasattr(result, "text") else str(result)


async def _call_tool(mcp_server, tool_name: str, params: dict):
    """Call a FastMCP tool in-process and return parsed result."""
    async with Client(mcp_server) as client:
        result = await client.call_tool(tool_name, params)
    if result:
        text = _tool_result_text(result)
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
    top = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
    queries_per_hypothesis = [
        "{label} diagnosis clinical criteria",
        "{label} pathophysiology laboratory findings",
    ]
    print("\n[MCP:PubMed] Seeding vector store — fetching articles for top hypotheses...")
    total_seeded = 0
    for h in top:
        for query_tpl in queries_per_hypothesis:
            query = query_tpl.format(label=h.label)
            articles = _call(pubmed_mcp, "pubmed_search", {
                "query": query,
                "n": 4,
            }) or []
            # Guard against non-list response (e.g. raw string from failed JSON parse)
            if not isinstance(articles, list):
                print(f"  [PubMed] skipped '{query}' — unexpected response type")
                continue
            print(f"  query: '{query}' → {len(articles)} article(s)")
            for art in articles:
                if not isinstance(art, dict):
                    continue
                print(f"    [{art.get('study_type','?')}] PMID:{art.get('pmid','')} — {art.get('title','')[:80]}")
                text = f"[PubMed {art.get('pmid','')}] {art.get('title','')}\n{art.get('abstract','')}"
                meta = {"source": f"pubmed:{art.get('pmid','')}", "study_type": art.get("study_type","")}
                vector_store.add("cache", [text], [meta])
                total_seeded += 1
    print(f"[MCP:PubMed] Done — {total_seeded} article(s) seeded into vector store\n")


def _augment_with_web(hypotheses: list[Hypothesis]) -> None:
    """
    Supplement the vector store with medical web results for top hypotheses.
    Called after PubMed seeding as an additional evidence source.
    """
    from agentcds import vector_store
    top = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
    print("\n[MCP:WebSearch] Fetching medical web results for top hypotheses...")
    total_seeded = 0
    for h in top:
        query = f"{h.label} diagnosis management clinical guidelines"
        try:
            results = _call(websearch_mcp, "web_search_medical", {"query": query, "n": 4}) or []
        except Exception as exc:
            print(f"  [WebSearch] error for '{query}': {exc}")
            continue
        if not isinstance(results, list):
            print(f"  [WebSearch] skipped '{query}' — unexpected response type")
            continue
        print(f"  query: '{query}' → {len(results)} result(s)")
        for r in results:
            if not isinstance(r, dict) or not r.get("snippet"):
                continue
            print(f"    {r.get('title','')[:80]}")
            print(f"    {r.get('url','')}")
            text = f"[Web] {r.get('title','')}\n{r.get('snippet','')}"
            meta = {"source": r.get("url", ""), "type": "web"}
            vector_store.add("cache", [text], [meta])
            total_seeded += 1
    print(f"[MCP:WebSearch] Done — {total_seeded} web snippet(s) seeded into vector store\n")


def _drug_safety_check(patient: Patient) -> list[str]:
    """Check drug interactions via RxNorm MCP. Returns list of warning strings."""
    if len(patient.medications) < 2:
        return []
    drug_names = [m.split()[0] for m in patient.medications]  # first word = drug name
    print(f"[MCP:RxNorm] Checking interactions for: {', '.join(drug_names)}")
    interactions = _call(rxnorm_mcp, "check_interactions", {"drug_names": drug_names}) or []

    warnings = []
    if interactions and isinstance(interactions, list):
        # Sort by severity: major first
        severity_order = {"major": 0, "moderate": 1, "minor": 2}
        interactions = sorted(interactions, key=lambda x: severity_order.get(x.get("severity", "").lower(), 3))
        for ix in interactions:
            if isinstance(ix, dict):
                sev  = ix.get('severity', 'unknown').upper()
                d1   = ix.get('drug_1', '?')
                d2   = ix.get('drug_2', '?')
                desc = ix.get('description', '')[:120]
                msg  = f"[{sev}] {d1} + {d2}: {desc}"
                print(f"  ⚠ {msg}")
                warnings.append(msg)
            else:
                print(f"  ⚠ {ix}")
                warnings.append(str(ix))
    else:
        print("  ✓ No significant interactions found")

    return warnings


def _synthesis(patient: Patient, hypotheses: list[Hypothesis]) -> tuple[list[str], list[str]]:
    """Get recommended next steps and safety flags from the LLM."""
    diff_str = "\n".join(
        f"  {h.label} ({h.confidence:.0%}) [{h.urgency}]"
        for h in sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
    )
    prompt = SYNTHESIS_PROMPT.format(patient=patient.to_text()[:500], differential=diff_str)
    raw = llm.ask_json(prompt, max_tokens=600)

    next_steps, safety = [], []
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            next_steps = data.get("next_steps", [])
            safety     = data.get("safety", [])
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass

    if next_steps:
        print(f"[Synthesis] {len(next_steps)} next step(s) generated")
    else:
        print(f"[Synthesis] WARNING — could not parse next steps (raw={raw[:120]!r})")

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

    # 2b. Supplement with medical web results
    _augment_with_web(hypotheses)

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
