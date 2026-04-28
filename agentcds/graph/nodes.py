"""
AgentCDS LangGraph Nodes
=========================
One function per node in the graph.  Every function receives the full
AgentCDSState and returns a *partial* dict that gets merged back into
state by LangGraph's reducer.

Node execution order (see graph.py for wiring):
  START
    └─ form_differential          (Layer 1+2 bootstrap)
         ├─ lab_agent_node         ┐
         ├─ radiology_agent_node   ├─ parallel (Layer 2)
         └─ pharmacology_agent_node┘
              └─ merge_agent_outputs
                   └─ seed_knowledge         (Layer 3)
                        └─ rag_node          (Layer 4, loops)
                             └─ synthesize   (Layer 5)
                                  └─ clarify
                                       └─ build_result
                                            └─ END
"""
from __future__ import annotations

import json
import re

from agentcds import llm, config
from agentcds.schemas import Patient, Hypothesis, Evidence, DiagnosticResult
from agentcds.graph.state import AgentCDSState
from agentcds.graph.agents import (
    run_lab_agent,
    run_radiology_agent,
    run_pharmacology_agent,
)

# ──────────────────────────────────────────────────────────────────────────────
# Layer 1+2 bootstrap
# ──────────────────────────────────────────────────────────────────────────────

DIFFERENTIAL_PROMPT = r"""You are a clinical attending physician.

Patient:
{patient}

Enriched clinical context (specialist agents):
{context}

Generate a differential diagnosis of up to {n} conditions.
Assign an initial probability (0.0–1.0) based on the presentation AND
the specialist context above.

Respond with ONLY a JSON array:
[
  {{"label": "Condition Name", "confidence": 0.XX, "icd11": "X.X",
    "urgency": "routine|urgent|emergent", "workup": ["test1", "test2"]}},
  ...
]"""


def form_differential(state: AgentCDSState) -> dict:
    """
    Layer 2 — bootstrap node.

    Runs the three specialist agents *before* the initial differential so
    the LLM has enriched context when it assigns confidence scores.
    This keeps the graph simple: we do NOT need a separate bootstrap step.

    Returns: hypotheses, lab_signals, radiology_signals, pharma_signals,
             enriched_context, drug_warnings (from pharma)
    """
    patient: Patient = state["patient"]
    print("\n" + "="*60)
    print(f"[Graph] Processing patient {patient.patient_id}")
    print("="*60)

    # ── Run specialist agents ──────────────────────────────────────
    print("\n[Layer 2] Running specialist agents …")

    print("\n  → Lab Agent")
    lab_sig = run_lab_agent(patient)

    print("\n  → Radiology Agent")
    rad_sig = run_radiology_agent(patient)

    print("\n  → Pharmacology Agent")
    pharma_sig, pharma_warnings = run_pharmacology_agent(patient)

    enriched = "\n\n".join(filter(None, [lab_sig, rad_sig, pharma_sig]))

    # ── Form initial differential using enriched context ──────────
    print("\n[Layer 2] Initial differential formed …")
    prompt = DIFFERENTIAL_PROMPT.format(
        patient=patient.to_text(),
        context=enriched[:800],
        n=config.MAX_DIFFERENTIAL,
    )
    raw = llm.ask_json(prompt, max_tokens=700)

    hypotheses: list[Hypothesis] = []
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            for item in json.loads(m.group()):
                hypotheses.append(Hypothesis(
                    label=item["label"],
                    confidence=float(item.get("confidence", 0.2)),
                    icd11=item.get("icd11"),
                    urgency=item.get("urgency", "routine"),
                    workup=item.get("workup", []),
                    supporting_factors=item.get("supporting_factors", []),
                    opposing_factors=item.get("opposing_factors", []),
                    missing_data=item.get("missing_data", []),
                    confidence_components=item.get("confidence_components", {}),
                ))
    except (json.JSONDecodeError, KeyError):
        hypotheses = [Hypothesis(label="Undetermined — workup required", confidence=0.1)]

    print(f"\n[Layer 2] Initial differential ({len(hypotheses)} items):")
    for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
        print(f"  {h.confidence:.0%}  {h.label}  [{h.urgency}]")

    support_features = _patient_support_features(patient)
    missing_features = _patient_missing_features(patient)
    for h in hypotheses:
        if not h.supporting_factors:
            h.supporting_factors = list(support_features)
        if not h.opposing_factors:
            h.opposing_factors = list((patient.absent or [])[:2])
        if not h.missing_data:
            h.missing_data = list(missing_features)
        if not h.confidence_components:
            h.confidence_components = {
                "initial_clinical_fit": round(h.confidence, 3),
                "support_signal_count": float(len(h.supporting_factors)),
                "missing_data_penalty": float(len(h.missing_data)) * 0.05,
            }

    return {
        "hypotheses": hypotheses,
        "lab_signals": lab_sig,
        "radiology_signals": rad_sig,
        "pharma_signals": pharma_sig,
        "enriched_context": enriched,
        "drug_warnings": pharma_warnings,   # operator.add channel
        "reasoning_trace": [
            f"Formed initial differential with {len(hypotheses)} hypotheses.",
            "Specialist agents contributed lab, radiology, and pharmacology context.",
        ],
        "rag_iteration": 0,
        "rag_done": False,
        "knowledge_seeded": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3 — Knowledge seeding
# ──────────────────────────────────────────────────────────────────────────────

def seed_knowledge(state: AgentCDSState) -> dict:
    """
    Layer 3 — Seed the vector store with PubMed abstracts and web
    guidelines for the top hypotheses before the RAG loop starts.
    """
    import asyncio
    import json as _json
    from fastmcp import Client
    from agentcds import vector_store
    from agentcds.mcp.pubmed import mcp as pubmed_mcp
    from agentcds.mcp.web_search import mcp as websearch_mcp

    hypotheses: list[Hypothesis] = state["hypotheses"]
    top = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]

    async def _pubmed(query: str, n: int = 4):
        async with Client(pubmed_mcp) as c:
            result = await c.call_tool("pubmed_search", {"query": query, "n": n})
        if not result:
            return []
        content = getattr(result, "content", None)
        text = ("".join(ch.text if hasattr(ch, "text") else str(ch) for ch in content)
                if content else str(result))
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            return []

    async def _websearch(query: str, n: int = 4):
        async with Client(websearch_mcp) as c:
            result = await c.call_tool("web_search_medical", {"query": query, "n": n})
        if not result:
            return []
        content = getattr(result, "content", None)
        text = ("".join(ch.text if hasattr(ch, "text") else str(ch) for ch in content)
                if content else str(result))
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            return []

    print("\n[Layer 3] Seeding knowledge base …")
    total = 0

    for h in top:
        # PubMed — two targeted queries per hypothesis
        for q_tpl in [
            "{label} diagnosis clinical criteria",
            "{label} pathophysiology laboratory findings",
        ]:
            query = q_tpl.format(label=h.label)
            try:
                articles = asyncio.run(_pubmed(query)) or []
                if not isinstance(articles, list):
                    continue
                for art in articles:
                    if not isinstance(art, dict):
                        continue
                    text = f"[PubMed {art.get('pmid','')}] {art.get('title','')}\n{art.get('abstract','')}"
                    meta = {"source": f"pubmed:{art.get('pmid','')}", "study_type": art.get("study_type", "")}
                    vector_store.add("cache", [text], [meta])
                    total += 1
                print(f"  [PubMed]  '{query[:50]}' → {len(articles)} article(s)")
            except Exception as exc:
                print(f"  [PubMed]  error for '{query[:40]}': {exc}")

        # Web — clinical guidelines
        try:
            results = asyncio.run(_websearch(f"{h.label} diagnosis management guidelines")) or []
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict) or not r.get("snippet"):
                        continue
                    text = f"[Web] {r.get('title','')}\n{r.get('snippet','')}"
                    meta = {"source": r.get("url", ""), "type": "web"}
                    vector_store.add("cache", [text], [meta])
                    total += 1
                print(f"  [Web]     '{h.label[:40]}' → {len(results)} snippet(s)")
        except Exception as exc:
            print(f"  [Web]     error for '{h.label[:40]}': {exc}")

    print(f"[Layer 3] Done — {total} document(s) seeded\n")
    return {
        "knowledge_seeded": True,
        "reasoning_trace": [f"Knowledge seeding added {total} evidence document(s) to retrieval cache."],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Layer 4 — Agentic RAG iteration node
# ──────────────────────────────────────────────────────────────────────────────

CONFIDENCE_UPDATE_PROMPT = r"""You are a clinical reasoning AI.

Patient: {patient}

Current differential:
{differential}

New evidence retrieved:
{evidence}

Update the confidence score (0.0-1.0) for each hypothesis based on
this evidence.  Respond with ONLY a JSON object mapping label → score:
{{"Diagnosis A": 0.65, "Diagnosis B": 0.20}}"""


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "moderate"
    return "low"


def _patient_support_features(patient: Patient) -> list[str]:
    features: list[str] = []
    features.extend((patient.findings or [])[:3])
    abnormal = [f"{lab.name} abnormal ({lab.value} {lab.unit})" for lab in patient.labs if lab.abnormal]
    features.extend(abnormal[:2])
    return features


def _patient_missing_features(patient: Patient) -> list[str]:
    missing: list[str] = []
    if not patient.imaging:
        missing.append("No imaging evidence available yet")
    if not patient.hpi:
        missing.append("Limited history of present illness detail")
    if not patient.absent:
        missing.append("Pertinent negatives not fully documented")
    return missing


def rag_node(state: AgentCDSState) -> dict:
    """
    Layer 4 — One iteration of the Agentic RAG loop.

    Steps:
      1. HyDE retrieval  — embed a hypothetical clinical note
      2. Self-RAG grading — relevance / support / utility per chunk
      3. CRAG            — corrective re-retrieval on contradictions
      4. Confidence update — LLM re-scores all hypotheses
    """
    from agentcds.rag import hyde, self_rag, crag
    from agentcds import vector_store as vs

    hypotheses: list[Hypothesis] = state["hypotheses"]
    patient: Patient = state["patient"]
    iteration: int = state["rag_iteration"] + 1

    top_conf = max(h.confidence for h in hypotheses)
    print(f"\n[RAG iteration {iteration}/{config.MAX_RAG_ITERATIONS}]  "
          f"top_conf={top_conf:.2f}  (stop if ≥{config.CONFIDENCE_HIGH})")

    # Work on the top-3 hypotheses each iteration
    active = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
    conf_before = {h.label: h.confidence for h in active}
    new_evidence_lines: list[str] = []

    for hyp in active:
        # ── HyDE retrieval ───────────────────────────────────────
        chunks = hyde.retrieve(patient, hyp)
        print(f"  [HyDE]     '{hyp.label[:40]}' → {len(chunks)} chunks")

        # ── Self-RAG grading ─────────────────────────────────────
        graded = self_rag.grade_all(chunks, hyp.label, patient.to_text())
        used    = sum(1 for g in graded if g.action == "use")
        crag_ct = sum(1 for g in graded if g.action == "crag")
        discard = sum(1 for g in graded if g.action == "discard")
        print(f"  [Self-RAG] graded {len(graded)} → use={used}  crag={crag_ct}  discard={discard}")

        # Fallback: if everything discarded, fetch fresh PubMed articles
        if used == 0 and crag_ct == 0 and iteration < config.MAX_RAG_ITERATIONS:
            import asyncio, json as _j
            from fastmcp import Client
            from agentcds.mcp.pubmed import mcp as _pmcp

            async def _fetch(q):
                async with Client(_pmcp) as c:
                    r = await c.call_tool("pubmed_search", {"query": q, "n": 4})
                raw = r.content[0].text if getattr(r, "content", None) else str(r)
                try:
                    return _j.loads(raw)
                except Exception:
                    return []

            q = f"{hyp.label} diagnosis clinical criteria management"
            fresh = asyncio.run(_fetch(q))
            for art in (fresh or []):
                t = f"[PubMed {art.get('pmid','')}] {art.get('title','')}\n{art.get('abstract','')}"
                m = {"source": f"pubmed:{art.get('pmid','')}", "study_type": art.get("study_type","")}
                vs.add("cache", [t], [m])
            if fresh:
                print(f"  [fallback] fetched {len(fresh)} fresh articles — re-retrieving")
                chunks = hyde.retrieve(patient, hyp)
                graded = self_rag.grade_all(chunks, hyp.label, patient.to_text())
                used    = sum(1 for g in graded if g.action == "use")
                crag_ct = sum(1 for g in graded if g.action == "crag")
                print(f"  [Self-RAG] re-graded → use={used}  crag={crag_ct}")

        for gc in graded:
            if gc.action == "use" and gc.key_finding:
                hyp.evidence.append(Evidence(
                    text=gc.key_finding,
                    support=gc.support,
                    weight=gc.utility * gc.relevance,
                    source=gc.source,
                ))
                if gc.support == "supports":
                    hyp.supporting_factors.append(gc.key_finding)
                elif gc.support == "contradicts":
                    hyp.opposing_factors.append(gc.key_finding)
                new_evidence_lines.append(
                    f"[{gc.support}] {gc.key_finding} (source: {gc.source})"
                )

            # ── CRAG on contradictions ────────────────────────────
            elif gc.action == "crag":
                delta = crag.correct(gc, hyp, patient.to_text())
                print(f"  [CRAG]     contradiction resolved  delta={delta:+.2f}")
                hyp.adjust(delta)
                if gc.key_finding:
                    hyp.opposing_factors.append(f"CRAG contradiction check: {gc.key_finding}")

    # ── LLM confidence update ─────────────────────────────────────
    if new_evidence_lines:
        diff_str = "\n".join(f"  {h.label}: {h.confidence:.2f}" for h in hypotheses)
        ev_str   = "\n".join(f"  - {e}" for e in new_evidence_lines[:8])
        prompt   = CONFIDENCE_UPDATE_PROMPT.format(
            patient=patient.to_text()[:400],
            differential=diff_str,
            evidence=ev_str,
        )
        raw = llm.ask_json(prompt, max_tokens=200)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                updates = json.loads(m.group())
                for h in hypotheses:
                    for label, score in updates.items():
                        if label.lower() == h.label.lower():
                            h.confidence = max(0.0, min(1.0, float(score)))
        except (json.JSONDecodeError, ValueError):
            pass

    # Print confidence delta table
    print(f"  ── confidence update {'─'*30}")
    for h in sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]:
        before = conf_before.get(h.label, h.confidence)
        arrow = (f"{before:.2f} → {h.confidence:.2f}"
                 if abs(h.confidence - before) > 0.001
                 else f"{h.confidence:.2f} (unchanged)")
        print(f"    {arrow}  {h.label}")
        h.confidence_components["post_rag_confidence"] = round(h.confidence, 3)
        h.confidence_components["rag_iteration"] = float(iteration)

    # Determine whether to stop the loop
    new_top_conf = max(h.confidence for h in hypotheses)
    rag_done = (
        new_top_conf >= config.CONFIDENCE_HIGH
        or iteration >= config.MAX_RAG_ITERATIONS
    )
    if rag_done:
        reason = (f"confidence {new_top_conf:.2f} ≥ {config.CONFIDENCE_HIGH}"
                  if new_top_conf >= config.CONFIDENCE_HIGH
                  else f"max iterations ({config.MAX_RAG_ITERATIONS}) reached")
        print(f"  → RAG done: {reason}")

    return {
        "hypotheses": hypotheses,
        "rag_iteration": iteration,
        "rag_done": rag_done,
        "reasoning_trace": [
            f"RAG iteration {iteration} completed; top confidence is {new_top_conf:.0%} ({_confidence_band(new_top_conf)})."
        ],
        "rag_iterations_run": iteration,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Layer 5 — Output nodes
# ──────────────────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = r"""You are a clinical attending physician summarizing a diagnostic workup.

Patient: {patient}

Final differential after evidence review:
{differential}

Specialist agent context:
{context}

Provide:
1. 3–5 specific recommended next steps
2. Any safety concerns

Respond with ONLY a JSON object:
{{"next_steps": ["step1", "step2"], "safety": ["concern1"]}}"""


CLARIFICATION_PROMPT = r"""The diagnostic confidence is low (top: {conf:.0%}).

Patient: {patient}
Differential: {differential}

Write 1–2 targeted follow-up questions that would most resolve the
uncertainty.  Focus on pertinent positives/negatives the clinician
could quickly check.

Respond with ONLY a JSON array of question strings:
["Question 1?", "Question 2?"]"""


def synthesize(state: AgentCDSState) -> dict:
    """
    Layer 5 — Synthesis node.
    Generates recommended next steps and additional safety concerns.
    """
    patient: Patient = state["patient"]
    hypotheses: list[Hypothesis] = state["hypotheses"]
    enriched: str = state.get("enriched_context", "")

    print("\n[Layer 5] Synthesizing output …")

    diff_str = "\n".join(
        f"  {h.label} ({h.confidence:.0%}) [{h.urgency}]"
        for h in sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
    )
    prompt = SYNTHESIS_PROMPT.format(
        patient=patient.to_text()[:500],
        differential=diff_str,
        context=enriched[:400],
    )
    raw = llm.ask_json(prompt, max_tokens=600)

    next_steps: list[str] = []
    safety_flags: list[str] = []
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            next_steps   = data.get("next_steps", [])
            safety_flags = data.get("safety", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    print(f"  [Synthesis] next_steps={len(next_steps)}  safety_flags={len(safety_flags)}")

    return {
        "next_steps": next_steps,
        "drug_warnings": safety_flags,   # operator.add — appended to pharma warnings
        "reasoning_trace": [
            f"Synthesis produced {len(next_steps)} next step(s) and {len(safety_flags)} safety flag(s)."
        ],
    }


def clarify(state: AgentCDSState) -> dict:
    """
    Layer 5 — Clarification node.
    Generates follow-up questions for the clinician if top confidence
    is still below the CONFIDENCE_LOW threshold.
    """
    hypotheses: list[Hypothesis] = state["hypotheses"]
    patient: Patient = state["patient"]

    top_conf = max(h.confidence for h in hypotheses)
    if top_conf >= config.CONFIDENCE_LOW:
        print(f"[Layer 5] Confidence {top_conf:.0%} ≥ threshold — no clarifications needed")
        return {
            "clarifications": [],
            "uncertainty_factors": [],
            "reasoning_trace": ["Confidence above low threshold; no clarification prompts required."],
        }

    print(f"[Layer 5] Confidence {top_conf:.0%} below threshold — generating clarifications")

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
    clarifications: list[str] = []
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            clarifications = json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    print(f"  [Clarify]  {len(clarifications)} question(s) generated")
    return {
        "clarifications": clarifications,
        "uncertainty_factors": clarifications,
        "reasoning_trace": [f"Generated {len(clarifications)} clarification question(s) due to low confidence."],
    }


def build_result(state: AgentCDSState) -> dict:
    """
    Layer 5 — Final assembly node.
    Packages everything into a DiagnosticResult and prints the summary.
    """
    ordered = sorted(state["hypotheses"], key=lambda h: h.confidence, reverse=True)
    top_conf = ordered[0].confidence if ordered else 0.0
    top_hyp = ordered[0] if ordered else None
    uncertainty = list(state.get("uncertainty_factors", []))
    if top_hyp is not None:
        uncertainty.extend(top_hyp.missing_data[:3])

    result = DiagnosticResult(
        patient_id=state["patient"].patient_id,
        differential=ordered,
        next_steps=state.get("next_steps", []),
        clarifications=state.get("clarifications", []),
        drug_warnings=state.get("drug_warnings", []),
        rag_iterations=state.get("rag_iterations_run", 0),
        confidence_band=_confidence_band(top_conf),
        uncertainty_factors=list(dict.fromkeys(uncertainty))[:6],
        reasoning_trace=state.get("reasoning_trace", []),
    )
    print(result.summary())
    return {"result": result}


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge function for the RAG loop
# ──────────────────────────────────────────────────────────────────────────────

def should_continue_rag(state: AgentCDSState) -> str:
    """
    Routing function consumed by a conditional edge.

    Returns:
        "rag_node"  — loop back for another RAG iteration
        "synthesize" — exit the loop and proceed to output layer
    """
    if state.get("rag_done", False):
        return "synthesize"
    return "rag_node"
