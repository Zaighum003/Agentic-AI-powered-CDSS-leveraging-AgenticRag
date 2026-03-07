"""
Agentic RAG Pipeline
=====================
The core research contribution: iterative, hypothesis-driven retrieval.

Each iteration:
  1. HyDE retrieval — embed a hypothetical note, not the raw query
  2. Self-RAG grading — relevance, support type, utility scoring
  3. CRAG — corrective re-retrieval on contradictions
  4. Confidence update — LLM updates all hypothesis scores based on evidence
  5. Stop early if top confidence >= CONFIDENCE_HIGH

This loop runs up to MAX_RAG_ITERATIONS times.
"""
import json
import re
from agentcds import llm, config
from agentcds.schemas import Patient, Hypothesis
from agentcds.rag import hyde, self_rag, crag


CONFIDENCE_UPDATE_PROMPT = r"""You are a clinical reasoning AI.

Patient: {patient}

Current differential:
{differential}

New evidence retrieved:
{evidence}

Update the confidence score (0.0-1.0) for each hypothesis based on this evidence.
Respond with ONLY a JSON object mapping each label to its new confidence:
{{"Diagnosis A": 0.65, "Diagnosis B": 0.20}}"""


def _update_confidences(
    hypotheses: list[Hypothesis],
    evidence_lines: list[str],
    patient_summary: str,
) -> None:
    """Ask the LLM to update all hypothesis confidences given the new evidence."""
    if not evidence_lines:
        return

    diff_str = "\n".join(f"  {h.label}: {h.confidence:.2f}" for h in hypotheses)
    ev_str   = "\n".join(f"  - {e}" for e in evidence_lines[:8])

    prompt = CONFIDENCE_UPDATE_PROMPT.format(
        patient=patient_summary[:400],
        differential=diff_str,
        evidence=ev_str,
    )
    raw = llm.ask_json(prompt, max_tokens=200)

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return
        updates = json.loads(match.group())
        for h in hypotheses:
            for label, score in updates.items():
                if label.lower() == h.label.lower():
                    h.confidence = max(0.0, min(1.0, float(score)))
    except (json.JSONDecodeError, ValueError):
        pass


def run(patient: Patient, hypotheses: list[Hypothesis]) -> int:
    """
    Run the Agentic RAG loop.

    Mutates hypothesis.confidence and hypothesis.evidence in-place.
    Returns the number of iterations actually run.
    """
    iterations = 0

    for iteration in range(config.MAX_RAG_ITERATIONS):
        iterations += 1
        top_conf = max(h.confidence for h in hypotheses)

        # Stop early if already confident
        if top_conf >= config.CONFIDENCE_HIGH and iteration > 0:
            break

        # Only work on the top 3 hypotheses per iteration (efficiency)
        active = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
        new_evidence_lines = []

        for hypothesis in active:
            # Step 1: HyDE retrieval
            chunks = hyde.retrieve(patient, hypothesis)

            # Step 2: Self-RAG grading
            graded = self_rag.grade_all(chunks, hypothesis.label, patient.to_text())

            for gc in graded:
                if gc.action == "use" and gc.key_finding:
                    from agentcds.schemas import Evidence
                    hypothesis.evidence.append(Evidence(
                        text=gc.key_finding,
                        support=gc.support,
                        weight=gc.utility * gc.relevance,
                        source=gc.source,
                    ))
                    new_evidence_lines.append(
                        f"[{gc.support}] {gc.key_finding} (source: {gc.source})"
                    )

                # Step 3: CRAG on contradictions
                elif gc.action == "crag":
                    delta = crag.correct(gc, hypothesis, patient.to_text())
                    hypothesis.adjust(delta)

        # Step 4: LLM updates all confidences based on accumulated evidence
        _update_confidences(hypotheses, new_evidence_lines, patient.to_text())

    return iterations
