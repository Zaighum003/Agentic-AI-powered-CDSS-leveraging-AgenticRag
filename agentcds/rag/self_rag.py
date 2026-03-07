"""
Self-RAG — Evidence Grading
=============================
After retrieval, each chunk is graded by the LLM on:
  - relevance_score (0–1): how relevant to this hypothesis?
  - support: "supports" | "contradicts" | "neutral"
  - utility (0–1): will this meaningfully change confidence?

Chunks below the relevance cutoff are discarded.
High-relevance contradicting chunks trigger CRAG.
"""
import json
import re
from dataclasses import dataclass
from agentcds import llm, config


GRADING_PROMPT = """You are grading a medical text chunk for its relevance to a clinical hypothesis.

Hypothesis: {hypothesis}
Patient context: {patient_summary}

Text to grade:
---
{chunk}
---

Respond with ONLY a JSON object (no markdown):
{{"relevance": <0.0-1.0>, "support": "<supports|contradicts|neutral>", "utility": <0.0-1.0>, "key_finding": "<one sentence>"}}"""


@dataclass
class GradedChunk:
    text:        str
    source:      str
    relevance:   float
    support:     str    # supports | contradicts | neutral
    utility:     float
    key_finding: str
    action:      str    # use | discard | crag


def grade(chunk: dict, hypothesis_label: str, patient_summary: str) -> GradedChunk:
    """Grade a single retrieved chunk against the current hypothesis."""
    prompt = GRADING_PROMPT.format(
        hypothesis=hypothesis_label,
        patient_summary=patient_summary[:400],
        chunk=chunk["text"][:700],
    )
    raw = llm.ask(prompt, max_tokens=150, temperature=0.05)

    # Parse — with a safe fallback
    parsed = {"relevance": 0.3, "support": "neutral", "utility": 0.2, "key_finding": ""}
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    relevance = float(parsed.get("relevance", 0.3))
    support   = str(parsed.get("support", "neutral")).lower()
    utility   = float(parsed.get("utility", 0.2))

    # Decide action
    if relevance < config.RELEVANCE_CUTOFF:
        action = "discard"
    elif support == "contradicts" and relevance > config.CRAG_THRESHOLD:
        action = "crag"
    elif utility < 0.3:
        action = "discard"
    else:
        action = "use"

    return GradedChunk(
        text=chunk["text"],
        source=chunk.get("meta", {}).get("source", "vector_store"),
        relevance=relevance, support=support,
        utility=utility,
        key_finding=str(parsed.get("key_finding", "")),
        action=action,
    )


def grade_all(chunks: list[dict], hypothesis_label: str, patient_summary: str) -> list[GradedChunk]:
    """Grade a list of chunks and return all results."""
    return [grade(c, hypothesis_label, patient_summary) for c in chunks]
