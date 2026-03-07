"""
CRAG — Corrective RAG
======================
When a high-relevance chunk contradicts the working hypothesis,
CRAG fires a targeted corrective search to adjudicate the conflict:

  1. Ask the LLM: "Generate a search query that resolves this contradiction"
  2. Re-retrieve from vector store with the new query
  3. Ask the LLM: "Does this new evidence resolve the contradiction?"
  4. Return a confidence delta based on the adjudication

This prevents the system from silently ignoring contradicting evidence.
"""
import json
import re
from agentcds import llm, vector_store
from agentcds.schemas import Hypothesis
from agentcds.rag.self_rag import GradedChunk


QUERY_PROMPT = """A retrieved document CONTRADICTS the hypothesis: "{hypothesis}"

The contradiction: {contradiction}

Write a focused medical search query to find literature that would
adjudicate whether this finding truly rules out the diagnosis.
Respond with ONLY the search query, nothing else."""


ADJUDICATION_PROMPT = """Hypothesis: {hypothesis}
Contradiction: {contradiction}

Adjudication literature:
---
{evidence}
---

Does this literature resolve the contradiction in favor of the hypothesis,
or does it confirm the hypothesis should be down-ranked?

Respond with ONLY a JSON object:
{{"resolved": <true|false>, "confidence_delta": <-0.25 to +0.15>, "explanation": "<one sentence>"}}"""


def correct(contradiction: GradedChunk, hypothesis: Hypothesis, patient_summary: str) -> float:
    """
    Run CRAG for one contradicting chunk.
    Returns a confidence delta (negative = down-rank, positive = resolved).
    """
    # Step 1: generate corrective query
    query_prompt = QUERY_PROMPT.format(
        hypothesis=hypothesis.label,
        contradiction=contradiction.key_finding or contradiction.text[:300],
    )
    corrective_query = llm.ask(query_prompt, max_tokens=60, temperature=0.1).strip()

    # Step 2: re-retrieve
    new_chunks = vector_store.search_all(corrective_query, k=3)
    adj_text = "\n\n".join(c["text"][:400] for c in new_chunks[:3]) or "No adjudication evidence found."

    # Step 3: adjudicate
    adj_prompt = ADJUDICATION_PROMPT.format(
        hypothesis=hypothesis.label,
        contradiction=contradiction.key_finding or contradiction.text[:200],
        evidence=adj_text,
    )
    raw = llm.ask(adj_prompt, max_tokens=150, temperature=0.05)

    delta = -0.05  # default: slight down-rank if adjudication fails
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            delta  = float(result.get("confidence_delta", -0.05))
    except (json.JSONDecodeError, AttributeError):
        pass

    return delta
