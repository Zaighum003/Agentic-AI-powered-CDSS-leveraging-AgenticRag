"""
HyDE — Hypothetical Document Embeddings
========================================
Instead of embedding the raw patient query (which may use lay language),
HyDE asks the LLM to write a hypothetical ideal clinical note assuming
a given diagnosis is correct, then embeds THAT document.

Why this helps: the hypothetical note uses the same clinical vocabulary
as the indexed literature, so cosine similarity retrieval works much better.
"""
from agentcds import llm, vector_store
from agentcds.schemas import Patient, Hypothesis


PROMPT = """You are a senior physician. Write a concise 2-paragraph clinical note
for a patient with the diagnosis: {diagnosis}

Assume the diagnosis is correct. Include typical clinical features,
relevant lab findings, and key pertinent negatives.
Use standard medical terminology. Be specific."""


def retrieve(patient: Patient, hypothesis: Hypothesis, k: int = None) -> list[dict]:
    """
    Retrieve evidence chunks using the HyDE strategy.

    1. Generate a hypothetical clinical note for this diagnosis
    2. Embed the hypothetical note (not the patient text)
    3. Retrieve similar chunks from the vector store
    """
    prompt   = PROMPT.format(diagnosis=hypothesis.label)
    hypo_doc = llm.ask(prompt, max_tokens=300, temperature=0.2)
    return vector_store.search_all(hypo_doc, k=k)
