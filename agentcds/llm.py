"""
Thin wrapper around HuggingFace Inference API.
Uses the free serverless endpoint — no local GPU needed.

Swap LLM_MODEL in .env to try different medical models:
  BioMistral/BioMistral-7B   (general medical reasoning)
  epfl-llm/meditron-7b        (clinical guideline focus)
  microsoft/BioGPT-Large      (biomedical generation, smaller)
"""
from huggingface_hub import InferenceClient
from agentcds import config


_client = None

def get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(model=config.LLM_MODEL, token=config.HF_TOKEN)
    return _client


def ask(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Send a prompt to the medical LLM and return the response text."""
    formatted = f"<s>[INST] {prompt} [/INST]"  # Mistral instruction format
    response = get_client().text_generation(
        formatted,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        do_sample=True,
        return_full_text=False,
    )
    return response.strip()


def ask_json(prompt: str, max_tokens: int = 400) -> str:
    """
    Ask the LLM for a JSON response.
    Appends explicit JSON-only instruction and returns the raw string.
    Caller is responsible for json.loads().
    """
    json_prompt = (
        prompt
        + "\n\nRespond ONLY with valid JSON — no markdown, no explanation, just the JSON object or array."
    )
    return ask(json_prompt, max_tokens=max_tokens, temperature=0.05)
