"""LLM wrapper with pluggable backends (local transformers or OpenAI API)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agentcds import config


_tokenizer = None
_model = None
_openai_client = None


def _get_local_model():
    """Load tokenizer/model once and keep them cached for this process."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL,
        token=config.HF_TOKEN or None,
        use_fast=True,
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        token=config.HF_TOKEN or None,
        torch_dtype=dtype,
    )
    _model.eval()
    return _tokenizer, _model


def _get_openai_client():
    """Create and cache OpenAI client for hosted chat-completions models."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OpenAI LLM selected but OPENAI_API_KEY is missing.")

    # Lazy import keeps local mode working even if openai package is absent.
    from openai import OpenAI

    base_url = (config.OPENAI_BASE_URL or "").strip() or "https://api.openai.com/v1"
    kwargs = {
        "api_key": config.OPENAI_API_KEY,
        "base_url": base_url,
    }

    _openai_client = OpenAI(**kwargs)
    return _openai_client


def _ask_openai(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate with OpenAI Chat Completions API."""
    client = _get_openai_client()
    model = config.LLM_MODEL
    # o1/o3/o4 and gpt-5.x are reasoning/new-gen models: use max_completion_tokens
    # and skip temperature (not supported or ignored).
    reasoning_or_new = (
        any(model.startswith(p) for p in ("o1", "o3", "o4"))
        or model.startswith("gpt-5")
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if not reasoning_or_new:
        kwargs["temperature"] = max(temperature, 0.01)
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message.content if response.choices else ""
    return (msg or "").strip()


def _ask_local(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate a response using a local causal LLM."""
    tokenizer, model = _get_local_model()

    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = f"<s>[INST] {prompt} [/INST]"

    inputs = tokenizer(formatted, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.01),
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def ask(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate a response using configured backend in config.LLM_PROVIDER."""
    provider = (config.LLM_PROVIDER or "local").strip().lower()
    if provider == "openai":
        return _ask_openai(prompt, max_tokens=max_tokens, temperature=temperature)
    if provider == "local":
        return _ask_local(prompt, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER!r}. Use 'local' or 'openai'.")


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
