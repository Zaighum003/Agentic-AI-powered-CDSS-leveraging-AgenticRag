"""
Project configuration — reads from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN   = os.getenv("HF_TOKEN")
LLM_MODEL  = os.getenv("LLM_MODEL", "BioMistral/BioMistral-7B")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PUBMED_KEY = (os.getenv("PUBMED_KEY") or "").strip() or None

# LLM backend selection
# local: run model via transformers on this machine
# openai: call OpenAI Chat Completions API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

# OpenAI settings (used when LLM_PROVIDER=openai)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip() or None
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

# RAG settings
TOP_K              = 6      # chunks retrieved per hypothesis
MAX_RAG_ITERATIONS = 3      # max agentic loop iterations
RELEVANCE_CUTOFF   = 0.55   # Self-RAG: discard below this relevance
CRAG_THRESHOLD     = 0.70   # Self-RAG: trigger CRAG above this when contradicting
CONFIDENCE_HIGH    = 0.75   # stop looping, confident enough
CONFIDENCE_LOW     = 0.40   # trigger clarification questions
MAX_DIFFERENTIAL   = 5      # top-N diagnoses to track
