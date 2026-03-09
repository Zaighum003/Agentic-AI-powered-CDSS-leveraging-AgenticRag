"""
ChromaDB vector store for clinical knowledge retrieval.

Collections:
  guidelines   — Clinical practice guidelines (load once, persists)
  cache        — PubMed abstracts fetched during sessions (ephemeral)
"""
import hashlib
import os
from pathlib import Path
from openai import OpenAI
from agentcds import config


# ChromaDB reads ".env" during import via pydantic Settings.
# Temporarily import it from this package directory (which has no .env)
# so project-level app keys don't get treated as Chroma settings.
_ORIG_CWD = os.getcwd()
try:
    os.chdir(Path(__file__).resolve().parent)
    import chromadb
finally:
    os.chdir(_ORIG_CWD)

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        base_url = (config.OPENAI_BASE_URL or "").strip() or "https://api.openai.com/v1"
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=base_url)
    return _openai_client


def _embed(texts: list[str]) -> list[list[float]]:
    response = _get_client().embeddings.create(input=texts, model=config.EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


_db     = chromadb.Client()  # in-memory; pass PersistentClient(path=...) for persistence
_cols   = {
    name: _db.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    for name in ("guidelines", "cache")
}


def add(collection: str, texts: list[str], metas: list[dict] = None):
    """Embed and store documents in a collection."""
    embs = _embed(texts)
    ids  = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]
    metas = metas or [{} for _ in texts]
    _cols[collection].upsert(embeddings=embs, documents=texts, metadatas=metas, ids=ids)


def search(collection: str, query: str, k: int = None) -> list[dict]:
    """
    Retrieve top-k chunks most similar to query.
    Returns list of {"text": ..., "meta": ..., "score": ...}
    """
    k = k or config.TOP_K
    col = _cols[collection]
    if col.count() == 0:
        return []

    emb = _embed([query])
    res = col.query(
        query_embeddings=emb,
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({"text": doc, "meta": meta or {}, "score": 1.0 - dist})
    return chunks


def search_all(query: str, k: int = None) -> list[dict]:
    """Search across all collections and merge results."""
    results = []
    for col in _cols:
        results.extend(search(col, query, k=k))
    return sorted(results, key=lambda x: x["score"], reverse=True)[: k or config.TOP_K]
