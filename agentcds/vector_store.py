"""
ChromaDB vector store for clinical knowledge retrieval.

Collections:
  guidelines   — Clinical practice guidelines (load once, persists)
  cache        — PubMed abstracts fetched during sessions (ephemeral)
"""
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer
from agentcds import config

# Medical-domain embedding model (loads once on first use)
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    return _embedder


_db     = chromadb.Client()  # in-memory; pass PersistentClient(path=...) for persistence
_cols   = {
    name: _db.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    for name in ("guidelines", "cache")
}


def add(collection: str, texts: list[str], metas: list[dict] = None):
    """Embed and store documents in a collection."""
    embs = get_embedder().encode(texts).tolist()
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

    emb = get_embedder().encode([query]).tolist()
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
