"""Semantic retrieval layer: per-node embeddings stored in SQLite.

Embeddings come from LM Studio's local embedding model (nomic-embed), so the
privacy story is unchanged. At hundreds-to-thousands of tasks, brute-force
cosine over normalized float32 vectors is sub-millisecond — a dedicated vector
database would be infrastructure without a problem. Everything degrades
gracefully to FTS-only retrieval when the embedding server is unreachable.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from app.config import config
from app.database import models

# nomic-embed is trained with task prefixes; using them measurably improves retrieval
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

# max_retries=0: embedding is best-effort; a down server must fail fast,
# not stall captures with retry backoff
_client = OpenAI(base_url=config.AI_BASE_URL, api_key="not-needed-for-local", max_retries=0)


def embed_text(text: str, is_query: bool = False) -> Optional[np.ndarray]:
    """Returns a unit-normalized float32 vector, or None if the server is down."""
    prefix = _QUERY_PREFIX if is_query else _DOC_PREFIX
    try:
        response = _client.embeddings.create(
            model=config.EMBEDDING_MODEL_NAME,
            input=prefix + text,
        )
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None
    except Exception:
        return None


def index_node(node_id: int, content: str) -> bool:
    """Computes and stores the embedding for one node. Returns False on failure
    (callers proceed regardless; backfill() will catch up later)."""
    vector = embed_text(content)
    if vector is None:
        return False
    with models.DatabaseSession() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO node_embeddings (node_id, model, vector) VALUES (?, ?, ?)",
            (node_id, config.EMBEDDING_MODEL_NAME, vector.tobytes())
        )
    return True


def backfill() -> int:
    """Indexes every node that has no embedding yet (or one from another model).
    Run at server startup and after seeding. Returns how many were indexed."""
    with models.DatabaseSession() as conn:
        rows = conn.execute(
            '''SELECT n.id, n.content FROM nodes n
               LEFT JOIN node_embeddings e
                 ON n.id = e.node_id AND e.model = ?
               WHERE e.node_id IS NULL''',
            (config.EMBEDDING_MODEL_NAME,)
        ).fetchall()
    indexed = 0
    for row in rows:
        if index_node(row["id"], row["content"]):
            indexed += 1
        else:
            break  # server unreachable; don't hammer it for every row
    return indexed


def semantic_search(text: str, limit: int = 10, min_similarity: float = 0.52) -> List[Dict[str, Any]]:
    """Active nodes ranked by cosine similarity to the query text.
    Returns [] when the embedding server is down or nothing clears the floor.

    The floor is deliberately permissive (calibrated on nomic-embed: true
    matches score 0.54-0.71, noise 0.50-0.62 — the bands overlap). These are
    candidates for the LLM to judge, so recall beats precision; the caller
    caps how many candidates it takes."""
    query_vector = embed_text(text, is_query=True)
    if query_vector is None:
        return []

    with models.DatabaseSession() as conn:
        rows = conn.execute(
            '''SELECT n.*, e.vector AS _vector FROM nodes n
               JOIN node_embeddings e ON n.id = e.node_id
               WHERE n.status = 'active' AND e.model = ?''',
            (config.EMBEDDING_MODEL_NAME,)
        ).fetchall()
    if not rows:
        return []

    matrix = np.frombuffer(b"".join(row["_vector"] for row in rows), dtype=np.float32)
    matrix = matrix.reshape(len(rows), -1)
    similarities = matrix @ query_vector

    ranked = sorted(zip(rows, similarities), key=lambda pair: pair[1], reverse=True)
    results = []
    for row, similarity in ranked[:limit]:
        if similarity < min_similarity:
            break
        node = {k: row[k] for k in row.keys() if k != "_vector"}
        node["similarity"] = round(float(similarity), 3)
        results.append(node)
    return results
