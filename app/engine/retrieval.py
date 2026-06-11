"""Hybrid task retrieval: FTS5 keyword matches plus semantic embedding matches.

FTS catches exact/stemmed word overlap ("cage" → The Cage tasks); embeddings
catch paraphrase and category references ("the documentary", "errands"). FTS
results rank first — an exact word match is stronger evidence than cosine
proximity — and semantic hits fill the remaining slots.
"""
from typing import Any, Dict, List

from app.database import models
from app.engine import embeddings


def search_active(text: str, limit: int = 15) -> List[Dict[str, Any]]:
    keyword_hits = models.search_active_nodes(text, limit=limit)
    found = {node["id"] for node in keyword_hits}
    merged = list(keyword_hits)
    # Semantic candidates are noisier than keyword hits; cap their share so a
    # vague query can't fill the whole pool with near-floor matches.
    semantic_cap = min(6, limit)
    for node in embeddings.semantic_search(text, limit=semantic_cap):
        if node["id"] not in found:
            merged.append(node)
    return merged[:limit]
