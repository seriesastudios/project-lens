"""Embedding store, cosine search, and hybrid retrieval — embed_text is mocked
so tests run without LM Studio."""
import numpy as np
import pytest

from app.database import models
from app.engine import embeddings, retrieval

# Hand-built unit vectors on distinct axes; "similar" texts share an axis
VECTORS = {
    "Buy groceries for the week": [1.0, 0.0, 0.0],
    "shopping": [0.9, 0.1, 0.0],
    "Finish The Cage rough cut": [0.0, 1.0, 0.0],
    "the documentary": [0.1, 0.9, 0.0],
    "Send invoices to clients": [0.0, 0.0, 1.0],
}


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def embed(text, is_query=False):
        if text not in VECTORS:
            return None
        vector = np.asarray(VECTORS[text], dtype=np.float32)
        return vector / np.linalg.norm(vector)
    monkeypatch.setattr(embeddings, "embed_text", embed)


def seed_indexed():
    ids = {}
    for content in ("Buy groceries for the week", "Finish The Cage rough cut",
                    "Send invoices to clients"):
        node_id = models.add_node(content)
        assert embeddings.index_node(node_id, content)
        ids[content] = node_id
    return ids


def test_semantic_search_finds_paraphrase():
    ids = seed_indexed()
    results = embeddings.semantic_search("the documentary")
    assert results
    assert results[0]["id"] == ids["Finish The Cage rough cut"]
    assert results[0]["similarity"] > 0.8


def test_semantic_search_respects_similarity_floor():
    seed_indexed()
    # "shopping" is close to groceries only; others fall below the floor
    results = embeddings.semantic_search("shopping", min_similarity=0.55)
    assert [r["content"] for r in results] == ["Buy groceries for the week"]


def test_semantic_search_excludes_completed():
    ids = seed_indexed()
    models.complete_nodes([ids["Finish The Cage rough cut"]])
    assert embeddings.semantic_search("the documentary") == []


def test_semantic_search_empty_when_server_down(monkeypatch):
    seed_indexed()
    monkeypatch.setattr(embeddings, "embed_text", lambda text, is_query=False: None)
    assert embeddings.semantic_search("the documentary") == []


def test_hybrid_retrieval_merges_keyword_and_semantic():
    ids = seed_indexed()
    # "documentary" has zero keyword overlap with any task — semantic only
    results = retrieval.search_active("the documentary")
    assert ids["Finish The Cage rough cut"] in {n["id"] for n in results}
    # keyword match still works and ranks first when both fire
    results = retrieval.search_active("Buy groceries for the week")
    assert results[0]["id"] == ids["Buy groceries for the week"]


def test_backfill_indexes_missing_nodes(monkeypatch):
    node_id = models.add_node("Buy groceries for the week")
    assert embeddings.backfill() == 1
    assert embeddings.backfill() == 0  # idempotent
    row = models.get_node(node_id)
    assert row is not None
