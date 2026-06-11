"""Tests for retrieval-based prompt context: tasks live in the DB; the prompt
only gets the subset relevant to the conversation."""
from app.database import models
from app.engine import brain


def seed_many(count=40):
    ids = {}
    for i in range(count):
        models.add_node(f"Generic background chore number {i}")
    ids["invoices"] = models.add_node("Send invoices to clients")
    ids["deadline"] = models.add_node("File quarterly taxes", target_date="2026-06-13")
    ids["focused"] = models.add_node("Design new logo")
    models.set_focus([ids["focused"]])
    return ids


def test_fts_search_matches_stemmed_words():
    models.add_node("Send invoices to clients")
    models.add_node("Walk the dog")
    results = models.search_active_nodes("just sent off the invoices")
    assert [r["content"] for r in results][0] == "Send invoices to clients"


def test_fts_search_excludes_completed():
    node_id = models.add_node("Send invoices to clients")
    models.complete_nodes([node_id])
    assert models.search_active_nodes("invoices") == []


def test_small_graphs_include_everything():
    for i in range(5):
        models.add_node(f"Task {i}")
    active = models.get_active_nodes()
    assert brain.select_context_tasks(active, "unrelated message") == active


def test_large_graphs_get_relevant_subset_only():
    ids = seed_many()
    active = models.get_active_nodes()
    chosen = brain.select_context_tasks(active, "I just sent the invoices")
    chosen_ids = {n["id"] for n in chosen}

    assert ids["invoices"] in chosen_ids   # FTS match on the message
    assert ids["deadline"] in chosen_ids   # deadline inside the window
    assert ids["focused"] in chosen_ids    # currently focused
    assert len(chosen) <= brain.CONTEXT_MAX_TASKS
    assert len(chosen) < len(active)


def test_prompt_notes_omitted_tasks():
    seed_many()
    prompt = brain.format_system_prompt("I just sent the invoices")
    assert "Send invoices to clients" in prompt
    assert "more active tasks" in prompt
    assert "Generic background chore number 7" not in prompt
