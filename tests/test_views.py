"""Tests for the view state machine: sticky navigation, deterministic project
resolution, and stale-view fallbacks."""
from datetime import datetime, timedelta

from app.database import models
from app.engine import views


def today_plus(days=0):
    return (datetime.now().date() + timedelta(days=days)).isoformat()


def seed_project(name="The Cage", task_count=3):
    project = models.add_node(name, node_type="project")
    children = []
    for i in range(task_count):
        child = models.add_node(f"{name} step {i}", target_date=today_plus(i))
        models.add_edge(project, child, "is_part_of")
        children.append(child)
    return project, children


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def test_default_view_is_today():
    assert views.get_view() == {"mode": "today"}


def test_view_is_sticky_in_the_database():
    project, _ = seed_project()
    views.set_view({"mode": "node", "path": [project]})
    # a fresh read (as after a server restart) returns the same view
    assert views.get_view() == {"mode": "node", "path": [project]}


def test_corrupt_or_invalid_state_falls_back_to_today():
    models.set_state(views.VIEW_STATE_KEY, "not json{")
    assert views.get_view() == {"mode": "today"}
    models.set_state(views.VIEW_STATE_KEY, '{"mode": "warp"}')
    assert views.get_view() == {"mode": "today"}
    models.set_state(views.VIEW_STATE_KEY, '{"mode": "node"}')  # missing path
    assert views.get_view() == {"mode": "today"}


def test_set_view_rejects_invalid_shapes():
    import pytest
    with pytest.raises(ValueError):
        views.set_view({"mode": "nonsense"})
    with pytest.raises(ValueError):
        views.set_view({"mode": "node", "path": []})        # empty path
    with pytest.raises(ValueError):
        views.set_view({"mode": "list", "node_ids": ["a"]})


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------

def test_resolve_project_exact_and_case_insensitive():
    project, _ = seed_project("The Cage")
    resolved = views.resolve_project("the cage")
    assert resolved["id"] == project


def test_resolve_project_partial_name():
    project, _ = seed_project("Mr. Spring and Mrs. Fresh")
    resolved = views.resolve_project("mr. spring")
    assert resolved["id"] == project


def test_resolve_project_ambiguous_returns_candidates():
    a, _ = seed_project("Website redesign", task_count=1)
    b, _ = seed_project("Website copywriting", task_count=1)
    resolved = views.resolve_project("website")
    assert isinstance(resolved, list)
    assert {p["id"] for p in resolved} == {a, b}


def test_resolve_project_no_match_returns_empty_list():
    seed_project("The Cage", task_count=1)
    assert views.resolve_project("quarterly taxes") == []


def test_resolve_project_ignores_tasks_with_matching_names():
    models.add_node("Email the cage distributor")  # a task, not a project
    project, _ = seed_project("The Cage", task_count=1)
    resolved = views.resolve_project("cage")
    assert resolved["id"] == project


# ---------------------------------------------------------------------------
# compute_view_cards dispatch + fallbacks
# ---------------------------------------------------------------------------

def test_compute_view_cards_node_mode_shows_all_children():
    project, children = seed_project(task_count=4)
    # noise outside the project must not appear
    models.add_node("Unrelated urgent", target_date=today_plus(0))
    views.set_view({"mode": "node", "path": [project]})

    state = views.compute_view_cards()
    assert state["view"]["mode"] == "node"
    assert state["view"]["label"] == "The Cage"
    assert state["view"]["breadcrumb"] == [{"id": project, "label": "The Cage"}]
    assert state["view"]["back"] is True
    assert {c["id"] for c in state["cards"]} == set(children)


def test_drill_into_a_subtask_container():
    project, children = seed_project(task_count=2)
    parent = children[0]
    sub_a = models.add_node("Sub A", target_date=today_plus(1))
    sub_b = models.add_node("Sub B")
    models.add_edge(parent, sub_a, "is_part_of")
    models.add_edge(parent, sub_b, "is_part_of")
    views.set_view({"mode": "node", "path": [project, parent]})

    state = views.compute_view_cards()
    assert {c["id"] for c in state["cards"]} == {sub_a, sub_b}
    assert [b["id"] for b in state["view"]["breadcrumb"]] == [project, parent]


def test_stale_path_tail_is_trimmed():
    project, children = seed_project(task_count=2)
    parent = children[0]
    sub = models.add_node("Sub")
    models.add_edge(parent, sub, "is_part_of")
    views.set_view({"mode": "node", "path": [project, parent]})
    models.complete_nodes([parent])  # the deeper container is gone

    state = views.compute_view_cards()
    assert views.get_view() == {"mode": "node", "path": [project]}  # trimmed, not dropped
    assert {c["id"] for c in state["cards"]} == set(children) - {parent}


def test_completed_project_view_falls_back_to_today():
    project, _children = seed_project()
    views.set_view({"mode": "node", "path": [project]})
    models.complete_nodes([project])

    state = views.compute_view_cards()
    assert state["view"]["mode"] == "today"
    assert views.get_view() == {"mode": "today"}  # fallback persisted


def test_emptied_list_view_falls_back_to_today():
    a = models.add_node("Errand A")
    views.set_view({"mode": "list", "node_ids": [a], "label": "errands"})
    models.complete_nodes([a])

    state = views.compute_view_cards()
    assert state["view"]["mode"] == "today"


def test_list_view_filters_to_still_active():
    a = models.add_node("Errand A", target_date=today_plus(0))
    b = models.add_node("Errand B")
    views.set_view({"mode": "list", "node_ids": [a, b], "label": "errands"})
    models.complete_nodes([b])

    state = views.compute_view_cards()
    assert state["view"]["label"] == "errands"
    assert [c["id"] for c in state["cards"]] == [a]


def test_view_member_ids_for_context():
    project, children = seed_project()
    views.set_view({"mode": "node", "path": [project]})
    assert set(views.view_member_ids()) == {project, *children}
    views.set_view({"mode": "today"})
    assert views.view_member_ids() == []


# ---------------------------------------------------------------------------
# Filter views
# ---------------------------------------------------------------------------

def test_filter_view_overdue_shows_only_past_dated():
    models.add_node("late thing", target_date=today_plus(-5))
    models.add_node("future thing", target_date=today_plus(5))
    views.set_view({"mode": "filter", "filter": "overdue"})
    out = views.compute_view_cards()
    assert out["view"]["label"] == "Overdue"
    contents = {c["content"] for c in out["cards"]}
    assert "late thing" in contents
    assert "future thing" not in contents


def test_filter_view_waiting_reads_on_hold():
    held = models.add_node("parked task")
    models.update_node(held, status="on_hold")
    views.set_view({"mode": "filter", "filter": "waiting"})
    out = views.compute_view_cards()
    assert out["view"]["label"] == "Waiting on"
    assert any(c["content"] == "parked task" for c in out["cards"])


def test_filter_view_done_reads_completed_and_flags_them():
    d = models.add_node("did this one")
    models.complete_nodes([d])
    views.set_view({"mode": "filter", "filter": "done"})
    out = views.compute_view_cards()
    assert out["view"]["label"] == "Done this week"
    card = next(c for c in out["cards"] if c["content"] == "did this one")
    assert card.get("done") is True


def test_filter_view_high_priority():
    models.add_node("flagged", priority="high")
    models.add_node("ordinary")
    views.set_view({"mode": "filter", "filter": "high"})
    out = views.compute_view_cards()
    assert out["view"]["label"] == "High priority"
    contents = {c["content"] for c in out["cards"]}
    assert "flagged" in contents and "ordinary" not in contents


def test_empty_filter_view_renders_cleanly():
    views.set_view({"mode": "filter", "filter": "overdue"})
    out = views.compute_view_cards()
    assert out["cards"] == []
    assert out["view"]["label"] == "Overdue"
