"""Date cascade: when a parent's deadline moves, its is_part_of subtree stays
consistent — subtasks on the old deadline ride along, subtasks left past the new
date clamp in, and independent/undated subtasks are left alone."""
import json

from app.database import models
from app.engine.brain import execute_tool_call, _confirmation_sentences


def update(node_id, **fields):
    return json.loads(execute_tool_call(fake_update(node_id, **fields)))


def fake_update(node_id, **fields):
    from tests.test_tools import fake_call
    return fake_call("update_task", {"node_id": node_id, **fields})


def project_with_children(deadline, children):
    """children: list of (content, date_or_None). Returns (project_id, {content: id})."""
    project = models.add_node("Launch site", node_type="project", target_date=deadline)
    ids = {}
    for content, date in children:
        cid = models.add_node(content, target_date=date)
        models.add_edge(project, cid, "is_part_of")
        ids[content] = cid
    return project, ids


def date_of(node_id):
    return models.get_node(node_id)["target_date"]


# ---------------------------------------------------------------------------
# Moving later — ride the deadline
# ---------------------------------------------------------------------------

def test_move_later_rides_deadline_bound_subtask_only():
    project, ids = project_with_children("2026-06-01", [
        ("write copy", "2026-06-01"),     # on the deadline — should ride out
        ("register domain", "2026-05-20"),  # earlier interim date — should stay
        ("pick a name", None),              # undated — should stay undated
    ])
    result = update(project, deadline="2026-06-15")

    assert date_of(ids["write copy"]) == "2026-06-15"
    assert date_of(ids["register domain"]) == "2026-05-20"
    assert date_of(ids["pick a name"]) is None
    assert len(result["cascaded"]) == 1
    assert result["cascade_direction"] == "out"


# ---------------------------------------------------------------------------
# Moving earlier — clamp the ceiling
# ---------------------------------------------------------------------------

def test_move_earlier_clamps_violators_and_rides_pinned():
    project, ids = project_with_children("2026-06-01", [
        ("final QA", "2026-06-20"),       # past the new date — clamp in
        ("write copy", "2026-06-01"),     # on the old deadline — rides in
        ("register domain", "2026-05-05"),  # safely before — stays
    ])
    result = update(project, deadline="2026-05-15")

    assert date_of(ids["final QA"]) == "2026-05-15"
    assert date_of(ids["write copy"]) == "2026-05-15"
    assert date_of(ids["register domain"]) == "2026-05-05"
    assert len(result["cascaded"]) == 2
    assert result["cascade_direction"] == "in"


# ---------------------------------------------------------------------------
# When nothing should cascade
# ---------------------------------------------------------------------------

def test_no_deadline_change_does_not_cascade():
    project, ids = project_with_children("2026-06-01", [("final QA", "2026-06-20")])
    result = update(project, priority="high")  # touches priority, not the date
    assert "cascaded" not in result
    assert date_of(ids["final QA"]) == "2026-06-20"  # left as-is


def test_same_date_is_a_noop():
    project, ids = project_with_children("2026-06-01", [("final QA", "2026-06-20")])
    result = update(project, deadline="2026-06-01")  # unchanged
    assert "cascaded" not in result
    assert date_of(ids["final QA"]) == "2026-06-20"


def test_undated_parent_gaining_a_date_only_clamps():
    project, ids = project_with_children(None, [
        ("final QA", "2026-06-20"),       # past the new ceiling — clamp
        ("register domain", "2026-05-01"),  # before it — stays
    ])
    result = update(project, deadline="2026-06-01")
    assert date_of(ids["final QA"]) == "2026-06-01"
    assert date_of(ids["register domain"]) == "2026-05-01"
    assert result["cascade_direction"] == "in"  # no old date to ride from


# ---------------------------------------------------------------------------
# Nesting, completed nodes, multi-home
# ---------------------------------------------------------------------------

def test_nested_grandchild_rides_when_its_parent_shifts():
    project, ids = project_with_children("2026-06-01", [("milestone", "2026-06-01")])
    grandchild = models.add_node("sub-step", target_date="2026-06-01")
    models.add_edge(ids["milestone"], grandchild, "is_part_of")

    result = update(project, deadline="2026-06-15")
    assert date_of(ids["milestone"]) == "2026-06-15"   # rides
    assert date_of(grandchild) == "2026-06-15"          # rides via its parent's shift
    assert len(result["cascaded"]) == 2


def test_completed_subtask_is_never_touched():
    project, ids = project_with_children("2026-06-01", [("final QA", "2026-06-20")])
    done = models.add_node("old step", target_date="2026-06-25")
    models.add_edge(project, done, "is_part_of")
    models.complete_nodes([done])

    update(project, deadline="2026-05-15")
    assert date_of(ids["final QA"]) == "2026-05-15"  # active one clamps
    assert date_of(done) == "2026-06-25"             # completed one untouched


def test_multi_home_grandchild_adjusted_once():
    # Diamond: project → A and B, both → shared grandchild G, all on the deadline.
    project, ids = project_with_children("2026-06-01", [
        ("track A", "2026-06-01"), ("track B", "2026-06-01")])
    shared = models.add_node("shared step", target_date="2026-06-01")
    models.add_edge(ids["track A"], shared, "is_part_of")
    models.add_edge(ids["track B"], shared, "is_part_of")

    result = update(project, deadline="2026-06-15")
    assert date_of(shared) == "2026-06-15"
    # A, B, and the shared grandchild — counted exactly once despite two paths.
    assert len(result["cascaded"]) == 3
    assert sum(1 for c in result["cascaded"] if c["id"] == shared) == 1


# ---------------------------------------------------------------------------
# Confirmation sentence
# ---------------------------------------------------------------------------

def test_confirmation_mentions_cascade_direction():
    project, _ = project_with_children("2026-06-01", [("write copy", "2026-06-01")])
    out = update(project, deadline="2026-06-15")
    lines = _confirmation_sentences("update_task", out)
    assert any("out to keep pace" in s for s in lines)

    project2, _ = project_with_children("2026-06-01", [("final QA", "2026-06-20")])
    inn = update(project2, deadline="2026-05-15")
    lines2 = _confirmation_sentences("update_task", inn)
    assert any("in so they stay before it" in s for s in lines2)
