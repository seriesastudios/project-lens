from datetime import datetime, timedelta, timezone

from app.engine import scoring


NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_node(node_id, content="task", *, focus_score=0.0, focused_at=None,
              target_date=None, created_at=None, status="active"):
    return {
        "id": node_id,
        "content": content,
        "status": status,
        "focus_score": focus_score,
        "focused_at": focused_at,
        "target_date": target_date,
        "created_at": created_at or "2026-01-01 00:00:00",
        "node_type": "task",
    }


def ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def test_lens_shows_only_qualifying_items_not_padded_to_seven():
    today = datetime.now().date()
    nodes = [
        make_node(1, target_date=today.isoformat()),
        make_node(2, target_date=(today + timedelta(days=3)).isoformat()),
        make_node(3, focus_score=10.0, focused_at=ts(NOW - timedelta(hours=1))),
    ]
    # Plenty of active-but-irrelevant nodes that must NOT appear
    nodes += [make_node(10 + i) for i in range(10)]

    lens = scoring.compute_lens(nodes, [], now=NOW)
    assert {entry["id"] for entry in lens} == {1, 2, 3}


def test_lens_caps_at_seven():
    today = datetime.now().date()
    nodes = [make_node(i, target_date=today.isoformat()) for i in range(1, 15)]
    lens = scoring.compute_lens(nodes, [], now=NOW)
    assert len(lens) == scoring.MAX_LENS_CARDS


def test_empty_lens_when_nothing_qualifies():
    nodes = [make_node(i) for i in range(1, 6)]
    assert scoring.compute_lens(nodes, [], now=NOW) == []


def test_focus_decays_over_time():
    fresh = make_node(1, focus_score=10.0, focused_at=ts(NOW - timedelta(hours=1)))
    stale = make_node(2, focus_score=10.0, focused_at=ts(NOW - timedelta(hours=48)))
    lens = scoring.compute_lens([fresh, stale], [], now=NOW)
    assert [entry["id"] for entry in lens] == [1]


def test_blocker_of_qualifying_node_is_promoted_and_ranked_above_it():
    today = datetime.now().date()
    blocked = make_node(1, target_date=today.isoformat())
    blocker = make_node(2)  # no deadline/focus of its own
    edges = [{"parent_id": 2, "child_id": 1, "relationship": "blocks"}]
    lens = scoring.compute_lens([blocked, blocker], edges, now=NOW)
    ids = [entry["id"] for entry in lens]
    assert ids.index(2) < ids.index(1)
    assert next(e for e in lens if e["id"] == 2)["category"] == scoring.CATEGORY_NEXT


def test_recently_captured_node_qualifies_briefly():
    new = make_node(1, created_at=ts(NOW - timedelta(minutes=10)))
    old = make_node(2, created_at=ts(NOW - timedelta(hours=10)))
    lens = scoring.compute_lens([new, old], [], now=NOW)
    assert [entry["id"] for entry in lens] == [1]


def test_categories():
    today = datetime.now().date()
    focused = make_node(1, focus_score=10.0, focused_at=ts(NOW - timedelta(minutes=5)))
    due_today = make_node(2, target_date=today.isoformat())
    horizon = make_node(3, target_date=(today + timedelta(days=6)).isoformat())
    lens = scoring.compute_lens([focused, due_today, horizon], [], now=NOW)
    by_id = {entry["id"]: entry["category"] for entry in lens}
    assert by_id[1] == scoring.CATEGORY_FOCUS
    assert by_id[2] == scoring.CATEGORY_NEXT
    assert by_id[3] == scoring.CATEGORY_HORIZON


def test_high_priority_outranks_normal_with_same_deadline():
    today = datetime.now().date()
    normal = make_node(1, target_date=today.isoformat())
    important = make_node(2, target_date=today.isoformat())
    important["priority"] = "high"
    lens = scoring.compute_lens([normal, important], [], now=NOW)
    assert [e["id"] for e in lens] == [2, 1]


def test_high_priority_qualifies_without_deadline():
    important = make_node(1)
    important["priority"] = "high"
    plain = make_node(2)
    lens = scoring.compute_lens([important, plain], [], now=NOW)
    assert [e["id"] for e in lens] == [1]
    assert lens[0]["category"] == scoring.CATEGORY_HORIZON


def test_low_priority_needs_imminent_deadline():
    today = datetime.now().date()
    low_far = make_node(1, target_date=(today + timedelta(days=5)).isoformat())
    low_far["priority"] = "low"
    low_now = make_node(2, target_date=today.isoformat())
    low_now["priority"] = "low"
    lens = scoring.compute_lens([low_far, low_now], [], now=NOW)
    assert [e["id"] for e in lens] == [2]


def test_high_priority_ranks_below_due_today_normal():
    today = datetime.now().date()
    due_today = make_node(1, target_date=today.isoformat())
    important_undated = make_node(2)
    important_undated["priority"] = "high"
    lens = scoring.compute_lens([due_today, important_undated], [], now=NOW)
    assert [e["id"] for e in lens] == [1, 2]


def test_get_lens_state_annotates_projects_and_focus(monkeypatch):
    from app.database import models
    project = models.add_node("The Cage", node_type="project")
    task_a = models.add_node("Submit FIN Atlantic", target_date=datetime.now().date().isoformat())
    task_b = models.add_node("Review composer contract", target_date=datetime.now().date().isoformat())
    models.add_edge(project, task_a, "is_part_of")
    models.add_edge(project, task_b, "is_part_of")
    models.set_focus([task_a, task_b])

    state = scoring.get_lens_state()
    cards = {c["id"]: c for c in state["cards"]}
    assert cards[task_a]["project_name"] == "The Cage"
    assert cards[task_a]["project_open_total"] == 2

    focus = state["focus"]
    assert focus["label"] == "The Cage"
    assert focus["count"] == 2
    assert focus["hours_left"] is not None and focus["hours_left"] > 0

    models.clear_all_focus()
    assert scoring.get_lens_state()["focus"] is None
