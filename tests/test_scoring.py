from datetime import datetime, timedelta, timezone

from app.engine import scoring


NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_node(node_id, content="task", *, target_date=None, created_at=None,
              status="active", node_type="task", priority="normal"):
    return {
        "id": node_id,
        "content": content,
        "status": status,
        "target_date": target_date,
        "created_at": created_at or "2026-01-01 00:00:00",
        "node_type": node_type,
        "priority": priority,
    }


def ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Today view
# ---------------------------------------------------------------------------

def test_today_shows_only_qualifying_items_not_padded_to_seven():
    today = datetime.now().date()
    nodes = [
        make_node(1, target_date=today.isoformat()),
        make_node(2, target_date=(today + timedelta(days=3)).isoformat()),
        make_node(3, priority="high"),
    ]
    # Plenty of active-but-irrelevant nodes that must NOT appear
    nodes += [make_node(10 + i) for i in range(10)]

    lens = scoring.compute_today(nodes, [], now=NOW)
    assert {entry["id"] for entry in lens} == {1, 2, 3}


def test_today_caps_at_seven():
    today = datetime.now().date()
    nodes = [make_node(i, target_date=today.isoformat()) for i in range(1, 15)]
    lens = scoring.compute_today(nodes, [], now=NOW)
    assert len(lens) == scoring.MAX_LENS_CARDS


def test_empty_today_when_nothing_qualifies():
    nodes = [make_node(i) for i in range(1, 6)]
    assert scoring.compute_today(nodes, [], now=NOW) == []


def test_blocker_of_qualifying_node_is_promoted_and_ranked_above_it():
    today = datetime.now().date()
    blocked = make_node(1, target_date=today.isoformat())
    blocker = make_node(2)  # no deadline of its own
    edges = [{"parent_id": 2, "child_id": 1, "relationship": "blocks"}]
    lens = scoring.compute_today([blocked, blocker], edges, now=NOW)
    ids = [entry["id"] for entry in lens]
    assert ids.index(2) < ids.index(1)
    assert next(e for e in lens if e["id"] == 2)["category"] == scoring.CATEGORY_NEXT


def test_recently_captured_node_qualifies_briefly():
    new = make_node(1, created_at=ts(NOW - timedelta(minutes=10)))
    old = make_node(2, created_at=ts(NOW - timedelta(hours=10)))
    lens = scoring.compute_today([new, old], [], now=NOW)
    assert [entry["id"] for entry in lens] == [1]


def test_categories_encode_urgency():
    today = datetime.now().date()
    due_today = make_node(2, target_date=today.isoformat())
    due_this_week = make_node(3, target_date=(today + timedelta(days=6)).isoformat())
    fresh_undated = make_node(1, created_at=ts(NOW - timedelta(minutes=5)))
    lens = scoring.compute_today([fresh_undated, due_today, due_this_week], [], now=NOW)
    by_id = {entry["id"]: entry for entry in lens}
    assert by_id[1]["category"] == scoring.CATEGORY_UNDATED
    assert by_id[2]["category"] == scoring.CATEGORY_NEXT
    assert by_id[3]["category"] == scoring.CATEGORY_SOON


def test_high_priority_outranks_normal_with_same_deadline():
    today = datetime.now().date()
    normal = make_node(1, target_date=today.isoformat())
    important = make_node(2, target_date=today.isoformat(), priority="high")
    lens = scoring.compute_today([normal, important], [], now=NOW)
    assert [e["id"] for e in lens] == [2, 1]


def test_high_priority_qualifies_without_deadline():
    important = make_node(1, priority="high")
    plain = make_node(2)
    lens = scoring.compute_today([important, plain], [], now=NOW)
    assert [e["id"] for e in lens] == [1]
    assert lens[0]["category"] == scoring.CATEGORY_UNDATED  # stripe carries importance


def test_low_priority_needs_imminent_deadline():
    today = datetime.now().date()
    low_far = make_node(1, target_date=(today + timedelta(days=5)).isoformat(), priority="low")
    low_now = make_node(2, target_date=today.isoformat(), priority="low")
    lens = scoring.compute_today([low_far, low_now], [], now=NOW)
    assert [e["id"] for e in lens] == [2]


# ---------------------------------------------------------------------------
# Project detail view
# ---------------------------------------------------------------------------

def _project_graph():
    today = datetime.now().date()
    nodes = [
        make_node(1, "The Cage", node_type="project"),
        make_node(2, "Undated chore"),
        make_node(3, "Due tomorrow", target_date=(today + timedelta(days=1)).isoformat()),
        make_node(4, "Far out but high", target_date=(today + timedelta(days=40)).isoformat(),
                  priority="high"),
        make_node(5, "Other project's task"),
        make_node(6, "Low priority extra", priority="low"),
    ]
    edges = [
        {"parent_id": 1, "child_id": 2, "relationship": "is_part_of"},
        {"parent_id": 1, "child_id": 3, "relationship": "is_part_of"},
        {"parent_id": 1, "child_id": 4, "relationship": "is_part_of"},
        {"parent_id": 1, "child_id": 6, "relationship": "is_part_of"},
    ]
    return nodes, edges


def test_project_detail_shows_all_members_no_cap():
    nodes, edges = _project_graph()
    cards = scoring.compute_project_detail(nodes, edges, project_id=1)
    assert {c["id"] for c in cards} == {2, 3, 4, 6}  # everything, even undated/low


def test_project_detail_sorted_by_urgency_then_importance():
    nodes, edges = _project_graph()
    cards = scoring.compute_project_detail(nodes, edges, project_id=1)
    ids = [c["id"] for c in cards]
    assert ids[0] == 3                    # imminent deadline first
    assert ids.index(4) < ids.index(2)    # dated+high beats undated
    assert ids[-1] == 6                   # low priority sinks


def test_project_detail_excludes_other_projects_tasks():
    nodes, edges = _project_graph()
    cards = scoring.compute_project_detail(nodes, edges, project_id=1)
    assert 5 not in {c["id"] for c in cards}


# ---------------------------------------------------------------------------
# Projects overview
# ---------------------------------------------------------------------------

def test_projects_overview_counts_and_nearest_deadline():
    today = datetime.now().date()
    soon = (today + timedelta(days=2)).isoformat()
    later = (today + timedelta(days=20)).isoformat()
    nodes = [
        make_node(1, "Film", node_type="project"),
        make_node(2, "Cut trailer", target_date=later),
        make_node(3, "Submit festival", target_date=soon),
        make_node(4, "Quiet project", node_type="project"),
        make_node(5, "Someday step"),
    ]
    edges = [
        {"parent_id": 1, "child_id": 2, "relationship": "is_part_of"},
        {"parent_id": 1, "child_id": 3, "relationship": "is_part_of"},
        {"parent_id": 4, "child_id": 5, "relationship": "is_part_of"},
    ]
    cards = scoring.compute_projects_overview(nodes, edges)
    film = next(c for c in cards if c["id"] == 1)
    assert film["project_open_total"] == 2
    assert film["target_date"] == soon          # nearest deadline wins
    assert cards[0]["id"] == 1                  # urgent project ranks first
    assert next(c for c in cards if c["id"] == 4)["target_date"] is None


def test_projects_overview_loose_tasks_pseudo_card():
    nodes = [
        make_node(1, "Project", node_type="project"),
        make_node(2, "Project step"),
        make_node(3, "Floating errand"),
    ]
    edges = [{"parent_id": 1, "child_id": 2, "relationship": "is_part_of"}]
    cards = scoring.compute_projects_overview(nodes, edges)
    pseudo = cards[-1]
    assert pseudo["pseudo"] == "loose"
    assert pseudo["project_open_total"] == 1    # only the floating errand
    # and a graph with nothing loose gets no pseudo-card
    cards = scoring.compute_projects_overview(nodes[:2], edges)
    assert all(not c.get("pseudo") for c in cards)


def test_loose_tasks_view_excludes_parented_and_projects():
    nodes = [
        make_node(1, "Project", node_type="project"),
        make_node(2, "Project step"),
        make_node(3, "Floating errand"),
    ]
    edges = [{"parent_id": 1, "child_id": 2, "relationship": "is_part_of"}]
    cards = scoring.compute_loose_tasks(nodes, edges)
    assert [c["id"] for c in cards] == [3]


# ---------------------------------------------------------------------------
# Annotation (project grouping + suppression)
# ---------------------------------------------------------------------------

def test_annotate_adds_project_names():
    from app.database import models
    project = models.add_node("The Cage", node_type="project")
    task_a = models.add_node("Submit FIN Atlantic",
                             target_date=datetime.now().date().isoformat())
    models.add_edge(project, task_a, "is_part_of")

    nodes = models.get_active_nodes()
    edges = models.get_all_edges()
    cards = scoring.annotate_projects(scoring.compute_today(nodes, edges), nodes, edges)
    card = next(c for c in cards if c["id"] == task_a)
    assert card["project_name"] == "The Cage"
    assert card["project_open_total"] == 1


def test_multi_home_task_appears_in_both_project_details():
    nodes = [
        make_node(1, "The Cage", node_type="project"),
        make_node(2, "AI Ethics Brand", node_type="project"),
        make_node(3, "Draft Globe op-ed"),
    ]
    # filed under The Cage FIRST, then AI Ethics Brand
    edges = [
        {"parent_id": 1, "child_id": 3, "relationship": "is_part_of"},
        {"parent_id": 2, "child_id": 3, "relationship": "is_part_of"},
    ]
    assert [c["id"] for c in scoring.compute_project_detail(nodes, edges, 1)] == [3]
    assert [c["id"] for c in scoring.compute_project_detail(nodes, edges, 2)] == [3]


def test_multi_home_primary_project_is_first_assigned():
    from app.database import models
    cage = models.add_node("The Cage", node_type="project")
    ethics = models.add_node("AI Ethics Brand", node_type="project")
    op_ed = models.add_node("Draft Globe op-ed", target_date=datetime.now().date().isoformat())
    models.add_edge(cage, op_ed, "is_part_of")     # first → primary
    models.add_edge(ethics, op_ed, "is_part_of")

    nodes = models.get_active_nodes()
    edges = models.get_all_edges()
    cards = scoring.annotate_projects(scoring.compute_today(nodes, edges), nodes, edges)
    card = next(c for c in cards if c["id"] == op_ed)
    assert card["project_name"] == "The Cage"  # first-assigned wins, not last


def test_project_card_suppressed_when_its_tasks_are_visible():
    from app.database import models
    project = models.add_node("Visible Project", node_type="project",
                              target_date=datetime.now().date().isoformat())
    task = models.add_node("Visible task", target_date=datetime.now().date().isoformat())
    models.add_edge(project, task, "is_part_of")
    lonely = models.add_node("Lonely Project", node_type="project",
                             target_date=datetime.now().date().isoformat())

    nodes = models.get_active_nodes()
    edges = models.get_all_edges()
    cards = scoring.annotate_projects(scoring.compute_today(nodes, edges), nodes, edges)
    ids = {c["id"] for c in cards}
    assert task in ids
    assert project not in ids       # header represents it; card would be noise
    assert lonely in ids            # sole representation of its project stays
