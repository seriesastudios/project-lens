from app.database import models


def test_add_and_get_node():
    node_id = models.add_node("Write report", target_date="2026-06-20")
    node = models.get_node(node_id)
    assert node["content"] == "Write report"
    assert node["status"] == "active"
    assert node["node_type"] == "task"


def test_existing_node_ids_filters_hallucinated_ids():
    real = models.add_node("Real task")
    assert models.existing_node_ids([real, 9999]) == [real]


def test_complete_nodes_stamps_completed_at():
    node_id = models.add_node("Finish thing")
    completed = models.complete_nodes([node_id, 4242])
    assert completed == [node_id]
    node = models.get_node(node_id)
    assert node["status"] == "completed"
    assert node["completed_at"] is not None


def test_description_roundtrip_and_partial_update():
    node_id = models.add_node("Lock credits in offline edit",
                              description="UHD page roll from Poncho, no stage-hour fixes")
    assert models.get_node(node_id)["description"] == "UHD page roll from Poncho, no stage-hour fixes"
    # updating other fields leaves the description intact
    models.update_node(node_id, priority="high")
    assert models.get_node(node_id)["description"] == "UHD page roll from Poncho, no stage-hour fixes"
    # and it can be replaced on its own
    models.update_node(node_id, description="new note")
    node = models.get_node(node_id)
    assert node["description"] == "new note" and node["content"] == "Lock credits in offline edit"


def test_multi_home_task_has_two_parents():
    cage = models.add_node("The Cage", node_type="project")
    ethics = models.add_node("AI Ethics Brand", node_type="project")
    op_ed = models.add_node("Draft Globe & Mail op-ed")
    models.add_edge(cage, op_ed, "is_part_of")
    models.add_edge(ethics, op_ed, "is_part_of")
    # the task is an active child of BOTH projects
    assert op_ed in models.get_active_child_ids(cage)
    assert op_ed in models.get_active_child_ids(ethics)


def test_app_state_roundtrip_and_overwrite():
    assert models.get_state("view") is None
    models.set_state("view", '{"mode": "today"}')
    assert models.get_state("view") == '{"mode": "today"}'
    models.set_state("view", '{"mode": "projects"}')
    assert models.get_state("view") == '{"mode": "projects"}'


def test_update_node_partial_fields():
    node_id = models.add_node("Draft email")
    assert models.update_node(node_id, target_date="2026-07-01")
    node = models.get_node(node_id)
    assert node["target_date"] == "2026-07-01"
    assert node["content"] == "Draft email"
    assert node["updated_at"] is not None
    assert not models.update_node(31337, content="ghost")


def test_edges_dedupe_on_reinsert():
    parent = models.add_node("Project", node_type="project")
    child = models.add_node("Step 1")
    models.add_edge(parent, child, "is_part_of")
    models.add_edge(parent, child, "is_part_of")
    assert len(models.get_edges_for_node(parent)) == 1


def test_find_node_by_content():
    models.add_node("Lens MVP", node_type="project")
    assert models.find_node_by_content("Lens MVP", node_type="project") is not None
    assert models.find_node_by_content("Lens MVP", node_type="task") is None


def test_update_node_changes_node_type():
    node_id = models.add_node("Wayfinder")
    assert models.get_node(node_id)["node_type"] == "task"
    assert models.update_node(node_id, node_type="project")
    assert models.get_node(node_id)["node_type"] == "project"


def test_detach_parents_removes_only_is_part_of_edges():
    project = models.add_node("New Scripts", node_type="project")
    wayfinder = models.add_node("Wayfinder")
    blocker = models.add_node("Some blocker")
    child = models.add_node("A subtask")
    models.add_edge(project, wayfinder, "is_part_of")   # parent link to drop
    models.add_edge(blocker, wayfinder, "blocks")       # different relationship: keep
    models.add_edge(wayfinder, child, "is_part_of")     # wayfinder as PARENT: keep

    removed = models.detach_parents(wayfinder)
    assert removed == 1
    remaining = models.get_edges_for_node(wayfinder)
    rels = {(e["parent_id"], e["child_id"], e["relationship"]) for e in remaining}
    assert (project, wayfinder, "is_part_of") not in rels
    assert (blocker, wayfinder, "blocks") in rels
    assert (wayfinder, child, "is_part_of") in rels


def test_get_nodes_by_status_on_hold():
    models.add_node("active one")
    held = models.add_node("held one")
    models.update_node(held, status="on_hold")
    result = models.get_nodes_by_status("on_hold")
    assert [n["id"] for n in result] == [held]


def test_get_recently_completed_includes_recent_excludes_active():
    done = models.add_node("finished recently")
    models.complete_nodes([done])
    active = models.add_node("still going")
    ids = {n["id"] for n in models.get_recently_completed()}
    assert done in ids
    assert active not in ids


def test_get_recently_completed_excludes_old():
    old = models.add_node("done long ago")
    models.complete_nodes([old])
    with models.DatabaseSession() as conn:
        conn.execute("UPDATE nodes SET completed_at = datetime('now', '-30 days') WHERE id = ?", (old,))
    assert old not in {n["id"] for n in models.get_recently_completed()}
