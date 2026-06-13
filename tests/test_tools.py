import json
from types import SimpleNamespace

from app.database import models
from app.engine.brain import execute_tool_call


def fake_call(name, arguments):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_capture_multiple_tasks_with_subtasks():
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [
            {"content": "Plan launch", "node_type": "project",
             "subtasks": ["Draft email", "Book venue"]},
            {"content": "Renew passport", "deadline": "2026-08-31"},
        ]
    })))
    assert result["success"]
    assert len(result["created"]) == 4  # project + 2 subtasks + standalone task

    project = models.find_node_by_content("Plan launch", node_type="project")
    children = models.get_edges_for_node(project["id"])
    assert len(children) == 2
    assert all(edge["relationship"] == "is_part_of" for edge in children)

    passport = models.find_node_by_content("Renew passport")
    assert passport["target_date"] == "2026-08-31"


def test_capture_rejects_hallucinated_parent_id():
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [{"content": "Orphan", "parent_id": 555}]
    })))
    assert "error" in result
    assert models.find_node_by_content("Orphan") is None


def test_capture_rejects_garbage_deadline():
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [{"content": "Bad date", "deadline": "whenever the vibes align"}]
    })))
    assert "error" in result


def test_capture_normalizes_natural_language_deadline():
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [{"content": "Natural date", "deadline": "tomorrow"}]
    })))
    assert result["success"]
    node = models.find_node_by_content("Natural date")
    assert node["target_date"] is not None
    assert len(node["target_date"]) == 10  # YYYY-MM-DD


def test_complete_tasks_reports_unknown_ids():
    real = models.add_node("Real task")
    result = json.loads(execute_tool_call(fake_call("complete_tasks", {
        "node_ids": [real, 9999]
    })))
    assert result["completed_ids"] == [real]
    assert result["unknown_ids"] == [9999]
    assert models.get_node(real)["status"] == "completed"


def test_update_task_unknown_id_errors():
    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": 31337, "deadline": "2026-07-01"
    })))
    assert "error" in result


def test_update_task_changes_status_and_deadline():
    node_id = models.add_node("Pausable")
    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": node_id, "status": "on_hold", "deadline": "2026-07-15"
    })))
    assert result["success"]
    node = models.get_node(node_id)
    assert node["status"] == "on_hold"
    assert node["target_date"] == "2026-07-15"


def test_link_tasks_blocks_relationship():
    blocker = models.add_node("Finish mockups")
    blocked = models.add_node("Build landing page")
    result = json.loads(execute_tool_call(fake_call("link_tasks", {
        "parent_id": blocker, "child_id": blocked, "relationship": "blocks"
    })))
    assert result["success"]
    edges = models.get_edges_for_node(blocker)
    assert edges[0]["relationship"] == "blocks"


def test_open_view_project_resolves_name_and_sets_view():
    from app.engine import views
    project = models.add_node("Website redesign", node_type="project")
    child = models.add_node("Finish mockups for site")
    models.add_edge(project, child, "is_part_of")

    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "project", "project_name": "website redesign"})))
    assert result["success"]
    assert result["project"] == "Website redesign"
    assert result["open_tasks"] == 1
    assert views.get_view() == {"mode": "node", "path": [project]}


def test_open_view_unknown_project_errors_with_guidance():
    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "project", "project_name": "moon base"})))
    assert "error" in result


def test_open_view_ambiguous_project_returns_candidates():
    models.add_node("Website redesign", node_type="project")
    models.add_node("Website copywriting", node_type="project")
    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "project", "project_name": "website"})))
    assert "error" in result
    assert len(result["candidates"]) == 2


def test_open_view_on_hold_project_suggests_resume():
    models.add_node("Shelved film", node_type="project", status="on_hold")
    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "project", "project_name": "shelved film"})))
    assert "error" in result and "on hold" in result["error"]


def test_open_view_today_and_projects_modes():
    from app.engine import views
    models.add_node("Some project", node_type="project")
    result = json.loads(execute_tool_call(fake_call("open_view", {"view": "projects"})))
    assert result["success"] and result["project_count"] == 1
    assert views.get_view() == {"mode": "projects"}

    result = json.loads(execute_tool_call(fake_call("open_view", {"view": "today"})))
    assert result["success"]
    assert views.get_view() == {"mode": "today"}


def test_open_view_list_filters_inactive_ids():
    from app.engine import views
    a = models.add_node("Errand A")
    b = models.add_node("Errand B")
    models.complete_nodes([b])
    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "list", "node_ids": [a, b, 999], "label": "errands"})))
    assert result["success"] and result["shown"] == 1
    assert views.get_view() == {"mode": "list", "node_ids": [a], "label": "errands"}

    result = json.loads(execute_tool_call(fake_call("open_view", {
        "view": "list", "node_ids": [999]})))
    assert "error" in result


def test_open_view_validation_requires_mode_fields():
    result = json.loads(execute_tool_call(fake_call("open_view", {"view": "project"})))
    assert "error" in result
    result = json.loads(execute_tool_call(fake_call("open_view", {"view": "list"})))
    assert "error" in result


def test_invalid_json_arguments():
    call = SimpleNamespace(id="x", function=SimpleNamespace(name="complete_tasks", arguments="{not json"))
    result = json.loads(execute_tool_call(call))
    assert "error" in result


def test_unknown_function():
    result = json.loads(execute_tool_call(fake_call("drop_database", {})))
    assert "error" in result


def test_salvage_parses_leaked_open_view_json_and_kwargs():
    from app.engine.brain import salvage_tool_call
    call = salvage_tool_call('open_view {"view": "project", "project_name": "The Cage"}')
    assert call.function.name == "open_view"
    assert json.loads(call.function.arguments) == {"view": "project", "project_name": "The Cage"}

    call = salvage_tool_call('open_view[view="project", project_name="The Cage"]')
    assert json.loads(call.function.arguments) == {"view": "project", "project_name": "The Cage"}

    call = salvage_tool_call("open_view(view=today)")
    assert json.loads(call.function.arguments) == {"view": "today"}


def test_salvage_parses_bare_id_list_and_dict_args():
    from app.engine.brain import salvage_tool_call
    call = salvage_tool_call("completed: complete_tasks([2, 6])")
    assert json.loads(call.function.arguments) == {"node_ids": [2, 6]}
    call = salvage_tool_call('update_task {"node_id": 3, "status": "on_hold"}')
    assert call.function.name == "update_task"


def test_salvage_returns_none_for_plain_text():
    from app.engine.brain import salvage_tool_call
    assert salvage_tool_call("Sounds like a busy day! Anything to capture?") is None
    assert salvage_tool_call("") is None


def test_normalize_deadline_relative_phrases():
    from datetime import datetime
    from app.engine.brain import normalize_deadline
    base = datetime(2026, 6, 11, 12, 0)  # a Thursday
    assert normalize_deadline("Friday", base) == "2026-06-12"
    assert normalize_deadline("by Friday", base) == "2026-06-12"
    assert normalize_deadline("next Monday", base) == "2026-06-15"
    assert normalize_deadline("end of month", base) == "2026-06-30"
    assert normalize_deadline("end of the week", base) == "2026-06-14"
    assert normalize_deadline("2026-08-31", base) == "2026-08-31"
    assert normalize_deadline(None, base) is None


def test_capture_skips_exact_duplicate_of_active_task():
    models.add_node("Pick a template")
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [{"content": "Pick a template"}, {"content": "Genuinely new task"}]
    })))
    assert result["success"]
    assert [c["content"] for c in result["created"]] == ["Genuinely new task"]
    assert result["skipped_duplicates"][0]["content"] == "Pick a template"


def test_salvage_parses_kwargs_style_leak():
    from app.engine.brain import salvage_tool_call
    call = salvage_tool_call("complete_tasks[node_ids=[6,5]]")
    assert call.function.name == "complete_tasks"
    assert json.loads(call.function.arguments) == {"node_ids": [6, 5]}


def test_search_tasks_tool_returns_ids_for_followup():
    models.add_node("Colour grade The Cage final cut")
    models.add_node("Walk the dog")
    result = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "cage grade"})))
    assert result["results"][0]["content"] == "Colour grade The Cage final cut"
    assert "open_view" in result["hint"]
    empty = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "zzzqqq"})))
    assert empty["results"] == []


def test_grounding_rejects_unseen_ids_and_allows_seen():
    from app.engine import brain
    brain.reset_session()
    real = models.add_node("Grounded task")
    result = json.loads(execute_tool_call(
        fake_call("open_view", {"view": "list", "node_ids": [real], "label": "x"}),
        enforce_grounding=True))
    assert "error" in result and "search_tasks" in result["error"]

    search = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "grounded"})))
    assert search["results"][0]["id"] == real
    result = json.loads(execute_tool_call(
        fake_call("open_view", {"view": "list", "node_ids": [real], "label": "x"}),
        enforce_grounding=True))
    assert result["success"]
    brain.reset_session()


def test_open_view_project_needs_no_grounding():
    # Navigation by NAME must work even for projects never shown to the model —
    # resolution happens in Python, so there are no IDs to ground.
    from app.engine import brain
    brain.reset_session()
    models.add_node("Never-mentioned project", node_type="project")
    result = json.loads(execute_tool_call(
        fake_call("open_view", {"view": "project", "project_name": "never-mentioned project"}),
        enforce_grounding=True))
    assert result["success"]
    brain.reset_session()


def test_capture_and_update_priority():
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [{"content": "Ship the critical fix", "priority": "high"}]
    })))
    node_id = result["created"][0]["id"]
    assert models.get_node(node_id)["priority"] == "high"

    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": node_id, "priority": "low"
    })))
    assert result["changed"] == {"priority": "low"}
    assert models.get_node(node_id)["priority"] == "low"

    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": node_id, "priority": "urgent-ish"
    })))
    assert "error" in result


def test_capture_subtasks_under_an_existing_task():
    # parent_id may be any task, so capture files new items as its subtasks.
    parent = models.add_node("Prep for Picture Shop")
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [
            {"content": "Lock credits", "parent_id": parent},
            {"content": "Foley check", "parent_id": parent},
        ]
    })))
    assert result["success"]
    child_ids = models.get_active_child_ids(parent)
    assert len(child_ids) == 2
    contents = {models.get_node(c)["content"] for c in child_ids}
    assert contents == {"Lock credits", "Foley check"}


def test_update_task_promotes_task_to_project_and_detaches():
    # "make X its own project" flips node_type AND removes the is_part_of edge
    # filing it under its old project; its own subtasks stay attached.
    project = models.add_node("New Scripts", node_type="project")
    wayfinder = models.add_node("Wayfinder")
    sub = models.add_node("Existing beat sheet")
    models.add_edge(project, wayfinder, "is_part_of")
    models.add_edge(wayfinder, sub, "is_part_of")

    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": wayfinder, "node_type": "project"
    })))
    assert result["success"]
    assert result["changed"]["node_type"] == "project"
    assert result["detached"] == 1

    assert models.get_node(wayfinder)["node_type"] == "project"
    # detached from New Scripts, but keeps its own child
    parent_ids = {e["parent_id"] for e in models.get_edges_for_node(wayfinder)
                  if e["child_id"] == wayfinder}
    assert project not in parent_ids
    assert models.get_active_child_ids(wayfinder) == [sub]

    # the confirmation reads as a promotion
    from app.engine.brain import template_confirmation
    assert template_confirmation([("update_task", result)]) == \
        "Promoted **Wayfinder** to its own project."


def test_promote_then_capture_new_tasks_under_it():
    # full sequence: promote an existing task, then file new tasks under it.
    project = models.add_node("New Scripts", node_type="project")
    wayfinder = models.add_node("Wayfinder")
    models.add_edge(project, wayfinder, "is_part_of")

    json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": wayfinder, "node_type": "project"
    })))
    result = json.loads(execute_tool_call(fake_call("capture_tasks", {
        "tasks": [
            {"content": "Write logline", "parent_id": wayfinder},
            {"content": "Draft treatment", "parent_id": wayfinder},
            {"content": "Build pitch deck", "parent_id": wayfinder},
        ]
    })))
    assert result["success"]
    assert len(models.get_active_child_ids(wayfinder)) == 3

    # it now appears in the all-projects overview
    from app.engine import scoring
    overview = scoring.compute_projects_overview(
        models.get_active_nodes(), models.get_all_edges())
    assert wayfinder in {c["id"] for c in overview}


def test_update_task_rejects_bad_node_type():
    node_id = models.add_node("Some task")
    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": node_id, "node_type": "epic"
    })))
    assert "error" in result


def _is_part_of_parents(node_id):
    return {e["parent_id"] for e in models.get_edges_for_node(node_id)
            if e["child_id"] == node_id and e["relationship"] == "is_part_of"}


def test_move_task_to_project_relocates_single_home():
    a = models.add_node("Project A", node_type="project")
    b = models.add_node("Project B", node_type="project")
    task = models.add_node("Draft the op-ed")
    models.add_edge(a, task, "is_part_of")

    result = json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": task, "to_project_name": "Project B"
    })))
    assert result["success"]
    assert result["to"] == "Project B"
    assert result["detached"] == 1
    assert _is_part_of_parents(task) == {b}          # only B now
    assert models.get_node(task)["node_type"] == "task"  # type unchanged


def test_move_task_is_a_true_move_for_multihome():
    a = models.add_node("The Cage", node_type="project")
    b = models.add_node("AI Ethics", node_type="project")
    c = models.add_node("Press", node_type="project")
    op_ed = models.add_node("Globe op-ed")
    models.add_edge(a, op_ed, "is_part_of")
    models.add_edge(b, op_ed, "is_part_of")

    result = json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": op_ed, "to_project_name": "Press"
    })))
    assert result["success"]
    assert result["detached"] == 2
    assert _is_part_of_parents(op_ed) == {c}         # dropped BOTH old homes


def test_move_task_under_another_task_by_id():
    prep = models.add_node("Picture Shop prep")
    foley = models.add_node("Foley check")
    result = json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": foley, "new_parent_id": prep
    })))
    assert result["success"]
    assert foley in models.get_active_child_ids(prep)


def test_move_task_validation_errors():
    t = models.add_node("Lonely task")
    other = models.add_node("Somewhere", node_type="project")
    # neither destination
    assert "error" in json.loads(execute_tool_call(fake_call("move_task", {"node_id": t})))
    # both destinations
    assert "error" in json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": t, "to_project_name": "Somewhere", "new_parent_id": other})))
    # under itself
    assert "error" in json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": t, "new_parent_id": t})))
    # non-existent project destination
    assert "error" in json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": t, "to_project_name": "No Such Project Anywhere"})))


def test_move_task_confirmation_and_grounding():
    from app.engine import brain
    from app.engine.brain import template_confirmation
    dest = models.add_node("Destination project", node_type="project")
    task = models.add_node("Wandering task")

    # confirmation reads as a move
    result = json.loads(execute_tool_call(fake_call("move_task", {
        "node_id": task, "to_project_name": "Destination project"})))
    assert template_confirmation([("move_task", result)]) == \
        "Moved **Wandering task** to **Destination project**."

    # grounding: an unseen node_id is rejected, but a name destination needs no ID grounding
    brain.reset_session()
    grounded = models.add_node("Seen task")
    rejected = json.loads(execute_tool_call(
        fake_call("move_task", {"node_id": grounded, "to_project_name": "Destination project"}),
        enforce_grounding=True))
    assert "error" in rejected  # node_id not yet seen
    models.add_node("noise")
    search = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "seen"})))
    assert any(r["id"] == grounded for r in search["results"])
    ok = json.loads(execute_tool_call(
        fake_call("move_task", {"node_id": grounded, "to_project_name": "Destination project"}),
        enforce_grounding=True))
    assert ok["success"]
    brain.reset_session()


def test_update_task_attaches_description():
    node_id = models.add_node("Lock credits in offline edit")
    result = json.loads(execute_tool_call(fake_call("update_task", {
        "node_id": node_id, "description": "UHD page roll from Poncho, no stage-hour fixes"
    })))
    assert result["changed"] == {"description": "UHD page roll from Poncho, no stage-hour fixes"}
    assert models.get_node(node_id)["description"] == "UHD page roll from Poncho, no stage-hour fixes"


def test_template_confirmation_routine_outcomes():
    from app.engine.brain import template_confirmation
    node_id = models.add_node("Order business cards")
    assert template_confirmation([
        ("capture_tasks", {"success": True, "created": [
            {"id": node_id, "content": "Order business cards", "deadline": "2026-06-17"}]})
    ]) == "Captured **Order business cards** — due 2026-06-17."

    assert template_confirmation([
        ("complete_tasks", {"success": True, "completed_ids": [node_id]})
    ]) == "Checked off **Order business cards**. ✓"

    assert template_confirmation([
        ("update_task", {"success": True, "node_id": node_id,
                         "changed": {"deadline": "2026-07-01", "priority": "high"}})
    ]) == "Updated **Order business cards** — now due 2026-07-01, high priority."

    assert template_confirmation([
        ("open_view", {"success": True, "view": "project", "project": "The Cage", "open_tasks": 11})
    ]) == "Opened **The Cage** — 11 open tasks."
    assert template_confirmation([
        ("open_view", {"success": True, "view": "projects", "project_count": 9})
    ]) == "Showing your **9 projects**."
    assert template_confirmation([
        ("open_view", {"success": True, "view": "list", "label": "errands", "shown": 4})
    ]) == "Showing **errands** — 4 tasks."
    assert template_confirmation([
        ("open_view", {"success": True, "view": "today"})
    ]) == "Here's your day."


def test_template_confirmation_defers_nuanced_cases():
    from app.engine.brain import template_confirmation
    # errors and unknown IDs need the model's judgment
    assert template_confirmation([("capture_tasks", {"error": "bad args"})]) is None
    assert template_confirmation([
        ("complete_tasks", {"success": True, "completed_ids": [1], "unknown_ids": [99]})
    ]) is None
    # a search with no follow-up action has nothing to confirm
    assert template_confirmation([("search_tasks", {"results": [], "hint": "x"})]) is None


def test_template_confirmation_link():
    from app.engine.brain import template_confirmation
    blocker = models.add_node("Finish mockups")
    blocked = models.add_node("Build page")
    text = template_confirmation([
        ("link_tasks", {"success": True, "parent_id": blocker, "child_id": blocked,
                        "relationship": "blocks"})
    ])
    assert text == "Noted: **Finish mockups** blocks **Build page**."
