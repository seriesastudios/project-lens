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


def test_focus_lens_and_clear_focus():
    a = models.add_node("Task A")
    b = models.add_node("Task B")
    result = json.loads(execute_tool_call(fake_call("focus_lens", {"node_ids": [a, b, 777]})))
    assert result["focused_ids"] == [a, b]
    assert result["unknown_ids"] == [777]
    assert models.get_node(a)["focus_score"] == 10.0

    result = json.loads(execute_tool_call(fake_call("clear_focus", {})))
    assert result["success"]
    assert models.get_node(a)["focus_score"] == 0.0


def test_invalid_json_arguments():
    call = SimpleNamespace(id="x", function=SimpleNamespace(name="complete_tasks", arguments="{not json"))
    result = json.loads(execute_tool_call(call))
    assert "error" in result


def test_unknown_function():
    result = json.loads(execute_tool_call(fake_call("drop_database", {})))
    assert "error" in result


def test_salvage_parses_leaked_focus_lens_with_wrong_arg_shape():
    from app.engine.brain import salvage_tool_call
    call = salvage_tool_call('focus_lens [{"node_id": 1, "content": "Buy groceries"}, {"node_id": 2}]')
    assert call.function.name == "focus_lens"
    assert json.loads(call.function.arguments) == {"node_ids": [1, 2]}


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
    call = salvage_tool_call("focus_lens[node_ids=[6,5]]")
    assert call.function.name == "focus_lens"
    assert json.loads(call.function.arguments) == {"node_ids": [6, 5]}


def test_search_tasks_tool_returns_ids_for_followup():
    models.add_node("Colour grade The Cage final cut")
    models.add_node("Walk the dog")
    result = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "cage grade"})))
    assert result["results"][0]["content"] == "Colour grade The Cage final cut"
    assert "focus_lens" in result["hint"]
    empty = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "zzzqqq"})))
    assert empty["results"] == []


def test_grounding_rejects_unseen_ids_and_allows_seen():
    from app.engine import brain
    brain.reset_session()
    real = models.add_node("Grounded task")
    result = json.loads(execute_tool_call(
        fake_call("focus_lens", {"node_ids": [real]}), enforce_grounding=True))
    assert "error" in result and "search_tasks" in result["error"]

    search = json.loads(execute_tool_call(fake_call("search_tasks", {"query": "grounded"})))
    assert search["results"][0]["id"] == real
    result = json.loads(execute_tool_call(
        fake_call("focus_lens", {"node_ids": [real]}), enforce_grounding=True))
    assert result["success"]
    brain.reset_session()
