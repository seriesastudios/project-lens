"""View state machine: the Lens is navigable, like a project manager app.

The current view is a small JSON blob persisted in the app_state table, so it
is sticky — it survives turns and server restarts until the user navigates
(by chat or by click). Modes:

    {"mode": "today"}                                  — default: most urgent, capped
    {"mode": "projects"}                               — overview of all projects
    {"mode": "node", "path": [526]}                    — inside one container (project)
    {"mode": "node", "path": [526, 612]}               — Cage ▸ Picture Shop prep (subtasks)
    {"mode": "list", "node_ids": [...], "label": "…"}  — ad-hoc query results
    {"mode": "loose"}                                  — tasks belonging to no project

A node is a *container* iff it has active is_part_of children; the path is the
breadcrumb trail and path[-1] is the container whose children are shown. The
model never hand-picks view membership; it names a target ("The Cage") and
resolve_project / the compute functions turn that into queries deterministically.
"""
import json
from typing import Any, Dict, List, Optional, Union

from app.database import models
from app.engine import scoring

VIEW_STATE_KEY = "view"
DEFAULT_VIEW: Dict[str, Any] = {"mode": "today"}
VALID_MODES = ("today", "projects", "node", "list", "loose", "filter")
VALID_FILTERS = ("overdue", "high", "waiting", "done")


def _valid(view: Any) -> bool:
    if not isinstance(view, dict) or view.get("mode") not in VALID_MODES:
        return False
    if view["mode"] == "node":
        path = view.get("path")
        if not isinstance(path, list) or not path or not all(isinstance(i, int) for i in path):
            return False
    if view["mode"] == "list":
        ids = view.get("node_ids")
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            return False
    if view["mode"] == "filter" and view.get("filter") not in VALID_FILTERS:
        return False
    return True


def get_view() -> Dict[str, Any]:
    raw = models.get_state(VIEW_STATE_KEY)
    if raw:
        try:
            view = json.loads(raw)
        except ValueError:
            view = None
        if isinstance(view, dict) and _valid(view):
            return view
    return dict(DEFAULT_VIEW)


def set_view(view: Dict[str, Any]) -> Dict[str, Any]:
    if not _valid(view):
        raise ValueError(f"Invalid view state: {view!r}")
    models.set_state(VIEW_STATE_KEY, json.dumps(view))
    return view


def resolve_project(name: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Deterministic name → project resolution. Returns the project's node dict
    on a confident unique match; otherwise a (possibly empty) candidate list.
    On-hold projects resolve too — the caller decides how to handle them."""
    from app.engine import retrieval

    needle = (name or "").strip().casefold()
    if not needle:
        return []
    with models.DatabaseSession() as conn:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE node_type = 'project' "
            "AND status IN ('active', 'on_hold')").fetchall()
    projects = [dict(r) for r in rows]

    exact = [p for p in projects if p["content"].casefold() == needle]
    if len(exact) == 1:
        return exact[0]

    partial = [p for p in projects
               if needle in p["content"].casefold() or p["content"].casefold() in needle]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        return partial

    # Fall back to hybrid retrieval (FTS + semantic) filtered to projects, so
    # "the documentary" can still find "Mr. Spring and Mrs. Fresh".
    hits = [n for n in retrieval.search_active(name, limit=10)
            if n.get("node_type") == "project"]
    if len(hits) == 1:
        return hits[0]
    return hits


def _active_path(path: List[int]) -> List[int]:
    """Trims a node path to its valid prefix: stops at the first id that is
    missing or inactive (a completed container collapses the trail below it)."""
    valid: List[int] = []
    for nid in path:
        node = models.get_node(nid)
        if not node or node["status"] != "active":
            break
        valid.append(nid)
    return valid


def view_meta(view: Dict[str, Any], cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The header payload the UI renders: mode, label, breadcrumb trail, back."""
    mode = view["mode"]
    breadcrumb: List[Dict[str, Any]] = []
    if mode == "today":
        label = "Today"
    elif mode == "projects":
        label = "Projects"
    elif mode == "node":
        for nid in view["path"]:
            node = models.get_node(nid)
            breadcrumb.append({"id": nid, "label": node["content"] if node else "?"})
        label = breadcrumb[-1]["label"] if breadcrumb else "?"
    elif mode == "loose":
        label = "Loose tasks"
    elif mode == "filter":
        label = {"overdue": "Overdue", "high": "High priority",
                 "waiting": "Waiting on", "done": "Done this week"}.get(view.get("filter"), "Filtered")
    else:
        label = view.get("label") or "Selection"
    return {"mode": mode, "label": label, "breadcrumb": breadcrumb,
            "count": len(cards), "back": mode != "today"}


def compute_view_cards() -> Dict[str, Any]:
    """Reads the graph and the current view, returns {'view': meta, 'cards': [...]}.
    Stale views (entered container completed, list emptied) fall back gracefully."""
    nodes = models.get_active_nodes()
    edges = models.get_all_edges()
    view = get_view()

    if view["mode"] == "node":
        trimmed = _active_path(view["path"])
        if not trimmed:
            view = set_view(dict(DEFAULT_VIEW))
        elif trimmed != view["path"]:
            view = set_view({"mode": "node", "path": trimmed})

    if view["mode"] == "projects":
        cards = scoring.compute_projects_overview(nodes, edges)
    elif view["mode"] == "node":
        cards = scoring.compute_container_detail(nodes, edges, view["path"][-1])
        cards = scoring.annotate_projects(cards, nodes, edges)
    elif view["mode"] == "loose":
        cards = scoring.compute_loose_tasks(nodes, edges)
    elif view["mode"] == "filter":
        name = view["filter"]
        if name == "waiting":
            candidates = models.get_nodes_by_status("on_hold")
        elif name == "done":
            candidates = models.get_recently_completed()
        else:  # overdue / high — drawn from the active set
            candidates = nodes
        cards = scoring.compute_filter(name, candidates, edges)
        cards = scoring.annotate_projects(cards, nodes, edges)
    elif view["mode"] == "list":
        cards = scoring.compute_list(nodes, view["node_ids"])
        cards = scoring.annotate_projects(cards, nodes, edges)
        if not cards:
            view = set_view(dict(DEFAULT_VIEW))
            cards = scoring.annotate_projects(scoring.compute_today(nodes, edges), nodes, edges)
    else:  # today
        cards = scoring.annotate_projects(scoring.compute_today(nodes, edges), nodes, edges)

    scoring.annotate_child_counts(cards, nodes, edges)
    return {"view": view_meta(view, cards), "cards": cards}


def view_member_ids(view: Optional[Dict[str, Any]] = None) -> List[int]:
    """Node IDs the current view is 'about' — used to prioritize prompt context
    (the tasks the user is looking at are what 'this'/'here' refer to)."""
    view = view or get_view()
    if view["mode"] == "node":
        current = view["path"][-1]
        return [current] + models.get_active_child_ids(current)
    if view["mode"] == "list":
        return list(view["node_ids"])
    return []
