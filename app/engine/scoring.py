"""Per-view card computation: decides which nodes each Lens view shows and why.

The LLM never picks scores or card counts. It only signals navigation intent
(open_view); everything here is deterministic Python so each view behaves
consistently:
- Today: items qualify on merit (deadline window, high priority, fresh capture,
  blocker of a qualifier), capped at MAX_LENS_CARDS — never padded.
- Project detail / lists: ALL the members, sorted by urgency and importance.
- Projects overview: every active project, ranked by its nearest deadline.
"""
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional

MAX_LENS_CARDS = 7            # Today view only; entered views show everything

DEADLINE_WINDOW_DAYS = 7      # deadlines further out than this don't qualify on their own
RECENT_CAPTURE_HOURS = 2.0    # just-captured items stay visible briefly
RECENT_CAPTURE_SCORE = 4.0

# Importance is independent of urgency: a high-priority task qualifies for the
# Today view even without a deadline, and outranks a same-deadline normal
# task; low-priority tasks need an imminent deadline to take a Today slot.
PRIORITY_BONUS = {"high": 3.0, "normal": 0.0, "low": 0.0}
PRIORITY_MULTIPLIER = {"high": 1.5, "normal": 1.0, "low": 0.5}
IMMINENT_DEADLINE_SCORE = 7.0  # deadline part of the scale at ~2 days out

# Card categories encode URGENCY (the wash color answers "when is this due?").
# Priority is the edge stripe — independent channels.
CATEGORY_NEXT = "next"          # peach — overdue or due within ~2 days
CATEGORY_SOON = "soon"          # butter — due within the week
CATEGORY_HORIZON = "horizon"    # sage — dated, but further out
CATEGORY_UNDATED = "undated"    # neutral — no deadline at all


def _parse_db_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parses SQLite 'YYYY-MM-DD HH:MM:SS' (UTC) or ISO strings into aware UTC datetimes."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_deadline(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _deadline_score(node: Dict[str, Any], today: date) -> float:
    """0 outside the window; 3..9 inside, rising as the deadline approaches."""
    deadline = _parse_deadline(node.get("target_date"))
    if deadline is None:
        return 0.0
    days_left = (deadline - today).days
    if days_left < 0:
        return 9.0  # overdue
    if days_left > DEADLINE_WINDOW_DAYS:
        return 0.0
    return 9.0 - (days_left / DEADLINE_WINDOW_DAYS) * 6.0


def _recency_score(node: Dict[str, Any], now: datetime) -> float:
    created = _parse_db_timestamp(node.get("created_at"))
    if created is None:
        return 0.0
    hours = (now - created).total_seconds() / 3600.0
    if hours < 0 or hours > RECENT_CAPTURE_HOURS:
        return 0.0
    return RECENT_CAPTURE_SCORE * (1.0 - hours / RECENT_CAPTURE_HOURS)


def _urgency_rank(node: Dict[str, Any], today: date) -> float:
    """Sort key for entered views (project detail / lists): deadline proximity
    weighted by importance. Far-out deadlines still order by date here — inside
    a project nothing is hidden, only ranked."""
    deadline = _parse_deadline(node.get("target_date"))
    if deadline is None:
        base = 0.0
    else:
        days_left = max((deadline - today).days, 0)
        base = max(9.0 - (days_left / DEADLINE_WINDOW_DAYS) * 6.0, 1.0)
    priority = node.get("priority") or "normal"
    return (base + PRIORITY_BONUS.get(priority, 0.0)) * PRIORITY_MULTIPLIER.get(priority, 1.0)


def _categorize(entry: Dict[str, Any], today: date) -> str:
    """Urgency only: the wash answers 'when?'. Priority has its own stripe."""
    if entry.get("_blocker"):
        return CATEGORY_NEXT
    deadline = _parse_deadline(entry.get("target_date"))
    if deadline is None:
        return CATEGORY_UNDATED
    days_left = (deadline - today).days
    if days_left <= 2:
        return CATEGORY_NEXT
    if days_left <= DEADLINE_WINDOW_DAYS:
        return CATEGORY_SOON
    return CATEGORY_HORIZON


def _finalize(entries: List[Dict[str, Any]], today: date) -> List[Dict[str, Any]]:
    for entry in entries:
        entry["category"] = _categorize(entry, today)
        for key in ("_deadline", "_qualifies", "_blocker"):
            entry.pop(key, None)
    return entries


# ---------------------------------------------------------------------------
# Today view — merit-based qualification, capped
# ---------------------------------------------------------------------------

def compute_today(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                  now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Returns up to MAX_LENS_CARDS qualifying nodes, each annotated with
    'lens_score' and 'category'. Non-qualifying nodes are excluded entirely."""
    now = now or datetime.now(timezone.utc)
    today = datetime.now().date()

    scored: Dict[int, Dict[str, Any]] = {}
    for node in nodes:
        deadline = _deadline_score(node, today)
        recency = _recency_score(node, now)
        priority = node.get("priority") or "normal"

        if priority == "low":
            # Movable work doesn't take a Today slot unless it's truly imminent
            qualifies = deadline >= IMMINENT_DEADLINE_SCORE
        else:
            qualifies = (
                deadline > 0
                or recency > 0
                or priority == "high"  # important work is the default view
            )
        entry = dict(node)
        base = deadline + recency + PRIORITY_BONUS.get(priority, 0.0)
        entry["lens_score"] = base * PRIORITY_MULTIPLIER.get(priority, 1.0)
        entry["_deadline"] = deadline
        entry["_qualifies"] = qualifies
        scored[node["id"]] = entry

    # Blocker promotion: a node that blocks a qualifying node qualifies too,
    # ranked just above the thing it blocks. Two passes cover short chains.
    for _ in range(2):
        for edge in edges:
            if edge.get("relationship") != "blocks":
                continue
            blocker = scored.get(edge["parent_id"])
            blocked = scored.get(edge["child_id"])
            if blocker and blocked and blocked["_qualifies"]:
                if not blocker["_qualifies"] or blocker["lens_score"] <= blocked["lens_score"]:
                    blocker["_qualifies"] = True
                    blocker["lens_score"] = blocked["lens_score"] + 0.5
                    blocker["_blocker"] = True

    qualifying = [entry for entry in scored.values() if entry["_qualifies"]]
    qualifying.sort(key=lambda entry: entry["lens_score"], reverse=True)
    return _finalize(qualifying[:MAX_LENS_CARDS], today)


# ---------------------------------------------------------------------------
# Entered views — everything, ranked
# ---------------------------------------------------------------------------

def compute_project_detail(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                           project_id: int) -> List[Dict[str, Any]]:
    """ALL active tasks of one project, most urgent/important first. No cap —
    entering a project means seeing it whole, like opening it in a PM app."""
    today = datetime.now().date()
    child_ids = {e["child_id"] for e in edges
                 if e["relationship"] == "is_part_of" and e["parent_id"] == project_id}
    members = [dict(n) for n in nodes if n["id"] in child_ids]
    members.sort(key=lambda n: _urgency_rank(n, today), reverse=True)
    return _finalize(members, today)


def compute_list(nodes: List[Dict[str, Any]], node_ids: List[int]) -> List[Dict[str, Any]]:
    """The still-active subset of an ad-hoc selection, most urgent first."""
    today = datetime.now().date()
    wanted = set(node_ids)
    members = [dict(n) for n in nodes if n["id"] in wanted]
    members.sort(key=lambda n: _urgency_rank(n, today), reverse=True)
    return _finalize(members, today)


def compute_loose_tasks(nodes: List[Dict[str, Any]],
                        edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Active tasks that belong to no project (and aren't projects themselves)."""
    today = datetime.now().date()
    parented = {e["child_id"] for e in edges if e["relationship"] == "is_part_of"}
    members = [dict(n) for n in nodes
               if n["id"] not in parented and n.get("node_type") != "project"]
    members.sort(key=lambda n: _urgency_rank(n, today), reverse=True)
    return _finalize(members, today)


def compute_projects_overview(nodes: List[Dict[str, Any]],
                              edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every active project as a card: open-task count and the nearest deadline
    among the project and its open tasks. Sorted by urgency. Tasks that belong
    to no project surface as one 'Loose tasks' pseudo-card at the end."""
    today = datetime.now().date()
    by_id = {n["id"]: n for n in nodes}

    children: Dict[int, List[Dict[str, Any]]] = {}
    for edge in edges:
        if edge["relationship"] != "is_part_of":
            continue
        parent, child = by_id.get(edge["parent_id"]), by_id.get(edge["child_id"])
        if parent and child and parent.get("node_type") == "project":
            children.setdefault(parent["id"], []).append(child)

    cards = []
    for node in nodes:
        if node.get("node_type") != "project":
            continue
        entry = dict(node)
        kids = children.get(node["id"], [])
        entry["project_open_total"] = len(kids)
        deadlines = [d for d in
                     (_parse_deadline(n.get("target_date")) for n in kids + [node])
                     if d is not None]
        entry["target_date"] = min(deadlines).isoformat() if deadlines else None
        cards.append(entry)
    cards.sort(key=lambda n: _urgency_rank(n, today), reverse=True)
    cards = _finalize(cards, today)

    loose = compute_loose_tasks(nodes, edges)
    if loose:
        dated = [_parse_deadline(n.get("target_date")) for n in loose]
        dated = [d for d in dated if d is not None]
        pseudo = {
            "id": None, "pseudo": "loose", "node_type": "project",
            "content": "Loose tasks", "priority": "normal", "status": "active",
            "project_open_total": len(loose),
            "target_date": min(dated).isoformat() if dated else None,
        }
        pseudo["category"] = _categorize(pseudo, today)
        cards.append(pseudo)
    return cards


# ---------------------------------------------------------------------------
# Annotation shared by views: which project does each card belong to?
# ---------------------------------------------------------------------------

def annotate_projects(cards: List[Dict[str, Any]], nodes: List[Dict[str, Any]],
                      edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adds project_id / project_name / project_open_total to task cards, and
    drops project cards whose own tasks are already visible (the group header
    represents them — a project card between its own tasks is noise)."""
    card_ids = {c["id"] for c in cards}
    children_present = {
        e["parent_id"] for e in edges
        if e["relationship"] == "is_part_of" and e["child_id"] in card_ids
    }
    cards = [c for c in cards
             if not (c.get("node_type") == "project" and c["id"] in children_present)]

    by_id = {n["id"]: n for n in nodes}
    project_of: Dict[int, Dict[str, Any]] = {}
    open_count: Dict[int, int] = {}
    for edge in edges:  # edges arrive in insertion order (models.get_all_edges)
        if edge["relationship"] != "is_part_of":
            continue
        parent = by_id.get(edge["parent_id"])
        child = by_id.get(edge["child_id"])
        if parent and child and parent.get("node_type") == "project":
            # A task may belong to several projects (multi-home); the FIRST one
            # it was filed under is its primary home for grouping/the chip.
            if child["id"] not in project_of:
                project_of[child["id"]] = parent
            open_count[parent["id"]] = open_count.get(parent["id"], 0) + 1

    for card in cards:
        if card.get("node_type") == "project":
            project = by_id.get(card["id"])
        else:
            project = project_of.get(card["id"])
        if project:
            card["project_id"] = project["id"]
            card["project_name"] = project["content"]
            card["project_open_total"] = open_count.get(project["id"], 0)
    return cards
