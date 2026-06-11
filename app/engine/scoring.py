"""Lens selection: decides which active nodes qualify for the focus stack and why.

The LLM never picks scores or card counts. It only signals intent (focus_lens /
clear_focus); everything here is deterministic Python so the Lens behaves
consistently: items qualify on merit, fade as focus decays, and the pane shows
*up to* MAX_LENS_CARDS — never padded with unrelated items.
"""
import math
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional

from app.database import models

MAX_LENS_CARDS = 7

# Focus halves every 8 hours, so an explicit "focus on X" (score 10.0) keeps an
# item qualified for roughly a working day, then it fades out on its own.
FOCUS_HALF_LIFE_HOURS = 8.0
FOCUS_QUALIFY_THRESHOLD = 2.0

DEADLINE_WINDOW_DAYS = 7      # deadlines further out than this don't qualify on their own
RECENT_CAPTURE_HOURS = 2.0    # just-captured items stay visible briefly
RECENT_CAPTURE_SCORE = 4.0

# Importance is independent of urgency: a high-priority task qualifies for the
# default view even without a deadline, and outranks a same-deadline normal
# task; low-priority tasks need an imminent deadline (or focus) to take a slot.
PRIORITY_BONUS = {"high": 3.0, "normal": 0.0, "low": 0.0}
PRIORITY_MULTIPLIER = {"high": 1.5, "normal": 1.0, "low": 0.5}
IMMINENT_DEADLINE_SCORE = 7.0  # deadline part of the scale at ~2 days out

# Categories map 1:1 to the spec's pastel tokens (computed here, not by card index)
CATEGORY_FOCUS = "focus"        # peach — explicitly hoisted in conversation
CATEGORY_NEXT = "next"          # butter — overdue/imminent deadline or blocker of a focused item
CATEGORY_HORIZON = "horizon"    # sage — deadline inside the window
CATEGORY_ADMIN = "admin"        # lavender — qualified for secondary reasons


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


def _effective_focus(node: Dict[str, Any], now: datetime) -> float:
    score = node.get("focus_score") or 0.0
    if score <= 0:
        return 0.0
    focused_at = _parse_db_timestamp(node.get("focused_at"))
    if focused_at is None:
        return score
    hours = max((now - focused_at).total_seconds() / 3600.0, 0.0)
    return score * math.pow(2.0, -hours / FOCUS_HALF_LIFE_HOURS)


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


def compute_lens(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                 now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Returns up to MAX_LENS_CARDS qualifying nodes, each annotated with
    'lens_score' and 'category'. Non-qualifying nodes are excluded entirely."""
    now = now or datetime.now(timezone.utc)
    today = datetime.now().date()

    scored: Dict[int, Dict[str, Any]] = {}
    for node in nodes:
        focus = _effective_focus(node, now)
        deadline = _deadline_score(node, today)
        recency = _recency_score(node, now)
        priority = node.get("priority") or "normal"

        if priority == "low":
            # Movable work doesn't take a Lens slot unless it's truly imminent
            qualifies = (
                focus >= FOCUS_QUALIFY_THRESHOLD
                or deadline >= IMMINENT_DEADLINE_SCORE
            )
        else:
            qualifies = (
                focus >= FOCUS_QUALIFY_THRESHOLD
                or deadline > 0
                or recency > 0
                or priority == "high"  # important work is the default view
            )
        entry = dict(node)
        base = focus + deadline + recency + PRIORITY_BONUS.get(priority, 0.0)
        entry["lens_score"] = base * PRIORITY_MULTIPLIER.get(priority, 1.0)
        entry["_focus"] = focus
        entry["_deadline"] = deadline
        entry["_priority"] = priority
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
    lens = qualifying[:MAX_LENS_CARDS]

    for entry in lens:
        entry["category"] = _categorize(entry)
        for key in ("_focus", "_deadline", "_priority", "_qualifies", "_blocker"):
            entry.pop(key, None)
    return lens


def _categorize(entry: Dict[str, Any]) -> str:
    if entry["_focus"] >= FOCUS_QUALIFY_THRESHOLD:
        return CATEGORY_FOCUS
    if entry.get("_blocker") or entry["_deadline"] >= IMMINENT_DEADLINE_SCORE:
        return CATEGORY_NEXT
    if entry["_deadline"] > 0:
        return CATEGORY_HORIZON
    if entry.get("_priority") == "high":
        return CATEGORY_HORIZON  # important-but-undated reads as "coming up", not admin
    return CATEGORY_ADMIN


# Hours until a fresh focus (score 10) decays below the qualify threshold
FOCUS_WINDOW_HOURS = FOCUS_HALF_LIFE_HOURS * math.log2(10.0 / FOCUS_QUALIFY_THRESHOLD)


def get_lens_state() -> Dict[str, Any]:
    """Reads the graph and returns {'cards': [...], 'focus': {...}|None}.

    Cards are annotated with their project (name + open-task count) so the UI
    can show structure; 'focus' summarizes the active focus set for the header."""
    nodes = models.get_active_nodes()
    edges = models.get_all_edges()
    cards = compute_lens(nodes, edges)

    by_id = {n["id"]: n for n in nodes}
    project_of: Dict[int, Dict[str, Any]] = {}
    open_count: Dict[int, int] = {}
    for edge in edges:
        if edge["relationship"] != "is_part_of":
            continue
        parent = by_id.get(edge["parent_id"])
        child = by_id.get(edge["child_id"])
        if parent and child and parent.get("node_type") == "project":
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

    focused = [n for n in nodes if (n.get("focus_score") or 0) > 0]
    focus = None
    if focused:
        # Label by what's visible: the project(s) of the focused cards that
        # actually made the Lens, not of every node the model boosted.
        visible_focused = [c for c in cards if c.get("category") == CATEGORY_FOCUS]
        names = {c["project_name"] for c in visible_focused if c.get("project_name")}
        label = next(iter(names)) if len(names) == 1 else f"{len(focused)} tasks"

        stamps = [_parse_db_timestamp(n.get("focused_at")) for n in focused]
        stamps = [s for s in stamps if s]
        hours_left = None
        if stamps:
            elapsed = (datetime.now(timezone.utc) - max(stamps)).total_seconds() / 3600.0
            hours_left = max(0, round(FOCUS_WINDOW_HOURS - elapsed))
        focus = {"label": label, "count": len(focused), "hours_left": hours_left}

    return {"cards": cards, "focus": focus}
