"""Lens-Brain: converts natural-language chat into validated tool calls.

Design rules:
- The LLM expresses *intent* only; it never picks scores, card counts, or SQL.
- Every tool argument set is validated with Pydantic before touching the DB,
  and node IDs are checked for existence (small models hallucinate IDs).
- A rolling conversation history is kept so references like "that one" and
  corrections like "no, I meant Friday" resolve correctly.
"""
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import calendar as _calendar

import dateparser
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import config
from app.database import models
from app.engine import embeddings

client = AsyncOpenAI(
    base_url=config.AI_BASE_URL,
    api_key="not-needed-for-local"
)

MAX_TOOL_ROUNDS = 5

# Session memory sizing. History is trimmed in BLOCKS (when it exceeds
# HISTORY_MAX, cut back to HISTORY_TURNS) rather than one-out-one-in: the
# llama.cpp engine caches the longest unchanged prompt PREFIX, and a sliding
# window shifts every message every turn, invalidating the whole cache.
HISTORY_TURNS = 8
HISTORY_MAX = 16

# Small models occasionally write the tool call as prose ("open_view [{...}]")
# instead of emitting a real one. LM Studio ignores tool_choice="required" and
# corrective retries come back empty, so we salvage by parsing the leaked text.
TOOL_NAME_LEAK = re.compile(
    r"\b(capture_tasks|complete_tasks|update_task|link_tasks|move_task|open_view|search_tasks)\b")


class _SalvagedFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _SalvagedCall:
    def __init__(self, name: str, args: dict, index: int = 0):
        self.id = f"salvaged_{index}"
        self.type = "function"
        self.function = _SalvagedFunction(name, json.dumps(args))


def _coerce_ids(items) -> List[int]:
    ids = []
    for item in items:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, dict):
            value = item.get("node_id") or item.get("id")
            if isinstance(value, int):
                ids.append(value)
    return ids


def salvage_tool_call(text: str) -> Optional[_SalvagedCall]:
    """Parses a tool call the model wrote as prose into an executable call.
    Normalizes the common argument mistakes (bare lists, node_id vs node_ids)."""
    match = TOOL_NAME_LEAK.search(text or "")
    if not match:
        return None
    name = match.group(1)
    rest = text[match.end():]

    parsed = None
    for i, ch in enumerate(rest):
        if ch in "[{":
            try:
                parsed, _ = json.JSONDecoder().raw_decode(rest[i:])
            except ValueError:
                parsed = None
            break

    if name == "open_view":
        if isinstance(parsed, dict) and parsed.get("view"):
            return _SalvagedCall(name, parsed)
        # Non-JSON leak ('open_view[view="project", project_name="The Cage"]'):
        # pull the fields out of the prose.
        view_match = re.search(r"view\s*[=:]\s*['\"]?(today|projects|project|list|loose)", rest[:200])
        if not view_match:
            return None
        args: dict = {"view": view_match.group(1)}
        name_match = re.search(r"project_name\s*[=:]\s*['\"]([^'\"]+)['\"]", rest[:300])
        if name_match:
            args["project_name"] = name_match.group(1)
        if args["view"] == "list":
            ids = [int(n) for n in re.findall(r"\d+", rest[:300])]
            if not ids:
                return None
            args["node_ids"] = ids
        return _SalvagedCall(name, args)
    if name == "complete_tasks":
        if isinstance(parsed, list):
            ids = _coerce_ids(parsed)
            return _SalvagedCall(name, {"node_ids": ids}) if ids else None
        if isinstance(parsed, dict):
            if "node_ids" in parsed:
                return _SalvagedCall(name, parsed)
            ids = _coerce_ids([parsed])
            return _SalvagedCall(name, {"node_ids": ids}) if ids else None
        # Non-JSON leak ("complete_tasks[node_ids=[6,5]]", "complete_tasks 3 and 5"):
        # extract the integers — bogus IDs get filtered against the DB downstream.
        ids = [int(n) for n in re.findall(r"\d+", rest[:200])]
        return _SalvagedCall(name, {"node_ids": ids}) if ids else None
    if name == "capture_tasks":
        if isinstance(parsed, list):
            return _SalvagedCall(name, {"tasks": parsed})
        if isinstance(parsed, dict):
            return _SalvagedCall(name, parsed if "tasks" in parsed else {"tasks": [parsed]})
        return None
    if isinstance(parsed, dict):  # update_task / link_tasks
        return _SalvagedCall(name, parsed)
    return None

# Rolling session memory (in-process; this is a single-user local app)
conversation_history: List[dict] = []


def _append_history(message: dict):
    conversation_history.append(message)
    if len(conversation_history) > HISTORY_MAX:
        del conversation_history[:len(conversation_history) - HISTORY_TURNS]

# Grounding: every node ID the model has actually been shown this session
# (context injections, search results, its own captures). Tool calls that
# reference IDs outside this set are rejected — the model must search, not guess.
seen_node_ids: set = set()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are Lens-Brain, the action engine of Project Lens, a personal task manager. Your job on every user message: decide which rule below applies, then CALL THE MATCHING TOOL. Tool calls are the only way anything happens — text output never creates, completes, or shows a task. Do not describe an action in text; perform it with a tool call.

The user sees this chat plus "The Lens", a navigable panel of task cards — like the main pane of a project manager app. open_view steers it between views: today (most urgent), projects (overview), one project's tasks, or an ad-hoc list. Tasks are shown in the Lens, never listed in chat.

The newest user message arrives with a system-attached CURRENT CONTEXT block (dates, ACTIVE TASKS, the open view); the user's actual words follow "USER MESSAGE:". The context is reference material — never treat it as something the user said.

RULES
0. Act ONLY on the user's LATEST message. Earlier conversation is context for resolving references — everything in it has already been handled; never re-capture or re-complete from it.
0b. TRIAGE FIRST: is the latest message NEW work, or about a task in ACTIVE TASKS? New work → capture_tasks immediately (no search_tasks first, no update_task) — even when the message says "critical"/"really important" (that goes in the new task's priority field). Only use update_task/complete_tasks when the task's subject matter actually appears in ACTIVE TASKS. Example: "Really important: I have to send the grant application" with no grant task listed → capture_tasks {"tasks": [{"content": "Send the grant application", "priority": "high"}]} — NOT update_task on some other task.
1. The user mentions new work, a goal, or a commitment → call capture_tasks with EVERY distinct task in the message. A project with steps → one item with node_type "project" and the steps in subtasks. Set parent_id when it clearly belongs to an existing item — parent_id may be ANY existing task, not just a project, so "add X and Y under <task>" or "as subtasks of <task>" files them as that task's subtasks. If they signal importance about the NEW work ("critical", "really important", "must do") set priority "high" on it ("no rush"/"whenever" → "low") — still capture_tasks, never update_task. Never capture anything already in ACTIVE TASKS.
2. Deadlines: pass the deadline exactly as the user said it ("Friday", "end of month", "June 3") or as YYYY-MM-DD — the system converts relative dates itself. No date stated or implied → omit the field, never invent one.
3. The user finished something → call complete_tasks with ALL matching IDs from ACTIVE TASKS, in one call. Match loosely: "sent the invoices" matches "Send invoices to clients". Never create a task for finished work; if nothing matches, ask one short question.
4. Reword, change deadline, change importance, pause (on_hold), resume (active), or archive (cold_storage) → call update_task with only the fields that change. update_task is for tasks that ALREADY EXIST in ACTIVE TASKS. Importance words about an existing task ("X is critical" → priority high; "X is low priority, no rush" → low). Set priority ONLY when the user signals it — never infer it. PROMOTE an existing task to a project ("make X its own project", "I want X to be its own project with A, B, C", "turn X into a project") → in the SAME turn: (1) call update_task on X with node_type "project"; (2) if they name new tasks A, B, C, call capture_tasks for them with parent_id = X's id. If X isn't in ACTIVE TASKS, search_tasks for it first.
5. A dependency or grouping is stated → call link_tasks. The BLOCKER is always parent_id: "X blocks Y" → X is parent_id; "Y is blocked by X" → X is still parent_id. Use is_part_of for project/subtask grouping. RELOCATE an existing task ("move X to Y", "file X under Y instead", "put X in Y") → call move_task — it pulls X out of its current project and refiles it. Pass to_project_name for a project destination, or new_parent_id to file X under another task. Reserve link_tasks (is_part_of) for "X is ALSO part of Y" (keep both homes). If X (or an ID destination) isn't in ACTIVE TASKS, search_tasks first.
6. ANY request to see tasks is NAVIGATION — call open_view, never answer with a text list:
   - "focus on X" / "let's work on X" / "open X" / "what's left on X" / "where am I on X" → open_view {"view": "project", "project_name": "X"}. Pass the project's name as the user said it — the system finds the project and shows ALL its open tasks itself. Do NOT pass node_ids and do NOT search first.
   - "what are my projects" / "show my projects" / "what's on my plate" → open_view {"view": "projects"}.
   - "what should I work on" / "show today" / "what's urgent" / "what's most urgent" / "what's due this week" / "what's coming up" / "what are the urgent ones" / "go back" / "clear the lens" → open_view {"view": "today"} (Today IS the urgent/this-week view — never answer these in chat).
   - Category words ("errands", "chores", "admin"): judge which ACTIVE TASKS fit (search_tasks first if needed) → open_view {"view": "list", "node_ids": [...], "label": "errands"}.
   EXCEPTION — a question about ONE specific named task ("when is the dentist due?", "what's the status of the op-ed?", "is the report done yet?") is NOT navigation: answer it in one plain sentence from ACTIVE TASKS, do NOT call open_view. This exception is ONLY for a single named task. A question about a SET of tasks ("what's urgent", "what's due soon", "which ones are high priority", "what are the most urgent tasks this week") is navigation — open_view, never a chat list.
   NEVER list tasks in chat, and never say the Lens shows something unless you called open_view this turn.
7. "that", "it", "the second one" → resolve from the recent conversation and ACTIVE TASKS; if genuinely ambiguous, ask one short question — never guess an ID. A correction ("no, I meant Friday") → call update_task on the existing task, never capture a new one.
8. ACTIVE TASKS is only the slice of the database relevant to this conversation. If the user refers to an EXISTING task or project that is NOT listed, call search_tasks with 2-3 keywords first, then act on the results. Never claim a task doesn't exist without searching. Do NOT search before capturing new work — rule 1 applies directly. Search results marked "on_hold" are shelved projects/tasks; when the user wants to resume or activate one, call update_task with status "active".

Reply in plain text ONLY when no rule applies: a single-fact question (one deadline, one status), a greeting, venting, or reflection — answer briefly, capture nothing. For mixed messages, call tools for the actionable part only."""

# Volatile context is attached to the FRONT OF THE NEWEST USER MESSAGE, not
# sent as a system message. The chat template hoists all system messages to
# the top of the rendered prompt (verified empirically: reordering system
# messages produced byte-identical KV cache hits), so volatile content in ANY
# system message lands before the history and invalidates the llama.cpp
# prefix cache every turn (~8s of prompt processing). Inside the latest user
# message it renders at the END of the prompt: the static rules, tool schemas,
# and append-only history stay a byte-identical cached prefix, and only the
# newest exchange reprocesses (~0.5-3s). NOW is hour-granular for the same
# reason. History stores the user's bare text, so each turn's context blob
# costs cache reuse for exactly one turn, not the whole session.
CONTEXT_TEMPLATE = """CURRENT CONTEXT
NOW: {now}
CALENDAR (use this table for any relative date — do not compute weekdays yourself):
{calendar}
ACTIVE TASKS (use these exact IDs in tool calls):
{task_list}
LENS CURRENTLY SHOWS: {view_desc}"""

POST_TOOL_REMINDER = (
    "Tools executed. Now reply to the user in at most two short sentences confirming what "
    "changed, bolding task names with **double asterisks**. NEVER list tasks line by line — "
    "name at most two, otherwise summarize the count (\"Added 3 tasks — they're in your "
    "Lens.\"). The Lens pane shows the tasks; chat does not. Never mention IDs, tools, or "
    "technical details."
)


def _calendar_table(now: datetime, days: int = 8) -> str:
    """The next two weeks as weekday→date lines; small models fail weekday math."""
    lines = []
    for offset in range(days):
        day = now + timedelta(days=offset)
        label = {0: " (today)", 1: " (tomorrow)"}.get(offset, "")
        lines.append(f"{day.strftime('%A')} = {day.strftime('%Y-%m-%d')}{label}")
    return "\n".join(lines)


# Below this many active tasks, just show them all; above it, retrieve a
# relevant subset so the prompt stays small no matter how large the graph grows.
# Every context line costs prompt-processing latency (~0.1s per 20 tokens on
# local hardware), so the cap leans tight.
CONTEXT_FULL_THRESHOLD = 20
CONTEXT_MAX_TASKS = 20


def select_context_tasks(active_nodes: List[dict], conversation_text: str,
                         now: Optional[datetime] = None,
                         view_ids: Optional[set] = None) -> List[dict]:
    """Tasks live in the database; the prompt gets only what this turn could
    plausibly touch: what's on screen (the open view), near deadlines, fresh
    captures, and full-text matches against the recent conversation."""
    if len(active_nodes) <= CONTEXT_FULL_THRESHOLD:
        return active_nodes

    from app.engine import scoring
    now = now or datetime.now(timezone.utc)
    today = datetime.now().date()
    view_ids = view_ids or set()

    # Priority buckets: when the cap bites, what the user is LOOKING AT and
    # conversation matches survive; "recently captured" noise is first to drop.
    from app.engine import retrieval
    on_screen, matched, due_soon, recent = [], [], [], []
    matched_ids = {n["id"] for n in retrieval.search_active(conversation_text, limit=15)}

    for node in active_nodes:
        deadline = scoring._parse_deadline(node.get("target_date"))
        if node["id"] in view_ids:
            on_screen.append(node)
        elif node["id"] in matched_ids:
            matched.append(node)
        elif deadline is not None and (deadline - today).days <= scoring.DEADLINE_WINDOW_DAYS:
            due_soon.append(node)
        elif scoring._recency_score(node, now) > 0:
            recent.append(node)

    # Recent captures are a courtesy (FTS on the conversation usually re-finds
    # them anyway); cap them so a bulk seed doesn't flood the prompt.
    return (on_screen + matched + due_soon + recent[-10:])[:CONTEXT_MAX_TASKS]


def format_context_prompt(user_text: str = "") -> str:
    from app.engine import views

    now = datetime.now()
    active_nodes = models.get_active_nodes()
    view = views.get_view()
    view_ids = set(views.view_member_ids(view))

    # Search against the latest message plus recent user turns, so references
    # like "that one" still surface the task discussed a moment ago.
    recent_user_text = " ".join(
        msg["content"] for msg in list(conversation_history)[-6:]
        if isinstance(msg, dict) and msg.get("role") == "user"
    )
    context_nodes = select_context_tasks(
        active_nodes, f"{user_text} {recent_user_text}", view_ids=view_ids)

    seen_node_ids.update(n["id"] for n in context_nodes)
    seen_node_ids.update(view_ids)

    if context_nodes:
        lines = []
        for node in context_nodes:
            parts = [f"[ID {node['id']}]", node["content"]]
            if node.get("node_type") == "project":
                parts.append("(project)")
            if node.get("target_date"):
                parts.append(f"(due {node['target_date']})")
            if node.get("priority") in ("high", "low"):
                parts.append(f"({node['priority']} priority)")
            lines.append(" ".join(parts))
        task_list = "\n".join(lines)
        if len(context_nodes) < len(active_nodes):
            task_list += (f"\n(+{len(active_nodes) - len(context_nodes)} more active tasks "
                          "in the database, omitted as unrelated to this conversation)")
    elif active_nodes:
        task_list = f"None relevant to this conversation ({len(active_nodes)} active tasks in the database)."
    else:
        task_list = "No active tasks."

    mode = view["mode"]
    if mode == "node":
        current = models.get_node(view["path"][-1])
        current_name = current["content"] if current else "a project"
        view_desc = (f"'{current_name}' (ID {view['path'][-1]}) and its open items — "
                     "'here'/'this' in the user's message means this one")
    elif mode == "projects":
        view_desc = "the all-projects overview"
    elif mode == "list":
        view_desc = f"a list: {view.get('label') or 'selected tasks'}"
    elif mode == "loose":
        view_desc = "tasks that belong to no project"
    else:
        view_desc = "Today (the most urgent tasks)"

    return CONTEXT_TEMPLATE.format(
        now=f"{now.strftime('%A, %Y-%m-%d')}, around {now.strftime('%-I %p')}",
        calendar=_calendar_table(now),
        task_list=task_list,
        view_desc=view_desc,
    )


# ---------------------------------------------------------------------------
# Tools: schemas the model sees + Pydantic validation before execution
# ---------------------------------------------------------------------------

def normalize_deadline(value: Optional[str], base: Optional[datetime] = None) -> Optional[str]:
    """Converts a deadline — ISO or the user's own words ("Friday", "end of month") —
    to YYYY-MM-DD. Date math lives here, deterministically; the model just relays
    the phrase. Raises ValueError when the phrase can't be parsed."""
    if value is None:
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        pass

    base = base or datetime.now()
    phrase = text.lower()
    for prefix in ("by ", "due ", "on ", "before ", "sometime "):
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):]

    if re.search(r"end of (the )?month", phrase) or phrase in ("month", "this month"):
        last_day = _calendar.monthrange(base.year, base.month)[1]
        return base.replace(day=last_day).date().isoformat()
    if re.search(r"end of (the )?week", phrase):
        return (base + timedelta(days=(6 - base.weekday()))).date().isoformat()

    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base}
    # dateparser handles "next month"/"in two weeks" natively but not
    # "next monday"; retry with the qualifier stripped before giving up.
    parsed = dateparser.parse(phrase, settings=settings)
    if parsed is None:
        parsed = dateparser.parse(
            phrase.replace("next ", "").replace("this ", ""), settings=settings)
    if parsed is None:
        raise ValueError(f"Could not understand deadline {value!r}. Pass YYYY-MM-DD.")
    return parsed.date().isoformat()


class TaskItem(BaseModel):
    content: str = Field(min_length=1)
    deadline: Optional[str] = None
    parent_id: Optional[int] = None
    relationship: str = "is_part_of"
    node_type: str = "task"
    priority: str = "normal"
    subtasks: List[str] = Field(default_factory=list)

    @field_validator("priority")
    @classmethod
    def check_priority(cls, value):
        if value not in models.VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {models.VALID_PRIORITIES}")
        return value

    @field_validator("deadline")
    @classmethod
    def check_deadline(cls, value):
        return normalize_deadline(value)

    @field_validator("relationship")
    @classmethod
    def check_relationship(cls, value):
        if value not in models.VALID_RELATIONSHIPS:
            raise ValueError(f"relationship must be one of {models.VALID_RELATIONSHIPS}")
        return value

    @field_validator("node_type")
    @classmethod
    def check_node_type(cls, value):
        if value not in models.VALID_NODE_TYPES:
            raise ValueError(f"node_type must be one of {models.VALID_NODE_TYPES}")
        return value


class CaptureTasksArgs(BaseModel):
    tasks: List[TaskItem] = Field(min_length=1)


class CompleteTasksArgs(BaseModel):
    node_ids: List[int] = Field(min_length=1)


class UpdateTaskArgs(BaseModel):
    node_id: int
    content: Optional[str] = None
    deadline: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    description: Optional[str] = None
    node_type: Optional[str] = None

    @field_validator("node_type")
    @classmethod
    def check_node_type(cls, value):
        if value is not None and value not in models.VALID_NODE_TYPES:
            raise ValueError(f"node_type must be one of {models.VALID_NODE_TYPES}")
        return value

    @field_validator("priority")
    @classmethod
    def check_priority(cls, value):
        if value is not None and value not in models.VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {models.VALID_PRIORITIES}")
        return value

    @field_validator("deadline")
    @classmethod
    def check_deadline(cls, value):
        return normalize_deadline(value)

    @field_validator("status")
    @classmethod
    def check_status(cls, value):
        if value is not None and value not in models.VALID_STATUSES:
            raise ValueError(f"status must be one of {models.VALID_STATUSES}")
        return value


class LinkTasksArgs(BaseModel):
    parent_id: int
    child_id: int
    relationship: str

    @field_validator("relationship")
    @classmethod
    def check_relationship(cls, value):
        if value not in models.VALID_RELATIONSHIPS:
            raise ValueError(f"relationship must be one of {models.VALID_RELATIONSHIPS}")
        return value


class MoveTaskArgs(BaseModel):
    node_id: int
    to_project_name: Optional[str] = None
    new_parent_id: Optional[int] = None

    @model_validator(mode="after")
    def check_destination(self):
        has_name = bool((self.to_project_name or "").strip())
        has_id = self.new_parent_id is not None
        if has_name == has_id:
            raise ValueError("move_task needs exactly one destination: to_project_name "
                             "(a project by name) OR new_parent_id (another task by ID).")
        if has_id and self.new_parent_id == self.node_id:
            raise ValueError("A task cannot be moved under itself.")
        return self


class OpenViewArgs(BaseModel):
    view: str
    project_name: Optional[str] = None
    node_ids: Optional[List[int]] = None
    label: Optional[str] = None

    @field_validator("view")
    @classmethod
    def check_view(cls, value):
        if value not in ("today", "projects", "project", "list", "loose"):
            raise ValueError("view must be one of: today, projects, project, list")
        return value

    @model_validator(mode="after")
    def check_required_fields(self):
        if self.view == "project" and not (self.project_name or "").strip():
            raise ValueError("view 'project' requires project_name (the project as the user named it)")
        if self.view == "list" and not self.node_ids:
            raise ValueError("view 'list' requires node_ids (IDs from ACTIVE TASKS or search results)")
        return self


class SearchTasksArgs(BaseModel):
    query: str = Field(min_length=1)


LENS_TOOLS: Any = [
    {
        "type": "function",
        "function": {
            "name": "capture_tasks",
            "description": "Create one or more new tasks/projects from the user's message. Put EVERY distinct task mentioned into the tasks array. Use subtasks for a project's steps, parent_id to attach to an existing item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Concise imperative description of the task or project."},
                                "deadline": {"type": "string", "description": "The deadline as the user said it ('Friday', 'end of month') or YYYY-MM-DD. Omit if no date was mentioned."},
                                "parent_id": {"type": "integer", "description": "ID of an EXISTING active task/project this belongs to."},
                                "relationship": {"type": "string", "enum": ["is_part_of", "blocks", "depends_on", "related_to"], "description": "How this attaches to parent_id. Default is_part_of."},
                                "node_type": {"type": "string", "enum": ["task", "project"], "description": "Use 'project' for umbrella goals with steps."},
                                "priority": {"type": "string", "enum": ["high", "normal", "low"], "description": "Only when the user signals it: 'critical'/'really important'/'must do' = high; 'low priority'/'whenever'/'no rush' = low. Omit otherwise."},
                                "subtasks": {"type": "array", "items": {"type": "string"}, "description": "Step descriptions to create as children of this item."}
                            },
                            "required": ["content"]
                        }
                    }
                },
                "required": ["tasks"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_tasks",
            "description": "Mark one or more EXISTING active tasks as completed. Use the IDs from the active task list. Include every task the user said they finished, in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {"type": "array", "items": {"type": "integer"}, "description": "IDs of the finished tasks."}
                },
                "required": ["node_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": "Edit an existing task: reword it, change/set its deadline, change its status (active, on_hold, cold_storage), or promote it to a project (node_type 'project'). Only pass the fields that change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer", "description": "ID of the task to edit."},
                    "content": {"type": "string", "description": "New wording, if rewording."},
                    "deadline": {"type": "string", "description": "New deadline, as the user said it ('next Monday') or YYYY-MM-DD."},
                    "status": {"type": "string", "enum": ["active", "on_hold", "cold_storage"], "description": "New status (use complete_tasks for completion)."},
                    "priority": {"type": "string", "enum": ["high", "normal", "low"], "description": "New importance level, when the user raises or lowers it."},
                    "description": {"type": "string", "description": "Longer detail/notes for the task, when the user dictates context to attach. Use their words; never invent."},
                    "node_type": {"type": "string", "enum": ["task", "project"], "description": "Set 'project' to promote a task into a top-level project (it detaches from any parent project and its subtasks become the project's tasks)."}
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "link_tasks",
            "description": "Create a relationship between two EXISTING tasks. For 'X blocks Y', X is parent_id and Y is child_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {"type": "integer", "description": "Source task ID (the blocker, or the project)."},
                    "child_id": {"type": "integer", "description": "Target task ID (the blocked task, or the part)."},
                    "relationship": {"type": "string", "enum": ["is_part_of", "blocks", "depends_on", "related_to"]}
                },
                "required": ["parent_id", "child_id", "relationship"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_task",
            "description": "RELOCATE an existing task to a new home — detaches it from its current project(s) and files it under the destination. Use for 'move X to Y' / 'file X under Y instead'. (To ADD a home while keeping the others, use link_tasks with is_part_of instead.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer", "description": "ID of the task to move."},
                    "to_project_name": {"type": "string", "description": "Destination project, named as the user said it (the system resolves it). Use this when moving into a project."},
                    "new_parent_id": {"type": "integer", "description": "Destination task ID, when moving the task UNDER another existing task (making it a subtask). Use an ID from ACTIVE TASKS or search results."}
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_view",
            "description": "Navigate the Lens pane, like clicking around a project manager app. 'project' opens one project and shows ALL its open tasks — pass project_name as the user said it, the system finds it. 'projects' shows the all-projects overview. 'today' returns to the default most-urgent view (also for 'go back' / 'clear'). 'list' shows specific tasks by ID (for category requests, after picking IDs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "enum": ["today", "projects", "project", "list"], "description": "Which view to open."},
                    "project_name": {"type": "string", "description": "For view 'project': the project's name as the user referred to it (e.g. 'The Cage')."},
                    "node_ids": {"type": "array", "items": {"type": "integer"}, "description": "For view 'list': the task IDs to show."},
                    "label": {"type": "string", "description": "For view 'list': a short title, e.g. 'errands'."}
                },
                "required": ["view"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_tasks",
            "description": "Search the task database by keywords. Use when the user mentions a task or project that is not in the ACTIVE TASKS list — search first, then act on the returned IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "2-3 distinctive keywords from the user's request (e.g. 'cage film', 'invoices')."}
                },
                "required": ["query"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _existing_active(content: str) -> Optional[dict]:
    node = models.find_node_by_content(content)
    return node if node and node["status"] == "active" else None


def _execute_capture(args: CaptureTasksArgs) -> dict:
    created, skipped = [], []
    for item in args.tasks:
        if item.parent_id is not None and not models.existing_node_ids([item.parent_id]):
            return {"error": f"parent_id {item.parent_id} does not exist. Use an ID from the active task list or omit parent_id."}

    for item in args.tasks:
        # Models sometimes re-capture work from conversation history; an exact
        # active-content match is reused instead of duplicated.
        existing = _existing_active(item.content)
        if existing:
            skipped.append({"id": existing["id"], "content": item.content,
                            "note": "already existed, not duplicated"})
            node_id = existing["id"]
        else:
            node_id = models.add_node(content=item.content, target_date=item.deadline,
                                      node_type=item.node_type, priority=item.priority)
            created.append({"id": node_id, "content": item.content,
                            "deadline": item.deadline or "none",
                            "priority": item.priority})
            embeddings.index_node(node_id, item.content)
        seen_node_ids.add(node_id)
        if item.parent_id is not None:
            models.add_edge(parent_id=item.parent_id, child_id=node_id, relationship=item.relationship)
        for subtask in item.subtasks:
            existing_sub = _existing_active(subtask)
            if existing_sub:
                skipped.append({"id": existing_sub["id"], "content": subtask,
                                "note": "already existed, not duplicated"})
                models.add_edge(parent_id=node_id, child_id=existing_sub["id"], relationship="is_part_of")
                continue
            child_id = models.add_node(content=subtask, node_type="task")
            models.add_edge(parent_id=node_id, child_id=child_id, relationship="is_part_of")
            created.append({"id": child_id, "content": subtask})
            embeddings.index_node(child_id, subtask)
            seen_node_ids.add(child_id)

    result = {"success": True, "created": created}
    if skipped:
        result["skipped_duplicates"] = skipped
    return result


def _execute_complete(args: CompleteTasksArgs) -> dict:
    completed = models.complete_nodes(args.node_ids)
    missing = [node_id for node_id in args.node_ids if node_id not in completed]
    result = {"success": bool(completed), "completed_ids": completed}
    if missing:
        result["unknown_ids"] = missing
        result["hint"] = "These IDs do not exist. Re-check the active task list."
    return result


def _execute_update(args: UpdateTaskArgs) -> dict:
    if not models.existing_node_ids([args.node_id]):
        return {"error": f"Task {args.node_id} does not exist. Use an ID from the active task list."}
    models.update_node(args.node_id, content=args.content, status=args.status,
                       target_date=args.deadline, priority=args.priority,
                       description=args.description, node_type=args.node_type)
    if args.content is not None:
        embeddings.index_node(args.node_id, args.content)
    result: dict = {"success": True, "node_id": args.node_id}
    # Promoting to a project makes the node top-level: drop the is_part_of edges
    # that filed it under its old project(s); its own subtasks come along.
    if args.node_type == "project":
        result["detached"] = models.detach_parents(args.node_id)
    changed = {k: v for k, v in (("content", args.content), ("status", args.status),
                                 ("deadline", args.deadline), ("priority", args.priority),
                                 ("description", args.description), ("node_type", args.node_type))
               if v is not None}
    result["changed"] = changed
    return result


def _execute_link(args: LinkTasksArgs) -> dict:
    missing = [i for i in (args.parent_id, args.child_id) if not models.existing_node_ids([i])]
    if missing:
        return {"error": f"Task IDs {missing} do not exist. Use IDs from the active task list."}
    models.add_edge(args.parent_id, args.child_id, args.relationship)
    return {"success": True, "parent_id": args.parent_id, "child_id": args.child_id,
            "relationship": args.relationship}


def _execute_move(args: MoveTaskArgs) -> dict:
    """Relocate a task: drop ALL its current is_part_of parents, then file it
    under exactly one destination (a project by name, or another task by ID)."""
    from app.engine import views

    if not models.existing_node_ids([args.node_id]):
        return {"error": f"Task {args.node_id} does not exist. Use an ID from the active task list."}

    if args.to_project_name:
        resolved = views.resolve_project(args.to_project_name)
        if isinstance(resolved, list):
            if not resolved:
                return {"error": (f"No project found matching {args.to_project_name!r}. "
                                  "If it's a NEW project, capture it first; to move under an "
                                  "existing task, pass new_parent_id instead.")}
            return {"error": f"Multiple projects match {args.to_project_name!r} — ask the user which one.",
                    "candidates": [{"id": p["id"], "name": p["content"]} for p in resolved]}
        target_id = resolved["id"]
    else:
        if not models.existing_node_ids([args.new_parent_id]):
            return {"error": f"Destination task {args.new_parent_id} does not exist. Use an ID from the active task list."}
        target_id = args.new_parent_id

    detached = models.detach_parents(args.node_id)
    models.add_edge(target_id, args.node_id, "is_part_of")
    seen_node_ids.update({args.node_id, target_id})
    return {"success": True, "node_id": args.node_id, "to": _node_content(target_id),
            "detached": detached}


def _execute_open_view(args: OpenViewArgs) -> dict:
    """Navigation: the model names a destination; Python resolves it into a view.
    The model never picks which tasks a project view contains — that's a query."""
    from app.engine import views

    if args.view == "project":
        resolved = views.resolve_project(args.project_name or "")
        if isinstance(resolved, list):
            if not resolved:
                return {"error": (f"No project found matching {args.project_name!r}. "
                                  "If the user wants tasks (not a project), use search_tasks + "
                                  "a 'list' view; if it's a NEW project, capture it first.")}
            return {"error": f"Multiple projects match {args.project_name!r} — ask the user which one.",
                    "candidates": [{"id": p["id"], "name": p["content"]} for p in resolved]}
        if resolved["status"] != "active":
            return {"error": (f"Project '{resolved['content']}' is on hold. To work on it, "
                              "resume it first with update_task status 'active', then open it.")}
        views.set_view({"mode": "node", "path": [resolved["id"]]})
        open_ids = models.get_active_child_ids(resolved["id"])
        seen_node_ids.update(open_ids)
        seen_node_ids.add(resolved["id"])
        return {"success": True, "view": "project", "project": resolved["content"],
                "open_tasks": len(open_ids)}

    if args.view == "list":
        active = [i for i in models.existing_node_ids(args.node_ids or [])
                  if (models.get_node(i) or {}).get("status") == "active"]
        if not active:
            return {"error": "None of those IDs are active tasks. Use IDs from ACTIVE TASKS or search_tasks results."}
        views.set_view({"mode": "list", "node_ids": active, "label": args.label or "selected tasks"})
        return {"success": True, "view": "list", "label": args.label or "selected tasks",
                "shown": len(active)}

    views.set_view({"mode": args.view})
    result = {"success": True, "view": args.view}
    if args.view == "projects":
        result["project_count"] = sum(
            1 for n in models.get_active_nodes() if n.get("node_type") == "project")
    return result


def _execute_search(args: SearchTasksArgs) -> dict:
    from app.engine import retrieval
    results = retrieval.search_active(args.query, limit=15)
    seen_node_ids.update(n["id"] for n in results)
    return {
        "results": [
            {"id": n["id"], "content": n["content"],
             **({"due": n["target_date"]} if n.get("target_date") else {}),
             **({"status": "on_hold"} if n.get("status") == "on_hold" else {})}
            for n in results
        ],
        "hint": "Use these IDs with open_view (list) / complete_tasks / update_task / link_tasks."
        if results else ("No matches in the database. If the user is describing NEW work, "
                         "call capture_tasks now. If they referred to an existing task, "
                         "try other keywords or tell them it wasn't found."),
    }


TOOL_HANDLERS = {
    "capture_tasks": (CaptureTasksArgs, _execute_capture),
    "complete_tasks": (CompleteTasksArgs, _execute_complete),
    "update_task": (UpdateTaskArgs, _execute_update),
    "link_tasks": (LinkTasksArgs, _execute_link),
    "move_task": (MoveTaskArgs, _execute_move),
    "open_view": (OpenViewArgs, _execute_open_view),
    "search_tasks": (SearchTasksArgs, _execute_search),
}


def _referenced_ids(func_name: str, args) -> List[int]:
    if func_name == "complete_tasks":
        return list(args.node_ids)
    if func_name == "open_view":
        return list(args.node_ids or [])  # 'list' views must use IDs the model has seen
    if func_name == "update_task":
        return [args.node_id]
    if func_name == "link_tasks":
        return [args.parent_id, args.child_id]
    if func_name == "move_task":
        # to_project_name is name-resolved (no ID to ground); an explicit
        # new_parent_id destination must be grounded, like link_tasks.
        return [args.node_id] + ([args.new_parent_id] if args.new_parent_id is not None else [])
    if func_name == "capture_tasks":
        return [item.parent_id for item in args.tasks if item.parent_id is not None]
    return []


def execute_tool_call(call, enforce_grounding: bool = False) -> str:
    """Validates and executes one tool call; returns a JSON result string for the model.

    With enforce_grounding, any referenced node ID the model was never shown
    (context, search results, own captures) is rejected — models guess
    plausible-looking IDs, and a guessed ID can hit the wrong real task."""
    func_name = call.function.name

    handler = TOOL_HANDLERS.get(func_name)
    if handler is None:
        return json.dumps({"error": f"Unknown function: {func_name}"})

    schema, executor = handler
    try:
        raw_args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "Arguments were not valid JSON. Retry with valid JSON."})

    try:
        args = schema.model_validate(raw_args)
    except ValidationError as exc:
        issues = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors())
        return json.dumps({"error": f"Invalid arguments: {issues}"})

    if enforce_grounding:
        unseen = [i for i in _referenced_ids(func_name, args) if i not in seen_node_ids]
        if unseen:
            return json.dumps({"error": (
                f"IDs {unseen} are not in your ACTIVE TASKS or any search result — do not "
                "guess IDs. Call search_tasks with keywords from the user's request, then "
                "use the IDs it returns.")})

    try:
        return json.dumps(executor(args))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Templated confirmations — routine outcomes don't need a second LLM call
# ---------------------------------------------------------------------------

def _node_content(node_id) -> str:
    node = models.get_node(node_id)
    return node["content"] if node else f"task {node_id}"


_STATUS_PHRASES = {"on_hold": "on hold", "active": "active again", "cold_storage": "archived"}


def _confirmation_sentences(name: str, result: dict) -> Optional[List[str]]:
    """Sentences for ONE clean tool result. Returns None when the outcome needs
    the model's judgment (errors, unknown IDs); [] when there is nothing to say
    (searches). Streaming emits these per action as they execute."""
    if "error" in result:
        return None
    sentences: List[str] = []

    if name == "capture_tasks":
        created = result.get("created", [])
        if len(created) == 1:
            item = created[0]
            due = f" — due {item['deadline']}" if item.get("deadline") not in (None, "none") else ""
            sentences.append(f"Captured **{item['content']}**{due}.")
        elif created:
            sentences.append(f"Captured {len(created)} tasks — they're in your Lens.")
        skipped = result.get("skipped_duplicates", [])
        if skipped:
            sentences.append(f"({len(skipped)} already existed, not duplicated.)")

    elif name == "complete_tasks":
        if result.get("unknown_ids"):
            return None  # the model should sort out what the user meant
        ids = result.get("completed_ids", [])
        if len(ids) == 1:
            sentences.append(f"Checked off **{_node_content(ids[0])}**. ✓")
        elif ids:
            sentences.append(f"Checked off {len(ids)} tasks. ✓")

    elif name == "update_task":
        changed = result.get("changed", {})
        content = _node_content(result["node_id"])
        if changed.get("node_type") == "project":
            sentences.append(f"Promoted **{content}** to its own project.")
            return sentences
        parts = []
        if "deadline" in changed:
            parts.append(f"now due {changed['deadline']}")
        if "status" in changed:
            parts.append(_STATUS_PHRASES.get(changed["status"], changed["status"]))
        if "priority" in changed:
            parts.append(f"{changed['priority']} priority")
        detail = f" — {', '.join(parts)}" if parts else ""
        sentences.append(f"Updated **{content}**{detail}.")

    elif name == "link_tasks":
        parent = _node_content(result["parent_id"])
        child = _node_content(result["child_id"])
        if result["relationship"] == "blocks":
            sentences.append(f"Noted: **{parent}** blocks **{child}**.")
        elif result["relationship"] == "is_part_of":
            sentences.append(f"Filed **{child}** under **{parent}**.")
        else:
            sentences.append(f"Linked **{parent}** and **{child}**.")

    elif name == "move_task":
        content = _node_content(result["node_id"])
        sentences.append(f"Moved **{content}** to **{result['to']}**.")

    elif name == "open_view":
        view = result.get("view")
        if view == "project":
            count = result.get("open_tasks", 0)
            plural = "task" if count == 1 else "tasks"
            sentences.append(f"Opened **{result['project']}** — {count} open {plural}.")
        elif view == "projects":
            count = result.get("project_count", 0)
            sentences.append(f"Showing your **{count} projects**.")
        elif view == "list":
            count = result.get("shown", 0)
            plural = "task" if count == 1 else "tasks"
            sentences.append(f"Showing **{result.get('label', 'selection')}** — {count} {plural}.")
        else:
            sentences.append("Here's your day.")

    # search_tasks contributes no sentence; an action that followed it speaks
    return sentences


def template_confirmation(executed: List[tuple]) -> Optional[str]:
    """Builds the user-facing confirmation in code when every executed tool had a
    clean, routine outcome. Returns None when nuance is needed (errors, unknown
    IDs, search-only turns) — those fall back to an LLM wrap-up call."""
    sentences: List[str] = []
    for name, result in executed:
        per_result = _confirmation_sentences(name, result)
        if per_result is None:
            return None
        sentences.extend(per_result)
    return " ".join(sentences[:2]) if sentences else None


# ---------------------------------------------------------------------------
# Conversation loop (streaming)
# ---------------------------------------------------------------------------
#
# Event protocol (NDJSON over /api/chat):
#   {"type": "status", "text": "Thinking…"}    transient placeholder
#   {"type": "action", "text": "Captured…"}    a templated confirmation, as it happens
#   {"type": "token",  "text": "The "}         streamed LLM reply text
#   {"type": "replace","text": ""}             retract optimistically streamed text
#   {"type": "lens"}                           internal: state changed, push to WS clients
#   {"type": "done",   "reply": "…"}           authoritative final reply

class _StreamedCall:
    """A tool call reassembled from streaming deltas (same shape execute_tool_call expects)."""
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _SalvagedFunction(name, arguments)


async def _collect_stream(stream, sink: List[str]):
    """Consumes a streaming completion. Yields ("token", text, None) for content
    deltas that arrive BEFORE any tool-call delta (also appended to sink so the
    caller knows what was shown), then one ("final", content, tool_calls).
    The caller decides whether forwarded tokens stand or get retracted."""
    content_parts: List[str] = []
    slots: dict = {}
    saw_tool_delta = False
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        for tc in (delta.tool_calls or []):
            saw_tool_delta = True
            slot = slots.setdefault(tc.index, {"id": tc.id or f"stream_{tc.index}",
                                               "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
        if delta.content:
            content_parts.append(delta.content)
            if not saw_tool_delta:
                sink.append(delta.content)
                yield ("token", delta.content, None)
    calls = [_StreamedCall(s["id"], s["name"], s["arguments"])
             for _, s in sorted(slots.items()) if s["name"]]
    yield ("final", "".join(content_parts), calls)


async def process_user_input_events(user_text: str):
    """Runs the tool-calling loop, yielding protocol events as the turn unfolds.
    The 'done' event always arrives last and carries the full reply text."""
    # KV-cache-friendly construction (see CONTEXT_TEMPLATE comment): the
    # volatile context rides inside the newest user message so the static
    # prompt + history remain a byte-identical, cacheable prefix.
    context_block = format_context_prompt(user_text)
    _append_history({"role": "user", "content": user_text})
    messages: List[Any] = (
        [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}]
        + conversation_history[:-1]
        + [{"role": "user",
            "content": f"{context_block}\n\nUSER MESSAGE: {user_text}"}]
    )

    reply = None
    ran_tools = False
    salvage_attempted = False
    executed: List[tuple] = []  # (tool_name, parsed_result) for templated confirmations
    streamed_reply = ""         # stage-1 tokens already shown to the user

    yield {"type": "status", "text": "Thinking…"}

    # Stage 1 — decide and act. The decision prompt carries no style rules, so a
    # small model doesn't skip the tool call and jump straight to a styled reply.
    for _ in range(MAX_TOOL_ROUNDS):
        content = ""
        tool_calls: List[Any] = []
        forwarded: List[str] = []
        try:
            stream = await client.chat.completions.create(
                model=config.AI_MODEL_NAME,
                messages=messages,
                tools=LENS_TOOLS,
                tool_choice="auto",
                temperature=0.2,  # small models drift into skipping tool calls at higher temps
                stream=True,
            )
            async for kind, first, second in _collect_stream(stream, forwarded):
                if kind == "token":
                    # Optimistic: most no-tool responses are the actual reply.
                    yield {"type": "token", "text": first}
                else:
                    content, tool_calls = first, list(second or [])
        except Exception as exc:
            reply = (f"Warning: inference engine unreachable — is LM Studio running "
                     f"with {config.AI_MODEL_NAME} loaded? Details: {exc}")
            break
        streamed_reply = "".join(forwarded)

        if not tool_calls:
            salvaged = None
            if not salvage_attempted and TOOL_NAME_LEAK.search(content):
                salvage_attempted = True
                if streamed_reply:  # what we streamed was a leaked call, not a reply
                    yield {"type": "replace", "text": ""}
                    streamed_reply = ""
                salvaged = salvage_tool_call(content)
                if salvaged is None:
                    # Leak detected but unparseable ("open_view with the errand
                    # tasks...") — nudge once and let the model emit it properly.
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "system", "content": (
                        "You described a tool call in plain text — nothing happened. "
                        "Now CALL the tool for real, with arguments matching its schema "
                        "and IDs from ACTIVE TASKS.")})
                    continue
            if salvaged is None:
                if not ran_tools:
                    reply = content  # no rule applied; the plain answer IS the reply
                break
            # The model wrote the call as prose; execute the parsed equivalent.
            tool_calls = [salvaged]

        if streamed_reply and tool_calls:
            # Content preceding a real tool call is reasoning spill, not a reply.
            yield {"type": "replace", "text": ""}
            streamed_reply = ""
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name,
                             "arguments": call.function.arguments},
            } for call in tool_calls],
        })

        ran_tools = True
        round_results = []
        for raw_call in tool_calls:
            tool_call: Any = raw_call
            result_json = execute_tool_call(tool_call, enforce_grounding=True)
            result = json.loads(result_json)
            round_results.append((tool_call.function.name, result))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result_json,
            })
            for sentence in (_confirmation_sentences(tool_call.function.name, result) or []):
                yield {"type": "action", "text": sentence}
        executed.extend(round_results)

        if any(name != "search_tasks" and "error" not in result
               for name, result in round_results):
            yield {"type": "lens"}  # something changed — update the cards now

        # A clean round of pure actions is finished — don't spend another LLM
        # call asking the model whether it's done. Searches and errors continue
        # the loop so the model can act on results or recover.
        if all(name != "search_tasks" and "error" not in result
               for name, result in round_results):
            break
        if round_results and all(name == "search_tasks" for name, _ in round_results):
            yield {"type": "status", "text": "Searching your tasks…"}

    # Stage 2 — speak. Routine outcomes are confirmed from code (no second
    # LLM call — this roughly halves actioned-turn latency); anything nuanced
    # falls back to a wrap-up call with style rules and no tools on offer.
    if reply is None and ran_tools:
        reply = template_confirmation(executed)

    if reply is None and not streamed_reply:
        messages.append({"role": "system", "content": POST_TOOL_REMINDER})
        try:
            stream = await client.chat.completions.create(
                model=config.AI_MODEL_NAME, messages=messages,
                temperature=0.2, stream=True)
            forwarded = []
            async for kind, first, _second in _collect_stream(stream, forwarded):
                if kind == "token":
                    yield {"type": "token", "text": first}
                else:
                    reply = first
        except Exception as exc:
            reply = f"Done, but I couldn't generate a summary. Details: {exc}"
    elif reply is None:
        reply = streamed_reply  # the streamed stage-1 text was the reply

    reply = reply or "Done."
    _append_history({"role": "assistant", "content": reply})
    if ran_tools:
        models.add_digest("activity_log", f"User: {user_text}\nLens-Brain: {reply}")
    yield {"type": "done", "reply": reply}


async def process_user_input(user_text: str) -> str:
    """Non-streaming wrapper: drains the event stream and returns the final reply."""
    reply = "Done."
    async for event in process_user_input_events(user_text):
        if event["type"] == "done":
            reply = event["reply"]
    return reply


def reset_session():
    """Clears in-memory conversation state (used by tests and on server restart)."""
    conversation_history.clear()
    seen_node_ids.clear()
