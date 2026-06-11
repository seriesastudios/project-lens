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
from collections import deque
from datetime import datetime, timedelta
from typing import Any, List, Optional

import calendar as _calendar

import dateparser
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import config
from app.database import models

client = AsyncOpenAI(
    base_url=config.AI_BASE_URL,
    api_key="not-needed-for-local"
)

MAX_TOOL_ROUNDS = 5
HISTORY_TURNS = 20  # user+assistant messages kept as session memory

# Small models occasionally write the tool call as prose ("focus_lens [{...}]")
# instead of emitting a real one. LM Studio ignores tool_choice="required" and
# corrective retries come back empty, so we salvage by parsing the leaked text.
TOOL_NAME_LEAK = re.compile(
    r"\b(capture_tasks|complete_tasks|update_task|link_tasks|focus_lens|clear_focus)\b")


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

    if name == "clear_focus":
        return _SalvagedCall(name, {})
    if name in ("focus_lens", "complete_tasks"):
        if isinstance(parsed, list):
            ids = _coerce_ids(parsed)
            return _SalvagedCall(name, {"node_ids": ids}) if ids else None
        if isinstance(parsed, dict):
            if "node_ids" in parsed:
                return _SalvagedCall(name, parsed)
            ids = _coerce_ids([parsed])
            return _SalvagedCall(name, {"node_ids": ids}) if ids else None
        # Non-JSON leak ("focus_lens[node_ids=[6,5]]", "focus_lens for tasks 3 and 5"):
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
conversation_history: deque = deque(maxlen=HISTORY_TURNS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are Lens-Brain, the action engine of Project Lens, a personal task manager. Your job on every user message: decide which rule below applies, then CALL THE MATCHING TOOL. Tool calls are the only way anything happens — text output never creates, completes, or shows a task. Do not describe an action in text; perform it with a tool call.

The user sees this chat plus "The Lens", a panel of task cards. focus_lens controls what appears there. Tasks are shown in the Lens, never listed in chat.

NOW: {now}
CALENDAR (use this table for any relative date — do not compute weekdays yourself):
{calendar}
ACTIVE TASKS (use these exact IDs in tool calls):
{task_list}
FOCUSED NOW: {focus_list}

RULES
0. Act ONLY on the user's LATEST message. Earlier conversation is context for resolving references — everything in it has already been handled; never re-capture or re-complete from it.
1. The user mentions new work, a goal, or a commitment → call capture_tasks with EVERY distinct task in the message. A project with steps → one item with node_type "project" and the steps in subtasks. Set parent_id when it clearly belongs to an existing item. Never capture anything already in ACTIVE TASKS.
2. Deadlines: pass the deadline exactly as the user said it ("Friday", "end of month", "June 3") or as YYYY-MM-DD — the system converts relative dates itself. No date stated or implied → omit the field, never invent one.
3. The user finished something → call complete_tasks with ALL matching IDs from ACTIVE TASKS, in one call. Match loosely: "sent the invoices" matches "Send invoices to clients". Never create a task for finished work; if nothing matches, ask one short question.
4. Reword, change deadline, pause (on_hold), resume (active), or archive (cold_storage) → call update_task with only the fields that change.
5. A dependency or grouping is stated → call link_tasks. For "X blocks Y", X is parent_id. Use is_part_of for project/subtask grouping.
6. The user asks to focus on something, asks what their tasks are, or asks what to work on → call focus_lens with ALL matching active IDs (include a project's subtasks). NEVER list tasks in chat — the Lens shows them. Call clear_focus when asked to clear or reset the Lens.
7. "that", "it", "the second one" → resolve from the recent conversation and ACTIVE TASKS; if genuinely ambiguous, ask one short question — never guess an ID. A correction ("no, I meant Friday") → call update_task on the existing task, never capture a new one.

Reply in plain text ONLY when no rule applies: a single-fact question (one deadline, one status), a greeting, venting, or reflection — answer briefly, capture nothing. For mixed messages, call tools for the actionable part only."""

POST_TOOL_REMINDER = (
    "Tools executed. Now reply to the user in at most two short sentences confirming what "
    "changed, bolding task names with **double asterisks**. NEVER list tasks line by line — "
    "name at most two, otherwise summarize the count (\"Added 3 tasks — they're in your "
    "Lens.\"). The Lens pane shows the tasks; chat does not. Never mention IDs, tools, or "
    "technical details."
)


def _calendar_table(now: datetime, days: int = 14) -> str:
    """The next two weeks as weekday→date lines; small models fail weekday math."""
    lines = []
    for offset in range(days):
        day = now + timedelta(days=offset)
        label = {0: " (today)", 1: " (tomorrow)"}.get(offset, "")
        lines.append(f"{day.strftime('%A')} = {day.strftime('%Y-%m-%d')}{label}")
    return "\n".join(lines)


def format_system_prompt() -> str:
    now = datetime.now()
    active_nodes = models.get_active_nodes()

    if active_nodes:
        lines = []
        for node in active_nodes:
            parts = [f"[ID {node['id']}]", node["content"]]
            if node.get("node_type") == "project":
                parts.append("(project)")
            if node.get("target_date"):
                parts.append(f"(due {node['target_date']})")
            lines.append(" ".join(parts))
        task_list = "\n".join(lines)
        focused = [f"[ID {n['id']}] {n['content']}" for n in active_nodes if (n.get("focus_score") or 0) > 0]
        focus_list = "; ".join(focused) if focused else "nothing"
    else:
        task_list = "No active tasks."
        focus_list = "nothing"

    return SYSTEM_PROMPT_TEMPLATE.format(
        now=now.strftime("%A, %Y-%m-%d %H:%M"),
        calendar=_calendar_table(now),
        task_list=task_list,
        focus_list=focus_list,
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
    for prefix in ("by ", "due ", "on ", "before "):
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):]
    phrase = phrase.replace("next ", "").replace("this ", "")

    if "end of month" in phrase or "end of the month" in phrase:
        last_day = _calendar.monthrange(base.year, base.month)[1]
        return base.replace(day=last_day).date().isoformat()
    if "end of week" in phrase or "end of the week" in phrase:
        return (base + timedelta(days=(6 - base.weekday()))).date().isoformat()

    parsed = dateparser.parse(phrase, settings={
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base,
    })
    if parsed is None:
        raise ValueError(f"Could not understand deadline {value!r}. Pass YYYY-MM-DD.")
    return parsed.date().isoformat()


class TaskItem(BaseModel):
    content: str = Field(min_length=1)
    deadline: Optional[str] = None
    parent_id: Optional[int] = None
    relationship: str = "is_part_of"
    node_type: str = "task"
    subtasks: List[str] = Field(default_factory=list)

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


class FocusLensArgs(BaseModel):
    node_ids: List[int] = Field(min_length=1)


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
            "description": "Edit an existing task: reword it, change/set its deadline, or change its status (active, on_hold, cold_storage). Only pass the fields that change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer", "description": "ID of the task to edit."},
                    "content": {"type": "string", "description": "New wording, if rewording."},
                    "deadline": {"type": "string", "description": "New deadline, as the user said it ('next Monday') or YYYY-MM-DD."},
                    "status": {"type": "string", "enum": ["active", "on_hold", "cold_storage"], "description": "New status (use complete_tasks for completion)."}
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
            "name": "focus_lens",
            "description": "Show specific tasks in the Lens pane. Replaces the current focus set. Use whenever the user asks to focus on something, asks what their tasks are, or asks what to work on. Include ALL matching active task IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {"type": "array", "items": {"type": "integer"}, "description": "IDs of every task to spotlight."}
                },
                "required": ["node_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_focus",
            "description": "Clear the Lens focus so it returns to showing only deadline-driven items. Use when the user asks to clear/reset the Lens.",
            "parameters": {"type": "object", "properties": {}}
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
            node_id = models.add_node(content=item.content, target_date=item.deadline, node_type=item.node_type)
            created.append({"id": node_id, "content": item.content,
                            "deadline": item.deadline or "none"})
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
    models.update_node(args.node_id, content=args.content, status=args.status, target_date=args.deadline)
    changed = {k: v for k, v in (("content", args.content), ("status", args.status),
                                 ("deadline", args.deadline)) if v is not None}
    return {"success": True, "node_id": args.node_id, "changed": changed}


def _execute_link(args: LinkTasksArgs) -> dict:
    missing = [i for i in (args.parent_id, args.child_id) if not models.existing_node_ids([i])]
    if missing:
        return {"error": f"Task IDs {missing} do not exist. Use IDs from the active task list."}
    models.add_edge(args.parent_id, args.child_id, args.relationship)
    return {"success": True}


def _execute_focus(args: FocusLensArgs) -> dict:
    focused = models.set_focus(args.node_ids)
    missing = [node_id for node_id in args.node_ids if node_id not in focused]
    result = {"success": bool(focused), "focused_ids": focused}
    if missing:
        result["unknown_ids"] = missing
    return result


def _execute_clear_focus() -> dict:
    models.clear_all_focus()
    return {"success": True}


TOOL_HANDLERS = {
    "capture_tasks": (CaptureTasksArgs, _execute_capture),
    "complete_tasks": (CompleteTasksArgs, _execute_complete),
    "update_task": (UpdateTaskArgs, _execute_update),
    "link_tasks": (LinkTasksArgs, _execute_link),
    "focus_lens": (FocusLensArgs, _execute_focus),
}


def execute_tool_call(call) -> str:
    """Validates and executes one tool call; returns a JSON result string for the model."""
    func_name = call.function.name

    if func_name == "clear_focus":
        return json.dumps(_execute_clear_focus())

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

    try:
        return json.dumps(executor(args))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

async def process_user_input(user_text: str) -> str:
    """Runs the tool-calling loop and returns the assistant's reply text."""
    conversation_history.append({"role": "user", "content": user_text})
    messages: List[Any] = [{"role": "system", "content": format_system_prompt()}] + list(conversation_history)

    reply = None
    ran_tools = False
    salvage_attempted = False

    # Stage 1 — decide and act. The decision prompt carries no style rules, so a
    # small model doesn't skip the tool call and jump straight to a styled reply.
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = await client.chat.completions.create(
                model=config.AI_MODEL_NAME,
                messages=messages,
                tools=LENS_TOOLS,
                tool_choice="auto",
                temperature=0.2,  # small models drift into skipping tool calls at higher temps
            )
        except Exception as exc:
            return f"Warning: inference engine unreachable — is LM Studio running with {config.AI_MODEL_NAME} loaded? Details: {exc}"

        message = response.choices[0].message

        tool_calls: List[Any] = list(message.tool_calls or [])

        if not tool_calls:
            content = message.content or ""
            salvaged = None
            if not ran_tools and not salvage_attempted:
                salvage_attempted = True
                salvaged = salvage_tool_call(content)
            if salvaged is None:
                if not ran_tools:
                    reply = content  # no rule applied; the plain answer IS the reply
                break
            # The model wrote the call as prose; execute the parsed equivalent.
            tool_calls = [salvaged]
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": salvaged.id,
                    "type": "function",
                    "function": {"name": salvaged.function.name,
                                 "arguments": salvaged.function.arguments},
                }],
            })
        else:
            messages.append(message)

        ran_tools = True
        for raw_call in tool_calls:
            tool_call: Any = raw_call
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": execute_tool_call(tool_call),
            })

    # Stage 2 — speak. After tools ran, generate the user-facing confirmation
    # in a separate call with style rules and no tools on offer.
    if reply is None:
        messages.append({"role": "system", "content": POST_TOOL_REMINDER})
        try:
            final = await client.chat.completions.create(
                model=config.AI_MODEL_NAME, messages=messages, temperature=0.2)
            reply = final.choices[0].message.content
        except Exception as exc:
            reply = f"Done, but I couldn't generate a summary. Details: {exc}"

    reply = reply or "Done."
    conversation_history.append({"role": "assistant", "content": reply})
    if ran_tools:
        models.add_digest("activity_log", f"User: {user_text}\nLens-Brain: {reply}")
    return reply


def reset_session():
    """Clears in-memory conversation state (used by tests and on server restart)."""
    conversation_history.clear()
