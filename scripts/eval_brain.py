"""Live eval for the Lens-Brain prompt: runs the situation inventory against the
local model and reports which situations produce the expected tool call.

Usage: ./venv/bin/python scripts/eval_brain.py [passes]

Uses a throwaway database seeded with representative tasks; your real lens.db
is never touched. Each case checks the model's FIRST response only (which tool
it chose and rough argument sanity), which is where a small model fails.
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
config.DATABASE_PATH = _tmp.name

from app.database import models
from app.engine import brain


def seed_tasks():
    website = models.add_node("Website redesign", node_type="project")
    mockups = models.add_node("Finish Figma mockups", target_date="2026-06-15")
    landing = models.add_node("Build landing page")
    invoices = models.add_node("Send invoices to clients")
    dentist = models.add_node("Book dentist appointment", target_date="2026-06-20")
    groceries = models.add_node("Buy groceries for the week")
    models.add_edge(website, mockups, "is_part_of")
    models.add_edge(website, landing, "is_part_of")
    models.add_edge(mockups, landing, "blocks")
    return {"website": website, "mockups": mockups, "landing": landing,
            "invoices": invoices, "dentist": dentist, "groceries": groceries}


async def run_decision_stage(user_text, max_rounds=4):
    """Mirrors the app's stage-1 loop: multi-round tool calling with execution,
    leak salvage, and grounding. Returns every successfully EXECUTED call."""
    brain.reset_session()
    messages = [
        {"role": "system", "content": brain.SYSTEM_PROMPT_TEMPLATE},
        {"role": "system", "content": brain.format_context_prompt(user_text)},
        {"role": "user", "content": user_text},
    ]
    executed = []
    final_text = ""
    salvage_attempted = False
    ran_tools = False

    for _ in range(max_rounds):
        try:
            response = await brain.client.chat.completions.create(
                model=config.AI_MODEL_NAME, messages=messages,
                tools=brain.LENS_TOOLS, tool_choice="auto", temperature=0.2,
            )
        except Exception:
            # Transient LM Studio hiccup (model loading/swapped) — wait and retry once
            await asyncio.sleep(10)
            response = await brain.client.chat.completions.create(
                model=config.AI_MODEL_NAME, messages=messages,
                tools=brain.LENS_TOOLS, tool_choice="auto", temperature=0.2,
            )
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])

        if not tool_calls:
            content = message.content or ""
            if not salvage_attempted and brain.TOOL_NAME_LEAK.search(content):
                salvage_attempted = True
                salvaged = brain.salvage_tool_call(content)
                if salvaged is None:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "system", "content": (
                        "You described a tool call in plain text — nothing happened. "
                        "Now CALL the tool for real, with arguments matching its schema "
                        "and IDs from ACTIVE TASKS.")})
                    continue
                tool_calls = [salvaged]
                messages.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": salvaged.id, "type": "function",
                    "function": {"name": salvaged.function.name,
                                 "arguments": salvaged.function.arguments}}]})
            else:
                final_text = content
                break
        else:
            messages.append(message)

        ran_tools = True
        for c in tool_calls:
            result = brain.execute_tool_call(c, enforce_grounding=True)
            if "error" not in json.loads(result):
                executed.append((c.function.name, json.loads(c.function.arguments or "{}")))
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "name": c.function.name, "content": result})

    return executed, final_text


def build_cases(ids):
    """Each case: (label, user message, checker(calls, text) -> True/str)."""

    def expect_tool(name, arg_check=None):
        def check(calls, text):
            named = [args for tool, args in calls if tool == name]
            if not named:
                got = [tool for tool, _ in calls] or f"text: {text[:60]!r}"
                return f"expected {name}, got {got}"
            if arg_check:
                return arg_check(named[0])
            return True
        return check

    def expect_no_tool(calls, text):
        # search_tasks is read-only; only mutations count as wrongly acting
        mutating = [t for t, _ in calls if t != "search_tasks"]
        return True if not mutating else f"expected no action, got {mutating}"

    return [
        ("capture single + deadline", "I need to renew my passport before my Japan trip in September",
         expect_tool("capture_tasks", lambda a: True if a["tasks"] and a["tasks"][0].get("deadline") else "missing deadline")),
        ("capture multiple", "I have to call the accountant and also pick up the dry cleaning",
         expect_tool("capture_tasks", lambda a: True if len(a["tasks"]) >= 2 else f"only {len(a['tasks'])} task(s)")),
        ("project with steps", "New project: plan mom's birthday party. I need to book a restaurant, order a cake, and send invites.",
         expect_tool("capture_tasks", lambda a: True if any(t.get("subtasks") for t in a["tasks"]) or len(a["tasks"]) >= 3 else "no subtasks")),
        ("fuzzy completion", "just sent off the invoices",
         expect_tool("complete_tasks", lambda a: True if ids["invoices"] in a["node_ids"] else f"wrong ids {a['node_ids']}")),
        ("multiple completions", "finished the figma mockups and bought the groceries",
         expect_tool("complete_tasks", lambda a: True if {ids["mockups"], ids["groceries"]} <= set(a["node_ids"]) else f"ids {a['node_ids']}")),
        ("capture with importance", "Really important: I have to send the grant application, it's critical",
         expect_tool("capture_tasks", lambda a: True if a["tasks"][0].get("priority") == "high" else f"priority {a['tasks'][0].get('priority')}")),
        ("deprioritize existing task", "the groceries thing is low priority, no rush on it",
         expect_tool("update_task", lambda a: True if a["node_id"] == ids["groceries"] and a.get("priority") == "low" else f"args {a}")),
        ("edit deadline", "push the dentist appointment to July 3rd",
         expect_tool("update_task", lambda a: True if a["node_id"] == ids["dentist"] and a.get("deadline") == "2026-07-03" else f"args {a}")),
        ("pause task", "put the landing page on hold for now",
         expect_tool("update_task", lambda a: True if a["node_id"] == ids["landing"] and a.get("status") == "on_hold" else f"args {a}")),
        ("dependency", "buying groceries is blocked by sending the invoices",
         expect_tool("link_tasks", lambda a: True if a["relationship"] == "blocks" and a["parent_id"] == ids["invoices"] else f"args {a}")),
        ("focus project opens project view", "let's focus on the website redesign",
         expect_tool("open_view", lambda a: True if a["view"] == "project" and "website" in (a.get("project_name") or "").lower() else f"args {a}")),
        ("what to work on", "what should I be working on right now?",
         expect_tool("open_view", lambda a: True if a["view"] in ("today", "projects") else f"args {a}")),
        ("category query becomes a list view", "what are my errands?",
         expect_tool("open_view", lambda a: True if a["view"] == "list" and a.get("node_ids") else f"args {a}")),
        ("what's-left query opens project view", "what's left to do on the website redesign?",
         expect_tool("open_view", lambda a: True if a["view"] == "project" and "website" in (a.get("project_name") or "").lower() else f"args {a}")),
        ("show all projects", "show me all my projects",
         expect_tool("open_view", lambda a: True if a["view"] == "projects" else f"args {a}")),
        ("back to today", "ok go back to my day",
         expect_tool("open_view", lambda a: True if a["view"] == "today" else f"args {a}")),
        ("single fact in chat", "when is the dentist appointment due?",
         expect_no_tool),
        ("chitchat no capture", "ugh, today was completely exhausting",
         expect_no_tool),
        ("greeting no capture", "good morning!",
         expect_no_tool),
    ]


def reset_db():
    with models.DatabaseSession() as conn:
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM history_digest")
        conn.execute("DELETE FROM app_state")  # view state is sticky by design


async def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    models.init_db()
    case_count = len(build_cases({k: 0 for k in
                                  ("website", "mockups", "landing", "invoices", "dentist", "groceries")}))
    failures = 0

    for index in range(case_count):
        results, label = [], ""
        for _ in range(passes):
            # Tools now execute for real, so every pass gets a fresh seeded DB
            reset_db()
            ids = seed_tasks()
            label, message, checker = build_cases(ids)[index]
            calls, text = await run_decision_stage(message)
            results.append(checker(calls, text))
        ok = sum(1 for v in results if v is True)
        status = "PASS" if ok == passes else ("FLAKY" if ok else "FAIL")
        if ok < passes:
            failures += 1
        detail = "" if ok == passes else "  " + "; ".join(str(v) for v in results if v is not True)
        print(f"{status:5} {ok}/{passes}  {label}{detail}")

    print(f"\n{case_count - failures}/{case_count} situations fully reliable")
    os.unlink(_tmp.name)


if __name__ == "__main__":
    asyncio.run(main())
