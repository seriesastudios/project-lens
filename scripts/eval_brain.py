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


def seed():
    models.init_db()
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


async def first_response(user_text):
    """Mirrors the app's stage-1 decision call, including the tool-name-leak salvage."""
    messages = [
        {"role": "system", "content": brain.format_system_prompt()},
        {"role": "user", "content": user_text},
    ]
    response = await brain.client.chat.completions.create(
        model=config.AI_MODEL_NAME, messages=messages,
        tools=brain.LENS_TOOLS, tool_choice="auto", temperature=0.2,
    )
    message = response.choices[0].message
    tool_calls = list(message.tool_calls or [])
    if not tool_calls:
        salvaged = brain.salvage_tool_call(message.content or "")
        if salvaged:
            tool_calls = [salvaged]
    calls = [(c.function.name, json.loads(c.function.arguments or "{}"))
             for c in tool_calls]
    return calls, message.content or ""


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
        return True if not calls else f"expected no tool, got {[t for t, _ in calls]}"

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
        ("edit deadline", "push the dentist appointment to July 3rd",
         expect_tool("update_task", lambda a: True if a["node_id"] == ids["dentist"] and a.get("deadline") == "2026-07-03" else f"args {a}")),
        ("pause task", "put the landing page on hold for now",
         expect_tool("update_task", lambda a: True if a["node_id"] == ids["landing"] and a.get("status") == "on_hold" else f"args {a}")),
        ("dependency", "buying groceries is blocked by sending the invoices",
         expect_tool("link_tasks", lambda a: True if a["relationship"] == "blocks" and a["parent_id"] == ids["invoices"] else f"args {a}")),
        ("focus project", "let's focus on the website redesign",
         expect_tool("focus_lens", lambda a: True if ids["mockups"] in a["node_ids"] or ids["website"] in a["node_ids"] else f"ids {a['node_ids']}")),
        ("what to work on", "what should I be working on right now?",
         expect_tool("focus_lens")),
        ("list query goes to lens", "what are my errands?",
         expect_tool("focus_lens")),
        ("clear lens", "clear the lens please",
         expect_tool("clear_focus")),
        ("single fact in chat", "when is the dentist appointment due?",
         expect_no_tool),
        ("chitchat no capture", "ugh, today was completely exhausting",
         expect_no_tool),
        ("greeting no capture", "good morning!",
         expect_no_tool),
    ]


async def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ids = seed()
    cases = build_cases(ids)
    failures = 0

    for label, message, checker in cases:
        results = []
        for _ in range(passes):
            calls, text = await first_response(message)
            verdict = checker(calls, text)
            results.append(verdict)
        ok = sum(1 for v in results if v is True)
        status = "PASS" if ok == passes else ("FLAKY" if ok else "FAIL")
        if ok < passes:
            failures += 1
        detail = "" if ok == passes else "  " + "; ".join(str(v) for v in results if v is not True)
        print(f"{status:5} {ok}/{passes}  {label}{detail}")

    print(f"\n{len(cases) - failures}/{len(cases)} situations fully reliable")
    os.unlink(_tmp.name)


if __name__ == "__main__":
    asyncio.run(main())
