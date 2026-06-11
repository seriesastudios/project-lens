"""Seeds Lens from Second Brain TASKS.md files.

Import policy (agreed 2026-06-11):
- Project tiers: frontmatter status 'active' imports normally; 'on-deck' and
  'waiting' import with everything on_hold (in the DB, findable via search,
  excluded from the Lens until activated in chat). Paused/complete/archived
  are skipped.
- Section headers set task priority: "This Week"/🔥-style sections → high,
  "On Deck"/paused-style sections → low, "⏸️ Waiting On" sections → on_hold,
  anything else inherits the project's frontmatter priority (1=high, 3+=low).
- Each raw task line is rewritten by the local LLM into a natural to-do
  (markdown/wiki-links/provenance stripped; regex fallback), with an explicit
  date in the line extracted as the deadline.
- The project node carries the earliest upcoming frontmatter deadline.
- Tasks whose dates have already passed import as-is and are listed in a
  review report at the end — confirm or kill them in chat.

Usage:
    ./venv/bin/python seed_lens.py            # add new tasks (dedupes)
    ./venv/bin/python seed_lens.py --reset    # wipe tasks first, then seed
"""
import glob
import json
import os
import re
import sys
from datetime import date

import yaml
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import config
from app.database import models
from app.engine import embeddings

SECOND_BRAIN_DIR = os.path.expanduser("~/My Drive/Second-Brain")

# Frontmatter project status → node status (anything else: skip the file)
STATUS_MAP = {
    "active": "active",
    "on-deck": "on_hold",
    "waiting": "on_hold",
}

client = OpenAI(base_url=config.AI_BASE_URL, api_key="not-needed-for-local")

HUMANIZE_PROMPT = """You clean up terse task notes. Rewrite the task note below as ONE short, natural to-do phrase a person would actually say (imperative, under 15 words where possible). Keep essential specifics: names, places, amounts, dates. Drop markdown symbols, file paths, wiki-links, emoji, and provenance notes like "(per email 2026-06-09)".

Return ONLY a JSON object, no other text:
{"task": "<the rewritten to-do>", "deadline": "<YYYY-MM-DD if the note contains an explicit calendar date, else null>"}

The year is 2026 when a date has no year. Only set deadline for real dates ("Jun 12"), never vague timing ("mid-July", "this week")."""


def classify_section(header: str) -> str:
    """Maps a markdown section header to 'high' | 'low' | 'waiting' | 'inherit'."""
    text = (header or "").lower()
    if "waiting on" in text:
        return "waiting"
    if any(k in text for k in ("this week", "next action", "current priority", "next week")):
        return "high"
    if "🔥" in (header or "") and "status" not in text:
        return "high"
    if any(k in text for k in ("on deck", "paused", "deprioritized", "when resumed",
                               "someday", "later", "backlog")):
        return "low"
    return "inherit"


def map_priority(value) -> str:
    """Second Brain frontmatter uses numeric priority (1 = top). Map to Lens levels."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "normal"
    if number <= 1:
        return "high"
    if number >= 3:
        return "low"
    return "normal"


def earliest_upcoming_deadline(deadlines) -> str | None:
    """Frontmatter deadlines look like '2026-10-08 OAC application — ...'."""
    today = date.today().isoformat()
    dates = []
    for entry in deadlines or []:
        match = re.match(r"(\d{4}-\d{2}-\d{2})", str(entry).strip())
        if match and match.group(1) >= today:
            dates.append(match.group(1))
    return min(dates) if dates else None


def regex_clean(text: str) -> str:
    """Deterministic fallback cleanup when the LLM is unavailable or returns garbage."""
    text = re.sub(r"\*\((?:from )?\[\[[^\]]*\]\][^)]*\)\*", "", text)   # *(from [[...]])*
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]+)\]\]", r"\2", text)          # [[X|Y]] -> Y
    text = re.sub(r"`[^`]*`", "", text)                                   # backtick paths
    text = text.replace("**", "")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" —-–")


def humanize(raw_task: str, project: str) -> dict:
    """Returns {"task": str, "deadline": str|None}; falls back to regex_clean."""
    fallback = {"task": regex_clean(raw_task), "deadline": None}
    try:
        response = client.chat.completions.create(
            model=config.AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": HUMANIZE_PROMPT},
                {"role": "user", "content": f"Project: {project}\nTask note: {raw_task}"},
            ],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        task = (parsed.get("task") or "").strip()
        if not task:
            return fallback
        deadline = parsed.get("deadline")
        if deadline and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(deadline)):
            deadline = None
        return {"task": task, "deadline": deadline}
    except Exception:
        return fallback


def parse_markdown_tasks(filepath):
    """Returns (project_name, project_status, project_priority, project_deadline,
    [{'raw': str, 'section': 'high'|'low'|'waiting'|'inherit'}]) — or (None, ...) to skip."""
    skip = (None, None, "normal", None, [])
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return skip

    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not yaml_match:
        return skip
    try:
        frontmatter = yaml.safe_load(yaml_match.group(1)) or {}
    except yaml.YAMLError:
        return skip
    if not isinstance(frontmatter, dict):
        return skip

    project_status = STATUS_MAP.get(str(frontmatter.get('status', '')).lower())
    project_name = frontmatter.get('project') or frontmatter.get('area')
    if not project_status or not project_name:
        return skip

    priority = map_priority(frontmatter.get('priority'))
    deadline = earliest_upcoming_deadline(frontmatter.get('deadlines'))

    items = []
    section = "inherit"
    for line in content.split("\n"):
        header = re.match(r"^#{2,3}\s+(.*)", line)
        if header:
            section = classify_section(header.group(1))
            continue
        match = re.match(r"^\s*-\s*\[ \]\s*(.+)", line)
        if match:
            items.append({"raw": match.group(1).strip(), "section": section})
    return project_name, project_status, priority, deadline, items


def reset_tasks():
    with models.DatabaseSession() as conn:
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
    print("Cleared all existing tasks and edges.")


def seed_database(reset: bool = False):
    models.init_db()
    if reset:
        reset_tasks()

    print(f"Scanning for TASKS.md files in {SECOND_BRAIN_DIR}...")
    task_files = glob.glob(os.path.join(SECOND_BRAIN_DIR, "**", "*TASKS.md"), recursive=True)
    if not task_files:
        print("No TASKS.md files found in the Second Brain.")
        return

    project_nodes = {}
    total_tasks = 0
    overdue = []
    today = date.today().isoformat()

    for filepath in sorted(task_files):
        project_name, project_status, project_priority, project_deadline, items = \
            parse_markdown_tasks(filepath)
        if not project_name or not project_status or not items:
            continue
        hold = " [ON HOLD]" if project_status == "on_hold" else ""
        print(f"\n{project_name}{hold} — {len(items)} open tasks (priority: {project_priority})")

        if project_name not in project_nodes:
            existing = models.find_node_by_content(project_name, node_type="project")
            if existing:
                project_nodes[project_name] = existing["id"]
            else:
                project_nodes[project_name] = models.add_node(
                    content=project_name, status=project_status, node_type="project",
                    priority=project_priority, target_date=project_deadline)
        proj_id = project_nodes[project_name]

        for item in items:
            result = humanize(item["raw"], project_name)
            task_text, deadline = result["task"], result["deadline"]
            if models.find_node_by_content(task_text):
                print(f"  = (exists) {task_text}")
                continue

            section = item["section"]
            status = project_status
            priority = project_priority
            if section == "waiting":
                status = "on_hold"
            elif section == "high":
                priority = "high"
            elif section == "low":
                priority = "low"

            task_id = models.add_node(content=task_text, status=status,
                                      target_date=deadline, priority=priority)
            models.add_edge(parent_id=proj_id, child_id=task_id, relationship="is_part_of")
            total_tasks += 1
            if deadline and deadline < today and status == "active":
                overdue.append((project_name, task_text, deadline))

            tags = []
            if deadline:
                tags.append(f"due {deadline}")
            if priority != "normal":
                tags.append(priority)
            if status == "on_hold":
                tags.append("on hold")
            suffix = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  + {task_text}{suffix}")

    print(f"\nSeeding complete: {len(project_nodes)} projects, {total_tasks} tasks.")
    indexed = embeddings.backfill()
    print(f"Embeddings: indexed {indexed} nodes.")

    if overdue:
        print("\n" + "=" * 60)
        print("REVIEW: these imported with dates already in the past.")
        print("Tell the app in chat which are done or dead:")
        for project_name, task_text, deadline in overdue:
            print(f"  ! [{project_name}] {task_text} (was due {deadline})")


if __name__ == "__main__":
    seed_database(reset="--reset" in sys.argv)
