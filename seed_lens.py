"""Seeds Lens from Second Brain TASKS.md files (active projects only).

"Smart" loading: each raw task line is stripped of Second Brain notation
(markdown bold, backtick file paths, [[wiki-links]], "(per email ...)"
provenance) and rewritten by the local LLM into a short, natural to-do —
with an explicit date in the line extracted as the deadline.

Usage:
    ./venv/bin/python seed_lens.py            # add new tasks (dedupes)
    ./venv/bin/python seed_lens.py --reset    # wipe tasks first, then seed
"""
import glob
import json
import os
import re
import sys

import yaml
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import config
from app.database import models
from app.engine import embeddings

SECOND_BRAIN_DIR = os.path.expanduser("~/My Drive/Second-Brain")

client = OpenAI(base_url=config.AI_BASE_URL, api_key="not-needed-for-local")

HUMANIZE_PROMPT = """You clean up terse task notes. Rewrite the task note below as ONE short, natural to-do phrase a person would actually say (imperative, under 15 words where possible). Keep essential specifics: names, places, amounts, dates. Drop markdown symbols, file paths, wiki-links, emoji, and provenance notes like "(per email 2026-06-09)".

Return ONLY a JSON object, no other text:
{"task": "<the rewritten to-do>", "deadline": "<YYYY-MM-DD if the note contains an explicit calendar date, else null>"}

The year is 2026 when a date has no year. Only set deadline for real dates ("Jun 12"), never vague timing ("mid-July", "this week")."""


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


def parse_markdown_tasks(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, [], "normal"

    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    project_name = None
    priority = "normal"
    if yaml_match:
        try:
            frontmatter = yaml.safe_load(yaml_match.group(1))
            if frontmatter and isinstance(frontmatter, dict):
                if str(frontmatter.get('status', '')).lower() != 'active':
                    return None, [], "normal"
                project_name = frontmatter.get('project') or frontmatter.get('area')
                priority = map_priority(frontmatter.get('priority'))
        except yaml.YAMLError:
            pass
    if not project_name:
        return None, [], "normal"

    tasks = []
    for line in content.split("\n"):
        match = re.match(r"^\s*-\s*\[ \]\s*(.+)", line)
        if match:
            tasks.append(match.group(1).strip())
    return project_name, tasks, priority


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

    for filepath in task_files:
        project_name, raw_tasks, priority = parse_markdown_tasks(filepath)
        if not raw_tasks or not project_name:
            continue
        print(f"\n{project_name} — {len(raw_tasks)} open tasks (priority: {priority})")

        if project_name not in project_nodes:
            existing = models.find_node_by_content(project_name, node_type="project")
            project_nodes[project_name] = (existing["id"] if existing else
                models.add_node(content=project_name, status="active", node_type="project",
                                priority=priority))
        proj_id = project_nodes[project_name]

        for raw in raw_tasks:
            result = humanize(raw, project_name)
            task_text, deadline = result["task"], result["deadline"]
            if models.find_node_by_content(task_text):
                print(f"  = (exists) {task_text}")
                continue
            task_id = models.add_node(content=task_text, status="active", target_date=deadline,
                                      priority=priority)
            models.add_edge(parent_id=proj_id, child_id=task_id, relationship="is_part_of")
            total_tasks += 1
            due = f"  [due {deadline}]" if deadline else ""
            print(f"  + {task_text}{due}")

    print(f"\nSeeding complete: {len(project_nodes)} projects, {total_tasks} tasks.")
    indexed = embeddings.backfill()
    print(f"Embeddings: indexed {indexed} nodes.")


if __name__ == "__main__":
    seed_database(reset="--reset" in sys.argv)
