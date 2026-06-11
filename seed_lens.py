import os
import glob
import re
import yaml
from pathlib import Path

# Adjust the python path to include the current project
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import models

SECOND_BRAIN_DIR = os.path.expanduser("~/My Drive/Second-Brain")

def parse_markdown_tasks(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, []

    # Extract YAML frontmatter
    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    project_name = "Unknown Project"
    if yaml_match:
        try:
            frontmatter = yaml.safe_load(yaml_match.group(1))
            if frontmatter and isinstance(frontmatter, dict):
                status = frontmatter.get('status', '').lower()
                if status != 'active':
                    return None, []
                project_name = frontmatter.get('project') or frontmatter.get('area') or "Unknown Project"
        except yaml.YAMLError:
            pass

    # Extract uncompleted tasks
    # Match lines like "- [ ] Task description"
    tasks = []
    for line in content.split("\n"):
        match = re.match(r"^\s*-\s*\[ \]\s*(.+)", line)
        if match:
            task_text = match.group(1).strip()
            tasks.append(task_text)

    return project_name, tasks

def seed_database():
    models.init_db()
    print(f"Scanning for TASKS.md files in {SECOND_BRAIN_DIR}...")
    search_pattern = os.path.join(SECOND_BRAIN_DIR, "**", "*TASKS.md")
    task_files = glob.glob(search_pattern, recursive=True)
    
    if not task_files:
        print("No TASKS.md files found in the Second Brain.")
        return

    # Track project nodes to avoid duplicates
    project_nodes = {}
    total_tasks = 0

    for filepath in task_files:
        project_name, tasks = parse_markdown_tasks(filepath)
        
        if not tasks or not project_name:
            continue
            
        print(f"Processing '{project_name}': found {len(tasks)} tasks.")
        
        # Get or create project node (idempotent: re-running the seeder must not duplicate)
        if project_name not in project_nodes:
            existing = models.find_node_by_content(project_name, node_type="project")
            if existing:
                project_nodes[project_name] = existing["id"]
            else:
                project_nodes[project_name] = models.add_node(
                    content=project_name, status="active", node_type="project"
                )

        proj_id = project_nodes[project_name]

        for task in tasks:
            if models.find_node_by_content(task):
                continue
            task_id = models.add_node(content=task, status="active")
            models.add_edge(parent_id=proj_id, child_id=task_id, relationship="is_part_of")
            total_tasks += 1

    print(f"\nSeeding complete! Added {len(project_nodes)} projects and {total_tasks} tasks to Lens.")

if __name__ == "__main__":
    seed_database()
