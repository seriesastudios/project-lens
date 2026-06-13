"""Populate Project Lens with a small, made-up set of projects and tasks so a
fresh clone has something to explore — no LLM and no external files involved,
just direct writes to the local SQLite graph.

    python scripts/seed_demo.py            # seed an empty database
    python scripts/seed_demo.py --reset    # wipe existing tasks first, then seed

The data is entirely fictional. It's shaped to show off every part of the UI:
a default "Today" view, a "Projects" overview, drill-down into a container
task's subtasks, a task that lives under two projects at once (the graph, not a
tree), and the deadline/priority colour coding.
"""
import os
import sys
from datetime import date, timedelta

# Allow running as `python scripts/seed_demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import models  # noqa: E402
from app.engine import embeddings  # noqa: E402

TODAY = date.today()


def in_days(n: int) -> str:
    """A YYYY-MM-DD deadline n days from today, so the demo always looks fresh."""
    return (TODAY + timedelta(days=n)).isoformat()


def project(content, priority="normal", deadline=None):
    return models.add_node(content, node_type="project", priority=priority,
                           target_date=deadline)


def task(content, priority="normal", deadline=None, description=None):
    return models.add_node(content, node_type="task", priority=priority,
                           target_date=deadline, description=description)


def under(parent_id, child_id):
    models.add_edge(parent_id, child_id, "is_part_of")


def reset():
    with models.DatabaseSession() as conn:
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM history_digest")
        conn.execute("DELETE FROM app_state")


def seed():
    # 1. Launch personal blog — a high-priority project with a container task
    #    ("Set up hosting") whose steps are real subtasks you can drill into.
    blog = project("Launch personal blog", priority="high")
    hosting = task("Set up hosting",
                   description="Get the site online before writing anything.")
    under(blog, hosting)
    under(hosting, task("Buy a domain name"))
    under(hosting, task("Configure DNS records"))
    under(hosting, task("Deploy the starter site"))
    under(blog, task("Write the first post", deadline=in_days(5)))
    under(blog, task("Pick a theme", priority="low"))

    # 2. Plan camping trip — has the most urgent item in the whole demo.
    camping = project("Plan camping trip")
    under(camping, task("Book the campsite", priority="high", deadline=in_days(2),
                        description="Sites for the long weekend are going fast."))
    under(camping, task("Rent gear", deadline=in_days(6)))
    under(camping, task("Plan the meals", priority="low"))

    # 3. Learn Spanish — a steady, low-pressure project.
    spanish = project("Learn Spanish")
    under(spanish, task("Finish Duolingo unit 3"))
    under(spanish, task("Schedule a weekly tutor session", deadline=in_days(4)))
    under(spanish, task("Watch a movie in Spanish", priority="low"))

    # 4. Home refresh — low priority, longer horizon.
    home = project("Home refresh", priority="low")
    under(home, task("Get three paint quotes", deadline=in_days(10)))
    under(home, task("Declutter the garage"))

    # A task that belongs to TWO projects at once — the graph, not a tree.
    budget = task("Draft a budget spreadsheet",
                  description="One sheet covering both the trip and the home work.")
    under(camping, budget)
    under(home, budget)

    # Loose tasks with no project — they still surface in Today on their merits.
    task("Renew passport", deadline=in_days(20))
    task("Call the dentist")


def main():
    do_reset = "--reset" in sys.argv
    models.init_db()

    existing = models.get_active_nodes()
    if existing and not do_reset:
        print(f"Database already has {len(existing)} active task(s). "
              "Re-run with --reset to wipe and reseed the demo.")
        return

    if do_reset:
        reset()

    seed()
    indexed = embeddings.backfill()
    nodes = models.get_active_nodes()
    projects = sum(1 for n in nodes if n.get("node_type") == "project")
    print(f"Seeded demo data: {projects} projects, {len(nodes) - projects} tasks.")
    if indexed:
        print(f"Indexed {indexed} nodes for semantic search.")
    else:
        print("No embedding server reached — semantic search will fall back to "
              "keyword search until you index (the app backfills on startup).")
    print("Start the app with `python -m app.main` and open http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
