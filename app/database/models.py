import re
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.config import config

VALID_STATUSES = ('active', 'completed', 'on_hold', 'cold_storage')
VALID_RELATIONSHIPS = ('is_part_of', 'blocks', 'depends_on', 'related_to')
VALID_NODE_TYPES = ('task', 'project')
VALID_PRIORITIES = ('high', 'normal', 'low')


def utc_now_str() -> str:
    """UTC timestamp matching SQLite's CURRENT_TIMESTAMP format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class DatabaseSession:
    def __init__(self, db_path: Optional[str] = None):
        # Resolve at call time so tests can repoint config.DATABASE_PATH
        self.db_path = db_path or config.DATABASE_PATH

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()


# Columns added after v1; applied idempotently on startup so an existing
# lens.db migrates in place.
_MIGRATION_COLUMNS = [
    ("node_type", "TEXT DEFAULT 'task'"),
    ("focus_score", "REAL DEFAULT 0.0"),
    ("focused_at", "DATETIME NULL"),
    ("completed_at", "DATETIME NULL"),
    ("updated_at", "DATETIME NULL"),
    ("priority", "TEXT DEFAULT 'normal'"),
    ("description", "TEXT NULL"),  # longer detail shown when a card is expanded
]


def init_db():
    """Initializes the SQLite schema and applies migrations."""
    with DatabaseSession() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                status TEXT CHECK(status IN ('active', 'completed', 'on_hold', 'cold_storage')) DEFAULT 'active',
                urgency_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                target_date DATETIME NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                parent_id INTEGER,
                child_id INTEGER,
                relationship TEXT CHECK(relationship IN ('is_part_of', 'blocks', 'depends_on', 'related_to')),
                PRIMARY KEY (parent_id, child_id, relationship),
                FOREIGN KEY (parent_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (child_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history_digest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                digest_type TEXT CHECK(digest_type IN ('daily_summary', 'activity_log')),
                body TEXT NOT NULL
            )
        ''')

        existing = {row[1] for row in cursor.execute("PRAGMA table_info(nodes)")}
        for column, definition in _MIGRATION_COLUMNS:
            if column not in existing:
                cursor.execute(f"ALTER TABLE nodes ADD COLUMN {column} {definition}")

        # Full-text index over node content (porter stemming: "sent" matches "send").
        # Tasks live in the DB; the brain retrieves only what's relevant per message.
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                content, content='nodes', content_rowid='id', tokenize='porter unicode61')
        ''')
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_fts_insert AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        ''')
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_fts_delete AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content) VALUES ('delete', old.id, old.content);
            END
        ''')
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS nodes_fts_update AFTER UPDATE OF content ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content) VALUES ('delete', old.id, old.content);
                INSERT INTO nodes_fts(rowid, content) VALUES (new.id, new.content);
            END
        ''')
        # Sync rows that predate the index (cheap at this scale)
        cursor.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")

        # Semantic vectors live alongside the relational data — SQLite stays the
        # single source of truth; no separate vector database needed at this scale.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                vector BLOB NOT NULL
            )
        ''')

        # Small key-value store for app state, e.g. the current Lens view —
        # views are sticky across restarts, so they live in the DB.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')


def add_node(content: str, status: str = 'active', target_date: Optional[str] = None,
             node_type: str = 'task', priority: str = 'normal',
             description: Optional[str] = None) -> int:
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO nodes (content, status, target_date, node_type, priority, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, status, target_date, node_type, priority, description)
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def add_edge(parent_id: int, child_id: int, relationship: str):
    with DatabaseSession() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO edges (parent_id, child_id, relationship) VALUES (?, ?, ?)",
            (parent_id, child_id, relationship)
        )


def detach_parents(child_id: int, relationship: str = "is_part_of") -> int:
    """Remove all edges that file `child_id` under a parent (used when a task is
    promoted to a top-level project). Returns the number of edges deleted."""
    with DatabaseSession() as conn:
        cursor = conn.execute(
            "DELETE FROM edges WHERE child_id = ? AND relationship = ?",
            (child_id, relationship)
        )
        return cursor.rowcount


def get_node(node_id: int) -> Optional[Dict[str, Any]]:
    with DatabaseSession() as conn:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return dict(row) if row else None


def existing_node_ids(node_ids: List[int]) -> List[int]:
    """Returns the subset of node_ids that actually exist (guards against hallucinated IDs)."""
    if not node_ids:
        return []
    with DatabaseSession() as conn:
        placeholders = ",".join("?" * len(node_ids))
        rows = conn.execute(f"SELECT id FROM nodes WHERE id IN ({placeholders})", node_ids).fetchall()
        return [row["id"] for row in rows]


def find_node_by_content(content: str, node_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Exact-content lookup, used by the seeder to avoid duplicates."""
    query = "SELECT * FROM nodes WHERE content = ?"
    params: List[Any] = [content]
    if node_type:
        query += " AND node_type = ?"
        params.append(node_type)
    with DatabaseSession() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def update_node(node_id: int, content: Optional[str] = None, status: Optional[str] = None,
                target_date: Optional[str] = None, priority: Optional[str] = None,
                description: Optional[str] = None, node_type: Optional[str] = None) -> bool:
    """Partial update; stamps updated_at, and completed_at when completing. Returns False if node missing."""
    fields, params = [], []
    if content is not None:
        fields.append("content = ?")
        params.append(content)
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if node_type is not None:
        fields.append("node_type = ?")
        params.append(node_type)
    if priority is not None:
        fields.append("priority = ?")
        params.append(priority)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
        if status == 'completed':
            fields.append("completed_at = ?")
            params.append(utc_now_str())
            fields.append("focus_score = 0.0")
    if target_date is not None:
        fields.append("target_date = ?")
        params.append(target_date)
    if not fields:
        return get_node(node_id) is not None
    fields.append("updated_at = ?")
    params.append(utc_now_str())
    params.append(node_id)
    with DatabaseSession() as conn:
        cursor = conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE id = ?", params)
        return cursor.rowcount > 0


def complete_nodes(node_ids: List[int]) -> List[int]:
    """Marks nodes completed; returns the IDs that were actually updated."""
    valid = existing_node_ids(node_ids)
    now = utc_now_str()
    with DatabaseSession() as conn:
        for node_id in valid:
            conn.execute(
                "UPDATE nodes SET status = 'completed', completed_at = ?, updated_at = ?, focus_score = 0.0 WHERE id = ?",
                (now, now, node_id)
            )
    return valid


def get_state(key: str) -> Optional[str]:
    with DatabaseSession() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str):
    with DatabaseSession() as conn:
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value)
        )


def get_active_nodes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """All active nodes (lens qualification happens in scoring.py, not SQL)."""
    query = "SELECT * FROM nodes WHERE status = 'active' ORDER BY created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with DatabaseSession() as conn:
        return [dict(row) for row in conn.execute(query).fetchall()]


# Generic words that would make every task a "match" in an OR query
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "at", "is",
    "it", "its", "im", "ive", "just", "that", "this", "these", "those", "my",
    "me", "we", "our", "you", "your", "was", "were", "be", "been", "am", "are",
    "do", "did", "have", "has", "had", "with", "off", "up", "out", "by", "as",
    "so", "now", "then", "what", "whats", "when", "where", "which", "who",
    "how", "can", "could", "should", "would", "will", "dont", "didnt", "cant",
    "also", "please", "okay", "ok", "about", "all", "any", "some", "more",
    "really", "actually", "instead", "task", "tasks", "todo",
}


def search_active_nodes(text: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Full-text search of open (active or on-hold) nodes against free-form
    user text. Meaningful words are OR-ed so any overlap surfaces a candidate.
    On-hold items are included so shelved projects stay discoverable — the
    Lens itself only ever shows active nodes."""
    words = [w for w in re.findall(r"[A-Za-z0-9]{2,}", text or "")
             if w.lower() not in _STOPWORDS]
    if not words:
        return []
    query = " OR ".join(words)
    with DatabaseSession() as conn:
        rows = conn.execute(
            '''SELECT n.* FROM nodes n
               JOIN nodes_fts f ON n.id = f.rowid
               WHERE nodes_fts MATCH ? AND n.status IN ('active', 'on_hold')
               ORDER BY rank LIMIT ?''',
            (query, limit)
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_edges() -> List[Dict[str, Any]]:
    # Ordered by insertion (rowid) so "first is_part_of edge = primary project"
    # is deterministic for multi-home tasks.
    with DatabaseSession() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM edges ORDER BY rowid").fetchall()]


def get_active_child_ids(parent_id: int, relationship: str = "is_part_of") -> List[int]:
    """Active children of a node (e.g. a project's open tasks)."""
    with DatabaseSession() as conn:
        rows = conn.execute(
            '''SELECT n.id FROM nodes n
               JOIN edges e ON n.id = e.child_id
               WHERE e.parent_id = ? AND e.relationship = ? AND n.status = 'active' ''',
            (parent_id, relationship)
        ).fetchall()
        return [row["id"] for row in rows]


def get_edges_for_node(node_id: int) -> List[Dict[str, Any]]:
    with DatabaseSession() as conn:
        rows = conn.execute(
            "SELECT * FROM edges WHERE parent_id = ? OR child_id = ?",
            (node_id, node_id)
        ).fetchall()
        return [dict(row) for row in rows]


def add_digest(digest_type: str, body: str):
    with DatabaseSession() as conn:
        conn.execute(
            "INSERT INTO history_digest (digest_type, body) VALUES (?, ?)",
            (digest_type, body)
        )
