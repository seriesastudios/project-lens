import sqlite3
from typing import List, Dict, Any, Optional
from app.config import config

class DatabaseSession:
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enforce foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()

def init_db():
    """Initializes the SQLite schema."""
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        
        # Enable Full-Text Search (FTS5) - Optional, creating basic schema first
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

def add_node(content: str, status: str = 'active', target_date: Optional[str] = None) -> int:
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO nodes (content, status, target_date) VALUES (?, ?, ?)",
            (content, status, target_date)
        )
        return cursor.lastrowid

def add_edge(parent_id: int, child_id: int, relationship: str):
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO edges (parent_id, child_id, relationship) VALUES (?, ?, ?)",
            (parent_id, child_id, relationship)
        )

def update_node_status(node_id: int, status: str):
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE nodes SET status = ? WHERE id = ?",
            (status, node_id)
        )

def update_node_urgency(node_id: int, urgency_score: float):
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE nodes SET urgency_score = ? WHERE id = ?",
            (urgency_score, node_id)
        )

def get_active_nodes(limit: int = 7) -> List[Dict[str, Any]]:
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM nodes WHERE status = 'active' ORDER BY urgency_score DESC, created_at ASC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_edges_for_node(node_id: int) -> List[Dict[str, Any]]:
    with DatabaseSession() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM edges WHERE parent_id = ? OR child_id = ?",
            (node_id, node_id)
        )
        return [dict(row) for row in cursor.fetchall()]

if __name__ == "__main__":
    # Test execution
    init_db()
    node1_id = add_node("Test Node 1")
    node2_id = add_node("Test Node 2")
    add_edge(node1_id, node2_id, "blocks")
    print("Active nodes:", get_active_nodes())
    print("Edges for Node 1:", get_edges_for_node(node1_id))
