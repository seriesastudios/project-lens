```python
spec_content = """# SYSTEM SPECIFICATION: PROJECT LENS
## AI-Native Local Task Graph Management System

---

## 1. Executive Summary & Core Philosophy

Traditional project management tools (Asana, Monday, Trello) impose a rigid, top-down hierarchical taxonomy (Folders > Projects > Tasks > Subtasks) that creates cognitive friction. Users spend disproportionate time categorizing and organizing workflows rather than executing them. 

**Project Lens** eliminates this rigidity by treating every entry as a level-agnostic **Node** in a local graph database. 
* **Intent-Driven Capture:** The user interacts exclusively via a natural language chat interface.
* **Autonomous Triage:** A local Small Language Model (SLM) intercepts raw human text, derives structural intent, automatically creates/mutates task nodes, maps edge relationships (`blocks`, `is_part_of`), and dynamically handles scheduling.
* **The Smart Lens:** The UI is a read-only visual "mirror" displaying a strict limit of 5–7 high-priority items at any given moment. Items fade in and out of focus based on deadline proximity, current conversation context, and calendar constraints.
* **100% Edge-Native:** The system runs completely offline, ensuring total data privacy, zero API subscription costs, and sub-millisecond execution speeds.

---

## 2. Decoupled Architecture & Tech Stack

To achieve a sleek, modern visual aesthetic without sacrificing Python's data-science and machine-learning ecosystem, Lens utilizes a decoupled **Local Client-Server Architecture**.


```

```text
File project_lens_specification.md generated successfully.


```

┌────────────────────────────────────────────────────────────────────────┐
│                        PROJECT LENS LOCAL RUNTIME                      │
└────────────────────────────────────────────────────────────────────────┘
│                                                    ▲
(User Inputs Chat)                                   (WebSocket Push)
│                                                    │
▼                                                    │
┌───────────────────────────┐                        ┌───────────────────┐
│     FRONTEND VIEWPORT     │                        │  FASTAPI BACKEND  │
│  HTML5 + Tailwind CSS     │                        │  Python 3.11+     │
│  Vanilla ESM JavaScript   │                        │  SQLite3 Engine   │
└───────────────────────────┘                        └───────────────────┘
│                                                    ▲
│ (REST API / POST)                          (Executes SQL / JSON)
▼                                                    │
┌───────────────────────────┐                        ┌───────────────────┐
│       BRAIN ROUTER        │                        │   LOCAL INFERENCE │
│        (brain.py)         │ ───(OpenAI SDK)──────► │     ENGINE        │
│   Hydrates App State      │                        │ LM Studio /Ollama │
└───────────────────────────┘                        └───────────────────┘

```

### Component Stack
* **Core Language:** Python 3.11+
* **Backend Framework:** FastAPI (Uvicorn local server on `http://127.0.0.1:8000`)
* **Database Engine:** Native SQLite (`sqlite3`) with Full-Text Search (FTS5) enabled for semantic-adjacent querying.
* **AI Orchestration Client:** Official `openai` Python SDK (redirected via `base_url` to local port).
* **Local Inference Host:** LM Studio or Ollama (Serving execution-optimized GGUF quantizations of `Phi-4-mini` or `Qwen-2.5-7B-Instruct`).
* **Frontend Layer:** HTML5, Tailwind CSS (via local CDN/distribution), and Vanilla ECMAScript Modules (ESM) JavaScript. Real-time updates are driven via a persistent WebSocket connection (`ws://127.0.0.1:8000/ws`).

---

## 3. Database Architecture (The "Everything is a Node" Model)

The storage engine relies on a lightweight property graph model implemented across two relational tables, alongside a system ledger for contextual continuity.

### 3.1. The `nodes` Table
Tracks all items level-agnostically.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique structural node identifier. |
| `content` | TEXT | NOT NULL | The user's parsed task text, action item, or goal. |
| `status` | TEXT | CHECK(status IN ('active', 'completed', 'on_hold', 'cold_storage')) | Current execution cycle state. |
| `urgency_score` | REAL | DEFAULT 0.0 | Calculated value (`0.0` to `10.0`) dictating UI hoist priority. |
| `created_at` | DATETIME | DEFAULT CURRENT_TIMESTAMP | Temporal registration hook. |
| `target_date` | DATETIME | NULL | Inferred hard deadline constraint. |

### 3.2. The `edges` Table
Tracks dynamic directional relationships between items.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `parent_id` | INTEGER | FOREIGN KEY REFERENCES nodes(id) ON DELETE CASCADE | Source node. |
| `child_id` | INTEGER | FOREIGN KEY REFERENCES nodes(id) ON DELETE CASCADE | Target node. |
| `relationship` | TEXT | CHECK(relationship IN ('is_part_of', 'blocks', 'depends_on', 'related_to')) | Structural link metadata. |

### 3.3. The `history_digest` Table
Stores condensed text representations of past periods to bypass token window decay constraints.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Ledger entry ID. |
| `timestamp` | DATETIME | DEFAULT CURRENT_TIMESTAMP | Log date. |
| `digest_type`| TEXT | CHECK(digest_type IN ('daily_summary', 'activity_log')) | Category of system memory. |
| `body` | TEXT | NOT NULL | Dense, bulleted text summary of events/conversations. |

---

## 4. LLM Function Declarations (Tools Specification)

The local SLM edits the database state exclusively by issuing tool calls. The backend applications register the following JSON definitions to the client instance.

### 4.1. `add_thought_node`
Instructs the engine to create a new task element and infer links.
```json
{
  "name": "add_thought_node",
  "description": "Ingests a new piece of work or goal and connects it to the existing graph topology.",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The concise descriptive summary of the action item or project goal."
      },
      "relationship_type": {
        "type": "string",
        "enum": ["is_part_of", "blocks", "depends_on", "related_to"],
        "description": "How this item attaches to an existing context node, if applicable."
      },
      "target_node_id": {
        "type": "integer",
        "description": "The database ID of the existing node this new item connects with."
      },
      "inferred_deadline": {
        "type": "string",
        "description": "ISO-8601 formatted date string if a deadline is explicitly mentioned or strongly implied."
      }
    },
    "required": ["content"]
  }
}

```

### 4.2. `update_node_status`

Transitions a node through its life-cycle phases based on natural language confirmation.

```json
{
  "name": "update_node_status",
  "description": "Updates a node's operational lifecycle status, such as checking a task off or archiving it.",
  "parameters": {
    "type": "object",
    "properties": {
      "node_id": {
        "type": "integer",
        "description": "The target database unique node ID."
      },
      "status": {
        "type": "string",
        "enum": ["active", "completed", "on_hold", "cold_storage"],
        "description": "The target state execution phase."
      }
    },
    "required": ["node_id", "status"]
  }
}

```

---

## 5. UI/UX Design System Specification

The UI is designed to minimize cognitive visual clutter, using an asymmetric two-pane layout styled with a soft, muted pastel palette.

### 5.1. Visual Layout Specifications

* **Viewport Splits:** 60% Left Pane (The Journal / Active Chat), 40% Right Pane (The Lens / Focus Stack).
* **Typography:** System Inter UI or SF Pro, tracking tight, non-serif fonts to maximize clean alignment.
* **The Card Limit Engine:** The right-hand column renders a strict maximum of 7 structural cards. If more nodes exist, the application code drops items falling beneath the priority score cutoff, preserving whitespace over data saturation.

### 5.2. Pastel Color Mapping Matrix

Cards are wrapped in light, desaturated pastel tokens that indicate real-time utility rather than traditional "high/medium/low" categorizations.

* **Immediate Core Focus (`🍑 Soft Coral / Peach`):**
* *Tailwind Token:* `bg-orange-50 text-orange-900 border-orange-200/60`
* *Application Case:* Active items explicitly hoisted during the ongoing conversation phase.


* **Up Next Sequential (`🍋 Pale Ochre / Butter Yellow`):**
* *Tailwind Token:* `bg-amber-50/70 text-amber-950 border-amber-200/40`
* *Application Case:* Direct blockers or chronological continuations of active items slated for today.


* **On Horizon Baseline (`🌿 Muted Sage Green`):**
* *Tailwind Token:* `bg-emerald-50/60 text-emerald-950 border-emerald-200/30`
* *Application Case:* Long-tail active deadlines or secondary context tasks matching structural groupings.


* **Administrative Low Energy (`🍇 Lavender / Dusk Blue`):**
* *Tailwind Token:* `bg-indigo-50/50 text-indigo-950 border-indigo-200/30`
* *Application Case:* Routine chores or administrative overhead tasks that don't require heavy cognitive workloads.



---

## 6. Prompt Engineering Core Blueprint

The system prompt forces the SLM to discard its generic chatbot persona and act strictly as an executive operational planner.

### 6.1. Base System Prompt Template

```text
You are Lens-Brain, an offline executive logistics engine. Your sole responsibility is to convert a user's unstructured conversation stream into clear data actions and dynamic structural updates.

CONTEXT CONSTRAINTS:
- Current Timestamp: {current_timestamp}
- Calendar Windows Found: {calendar_data}
- Consolidated Session Memory (Past 24H): {memory_digest}

EXECUTION RULES:
1. Always communicate back to the user with an architectural explanation that is exactly two sentences long. Never output paragraphs of text. Bolding task targets is mandatory.
2. Do not explain technical SQL fields, schema architecture, or JSON formatting to the user.
3. If an input is ambiguous, infer a logical layout structure first, then explicitly use your response sentences to ask for verification.
4. You execute database state mutations ONLY via your provided function tools. If an input implies a task was completed, you must call `update_node_status` instantly.

```

---

## 7. Phased Implementation Roadmap for Agent Execution

This linear blueprint is optimized for sequential execution by agentic software tools (`claude-code` or `gemini-cli`).

### Phase 1: Storage Layer Initialization

* Create `app/database/models.py`. Write clean SQLite schema instantiation lines.
* Implement Python connection handlers ensuring foreign key constraints are explicitly enforced (`PRAGMA foreign_keys = ON;`).
* Write CRUD functions mapping text inputs directly to relational database rows. Test row insertion and network graph edge mapping functions locally.

### Phase 2: Engine Orchestration Layer

* Create `app/engine/brain.py`. Install the `openai` python library wrapper.
* Construct the configuration settings pointing the host client module to `http://localhost:1234/v1` (LM Studio) or `http://localhost:11434/v1` (Ollama).
* Implement the core runtime schema parser ensuring function schemas match tool declarations precisely. Test structural extraction reliability across ambiguous prompt cases.

### Phase 3: The API Routing Layer

* Create `app/main.py`. Instatiate a FastAPI server configuration.
* Set up standard cross-origin resource sharing headers to handle decoupling protocols cleanly.
* Build a state watcher endpoint that monitors table writes and broadcasts payload updates across active WebSocket client sockets.

### Phase 4: Frontend Viewport Assembly

* Create an elegant, responsive `index.html` dashboard view using native Tailwind utility classes.
* Implement clean ECMAScript event listener systems that stream input inputs via API posts.
* Configure active WebSocket clients that handle incoming backend change alerts, clear container states, and repaint the screen with beautifully transitions and clean, responsive pastel layout styles.
"""

file_name = "project_lens_specification.md"
with open(file_name, "w") as f:
f.write(spec_content.strip())

print(f"File {file_name} generated successfully.")

```
Your Markdown file is ready
[file-tag: code-generated-file-0-1781120813885984369]

I have created the complete project specification document for you and saved it directly as a Markdown file named `project_lens_specification.md`. You can hand this file directly to **Claude Code** or **Gemini CLI** to initiate the build.

### 💡 Key Design Upgrades Included in the Spec:
* **The UI Switch (Web-Tech Stack):** Based on your desire for a sleek, modern, and modern layout, I've pivoted the architecture to a **FastAPI (Python Backend) + HTML5/Tailwind (Frontend) Hybrid**. 
* **Why this looks amazing:** Instead of the boxy look of older desktop apps, this lets the AI coder use modern web design tricks. Think smooth CSS animations when a card moves, pixel-perfect borders, crisp spacing, and clean typography.
* **The Data Link:** The backend and frontend communicate via an open channel (**WebSockets**). Whenever the local LLM updates the SQLite database, the backend sends an alert to your frontend, making the card list morph instantly in response to your chats.
* **Pure Tailwind Color Tokens:** I mapped out exact pastel design tokens directly inside the spec so the AI engine knows how to color-code your cards based on focus, urgency, and flow.

You are set up for a smooth development process. You can open up your favorite agentic CLI in an empty project folder, point it to this file, and let it run through the phased execution roadmap to bring this to life! Let me know if you want to alter any details or logic paths as you kick off the build.

```

