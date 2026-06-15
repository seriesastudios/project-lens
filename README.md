# Project Lens

**A local-first, AI-native task manager. Chat is the only input.**

You don't click buttons, fill forms, or drag cards. You talk to it — *"I need to renew my passport before September,"* *"focus on the website redesign,"* *"make Wayfinder its own project with a logline and a pitch deck"* — and a small language model running **entirely on your machine** turns that into structured changes to a task graph. A read-only pane called **the Lens** shows you what matters right now.

No cloud. No API keys. No subscription. Your tasks never leave your computer.

> _Screenshot coming — drop a PNG here._

---

## Why

Traditional task apps (Asana, Trello, Notion) make you do the filing: pick a project, set a priority, choose a due date, nest the subtask. That overhead is where to-do systems go to die. Project Lens flips it: **you describe, the model files.** Every entry is a node in a local graph, so the rigid "Folder > Project > Task > Subtask" hierarchy dissolves — a task can live under several projects at once, and you navigate into anything that has children.

The catch with chat-driven apps is usually latency and trust. Lens is built around both: it streams feedback in well under a second, it never lets the model invent task IDs, and it does all the date math and scoring in plain Python so a 4B model doesn't have to.

## Features

- **Natural-language everything** — capture, complete, reword, reschedule, reprioritize, group, all from chat.
- **The Lens, a navigable view** — a default **Today** view (only what's genuinely urgent, never padded), a **Projects** overview, and drill-down into *any* node that has children, with a clickable breadcrumb. View state is sticky across restarts.
- **A graph, not a tree** — tasks can belong to multiple projects (complete once, done everywhere); subtasks nest arbitrarily deep.
- **Restructure by voice** — *"make X its own project"* promotes a task; *"move X to Y"* relocates it between projects.
- **Streaming responses** — status, actions, and tokens stream as NDJSON; the Lens updates mid-turn over a WebSocket.
- **Stays fast as it grows** — a hybrid retrieval layer (SQLite FTS5 + local embeddings) puts only the conversation-relevant slice of your tasks in the prompt, so the context stays small whether you have 20 tasks or 2,000.
- **Grounded** — the model can only act on task IDs it was actually shown; guessed IDs are rejected before they touch your data.
- **Visual urgency** — card wash encodes deadline proximity, an edge stripe encodes priority.

## Requirements

- **Python 3.10+** (developed on 3.12).
- **A local, OpenAI-compatible LLM server** running two models:
  - a **chat model** with tool-calling — default `qwen/qwen3-4b-2507`;
  - an **embedding model** — default `text-embedding-nomic-embed-text-v1.5`.
- [LM Studio](https://lmstudio.ai/) is the easiest way to get both (load the two models, start its server on `http://127.0.0.1:1234/v1`). Any OpenAI-compatible server works — **Ollama**, **llama.cpp**'s server, etc. — just point `.env` at it.

The chat model needs to be reasonably good at tool calls. A 4B-class instruct model is the sweet spot of speed and reliability for this; smaller models start to slip.

## Setup

```bash
git clone https://github.com/seriesastudios/project-lens.git
cd project-lens

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # edit if your model names or server URL differ
```

`.env` controls the connection — base URL, chat model name, embedding model name, and the database path:

```
AI_BASE_URL="http://127.0.0.1:1234/v1"
AI_MODEL_NAME="qwen/qwen3-4b-2507"
EMBEDDING_MODEL_NAME="text-embedding-nomic-embed-text-v1.5"
DATABASE_PATH="lens.db"
```

## Run

1. Start your local LLM server (e.g. LM Studio) with the chat **and** embedding models loaded.
2. Start Lens:
   ```bash
   python -m app.main
   ```
3. Open **http://127.0.0.1:8000**.

The SQLite database is created automatically on first launch, and embeddings backfill at startup.

### Run as a desktop app (macOS)

To launch Lens as a single native window instead of a browser tab:

```bash
./lens
```

This opens a Project Lens window and returns your shell prompt immediately — the
app is detached, so it keeps running even if you close that terminal. Everything
is one process: **closing the window stops the server** (no stray port to kill).
To stop it from the command line: `pkill -f app.desktop`.

The LLM still runs separately in LM Studio — the desktop app does not bundle or
manage it. For debugging, run the window attached with `python -m app.desktop`.

### Load demo data (optional)

A fresh database is empty. To explore with a made-up set of projects and tasks:

```bash
python scripts/seed_demo.py            # seed an empty database
python scripts/seed_demo.py --reset    # wipe and reseed
```

This populates a handful of fictional projects (Launch personal blog, Plan camping trip, …) that exercise every part of the UI — Today, Projects, drill-down subtasks, a multi-home task, and the deadline/priority colours.

## Try saying…

- "I need to renew my passport before my trip in September"
- "I have to call the accountant and pick up the dry cleaning"
- "just sent off the invoices" *(completes the matching task)*
- "focus on the camping trip" *(opens that project in the Lens)*
- "make 'Set up hosting' its own project"
- "move 'Draft a budget spreadsheet' to Home refresh"
- "what should I be working on right now?"

## Development

```bash
pytest tests/ -q                  # unit tests (no LLM needed)
python scripts/eval_brain.py 3    # live behavioural eval — needs the LLM server up
```

`eval_brain.py` runs a suite of real situations through the model with actual tool execution, N times each, and reports how reliably the model picks the right action. It's the fastest way to see whether a model swap or a prompt change helped or hurt.

## How it works

- **Backend:** FastAPI + Uvicorn over a SQLite graph (`nodes` + typed `edges`). The model is given a small set of tools (`capture_tasks`, `complete_tasks`, `update_task`, `link_tasks`, `move_task`, `open_view`, `search_tasks`) and the runtime executes them against the database.
- **Brain** (`app/engine/brain.py`): a two-stage loop — decide-with-tools, then speak — plus prompt shaping that keeps a byte-stable prefix so `llama.cpp`/LM Studio's KV cache stays warm between turns. The model expresses *intent* (names, not IDs); Python resolves names, computes scores, and does all date math.
- **Views & scoring** (`app/engine/views.py`, `scoring.py`): the Lens is a view state machine; each mode has its own pure scoring function.
- **Retrieval** (`app/engine/retrieval.py`, `embeddings.py`): hybrid keyword + semantic search over locally-stored vectors, degrading gracefully to keyword-only if the embedding server is down.
- **Frontend** (`index.html`): a single page of vanilla JS + Tailwind (CDN), updated over a WebSocket. No build step.

## License

MIT — see [LICENSE](LICENSE).
