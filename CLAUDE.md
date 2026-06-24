# Project Lens — contributor & agent guide

Local-first, AI-native task manager. **Chat is the primary, NLP input** — a small
local LLM turns natural language into validated tool calls that mutate a SQLite
task graph. A handful of **deterministic, no-LLM UI actions** complement it (the
card complete checkbox, click-navigation, back/forward, and quick-add a task via
the **+** button / **⌘N**); these are thin endpoints that mutate the graph and
push state, never going through the model. (Quick-add is the one hybrid: it
creates the task instantly, then does a *best-effort async LLM enrich* — clean
title + due date + priority from phrases like "send proposal tomorrow" — and the
card updates over the WebSocket a beat later, degrading to the raw text when the
model is offline.) The **Lens** pane renders cards. No cloud, no accounts.

This file is for anyone (human or AI) modifying the code. End users should start
with `README.md`. Read this before changing the engine — several design rules are
load-bearing and non-obvious.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # FastAPI, uvicorn, openai, dateparser, …
python -m app.main                       # → http://127.0.0.1:8000
```

Needs **LM Studio** (or any OpenAI-compatible server) at `AI_BASE_URL` with a chat
model and an embedding model loaded. Defaults live in `app/config.py` and are
overridable via `.env` (`AI_BASE_URL`, `AI_MODEL_NAME`, `EMBEDDING_MODEL_NAME`,
`DATABASE_PATH`). The DB self-creates on first startup (`models.init_db()`); no
seed step required. `python scripts/seed_demo.py --reset` loads fake demo data.

Kill a stuck server: `lsof -ti :8000 | xargs kill`.

**Desktop app.** `./lens` launches a native macOS window (pywebview) detached
from the terminal; `app/desktop.py` runs uvicorn on a *dynamic free port* in a
daemon thread and opens the window — one process, so closing the window stops the
server. The LLM stays external (LM Studio); the desktop shell never manages it.
`python -m app.desktop` runs it attached for debugging. Stop a detached instance
with `pkill -f app.desktop`.

## Tests & eval — run BOTH before committing engine changes

```bash
./venv/bin/python -m pytest tests/ -q          # ~130 deterministic unit tests
./venv/bin/python scripts/eval_brain.py 3      # behavioral eval, 3 rounds, needs LM Studio
```

- **pytest** covers models, scoring, views, tool execution, and the streaming
  loop. The LLM is mocked there — these are fast and deterministic.
- **eval_brain.py** drives the *real* local model through ~27 scenarios and
  reports how many are reliable across N runs. It catches prompt regressions the
  unit tests can't. Target is **27/27 across 3 runs**.

## Architecture

```
app/main.py        FastAPI: /api/chat (NDJSON stream), /ws (state push), deterministic
                   actions (/api/tasks add, /api/nodes/{id}/complete, /api/view, /api/nav),
                   static index.html
app/config.py      env-driven config
app/database/models.py   SQLite schema + all DB primitives (the ONLY module that writes SQL)
app/engine/brain.py      LLM tool-calling loop, tool schemas, Pydantic validation, prompt
app/engine/scoring.py    pure functions: which cards each view shows + their order
app/engine/views.py      view state machine; turns the sticky view into card lists
app/engine/retrieval.py  FTS5 + embedding search over active nodes
app/engine/embeddings.py local embedding calls + backfill
index.html         single-file vanilla-JS + Tailwind-CDN frontend (no build step)
scripts/           eval_brain.py (behavioral eval), seed_demo.py (fake data)
```

### Data model (graph in SQLite)

- **nodes**: `id, content, status (active/completed/on_hold/cold_storage),
  target_date, node_type (task|project), priority (high|normal|low), description,
  completed_at, created_at, updated_at`.
- **edges**: `parent_id, child_id, relationship (is_part_of|blocks|depends_on|
  related_to)`. A task can have multiple `is_part_of` parents (multi-home).
- **app_state**: the sticky current view (JSON). **nodes_fts**: FTS5 index.
  **node_embeddings**: vectors for semantic search.

A node is a **container** iff it has active `is_part_of` children → the UI makes
it enterable (drill-down). A leaf with a `description` expands inline instead.

### The chat turn (`brain.py` → `process_user_input_events`)

Two-stage, streamed as NDJSON (`status` → `action`/`token` → `lens` → `done`):

1. **Decide & act.** The model is offered the 7 tools (`capture_tasks`,
   `complete_tasks`, `update_task`, `link_tasks`, `move_task`, `open_view`,
   `search_tasks`). The decision prompt carries **no style rules** so a small
   model doesn't skip the tool call and just chat. Every tool's args go through a
   **Pydantic** model; node IDs are checked against the DB (`enforce_grounding`)
   because small models hallucinate IDs.
2. **Speak.** Routine **mutations** are confirmed by a **templated** sentence
   built in Python (`template_confirmation`) — no second LLM call, ~halves
   latency. **Navigation/query** turns (`open_view`) instead defer to a second
   "speak" call (`NAV_SPEAK_REMINDER`) that gets an `on_screen` snapshot of the
   cards and answers in one warm sentence. Errors/unknown-IDs also fall back to
   the LLM.

The **Lens updates before the speak call**: the loop yields a `lens` event the
moment a tool runs, and `chat_endpoint` pushes new cards over the WebSocket
before the generator resumes into the speak call. Cards land instantly; the
sentence follows.

Other key mechanisms: `salvage_tool_call` parses tool calls the model wrote as
prose (gated by the `TOOL_NAME_LEAK` regex — **add any new tool name to it**);
volatile context rides at the END of the newest user message to keep the
llama.cpp/GGUF prefix cache warm (`format_context_prompt`).

### Views (`views.py`)

The sticky view in `app_state` is a state machine. Modes: `today`, `projects`,
`node` (drill-down with a breadcrumb `path`), `list` (ad-hoc id set), `loose`
(no project), `filter` (`overdue|high|waiting|done`). `compute_view_cards()`
reads the graph + current view and returns `{view: meta, cards: [...]}`.

### Scoring (`scoring.py`) — read this before touching ranking

Deliberately simple after a simplification pass. Importance never changes the
displayed **order** (always deadline-first); it only affects what *qualifies* and
ties. Three pieces:

- **`_order_key`** — THE display order, used by Today *and* every entered view:
  soonest/overdue deadline first, undated last, priority as the tiebreak only. A
  blocker borrows the deadline of what it gates so it sorts beside it.
- **`_qualifies_today`** — a boolean: does a node earn a Today slot? Due within
  `DEADLINE_WINDOW_DAYS`, OR high priority, OR captured in the last
  `RECENT_CAPTURE_HOURS`. Low priority needs a deadline within `IMMINENT_DAYS`.
- **`_slot_key`** — used ONLY when more than `MAX_LENS_CARDS` (7) qualify, to
  pick survivors. A small integer score (urgency tier 0–3 + importance 0–2). It
  **must** weigh urgency against importance: a pure priority-first cap once
  dropped a task due *tomorrow* in favor of undated high-priority work. Don't
  reintroduce that.

Entered views (project detail, lists, loose, filters, projects overview) show
*everything*, no cap, ordered by `_order_key`. Card categories (`next`/`soon`/
`horizon`/`undated`) encode urgency for the wash color; priority is a separate
edge stripe.

## Conventions & gotchas

- **All SQL lives in `models.py`.** Other modules call its functions; they never
  open the DB directly (except deliberate test fixtures).
- **The model names intent; Python resolves it.** The LLM passes task/project
  *names* and filter *names*; Python resolves them to IDs and computes membership
  and order. Never let the model pick scores, counts, or which tasks are in a
  view — that's how grounding stays intact.
- **The prompt is length-sensitive — this is the #1 recurring lesson.** The chat
  model is a 4B. Adding verbose rules has repeatedly *regressed core
  tool-calling* (e.g. a 5-bullet filter section broke basic "capture two tasks").
  Keep prompt additions terse; prefer a targeted `next`-hint returned from a tool
  result over a new global prompt rule. **Always run `eval_brain.py 3` after any
  prompt change** — unit tests won't catch these regressions.
- **New tool?** Add: the Pydantic args model, the `_execute_*` handler, an entry
  in `TOOL_HANDLERS`, the JSON schema in `LENS_TOOLS`, the name in
  `TOOL_NAME_LEAK`, a `_confirmation_sentences` branch (or return `None` to let
  the LLM speak), and tests in `tests/test_tools.py` + an eval case.
- **Frontend is one file**, no build. Cards render via `buildCard` in
  `index.html`; all text injected with `textContent`/`escapeHtml` (never raw
  HTML). The chat bubble's `renderRich` supports **`**bold**` only** — no lists.
- `app/main.py` runs each chat turn in its own task so a client disconnect can't
  abort half-finished DB writes.
