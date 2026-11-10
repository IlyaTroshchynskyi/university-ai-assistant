# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI university-admissions assistant, built twice over the same data: once as a CrewAI flow and
once as a LangChain/LangGraph agent. Active development is on the LangGraph one. It follows a course
whose milestones are referred to throughout the docs as `M06`, `T06.3` and so on; the course files
are not in this repository.

Storage is DynamoDB (reference data, appointment slots, the agent's checkpoints, the transcript) and
Qdrant (the handbook, for retrieval). There is no authentication, by design.

The chat front end is a separate repository next to this one, `../universal-agent-chat`. Its
`.env.example` lists the routes it calls here and its README the wire contract it expects, so a
change to what the `/langchain-assistant` endpoints take or answer has to be made there too.

## Commands

The Makefile calls `ruff`, `mypy`, `pytest` and `uvicorn` bare, so run it from the activated `.venv`
or prefix with `uv run`.

```bash
uv sync                                  # install, including the dev group
docker compose up -d dynamodb            # local DynamoDB on :8001 — every test run needs it
docker compose up -d qdrant              # :6333 — for ingesting, the retrieval evals and a live turn
make seed                                # wipe and refill the reference tables from db/seed/
make run_app                             # uvicorn on :8000
make stream_turn                         # one SSE turn against the running app
make lint                                # ruff format + ruff check --fix over the whole repo, mypy over app and tests
make check                               # format check + lint + mypy + pytest; writes nothing, needs DynamoDB on :8001
```

Tests:

```bash
pytest                                                   # everything that is free and offline
pytest tests/dynamodb/test_schedule.py -v                # one file
pytest tests/api/test_rooms.py::TestRooms::test_create_room   # one test
make test-checkpointer
```

A plain `pytest` talks to the local DynamoDB only: it creates `*_test` tables for the session and
drops them, and runs the agent on a stub model. It never calls OpenAI or Qdrant.

Evaluations (`make eval`, `make eval-routing`, `eval-retriever`, `eval-agent`, `eval-tables`,
`eval-multiturn`, `eval-booking`, `eval-safety`, `eval-documents`) are the suites in
`tests/integration`, DeepEval ones but for the last. **They call the real OpenAI API and cost
money.** They are collected and skipped unless `--run-eval` is passed, which the make targets do.
What each needs before it runs: `docs/evals.md`.

`.env` must define `OPENAI_API_KEY` and all four `LANGSMITH_*` names — `Settings` declares them
without defaults, so the app and the tests refuse to start otherwise. To run untraced, set
`LANGSMITH_TRACING=false` and leave the other three empty. `.env` does reach the LangSmith SDK:
`app/settings.py` imports `crewai`, which calls `load_dotenv()`, so `LANGSMITH_TRACING=true` in
`.env` traces every turn. `tests/conftest.py` sets `LANGSMITH_TRACING=false` for every run, the
evaluations included.

## Working on a task

- Done means `make check` passes, unless the task was only to write tests. `make lint` is the one
  that rewrites files.
- Ask first:
  - a run of the evaluations (`make eval*`, `pytest --run-eval`): each one bills the OpenAI key;
  - stored data: a new attribute, an index or a re-seed is an option to offer, not a step of a
    design; `make seed` wipes the reference tables;
  - the wire contract of `/langchain-assistant`: the change also goes into `../universal-agent-chat`.
- Before a PR: `/custom-review`. The PR description fills `.github/pull_request_template.md`.

## Architecture

### Two assistants in one app

| | CrewAI | LangGraph |
|---|---|---|
| Code | `app/ai_assistant/` | `app/ai_assistant_langchain/` |
| Endpoint | `POST /ask` (multipart, `session_id`, optional files) | `POST /langchain-assistant` (a form: `query`, optional `files`, or `decisions`), `GET /langchain-assistant/{user_id}/history` |
| Memory | a SQLite file or process memory, chosen by `STORE_BACKEND` | DynamoDB |

They share the knowledge base and the DynamoDB repositories and nothing else. Each verifies an
applicant's documents its own way: the CrewAI flow in one vision call that also does the judging,
the LangGraph agent by reading fields off the images and judging them in code. The CrewAI one is
frozen: `docs/crewai-assistant.md`.

### The LangGraph assistant

`graph/main_graph.py` is the whole shape: a router (one structured-output LLM call) sends each
message to one of two subagents, `qa` or `booking`, each a `create_agent` graph added **as a node**.
Both are compiled without a checkpointer; the main graph owns the only one, and that is what lets a
pause inside a subagent surface through the endpoint. The router sees the last few turns as text
(`docs/decisions/0004-router-sees-recent-turns.md`).

The package is laid out by who uses what: the root is the endpoint, `graph/` is what both agents
share, each agent is a folder under `agents/` with its own tools and prompts. Rules for changing it:
`app/ai_assistant_langchain/CLAUDE.md`.

- **QA agent** — `agents/qa/`, tools in `tools.py` (`ASSISTANT_TOOLS`). `report_issue` is its one
  write, behind `HumanInTheLoopMiddleware`; `validate_documents` reads `AgentRunContext.documents`
  and judges in `applicant_documents/` (`docs/applicant-documents.md`).
- **Booking agent** — `agents/booking/`, tools in `tools.py`. The two that write (`book_appointment`,
  `cancel_appointment`) sit behind `HumanInTheLoopMiddleware`: the graph interrupts, the API answers
  `status: "pending_approval"`, and the next request carries `decisions` (approve / edit / reject)
  instead of `query` to resume. A thread that is paused refuses a new message with a 409.
- **Request path** — `router.py` → `ChatService` → `AgentService` plus `ConversationHistoryService`:
  one path for a message, with files or without. The same endpoint streams on
  `Accept: text/event-stream` (`docs/streaming.md`).
- **Two tables per conversation**, both keyed by `user_id` as the thread: `agent_checkpoints` is the
  graph's state, written by the hand-rolled saver in `checkpointer/`; `conversation_history` is what
  was actually said, and is what the history endpoint replays. Do not read one to answer for the
  other (`docs/conversation-memory.md`).
- **`runs.py`** tracks in-flight turns in process memory, so a second message on a busy thread gets
  a 409. That is why the app must run as a **single worker**.
- **`app/lifespan.py`** holds the checkpointer's DynamoDB client open for the life of the process and
  drains streaming turns on shutdown.

The graph, both agents and the chat model are `lru_cache`d factories. A new `with_structured_output`
call goes on `get_structured_output_model`, not on `get_chat_model`.

### Knowledge base

`app/ai_assistant/university_knowladge/` — the misspelling is the real package name. `KnowledgeService`
ingests a PDF (`POST /documents`; a flat character-split path and a structure-aware one under
`structured/` that keeps tables whole and prefixes them with an LLM summary) and searches it. Every
point carries a dense vector (OpenAI) and a sparse one (FastEmbed BM25), fused by Qdrant with RRF.
Both assistants call the same `search`.

### DynamoDB

Chosen to learn it (`docs/decisions/0001-dynamodb-not-postgres.md`).

- `app/core/dynamodb/` — `DynamoDBService` is a generic layer over **one table** on the low-level
  async client. `TableItem` derives keys (`<ENTITY>#{id}` / `#META`). `indexes.py` holds the index
  and key-attribute enums and the name normalisers; a derived key only matches when the writer and
  the reader share one implementation, so those live there rather than per caller.
- One table per entity, except `academic_groups`, which holds groups and their class rows in one
  partition (`docs/decisions/0002-table-per-entity.md`, `db/03-table-split.md`).
- Every read is a numbered access pattern in the table in `db/02-dynamodb-model.md`. A new read is
  added there first; one that has no index is a Scan and is recorded as a deliberate one.
- `app/api/v1/<entity>/` — rooms, faculty and programs have HTTP APIs (router / service /
  repository, wired with `Depends`): create, list and delete, with no update on any of them yet.
  Professors, places, slots, schedule and issues are reached by the agents outside any
  request — slots and issues are written there too — through `open_*_repository()` /
  `open_schedule_service()` context managers that open their own client.
- `db/load_dynamodb.py` is the seeder and bypasses the repositories, so it maintains the
  `dependants` counters itself. Reference tables are wiped on every `make seed`;
  `agent_checkpoints` and `conversation_history` are created only when missing and never touched.

### Errors

Services raise the types in `app/core/exceptions.py`; `app/core/exception_handler.py` is the only
place they become HTTP status codes. ruff bans `HTTPException` everywhere else.

## Tests

- Test data is built in the test; no sample files are checked in, the eval goldens aside.
- Rows are seeded through `tests/factory_helpers/`, never through the agent.
- Requesting a table fixture is how a test declares which tables it may leave rows in.
- The `app` fixture builds `create_app()`, so `/ask`, `/documents`, `/slots` and `/health` do not
  exist in tests.
- The rest, including the stub model and event loops, is in `tests/CLAUDE.md`.

## Conventions the tooling enforces or the code relies on

- Formatting and import order are ruff's: run `make lint`, do not order imports by hand.
- mypy is strict and covers `tests/` too, so fixtures need annotations. `disallow_any_explicit`
  holds for the domain code, so no `Any` in new signatures there; `tests.*` and the SDK adapters
  listed in `pyproject.toml` are exempt. Existing code types a raw DynamoDB row as a bare `dict`.
- **Tools and prompts:** what a tool returns, what goes in its docstring and what in the prompt, and
  what adding a QA tool takes: `app/ai_assistant_langchain/CLAUDE.md`.
- Concurrent awaits use `asyncio.TaskGroup`; ruff bans `asyncio.gather`. A failure inside one
  arrives as an `ExceptionGroup`; `checkpointer/saver.py` shows the `except*` unwrap for when the
  original type has to surface.
- **Naming:** functions start with a verb that names the operation (`find_`, `build_`, `check_`,
  `format_`, …) and carry the domain noun; bare nouns are values; a boolean starts with `is_`,
  `has_`, `can_` or `should_`. Full rules: `.claude/rules/naming.md`.
- **Code:** no comments or docstrings in new code (a `@tool` docstring stays), `_private` functions
  below public ones, a variant as a sibling method and an optional part as a default-empty
  parameter, no `new or old` fallbacks. Full rules: `.claude/rules/code-conventions.md`.
- **Terms:** `docs/glossary.md` — read before naming a new domain thing.

## Things that will mislead you

- **`db/seed/` and every `*.pdf` are gitignored.** The seed JSON and `docs/handbook.pdf` are on disk
  here but not in a fresh clone; `make seed` and the retrieval evals depend on them.
- `app/api/v1/rooms/schemas.py` and `enums.py` hold the schemas and enums of every seeded entity
  without a package of its own (groups, professors, courses, places, schedule, slots, issues), not
  just rooms. `app/core/dynamodb/schemas.py` has a second `Slot`, `SlotStatus`, `Professor` and
  `Place`, and those are what the slots, professors and places repositories read.

## Design documents

Docs change in the same PR as the code. `README.md` lists what is in `docs/` and when to read it.

A spec is one-off: `superpowers:brainstorming` writes it to
`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, and it is not committed. Read the current
task's spec before writing code; each acceptance criterion in it names the test that proves it.

A spec that chose between real alternatives leaves an ADR in `docs/decisions/` before it is
deleted; an agreement without alternatives is a line in this file instead. A new domain term is a
row in `docs/glossary.md`.
