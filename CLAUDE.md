# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI university-admissions assistant, built twice over the same data: once as a CrewAI flow and
once as a LangChain/LangGraph agent. Active development is on the LangGraph one. It follows a course
whose milestones are referred to throughout the docs as `M06`, `T06.3` and so on; the course files
are not in this repository.

Storage is DynamoDB (reference data, appointment slots, the agent's checkpoints, the transcript) and
Qdrant (the handbook, for retrieval). There is no authentication, by design.

## Commands

The Makefile calls `ruff`, `mypy`, `pytest` and `uvicorn` bare, so run it from the activated `.venv`
or prefix with `uv run`.

```bash
uv sync                                  # install, including the dev group
docker compose up -d dynamodb            # local DynamoDB on :8001 — every test run needs it
docker compose up -d qdrant              # :6333 — only for ingesting and for the retrieval evals
make seed                                # wipe and refill the reference tables from db/seed/
make run_app                             # uvicorn on :8000
make stream_turn                         # one SSE turn against the running app
make lint                                # ruff format + ruff check --fix + mypy, over the whole repo
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
`eval-multiturn`, `eval-booking`, `eval-safety`) are the DeepEval suites in `tests/integration`.
**They call the real OpenAI API and cost money.** They are collected and skipped unless `--run-eval`
is passed, which the make targets do. `eval-routing` stubs every tool backend and needs no Qdrant;
the scoring suites need the handbook ingested first (`POST /documents`).

`.env` must define `OPENAI_API_KEY` and all four `LANGSMITH_*` names — `Settings` declares them
without defaults, so the app and the tests refuse to start otherwise. To run untraced, set
`LANGSMITH_TRACING=false` and leave the other three empty. Tracing itself only switches on when the
names are exported into the process environment; `.env` alone does not reach the LangSmith SDK.

## Architecture

### Two assistants in one app

| | CrewAI | LangGraph |
|---|---|---|
| Code | `app/ai_assistant/` | `app/ai_assistant_langchain/` |
| Endpoint | `POST /ask` (multipart, `session_id`, optional files) | `POST /langchain-assistant` (JSON), `GET /langchain-assistant/{user_id}/history` |
| Memory | a SQLite file or process memory, chosen by `STORE_BACKEND` | DynamoDB |

They share the knowledge base and the DynamoDB repositories and nothing else. Document verification
(vision) exists only in the CrewAI flow.

### The LangGraph assistant

`main_graph.py` is the whole shape: a router (one structured-output LLM call) sends each message to
one of two subagents, `qa` or `booking`, each a `create_agent` graph added **as a node**. Both are
compiled without a checkpointer; the main graph owns the only one, and that is what lets a pause
inside the booking agent surface through the endpoint.

- **QA agent** — `agent.py`, tools in `tools.py` (`ASSISTANT_TOOLS`). `compare_programs` is
  `return_direct` and runs its own graph (`graphs/compare_programs/`), which fans out one retrieval
  per programme with `Send` and merges.
  `check_scholarship` reads nothing: the handbook's scholarship rules are code in
  `app/api/v1/scholarships/scholarship_rules.py`, so a rule that changes in the handbook has to
  change there too — `TestRulesMatchTheHandbook` fails when the two drift. A faculty added through
  the API has to be added to the `Faculty` enum next to it as well, or the tool rejects it.
- **Booking agent** — tools in `booking_tools.py`. The two that write (`book_appointment`,
  `cancel_appointment`) sit behind `HumanInTheLoopMiddleware`: the graph interrupts, the API answers
  `status: "pending_approval"`, and the next request carries `decisions` (approve / edit / reject)
  instead of `query` to resume. A thread that is paused refuses a new message with a 409.
- **Request path** — `router.py` → `ChatService` (`service.py`: one turn, JSON or SSE, and what gets
  recorded) → `AgentService` (`agent_service.py`: invoke, resume, stream, validate decisions) plus
  `ConversationHistoryService` (`history/`). The same endpoint streams when the request sends
  `Accept: text/event-stream`.
- **Two tables per conversation**, both keyed by `user_id` as the thread: `agent_checkpoints` is the
  graph's state, written by the hand-rolled saver in `checkpointer/`; `conversation_history` is what
  was actually said, and is what the history endpoint replays. Do not read one to answer for the
  other.
- **`runs.py`** tracks in-flight turns in process memory, so a second message on a busy thread gets
  a 409. That is why the app must run as a **single worker**.
- **`lifespan.py`** holds the checkpointer's DynamoDB client open for the life of the process and
  drains streaming turns on shutdown.

The graph, both agents and the chat model are `lru_cache`d factories. The booking prompt is rebuilt
per call (it carries today's date); the QA prompt is a constant and knows no date.

### Knowledge base

`app/ai_assistant/university_knowladge/` — the misspelling is the real package name. `KnowledgeService`
ingests a PDF (`POST /documents`; a flat character-split path and a structure-aware one under
`structured/` that keeps tables whole and prefixes them with an LLM summary) and searches it. Every
point carries a dense vector (OpenAI) and a sparse one (FastEmbed BM25), fused by Qdrant with RRF.
Both assistants call the same `search`.

### DynamoDB

- `app/core/dynamodb/` — `DynamoDBService` is a generic layer over **one table** on the low-level
  async client. `TableItem` derives keys (`<ENTITY>#{id}` / `#META`). `indexes.py` holds the index
  and key-attribute enums and the name normalisers; a derived key only matches when the writer and
  the reader share one implementation, so those live there rather than per caller.
- One table per entity, except `academic_groups`, which holds groups and their class rows in one
  partition. The reasoning is `db/02-dynamodb-model.md` and `db/03-table-split.md`.
- Every read is a numbered access pattern in the table in `db/02-dynamodb-model.md`. A new read is
  added there first; one that has no index is a Scan and is recorded as a deliberate one.
- `app/api/v1/<entity>/` — rooms, faculty and programs have HTTP APIs (router / service /
  repository, wired with `Depends`): create, list and delete, with no update on any of them yet.
  Professors, places, slots and schedule are read by the agents outside any
  request, through `open_*_repository()` / `open_schedule_service()` context managers that open
  their own client.
- `db/load_dynamodb.py` is the seeder and bypasses the repositories, so it maintains the
  `dependants` counters itself. Reference tables are wiped on every `make seed`;
  `agent_checkpoints` and `conversation_history` are created only when missing and never touched.

### Errors

Services raise the types in `app/core/exceptions.py`; `app/core/exception_handler.py` is the only
place they become HTTP status codes.

## Tests

- `tests/api/` drives the app over HTTP with `StubChatModel` (`tests/agent_stubs.py`): scripted
  replies, scripted routes, real graph, real checkpointer. The stub is installed by patching
  `get_model_factory` under **each importing module's own name** and clearing all three graph caches.
- `tests/dynamodb/` calls savers and services directly against the local DynamoDB.
- `tests/unit/` is for pure functions and needs neither DynamoDB nor a model.
- `tests/factories/` holds row models that mirror what the seeder writes, plus creators, getters and
  deleters. Seed rows through these rather than through the agent.
- Each table has a fixture in `tests/conftest.py` that hands out a service and **empties the table
  afterwards**. Requesting one is how a test declares which tables it may leave rows in.
- Event loops: ordinary tests run on a per-test loop; the eval suites run on the session loop
  (`pytest.mark.asyncio(loop_scope='session')`). A function-scoped async fixture used from a
  session-loop test passes and then errors on teardown — use `session_dynamo_client` there. See
  `docs/specs/event-loop-scopes-in-tests.md`.
- The `app` fixture builds `create_app()`. `/ask`, `/documents`, `/slots` and `/health` are
  registered on the module-level `app` in `app/main.py`, not inside `create_app()`, so they do not
  exist in tests.

## Conventions the tooling enforces or the code relies on

- ruff: single quotes, 120 columns, isort with `force-sort-within-sections` and case-insensitive
  ordering. Run it rather than ordering imports by hand.
- mypy is strict and covers `tests/` too, with `disallow_any_explicit`: no `Any` in new signatures,
  and fixtures need annotations. Existing code types a raw DynamoDB row as a bare `dict`.
- A tool returns `model_dump()` dicts or a string, never pydantic models: LangChain renders a result
  with `json.dumps` and falls back to `str()`, so a model — or a `datetime.time` — reaches the LLM as
  a Python repr.
- What an agent should do lives in its system prompt; what a tool is lives in the tool. Every rule
  of behaviour — when to call a tool and when not, what to look up or ask first, what to do with
  each kind of result — is in `MAIN_CHAT_PROMPT`, which has one section per QA tool, or in the
  booking prompt, which is laid out by task (finding a time, booking, cancelling, moving). A tool's
  docstring says only what it does and what its result means, a `Field(description=...)` only what
  the argument is and its format, and a result only what happened — none of them names another tool
  or tells the model what to do next. `tests/unit/test_tool_texts.py` holds the part of that a test
  can check, over `ASSISTANT_TOOLS` and `BOOKING_TOOLS`.
- Adding a QA tool means: input schema in `agent_schemas.py`, the tool and `ASSISTANT_TOOLS` in
  `tools.py`, a section in `MAIN_CHAT_PROMPT`, and a stub plus cases in
  `tests/integration/test_tool_routing_eval.py`.
- Concurrent awaits use `asyncio.TaskGroup`. A failure inside one arrives as an `ExceptionGroup`;
  `checkpointer/saver.py` shows the `except*` unwrap for when the original type has to surface.

## Naming

New code follows these three rules. Code written before them does not all follow yet, so an older
name is not a precedent.

### Functions get verbs, values get nouns

A function's name starts with a verb — a one-line private helper, a method and a test helper
included:

- `_money(amount)` -> `_format_money`
- `scholarship.worth(annual_tuition)` -> `scholarship.compute_worth(annual_tuition)`
- `verdicts(check)` -> `build_verdict_map`

A bare noun is reserved for a value: `reason = find_scholarship_reason(check, MERIT)`. A property
reads like a value, so it is named like one (`scholarship.award`, `scholarship.label`), and so is a
constant (`ONE_SCHOLARSHIP_RULE`, not `ONE_AT_A_TIME`).

### The verb says what the function actually does

`get_` is only for handing back something that already exists: a field, or an object built once and
cached (`get_settings`, `get_knowledge_service`). Anything that looks something up or transforms it
takes the verb this repo already uses for that shape of work:

- `find_…`      looks something up by a name or a key and may find nothing - `find_schedule`,
  `find_professors_by_name`
- `list_…`      every row of one kind - `list_group_classes`, `list_free_slots`
- `build_…`     assembles a structure or a prompt - `build_main_graph`, `_build_scholarship_match`
- `compute_…`   numeric result - `compute_worth`
- `check_…`     holds a fact against a rule and returns the verdict - `check_scholarship`,
  `_check_gpa_minimum`
- `pick_…`      chooses one out of several - `_pick_awarded_scholarship`
- `parse_…`     unstructured input -> typed shape - `parse_pages`
- `normalize_…` one spelling out of many - `normalize_name`
- `format_…`    value -> display string - `_format_money`, `_format_results`
- `open_…`      an async context manager that owns its client - `open_schedule_service`

`_handle_gpa` is wrong where `_check_gpa_minimum` is right: the verb must name the operation, not
gesture at it.

### The noun names the domain type

Qualify with the type the function works on: `_build_scholarship_match`, not `_match`;
`_pick_awarded_scholarship`, not `_pick_awarded`; `find_scholarship_reason`, not `reason_for`.

## Things that will mislead you

- **README.md has drifted.** It describes `app/ai_assistant/main.py`, `make test-history` and
  `docs/manual-testing.md`, none of which exist, and `[project.scripts]` in `pyproject.toml` still
  points at `app.ai_assistant.main`, so `crewai run` and `uv run kickoff` do not work. Trust the
  code and the Makefile.
- **`db/seed/` and every `*.pdf` are gitignored.** The seed JSON and `docs/handbook.pdf` are on disk
  here but not in a fresh clone; `make seed` and the retrieval evals depend on them.
- `app/api/v1/rooms/schemas.py` and `enums.py` hold the schemas and enums for **every** seeded
  entity, not just rooms.

## Design documents

`docs/specs/` and `docs/superpowers/specs/` hold a design document per feature, written before the
code and revised after it; `docs/specs/crud-roadmap.md` is the plan for the remaining CRUD entities.
They record why a shape was chosen and what was measured, which the code alone does not.
