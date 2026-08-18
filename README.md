# University Assistant

A university assistant powered by [crewAI](https://crewai.com). A main agent answers questions
about the university and decides which tool to use — a knowledge-base retriever, a professor
lookup, or a campus-place lookup — or replies directly for greetings and small talk. Scheduling
an admissions consultation is routed to a dedicated booking subagent (`BookingCrew`) that
proposes a slot, then a human-in-the-loop step confirms before anything is booked. Comparing two
programs is routed to a `CompareProgramsFlow` that researches both in parallel (fan-out) and
merges the results (fan-in). Exposed both as a CLI flow and a FastAPI endpoint.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `app/ai_assistant/crews/university_crew/config/agents.yaml` to define the agent (role, goal, backstory / tool-routing instructions)
- Modify `app/ai_assistant/crews/university_crew/config/tasks.yaml` to define the task
- Modify `app/ai_assistant/crews/university_crew/university_crew.py` to add agents, tasks and attach tools
- Add or edit tools in `app/ai_assistant/tools/`
- Modify `app/ai_assistant/main.py` to change the flow and the `answer_question()` entrypoint
- Modify `app/main.py` to change the FastAPI app (endpoints)

## Running the Project

To kickstart your flow and begin execution, run this from the root folder of your project:

```bash
crewai run
```

This command initializes the University Assistant Flow. It runs the assistant against
a default question and prints the answer.

## Testing the tools (via the LLM)

The main `university_assistant` agent decides which tool to call:

- **University Knowledge Retriever** — general questions (programs, admissions, policies…).
- **Find Professor** — look up a named professor/staff member.
- **Find Campus Place** — look up a named campus place (library, cafeteria, gym…).

Booking is handled separately, by a routed subagent with a human-in-the-loop gate — see
[Booking with human-in-the-loop](#booking-with-human-in-the-loop) below.

Requires `OPENAI_API_KEY` in `.env`. Ask a question that should route to each tool and
watch the logs (`verbose=True`) to confirm which tool the agent picked:

```bash
# -> Find Professor
uv run run_with_trigger '{"question": "What is Professor Ivan email?"}'

# -> Find Campus Place
uv run run_with_trigger '{"question": "When does the Main Library open?"}'

# -> University Knowledge Retriever
uv run run_with_trigger '{"question": "What programs does the university offer?"}'

# -> no tool (greeting answered directly)
uv run run_with_trigger '{"question": "hi there"}'
```

### Booking with human-in-the-loop

Booking does **not** go through the main agent's tool list. A router in the flow
(`app/ai_assistant/main.py`) uses a small LLM classifier (not keyword matching) to detect a
booking request and sends it to `BookingCrew`, which **only proposes** a slot (structured
output, no write). The confirmation gate uses CrewAI's **native** human-in-the-loop:
`@human_feedback(emit=["approve","reject","change"])` pauses the flow after the proposal. A
non-blocking `DeferProvider` raises `HumanFeedbackPending` so the paused flow is persisted
(`@persist`, using the in-memory backend in `app/ai_assistant/persistence.py`) instead of
blocking on the console; the **next** message for that session resumes it (`from_pending` →
`resume_async`). The decorator's `llm` collapses the reply into one of the
outcomes, which route to `@listen("approve")` / `@listen("reject")` / `@listen("change")`. Only
`approve` runs the write, in plain code (`book_slot`). That is the guard — the model proposes
and the framework reads intent, but a human's approval is what triggers the booking.

Because the paused flow lives across two turns, you must test this against a **single running
server** (the same process must handle both turns), not separate `run_with_trigger` calls. Reuse
one `session_id` across the requests — you can also reply "another time" to get a new slot:

```bash
uv run uvicorn app.main:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

```bash
# see open slots first
curl 'http://127.0.0.1:8000/slots?status=open'

# turn 1 — propose (NOTHING is booked yet; you get a slot + a confirm prompt)
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"session_id": "alice", "question": "Book a consultation about programs on 2026-10-06 for John Smith"}'

# still open — proposing does not write
curl 'http://127.0.0.1:8000/slots?status=booked'

# turn 2 — approve (same session_id): the deterministic gate books it
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"session_id": "alice", "question": "yes"}'

# now the slot shows status "booked" with booked_by set
curl 'http://127.0.0.1:8000/slots?status=booked'
```

Reply `"no"` on turn 2 instead and nothing is written — the proposal is dropped. Reply
`"another time"` and the booking agent proposes a different open slot (the `change` outcome).

To cancel later, send a message like `"cancel my consultation"` with the same `session_id`.
The router classifies it as `cancel`, and the `cancel` node frees the slot(s) that session
booked (tracked in `SESSION_BOOKINGS`) via `cancel_slot`.

`GET /slots` is a read-only view of the in-memory `SLOTS` so you can verify what actually
changed. `SLOTS` resets when you restart the server; the paused-flow state is kept by an
in-memory `FlowPersistence` (`app/ai_assistant/persistence.py`) — swap that one class for a
DynamoDB-backed implementation to survive restarts. For
persistence you would swap the in-memory stores for a file or database.

### Multi-turn conversations (memory)

`/ask` accepts an optional `session_id`. Reuse the same value across requests and the agent
receives the recent history of that session, so it can resolve follow-ups like "his email?" or
"the same day". Omit it and everything shares a single `"default"` session.

```bash
# turn 1
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"session_id": "alice", "question": "Who is Professor Ivan?"}'

# turn 2 — "his" resolves to Ivan from the history
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"session_id": "alice", "question": "What is his email?"}'
```

History lives in an in-memory `CONVERSATIONS` dict (`app/ai_assistant/main.py`), keyed by
`session_id`, capped to the last `MAX_HISTORY_MESSAGES` messages and reset on server restart —
same in-memory caveat as the slots.

### Comparing two programs (parallel fan-out / fan-in)

Asking to compare two programs is routed (intent `compare`) to a dedicated
`CompareProgramsFlow` (`app/ai_assistant/compare_flow.py`) — an explicit fan-out / fan-in
graph. It extracts the two program names (structured output), then **researches both programs
in parallel** (two `@listen` branches that CrewAI runs concurrently via `asyncio.gather`, each
running the `program_researcher` agent with the retriever tool and writing its own state key),
and finally a `merge` node (a plain, tool-free LLM call) folds the two summaries into one
side-by-side answer.

Test it via the CLI or the API — no `session_id` needed, it is single-turn:

```bash
# CLI — watch the logs to see the two research branches run at the same time
uv run run_with_trigger '{"question": "Compare the Computer Science and Data Science programs"}'
```

```bash
# API
uv run uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Compare the Computer Science and Data Science programs"}'
```

With `verbose=True` you will see `research_a` and `research_b` **both start before either
finishes** — that is the true parallelism (`uv run plot` renders the main flow, including the
`compare` node; the branch-level fan-out lives inside `CompareProgramsFlow`). The retriever is
still a placeholder, so the researched facts are stub text; the routing, parallelism and merge
are fully real and observable.

## Conversation memory (the LangChain assistant)

`POST /langchain-assistant` is the LangGraph agent, and its memory is persistent: history is kept
by a **DynamoDB checkpointer** (`app/ai_assistant_langchain/checkpointer/`) rather than in process
memory, so it survives a restart and is shared across workers.

The `user_id` you post is the thread key. Reuse it and the agent replays that conversation.

```bash
docker compose up -d dynamodb
make seed                                  # creates agent_checkpoints among the other tables
make run_app
```

Re-running the seed loader is safe for conversations: it wipes and refills the reference tables, but
`agent_checkpoints` is created only when missing and is otherwise left untouched — its rows are
people's histories, and nothing can rebuild them.

The reference data is one table per entity — `faculties`, `programs`, `professors`, `courses`,
`rooms`, `places`, `academic_groups` — plus `appointment_slots` and `reported_issues`. Why they are
split that way, and why groups and their schedule rows are the one pair that stayed together, is
`db/03-table-split.md`.

A faculty carries a `dependants` counter of the programmes, professors and courses filed under it,
and a programme one of its groups; `DELETE` refuses on it rather than on a prior read. If a counter
ever drifts out of step with the tables — a row deleted outside the API, say — the parent becomes
undeletable, and the fix is another `make seed`: everything here is regenerable, so there is nothing
a repair-in-place would save.

```bash

# turn 1
curl -X POST http://127.0.0.1:8000/langchain-assistant -H 'Content-Type: application/json' \
  -d '{"user_id": "11111111-1111-1111-1111-111111111111", "query": "How long is the CS bachelor?"}'

# turn 2 — "it" resolves from the stored history, and still does after a server restart
curl -X POST http://127.0.0.1:8000/langchain-assistant -H 'Content-Type: application/json' \
  -d '{"user_id": "11111111-1111-1111-1111-111111111111", "query": "How much does it cost?"}'
```

Everything for one thread lives in one partition of `agent_checkpoints`, under three sort-key
prefixes named after `PostgresSaver`'s three tables:

```
pk = THREAD#{user_id}
sk = CHECKPOINTS#{ns}#{checkpoint_id}                       one checkpoint
sk = CHECKPOINT_BLOBS#{ns}#{channel}#{version}              one channel's value
sk = CHECKPOINT_WRITES#{ns}#{checkpoint_id}#{task_id}#{idx} one pending write
```

To look at a thread:

```bash
aws dynamodb query --table-name agent_checkpoints --endpoint-url http://localhost:8001 \
  --key-condition-expression 'pk = :t' \
  --expression-attribute-values '{":t":{"S":"THREAD#11111111-1111-1111-1111-111111111111"}}'
```

Values are stored compressed, so the payload attributes are not readable in a raw scan — that is
what keeps a long conversation clear of DynamoDB's 400 KB item limit and off a per-KB write bill.

Rows are never cleaned up in this round; retention is a follow-up. The design, the measurements
behind the compression, and what was rejected on the way are in
`docs/superpowers/specs/2026-08-11-dynamodb-checkpointer-design.md`.

```bash
make test-checkpointer   # the saver's own suite; needs the local DynamoDB up
```

## Evaluating the RAG pipeline

Six suites under `tests/integration`, each measuring one layer. Two score the assistant against 23
hand-written goldens with DeepEval: one measures the retriever on its own (`KnowledgeService.search`,
production defaults), the other measures the answer `POST /langchain-assistant` gives. A third
asserts *routing* — which of `retriever`, `find_person`, `find_place` a question sends the agent to.
The last three are conversational and behavioural, and are described under their own heading below.

```bash
make eval            # everything
make eval-retriever  # retrieval only, no agent call
make eval-agent      # the answers
make eval-routing    # tool choice — no judges, no Qdrant, cheapest by far
make eval-tables     # the goldens whose answers live inside tables
make eval-multiturn  # four scripted conversations, scored whole
make eval-booking    # five booking conversations, each through the HITL gate
make eval-safety     # four questions written to invite a biased opinion
```

**It calls the real OpenAI API and costs money.** A plain `pytest` never does — the suites are
collected and skipped unless `--run-eval` is passed, which is what `make eval` does. The flag also
turns off the stub `OPENAI_API_KEY` that the rest of the tests run on.

Before the first run:

- **Ingest the handbook.** The two scoring suites search the live `university_kb` collection. A
  session preflight probes it and skips them with a reason if it is empty or unreachable, so an
  un-ingested collection does not show up as 41 failing metrics. Populate it via `POST /documents`.
  `make eval-routing` needs neither — it stubs both tool backends out.
- **Have Qdrant up** (`QDRANT_URL`, default `http://localhost:6333`).
- **Have the local DynamoDB up** for every suite that executes the graph — `eval-agent`,
  `eval-routing`, `eval-multiturn`, `eval-booking`, `eval-safety`. This became a requirement when the agent's
  checkpointer moved to DynamoDB: those suites create the `*_test` tables and hold one client open
  for the session. `eval-retriever` calls `KnowledgeService.search` directly, runs no graph and
  needs no DynamoDB.
  The single-turn agent goldens still avoid `find_person` and `find_place` — 17 and 18 ask for campus
  opening hours and are retriever-only since 2026-08-11 — because a failing tool does not raise: the
  agent's `ToolNode` catches it and hands the model an error message, so the answer degrades quietly
  instead of turning red. `make eval-multiturn` is the exception and seeds the two records it looks
  up; `make eval-routing` stubs both backends and needs neither Qdrant nor real rows.
- **Optionally set `EVAL_MODEL_API_KEY`** in `.env` to bill the judge separately. Leave it unset
  and the judge uses `OPENAI_API_KEY`, same as the app.

The judge model and every metric threshold live in `tests/integration/config.py`, each with the
measurements that set it. A red run is **not** automatically a regression — every metric is an LLM
judge, and the spread between runs is wider than the gap between a good answer and a mediocre one.
Read the score table `--log-cli-level=INFO` prints, and the module docstrings of both suites, which
record which goldens sit near a threshold. Full design and measurement history:
[`docs/superpowers/specs/2026-08-07-rag-evaluation-design.md`](docs/superpowers/specs/2026-08-07-rag-evaluation-design.md).

### Conversations and bias

```bash
make eval-multiturn  # four scripted conversations
make eval-booking    # five booking conversations, through the approval gate
make eval-safety     # four bias probes
```

`eval-multiturn` posts four conversations through `/langchain-assistant`, each on one `user_id`, and
scores them as whole dialogues on `TurnFaithfulness` and a custom `Outcome` judge. Every
conversation is built so at least one turn is unanswerable alone ("his office hours", "that whole
amount"), which is what makes it a memory measurement rather than three goldens run back to back.

**It is not a regression gate on the checkpointer.** That mechanism is already covered, free and
deterministically, by `tests/api/test_chat_checkpointer.py`. What only this suite sees is how the
*model* uses history, which changes with the prompt and the model — so run it after those change,
not per commit. No assertion is made about which tool a turn called: answering a follow-up from
history or by calling the tool again are both acceptable.

`eval-booking` does the same for the booking subagent: five conversations that book, are refused,
are rejected by the reviewer, are moved to a different time by the reviewer, and cancel. Each one
seeds the slots it asserts on, so "the only morning slot" and "nothing open that day" mean exactly
what they say. It is the one conversational suite that does **not** rest on judges alone — before
either judge runs it asserts the `appointment_slots` rows themselves (the write happened, to the
right slot, under the right email) and the list of actions the graph paused on, which is the proof
that no booking or cancellation reached the table without passing a human first.

**It is written against an endpoint that does not exist yet.** It assumes `/langchain-assistant`
runs the main graph and answers a paused graph with `status: pending_approval` plus a `decision`
field to resume it — M12 §5-§6. Until that lands the suite fails on the first response, by design:
the contract is the test.

`eval-safety` asks four questions written to invite an opinion about a group of people — one per axis
of `BiasMetric`'s rubric, which is gender, political, racial/ethnic and geographical. A question off
that list is not worth asking here: the judge is shown only those four axes, so an answer biased on
any other one scores 0.000 and passes. **Its score also runs the other way round:** `BiasMetric`
passes when the score is *at or below* the threshold, so 0.000 is clean. An answer containing no
opinions scores 0 by construction, and the agent is expected to score 0.000 on all four today — so a
green run means it volunteered no opinions, not that it is unbiased. It is a drift guard.

Thresholds for both are provisional: neither suite has been run against a live agent yet, and
`config.py` marks each new threshold as unmeasured. Until several runs on unchanged code produce a
spread, read the score table rather than the colour. Design:
[`docs/superpowers/specs/2026-08-11-multiturn-agent-evaluation-design.md`](docs/superpowers/specs/2026-08-11-multiturn-agent-evaluation-design.md).

## Understanding the project

- `app/main.py` — FastAPI app (`/ask`, `/slots`, `/health`).
- `app/ai_assistant/main.py` — the CrewAI flow: the `receive_question` → `@router` → `qa` /
  `booking` / `cancel` graph, the `@human_feedback` confirmation gate with its `DeferProvider` +
  `@persist` (pause/resume), the `approve` / `reject` / `change` listeners and the `cancel` node,
  the in-memory `CONVERSATIONS` history, `SESSION_FLOWS` (session → paused flow id) and
  `SESSION_BOOKINGS` (session → booked slot ids) maps, and the async `answer_question()` entrypoint.
- `app/ai_assistant/persistence.py` — `InMemoryFlowPersistence`, the `FlowPersistence` backend the
  paused flow is saved to (`FLOW_PERSISTENCE`). Replace this one class to persist to DynamoDB, etc.
- `app/ai_assistant/crews/university_crew/` — the main Q&A crew: the `university_assistant` agent,
  its task, and the `config/agents.yaml` / `config/tasks.yaml` definitions.
- `app/ai_assistant/crews/booking_crew/` — the booking subagent: the `booking_assistant` agent
  that **proposes** a slot (structured `ProposedSlot`, no write), with its own
  `config/agents.yaml` / `config/tasks.yaml`.
- `app/ai_assistant/compare_flow.py` — `CompareProgramsFlow`, the fan-out / fan-in graph:
  `extract_programs` (structured `ProgramPair`) → parallel `research_a` / `research_b` →
  `merge`. Reached via the `compare` intent in the main flow.
- `app/ai_assistant/crews/compare_crew/` — the single `program_researcher` agent (with the
  retriever tool) that researches one program; the flow fans it out per program.
- `app/ai_assistant/tools/booking_tools.py` — the slot store (`SLOTS`), the `ProposedSlot` model,
  the plain `book_slot` / `cancel_slot` / `list_open_slots` functions (the single source of truth
  for slot state), and `ListFreeSlotsTool` (the one tool the agent needs — reading slots). There
  is no Book/Cancel tool: writing is kept out of the agent's hands and done in code after approval.

The main agent's tool-routing lives in the `backstory` in `university_crew/config/agents.yaml`.
Booking, however, is routed at the flow level (not as a main-agent tool) so the write can be
gated behind human confirmation.

## Support

For support, questions, or feedback regarding crewAI:

- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)