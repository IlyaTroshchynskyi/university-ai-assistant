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

The request is a form, not a JSON body: `user_id` and `query` as form fields, `multipart/form-data`
or URL-encoded. The same form carries the files of a message that has any (below) and, in place of
`query`, the `decisions` that answer a pending approval — a JSON array in one field, because a form
has no nesting. A JSON body is answered with a `422`.

```bash

# turn 1
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'query=How long is the CS bachelor?'

# turn 2 — "it" resolves from the stored history, and still does after a server restart
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'query=How much does it cost?'

# answering an approval the thread is paused on
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'decisions=[{"type": "approve"}]'
```

### Streaming

The same endpoint serves two representations, and the `Accept` header is the switch:
`Accept: text/event-stream` gets Server-Sent Events, anything else gets the JSON above, unchanged.

    make stream_turn

| `event:` | `data:` | When |
|---|---|---|
| `token` | `{"text": "…"}` | zero or more, in order |
| `done` | the full `ChatSchemaOut` | the turn finished |
| `pending` | the full `ChatSchemaOut`, `status: "pending_approval"` | the graph paused on an approval |
| `error` | `{"detail": "…"}` | the turn failed after the response had started |

**Tokens are a preview, `done` is the truth.** Render tokens as they arrive, then replace the
message with `done.message`: a failed tool replaces the answer wholesale, discarding whatever was
streamed before it failed. Tokens are also not all small — a `compare_programs` answer streams from
the tool's own graph, but falls back to the whole comparison in a single `token` when that graph
streams nothing (`docs/specs/streaming-responses.md` §4.4).

Two behaviours follow from the turn outliving its request:

- a turn is recorded even if the caller hangs up mid-answer — the graph had already checkpointed it,
  so the transcript would otherwise disagree with the agent's own memory;
- re-asking on that same thread while the run is still going gets a `409` for a few seconds —
  whichever representation the re-ask uses, streaming or JSON.

That 409 is held in the process's own memory, so **run a single worker**. Under
`uvicorn --workers 2`, or several replicas, a retry that lands on another process sees nothing in
flight and starts a second turn on a thread the first one is still writing to.
`docs/specs/streaming-responses.md` §7.3 has the alternative and why it is not taken yet.

`docs/manual-testing.md` has the `curl -N` scripts, and `docs/specs/streaming-responses.md` is the
design.

### Checking an applicant's documents

A message can carry files: the same `POST /langchain-assistant`, with up to five parts named
`files` next to `user_id` and `query` — PNG, JPEG or PDF, 5 MB each. Nothing says which file is
which document: the assistant tells the passport, the tax code and the school certificate apart by
reading them. They can come as a pack or one message at a time: what is read off a file is kept for
the rest of the conversation, and the verdict is always about everything received so far. A PDF is
read by its first page, drawn at
150 dpi and never more than 2000 pixels on its longer side, so a page of any declared size costs a
bounded amount of memory.

```bash
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 \
  -F files=@passport.png \
  -F files=@tax_code.png \
  -F files=@certificate.pdf
```

Without a `query` the turn is recorded in the transcript as "Please check my documents.". The
model is not shown that sentence — it is ours, in English, and the model answers in the language of
the last message it reads. The answer is the `ChatSchemaOut` every turn returns, and it streams like
any other turn when the request asks for `text/event-stream`. Files cannot travel with `decisions`.

A message that carries files skips the router and goes to the QA agent, the only one that can read
them. The model's copy of the message ends with a line counting what came with it —
`[Files attached to this message: 2]` — because nothing else tells it that an "is it fine now?"
arrived with a new certificate. A message sent with files and no text is that line alone, and the
prompt tells the model to keep answering in the language the applicant last wrote in. The line
carries a count and not the file names, which are the applicant's own text. The transcript keeps
the message as it was written.

**The files are not stored; what was read off them is, for the conversation.** The images travel
in the graph's run context for that one request, which is where the `validate_documents` tool reads
them, and they are gone when it ends. The fields the tool read are kept in the conversation's graph
state as `documents_on_file` — one entry per document, the latest file of each — so they are in
`agent_checkpoints` with the rest of the thread. That is what lets an applicant send the passport
now and the certificate two messages later, or ask "what is still missing?" with no file at all. A
file sent later replaces the earlier file of the same document, so a sharper scan corrects a
blurred one. A new conversation starts with nothing on file. A passport number and a tax id are
masked but for their last two characters (`*******67`) before they are kept or shown to the model:
the rules only need to know they were there, and the two characters are enough for an applicant to
recognise the document.

Checking is two stages that do not know about each other:

1. **Extraction** turns each image into fields, the first of which is `document_type`: what the
   image is — a passport, a tax code, a certificate, or none of the three. `Extractor` in
   `app/ai_assistant_langchain/applicant_documents/extractors.py` is the interface; `OpenAIVisionExtractor` is the
   one implementation, a multimodal message to `VISION_MODEL` with a structured answer. All
   attached files are read at the same time, one call each.
2. **Validation** is code, `check_applicant_documents` in `document_rules.py`: the same rules whatever
   read the fields. It puts each file of the message on file under the `document_type` it was read
   as, in place of an earlier one, and then judges the documents on file.

The verdict is `is_valid` plus `issues`, one code per problem with the documents on file, in the
vocabulary the course's sample sets use:

| Issue | Means |
|---|---|
| `missing_document:<document>` | that document has not been received in this conversation yet |
| `missing_field:<document>.<field>` | the document does not show a field it has to |
| `unreadable:<document>.<field>` | the field is printed but could not be read with confidence |
| `expired:<document>` | past its expiry date — the passport |
| `name_mismatch` | the documents do not agree on the full name |
| `date_of_birth_mismatch` | the documents do not agree on the date of birth |

A passport has to show a full name, a date of birth, a number and an expiry date; a tax code a full
name, a date of birth and a tax id; a certificate a full name and a GPA.

Two names are the same when they are made of the same words: letter case, punctuation and the order
of the words do not count, so `KOVALENKO, MARIA` on a passport matches `Maria Kovalenko` on a
certificate. A word one document has and another lacks — a middle name, a patronymic — still makes
a `name_mismatch`. A date printed in digits alone, with nothing on the document to show which number
is the day, is read day first.

What became of the files of the message itself is a second list, `file_outcomes`, one entry per
file in the order attached:

| Outcome | Means |
|---|---|
| `added` | the document was not on file and now is |
| `replaced` | the file took the place of an earlier file of the same document |
| `unchanged` | the file reads exactly like the one already on file, so nothing changed |
| `duplicate` | another file of the same message is the same document; neither is taken, and what was on file stays |
| `unrecognized` | the file is none of the three documents; it is not taken |

An outcome is not an issue: `is_valid` is about the documents on file alone, so a stray screenshot
sent after three good documents is mentioned, not held against them. The outcomes are what lets the
assistant say "I already had this passport" or "the new certificate replaced the earlier one"
instead of repeating the verdict. A replacement counts even when it is worse than what it replaced:
a certificate with another name on it, sent after a good one, makes the set invalid.

The verdict also carries `documents_on_file`, the documents it judged, and `extracted_fields`, what
was read off each file of this message, in the same order as the outcomes.

A file that is not a PNG, a JPEG or a PDF is refused with a `415` naming the file, and so is a
PDF that cannot be opened, is protected by a password or has no pages. A file over the size limit is
refused with a `413`, and a sixth file with a `422`. A thread that is busy or waiting for an
approval answers `409` before any file is read. All of this happens before anything reaches the
model, and before a stream starts.

Two limits are not handled in the application. The size limit is per file and is checked after
the request has arrived, so the total size of a request is whatever the server in front of the
application allows. And PDF pages are drawn one at a time, because PyMuPDF must not
run on two threads at once: a second PDF waits for the first.

When a file cannot be read — the vision model timed out, or refused the image — the tool says
that nothing was checked instead of failing the turn, and the assistant asks for the files again.
One unread file leaves all the files of that message unread — the reads run together and stop
together — and what was already on file as it was.

No applicant's document is kept in the repository. The two sets the course describes — a valid one,
and a broken one with a misspelt name, no tax code and no GPA — are built in code by
`tests/factories/applicant_factory.py`, which also draws each document as an image when a test
needs one. The rules are checked against both sets on every `pytest`; `make eval-documents` draws
the five images and has the real vision model read them, which costs money like every other
`make eval-*`.

### The transcript (`conversation_history`)

Two tables, because the agent's memory and the applicant's conversation are two different things.
`agent_checkpoints` is the graph's state — tool calls, their results, pending writes, compressed and
only meaningful to LangGraph. `conversation_history` is what was *said*: a row per message, written
by `app/ai_assistant_langchain/history/`.

Each turn is stored as one `TransactWriteItems` — the question and the answer land together or not
at all. Two `put_item` calls have a moment between them in which the question is stored and the
answer is not, and a request that dies there leaves that behind for good: a conversation that reads
as unanswered, or, the other way round, a reply to something nobody asked. A turn paused on an
approval stores its question alone and its answer when the decision comes in, and neither is a
different kind of write.

```
pk = THREAD#{user_id}      thread_key() in app/core/dynamodb/indexes.py — the checkpointer's too
sk = {created_at}          UTC, fixed width, so the sort key is the order the lines were said in
```

The question's `created_at` is taken *before* the agent runs, the answer's when it is written. That
is what keeps a slow turn in its place: a second question asked while the first was still being
answered is filed where the applicant asked it, not after the answer that arrived later.

`GET /langchain-assistant/{user_id}/history` replays that table as `[{role, content}]`, oldest
first — what a chat client needs to redraw a conversation it did not have open.

```bash
curl http://127.0.0.1:8000/langchain-assistant/11111111-1111-1111-1111-111111111111/history
```

The replay used to be reconstructed from the graph's state, and that meant a rule per LangGraph
detail standing between the applicant and what they were told: drop the tool calls and their
results, keep the result of a `return_direct` tool because that one *is* the answer, drop the empty
assistant turns, and inherit whatever a summarisation step had done to the messages before them.
The table stores the two lines that were actually exchanged, so a replay is now a query. An id
nobody has written under answers `[]` rather than 404: a conversation that has not started is not a
missing resource, and a client opening a fresh thread would otherwise have to treat its own first
visit as an error.

```bash
aws dynamodb query --table-name conversation_history --endpoint-url http://localhost:8001 \
  --key-condition-expression 'pk = :t' \
  --expression-attribute-values '{":t":{"S":"THREAD#11111111-1111-1111-1111-111111111111"}}'

make test-history   # the table, the repository and the service; needs the local DynamoDB up
```

Like `agent_checkpoints`, this table is created only when missing and never re-seeded — `make seed`
does not cost anybody their conversation. Retention is a follow-up for both.

A thread paused on an approval replays *with* it: the last row carries a `pending` array in the
same shape `POST /langchain-assistant` returns, and its `content` is the same sentence a live turn
would have shown. That row is appended rather than stored, because a pause is not something that
was said — it is addressed to a reviewer, it is built from the live interrupt (`aget_state`), and it
stops being true the moment the approval is answered. Both paths go through one `_describe_pause`,
so a reviewer who reloaded answers the same thing, described the same way, as one who did not.

That closes what would otherwise be a dead end: a client that reloaded mid-review saw a
conversation that had simply stopped, could not answer an approval it could not see, and could not
send a new message either, because a paused thread refuses one.

`pending` is absent on every ordinary row, so a client that ignores it still reads a correct
transcript.

### The agent's state (`agent_checkpoints`)

Everything for one thread lives in one partition, under three sort-key prefixes named after
`PostgresSaver`'s three tables:

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

## Tracing with LangSmith

The LangSmith SDK configures itself from the process environment, so export the four names before
starting the app:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_pt_...          # https://smith.langchain.com → Settings → API keys
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_PROJECT=university-assistant
```

`.env` on its own does not reach the SDK: pydantic-settings loads that file into `Settings` and never
into `os.environ`. Exporting covers both — pydantic-settings reads the environment first, so
`app/settings.py` sees the same values.

All four are declared without defaults, so the app refuses to start until they are set. To run
untraced, set `LANGSMITH_TRACING=false` and leave the other three empty.

Nothing is instrumented by hand — LangChain and LangGraph already emit a run tree, so one request
arrives in LangSmith as the whole thing: the router's classification, the subagent it chose, every
tool call with its arguments and its result, and, for `compare_programs`, one `gather` branch per
programme next to each other with `merge` waiting on all of them. That is the easiest way to *see*
the fan-out:
the branches overlap on the trace's timeline instead of running end to end.

One thing it does not cover: the CrewAI flow (`crewai run`) goes through its own entrypoint, not
`create_app()`, so it is not traced by this.

The evaluation suites drive the app through `create_app()`, so with tracing on a `make eval-*` run
lands in the same project — useful for reading a failed golden, and worth turning off when it is
not what you want.

## Evaluating the RAG pipeline

Seven suites under `tests/integration`, each measuring one layer. Two score the assistant against 23
hand-written goldens with DeepEval: one measures the retriever on its own (`KnowledgeService.search`,
production defaults), the other measures the answer `POST /langchain-assistant` gives. A third
asserts *routing* — which of `retriever`, `find_person`, `find_place`, `get_schedule`,
`check_scholarship`, `validate_documents`, `list_common_issues` a question sends the agent to, and
that a complaint to file waits for approval before `report_issue` runs.
Three more are conversational and behavioural, and are described under their own heading below. The
seventh draws the two applicant sets as images, has the vision extractor read them and compares the
verdict with the one each set should get.

```bash
make eval            # everything
make eval-retriever  # retrieval only, no agent call
make eval-agent      # the answers
make eval-routing    # tool choice — no judges, no Qdrant, cheapest by far
make eval-tables     # the goldens whose answers live inside tables
make eval-multiturn  # four scripted conversations, scored whole
make eval-booking    # five booking conversations, each through the HITL gate
make eval-safety     # four questions written to invite a biased opinion
make eval-documents  # the vision extractor on the two applicant sample sets
```

**It calls the real OpenAI API and costs money.** A plain `pytest` never does — the suites are
collected and skipped unless `--run-eval` is passed, which is what `make eval` does. The flag also
turns off the stub `OPENAI_API_KEY` that the rest of the tests run on.

Before the first run:

- **Ingest the handbook.** The two scoring suites search the live `university_kb` collection. A
  session preflight probes it and skips them with a reason if it is empty or unreachable, so an
  un-ingested collection does not show up as 41 failing metrics. Populate it via `POST /documents`.
  `make eval-routing` needs neither — it stubs every tool backend out.
- **Have Qdrant up** (`QDRANT_URL`, default `http://localhost:6333`).
- **Have the local DynamoDB up** for every suite that executes the graph — `eval-agent`,
  `eval-routing`, `eval-multiturn`, `eval-booking`, `eval-safety`. This became a requirement when the agent's
  checkpointer moved to DynamoDB: those suites create the `*_test` tables and hold one client open
  for the session. `eval-retriever` calls `KnowledgeService.search` directly, runs no graph and
  needs no DynamoDB. Neither does `eval-documents`, which calls the extractor and the rules
  directly and needs no Qdrant either.
  The single-turn agent goldens still avoid `find_person` and `find_place` — 17 and 18 ask for campus
  opening hours and are retriever-only since 2026-08-11 — because a failing tool does not raise: the
  agent's `ToolNode` catches it and hands the model an error message, so the answer degrades quietly
  instead of turning red. `make eval-multiturn` is the exception and seeds the two records it looks
  up; `make eval-routing` stubs every backend and needs neither Qdrant nor real rows.
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

### Running the same cases by hand

[`docs/manual-testing.md`](docs/manual-testing.md) is every question from the suites above as a
copy-paste script against a running server — the goldens, the routing questions, the four
conversations, the five booking flows through the approval gate, and the bias probes. Same cases, no
judges and no OpenAI bill for scoring, so it is the loop to use while iterating on a prompt or a tool
description. It also lists what the suites do *not* cover — other languages, input validation, the
approval protocol's error paths, lookup misses, prompt injection — which is only reachable by hand.

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