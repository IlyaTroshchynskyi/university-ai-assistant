# The CrewAI assistant

The first of the two assistants, `app/ai_assistant/`, served on `POST /ask`. It is **frozen**: kept
working, not developed. New work goes to the LangGraph assistant. It shares the knowledge base and
the DynamoDB repositories with it and nothing else.

`/ask`, `/documents`, `/slots` and `/health` are registered on the module-level `app` in
`app/main.py`, so they exist under `make run_app` but not in the `create_app()` the tests build.
There is no CLI entry point: `crewai run` and the old `uv run` scripts do not work.

## Asking

`/ask` is a form: `session_id` (required), `question`, and optional `files`.

```bash
make run_app

curl -X POST http://127.0.0.1:8000/ask -F session_id=alice -F 'question=Who is Professor Ivan?'

# "his" resolves from the session's history
curl -X POST http://127.0.0.1:8000/ask -F session_id=alice -F 'question=What is his email?'
```

A turn goes through `MainFlowService.answer_question` into the flow in `main_flow.py`, whose router
sends it to one of five branches:

- `verify` when files are attached: the documents are checked in one vision call that also judges
  them (`doc_verification/`);
- otherwise an LLM classification (`classify_intent`, never keywords) picks `qa`, `booking`,
  `cancel` or `compare`.

`qa` runs the `university_assistant` agent (`crews/university_crew/`), whose tools are the
knowledge retriever, `FindPersonTool` and `FindPlaceTool`; it answers greetings itself. Its
tool-routing instructions are the `backstory` in `university_crew/config/agents.yaml`.

History for a session is the last `MAX_HISTORY_MESSAGES` messages (`main_flow_service.py`), kept
in `CONVERSATION_STORE` (`conversation_store.py`).

## Booking with a human in the loop

Booking is routed at the flow level, not offered as a tool, so the write can sit behind a human's
approval:

1. `booking` runs `BookingCrew`, which only **proposes** a slot (structured `ProposedSlot`, no
   write).
2. `@human_feedback(emit=['approve', 'reject', 'change'])` pauses the flow after the proposal. Its
   `DeferProvider` raises `HumanFeedbackPending` instead of blocking on a console, and `@persist`
   saves the paused flow; `SESSION_FLOWS` maps the session to it.
3. The next message on that `session_id` resumes it (`from_pending`, `resume_async`). The
   decorator's `llm` collapses the reply into one outcome: `approve` books the slot in plain code
   through the slots repository, `reject` drops the proposal, `change` proposes another slot.
4. `cancel` frees the slots that session booked (`SESSION_BOOKINGS`).

The model proposes and the framework reads intent, but only a human's approval triggers the write.
Both turns must reach the same running server.

```bash
curl 'http://127.0.0.1:8000/slots?status=open'

curl -X POST http://127.0.0.1:8000/ask -F session_id=alice \
  -F 'question=Book a consultation about programs on 2026-10-06 for John Smith'

curl -X POST http://127.0.0.1:8000/ask -F session_id=alice -F 'question=yes'

curl 'http://127.0.0.1:8000/slots?status=booked'
```

`GET /slots` reads `appointment_slots` in DynamoDB, filtered by `status` when given.

## Comparing two programmes

The `compare` intent runs `CompareProgramsFlow` (`compare_flow.py`), an explicit fan-out and fan-in.
`extract_programs` pulls the two programme names out of the question; `research_a` and `research_b`
both listen to it, so CrewAI runs them concurrently, each kicking off the single-agent
`ProgramResearchCrew` (`crews/compare_crew/`) for one programme; `merge` waits on both and folds the
two summaries into one side-by-side answer with a plain LLM call that has no tool, so it cannot
fetch or invent facts beyond them. The branches write separate state keys (`info_a`, `info_b`)
because CrewAI Flow has no reducer: a branch that read a shared key, awaited, and wrote back would
lose its sibling's write.

## Storage

`STORE_BACKEND` (`db.py`) chooses where the paused flows and the conversation history live:
`sqlite` (the default, the file `APP_DB_PATH`, `app_state.sqlite3`) or `memory`, gone on restart.
Slots are in DynamoDB either way.
