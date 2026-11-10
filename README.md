# University Assistant

A FastAPI university-admissions assistant, built twice over the same data: once as a CrewAI flow
(`POST /ask`, frozen) and once as a LangChain/LangGraph agent (`POST /langchain-assistant`), where
development happens. A router sends each message to a QA agent (the handbook, staff, campus places,
timetables, scholarships, programme comparisons, applicant documents, reported issues) or a booking agent (consultation
slots); every write waits for a human's approval.

Storage is DynamoDB and Qdrant. There is no authentication, by design. The chat front end is a
separate repository, `../universal-agent-chat`.

## Quick start

Python 3.13 (`.python-version`), [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                          # install, including the dev group
docker compose up -d dynamodb    # local DynamoDB on :8001, needed by every test run
docker compose up -d qdrant      # :6333, needed for ingesting, the retrieval evals and a live turn
make seed                        # wipe and refill the reference tables from db/seed/
make run_app                     # uvicorn on :8000
make stream_turn                 # one streamed turn against the running app
```

`.env` must define `OPENAI_API_KEY` and `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
`LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`; to run untraced, set `LANGSMITH_TRACING=false` and leave
the other three empty. `db/seed/` and `docs/handbook.pdf` are gitignored, so a fresh clone needs
them copied in before `make seed` and before ingesting the handbook with `POST /documents`.

Run the Makefile from the activated `.venv`, or prefix commands with `uv run`.

```bash
make check    # format check, lint, mypy and pytest; writes nothing
make lint     # ruff format, ruff check --fix and mypy; rewrites files, runs no tests
pytest        # every free, offline test; needs the local DynamoDB
```

A plain `pytest` never calls OpenAI or Qdrant. The evaluations do, and cost money: `docs/evals.md`.

## Where to read what

| File | Read it when |
| --- | --- |
| `CLAUDE.md` | Working in the repository: architecture, commands, conventions, what to ask first |
| `app/ai_assistant_langchain/CLAUDE.md` | Changing the LangGraph assistant: its tools, prompts, request path and models |
| `tests/CLAUDE.md` | Writing tests: the stub model, fixtures, event loops |
| `docs/glossary.md` | Naming a domain thing, or reading a term you do not know |
| `docs/decisions/` | Wondering why something is built the way it is (ADRs) |
| `docs/conversation-memory.md` | Touching the checkpointer, the transcript or the history endpoint |
| `docs/streaming.md` | Touching the SSE path or what a client receives |
| `docs/applicant-documents.md` | Touching document checking |
| `docs/evals.md` | Running or changing an evaluation |
| `docs/tracing.md` | Turning on LangSmith |
| `docs/crewai-assistant.md` | Touching the frozen CrewAI assistant on `/ask` |
| `db/01-relational-schema.md` | The relational design the DynamoDB model started from |
| `db/02-dynamodb-model.md` | Adding a DynamoDB read: every access pattern, numbered |
| `db/03-table-split.md` | The table map and the `dependants` counter |
| `.github/pull_request_template.md` | Opening a PR |
