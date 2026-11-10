# 0003. Own DynamoDB checkpointer, not an existing package

Status: accepted
Date: 2026-08-11

## Context

The LangGraph agent pauses and resumes threads, so it needs a checkpointer. `InMemorySaver`, which
it started on, loses every conversation on restart and cannot be inspected. The rest of the app
already talks to DynamoDB through one shared `aiobotocore` session.

## Options

- `langgraph-checkpoint-amazon-dynamodb` or `justinram11/langgraph-checkpoint-dynamodb`.
- A saver of our own over `BaseCheckpointSaver`.

## Decision

We write our own saver in `app/ai_assistant_langchain/checkpointer/`. Both packages target the
LangGraph 0.x interface, which has since grown `delete_for_runs`, `copy_thread`, `prune` and
`DeltaChannel` support, and both bring a second AWS client (`boto3` / `aioboto3`).

## Consequences

- About 600 lines (`saver.py`, `items.py`) are ours to keep in step with `langgraph-checkpoint`.
- One AWS client for the whole app. `app/lifespan.py` opens it once for the life of the process:
  a lazily opened client bound itself to whichever event loop reached it first, and one opened per
  operation cost about 20 ms against 3.6 ms for a call on an open one, a dozen times a turn.
- How the saver writes (unique channel versions, compressed blob) is in
  `docs/conversation-memory.md`.
