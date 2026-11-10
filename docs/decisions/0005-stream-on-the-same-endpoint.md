# 0005. Stream on the same endpoint; the turn is detached from the request

Status: accepted
Date: 2026-09-19

## Context

The front end wanted answers to appear as they are written. `POST /langchain-assistant` already
answered with one JSON body, and the front end posts both messages (`send`) and approval decisions
(`decide`) to that one route. A turn writes two tables: LangGraph checkpoints while it runs, and the
transcript is recorded when the turn ends.

## Options

Where the stream is served:

- a second route, such as `/langchain-assistant/stream`;
- the same route, chosen per request by `Accept: text/event-stream`.

What a client disconnect does to a running turn:

- A. nothing special: the turn is lost from `conversation_history` but kept in
  `agent_checkpoints`, and the two tables disagree;
- B. the turn runs in its own `asyncio.Task` that feeds a queue the response drains, so a
  disconnect closes only the response;
- C. write the question first and the answer later, as two writes.

## Decision

The same route, negotiated on `Accept`, and B.

A client that sends nothing gets byte for byte the JSON it got before, so nothing that existed had
to change, and `send` can stream while `decide` does not without forking the front end's route
config. The guards (a paused or busy thread is a 409) run before the response starts, because once
a stream has started its status is 200. That is also why the stream is `sse_starlette`'s
`EventSourceResponse` and not `fastapi.sse`: the latter only encodes for a generator endpoint, which
cannot first answer 409 or JSON.

B keeps the one-transaction turn (C reintroduces the half-written turn) and keeps the two tables in
agreement (A does not).

## Consequences

- A turn outlives its request, so a second message on that thread is refused with a 409 until it
  ends. The registry of running turns is `runs.py`, in process memory: **the app runs as a single
  worker**. Under several workers a retry on another process starts a second run on the same
  thread. Making the guard cross-process would take a conditional write with a TTL in DynamoDB, a
  lock that wedges a thread when it goes stale.
- Shutdown waits for running turns (`wait_for_streaming_turns` in `app/lifespan.py`) inside the
  open checkpointer, for at most `DRAIN_TIMEOUT_SECONDS` (25 s by default). A value above the
  platform's grace period is fiction: `docker stop` kills at 10 s, Kubernetes at 30 s.
- The detached task is the one place `asyncio.TaskGroup` must not be used: it would await the run
  on exit and tie it to the request again.
- How the stream is filtered and what a client receives: `docs/streaming.md`.
