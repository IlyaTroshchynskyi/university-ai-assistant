# Streaming

How `POST /langchain-assistant` streams an answer. Why it is the same endpoint and why a turn
outlives its request: `docs/decisions/0005-stream-on-the-same-endpoint.md`.

`Accept: text/event-stream` gets Server-Sent Events; anything else gets the JSON `ChatSchemaOut`,
unchanged. `decisions` stream too, when the caller asks.

```bash
make stream_turn   # one streamed turn against the running app; -N makes curl print events as they come
```

## What a client receives

| `event:` | `data:` | When |
|---|---|---|
| `token` | `{"text": "…"}` | zero or more, in order |
| `done` | the full `ChatSchemaOut`, `status: "answer"` | the turn finished |
| `pending` | the full `ChatSchemaOut`, `status: "pending_approval"`, `pending: [...]` | the graph paused on an approval |
| `error` | `{"detail": "…"}` | the turn failed after the response had started |

**Tokens are a preview, `done` or `pending` is the truth.** A client renders tokens as they arrive,
then replaces the bubble's text with `data.message`. That is what makes three behaviours work without
a special case each:

- a failed tool replaces the answer wholesale with `TOOL_FAILURE_MESSAGE`, discarding what was
  streamed before it failed;
- a `return_direct` answer (`compare_programs`) is the tool's own graph writing, not the subagent's
  tokens; when that graph streamed nothing, the whole answer comes as one `token`, so a client must
  not assume tokens are small;
- a paused turn ends with a sentence addressed to the reviewer plus the `pending` array the approval
  card is drawn from.

`done.message` is byte for byte what the transcript stores, so a reload replays what was on screen.
The one exception is a turn whose tool failed: it is not recorded at all, so a reload shows neither
the question nor `TOOL_FAILURE_MESSAGE`, and the applicant asks again.

There is no `[DONE]` sentinel and no reconnection id: a stream is one turn, and a client that lost it
re-asks. `EventSourceResponse` sends keep-alive comments and sets `Cache-Control: no-store` and
`X-Accel-Buffering: no`. `Vary: Accept` is not set on purpose: no browser or CDN caches a POST, and
the 409 and 422 refusals are JSON whatever was asked.

Every event goes through `ChatService._build_server_sent_event`, which writes
`model_dump_json()`: `ServerSentEvent` puts `str(data)` on the wire, so a model handed to it directly
would arrive as a Python repr.

## Refusals come before the first byte

A paused thread, a busy thread, a bad `decisions` payload and a refused upload are all answered with
their status code (409, 422, 413, 415) before the stream starts, because once an
`EventSourceResponse` has started the status is already 200. A request asking for a stream that
trips a guard gets `409 application/json`: the header asks for a stream, it does not promise one.
`start_streaming_turn` checks that the thread is not busy and registers the run with no `await`
between them, so two requests arriving together cannot both pass.

After the first byte there is no status left to send. An OpenAI timeout, a Qdrant outage or a failed
DynamoDB write becomes an `error` event with a sentence for the applicant, never a traceback, and is
logged with `logger.exception`.

An interrupt mid-stream needs nothing special: the turn ends with a `pending` event carrying the same
`pending` array the JSON answer has.

## The turn runs detached

The graph runs in its own task, which feeds an unbounded `asyncio.Queue` that the response drains.
Everything that decides what is true about the turn (the final `ChatSchemaOut`, the failed-tool
check, recording the transcript) runs in that task, not in the response generator, because the task
is the half that survives a disconnect. So a turn is recorded even if the caller hangs up
mid-answer.

- The queue is unbounded on purpose: a bounded `put` would block on the consumer that just left, and
  the turn would never be recorded. A turn is bounded by the model's own timeout.
- `runs.py` holds a strong reference to each running task: CPython keeps only a weak one, and an
  unreferenced task can be collected mid-run.
- Re-asking on a thread whose turn is still running is a 409 for a few seconds, from either
  representation. The registry is process memory, hence a single worker.

## Which tokens are forwarded

LangGraph streams every model call in the graph, so most chunks are not the answer. A chunk is
forwarded only when `ChatService._is_answer_token` holds all of these, each added for a leak that
was measured:

- **Namespace depth 1 (`SUBAGENT_DEPTH`), a subagent.** The router's structured-output call
  streams at the root, so without this every answer would open with `{"destination": "qa"}`. A graph
  that a tool runs streams one level deeper, and its node could also be called `model`.
- **Node `model` (`MODEL_NODE`).** `SummarizationMiddleware` calls the same chat model in the same
  namespace; without this, "Here is a summary of the conversation so far…" lands in the answer. It
  only fires past 8000 tokens, so it passes every short test.
- **An `AIMessageChunk`.** `messages` mode carries whole `ToolMessage`s too; the retriever's raw
  passages would land ahead of the answer.
- **Non-empty `chunk.text`**, not `content`: a content-block answer has a non-empty `content` list and
  no text, which would send tokens that append nothing.

Two exceptions, both named rather than implied:

- `_is_direct_answer_token`: depth 2 (`TOOL_GRAPH_DEPTH`) and node `merge`
  (`DIRECT_ANSWER_NODE`), the `compare_programs` graph writing its answer. Without it the applicant
  waits through two searches and a whole generation. Once that text has streamed, the tool's
  `ToolMessage`, which repeats it, is dropped.
- `_is_direct_answer`: a `ToolMessage` from a tool in `DIRECT_ANSWER_TOOLS` at depth 1, with
  `status != 'error'`. It is the fallback when the graph streamed nothing, such as a failure before
  `merge` or the node renamed upstream: a rename costs the trickle, not the answer. The status check
  matters: a tool that raises still produces a `ToolMessage` named after it, carrying the exception
  text, which the applicant would read before the apology.

Tests on these filters assert the event names, not only `done`, or a leak passes:
`tests/api/test_chat_stream.py`.

## What the filters rest on

The filters above, and reading the pause off `values`, follow from measurements rather than from
LangGraph's documentation. They were taken on `langgraph 1.2.10`, `langchain 1.3.14`,
`langchain-core 1.5.3` and `fastapi 0.139.0`, against `build_main_graph()` with its checkpointer
swapped for `InMemorySaver`, the real model, Qdrant and the dev slots, in six scenarios: a greeting
with and without `subgraphs=True`, a `retriever` question, a `compare_programs` question, a booking
that pauses, and a turn after 24 long messages of history. The script was throwaway, so this table
is the record.

After a LangGraph or LangChain upgrade, write the script again and check the table before trusting
the filters. The suite cannot do that job: it scripts the model, so it catches a filter that
stopped working, not a stream whose shape changed underneath the filters.

| # | Question | Measured |
|---|---|---|
| P1 | Do the subagents' tokens reach the parent stream? | Only with `subgraphs=True`, at `ns == ('qa:<task_id>',)` under node `model`. Without it the answer arrives once, whole, as an `AIMessage` at the root under node `qa`. |
| P2 | Does the router's call stream? | Yes, on every turn: `ns == ()`, node `__start__`, the raw routing JSON. |
| P2 | Where does a pause appear? | In `updates` twice, at the subagent and at the root, and in the `interrupts` of the last `values` part at both levels. With `updates` left out of `stream_mode` the root `values` still carries it, which is why the stream asks for `['messages', 'values']` only. |
| P3 | What arrives on `messages`? | `AIMessageChunk`s, whole `ToolMessage`s (a retriever result in one part, node `tools`), and from the summariser a `RemoveMessage` and a `HumanMessage`. The applicant's own message is not echoed. |
| P3 | What does a `return_direct` tool stream? | Its tool call as empty `AIMessageChunk`s under `model`, then the work of its own graph, then the answer as one `ToolMessage` under `tools`. |
| P3 | Does a graph a tool runs stream? | Yes: `compare_programs`' `merge` streamed 183 to 210 chunks at `ns == ('qa:<task_id>', 'tools:<task_id>')`, node `merge`. |
| P4 | Where does `SummarizationMiddleware` stream? | Node `SummarizationMiddleware.before_model`, same `ns` as the subagent, as ordinary `AIMessageChunk`s carrying the summary. It is checked before every model call, so a turn past the trigger can summarise twice. |
| P2, P3 | Is there a final root `values` part? | Always, and last, on a pause too. Its `messages` are what `ainvoke` returns. |
| P5 | Can one route serve both representations? | Yes. `response_model=ChatSchemaOut` survives a `ChatSchemaOut \| EventSourceResponse` return annotation, and `/openapi.json` shows both. No `Accept` (httpx sends `*/*`) gets the JSON body. A tripped guard is `409 application/json` and a validation error `422 application/json`, even when SSE was asked for. |
| P6 | Why not `fastapi.sse`? | Its `EventSourceResponse` encodes only for a generator endpoint, which cannot first answer 409 or JSON. Returned from a plain `async def` it fails with `AttributeError: 'ServerSentEvent' object has no attribute 'encode'`. `sse_starlette.ServerSentEvent` writes `str(data)`, hence `model_dump_json()`. |
| P7 | Does a detached turn survive the client hanging up? | Yes: the turn ran to the end and wrote the transcript, and `runs.py` held nothing afterwards. |
| P8 | Can a cancelled task still await a write in `finally`? | Once: the await after the `CancelledError` runs normally, and a second cancel needs `asyncio.shield`. The shutdown drain waits for running turns instead of cancelling them, so it does not depend on either. |

What the suite cannot see: httpx's `ASGITransport` awaits the whole app before it returns a
response, so `aiter_lines()` walks a turn that is already over, and the app is never sent
`http.disconnect`. The tests assert which events came and in what order, never that they arrive
one by one; only `make stream_turn` shows that. A mid-turn disconnect cannot be expressed through
it at all, so `test_the_run_finishes_with_nobody_consuming_the_stream` starts the turn and never
iterates `stream_events`.
