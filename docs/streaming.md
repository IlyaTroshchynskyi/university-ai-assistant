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
