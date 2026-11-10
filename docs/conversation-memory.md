# Conversation memory

How the LangGraph assistant (`POST /langchain-assistant`) remembers a conversation. Why the saver is
our own: `docs/decisions/0003-own-dynamodb-checkpointer.md`.

The `user_id` a request posts is the thread key. Reuse it and the agent continues that conversation,
after a server restart too:

```bash
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'query=How long is the CS bachelor?'

# "it" resolves from the stored state
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'query=How much does it cost?'
```

The request is a form, `multipart/form-data` or URL-encoded, not a JSON body; a JSON body is a
`422`. In place of `query` the same form carries the `decisions` that answer a pending approval, as
a JSON array in one field, because a form has no nesting:

```bash
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 -F 'decisions=[{"type": "approve"}]'
```

## Two tables

The agent's memory and the applicant's conversation are two different things, so they are two
tables, both keyed by `user_id`:

- `agent_checkpoints` is the graph's state: tool calls, their results, pending writes, compressed
  and only meaningful to LangGraph. It is written by the saver in
  `app/ai_assistant_langchain/checkpointer/`.
- `conversation_history` is what was *said*: a row per message, written by
  `app/ai_assistant_langchain/history/`. It is what the history endpoint replays.

Do not read one to answer for the other. Both are created by `make seed` only when missing and are
never wiped by it: their rows are people's conversations, and nothing can rebuild them. Neither has
retention yet; why `agent_checkpoints` must not get a TTL is below.

The state is in DynamoDB, but the app still runs as a single worker: which turns are in flight is
kept in process memory (`runs.py`), see `docs/streaming.md`.

## The transcript (`conversation_history`)

Each turn is stored as one `TransactWriteItems`: the question and the answer land together or not
at all. Two `put_item` calls have a moment between them in which the question is stored and the
answer is not, and a request that dies there leaves behind a conversation that reads as unanswered,
or a reply to something nobody asked. A turn paused on an approval stores its question alone and
its answer when the decision comes in.

```
pk = THREAD#{user_id}      build_thread_key() in app/core/dynamodb/indexes.py, the checkpointer's too
sk = {created_at}          UTC, fixed width, so the sort key is the order the lines were said in
```

The question's `created_at` is taken *before* the agent runs, the answer's when it is written. That
keeps a slow turn in its place: a second question asked while the first was still being answered
is filed where the applicant asked it, not after the answer that arrived later.

`GET /langchain-assistant/{user_id}/history` replays the table as `[{role, content}]`, oldest
first, which is what a chat client needs to redraw a conversation it did not have open:

```bash
curl http://127.0.0.1:8000/langchain-assistant/11111111-1111-1111-1111-111111111111/history
```

The replay used to be rebuilt from the graph's state, which took a rule per LangGraph detail: drop
the tool calls and their results, keep the result of a `return_direct` tool because that one *is*
the answer, drop the empty assistant turns, inherit whatever summarisation had done. The table
stores the two lines actually exchanged, so a replay is a query. An id nobody has written under
answers `[]`, not 404: a conversation that has not started is not a missing resource.

A thread paused on an approval replays *with* it: the last row carries a `pending` array in the
shape `POST /langchain-assistant` returns, and its `content` is the sentence a live turn would have
shown. That row is appended, not stored: a pause is not something that was said, it is built from
the live interrupt (`aget_state`), and it stops being true once the approval is answered. Both
paths go through `ChatService._describe_pause`, so a reviewer who reloaded sees the same thing as
one who did not. Without it, a client that reloaded mid-review saw a conversation that had stopped,
could not answer an approval it could not see, and could not send a new message either, because a
paused thread refuses one. `pending` is absent on every ordinary row.

```bash
aws dynamodb query --table-name conversation_history --endpoint-url http://localhost:8001 \
  --key-condition-expression 'pk = :t' \
  --expression-attribute-values '{":t":{"S":"THREAD#11111111-1111-1111-1111-111111111111"}}'
```

Tests: `tests/api/test_chat_history.py`.

## The agent's state (`agent_checkpoints`)

Everything for one thread lives in one partition, under three sort-key prefixes named after
`PostgresSaver`'s three tables:

```
pk = THREAD#{user_id}
sk = CHECKPOINTS#{ns}#{checkpoint_id}                       one checkpoint
sk = CHECKPOINT_BLOBS#{ns}#{channel}#{version}              one channel's value
sk = CHECKPOINT_WRITES#{ns}#{checkpoint_id}#{task_id}#{idx} one pending write
```

```bash
aws dynamodb query --table-name agent_checkpoints --endpoint-url http://localhost:8001 \
  --key-condition-expression 'pk = :t' \
  --expression-attribute-values '{":t":{"S":"THREAD#11111111-1111-1111-1111-111111111111"}}'
```

Values are compressed, so the payload attributes are not readable in a raw query.

Tests: `pytest tests/dynamodb/test_checkpointer.py` (`make test-checkpointer`) for the saver,
`tests/api/test_chat_checkpointer.py` for the agent on it.

### How the saver writes, and why

- **A channel's value is written only when it changed.** A checkpoint row points at blob rows by
  `channel_versions`; channels absent from `new_versions` keep their existing rows. That is the
  point of splitting blobs out: a long conversation does not rewrite every channel every superstep.
- **One transaction per checkpoint**, so a checkpoint row never references a blob row whose write
  failed. It costs 2× WCU and caps a write at 100 actions; `new_versions` holds a handful.
- **Blobs are read by exact key** with `batch_get`, built from the checkpoint's `channel_versions`.
  A `begins_with` query on `CHECKPOINT_BLOBS#` would drag back every version ever written for the
  thread, a set that only grows.
- **Pending writes follow `InMemorySaver.put_writes`**, because Pregel depends on it: a retried task
  must not overwrite what the first attempt committed (a conditional `put_item`, its failure
  swallowed), while an error or interrupt replaces the earlier one. Not a transaction: a failed
  condition means skip this item, not abandon the batch. `idx` is also stored as a number, because
  the special writes have negative indexes, which do not sort as strings.
- **`alist` filters metadata in Python**: metadata is an opaque serde blob, so DynamoDB cannot filter
  on it. Listing N checkpoints costs 2N extra calls, acceptable off the request path.

### Channel versions are unique per write

`get_next_version` returns `f'{counter:032}.{random.random():016}'`, upstream's format for
upstream's reason. The base class answers `current + 1`, and a saver that keys a blob row by
`(channel, version)` cannot use that: two checkpoints forked from one parent would both call the
channel version 2, land on one key, and the second write would overwrite the first. Reading the
first branch back would then return the other branch's state, silently. The zero-padded counter
keeps versions ordered, the random tail keeps siblings apart. An `int` version is still read.

### Why the blob is compressed

DynamoDB caps an item at 400 KB, and `messages` is one channel, so one row. Measured with the real
`JsonPlusSerializer` and `tiktoken` on non-repeating conversations grown to 8000 tokens, where
`SummarizationMiddleware` fires (`agents/qa/agent.py`):

| Shape at 8000 tokens | msgs | raw | zlib | ratio | raw, % of 400 KB |
| --- | ---: | ---: | ---: | ---: | ---: |
| short chat, English | 313 | 168 KB | 21 KB | 8.0× | 41% |
| short chat, Russian | 159 | 115 KB | 12 KB | 9.4× | 28% |
| turns with retriever results | 81 | 92 KB | 15 KB | 6.3× | 23% |

Bytes per token is not bytes per character. A serialized `AIMessage` carries 250–300 bytes of fixed
OpenAI `response_metadata`, so the cost per token rises as messages get shorter: 7 B/token for long
tool results, 21 for ordinary answers, 37 for one-line exchanges. In that last shape a 400 KB item
holds about 11,000 tokens, and the peak at the summarisation trigger reaches about 71 % of the
limit: a margin of 1.4×.

zlib closes it. The ratio is 6–9× even on non-repeating text, because what compresses is the
structure, the same metadata keys once per message. The worst shape goes from 20.9 to 2.6 B/token,
and a 400 KB item from about 19,600 to 156,000 tokens.

The larger argument is cost. DynamoDB bills writes per KB, rounded up, and `aput` runs about five
times per user message: a 300 KB item costs 300 WCU per write, so compressing to 20 KB turns about
1500 WCU per message into 100. One `zlib.compress` on tens of KB costs microseconds against a
network round trip. `COMPRESSION_LEVEL` in `saver.py` is zlib's default, 6.

### No retention, and why not a TTL

Nothing deletes old checkpoints or superseded blob rows; they accumulate, as they do under
`PostgresSaver`, which leaves retention to the operator. A TTL on the rows is the obvious fix and
the wrong one. DynamoDB's TTL is per item, and a channel that stops changing keeps the blob row of
its last write, which every newer checkpoint goes on referencing. A conversation that outlives the
TTL loses the blob of its least active channel, and loses it silently: `batch_get` returns only the
rows it found, so the channel is simply missing from the state. Refreshing every referenced blob on
every `aput` would cost an extra write per unchanged channel per superstep.

Retention, when it is needed, is per thread: an `adelete_thread` for "clear my history", which the
saver does not implement yet, and a job that deletes whole threads by their last activity.
