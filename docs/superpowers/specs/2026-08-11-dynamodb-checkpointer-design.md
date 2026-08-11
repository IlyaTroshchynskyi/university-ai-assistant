# A DynamoDB checkpointer for the LangChain agent

**Date:** 2026-08-11
**Status:** Implemented. Client lifetime went through two wrong shapes before the current one — see
[Client lifetime](#client-lifetime). Code review is recorded separately in
`docs/superpowers/reviews/2026-08-11-dynamodb-checkpointer-review.md`.

## Overview

`app/ai_assistant_langchain/agent.py` compiles its graph with `checkpointer=InMemorySaver()`. The
conversation history therefore lives in a process dictionary: it is lost on restart, and two
workers serving the same user hold two different histories. This replaces it with a checkpointer
backed by DynamoDB, written against `BaseCheckpointSaver` rather than taken off the shelf.

The layout follows the one `PostgresSaver` uses — checkpoint rows, per-channel blob rows and
pending-write rows — collapsed onto one table the way `university` already collapses seven entity
types.

Nothing about the agent, the tools or the API contract changes. `create_assistant_agent()` gets a
different checkpointer and `create_app()` gains a lifespan; that is the whole externally visible
diff.

## Motivation

`InMemorySaver` is documented by LangGraph as a debugging store. Three things it cannot do:

- **survive a restart** — every deployment silently wipes every conversation;
- **be shared** — `uvicorn --workers 2` gives a user a different history per request;
- **be inspected** — there is no way to look at what a user's thread actually contains.

DynamoDB is already the project's store: local instance in `docker-compose.yml`, async client
wiring in `app/core/dynamodb/client.py`, a generic CRUD layer in `base_service.py`. A checkpointer
is a small repository over one more table.

### Why not an existing package

Two exist — `langgraph-checkpoint-amazon-dynamodb` (last release 2025-11) and
`justinram11/langgraph-checkpoint-dynamodb`. Both are rejected:

- they were written against the LangGraph 0.x checkpointer interface; this project runs
  `langgraph` 1.2.10 / `langgraph-checkpoint` 4.1.1, where the interface has since grown
  `delete_for_runs`, `copy_thread`, `prune` and the `DeltaChannel` support surface;
- they talk to DynamoDB through `boto3` / `aioboto3`, so adopting one puts a second, differently
  configured AWS client next to the `aiobotocore` session the rest of the app shares.

The interface is stable and small. Writing it here costs roughly 250 lines and keeps one client.

## What the interface actually requires

From `langgraph/checkpoint/base/__init__.py` (4.1.1), an async graph reaches exactly four methods:

| Method | When LangGraph calls it |
| --- | --- |
| `aget_tuple(config)` | start of every turn — loads the thread's latest state |
| `aput(config, checkpoint, metadata, new_versions)` | end of every superstep |
| `aput_writes(config, writes, task_id, task_path)` | after each task, before the superstep commits |
| `alist(config, ...)` | `aget_state_history`, time travel |

`aget_delta_channel_history` is inherited: the base class walks the parent chain through
`aget_tuple` (lines 651-690), so a correct `aget_tuple` supplies it for free.

`acopy_thread`, `aprune` and `adelete_for_runs` are **not implemented**. Grepping the installed
`langgraph` and `langchain` packages finds no caller for any of them — they are surface for
LangGraph Platform (thread forking, background checkpoint pruning, deleting a cancelled run's
checkpoints), a product this app does not use. `adelete_thread` is deliberately deferred too; see
[Retention](#retention-is-out-of-scope-for-this-round).

The synchronous half of the interface (`get_tuple`, `put`, `put_writes`, `list`) raises
`NotImplementedError` with a message pointing at the async methods. `aiobotocore` has no
synchronous path, and the graph is only ever driven through `ainvoke`.

## Table layout

One new table, `agent_checkpoints`, keyed `pk`/`sk` like every other table in the project. **No
GSIs** — every read is scoped to a thread, and the thread is the partition.

```
pk = THREAD#{thread_id}

sk = CHECKPOINTS#{ns}#{checkpoint_id}                       → one checkpoint, minus its channel values
sk = CHECKPOINT_BLOBS#{ns}#{channel}#{version}              → one channel's value at one version
sk = CHECKPOINT_WRITES#{ns}#{checkpoint_id}#{task_id}#{idx} → one pending write
```

The three prefixes are the three `PostgresSaver` table names — `checkpoints`, `checkpoint_blobs`,
`checkpoint_writes` — upper-cased to match how this project already namespaces a single-table sort
key (`SCHED#`, `TYPE#ROOM`). One row type here corresponds to one Postgres table there, so anyone
who knows the upstream schema can read this one. `checkpoint_migrations`, the fourth Postgres
table, has no counterpart: it tracks DDL versions, and DynamoDB is schemaless.

The prefixes do not collide under `begins_with`: `CHECKPOINTS#` and `CHECKPOINT_` diverge at the
tenth character (`S` vs `_`), so a query for one row type can never pick up another.

`{ns}` is `checkpoint_ns`: empty for the main graph, `node:uuid`-shaped for a subgraph. It sits in
the sort key rather than the partition key so that a whole thread — every namespace — is one
partition. That makes "everything for this user" a single query, which is what a future
`adelete_thread` needs. LangGraph joins nested namespaces with `|` and separates node from task
with `:`, so `#` never appears inside a namespace and the prefixes stay unambiguous.

`checkpoint_id` is a uuid6: monotonically increasing and lexicographically sortable, so
`CHECKPOINTS#{ns}#` in ascending order is chronological, and the newest checkpoint is the last one.

### Row models

Three Pydantic models with `@computed_field` keys, following `TableItem`'s approach in
`base_items.py` — the key is derived from the fields rather than assembled at each call site.
`TableItem` itself does not fit: it assumes `pk = ENTITY#{id}`, `sk = '#META'`, and these rows have
composite sort keys.

```python
class CheckpointRow(BaseModel):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    checkpoint_type: str    # serde tag from dumps_typed()[0]
    checkpoint_value: bytes # serde payload  from dumps_typed()[1]
    metadata_type: str
    metadata_value: bytes

class BlobRow(BaseModel):
    thread_id: str
    checkpoint_ns: str
    channel: str
    version: str            # str() of the ChannelVersions value (int by default)
    value_type: str         # 'empty' when the channel had no value at this version
    value: bytes

class WriteRow(BaseModel):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    task_id: str
    task_path: str
    idx: int
    channel: str
    value_type: str
    value: bytes
```

`serde.dumps_typed(obj)` returns `(type_str, bytes)`; the pair is stored as two attributes, `S` and
`B`. `TypeSerializer` in `base_service.py:210` already maps `bytes` to `B` and `None` to `NULL`, so
no serialization work is added to the shared layer.

**The `bytes` half is zlib-compressed before it is stored, and decompressed on read.** The type tag
is left alone — it is what routes deserialization, and it is a short string. Two helpers on the
saver, `_pack(obj) -> (type, bytes)` and `_unpack(type, bytes) -> obj`, are the only places that
know about it, so every row type gets it uniformly. See
[Payload size](#payload-size-and-why-the-blob-is-compressed) for the measurements behind this.

`idx` is stored as a number attribute as well as being part of the sort key, because pending writes
have to come back in a deterministic order and `idx` can be negative (`WRITES_IDX_MAP` maps
`__error__`/`__scheduled__`/`__interrupt__`/`__resume__` to `-1`…`-4`), which does not sort
correctly as a string.

### Channel versions must be unique per write, not merely increasing

`get_next_version` is overridden to return `f'{counter:032}.{random.random():016}'` — upstream's
format, for upstream's reason. The base class answers `current + 1`, and a saver that keys a blob
row by `(channel, version)` cannot use that: two checkpoints forked from the same parent would both
call the channel version 2, land on one key, and the second write would overwrite the first. Reading
the first branch back would then return the other branch's state, silently.

The zero-padded counter keeps versions ordered; the random tail keeps siblings apart. Reading an
`int` is still handled, so threads written before this keep resolving.

## How each method works

### `aput`

LangGraph hands over the checkpoint, its metadata and `new_versions` — the channels that changed
during this superstep.

1. `values = checkpoint.copy().pop('channel_values')`.
2. For every `channel, version` in `new_versions`: a `BlobRow` holding `_pack(values[channel])`, or
   `('empty', b'')` when the channel has no value.
3. One `CheckpointRow` holding the checkpoint without its channel values, and
   `get_checkpoint_metadata(config, metadata)` — the helper that folds `config['metadata']` and
   `config['configurable']` into the stored metadata. Skipping it loses `run_id` and any custom
   config keys.
4. `parent_checkpoint_id = config['configurable'].get('checkpoint_id')` — the checkpoint being
   written *from* is the parent of the one being written.
5. All of it through `DynamoDBService.transact_write` as one `TransactWriteItems`.
6. Return `{'configurable': {'thread_id', 'checkpoint_ns', 'checkpoint_id': checkpoint['id']}}`.

Channels absent from `new_versions` are **not rewritten** — the new checkpoint's
`channel_versions` still points at their existing blob rows. That is the entire point of splitting
blobs out, and it is why a long conversation does not rewrite every channel on every superstep.

A transaction rather than `batch_write_item` so that a checkpoint row can never reference a blob
row whose write failed. The cost is 2× WCU and a 100-action ceiling; `new_versions` holds a handful
of channels, so the ceiling is not in reach.

### `aget_tuple`

Three round trips, each one precise:

1. **The checkpoint.** With `checkpoint_id` in the config, `get_item` on the exact key. Without it,
   `query(pk, begins_with(sk, f'CHECKPOINTS#{ns}#'), ascending=False, limit=1)`.
2. **The channel values.** Build `CHECKPOINT_BLOBS#{ns}#{channel}#{version}` keys from the
   checkpoint's `channel_versions` and fetch them with `batch_get_item`. Rows whose `value_type` is
   `'empty'` are skipped, matching `InMemorySaver._load_blobs`.
3. **The pending writes.** `query(pk, begins_with(sk, f'CHECKPOINT_WRITES#{ns}#{checkpoint_id}#'))`,
   sorted by `(task_path, task_id, idx)` before being returned.

Returns a `CheckpointTuple` with `parent_config` built from `parent_checkpoint_id`, or `None` when
the thread has no checkpoint yet.

Reading blobs by exact key matters: a `begins_with(sk, 'CHECKPOINT_BLOBS#')` query would drag back
every version ever written for the thread, and that set only grows.

### `aput_writes`

The de-duplication rule is copied from `InMemorySaver.put_writes` (line 499-509), because Pregel
depends on it: a task that is retried must not overwrite the write the first attempt already
committed, while a later error or interrupt must replace the earlier one.

- `idx = WRITES_IDX_MAP.get(channel, position_in_list)`.
- `idx >= 0` → `put_item` with `ConditionExpression=Attr('sk').not_exists()`; a
  `ConditionalCheckFailedException` means the write is already there and is swallowed.
- `idx < 0` → plain `put_item`, last write wins.

The writes of one call are independent, so they run concurrently in an `asyncio.TaskGroup`, each
task swallowing its own conditional failure.

Not a transaction: a transaction is all-or-nothing, and here a failed condition means *skip this
one item*, not *abandon the batch*.

### `alist`

Query the partition on `begins_with(sk, f'CHECKPOINTS#{ns}#')` — or `'CHECKPOINTS#'` when the config
names no namespace, matching `InMemorySaver.list`, which iterates every namespace of the thread.
With no config at all, fall back to a table `scan`; that path is for inspection and is documented
as such.

`before` filters on `checkpoint_id < before_id`, `limit` caps the result. `filter` is applied in
Python: metadata is stored as an opaque serde blob, so DynamoDB cannot filter on it server-side.

Each yielded tuple needs its blobs and writes, so listing N checkpoints costs 2N extra calls. That
is acceptable for a time-travel/debug path and is not on the request hot path.

## Payload size, and why the blob is compressed

DynamoDB caps an item at 400 KB. `messages` is one channel, so it is one row — splitting blobs out
does not divide that budget, it only keeps other channels out of the way. So the question is
whether a conversation fits, and the answer needed measuring rather than estimating.

Measured with the real `JsonPlusSerializer` and `tiktoken`, on non-repeating generated
conversations grown to the 8000-token point where `SummarizationMiddleware` fires
(`agent.py:27`):

| Shape at 8000 tokens | msgs | raw | zlib | ratio | raw, % of 400 KB |
| --- | ---: | ---: | ---: | ---: | ---: |
| short chat, English | 313 | 168 KB | 21 KB | 8.0× | 41% |
| short chat, Russian | 159 | 115 KB | 12 KB | 9.4× | 28% |
| turns with retriever results | 81 | 92 KB | 15 KB | 6.3× | 23% |

Bytes per token is **not** bytes per character. Plain text costs about 5 B/token in English and
7 B/token in Russian, but a serialized `AIMessage` carries roughly 250-300 bytes of fixed OpenAI
`response_metadata` — `token_usage`, `model_name`, `system_fingerprint`, `finish_reason`. That
overhead is per message, not per token, so the cost per token rises as messages get shorter: 7
B/token for long tool results, 21 B/token for ordinary answers, and 37 B/token for a chat of
one-line exchanges. In that last shape a 400 KB item holds only about **11,000 tokens**, and the
peak at the summarization trigger reaches ~71% of the limit.

So the earlier claim in this design — that the middleware keeps the channel "far below" the limit —
was wrong. Raw does fit, but in the worst measured shape the margin is about 1.4×, which is not a
margin worth shipping.

zlib closes it. The ratio is 6-9× even on non-repeating text, because what compresses is not the
prose but the structure: the same `response_metadata` keys repeat once per message. Worst measured
shape goes from 20.9 to 2.6 B/token, and a 400 KB item goes from ~19,600 to ~156,000 tokens.

The second argument is cost, and it is the larger one. DynamoDB bills writes per KB, rounded up: a
300 KB item costs 300 WCU **per write**, and `aput` runs once per superstep — roughly five times
per user message. Compressing to 20 KB turns ~1500 WCU per message into ~100.

Cost: one `zlib.compress` / `zlib.decompress` per row, on payloads of tens of KB — microseconds,
against a network round trip. The scripts behind the table are throwaway; the numbers are recorded
here so nobody has to re-derive them.

## Two additions to `DynamoDBService`

Both are generic, and neither is shaped around the checkpointer:

- **`batch_get(keys)`** — `batch_get_item` with the usual chunking at 100 keys per request and a
  retry loop over `UnprocessedKeys`, deserializing like the existing readers do. Needed to fetch
  channel blobs by exact key.
- **`query(..., ascending: bool = True)`** — a new keyword argument setting `ScanIndexForward`.
  Needed for "the newest checkpoint". Existing callers are unaffected by the default.

## Where the code lives

```
app/ai_assistant_langchain/checkpointer/__init__.py   exports DynamoDBCheckpointer, get_checkpointer
app/ai_assistant_langchain/checkpointer/items.py      the three row models
app/ai_assistant_langchain/checkpointer/saver.py      DynamoDBCheckpointer
```

Next to the agent that uses it, not in `app/core/dynamodb/`. The `core/dynamodb` package is
framework-agnostic table plumbing; a `BaseCheckpointSaver` subclass would drag a LangGraph import
into it. The generic pieces the saver needs go into `base_service.py`, where they belong; the
LangGraph adapter stays on the LangGraph side.

## Client lifetime

`get_dynamo_client` (`client.py:40`) opens a client per HTTP request. The checkpointer cannot use
it: it is held by an `@lru_cache`d graph and outlives every request.

```python
@lru_cache
def get_checkpointer() -> DynamoDBCheckpointer:
    return DynamoDBCheckpointer(get_aioboto_session(), get_settings())
```

The saver holds no lifecycle logic. One `opened()` async context manager keeps the client for the
block, and a `table` property raises a named error outside it:

```python
# app/main.py
async with get_checkpointer().opened():
    yield
```

### What was tried first, and why it went

The original design opened the client lazily on first use. The event-loop risk it named duly
happened on the first test run — `HTTPClientError: 'NoneType' object has no attribute 'get'` from
inside aiohttp, an error that says nothing about loops — and the fallback this document proposed,
opening a client per operation, was measured and rejected: **~20 ms per open** against a local
DynamoDB versus ~3.6 ms for a call on an open one, and a single turn reaches the checkpointer about
a dozen times.

The second attempt kept the lazy open and made it loop-aware: remember the loop, reopen on a
change, abandon what could not be closed. It worked, and it was wrong. Code review found that the
`asyncio.Lock` guarding it was *itself* loop-bound (`asyncio.Lock` binds on its first contended
acquire, and nothing rebuilt it), that `aclose()` silently discarded a live client when called from
the wrong loop, and that `_abandon_stale_client` leaked by design.

All three were symptoms of one decision. Opening lazily means the object cannot know who owns it or
which loop it will first be reached from, so it has to infer both at runtime — and every piece of
that inference was a place to be wrong. Opening explicitly, where the owner and the loop are known,
deletes the question: `_loop`, `_open_lock`, `_stack`, `_abandon_stale_client` and `aclose()` are
all gone, and the lifespan gets its `try/finally` from `asynccontextmanager` rather than from
remembering to write one.

Tests open it per test rather than relying on the app, which also sidesteps `ASGITransport` not
running the lifespan.

## Retention is out of scope for this round

The first draft of this design put a TTL on every row, reusing the `expires_at` mechanism
`appointment_slots` already has. **It does not work, and the reason is worth writing down.**

DynamoDB's TTL is per item. A channel that stops changing keeps the `expires_at` of the last blob
row written for it, while newer checkpoints go on referencing that row. A conversation that stays
alive past the TTL window therefore loses the blob for its least-active channel — and loses it
silently, because `_load_blobs` skips keys it cannot find. Refreshing every referenced blob on
every `aput` would fix it, at the cost of one extra write per unchanged channel per superstep,
which is 25-75 extra writes per user message. That is a bad trade.

So: **no TTL, and no cleanup in this round.** Superseded blob rows and old checkpoints accumulate.
This matches `PostgresSaver`, which also has no retention mechanism — upstream treats retention as
an operator concern. The follow-up, when it is needed, is thread-level rather than item-level:
`adelete_thread` for an explicit "clear my history", and a job that deletes whole threads by last
activity. Both belong to a later round, per the decision to start with the required methods only.

## Testing

The tests run against the local DynamoDB, not against `InMemorySaver`.

`tests/agent_stubs.py::get_test_agent()` builds the *production* graph with a stubbed model, so it
picks up the real checkpointer automatically — the existing suite becomes an end-to-end test of the
saver at no cost.

**Wiring:**

- `tests/conftest.py::pytest_configure` sets `DYNAMODB_CHECKPOINTS_TABLE = 'agent_checkpoints_test'`
  alongside the two table names already set there;
- `tests/db_utils.py::get_table_specs()` gains
  `'checkpoints': TableSpec(name=..., gsi_numbers=[])`. `table_definition` already declines to emit
  an empty `GlobalSecondaryIndexes` (line 371), so a GSI-less table needs no change there;
- `TestBaseCheckpointerClass` in `tests/api/test_chat_checkpointer.py` gains the session-scoped
  `tables` fixture so the table exists before the first request.

**`tests/api/test_chat_checkpointer.py` needs no other change.** Every test already posts under its
own `uuid4()` `user_id`, so threads cannot collide, and the table is dropped when the session ends.
The tests keep asserting exactly what they assert today — what the model was sent — and now prove
it against DynamoDB.

**New: `tests/dynamodb/test_checkpointer.py`**, driving the saver directly:

- `aput` → `aget_tuple` round trip: channel values, metadata and `parent_config` come back intact;
- a channel absent from `new_versions` is still resolved, from the blob an earlier `aput` wrote —
  the test that proves the split layout works;
- `aput_writes` de-duplication: the same `(task_id, idx)` written twice keeps the first value;
- a `__error__` write (negative index) replaces an earlier one;
- two thread ids do not see each other's checkpoints;
- `alist` returns newest first, honours `limit` and `before`;
- `aget_tuple` on an unknown thread returns `None`;
- a message list large enough to exceed 400 KB uncompressed round-trips intact — the test that
  proves compression is actually on the write path and not merely configured.

Each test builds its own `DynamoDBCheckpointer` and closes it, so the suite does not depend on the
cached singleton or on which loop opened it.

## Settings

```python
DYNAMODB_CHECKPOINTS_TABLE: str = 'agent_checkpoints'
```

Read directly wherever it is needed — no fallback onto `DYNAMODB_UNIVERSITY_TABLE` or any other
existing name.

`db/load_dynamodb.py` creates the table in `main()`: `recreate_table(T_CHECKPOINTS, gsi_numbers=[])`.
It is not seeded — the table starts empty and the app fills it.

## Documentation

- `Makefile`: a `test-checkpointer` target running `pytest tests/dynamodb/test_checkpointer.py -v`,
  with a comment noting it needs the local DynamoDB up;
- `README.md`: a section on conversation persistence — which table, what a thread is, and the fact
  that `docker compose up -d dynamodb` is now required to run the API tests.

## Known limits

| Limit | Consequence |
| --- | --- |
| 400 KB per DynamoDB item | one channel's value must fit. `messages` is the channel at risk. Compressed, the worst measured shape leaves ~20× headroom at the summarization trigger; raw it would have been ~1.4×. Exceeding it raises `ValidationException` — loud, not silent. See [Payload size](#payload-size-and-why-the-blob-is-compressed). |
| `batch_get_item`: 100 keys, 16 MB | chunked and retried on `UnprocessedKeys`. A graph with >100 channels is not in sight. |
| `transact_write_items`: 100 actions, 4 MB, 2× WCU | `new_versions` is a handful of channels per superstep. |
| One partition per thread | all of a user's writes land on one partition. A chat thread's write rate is nowhere near a partition's 1000 WCU/s. |
| No retention | rows accumulate. See [Retention](#retention-is-out-of-scope-for-this-round). |