# Streaming responses for the LangChain assistant (spec)

Design document. The backend half **is** implemented — `tests/api/test_chat_stream.py` is the suite
§9 asks for; the frontend (§8) is not. Every code block below is a sketch of the shape, not a patch
to apply, and where implementing it measured something this design had wrong, the row says so.

Two repositories are in scope:

| Repo | What changes |
|---|---|
| `crewai-university-assistant` (this one) | `app/ai_assistant_langchain/` — a streaming representation of the existing entry point |
| `universal-agent-chat` (`../universal-agent-chat`) | an opt-in flag, a `fetch`-based transport, a hook beside `useSendMessage` |

The CrewAI flow (`app/ai_assistant/`, `POST /ask`) is **not** touched.

---

## 1. One endpoint, two representations

**`POST /langchain-assistant` stays the only chat path.** Streaming is opt-in per request, through
the header HTTP already has for this: a client that sends `Accept: text/event-stream` gets SSE, and
a client that sends nothing gets byte for byte the JSON it gets today.

```python
SSE_MEDIA_TYPE = 'text/event-stream'

@router.post(
    '/langchain-assistant',
    response_model=ChatSchemaOut,
    responses={200: {'content': {SSE_MEDIA_TYPE: {'schema': {'type': 'string'}}}}},
)
async def chat_with_user(
    input_data: UserQuery,
    request: Request,
    service: ChatService = Depends(),
) -> ChatSchemaOut | EventSourceResponse:
    if SSE_MEDIA_TYPE not in request.headers.get('accept', ''):
        return await service.process_turn(input_data)

    await service.assert_can_take(input_data)   # ConflictingStatusError -> 409, see §6.2
    return EventSourceResponse(service.stream_turn(input_data))
```

The sketch's two calls became three when it was implemented: `start_streaming_turn(query)` runs the guards
*and* creates the detached run with no `await` between them, and `stream_events(queue)` is what the
response iterates. `assert_can_take` followed by an `await` and only then a task lets two requests
that arrive together both pass the check and put two runs on one thread — §7.3's failure, inside the
guard meant to prevent it.

All of that is measured, not assumed — **P5** in §3. It holds because the JSON branch is the
default: nothing that exists today sends that header, so nothing that exists today changes.

- `tests/api/test_chat_checkpointer.py`, `tests/api/test_chat_history.py` — unchanged;
- the six eval suites that post whole conversations and read `status` / `pending` off the body
  (`test_agent_eval`, `test_agent_multiturn_eval`, `test_booking_multiturn_eval`,
  `test_tool_routing_eval`, `test_safety_eval`, `test_retriever_eval`) — unchanged;
- `README.md:216-220`, `Makefile:52`, `docs/manual-testing.md` `curl` scripts — unchanged;
- `universal-agent-chat`'s `decide`, which posts `decisions` to the same path — unchanged, and this
  is the case the header handles better than a second path would. `send` and `decide` are literally
  one route in `.env.example:21`. Per-request negotiation lets `send` stream while `decide` does
  not; a `/stream` path would have had to fork that config to keep them apart.

`response_model=ChatSchemaOut` still documents the JSON shape in `/docs` — the explicit parameter
takes priority over the union return annotation, so FastAPI does not try to build a response field
out of `EventSourceResponse`. The `responses=` entry advertises the SSE alternative on the same
operation. Both verified in P5(a); the whole handler type-checks under `mypy --strict`.

Two things to carry:

- **No `Vary: Accept`, on purpose.** One URL has two representations, and RFC 9110 says an origin
  SHOULD send `Vary` for that — but here it would change nothing. No browser or CDN caches a POST
  response, and the two refusals are neither heuristically cacheable nor different by `Accept`: a
  409 is JSON whatever was asked (P5d), and so is a 422 (P5e). A GET that negotiates on `Accept`
  would need it — set it in that route then.
- **The exception is `fastapi.sse`.** Its `EventSourceResponse` is a *marker* — `routing.py:392`
  switches on the declared `response_class`, and the encoding only runs for a path operation that is
  itself an async generator. A generator endpoint cannot choose to return JSON instead, so it cannot
  serve both. Use `sse_starlette` (§5.1); see **P6**.

`decisions` stream too, as soon as a caller asks: the endpoint takes the same `UserQuery`
(`query` XOR `decisions`) it always did, and both branches go through the same producer. Whether the
*frontend* opts in for `decide` is a separate question — §8.1.

---

## 2. What already exists

| Piece | Where | Note |
|---|---|---|
| Main graph: router → `qa` / `booking`, one checkpointer | `main_graph.py:68` | streams as-is; nothing to restructure |
| Router LLM call inside a `START` conditional edge | `main_graph.py:52` | **leaks into the token stream — see §4** |
| `SummarizationMiddleware(trigger=('tokens', 8000))` on the QA agent | `agent.py:42` | **leaks into the token stream — see §4** |
| `compare_programs`, `return_direct=True` | `tools.py:134` (`DIRECT_ANSWER_TOOLS`) | its own graph's `merge` call **streams tokens one namespace deeper**, and they are the answer — see §4.4 |
| `HumanInTheLoopMiddleware` interrupts | `agent.py` | reachable mid-stream — see §6 |
| `_to_chat_out`, `_get_failed_tool`, `_describe_pause` | `service.py:99-137` | **reused verbatim** — see §5.3 |
| `_refuse_while_paused` → `ConflictingStatusError` → 409 | `agent_service.py:101` | must run *before* the response starts — see §6.2 |
| `record_turn` / `record_resumed_turn`, one `TransactWriteItems` | `history/service.py` | unchanged, and still one transaction; it moves into a detached task — see §7 |
| `lifespan` holding the checkpointer's client open | `lifespan.py:18` | gains the run drain, nested inside it — see §7.2 |
| `sse-starlette 3.4.5` | `uv.lock` | installed, but only **transitively, via `mcp 1.26.0`** — must become a direct dependency, see §5.1 |
| `fastapi 0.139.0` | `uv.lock` | ships `fastapi.sse`, which this design does **not** use — see §1 and P6 |

---

## 3. Measurements

Everything in §4 follows from four probes run against this project's own venv
(`langgraph 1.2.10`, `langchain 1.3.14`, `langchain-core 1.5.3`, `fastapi 0.139.0`). **Reproduce them
before trusting §4** — they are the load-bearing part of this design, and a LangGraph upgrade can move
any of it. P5 and P6 are the same for §1 and §5.4: they are what makes one endpoint possible, and they
are why the transport is `sse_starlette`.

P1–P4 were first taken on graphs shaped like `main_graph.py`, then re-run on 2026-09-11 against the
real thing: `build_main_graph()` with only its checkpointer swapped for `InMemorySaver`, the real
model (`gpt-4o-mini`), Qdrant and the dev slots. Six scenarios — a greeting with and without
`subgraphs=True`, a `retriever` question, a `compare_programs` question, a booking that pauses, and a
turn after 24 long messages of history — plus the booking once more without `updates`. The re-run
corrected the `return_direct` row, added two, and extended most of the others.

The script that took them was throwaway by design and has been deleted, so **this table is the
artifact** — there is no file to re-run. Rechecking after a LangGraph or LangChain bump means
writing it again, and the paragraph above is what makes that cheap: that same graph and saver,
`astream` with the arguments named below, and those six scenarios. The suite cannot stand in for it.
§9's leak tests
drive a stub model, so they catch a filter that stopped working; they cannot catch a stream whose
shape changed underneath the filters, which is exactly what an upgrade moves.

With `stream_mode=['messages', 'updates', 'values'], subgraphs=True, version='v2'`, each part is a
dict `{'type', 'ns', 'data'}`, and a `values` part also carries `interrupts`:

| # | Question | Measured answer |
|---|---|---|
| P1 | Do the QA/booking subagents' tokens reach the parent stream? | Only with `subgraphs=True`. `ns == ('qa:<task_id>',)`, metadata `langgraph_node == 'model'`. Without it the answer still arrives, but once and whole: an `AIMessage` at `ns == ()` under `langgraph_node == 'qa'`. |
| P2 | Does the `START` router's LLM call appear? | **Yes, on every turn.** `ns == ()`, `langgraph_node == '__start__'`, content is the raw routing JSON: `{"destination":"booking"}`. |
| P2 | Where does a HITL interrupt appear? | In `updates`, **twice** — once at the subgraph `ns`, once at `ns == ()` — as `{'__interrupt__': (Interrupt(...),)}`. **And on `values`:** the subgraph's and the root's last `values` parts carry it in `interrupts`, one each. On a turn that does not pause, `interrupts` is empty. |
| P2 | With `updates` left out of `stream_mode`, does the root `values` still carry the pause? | **Yes.** With `['messages', 'values']` the last part is still the root `values`, its `interrupts` holds the pause, and no `updates` arrive at all. §5.2 streams without them. |
| P3 | What message types arrive on `messages`? | `AIMessageChunk`, **and also whole `ToolMessage`s** (a retriever result arrives complete, in one part, under `langgraph_node == 'tools'`), `RemoveMessage`, `HumanMessage`. The last two came only from the summariser (P4); the applicant's own message is not echoed. |
| P3 | What does a `return_direct` tool stream? | The tool call as **empty** `AIMessageChunk`s under `model` — 15 for `compare_programs`, as many as its arguments take, not a fixed two — then the work of the tool's own graph (next row), then the answer as a single **`ToolMessage`** under `tools`. |
| P3 | Does a graph that a tool invokes stream? | **Yes.** `compare_programs` runs its own graph, and its `merge` model call streamed 183 `AIMessageChunk`s at `ns == ('qa:<task_id>', 'tools:<task_id>')`, `langgraph_node == 'merge'` — the comparison being written, the text the `ToolMessage` then carries. An earlier version of this table said "no assistant tokens at all"; that was wrong. These are the tokens §4.4 forwards; re-measured on 2026-09-20 the same call streamed 210 chunks. See §4.1 and §4.4. |
| P4 | What node does `SummarizationMiddleware`'s own model call run under? | `langgraph_node == 'SummarizationMiddleware.before_model'`, same `ns`, **as ordinary `AIMessageChunk`s carrying the summary text**, followed by a `RemoveMessage` and the summary as a `HumanMessage` from the same node. It is checked before every model call, not once per turn: here the 20 messages it keeps were still past the trigger, so a turn that called `retriever` summarised twice. |
| P2/P3 | Is there a final root-level `values` part? | **Yes, always, and last** — on the plain path *and* on the interrupt path. Its `messages` equals the thread's state, which is what `ainvoke` returns today. On a pause that is only the applicant's message: the subagent's tool call has not reached the parent yet. |

P5 and P6 come from one script (not committed, ~90 lines): the §1 route over a `ChatSchemaOut`
stand-in, driven through `httpx.ASGITransport` — four requests and a read of `/openapi.json`.

| # | Question | Measured answer |
|---|---|---|
| P5a | Does `response_model` survive a `ChatSchemaOut \| EventSourceResponse` return annotation? | **Yes.** `/openapi.json` carries both `application/json` → `$ref: ChatSchemaOut` and `text/event-stream`. `mypy --strict` clean. |
| P5b | What does a client that sets no `Accept` get? (httpx sends `*/*`) | `200 application/json`, the unchanged body. The substring test never matches `*/*`. |
| P5c | What does `Accept: text/event-stream` get? | `200 text/event-stream; charset=utf-8`, `X-Accel-Buffering: no`, `Cache-Control: **no-store**` (not `no-cache` — sse-starlette's choice). Events arrive `event:` then `data:`. |
| P5d | Does the pre-stream guard still produce a real status code? | **Yes** — `409 application/json` with a normal `{"detail": …}` body, *even when* the request asked for SSE. |
| P5e | Does request validation still land first? | **Yes** — `422 application/json`, unchanged. |
| P5 | Is `Vary: Accept` set by either branch? | **No** — and left unset on purpose, §1. |
| P6 | Can `fastapi.sse.EventSourceResponse` be returned from a plain `async def`? | **No.** It is a marker class; `routing.py:392` gates encoding on the declared `response_class`, and only wraps a generator endpoint. Returning `EventSourceResponse(agen)` yields `AttributeError: 'ServerSentEvent' object has no attribute 'encode'`. |
| P6 | Does `sse_starlette.ServerSentEvent` JSON-encode a pydantic `data`? | **No** — it does `str(self.data)`. Pass `model_dump_json()` yourself; see §5.4. |
| P7 | Does a detached producer survive the client hanging up? | **Yes.** Client reads one token and drops; the generator's `finally` runs, the producer runs to completion and writes history. `_RUNS` is empty afterwards — nothing leaked. Note what this does *not* license: through `httpx.ASGITransport` a client cannot hang up mid-turn at all, so the suite pins the property at the queue instead — see the note under §9's table. |
| P8 | Can a cancelled task still `await` a write in `finally`? | **Yes, once.** CancelledError is delivered at the suspension point; the next await runs normally. A *second* cancel needs `asyncio.shield`. A lifespan drain avoids the question — §7.2. |

---

## 4. The filters

A **subagent** token is forwarded only if all three of §4.1-§4.3 hold. Each one exists because of a
specific measured leak; dropping any of them ships a bug that passes the current tests. §4.4 is the
one path that does not come from a subagent at all — a `return_direct` tool writing the answer in
its own graph — and it has a rule of its own.

```python
MODEL_NODE = 'model'   # what `create_agent` names its model node — asserted by a test, see §9
SUBAGENT_DEPTH = 1

class BaseChatService:
    ...

    @staticmethod
    def _is_answer_token(part: MessagesStreamPart) -> bool:
        chunk, metadata = part['data']
        return (
            len(part['ns']) == SUBAGENT_DEPTH                 # 4.1 — a subagent: not the router, not a tool's graph
            and metadata.get('langgraph_node') == MODEL_NODE  # 4.2 — not the summariser
            and isinstance(chunk, AIMessageChunk)             # 4.3 — not a tool result
            and bool(chunk.text)                              # `.text`, not `.content` — see below
        )
```

The filters are static methods of `BaseChatService`, next to `_forward_tokens` which is their only
caller — not module-level functions. They need no instance, so §9's filter tests call them on the
class (`ChatService._is_answer_token(part)`) with no graph and no fixtures. The constants stay at
module level.

The last condition reads `chunk.text` because that is what goes on the wire. The two disagree when a
model answers in content blocks (reasoning, tool use): `content` is a non-empty list while `text`
extracts nothing, so a filter on `content` sends `{"text": ""}` — a token that appends nothing.

### 4.1 `len(ns) == 1` — the router, and graphs inside tools

`route()` is a conditional-edge function on `START`, so its `with_structured_output` call streams at
the **root** namespace under `langgraph_node == '__start__'`. Unfiltered, every answer in the chat
would open with `{"destination": "qa"}`. This is not a rare edge: it happens on **every single
turn**, so it will be the first thing anyone sees.

The other end of the namespace leaks too. A graph that a tool invokes runs one level *below* the
subagent, and it streams (P3): `compare_programs`' `merge` call arrives at
`('qa:<task_id>', 'tools:<task_id>')`. An earlier draft of this filter said `ns != ()`, which lets that
through — the only thing that dropped those tokens was 4.2's node name, and only because the
comparison graph happens to call its node `merge`. A tool that runs a `create_agent` of its own has a
node called `model`, and its tokens would land in the bubble ahead of the answer — or, for a
`return_direct` tool, as well as it. Depth one is what "a subagent's answer" means here; the filter
should say that, not "anything but the root".

There is exactly one exception, and it is named rather than implied: `compare_programs`' `merge`
node, whose text *is* the answer — §4.4.

### 4.4 `merge` at depth two — a `return_direct` answer as it is written

`compare_programs` is `return_direct`, so the text its graph writes *is* the answer. Waiting for the
whole `ToolMessage` leaves the applicant on the typing dots through two searches and a full
generation — measured on the real agent: the first token would otherwise have arrived 1.6s before
`done` instead of at the same moment, on a 1030-character comparison of 210 chunks.

```python
TOOL_GRAPH_DEPTH = 2
DIRECT_ANSWER_NODE = MERGE   # imported from graphs/compare_programs/graph.py, not spelled again

    @staticmethod
    def _is_direct_answer_token(part: MessagesStreamPart) -> bool:
        chunk, metadata = part['data']
        return (
            len(part['ns']) == TOOL_GRAPH_DEPTH
            and metadata.get('langgraph_node') == DIRECT_ANSWER_NODE
            and isinstance(chunk, AIMessageChunk)
            and bool(chunk.text)
        )
```

Two things keep this from becoming "anything a tool streams":

- **The node is named**, so a tool that runs a `create_agent` of its own — node `model`, depth two —
  still streams nothing. That is §4.1's warning, and it stays covered.
- **The `ToolMessage` is dropped once its text has been streamed.** It carries the same comparison
  again; forwarded as well, the applicant reads it twice. `_forward_tokens` holds one flag for that
  and clears it on each `return_direct` result, so a second such call in one turn is judged on its
  own. When the graph streamed nothing — a failure before `merge`, or that node renamed under us —
  the flag is false and §4.3's whole event goes out as before. **The fallback is the point**: a
  rename upstream costs the trickle, not the answer.

A failure *after* tokens have gone out is already covered by §5.4's rule — `done` replaces the
bubble with `TOOL_FAILURE_MESSAGE`, discarding what was streamed.

### 4.2 `langgraph_node == 'model'` — the summariser

This is the trap. `SummarizationMiddleware` calls the same chat model, inside the same subgraph
namespace, and its output arrives as ordinary `AIMessageChunk`s. A filter built only on `ns` and
`AIMessageChunk` splices *"Here is a summary of the conversation so far…"* into the applicant's
answer.

It fires only past `trigger=('tokens', 8000)` — long conversations — so it passes every test in
`tests/api/`, passes a manual smoke test, and shows up in front of a real applicant. Past the trigger
it is not once per turn, either: it is checked before every model call, and in P4 a turn that called
`retriever` streamed a summary twice before the answer began. Filter on the node name, and cover it
with the regression test in §9.

### 4.3 `isinstance(chunk, AIMessageChunk)` — tool results

`messages` mode carries `ToolMessage`s too, whole. Forwarded blindly, the retriever's raw passages
land in the bubble ahead of the answer written from them.

**But `compare_programs` is `return_direct`, and its `ToolMessage` *is* the answer.** So there is one
deliberate exception, kept narrow by reusing the frozenset the router context already uses:

```python
class BaseChatService:
    ...

    @staticmethod
    def _is_direct_answer(part: MessagesStreamPart) -> bool:
        chunk, _ = part['data']
        return (
            len(part['ns']) == SUBAGENT_DEPTH    # 4.1 holds here too — see below
            and isinstance(chunk, ToolMessage)
            and chunk.name in DIRECT_ANSWER_TOOLS
            and chunk.status != 'error'          # a failed call is not an answer — see below
        )
```

The depth check is 4.1's rule, and it belongs here for the same reason: "this turn's answer" means a
subagent, one namespace down. Today `compare_programs` is reachable only from `ASSISTANT_TOOLS`, so it
is always at depth one and the check never fires. The moment a `return_direct` tool is reachable from
a graph nested inside another tool, its `ToolMessage` would otherwise go out as the *outer* turn's
answer.

The `ToolMessage` is the fallback rather than the normal path: §4.4 streams the same text as the
tool writes it, and drops this event when it did. It still arrives whole — as **one large token
event** — whenever the tool's graph streamed nothing this service recognises, so the client must not
assume tokens are small.

`status != 'error'` is not defensive padding, and the first implementation shipped without it. A
tool that raises still produces a `ToolMessage` **named after the tool it tried**
(`langgraph/prebuilt/tool_node.py`), so the name check cannot tell a result from a failure, and what
a failure carries is the exception text — for `compare_programs`, pydantic's `programs: Field
required`. `_to_chat_out` already replaces that turn with `TOOL_FAILURE_MESSAGE` (§5.4), so without
the status check the applicant reads the raw validation error first and the apology for it second —
the exact thing `TOOL_FAILURE_MESSAGE` exists to prevent. Assert on the event *names*, not only on
`done`, or the test passes while the leak ships.

That is a choice, not a limit. An earlier draft said there was no partial answer to show, and P3
disproves it: the comparison is written token by token under `merge`, one namespace down, and 4.1
drops it. Forwarding it would take a second exception, keyed on a node name inside one tool's graph,
plus a rule for skipping the `ToolMessage` that then repeats the same text — except when the tool
changes it on the way out, which it does when the applicant names more than `MAX_PROGRAMS`. One
whole event keeps 4.1 a single rule. Whether the wait is worth that is open — §13.

---

## 5. Backend

### 5.1 Files

```
app/ai_assistant_langchain/
  agent_service.py   EDIT  + stream_agent() / stream_resume() beside run_agent() / resume_agent();
                           assert_can_take(); _build_hitl_decisions() factored out of resume_agent()
  service.py         EDIT  + start_streaming_turn() / stream_events() / _run_streaming_turn() beside process_turn() (§1);
                           the §4 filters and _event() as static methods of BaseChatService
  stream_schemas.py  NEW   the SSE event models (§5.4)
  runs.py            NEW   the _RUNS task registry, its per-user guard and wait_for_streaming_turns() (§7.2-§7.3)
  router.py          EDIT  chat_with_user() grows the Accept branch (§1) — no new route
app/lifespan.py      EDIT  drain pending runs, inside the checkpointer context (§7.2)
pyproject.toml       EDIT  + "sse-starlette>=3.4.5"   (in the lock today only via mcp; see §2)
Makefile             EDIT  + a manual-check target (§10)
README.md            EDIT  + a "Streaming" section (§10)
docs/manual-testing.md EDIT + a curl -N script (§10)
tests/api/test_chat_stream.py  NEW  (§9)
```

The dependency line is the one easy thing to get wrong: `sse-starlette` resolves today only because
`crewai` pulls in `mcp`, which depends on it. Importing it without declaring it means a `crewai`
bump can uninstall the transport. `fastapi` needs **no** bump — `fastapi.sse` is not used (P6).

Nothing is rewritten in place: `run_agent`/`resume_agent`/`process_turn` keep working and keep their
tests. The streaming methods are siblings, and the JSON branch of the route calls exactly the
`process_turn` it calls today.

### 5.2 `AgentService` — two new methods

```python
async def stream_agent(self, messages: list[BaseMessage], user_id: str) -> AsyncIterator[StreamPart]:
  await self.refuse_while_paused(user_id)
  async for part in self._graph.astream(
          {'messages': messages},
          config=self._config(user_id),
          context=CustomContext(user_id=user_id),
          stream_mode=['messages', 'values'],  # no `updates`: the pause is read off `values` — P2, §5.3
          subgraphs=True,  # without it a subagent's answer arrives once, whole, not as tokens — P1
          version='v2',
  ):
    yield part
```

`stream_resume` is the same shape over `Command(resume={'decisions': hitl_decisions})`, and reuses
the whole of `resume_agent`'s validation (count, `allowed_decisions`, `_get_hitl_decision`)
unchanged — factor that block into a `_build_hitl_decisions(decisions, request)` helper both call,
rather than copying it. Approving a write against half-checked decisions is exactly the failure
`resume_agent`'s docstring is about.

### 5.3 The turn body — tokens now, the same `ChatSchemaOut` at the end

The point of the design: **the streaming path reconstructs the identical `AgentResponse` the JSON
path receives**, and then calls the existing `_to_chat_out`. Nothing about approvals, tool failures
or history is re-derived, so the two paths cannot drift.

The sketch below is written as one flat generator to keep that idea readable. It is **not** the
final shape: §7 splits it into a detached producer and a formatting generator, and every line after
the loop belongs to the producer. Read §7.1 before implementing it.

```python
async def stream_turn(self, query: UserQuery) -> AsyncIterator[ServerSentEvent]:
    ...
    last_values: ValuesStreamPart[MainState] | None = None

    async for part in parts:
        if part['type'] == 'values' and part['ns'] == ():
            last_values = part                    # P2/P3: a root `values` always arrives, last — on a pause too
        elif part['type'] == 'messages':
            if self._is_answer_token(part) or self._is_direct_answer(part):
                yield ServerSentEvent(
                    data=TokenEvent(text=part['data'][0].text).model_dump_json(),  # P6
                    event='token',
                )

    if last_values is None:                       # P2/P3 say it cannot happen; fail loudly if it does
        raise IncompleteStreamError('The graph ended without a root `values` part.')

    interrupts = list(last_values['interrupts'])  # P2: the pause, if there is one
    response = cast(AgentResponse, {**last_values['data'], **({'__interrupt__': interrupts} if interrupts else {})})
    result = self._to_chat_out(response)          # unchanged — §2
    if self._get_failed_tool(response) is None:
        await self._history_service.record_turn(user_id, message, asked_at, result)

    yield ServerSentEvent(
        data=result.model_dump_json(),
        event='pending' if result.status == 'pending_approval' else 'done',
    )
```

The pause comes off the same part as the messages. A v2 `values` part carries `interrupts` beside
`data`, and on a pause the root one still arrives last with the interrupt in it (P2) — so nothing
needs catching in `updates`, and §5.2 no longer streams them. `__interrupt__` is put back by hand
because that is the key `ainvoke` returns it under, and the one `_to_chat_out` reads.

`asked_at = utc_now()` is still taken **before** the graph runs, for the reason
`process_user_query` already documents: it is what files a slow turn where the applicant asked it.

### 5.4 The wire protocol

The event models are ordinary pydantic — document their fields with `Field(description=...)`, as
everywhere else here. **Serialize them yourself**: `sse_starlette.ServerSentEvent` writes
`str(self.data)` into `data:`, so handing it a model puts a Python repr on the wire. Always
`data=model.model_dump_json()` (P6). The code does that in exactly one place — `_event(payload, name)`,
a static method of `BaseChatService` that every token, `done`, `pending` and `error` event goes
through — so no call site can forget it. (The sketches in §5.3 and §7.1 spell the call out inline to
stay readable.) This is the one place where `fastapi.sse` would have been more convenient, and the
reason it cannot be used is §1, not this.

| `event:` | `data:` | When |
|---|---|---|
| `token` | `{"text": "…"}` | zero or more, in order |
| `done` | the full `ChatSchemaOut` (`status: "answer"`) | the turn finished |
| `pending` | the full `ChatSchemaOut` (`status: "pending_approval"`, `pending: [...]`) | the graph paused on an approval |
| `error` | `{"detail": "…"}` | the turn failed after the response had already started (§6.3) |

**The contract in one line: tokens are a preview, `done`/`pending` is the truth.** A client renders
tokens as they arrive and then *replaces* the bubble's text with `data.message` on `done`.

That is not belt-and-braces, it is what makes three existing behaviours work without a special case
each:

- a **tool failure** replaces the answer wholesale with `TOOL_FAILURE_MESSAGE` (`service.py:28`),
  discarding whatever tokens the model had already produced;
- a **`return_direct`** answer is not the subagent's tokens at all — it is the tool's own graph
  writing, streamed from depth two (§4.4), or the whole `ToolMessage` as one token when it did not
  (§4.3);
- a **paused** turn ends with a sentence addressed to a reviewer, not an answer, plus the `pending`
  array the approval card is drawn from.

`done.message` is also, byte for byte, what `record_turn` stored — so a reload replays exactly what
was on screen.

No `[DONE]` sentinel and no reconnection ids: this stream is one turn of one thread, and a client
that lost it re-asks rather than resumes. `EventSourceResponse` sends a keep-alive comment on its
own and sets `Cache-Control: no-store` and `X-Accel-Buffering: no` (P5c). `Vary: Accept` it does
not set, and neither does the route — §1 says why.

---

## 6. Interrupts, guards and errors

### 6.1 An interrupt mid-stream

Nothing special is needed. P2 shows the root `values` still arriving last on a pause, with the
interrupt in its `interrupts`, so `_to_chat_out` takes its usual `pending_approval` branch. The client
gets whatever tokens the model produced before the tool call (usually none) and then a `pending`
event carrying the same `pending[]` array the JSON endpoint returns — the approval card renders from
an unchanged shape.

### 6.2 The paused-thread guard must run before the first byte

`_refuse_while_paused` raises `ConflictingStatusError`, which `exception_handler.py` turns into a
409. **Once an `EventSourceResponse` has started, the status line is already `200` and cannot be
taken back.**

So the path operation must be a plain `async def` that awaits the guard and *then* returns the
streaming response — the shape in §1. That is also the constraint that rules out `fastapi.sse`
entirely: its encoding only runs for a generator endpoint (P6), and a generator has no "before the
first byte" to put a guard in.

P5d confirms it end to end: a request carrying `Accept: text/event-stream` that trips the guard
comes back `409 application/json`, body and all. The header asks for a stream; it does not promise
one.

**The same applies to `decisions`, and it is easy to miss.** A `decisions` turn is not refused by the
paused guard — a pause is what it exists to answer — but its own checks (one verdict per action, and
a verdict the pause accepts) are just as capable of rejecting the request, and they raise the same
`ConflictingStatusError`. Run them before the response starts too, or every reviewer mistake becomes
a 200 carrying "something went wrong" instead of a 409 saying what to send instead, and a routine
error is logged as a traceback from inside a detached task.

This costs one extra `aget_state` on the streaming path (the guard reads it, `stream_agent` reads it
again). Cheap, and the alternative — passing the already-read state down — couples the two for no
real gain. The JSON branch does not pay it at all: `process_turn` reaches the same guard inside
`run_agent` exactly as it does today. A 422 from `UserQuery`'s validators still lands before any of
this, unchanged (P5e).

### 6.3 Failures after the first byte

An OpenAI timeout, a Qdrant outage, a DynamoDB write that fails — after tokens have flowed there is
no status code left to send. Emit `event: error` with a `detail` and stop. The client keeps the
partial bubble and marks it failed, reusing the existing "Not delivered / Retry" affordance in
`MessageBubble`.

Log it at `logger.exception` on the way past; do not put a traceback in `detail`. Applicants read
`detail`, and `TOOL_FAILURE_MESSAGE` exists for exactly that reason.

The `except` that does this lives in the **producer** (§7.1), not the generator — same reason
`record_turn` does. A failure after the client has already gone still has to be logged and still has
to release the run; there is just nobody left to hand the `error` event to.

---

## 7. History and client disconnect — **decided: B, detach the run**

`record_turn` runs after the stream ends. If the browser goes away mid-turn and the run is owned by
the request, the SSE generator is closed, `record_turn` never runs — but LangGraph has already
checkpointed the turn. The result is a thread whose agent remembers a question the transcript does
not show.

| | What happens on disconnect | Cost |
|---|---|---|
| **A. Accept it** | Turn is lost from `conversation_history`, kept in `agent_checkpoints`. The two tables disagree until the next turn. | Zero work. Documented drift. |
| **B. Detach the run** ✅ | The graph runs in an `asyncio.Task` feeding an `asyncio.Queue` the generator drains. Disconnect closes the generator; the task finishes and writes history. | One task, one queue, a task registry and a shutdown drain — §7.1-§7.3. |
| **C. Write the question first** | Two writes instead of one `TransactWriteItems`. | Rejected — it reintroduces exactly the half-written turn the current design calls out. |

**B.** It keeps the single-transaction invariant *and* the property the codebase already argues for
in `_refuse_while_paused`: a state nobody can recover from is worse than a slow one. The run is
bounded already — the model has `timeout=30, max_retries=2` (`agent_model.py`).

P7 confirms the core claim on this stack: a client that reads one token and hangs up leaves the
producer running, and history is written anyway. No task leaked afterwards.

### 7.1 The split — everything that decides state lives in the producer

The generator must be a *formatter and nothing else*. Every step that decides what is true about the
turn — assembling `last_values`, `_to_chat_out`, `_get_failed_tool`, `record_turn` — runs in the
producer, because the producer is the half that survives a disconnect. If `_to_chat_out` runs in the
generator, a disconnected turn is unrecorded again and B has bought nothing.

```python
_RUNS: set[asyncio.Task[None]] = set()   # module level, see §7.2

async def _run_streaming_turn(self, queue: asyncio.Queue[ServerSentEvent | None], query: UserQuery) -> None:
    try:
        ...                                      # §5.3's loop, unchanged
        for event in ...:                        # filtered tokens
            queue.put_nowait(event)
        result = self._to_chat_out(response)     # decided here, not in the generator
        if self._get_failed_tool(response) is None:
            await self._history_service.record_turn(user_id, message, asked_at, result)
        queue.put_nowait(ServerSentEvent(data=result.model_dump_json(), event=...))
    except Exception:
        logger.exception('Streaming turn failed for user=%s', query.user_id)
        queue.put_nowait(ServerSentEvent(data=ErrorEvent(...).model_dump_json(), event='error'))
    finally:
        queue.put_nowait(None)                   # the sentinel, always

async def stream_turn(self, query: UserQuery) -> AsyncIterator[ServerSentEvent]:
    queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()
    task = asyncio.create_task(self._run_streaming_turn(queue, query))
    _RUNS.add(task)
    task.add_done_callback(_RUNS.discard)
    while (event := await queue.get()) is not None:
        yield event
```

Three things in that sketch are load-bearing and easy to drop:

- **The queue is unbounded** (`asyncio.Queue()`, no `maxsize`). A bounded queue makes `put` block,
  and the consumer that would unblock it is exactly the one that just disconnected — the producer
  would hang forever and never write history, which is the bug B exists to fix. Unbounded is safe
  here because a turn is bounded by the model's own `timeout=30, max_retries=2`; it is a few
  thousand short strings, not a firehose.
- **`_RUNS` holds a strong reference.** CPython keeps only a weak reference to a running task, so a
  bare `asyncio.create_task(...)` whose result nobody holds can be garbage-collected mid-run. This
  is the classic form of this bug and it is silent.
- **No `TaskGroup` here.** The house rule prefers `asyncio.TaskGroup` over `gather`, and this is the
  one place it must not be used: a `TaskGroup` awaits its children on exit, which re-couples the run
  to the request and undoes B. Say so in a comment, or someone will "fix" it.

### 7.2 Shutdown

A detached task has no owner, so process shutdown is where it can still be lost. Measured (P8):

| | Result |
|---|---|
| a single `task.cancel()`, then `await` inside `finally` | **the write still lands** — CancelledError is delivered once, at the suspension point, so the next await runs normally |
| the same, wrapped in `asyncio.shield` | lands, and survives a *second* cancel |
| a lifespan handler that waits on `_RUNS` instead of cancelling | lands, deterministically |

Take the third: a shutdown hook that does `await asyncio.wait(_RUNS, timeout=...)` before the app
goes down. It does not depend on cancellation semantics, and it is the only one that also covers the
case where the loop closes without cancelling anything. `asyncio.shield` around `record_turn` is a
reasonable belt to add; the first row means a plain `finally` is not as fragile as it looks, but it
is not something to rely on.

It goes in `app/lifespan.py`, and **nesting order is not free**: a pending run still needs the
checkpointer and the history table, so the drain must happen *inside* `get_checkpointer().opened()`,
not after it.

```python
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    async with get_checkpointer().opened():
        try:
            yield
        finally:
            await wait_for_streaming_turns(timeout=...)   # the checkpointer is still open here
```

The bound matters, and it is not the model's. Whatever the drain is waiting for, the platform kills
the process at *its* grace period — `docker stop` gives 10s, Kubernetes 30s by default — so a
timeout above that is fiction: the process dies mid-drain and even the "rows are lost" log never
runs. Pick a value under the grace period, make it a **setting** rather than a constant so each
deployment can match its own, and log what was still pending when it expires. The model's ceiling
(~30s × `max_retries`) is higher than either number, so a slow turn can still be cut short; buying
it more time means raising the platform's grace period first and the setting second.

### 7.3 The new failure mode B introduces

Detaching the run means a request no longer holds the thread for the duration of the turn. An
applicant who disconnects and immediately re-asks now has **two runs writing to one thread's
checkpoint at once**. `_refuse_while_paused` does not catch it — the thread is not paused, it is
busy.

The guard therefore belongs to the **thread**, not to the transport that asked. An earlier version
of this section said this was a state the JSON endpoint could not reach, and the implementation
followed it — the check sat on the streaming branch alone. It is reachable from either side, and the
reachable case is the ordinary one: the applicant's stream dropped, nothing is on screen, and the
client they re-ask from sends no `Accept` header at all. So `process_turn` claims the thread for the
length of its turn as well, and both branches consult one registry.

Guard it out of the same registry: key `_RUNS` by `user_id` and refuse a second turn for a user who
already has one in flight, with the same `ConflictingStatusError` → 409 the paused guard uses, from
the same place in the route (§6.2), so it too lands before the first byte. The message differs — "a
turn is still running" is not "a decision is pending" — but the shape a client handles does not.

This is worth writing down as a behaviour: a disconnected turn keeps its thread for up to the drain
timeout, so a fast retry can get a 409 for a few seconds. That is the price of not losing the turn,
and it is recoverable, which is exactly the trade this section opened with.

**The registry is process-local, and that is a deployment constraint rather than an implementation
detail.** Both registries are module state, so the 409 holds inside one worker and nowhere else:
under `uvicorn --workers 2`, or two replicas behind a load balancer, a retry that lands on another
process sees nothing in flight and starts a second run on the same thread — this very failure,
unguarded — and a restart forgets whatever it was holding. Making it real across processes means a
conditional write with a TTL on the thread key in DynamoDB, which is a lock with its own failure
mode: a stale one wedges a thread until it expires. Until that trade is worth taking, this app runs
a single worker, and `README.md` says so where it promises the 409.

---

## 8. Frontend — `universal-agent-chat`

`README.md` there lists Streaming under **Not included**. This extends the wire contract, so that
section and the contract table both change.

### 8.1 A flag, not a route

`VITE_API_ROUTES` does **not** change — there is no second path to point at. Streaming is a property
of how `send` is called, so it is a flag:

```
VITE_STREAMING=true
```

Default **off**, parsed in `src/config/` beside the routes. Off, the chat behaves exactly as it does
today, through `useSendMessage`. That preserves the rule the README already states — "Missing
optional routes degrade rather than break" — for the same reason: the app must stay pointable at an
agent system that does not stream, and `Accept: text/event-stream` sent at one that ignores it comes
back as JSON the SSE parser cannot read.

A flag is strictly weaker than route config, and that is the trade: with a `/stream` route the
backend advertised its own capability, and now the operator asserts it. Worth one line in the
README next to the variable.

`decide` is left non-streaming in this pass, and here the header earns its keep: `send` and `decide`
are the same route in `.env.example:21`, so the *request* opts in, not the config. `decide` streams
later by setting the header in its own hook, with no config change at all.

### 8.2 Transport: `fetch`, not axios, not `EventSource`

- **`EventSource`** (the browser built-in) is `GET`-only and cannot carry a body. The turn needs a
  JSON `POST`. Ruled out.
- **axios** in the browser is XHR-backed; `responseType: 'stream'` is Node-only. Ruled out.
- **`fetch` + a parser** is the only option. `@microsoft/fetch-event-source` is the name everyone
  reaches for and it has not been published since **2021-04-25** (v2.0.1) — do not add it. Use
  [`eventsource-parser`](https://www.npmjs.com/package/eventsource-parser) (v4.1.0, published
  2026-08-20), which is a parser and nothing else, over `response.body`.

New file `src/api/stream.ts`, beside `chat.ts`, keeping the rule that `api/` knows nothing about
React. It resolves the **`send`** route — the same one `chat.ts` posts to — and the only thing that
makes it stream is the header:

```ts
export async function* streamMessage(
  userId: string,
  query: string,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ user_id: userId, query }),
    signal,
  });
  // 409 / 422 still arrive as JSON with a real status (P5d/P5e) — reuse chat.ts's error path
  // before touching res.body.
}
```

Check `res.headers.get('content-type')` before parsing. A backend that ignores the header answers
`application/json`, and the honest failure there is "streaming is on but this backend does not
stream", not a parser that silently yields nothing — that is the failure mode §8.1's flag buys.

`src/types/chat.ts` gains the event union — `{type:'token', text} | {type:'done', response:
ChatResponse} | {type:'pending', response: ChatResponse} | {type:'error', detail: string}` — beside
the existing `ChatResponse`, which is unchanged and still what `done` carries.

### 8.3 `useStreamMessage`, beside `useSendMessage`

Same `SendVariables`, same optimistic user row, same retry-on-error behaviour, so `ChatPanel` picks
one or the other by the `VITE_STREAMING` flag (§8.1) and nothing below it changes:

1. `onMutate` — unchanged from `useSendMessage`.
2. First `token` — `appendMessage` an empty assistant row, remember its id.
3. Each `token` — `updateMessage(… content: prev.content + text)`.
4. `done` / `pending` — **replace** `content` with `response.message` and set `pending` from
   `fromResponse`. This is §5.4's rule; without it a tool failure leaves the discarded tokens on
   screen.
5. `error` — mark the row `status: 'error'`; the existing Retry button already handles the rest.

`streamedQuery` from TanStack Query was considered and rejected: it owns a *query*'s data, and this
transcript is a single cache entry written by mutations (`hooks/transcript.ts`). Two owners of
`historyKey(userId)` is a merge problem the current design deliberately does not have.

### 8.4 Three details that decide whether it feels good

- **Re-parsing.** `Markdown` re-parses the whole bubble on every token. At ~30 tokens/s over a long
  answer that is 30 full parses a second of a growing string. Batch tokens in the hook and flush on
  a ~50ms timer: one React render per flush, and the eye cannot tell.
- **Abort.** An `AbortController` per turn, aborted on unmount and on **New conversation**. Without
  it a reset leaves a stream writing into a transcript that no longer exists.
- **The typing indicator.** `isThinking` currently means "a request is in flight". It should mean
  "in flight **and** no token has arrived yet" — otherwise the dots sit under the answer that is
  already being written. Add a caret at the end of the streaming bubble instead; `theme.ts` already
  has the `blink` keyframe the indicator uses.

### 8.5 The dev proxy

Vite's `/api` proxy streams by default, and `EventSourceResponse` sets `X-Accel-Buffering: no` for
anything downstream. Verify it end to end anyway before believing the backend is at fault: a
buffering proxy and a broken stream look identical from the browser.

---

## 9. Tests

Backend, `tests/api/test_chat_stream.py`, following the house conventions — real graph, stub model
via `tests/agent_stubs.py`, stub opted into per test rather than in the shared `app` fixture, async
and fully typed. `StubChatModel` implemented `_generate` only, and needed a `_stream` before any of
this could assert anything; it has one now, next to `_generate`, so both suites share one stub.

| Test | Asserts |
|---|---|
| tokens arrive in order | the concatenation of every `token` equals the stub's reply |
| `done` is authoritative | `done.message` equals the reply, and equals the `conversation_history` row |
| **the router does not leak** | no `token` contains the routing JSON (§4.1) — script a route reply that would be visible |
| **the summariser does not leak** | drive a thread past the summarisation trigger; no `token` carries summary text (§4.2). This is the one worth writing first — it is the leak that passes every other test |
| a tool result does not leak | a scripted `retriever` call streams no passage text (§4.3) |
| `return_direct` | `compare_programs` yields **several** `token`s, and their concatenation equals `done.message` exactly — the assertion that catches the comparison being sent twice (§4.4) |
| the `return_direct` fallback | with `DIRECT_ANSWER_NODE` patched to a name the graph does not use, the same turn yields **one** `token` carrying the whole comparison — a node renamed upstream costs the trickle, not the answer |
| **a tool's own graph leaks nothing but `merge`** | unit tests over single parts: `merge` at depth two is a direct-answer token, `model` at depth two is not, and `merge` at depth *one* is not either (§4.4). Both integration rows above need `merge` on the stub: it gets its model from `graphs/compare_programs/nodes.py`, which neither patch in `tests/agent_stubs.py` reaches, and `gather` needs its knowledge service patched the way `test_tool_routing_eval.py` does |
| **the depth check, on its own** | a unit test over a single part: `ns == ('qa:…', 'tools:…')` with `langgraph_node == 'model'` is not forwarded. The row above cannot stand in for it, because `merge` is rejected by 4.2's node name whatever 4.1 does — delete the depth check and that row still passes (measured). This is the only test holding §4.1, and it is what a tool running a `create_agent` of its own would trip |
| a tool failure | `done.message == TOOL_FAILURE_MESSAGE` and **nothing** is written to `conversation_history` — and for a `return_direct` tool, **no `token` at all**. Assert the event names: its failed `ToolMessage` is still named after the tool, so asserting only on `done` lets the exception text stream to the applicant while the test stays green (§4.3 — it did) |
| an interrupt | the turn ends on `event: pending` with the same `pending[]` the JSON endpoint returns for the same input |
| the paused guard | posting with `Accept: text/event-stream` on a paused thread is a **409 with a JSON body**, not a 200 with an `error` event (§6.2) |
| `MODEL_NODE` is real | at least one forwarded part had `langgraph_node == 'model'` — so a LangGraph rename fails loudly instead of streaming silence |
| **the JSON representation is untouched** | the *same* request without the header returns `application/json` and today's `ChatSchemaOut` — §1's whole premise, and the one test that fails if someone later makes the route stream unconditionally |
| **the run finishes with nobody consuming the stream** | start the turn, never iterate `stream_events` at all, await the pending run, then assert the `conversation_history` row exists and matches — §7's entire reason for being. Without this test B silently degrades to A. Deliberately *not* written as "read one `token`, then close the response": through the test transport that reads a turn which already finished, and it passes with `record_turn` moved into the consumer — the one coupling it exists to forbid. Measured; see the note below |
| the producer is not owned by the request | after a turn that was read to the end, `_RUNS` drains to empty and no task was cancelled — catches a `TaskGroup` creeping back in (§7.1) |
| a second turn while one is in flight | 409, not two runs on one thread (§7.3) |
| the drain waits | with a run pending, the lifespan shutdown does not return until it finishes or the timeout expires (§7.2) |

`httpx.AsyncClient` reads SSE with `client.stream('POST', …, headers={'Accept': 'text/event-stream'})`
+ `aiter_lines()`, and the existing `not_auth_client` fixture works unchanged — **but it does not
stream.** `ASGITransport.handle_async_request` awaits the whole app, collects every body part and
only then returns a `Response` (`httpx/_transports/asgi.py`, 0.28.1), so `aiter_lines()` walks a
buffer of a turn that is already over, and the app is never sent `http.disconnect`.

Two consequences, and both were paid for by writing the tests the other way first:

- the suite can assert which events came and in what order, but never that they *arrive*
  incrementally — only `make stream_turn` (§10) shows that;
- a mid-turn disconnect is not expressible at all. Holding the turn open to force one deadlocks
  instead: the client waits to enter the stream, the producer waits to be let go.

So the property §7 exists for is tested at the seam where it lives — the queue is abandoned and the
generator is never iterated — not at a socket that cannot be cut.

No changes to `[tool.pytest.ini_options]`.

The eval suites are untouched — literally, not by choice: they send no `Accept` header, so they keep
getting JSON. That is the payoff of §1. They exercise the agent, not the transport, and opting them
into SSE would buy nothing and cost a rewrite of every assertion.

---

## 10. Documentation to update

Per the project's own rule that a new workflow gets all three:

- **`Makefile`** — a target that runs the streaming turn by hand (`curl -N -H 'Accept:
  text/event-stream'`), next to `run_app`. The existing `curl` lines at `README.md:216-220` and
  `Makefile:52` stay exactly as they are.
- **`README.md`** — a "Streaming" subsection under *Conversation memory (the LangChain assistant)*:
  that one endpoint serves both representations and the header is the switch, the event table from
  §5.4, the "tokens are a preview" rule, and the two behaviours §7 makes user-visible — a turn is
  recorded even if the caller hangs up, and a re-ask while that run is still going gets a 409.
- **`docs/manual-testing.md`** — a `curl -N` script for a plain answer, a `compare_programs` answer
  and a booking that pauses.
- **`universal-agent-chat/README.md`** — the *Not included* list loses "Streaming"; the contract
  table gains **no row** (no new route), and instead `send` is annotated as streaming when
  `VITE_STREAMING` is on. `.env.example` gains the variable with its comment.

---

## 11. Out of scope

Streaming `decide` (§8.1), resumable streams (`Last-Event-ID`), a stop/cancel button, streaming the
CrewAI `/ask` flow, per-step progress events ("searching the knowledge base…") — the `custom` stream
mode and `get_stream_writer` are the hook for that later, and nothing here forecloses it.

---

## 12. Reading list

Checked 2026-08-30, against the versions this project actually has installed.

**LangGraph / LangChain**

- [Streaming — LangGraph (Python)](https://docs.langchain.com/oss/python/langgraph/streaming) — the
  seven stream modes, `messages` with `version="v2"`, `subgraphs=True`, and filtering by
  `langgraph_node`. **Read this one first**; §4 is a direct consequence of it.
- [Streaming — LangChain agents](https://docs.langchain.com/oss/python/langchain/streaming/overview)
  — the same from `create_agent`'s side, plus streaming tool calls and the v2 format.
- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) — why `subgraphs=True`
  is mandatory here, and how an interrupt raised inside a subagent surfaces.
- [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) —
  `HumanInTheLoopMiddleware`, `interrupt_on`, and resuming with `Command(resume=…)`.
- [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) —
  `SummarizationMiddleware`; useful for understanding §4.2 rather than working around it.

**FastAPI / sse-starlette**

- [`sse-starlette`](https://github.com/sysid/sse-starlette) — **the transport this design uses.**
  `EventSourceResponse` as a real response class you can construct and return, its ping task, and
  `ServerSentEvent`'s `str(data)` encoding (P6).
- [Response Model — return type](https://fastapi.tiangolo.com/tutorial/response-model/) — that the
  `response_model` parameter takes priority over the return annotation, which is what lets §1 return
  a `ChatSchemaOut | EventSourceResponse` union and still document the JSON shape.
- [Server-Sent Events](https://fastapi.tiangolo.com/tutorial/server-sent-events/) — FastAPI's own
  SSE support, added in **0.135.0**. Read it to understand what is being passed over and why: it is
  `response_class=` on a generator endpoint, which cannot also serve JSON (P6). No `pyproject` bump
  is needed for it.
- [`fastapi.sse` reference](https://fastapi.tiangolo.com/reference/sse/) — same caveat; useful for
  the `data` vs `raw_data` distinction, which `sse_starlette` spells the same way.

**Browser**

- [Using server-sent events — MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
  — the wire format (`event` / `data` / `id` / `retry`), why `EventSource` cannot be used here, and
  the 6-connections-per-domain limit without HTTP/2.
- [`eventsource-parser`](https://www.npmjs.com/package/eventsource-parser) — the parser recommended
  in §8.2.
- [`experimental_streamedQuery` — TanStack Query](https://tanstack.com/query/latest/docs/framework/react/reference/functions/experimental_streamedQuery)
  — worth reading to understand why §8.3 does *not* use it.

---

## 13. Questions for whoever implements this

1. ~~What drain timeout, and what happens to a run still pending when it expires?~~ **Settled: 25
   seconds**, as `DRAIN_TIMEOUT_SECONDS` in `app/settings.py`. The first answer here was 90 — the
   model's own ceiling, `timeout=30` × three attempts — and it was wrong for a reason worth keeping:
   the process does not get to choose how long it lives. `docker stop` gives 10s and Kubernetes 30s
   by default, so a 90s drain would have been killed at 30 with nothing logged. A run still pending
   when the drain expires is logged at `ERROR` with a count and left alone, not cancelled; the turn
   is checkpointed either way and only the transcript row is lost. It is a setting, not a constant,
   because the right value is the deployment's grace period minus a margin. Whether that `ERROR`
   deserves an alert is an ops question, not a design one.
2. Does `decide` stream in this pass, or in the next one (§8.1)?
3. Does the streaming representation go into `docs/manual-testing.md` as a first-class script, or
   stay a `curl -N` footnote until the frontend uses it?
4. ~~Does `compare_programs` stream its comparison as it is written, or keep arriving as one
   event?~~ **Settled: it streams — §4.4.** It costs what the question said it would: a second,
   named exception to §4.1 and a rule that drops the `ToolMessage` whose text has already gone out.
   What decided it is that the alternative was not slower or faster but simply blind — measured on
   the real agent, the whole comparison landed at once after 4.4s, where it now begins at 2.8s. No
   extra model call either way: removing `return_direct` instead would have added one, and that is
   the version rejected.

Settled, and recorded here so it is not reopened by accident: **§7 is B** — the run is detached, and
a disconnected turn is still written. The cost is §7.3's per-user in-flight guard, which is a new
409 the JSON endpoint never returns. **One endpoint.** An earlier draft of
this document put streaming on `POST /langchain-assistant/stream` and rejected content negotiation
on three grounds — that `response_model` could describe neither shape, that a client could not know
what it was about to get, and that two shapes on one operation are undocumentable. P5 disproves the
first and third; the second is backwards, because JSON is the default and SSE is opt-in, so a client
gets the stream only by asking for it. What the earlier draft got right is now §6.2's constraint and
§1's `fastapi.sse` exclusion.

---

## 14. Where to start

The order below is not the only one that works, but each step is the one that makes the next step
cheap to get right. Steps 0-2 touch no code in `app/`.

**0. The probes — done.** P1-P4 were re-run on 2026-09-11 against the real graph, and §3, §4.1, §4.2,
§4.3, §5.2, §5.3 and §9 already carry what they corrected. Nothing to do now. The script was
throwaway and has been deleted, so after a LangGraph or LangChain bump, write it again from §3's
description and check the table before trusting §4 — the suite cannot do that job for you, because
it scripts the model.

**1. The dependency.** `uv add "sse-starlette>=3.4.5"`. It is in the lock today only through `mcp`,
and a `crewai` bump can take it away — §5.1.

**2. The stub, before any of the code it tests.** `StubChatModel._stream` next to `_generate` (§9):
without it a stubbed turn produces no tokens, so there is nothing for any of the token tests to
assert. There is a second gap in the same file, and it is the quieter one: `with_structured_output`
returns a `RunnableLambda` and never calls a chat model, so in tests the **router streams nothing**
— §9's "the router does not leak" passes with 4.1 deleted. Make the stubbed routing call go through
the model, streaming the routing JSON, or that test asserts nothing at all.

**3. The first slice that streams a plain answer**, written in §7.1's shape from the start — a
detached producer, a queue, and `_RUNS` holding the task; §5.3's flat generator is a reading aid, not
a stage to pass through. That is `stream_schemas.py`, `stream_agent`, `start_streaming_turn` /
`stream_events` / `_run_streaming_turn` / `assert_can_take`, the three filters of §4, and the `Accept` branch
in `router.py`. The tests to write with it, from §9: the JSON representation is untouched, tokens arrive
in order, `done` is authoritative, the three leak tests — **the summariser one first** — and the test
that a turn nobody is consuming still records itself, which is what tells B from A. Read the note
under §9's table before writing that last one: the obvious version of it asserts nothing.

**4. `stream_resume`.** Factor `_build_hitl_decisions` out of `resume_agent` first, then let both
call it (§5.2) — a copy is how the two drift apart on a write that needs approval. Its test is the
interrupt: the turn ends on `event: pending` carrying the same `pending[]` the JSON endpoint returns.

**5. The guards and the drain.** `_RUNS` keyed by `user_id` with the second-turn 409 (§7.3), and
`wait_for_streaming_turns` inside `get_checkpointer().opened()` (§7.2). Last because a turn streams and records
without either. The drain timeout was the open question that kept this last; it is settled at 90
seconds — §13.1.

**6. The documentation** — all four places in §10, once the code works.

The frontend (§8) is its own pass and needs nothing from this list past step 3.
