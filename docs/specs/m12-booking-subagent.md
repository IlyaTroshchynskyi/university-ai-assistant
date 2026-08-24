# M12 — Booking subagent + HITL (spec)

Design-only document. No implementation except the booking system prompt (§7), which is written out
in full because its wording is part of the contract.

Target package: `app/ai_assistant_langchain/` (the LangChain assistant). The CrewAI flow
(`app/ai_assistant/`) is **not** touched — its `ListFreeSlotsTool` and `main_flow.do_book` stay as
they are. This milestone builds the LangChain equivalent alongside them.

---

## 1. What already exists (and is therefore not part of this milestone)

| Piece | Where | Status |
|---|---|---|
| `appointment_slots` DynamoDB table (`DYNAMODB_SLOTS_TABLE`) | `app/settings.py:40` | exists |
| `Slot` / `SlotStatus` models | `app/core/dynamodb/schemas.py:9,32` | exists |
| `SlotsRepository`: `list_open_slots_on_date`, `list_slots_by_status`, `list_student_bookings`, `book_slot`, `cancel_slot` | `app/core/dynamodb/slots_repository.py` | exists |
| Conditional-write double-booking guard (`condition=Attr('status').eq(OPEN)` → returns `None`) | `slots_repository.book_slot:60` | exists — **T12.9 is already satisfied at the repository level** |
| Seed data (24 open slots, 2026-10-06 …) | `db/seed/appointment_slots.json` | exists |
| Checkpointer (`DynamoDBCheckpointer`, `get_checkpointer()`) | `app/ai_assistant_langchain/checkpointer/saver.py:472` | exists (M11) |
| Q&A agent (`create_assistant_agent`) | `app/ai_assistant_langchain/agent.py` | exists |
| `/langchain-assistant` endpoint → `ChatService` → `AgentService` | `router.py`, `service.py`, `agent_service.py` | exists |

**Consequence: T12.2 (Alembic migration + seed) is a no-op.** The milestone text assumes Postgres;
this project is on DynamoDB and the table is already modelled, seeded and covered by a repository.
The only data-layer work left is the optional `email` capture (T12.11, §9).

So the real scope is: **three `@tool`s → a booking subagent with a HITL gate → a main graph with a
router → the interrupt surfaced through the existing endpoint.**

---

## 2. Files touched / added

```
app/ai_assistant_langchain/
  booking_tools.py       NEW   list_free_slots / book_appointment / cancel_appointment
  agent_schemas.py       EDIT  + three tool input schemas, + pending-approval response types
  prompts.py             EDIT  + BOOKING_SYSTEM_PROMPT, + ROUTER_PROMPT
  booking_agent.py       NEW   build_booking_agent() — create_agent + HumanInTheLoopMiddleware
  agent.py               EDIT  create_assistant_agent() loses its checkpointer (see §5)
  main_graph.py          NEW   build_main_graph() — router → {qa, booking}, ONE checkpointer
  agent_service.py       EDIT  run on the main graph; detect __interrupt__; resume with Command
  service.py             EDIT  pass the decision through; build the pending-approval response
  schemas.py             EDIT  UserQuery gains an optional `decision`; ChatSchemaOut gains status
  router.py              EDIT  nothing structural — same endpoint, richer response model
tests/
  langchain/test_booking_hitl.py   NEW  approve / edit / reject
  langchain/test_routing.py        NEW  booking vs qa routing
docs/specs/m12-booking-subagent.md  this file
```

`app/main.py` is untouched (`assistant_router` is already included). No new endpoint is added and
none is removed — see §4 on why `/booking/try` is skipped.

---

## 3. T12.3 — the three tools (`booking_tools.py`)

Style follows `tools.py`: `@tool(args_schema=...)`, async, thin wrapper over the repository,
`logger.info` on the way out, human-readable string on the "nothing found" path.

### Input schemas (into `agent_schemas.py`)

```python
class ListFreeSlotsInput(BaseModel):
    date: str | None   # 'YYYY-MM-DD', None = all open slots
    topic: str | None  # substring match, applied in Python after the query

class BookAppointmentInput(BaseModel):
    slot_id: int
    date: str          # 'YYYY-MM-DD' — part of the primary key
    start_time: str    # 'HH:MM'      — part of the sort key
    applicant_name: str
    topic: str

class CancelAppointmentInput(BaseModel):
    slot_id: int
    date: str
    start_time: str
```

> **Design note — why `date` and `start_time` are tool arguments even though the milestone text
> shows only `slot_id`.** The DynamoDB key is `pk = date`, `sk = f'{start_time}#{slot_id}'`
> (`slots_repository.py:56`). Reaching a slot by `slot_id` alone would need a scan or a new GSI.
> The agent always calls `list_free_slots` first and therefore already holds all three fields, so
> passing them costs nothing. **This also makes the HITL payload self-describing:** the human sees
> the date and time in the proposed args, not an opaque id.

### Behaviour

| Tool | Repository call | Returns |
|---|---|---|
| `list_free_slots(date, topic)` | `list_open_slots_on_date(date)` if `date` else `list_slots_by_status(OPEN)`; then filter by `topic` in Python | `list[dict]` of `Slot.model_dump()`, or `'No open consultation slots …'` |
| `book_appointment(...)` | `book_slot(date, start_time, slot_id, booked_by=applicant_name, topic=topic)` | confirmation string with date/time/topic, or `'Slot … is no longer available …'` when the repo returns `None` |
| `cancel_appointment(...)` | `cancel_slot(date, start_time, slot_id)` | confirmation string, or `'Slot … is not booked …'` on `None` |

`None` from the repository is a **normal outcome, not an error**: it means the conditional write
lost. The tool turns it into a sentence the model can relay. That is the double-booking guard
(T12.9) surfacing — nothing extra to build.

`booked_by` currently carries the applicant *name*; §9 revisits this for email.

---

## 4. T12.4 / T12.5 — the booking subagent (`booking_agent.py`)

```python
@lru_cache
def build_booking_agent() -> CompiledStateGraph:
    return create_agent(
        model=_get_model_factory(),          # reuse the one in agent.py
        system_prompt=BOOKING_SYSTEM_PROMPT, # §7
        tools=[list_free_slots, book_appointment, cancel_appointment],
        context_schema=CustomContext,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={...})],
        # NO checkpointer — the main graph owns it
    )
```

### The gate

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        'book_appointment':   {'allowed_decisions': ['approve', 'edit', 'reject']},
        'cancel_appointment': {'allowed_decisions': ['approve', 'reject']},
    },
    description_prefix='Please confirm this appointment change',
)
```

`list_free_slots` is read-only → not in `interrupt_on` → runs freely.

Verified against the installed `langchain==1.3.14`
(`langchain/agents/middleware/human_in_the_loop.py`) — the exact payload shapes are in §6.
`allowed_decisions` on cancel deliberately omits `edit`: editing which booking gets cancelled is a
different request, not an amendment.

**Sync/async caveat.** `HumanInTheLoopMiddleware` implements `awrap_model_call`; the app drives the
graph with `ainvoke`, so this is fine. A sync `invoke` anywhere in a test would raise
`NotImplementedError` from the base middleware — the tests must be async (they already are).

### T12.5b — the temporary `/booking/try` endpoint: **skipped, deliberately**

The milestone wants it as a debugging aid. Here the same isolation is available for free from a
test: `tests/agent_stubs.py` already builds an agent against a `StubChatModel` and an opened
checkpointer. Part A's checkpoint is therefore "the three HITL tests pass against
`build_booking_agent()` compiled with the checkpointer *in the test*", not a throwaway route in
`app/main.py`. This avoids adding — and then having to remember to delete — a public endpoint.

If you would rather have the endpoint anyway, add it to `app/ai_assistant_langchain/router.py` (not
`main.py`) so deleting it is a one-file change.

---

## 5. T12.6 — the main graph (`main_graph.py`)

```
        ┌── router (conditional edge from START) ──┐
START ──┤                                          ├── qa      ──> END
        └──────────────────────────────────────────┴── booking ──> END
```

```python
class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

Use LangGraph's `MessagesState` unless you need extra channels; a bare `{messages}` state is what
both subagents already speak, which is why they can be added **as nodes with no adapter**.

```python
@lru_cache
def build_main_graph() -> CompiledStateGraph:
    builder = StateGraph(MainState, context_schema=CustomContext)
    builder.add_node('qa', create_assistant_agent())
    builder.add_node('booking', build_booking_agent())
    builder.add_conditional_edges(START, route, {'qa': 'qa', 'booking': 'booking'})
    builder.add_edge('qa', END)
    builder.add_edge('booking', END)
    return builder.compile(checkpointer=get_checkpointer())
```

### Checkpointer ownership — the one required change to existing code

`create_assistant_agent()` currently passes `checkpointer=get_checkpointer()` (`agent.py:25`). It
**must stop doing that**: the checkpointer moves to `builder.compile(...)`. A subgraph compiled with
its own checkpointer runs on its own thread, and the parent's `thread_id`/interrupt plumbing breaks
— this is the classic bug named in the milestone. Both subagents are compiled bare.

`create_assistant_agent`'s `SummarizationMiddleware` stays exactly as it is.

Because the compiled agents are added as nodes directly (not wrapped in functions), **there is no
`config` to forward** — LangGraph hands the parent config down itself, and the inner checkpoints are
namespaced by node name (`checkpoint_ns = 'booking'`). The `DynamoDBCheckpointer` already keys every
row by `checkpoint_ns` (`saver.py:148,313`), so this needs no storage change.

### The router

Per `no-keyword-routing`: **LLM classification, not substring matching.** A tiny structured call on
the last human message.

```python
class Route(BaseModel):
    destination: Literal['qa', 'booking']

async def route(state: MainState) -> Literal['qa', 'booking']:
    decision = await _get_model_factory().with_structured_output(Route).ainvoke(
        [SystemMessage(ROUTER_PROMPT), state['messages'][-1]]
    )
    return decision.destination
```

`ROUTER_PROMPT` (draft — tune during implementation):

> You route a university assistant's messages to one of two handlers.
> `booking` — the user wants to arrange, change, confirm or cancel an admissions consultation
> appointment, or is answering a question the scheduler asked them (a time, a date, their name,
> "yes, book it", "cancel that").
> `qa` — everything else: questions about programs, tuition, scholarships, deadlines, policies,
> professors, campus places, plus greetings and small talk.
> When the message merely *mentions* a deadline or an office's opening hours without asking to meet
> anyone, that is `qa`.

**Known limitation — routing is stateless per turn.** A mid-booking reply like "Monday works" is
routed on its own text. The prompt bullet about answering the scheduler covers the common cases; the
robust fix (sticky routing: remember the last node in state and stay there until the booking
finishes) is out of scope here — note it in the test file as the reason a follow-up test is absent.
A resume (§6) is *not* affected: it re-enters the paused node directly and never hits the router.

---

## 6. T12.7 — surfacing the interrupt through `/langchain-assistant`

### What comes back

`await graph.ainvoke(...)` returns a dict with an `'__interrupt__'` key: a list of `Interrupt`
objects. `interrupt.value` is a `HITLRequest`:

```python
{
  'action_requests': [
      {'name': 'book_appointment',
       'args': {'slot_id': 3, 'date': '2026-10-06', 'start_time': '11:00',
                'applicant_name': 'Anna', 'topic': 'scholarships'},
       'description': 'Please confirm this appointment change\n\n...'},
  ],
  'review_configs': [
      {'action_name': 'book_appointment',
       'allowed_decisions': ['approve', 'edit', 'reject'],
       'args_schema': {...}},
  ],
}
```

### Resume payloads (`HITLResponse` — exact shapes, verified in 1.3.14)

```python
Command(resume={'decisions': [{'type': 'approve'}]})
Command(resume={'decisions': [{'type': 'edit',
                               'edited_action': {'name': 'book_appointment',
                                                 'args': {...full arg set...}}}]})
Command(resume={'decisions': [{'type': 'reject', 'message': 'The applicant declined.'}]})
```

Note `edited_action` (a `{name, args}` object), **not** a bare `args`. `args` must be the complete
argument set, not a patch. One decision per action request, in order.

### API surface

`schemas.py`:

```python
class Decision(BaseModel):
    type: Literal['approve', 'edit', 'reject']
    args: dict[str, Any] | None = None   # required for 'edit'
    message: str | None = None           # optional for 'reject'

class UserQuery(BaseModel):
    user_id: UUID
    query: str | None = ...              # None when `decision` is set
    decision: Decision | None = None

class ChatSchemaOut(BaseModel):
    status: Literal['answer', 'pending_approval'] = 'answer'
    message: str                         # the reply, or the approval description
    pending: PendingApproval | None = None

class PendingApproval(BaseModel):
    action: str                          # 'book_appointment'
    args: dict[str, Any]
    allowed_decisions: list[str]
```

Validation: exactly one of `query` / `decision` must be set (`model_validator`).

`AgentService.run_agent` gets a sibling rather than growing a branch (per `add-alongside-dont-rewrite`):

```python
async def run_agent(self, messages, user_id) -> AgentResponse: ...          # unchanged shape
async def resume_agent(self, decision: Decision, user_id) -> AgentResponse: ...
```

Both call `self._graph.ainvoke(payload, config={'configurable': {'thread_id': user_id}}, context=...)`;
the second passes `Command(resume=...)` as the payload. `ChatService.process_user_query` inspects
the result: `'__interrupt__' in result` → build `PendingApproval` from
`result['__interrupt__'][0].value`; otherwise take `result['messages'][-1].content` as today.

The `thread_id` is the `user_id`, unchanged — the pause and its resume are the same thread by
construction.

---

## 7. The booking system prompt (`BOOKING_SYSTEM_PROMPT`)

```python
BOOKING_SYSTEM_PROMPT = """
You are the admissions front desk of the university: you schedule consultation
appointments with an admissions advisor, and that is the only thing you do. You are
warm, brief and concrete.

How you work:
- Never invent a slot. The only appointments that exist are the ones
  "list_free_slots" returns. Call it before you propose anything, every time — an
  appointment you remember from earlier in the conversation may already be taken.
- If the applicant named a day or a rough time ("Friday afternoon", "next week"),
  pass the date to "list_free_slots" and offer what comes back that fits. If nothing
  fits, say so plainly and offer the nearest alternatives you did find, rather than
  bending the request into a slot that does not match it.
- Propose ONE slot at a time, with its date, start time and topic, and ask the
  applicant to confirm it or ask for another. Offer a short list only when they gave
  you nothing to narrow by.
- Before you call "book_appointment" you need three things: the slot (id, date,
  start time), the applicant's name, and what they want to discuss. Ask for whichever
  of them you are missing — one short question at a time — and do not guess any of
  them. An empty topic is not a topic.
- One appointment per request. Once a booking succeeds, confirm it in one sentence
  with the date, time and topic, and stop. Do not offer to book anything else.

Booking and cancelling are reviewed by a human before they take effect. You will not
see that review happen; you will simply see the tool's result:
- If it succeeded, confirm the appointment to the applicant.
- If it came back saying the slot is no longer available, apologise briefly, call
  "list_free_slots" again and propose one of the slots that is actually still open.
- If the tool result says the action was rejected or edited by a reviewer, take that
  at face value and tell the applicant plainly what happened — that the booking was
  not made, or that it was made for the different time you were given. Never re-issue
  a rejected booking unless the applicant asks for it again.

Cancelling: to cancel, you need the appointment's date, start time and slot id.
Ask for the details you are missing; do not cancel a booking on a guess.

Anything that is not about arranging, changing or cancelling a consultation —
tuition, programs, deadlines, where a building is — is not yours to answer. Say in
one sentence that you only handle consultation appointments, and offer to book one if
that is what they came for.

Always reply in the language the applicant wrote in. Answer what was asked and stop
there — no generic "let me know if you need anything else".
"""
```

---

## 8. T12.8 — tests

New file `tests/langchain/test_booking_hitl.py`, built on the existing stub machinery
(`tests/agent_stubs.py`: `StubChatModel` + patched `_get_model_factory`) so no real LLM is called.
Per `test-tools-via-llm-only`, the tools are exercised *through* the agent: the stub model emits the
`book_appointment` tool call, it is not invoked directly.

The router also calls the model (`with_structured_output`) — `StubChatModel` must either gain
`with_structured_output` support or the router function is patched to return a fixed destination in
the HITL tests (routing gets its own test).

| Test | Script | Assertion |
|---|---|---|
| **approve** | stub emits `list_free_slots` → then `book_appointment(slot 3)`; first `ainvoke` returns `__interrupt__`; resume `approve` | slot 3 in DynamoDB is `booked`, `booked_by` set; final message confirms |
| **edit** | same, then resume with `edited_action.args` pointing at slot 8 | slot 8 is `booked`, **slot 3 is still `open`** |
| **reject** | same, then resume `reject` | **no** slot changed status; the last message says it was declined |
| **pause is before the write** | assert on the interrupt turn, before resuming | slot 3 still `open` at that point |
| **double booking (T12.9)** | slot pre-booked in the fixture; approve a booking for it | tool returns the "no longer available" string, status unchanged, no exception |
| **routing — booking** | "I'd like to book a consultation on Friday" | the booking node ran (booking tools called / `qa`'s tools not) |
| **routing — qa** | "How much is tuition for the CS program?" | `retriever` called, no booking tool called |

Fixtures: reuse `tests/conftest.py`'s DynamoDB setup and seed a small, known slot set per test
(don't lean on `db/seed/appointment_slots.json` — a test that asserts "slot 3 is open" should own
slot 3). Checkpointer is opened per test via the existing `TestBaseAgentClass` pattern
(`checkpointer.opened()`), and each test uses a fresh `thread_id`.

---

## 9. Bonus tasks — verdicts

- **T12.9 double-booking** — already implemented in `book_slot`'s conditional write. Scope here is
  only the test row above and making sure the tool renders `None` as a sentence.
- **T12.11 email** — `Slot.booked_by` is documented as "student email" but the tools currently pass
  a name, and `list_student_bookings` queries `STUDENT#{email}`. Decide one: either
  `book_appointment` takes `applicant_email` and puts it in `booked_by` (with the name in `topic`'s
  neighbourhood — i.e. a new `applicant_name` attribute), **or** add an explicit `email` attribute.
  Recommendation: **take email as a separate required tool arg and store it in `booked_by`, moving
  the name into a new `applicant_name` attribute** — that keeps GSI2 (`list_student_bookings`)
  meaningful, which it is not today.
- **T12.10 `.ics` / T12.12 no-show** — out of scope for this milestone. Both need an outbound
  channel (file or mail stub) that does not exist yet; do them after M12 lands, if at all.

---

## 10. Order of work

1. `booking_tools.py` + the three input schemas. *(Part A)*
2. `BOOKING_SYSTEM_PROMPT` + `build_booking_agent()` with the HITL gate. *(Part A)*
3. HITL tests (approve / edit / reject / pause-before-write) against the subagent compiled with the
   checkpointer **in the test** — this is the Part A checkpoint. *(Part A)*
4. Move the checkpointer out of `create_assistant_agent`; build `main_graph.py` with the router.
5. Router prompt + routing tests.
6. `schemas.py` / `agent_service.py` / `service.py`: resume path and the pending-approval response.
7. README + AGENTS.md + Makefile entries per `document-commands-everywhere` (new test target if the
   booking tests get their own marker).