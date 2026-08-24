# Router conversation context (follow-up to M12 §5)

Design-only document. It closes the "routing is stateless per turn" limitation left open in
`docs/specs/m12-booking-subagent.md` §5, and records why the obvious fix (sticky routing) is *not*
the one being taken.

Scope: `prompts.py`, `main_graph.py`, `tests/agent_stubs.py`, one new test file. No change to either
subagent, no change to the HITL path.

---

## 1. The problem

`route()` classifies **one message with no context** (`main_graph.py:28`):

```python
.ainvoke([SystemMessage(ROUTER_PROMPT), state["messages"][-1]])
```

```
assistant (booking): Which day suits you?
applicant:           Monday works.          ← routed on these two words alone
```

"Monday works" carries no evidence that it is about an appointment, so it can land in `qa`, which
has no idea a booking is half-finished. `ROUTER_PROMPT` currently compensates with a paragraph
telling the model that bare fragments (a day, a time, an email, "yes") are probably answers to the
scheduler. That is a guess dressed as a rule: it also drags "Monday works" into `booking` when the
conversation was never about appointments.

A HITL resume is **not** affected — `Command(resume=...)` re-enters the paused node directly and
never reaches the router.

---

## 2. Why not sticky routing

The obvious fix — remember the last node in `MainState` and stay there until the booking finishes —
is rejected. The flag is trivial; its **exit condition** is not.

- The booking agent ends its turn after *every* question it asks ("which day suits you?"). The graph
  reaches `END` each time, so "the turn ended" and "the booking ended" look identical in
  `{messages}`. Nothing in the state marks completion.
- Detecting completion needs one of: another LLM classifier call (the same stateless judgement,
  moved), a structured "done" signal out of the booking agent (a contract change to the agent that
  M12 just wrapped in a HITL gate), or the heuristic "stay until `book_appointment` succeeds" —
  which breaks on cancellations, on abandonment, and on any off-topic question mid-flow.
- Mechanically it is not one state field either: both nodes are **compiled agents added directly**
  (`main_graph.py:36-37`). They speak `AgentState` and return `{messages}` only; they cannot write
  an `active` channel. Writing it needs wrapper nodes — a structural change to the graph.

And the failure modes are asymmetric:

| | today | with sticky routing |
|---|---|---|
| wrong destination | user repeats themselves, gets through | user is **held** in booking |
| escape | always available — every turn is routed fresh | the router is bypassed, so none |

`BOOKING_SYSTEM_PROMPT_TEMPLATE` explicitly refuses everything that is not an appointment. A user
stuck in the booking node asking about tuition gets "I only handle consultation appointments" — and
no way out. That is worse than an occasional misroute.

---

## 3. The approach: show the router the recent turns

Give the router the last few turns of conversation as **text inside the system prompt**, and keep
the message being routed as the actual `HumanMessage`.

**Why text, not a message list.** Slicing `state['messages'][-6:]` can start the slice on a
`ToolMessage`, and providers reject a tool message that does not follow an assistant message with
matching `tool_calls`. Rendered text has no role-alternation rules at all. It also lets us drop the
tool traffic — `list_free_slots` returns a list of slot dicts that would dwarf the actual dialogue.

**Why the current message stays a real message.** Background in the system prompt, subject in the
human turn: the model is never in doubt about *which* message it is classifying.

**What this buys over sticky routing.** Context sets the default without removing the escape hatch:
"actually, how much is tuition?" after five booking turns still routes to `qa`, because the model
weighs the message against the background instead of being locked by it.

---

## 4. Implementation

### 4.1 `prompts.py`

`ROUTER_PROMPT` becomes `ROUTER_PROMPT_TEMPLATE` + a builder, mirroring
`build_booking_system_prompt()` right above it.

Drop the f-string and put **all three** substitutions through `.format` — mixing an f-string with
`.format` means every literal brace in the prompt has to be doubled, and the next person to edit the
text will not know that:

```python
ROUTER_PROMPT_TEMPLATE = """
...
"{booking}" — the applicant wants to arrange, move, confirm or cancel ...
"{qa}" — everything else the university gets asked ...

The conversation so far, for context only:
{history}
"""


def build_router_prompt(history: str) -> str:
    return ROUTER_PROMPT_TEMPLATE.format(booking=GraphNode.BOOKING, qa=GraphNode.QA, history=history)
```

Two paragraphs of the prompt text change. The one that claims the router is blind:

> You are shown that one message on its own, never the conversation it belongs to …

is replaced by a rule anchored on the background:

> The turns before it are shown above as background. Route the new message, not the background. The
> scheduler asks a lot of short questions, so a message that answers one — a day or a time
> ("Friday", "11:00", "the second one"), an email address, a subject to discuss, a bare "yes" /
> "that works" — is `booking` **when the turn above it came from the scheduler**. When there is no
> background, judge the message on its own.

And one paragraph is added, to keep the escape hatch explicit:

> Background sets the default; it does not overrule the message. When the applicant turns away from
> arranging an appointment and asks something the university answers — "actually, how much is
> tuition?" — that is `qa`, however much booking talk comes before it.

### 4.2 `main_graph.py`

```python
NO_HISTORY = '(no earlier messages — the conversation starts here)'


def _recent_turns(messages: Sequence[BaseMessage], limit: int = 4, max_chars: int = 300) -> str:
    """The last few spoken turns, newest last, as one line each.

    Tool calls and tool results are dropped: they are internal traffic (a slot listing is a page of
    dicts) and say nothing about what the applicant wants next.
    """
    lines: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            speaker = 'applicant'
        elif isinstance(message, AIMessage) and not message.tool_calls:
            speaker = 'assistant'
        else:
            continue
        text = ' '.join(message.text.split())   # `.text` is a property in langchain-core 1.5
        if not text:
            continue
        lines.append(f'{speaker}: {text[:max_chars]}')
        if len(lines) == limit:
            break
    return '\n'.join(reversed(lines)) if lines else NO_HISTORY


async def route(state: MainState) -> GraphNode:
    *earlier, current = state['messages']
    decision = (
        await _get_model_factory()
        .with_structured_output(Route)
        .ainvoke([SystemMessage(build_router_prompt(_recent_turns(earlier))), current])
    )
    return decision.destination
```

`limit=4` covers the fragment case (the scheduler's question is one turn back) without paying for a
long history on a call that only has to pick one of two words. `max_chars` truncates a long qa
answer — the router needs the gist of the turn, not its content.

### 4.3 Prerequisite

`main_graph.py:3` imports `SystemMessage` from
`instructor.v2.providers.anthropic.handlers` — an IDE autoimport from the wrong package. It must
come from `langchain_core.messages`. Nothing here works until that is fixed.

---

## 5. Edge cases

| Case | Behaviour |
|---|---|
| First message in a thread | `earlier` is empty → `NO_HISTORY` → the model judges the message alone, as today |
| History is all tool traffic | filtered out → `NO_HISTORY`; never an empty `{history}` block |
| `SummarizationMiddleware` compressed the qa history | the summary is a `SystemMessage`/`AIMessage` in the tail; anything that is not human or plain-AI is skipped, so at worst the history is shorter |
| Multimodal content blocks | `message.text` yields the text parts only |
| Empty `state['messages']` | `*earlier, current = ...` raises `ValueError`. The graph is never entered without a turn, so this stays unguarded rather than silently routing something invented |

**Cost:** no new model call — the router already makes one. It grows by at most 4 × 300 chars
(≈ 300 tokens) of prompt on a structured call that returns a single word.

---

## 6. Tests

New file `tests/integration/test_router_context.py` (routing lives next to the other eval-style
tests, not under a `tests/langchain/` directory that does not exist).

`StubChatModel` (`tests/agent_stubs.py`) has no `with_structured_output`; the base implementation
goes through `bind_tools`, which the stub deliberately no-ops. It needs an override that pops a
scripted `Route` from a queue **and records the messages it was called with**, so a test can assert
on the rendered history as well as the destination.

| Test | Setup | Assertion |
|---|---|---|
| fragment after a scheduler question | history: assistant "Which day suits you?"; new: "Monday works" | `booking` — the test M12 §5 said was missing |
| topic switch mid-booking | history: three booking turns; new: "actually, how much is tuition?" | `qa` — the case sticky routing would fail |
| qa follow-up | history: a tuition answer; new: "and for international students?" | `qa` |
| explicit booking request, no history | first message: "I'd like to book a consultation" | `booking` — unchanged from today |
| deadline mention | "when do applications close?" | `qa` |
| tool traffic is excluded | `_recent_turns` over a list holding an `AIMessage` with `tool_calls` and a `ToolMessage` of slot dicts | neither appears in the output; only spoken turns do |
| truncation and ordering | a long assistant turn plus six turns of history | ≤ 4 lines, newest last, each ≤ 300 chars |

The last two are unit tests of `_recent_turns` — it is a formatter, not a tool, so calling it
directly is right; the routing rows above still go through the model.

---

## 7. Order of work

1. Fix the `SystemMessage` import (§4.3).
2. `ROUTER_PROMPT_TEMPLATE` + `build_router_prompt`, with the two prompt paragraphs rewritten (§4.1).
3. `_recent_turns` + `route` (§4.2).
4. `StubChatModel.with_structured_output` (§6).
5. The test file.
6. Delete the "routing is stateless per turn" limitation note from `m12-booking-subagent.md` §5 and
   point it here.