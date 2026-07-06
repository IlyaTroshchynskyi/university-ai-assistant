# Compare Two Programs — fan-out / fan-in CrewAI Flow

**Date:** 2026-07-06
**Status:** Design approved (pending spec review)

## Overview

Add a "compare two programs" capability: given a message like *"compare the CS and
Data Science programs"*, gather information for **both** programs **in parallel**
(fan-out), then merge the two branches into one side-by-side answer (fan-in).

This is a learning exercise in expressing a **true parallel-branches-that-merge**
graph. The original goal was framed in LangGraph terms (`StateGraph`, typed state,
reducer, fan-out/fan-in). We deliberately build it on **CrewAI Flow** instead — the
framework the rest of this project already uses — because CrewAI Flow can express the
same shape natively, with no second orchestrator.

## Goals

- An explicit fan-out / fan-in graph: one node splits into two concurrent retrieval
  branches, a merge node waits for both and produces the comparison.
- **Genuine concurrency** — the two retrievals overlap in time, not run one-after-another.
- Reachable end-to-end from the running assistant via `/ask`.
- Teach the CrewAI equivalent of LangGraph's concurrent-write handling.

## Non-Goals (YAGNI for v1)

- Comparing more than two programs.
- A router guard for "only one program named" (see Known Simplifications).
- Session history / multi-turn memory inside the compare flow — it is single-turn.
- Connecting a real vector DB (the retriever stays a placeholder; the graph mechanics
  are observable regardless).

## Why CrewAI can do this (verified)

Inspected CrewAI **1.15.1** source (`crewai.flow.flow.Flow._execute_listeners`):
listeners triggered by the same upstream method are executed **in parallel** via
`asyncio.gather(*tasks)`, and `and_()` fan-in is supported. So two `@listen(extract)`
async nodes run concurrently, and a `@listen(and_(a, b))` node waits for both. The
"true parallel branches that merge" requirement is met on CrewAI natively.

## Architecture

New self-contained flow in `app/ai_assistant/compare_flow.py`:

```
          extract_programs   (@start)          LLM extracts 2 program names
                 │
        ┌────────┴────────┐                    fan-out: both listeners run
        ▼                 ▼                     CONCURRENTLY (asyncio.gather)
   retrieve_a         retrieve_b                each: await retrieve(program) -> own key
        └────────┬────────┘
                 ▼
              merge   (@listen(and_(retrieve_a, retrieve_b)))   fan-in: waits for both,
                 │                                               one LLM call -> side-by-side
                 ▼
              answer
```

## State model & the "reducer" question

```python
class CompareState(BaseModel):
    question: str = ''
    program_a: str = ''
    program_b: str = ''
    info_a: str = ''      # written ONLY by retrieve_a
    info_b: str = ''      # written ONLY by retrieve_b
    answer: str = ''
```

**Chosen approach: separate keys, no reducer.**

CrewAI Flow has **no reducer mechanism** (unlike LangGraph's `Annotated[list, add]`,
which declares how to merge concurrent writes to one key). CrewAI never merges state
for you. We don't need it, because parallel branches run cooperatively on one event
loop (`asyncio.gather`), not in threads:

- **Separate keys (`info_a` / `info_b`)** — no shared mutation, cannot race under any
  code. This is what we use.
- **Shared key, manual merge** — safe *if* each branch writes after its `await`, e.g.
  `self.state.results[program] = info` (synchronous item-assignment lands after the
  await). This is the hand-rolled equivalent of a reducer.
- **Unsafe pattern** — read a key, `await`, then write back a value derived from the
  stale read: the sibling branch's write is lost. This is exactly what a LangGraph
  reducer protects against, and why CrewAI's answer is "use separate keys."

The `info_a` / `info_b` split will carry a short comment documenting this contrast so
the learning point is explicit in the code.

## Components & reuse

- **`retrieve(query: str) -> str`** — a plain `async` function extracted into
  `app/ai_assistant/tools/retriever_tool.py` as the single source of truth for a
  lookup. `RetrieverTool._run` is refactored to call it. This mirrors the existing
  `booking_tools.py` pattern (plain `list_open_slots` + thin `BaseTool`). The two
  retrieval branches call `retrieve()` directly for genuine parallel async I/O.
- **`extract_programs`** (`@start`) — a small structured LLM call (same approach as
  `_CLASSIFIER_LLM` in `main.py`) that pulls the two program names from `question` into
  `program_a` / `program_b`.
- **`retrieve_a` / `retrieve_b`** (`@listen(extract_programs)`, `async`) — each awaits
  `retrieve(program_x)` and writes its own `info_x` key.
- **`merge`** (`@listen(and_(retrieve_a, retrieve_b))`) — one LLM call that turns
  `info_a` + `info_b` into a coherent side-by-side comparison written to `answer`.

## Integration with the main flow (`main.py`)

- `_classify_intent` gains a `'compare'` outcome (classifier prompt updated); its return
  type becomes `Literal['booking', 'cancel', 'qa', 'compare']`.
- `route_intent` can return `'compare'`.
- New node:
  ```python
  @listen('compare')
  async def compare_programs(self):
      sub = CompareProgramsFlow()
      await sub.kickoff_async(inputs={'question': self.state.question})
      self.state.answer = sub.state.answer
  ```
- `compare_programs` is added to `show_answer`'s `or_(...)`.
- Reachable via `POST /ask` with body `{"question": "compare the CS and DS programs"}`.

## Error handling

- The compare flow is single-turn and does not pause (no human-in-the-loop), so no
  persistence/resume concerns.
- If `retrieve()` raises, CrewAI's listener wrapper logs it; v1 lets the branch fail and
  the merge sees an empty `info_x`. Hardening (per-branch fallback text) is future work.

## Testing

Per project convention, exercise through the running assistant / LLM, not by calling
nodes directly:

1. Start the server (`uv run uvicorn app.main:app --reload`).
2. `POST /ask` with a comparison question.
3. Confirm in the `verbose` logs that `retrieve_a` and `retrieve_b` **start before
   either finishes** (interleaved) — evidence of true concurrency.
4. Confirm the answer is a single side-by-side comparison.
5. `plot()` the compare flow to visually verify the fan-out / fan-in shape.

Because the retriever is a placeholder, `info_a` / `info_b` are stub strings; the graph
structure and concurrency are still fully observable.

## Known Simplifications

- **Only one program named:** `extract_programs` still returns two fields (second may be
  empty); the merge handles a thin/empty side. A proper "too few programs → ask to
  clarify" router is deferred to a later iteration.

## Files touched

- `app/ai_assistant/compare_flow.py` — **new**: `CompareState`, `CompareProgramsFlow`.
- `app/ai_assistant/tools/retriever_tool.py` — extract plain `retrieve()`, thin the tool.
- `app/ai_assistant/main.py` — `'compare'` intent + `compare_programs` node + `or_`.
- `README.md` — document the compare capability (follow-up).
