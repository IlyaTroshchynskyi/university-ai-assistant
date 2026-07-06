# Compare Two Programs — fan-out / fan-in CrewAI Flow

**Date:** 2026-07-06
**Status:** Implemented (skeleton) — tool + main-flow wiring pending

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

- An explicit fan-out / fan-in graph: one node splits into two concurrent research
  branches, a merge node waits for both and produces the comparison.
- **Genuine, deterministic concurrency** — the two lookups overlap in time, and the
  parallelism is a visible part of the graph (not hidden inside an agent loop).
- Reachable end-to-end from the running assistant via `/ask`.
- Teach the CrewAI equivalent of LangGraph's concurrent-write handling.

## Non-Goals (YAGNI for v1)

- Comparing more than two programs.
- A router guard for "only one program named" (see Known Simplifications).
- Session history / multi-turn memory inside the compare flow — it is single-turn.
- Connecting a real vector DB (the retriever stays a placeholder; the graph mechanics
  are observable regardless).

## Two things verified against CrewAI 1.15.1 source

1. **Flow runs same-trigger listeners in parallel.** `Flow._execute_listeners` runs all
   listeners triggered by the same upstream method via `asyncio.gather(*tasks)`, and
   `and_()` fan-in is supported. So two `@listen(extract)` async nodes run concurrently
   and a `@listen(and_(a, b))` node waits for both.

2. **A single agent *can* run tools in parallel — but non-deterministically.**
   `CrewAgentExecutor._handle_native_tool_calls` runs multiple tool calls from one LLM
   turn through a `ThreadPoolExecutor` (unless a tool sets `result_as_answer` or
   `max_usage_count`, which forces sequential first-only). That means the "one agent
   fires two research tools at once" design is technically possible, **but** the
   parallelism only happens if the model chooses to emit both tool calls in a single
   turn, and it is hidden inside the agent loop (not plottable, not guaranteed).

Because this is a learning exercise about an **explicit parallel graph**, we express the
fan-out at the **Flow level** (finding #1), which is deterministic and visible, rather
than relying on an agent batching its tool calls (finding #2).

## Architecture

Self-contained flow in `app/ai_assistant/compare_flow.py`:

```
          extract_programs   (@start)          plain LLM call: extract 2 program names
                 │
        ┌────────┴────────┐                    fan-out: both listeners of the same
        ▼                 ▼                     trigger run CONCURRENTLY (asyncio.gather)
   research_a         research_b                each runs the research AGENT for ONE
        │                 │                     program -> writes its OWN state key
        └────────┬────────┘
                 ▼
              merge   (@listen(and_(research_a, research_b)))   fan-in: waits for both,
                 │                                              plain LLM call -> side-by-side
                 ▼
              answer
```

## State model & the "reducer" question

```python
class CompareState(BaseModel):
    question: str = ''
    program_a: str = ''
    program_b: str = ''
    info_a: str = ''      # written ONLY by research_a
    info_b: str = ''      # written ONLY by research_b
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

The `info_a` / `info_b` split carries a comment documenting this contrast so the learning
point is explicit in the code.

## Components

**One agent total** (research). Extraction and merge are plain, tool-free LLM calls.

- **`ProgramResearchCrew`** (`crews/compare_crew/compare_crew.py`) — a single-agent crew
  that researches **one** program. The flow fans it out (one kickoff per program), so the
  two runs execute in parallel. Its `program_researcher` agent carries the knowledge
  retriever tool (added by the user; a `# TODO` placeholder marks the spot). Config in
  `config/research_agents.yaml` + `config/research_tasks.yaml`.
- **`extract_programs`** (`@start`) — a plain `_LLM.call` that pulls the two program
  names out of `question` into `program_a` / `program_b` (parsed via `_parse_pair`).
- **`research_a` / `research_b`** (`@listen(extract_programs)`, `async`) — each awaits
  `ProgramResearchCrew().crew().kickoff_async(...)` for its program and writes its own
  `info_x` key.
- **`merge`** (`@listen(and_(research_a, research_b))`) — a plain `_LLM.call` that turns
  `info_a` + `info_b` into the side-by-side comparison in `answer`. It has **no tool on
  purpose**: it must not fetch or invent facts beyond the two summaries.

### Why merge is not a second agent

The fan-in is a pure text transform over two ready-made summaries — no tool use, no
reasoning loop, no crew machinery needed. A tool-wielding "comparison agent" would also
risk re-fetching and inventing facts beyond the summaries. So merge stays a plain LLM
call. (An earlier draft used a separate `ComparisonCrew`; it was removed.)

## Integration with the main flow (`main.py`) — pending

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
- Reachable via `POST /ask` with `{"question": "compare the CS and DS programs"}`.

## Error handling

- The compare flow is single-turn and does not pause (no human-in-the-loop), so no
  persistence/resume concerns.
- If a research kickoff raises, CrewAI's listener wrapper logs it; v1 lets the branch
  fail and merge sees an empty `info_x`. Per-branch fallback text is future work.
- If only one program is named, `research_b` gets an empty program and returns empty
  `info_b` (see `_research`'s empty-program guard).

## Testing

Per project convention, exercise through the running assistant / LLM, not by calling
nodes directly:

1. Add the retriever tool to `program_researcher`, then start the server.
2. `POST /ask` with a comparison question.
3. Confirm in the `verbose` logs that `research_a` and `research_b` **start before either
   finishes** (interleaved) — evidence of true concurrency.
4. Confirm the answer is a single side-by-side comparison.
5. `plot()` the compare flow to visually verify the fan-out / fan-in shape.

Because the retriever is a placeholder, `info_a` / `info_b` are stub strings; the graph
structure and concurrency are still fully observable.

## Known Simplifications

- **Only one program named:** `extract_programs` still returns two fields (second may be
  empty); the merge handles a thin/empty side. A proper "too few programs → ask to
  clarify" router is deferred to a later iteration.

## Files

- `app/ai_assistant/compare_flow.py` — `CompareState`, `CompareProgramsFlow`, the plain
  `extract`/`merge` LLM steps, `_parse_pair`, `_research`. **Done.**
- `app/ai_assistant/crews/compare_crew/compare_crew.py` — `ProgramResearchCrew`. **Done.**
- `app/ai_assistant/crews/compare_crew/config/research_agents.yaml`,
  `research_tasks.yaml` — the research agent + task. **Done.**
- Retriever tool on `program_researcher` — **pending (user adds).**
- `app/ai_assistant/main.py` — `'compare'` intent + `compare_programs` node + `or_`.
  **Pending.**
- `README.md` — document the compare capability. **Pending.**