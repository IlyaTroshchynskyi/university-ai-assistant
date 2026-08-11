# RAG & Agent Evaluation with DeepEval

**Date:** 2026-08-07
**Last synced with the code:** 2026-08-10
**Status:** Implemented and running, with several departures from the design and part of the
scope still outstanding — see *State as built*.

## Overview

Add a measurable quality gate for the LangChain assistant: a golden dataset built from
`docs/handbook.pdf` plus two DeepEval-driven integration suites — one for the retriever
(`KnowledgeService.search`) and one for the agent end-to-end (`POST /langchain-assistant`).

The suites live in `tests/integration/`, cost real money on every run (OpenAI calls for the
agent *and* for the LLM judge), and therefore run only behind an explicit `--run-eval` flag.
A plain `pytest` collects them and skips them with a reason.

A second, concrete motivation: the knowledge base is currently chunked by
`RecursiveCharacterTextSplitter`, which cuts tables mid-row. Three goldens target table-only
facts, so that replacing the splitter can be validated by re-running the same suite.

## State as built

What the code does that this design did not say, recorded so the two stop drifting apart.

| Area | Design | As built |
|---|---|---|
| Assertion | `assert_test` | Own `assert_metrics` coroutine — it measures on the metric objects themselves and logs the full score table, passes included. `assert_test` measures on *copies* and mentions only failures, so a passing score cannot be read back out of it |
| Preflight | Fixture counting points in `university_kb` through the `doc_type` filter | Written as a session-scoped autouse fixture, but it *probes* rather than counts: it runs one `search()` and skips both suites when nothing comes back or Qdrant is unreachable. Going through `search()` keeps the collection name and the `doc_type` filter in one place — the retriever's — instead of mirroring them in an `EVAL_DOC_TYPE`, which is why that setting is gone |
| Case ids | `pytest.param(id=str(case.id))` → `test_retriever[20]` | As designed — `golden_params()` wraps each case as `pytest.param(id=str(case.id))`. This is not cosmetic: pytest's own ids are positional in the *filtered* list, and the two layers filter differently, so `[golden19]` meant golden 21 in the agent suite and golden 20 in the retriever one. Table selection is a `tables` **marker** rather than the design's `-k table`, because `-k` matches substrings and a section named e.g. "Timetable" would join the run by accident |
| App behaviour | Non-goal: "evaluation is observation, not modification" | Broken on purpose twice, both times because a metric found a real defect. (1) `MAIN_CHAT_PROMPT` gained a "How you write the answer" block after `AnswerRelevancyMetric` scored 0.75 on a correct answer that ended in "let me know if you need anything else". (2) `MAIN_CHAT_PROMPT` and the `retriever` docstring gained the multi-hop rules — see *Multi-hop: one question, several searches* |
| Thresholds | 0.7 / 0.8 / 0.7 | Retuned against measured scores; see *Metrics and thresholds* |

Outstanding: `AGENTS.md` from *Deliverables outside `tests/`* is not written. Everything else in
that section landed on 2026-08-10 — the `README.md` section, all four Makefile targets, and the
`deepeval` move into the `dev` dependency group (which this document had recorded as done while it
was still a runtime dependency).

Both suites run their full sets again as of 2026-08-10: the agent slice was removed, and
`TestRetrieverEvaluation` was uncommented and un-sliced. `pytest tests/integration
--collect-only` reports **47** cases as of 2026-08-11 — 21 retriever, 20 agent, and the 6 routing
cases of the third suite, which this document predates.

Restoring the retriever class needed one change beyond deleting the comment markers: its
`pytestmark` now carries `pytest.mark.asyncio(loop_scope='session')`, for the same reason the
agent suite does. `get_qdrant_client` and the OpenAI client behind `get_embedder` are cached
singletons, so the first case binds them to whatever loop it ran on and every later case on a
fresh function loop would talk to a loop nobody is running.

## Goals

- Separate scores for the two layers, so a bad answer points at either retrieval or the LLM.
- A golden dataset that is a **file in git** — reviewable, diffable, and the single source of
  truth for what "correct" means.
- Per-case granularity: each golden is its own pytest case, named by id, with the judge's
  reason printed on failure.
- Zero cost and zero network on a normal `pytest` run.
- Table-integrity coverage that survives a chunking-strategy change.

## Non-goals

- No re-ingestion, no new collection, no embedding regeneration inside the tests. The suites
  read whatever is in the existing `university_kb` collection.
- No tool-selection metric (`ToolCorrectnessMetric`). The agent has exactly one tool today.
- No CI wiring. The suites are run locally, on demand.
- No synthetic golden generation (`Synthesizer`). Cases are written by hand from the handbook.
- No changes to `app/` behaviour. Evaluation is observation, not modification. *(Departed from
  once, knowingly: see the `MAIN_CHAT_PROMPT` row in *State as built*. The rule still stands —
  a metric may only drive an `app/` change when it has found a genuine defect, never to make a
  number go green.)*

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Layers | Retriever and agent, evaluated separately | A failing answer immediately attributes to search or to the LLM |
| Golden authoring | Hand-written from `docs/handbook.pdf`, in English | The handbook is English; a verbatim expected answer is checkable, and a synthesizer gets numbers wrong |
| Dataset shape | One JSON file, per-case `layers` tag | A fact lives in one place; small talk and negative cases are excluded from the retriever layer, where they are meaningless |
| Data source | Existing `university_kb` collection, as is | The user maintains ingestion; the eval must not spend money on embeddings or mutate the working collection |
| Gate | `--run-eval` pytest option + `evaluation` marker | Visible in `pytest --help`, skip carries a reason, and marker selection still composes with `-k` |
| Judge model | `gpt-4.1-mini`, configurable via `EVAL_JUDGE_MODEL` | Cheap enough to iterate on; one env var raises it to `gpt-4o` when a verdict must be trusted. **Not** `gpt-4o-mini`, which stalls on DeepEval's nested `Verdicts` schema — see the comment above `EVAL_JUDGE_MODEL` in `tests/integration/config.py` |
| Failure mode | One pytest case per golden, own `assert_metrics` | Shows exactly which question and which metric broke — and, unlike `assert_test`, prints every score against its threshold on passing runs too, so a near-miss is visible before it turns into a failure |
| Agent invocation | Real HTTP through `POST /langchain-assistant` | Covers router, schemas, service and agent — the path a user actually takes |

## Architecture

```
tests/integration/
├── __init__.py
├── config.py               # EvalSettings — judge model + one threshold per metric
├── conftest.py             # preflight fixture, metric factories, assert_metrics
├── goldens.py              # GoldenCase pydantic model, layer-filtered loader, golden_params
├── data/goldens.json       # 23 cases — source of truth
├── test_retriever_eval.py
└── test_agent_eval.py
```

The eval suites reuse the shared `app` / `not_auth_client` fixtures from `tests/conftest.py`.
That requires making the agent stub **opt-in** — see the next section.

### Step 0 — prerequisites in existing code

None of the eval work can start until the suite collects and the app imports. Two pre-existing
bugs (both present on `master`) block that:

1. `app/ai_assistant_langchain/agent.py:12` imports `CustomContext` from
   `app.ai_assistant_langchain.schemas`, but the class lives in `agent_schemas.py`. This breaks
   the import chain `main.py → router → service → agent_service → agent`, so **both** `pytest`
   (via `tests/conftest.py`) and application startup fail. Fix the import.
2. `tests/agent_stubs.py:21` patches `app.ai_assistant_langchain.agent.get_model_factory`; the
   function is named `_get_model_factory`. Raises `AttributeError` as soon as bug 1 is fixed.
   Point the patch target at the real name.

Then the agent stub stops being app-wide and moves to the only file that needs it:

- `tests/dependencies.py` — the `deps` list in `override_app_test_dependencies` no longer
  carries the agent entry. The function and `_DepOverride` stay, as the hook for future
  app-wide overrides. A `remove_dependency_override` helper is added: `app` is session-scoped,
  so a per-test override has to be taken back off or it leaks into every later test.
- `tests/conftest.py` — `TestBaseAgentClass` and `TestBaseClientAgentClass` are deleted.
- `tests/api/test_chat_checkpointer.py` — gains `TestBaseCheckpointerClass`, whose autouse
  fixture installs `create_assistant_agent → get_test_agent` and removes it on teardown. Only
  `TestChatMemoryCheckpointer` and `TestChatToolCallsCheckpointer` inherit it — those drive the
  real graph and never mock `run_agent`.
- `TestChatWithUserCheckpointer` drops to plain `TestBaseClientClass`: it patches
  `AgentService.run_agent`, so it never needs an agent at all.

Why an override and not `patch`: `AgentService` receives the graph through
`Depends(create_assistant_agent)`, and that `Depends` holds a reference to the function object
itself, so patching the module attribute never reaches it. `app.dependency_overrides` is the
only hook.

Two consequences worth knowing. In non-stubbed tests `Depends(create_assistant_agent)` now
resolves for real and constructs a `ChatOpenAI` object — construction only, no network. And a
test that hits `/langchain-assistant` without stubbing anything will reach OpenAI; without
`--run-eval` the key in the environment is `test-key`, so that surfaces as a loud 401, not as
silent spending.

**Status: done.** `pytest` reports 42 passed, `ruff check` is clean, and `create_app()` builds.

### Golden case schema

```json
{
  "id": 1,
  "question": "How much is the annual tuition for the Data Science program?",
  "expected_answer": "$26,000 per year.",
  "context": ["Data Science (BSc, 4 years) ... Annual tuition: $26,000"],
  "layers": ["retriever", "agent"],
  "section": "Programs"
}
```

- `context` — verbatim passages from the handbook. **Dropped from `GoldenCase` on 2026-08-10:**
  the design had it as ground truth for `ContextualRecallMetric` and `ContextualPrecisionMetric`,
  but neither reads `LLMTestCase.context` — both take `retrieval_context`, which the suites fill
  from the live retriever. `HallucinationMetric` is the one metric that would use these blocks.
  They are still in the JSON, and pydantic ignores them.
- `layers` — which suites consume the case. `["agent"]` only for small talk and the negative
  case.
- `section` — the handbook section, so a failure report shows which part of the knowledge base
  regressed. Not used by any metric.

`goldens.py` exposes:

```python
class Layer(StrEnum):
    RETRIEVER = 'retriever'
    AGENT = 'agent'

class GoldenCase(BaseModel):
    @property
    def is_table(self) -> bool: ...   # section ends in '(table)' -> the `tables` marker

def load_goldens(layer: Layer) -> list[GoldenCase]: ...
def golden_params(layer: Layer) -> list[pytest.param]: ...   # id=golden id, + `tables` marker
```

Validation happens at load time, so a malformed dataset fails loudly instead of surfacing as a
confusing metric error. `id` is a plain number — the case's position in the dataset. Numbers keep
the dataset short to read and rename-free to extend; `section` is what tells you which part of
the handbook a failing case came from.

The design had that `id` reach pytest through `pytest.param(id=str(case.id))`, giving node ids
such as `test_retriever[20]`. The suites instead parametrize the `GoldenCase` objects directly,
so pytest falls back to positional names: `test_agent_answer[golden0]`. The question is still
identified on failure — `assert_metrics` logs it — but the node id no longer says which golden
ran, and `-k` can no longer select a case by id. Restoring `pytest.param` is a one-line change in
each suite if that selection is wanted back (`eval-tables` below depends on it).

### Dataset inventory — 23 cases

| id | Topic | Section | Layers |
|---|---|---|---|
| 1 | Data Science tuition | Programs | retriever, agent |
| 2 | Architecture length and cost | Programs | retriever, agent |
| 3 | Monthly plan surcharge | Tuition & Fees | retriever, agent |
| 4 | Discount applies to tuition only | Tuition & Fees | retriever, agent |
| 5 | Merit scholarship rule | Scholarships | retriever, agent |
| 6 | Dean's Excellence requirements and award | Scholarships | retriever, agent |
| 7 | One scholarship per student | Scholarships | retriever, agent |
| 8 | Software Engineering with Merit | Programs + Scholarships | retriever, agent |
| 9 | Mechanical first-year total | Programs + Tuition & Fees | retriever, agent |
| 10 | Women in STEM + Data Science | Programs + Scholarships | retriever *(agent layer removed 2026-08-10)* |
| 11 | Minimum GPA and exam score | Admissions | retriever, agent |
| 12 | English test requirement | Admissions | retriever, agent |
| 13 | Application window opens and closes | Key Dates | retriever, agent |
| 14 | Deposit amount, deadline and credit | Key Dates | retriever, agent |
| 15 | Studio annual cost | Housing | retriever, agent |
| 16 | Housing application prerequisites | Housing | retriever, agent |
| 17 | Library hours | Campus Life | retriever *(agent layer removed 2026-08-11)* |
| 18 | Cafeteria closed Sunday | Campus Life | retriever *(agent layer removed 2026-08-11)* |
| 19 | Academic year dates | Key Dates (table) | retriever, agent |
| 20 | Lab fee and who pays it | Tuition & Fees (table) | retriever, agent |
| 21 | Civil Engineering summary row | Tuition summary (table) | retriever, agent |
| 22 | No medical faculty | — | agent |
| 23 | Greeting | — | agent |

21 cases feed the retriever suite, 20 feed the agent suite — 41 in total, not the 44 this
document originally planned.

Cases 17 and 18 lost their `agent` tag on 2026-08-11, when `find_place` was registered on the
agent. Both ask for a named campus place's opening hours, which is verbatim what the tool's
prompt bullet claims, so the agent stopped answering them out of the handbook and started
routing them to DynamoDB — a table the eval session does not create. The questions are still
worth asking; they are just a *routing* assertion now, and `test_tool_routing_eval.py` makes it
(`place-hours`, `place-where`). On the retriever side nothing changed: the Campus Life passages
are in the handbook and their retrieval is still scored.

Case 10 lost its `agent` tag on 2026-08-10. Its answer was right every time, but
`FaithfulnessMetric` kept scoring 0.667–0.714 on it (roughly four runs in six) for one reason:
the agent calls the award the *Northwood* Merit Scholarship, which is its name in the handbook,
while the chunks retrieved for this question say only "Merit Scholarship" — so the judge read
the correct name as a contradiction. The case still earns its keep in the retriever suite, where
that never arises.

The multi-hop cases (8–10) require passages from two different handbook sections in one answer
— `ContextualRecallMetric` shows whether hybrid search pulled both or only one. Two of the three
(8 and 9) still cover that ground on the agent side.

### Multi-hop: one question, several searches

Case 8 (*"admitted to Software Engineering with a 3.8 GPA, what will I pay?"*) failed at
`Correctness` 0.21–0.57 for a reason worth recording, because it is a defect the suite was built
to find. The agent issued a single query, `tuition for Software Engineering program`. That
brought back the $24,500 tuition **and** the handbook's worked example ending in $12,250 — but
not the eligibility rule (`GPA of 3.7 or higher`, `Northwood`, `50% tuition discount`). Having no
rule that connects a 3.8 GPA to the discount, the agent hedged: *"if you qualify for a
scholarship … your tuition would be…"*. Correct behaviour on the context it had; wrong answer.

The knowledge base was never the problem — the same `search()` with
`Software Engineering tuition with Merit Scholarship 3.8 GPA` returns all of it. Two
instructions were actively steering the agent to one query:

- the `retriever` docstring promised that "a single query with the program name returns all of
  its information at once", with no caveat that other *sections* are not covered;
- `MAIN_CHAT_PROMPT` said "pick the single most appropriate tool" and nothing about splitting a
  question.

Both were fixed on 2026-08-10: the docstring now scopes its promise to within one program and
says scholarship eligibility, fees, deadlines and policies need their own query; the prompt
gained a rule to search once per part when an answer combines a program's cost with a
scholarship (or an amount with the condition attached to it), and a rule never to state a
discounted amount without having found the rule that grants it — *"a worked example in the
handbook is not that rule."*

Result on case 8: two tool calls (`tuition for…`, `scholarships for…`), and
`Answer Relevancy 1.000 · Correctness 0.848 · Faithfulness 1.000`, all passing. Case 10 went
from `Correctness` 0.431 to 0.673. Case 9 was not affected — its problem is the question/answer
width mismatch described above, not retrieval.

Case 22 asserts the agent says NIT has no medical faculty instead of inventing one; case 23
asserts a greeting is answered directly, with no invented facts.

### The rule a golden must obey: the question must be as wide as its answer

**A `question` that asks for less than its `expected_answer` states cannot be satisfied.**
`Correctness` requires every figure in the expected answer to appear in the output;
`AnswerRelevancy` marks anything the question did not ask for as an irrelevant statement. A
narrow question with a broad expected answer therefore pits the two metrics against each other,
and no answer scores well on both.

Cases 6, 13 and 14 were written that way and were rewritten on 2026-08-10 once the failures
showed up:

| id | Was | Now |
|---|---|---|
| 6 | *"What do I need to qualify for the Dean's Excellence Award?"* — but the expected answer also states the $8,000 award | *"…, and what does it give?"* |
| 13 | *"When does the application … close?"* — but the expected answer also states the opening date | *"When does the application window … open and close?"* |
| 14 | *"By when do I have to pay the tuition deposit?"* — but the expected answer also states the $1,000 and the credit | *"How much is the tuition deposit, when is it due, and what happens to it?"* |

Case 6 scored `AnswerRelevancy` **0.500** before the rewrite and 0.800 after; 13 and 14 score
1.000. Case 5 was the model to copy — *"Who qualifies … **and what does it give?**"* — which is
why it never had the problem.

Cases 3, 11 and 16 have a milder version of the same mismatch (the expected answer adds a
contrastive fact the question did not ask for: the no-surcharge plans, the higher cutoffs, the
$500 housing deposit). They have not been rewritten — the judge has so far scored the contrast
as relevant — but they are the first place to look if `AnswerRelevancy` fails on them.

### The three table cases

These exist to measure whether chunking preserved table structure. They are cases 19–21, and all
three are sectioned `… (table)`, which `GoldenCase.is_table` turns into a `tables` marker, so
`make eval-tables` (`-m tables`) selects exactly those six cases — three per suite. Verified
2026-08-10.

| id | Question | What breaks if chunking splits the table |
|---|---|---|
| 19 | *"When does the Fall term start and end, and when does the Spring term run?"* | These dates appear **only** in the "Academic year" table — nowhere in prose. A broken table means no answer exists at all |
| 20 | *"How much is the lab fee and which students pay it?"* | The row is `Lab fee \| $300 / year \| students in Computer Science, Engineering, and Natural Sciences`. A cut between cells returns the amount without the audience |
| 21 | *"For Civil Engineering, what degree, how many years, and what is the annual tuition?"* | The row sits after the Tuition-summary table's page break (the extracted text repeats the header line). Exactly where a character-count splitter cuts blindly |

## Test flows

### Retriever suite (`test_retriever_eval.py`)

```
golden → KnowledgeService.search(question)          # list[ScoredPoint]
       → passages = [hit.payload['text'] for hit in hits if hit.payload]
       → LLMTestCase(input, actual_output='', retrieval_context=passages,
                     expected_output=expected_answer)
       → await assert_metrics(case, retriever_metrics())
```

`search()` returns the raw hits, so the suite takes `payload['text']` from each — the
contextual metrics score passages individually and must never receive one joined string.
`actual_output` is unused by the contextual metrics but required by `LLMTestCase`; the suite
omits it and lets `LLMTestCase` default it.

`search()` is called with its defaults — `limit=3`, `doc_type='general'` — because those are
the defaults the `retriever` tool uses. Evaluating the retriever under any other parameters
would measure a configuration the agent never runs. (`limit` was 4 when this was designed; it
was cut to 3 on the evidence of these very goldens — the fourth chunk never added coverage and
cost a quarter of the returned text. See the `search` docstring.)

Consequently the handbook must have been ingested with `doc_type='general'` (the default of
`POST /documents`), otherwise the filter excludes every chunk. The preflight that was to catch
this as a skip does not exist, so today that mistake shows up as 21 cases failing on empty
contexts.

### What the retriever suite measures, once it was actually run

Four full runs on 2026-08-10, 21 cases each:

| Metric | Result |
|---|---|
| `ContextualRecall` | **1.000 on every case, every run** — once golden 2 was fixed, below. Nothing the expected answers need is missing from what the search returns |
| `ContextualPrecision` | 1.000 on most cases. Three exceptions, of which one is real: see below |
| `ContextualRelevancy` | 0.095–0.682, median ~0.28, and **reproducible to within 0.03** — it measures chunk bulk, not search quality |

Three findings worth keeping:

1. **Recall is one verdict per sentence of the expected answer**, so a one-sentence golden makes
   it strictly binary. Golden 2's expected answer was the fragment *"5 years, at $22,000 per
   year."*, whose two facts sit in two different chunks; the judge alternated between "both are
   there, in nodes 1 and 3" and "not combined in one sentence", giving 1.000 · 0.000 · 1.000 ·
   0.000 across measurements. Split into two sentences it scores 1.000 three times running.
   Goldens 1, 15 and 21 are still single fragments; 21 packs three facts into one and is next.
2. **Precision quantises too.** With `limit=3` the attainable values around the bar are 0.833
   and 0.583, so a 0.7 threshold means, in practice, "the top-ranked chunk must be relevant".
   Worth remembering that the agent concatenates all three chunks into one tool result, so their
   order never reaches the model — this metric grades something the pipeline does not use.
3. **Golden 19 fails precision reproducibly** (0.500, four runs of four): asking when the terms
   run puts a payment-plans chunk above the academic-year table. Recall is still 1.000 — the
   dates come back, just ranked second.

### Agent suite (`test_agent_eval.py`)

```
golden → user_id = uuid4()
       → POST /langchain-assistant {query, user_id}
       → answer = response.message
       → retrieval_context = ToolMessage contents from the graph state for thread_id=user_id,
                             minus any whose text is tools.NO_RESULTS
       → LLMTestCase(input, actual_output=answer, expected_output, retrieval_context)
       → await assert_metrics(case, agent_metrics(has_context=bool(retrieval_context)))
```

The request goes through the shared `not_auth_client` fixture. After step 0 that client talks
to an unmodified app, so no eval-specific app wiring is needed. Because that client is
session-scoped, the suite must run on the session loop as well —
`pytest.mark.asyncio(loop_scope='session')` in the module's `pytestmark`. On the default
function loop the test hangs: the client and the agent's cached clients belong to a loop nobody
is running any more.

The `NO_RESULTS` filter matters: when the knowledge base has nothing, the tool returns that
sentence, and it is a message *about* the absence of context, not context. Counted as
`retrieval_context` it would both switch `FaithfulnessMetric` on and give it a "fact" to judge
against. The constant lives in `app/ai_assistant_langchain/tools.py` precisely so the suite can
recognise it without duplicating the wording.

A fresh `user_id` per case is mandatory: `create_assistant_agent` uses an `InMemorySaver`
keyed by `thread_id`, so a shared id would concatenate all 23 questions into one conversation
and eventually trigger `SummarizationMiddleware`.

`retrieval_context` is read back from the compiled graph's state rather than by patching
`get_knowledge_service`. No patching, no second run, and the passages are exactly what the LLM
saw. When the agent called no tool (small talk), both `FaithfulnessMetric` and
`AnswerRelevancyMetric` are omitted and the reply is scored on correctness alone — see
*Metrics and thresholds*.

## Reference material for writing the tests

### DeepEval documentation

Read in this order — the first three cover everything the suites do.

| Page | Why |
|---|---|
| [Test cases](https://deepeval.com/docs/evaluation-test-cases) | What each `LLMTestCase` field means and which metric reads which field. The single most useful page |
| [Unit testing in CI/CD](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd) | `assert_test` inside pytest — background only. The suites use their own `assert_metrics` instead; read this to see what it replaces |
| [Metrics introduction](https://deepeval.com/docs/metrics-introduction) | Thresholds, `model=`, `async_mode`, how a metric reports failure |
| [Getting started](https://deepeval.com/docs/getting-started) | Skim only — it pushes the Confident AI cloud, which we do not use |
| [RAG evaluation guide](https://deepeval.com/guides/guides-rag-evaluation) | Why retriever and generator are measured separately — the reasoning behind the two-suite split |
| [`ContextualRelevancy`](https://deepeval.com/docs/metrics-contextual-relevancy) · [`ContextualRecall`](https://deepeval.com/docs/metrics-contextual-recall) · [`ContextualPrecision`](https://deepeval.com/docs/metrics-contextual-precision) | The retriever suite's three metrics |
| [`AnswerRelevancy`](https://deepeval.com/docs/metrics-answer-relevancy) · [`Faithfulness`](https://deepeval.com/docs/metrics-faithfulness) | The agent suite's two ready-made metrics |
| [`GEval`](https://deepeval.com/docs/metrics-llm-evals) | The custom Correctness metric — `criteria` vs `evaluation_steps`, and `evaluation_params` |
| [pytest parametrize](https://docs.pytest.org/en/stable/how-to/parametrize.html) | One golden = one test case, with a readable node id |

### Code to read in this repo

| Where | What you get from it |
|---|---|
| `tests/integration/conftest.py` | `retriever_metrics()` / `agent_metrics(has_context)` — already built, just call them. Plus `assert_metrics`, which measures, logs the table and asserts, and the autouse preflight |
| `tests/integration/config.py` | `get_eval_settings()` — the judge model and every threshold, each with the measurements that set it |
| `tests/integration/goldens.py` | `golden_params(Layer.RETRIEVER \| Layer.AGENT)` — parametrize on this, not on `load_goldens`, or the node ids lose the golden id |
| `tests/conftest.py:72`, `:82`, `:129` | The `app` and `not_auth_client` session fixtures and `TestBaseClientClass`, which hands `self.not_auth_client` to the test |
| `tests/api/test_chat_checkpointer.py:76-89` | How a request to `/langchain-assistant` is posted and the response parsed |
| `tests/api/test_chat_checkpointer.py:202-205` | **The pattern the agent suite needs**: `await agent.aget_state({'configurable': {'thread_id': user_id}})`, then pick the `ToolMessage`s out of `state.values['messages']` |
| `app/ai_assistant_langchain/tools.py:13`, `:16-29` | `NO_RESULTS`, and the `retriever` tool calling `search(query)` with no extra arguments — that is why the retriever suite must use the same defaults |
| `app/ai_assistant/university_knowladge/knowledge_service.py:98-121` | `search()` — the defaults `limit=3`, `doc_type='general'`, and the `list[ScoredPoint]` it returns, one entry per passage |
| `app/ai_assistant/university_knowladge/knowledge_service.py:212` | `get_knowledge_service()` — the retriever suite's entry point, wired to the real Qdrant |
| `app/ai_assistant_langchain/schemas.py:6-12` | Request body `{query, user_id}` (`query` is capped at 128 chars) and the `{message}` response |
| `app/ai_assistant_langchain/agent.py:17-18` | `create_assistant_agent` is `@lru_cache`d — calling it in a test returns the *same* graph the endpoint used, which is what makes reading state back possible |

### API shapes, as verified against the installed deepeval 3.9.6

```python
await metric.a_measure(test_case)   # what assert_metrics awaits, one task per metric

LLMTestCase(
    input=...,             # the question
    actual_output=...,     # the answer; defaulted in the retriever suite
    expected_output=...,   # golden.expected_answer
    retrieval_context=[...],  # list of passages, not one joined string
)
```

Each metric is an independent judge call, so `assert_metrics` runs them inside an
`asyncio.TaskGroup` and only then reads `metric.score` / `.threshold` / `.success` / `.reason`
off the objects.

### Traps worth knowing before you start

1. **Async all the way.** The suites `await assert_metrics(...)`, which awaits
   `metric.a_measure`. DeepEval's synchronous `assert_test` was the original plan and is a trap
   inside an async test: with `run_async=True` it calls `loop.run_until_complete`, finds a loop
   already running, and applies `nest_asyncio` to re-enter it. That works only on the stdlib
   loop. `tests/conftest.py:88` forces the stdlib loop factory instead of uvloop for its own
   reasons — keep the hook, and the trap stays out of reach either way.
2. **The agent suite needs the session loop.** See *Agent suite* above: `pytestmark` carries
   `pytest.mark.asyncio(loop_scope='session')` because `not_auth_client` is session-scoped.
   Without it the test hangs rather than fails, which is a slow thing to debug.
3. **`GEval` scores are noisy.** Same answer, same judge, five runs: 0.675 to 0.839. It reads
   the logprobs of the judge's score tokens and returns a weighted sum, so a threshold placed
   inside that spread turns into a coin flip. See `EVAL_CORRECTNESS_THRESHOLD` in
   `tests/integration/config.py` for the measurements and the two real cures.

## Metrics and thresholds

All metrics use one judge model. It and the thresholds live in `tests/integration/config.py` — one
place to tune, overridable from `.env` without touching code.

| Suite | Metric | Threshold | Reads |
|---|---|---|---|
| retriever | `ContextualRelevancyMetric` | **0.08** | question + retrieved passages |
| retriever | `ContextualRecallMetric` | 0.7 | expected answer + retrieved passages |
| retriever | `ContextualPrecisionMetric` | 0.7 | question + expected answer + retrieved passages |
| agent | `AnswerRelevancyMetric` | **0.7**, temporarily | question + answer (skipped when no tool call) |
| agent | `FaithfulnessMetric` | **0.7** | answer + retrieval context (skipped when no tool call) |
| agent | `GEval` "Correctness" | **0.6** | answer + expected answer |

The design set a flat 0.7 / 0.8 / 0.7. Three of those were wrong once real scores existed, and
all are now one threshold per metric — they measure different things and are not comparable.
The reasoning is kept next to the fields in `tests/integration/config.py`; in short:

- **Contextual relevancy 0.08.** It scores the *share* of retrieved sentences that bear on the
  question, so chunk size caps it and a narrow question caps it harder. 0.25 was itself a guess
  and failed nine of the 21 cases while recall scored 1.000 on every one of them; measured, the
  spread is 0.095–0.682 with a median around 0.28. 0.08 sits under the floor. It is a regression
  detector, not a quality bar, and there is a TODO to raise it once the structured chunker lands.
- **Correctness 0.6.** `GEval` noise, measured at 0.675–0.839 on one unchanged correct answer.
  0.7 sat in the middle of that spread and failed roughly one run in five for no reason.
- **Faithfulness 0.7**, down from 0.8, on the same quantisation grounds — but this is the
  hallucination check, and the slack is real: one ungrounded claim in a four-claim answer now
  passes. Of the three lowered thresholds this is the one to raise first.
- **Answer relevancy 0.7, temporarily.** 0.8 first, then 0.75, then 0.7 — each step taken
  against a measured score, not on principle. See *quantisation* below.

### Quantisation — the trap behind both failures so far

`AnswerRelevancy` and `Faithfulness` both score a *ratio of statements*: relevant statements
over all statements, non-contradicted claims over all claims. A short answer produces three to
five of them, so the score can only take a few values — four statements give 1.00, 0.75, 0.50,
0.25, 0.00 and nothing in between.

Any threshold above `1 - 1/N` therefore means "every statement must land", and one wobble from
the judge is a red build. Both failures seen so far are exactly this: `AnswerRelevancy` 0.75 vs
0.8 (three of four statements) and `Faithfulness` 0.75 vs 0.8 (three of four claims). Neither
answer was wrong.

Two things follow. Thresholds for statement-ratio metrics must be read as "how many statements
may miss", not as a percentage of quality. And when the answer is short, the honest fix is a
metric that does not quantise this coarsely — explicit `evaluation_steps`, or `DAGMetric` — not
a threshold nudged down until the noise fits under it.

A threshold can also land in a gap no answer can reach. Case 6 draws seven statements, so its
only nearby scores are 6/7 = 0.857 and 5/7 = 0.714; the 0.75 threshold sat between them, and the
case flipped from pass to fail on a single verdict with nothing in between. Nothing was measured
by the 0.036 that separated 0.714 from the bar — which is why the threshold moved to 0.7, below
both attainable values. Note what it costs: a *three*-statement answer still needs all three to
land, since 2/3 = 0.667 is under 0.7.

**A greeting cannot be scored this way at all.** Answer relevancy needs a question for the
answer to be relevant *to*, and "Hi there!" is not one — the judge scored a perfectly good
"Hello! How can I assist you…" at 0.500 because it "uses 'Hello!' instead of matching the input
greeting". That is why `agent_metrics` now drops answer relevancy along with faithfulness when
the agent called no tool, leaving correctness — which judges the reply against the golden's
description of a good greeting, and gives it 1.000.

The `GEval` criterion states that numbers, amounts and dates must match `expected_answer`
exactly, while wording — including extra helpful phrasing — may differ freely. Note the
division of labour that creates: *Correctness* deliberately forgives padding, so it is
`AnswerRelevancy` that polices it. That is how the "let me know if you need anything else"
closer was caught, at 0.75, while Correctness scored the same answer 1.0.

## Configuration

The evaluation knobs live in `tests/integration/config.py`, in an `EvalSettings(BaseSettings)`,
under an `EVAL_` prefix. They started out in `app/settings.py` and moved on 2026-08-10: the running
FastAPI app imports that module and has no use for a judge model, while the measurement history
behind each threshold is test documentation, not application configuration.

A test-only settings class had been tried and dropped once before, because it also carried
`QDRANT_COLLECTION` and two independent collection names let the preflight pass on one collection
while the retriever searched another. That objection is answered rather than ignored: `EvalSettings`
holds `EVAL_` fields **only**, and the preflight calls `search()` instead of naming a collection, so
where to search is still written in exactly one place.

The design had one threshold field per *suite*; the code has one per *metric*, for the reasons
in the section above.

| Field | Default | Purpose |
|---|---|---|
| `EVAL_JUDGE_MODEL` | `gpt-4.1-mini` | LLM judge for every DeepEval metric |
| `EVAL_MODEL_API_KEY` | `''` | Key for the judge. Unset means "bill the same one as the app" — but only because `judge_model()` passes `None`; deepeval treats `''` as a real key and raises before any metric runs |
| `EVAL_CONTEXTUAL_RELEVANCY_THRESHOLD` | `0.08` | Capped by chunk size; a regression detector, set just under the measured floor |
| `EVAL_CONTEXTUAL_RECALL_THRESHOLD` | `0.7` | Coverage of what the expected answer needs |
| `EVAL_CONTEXTUAL_PRECISION_THRESHOLD` | `0.7` | Useful passages ranked above the noise |
| `EVAL_ANSWER_RELEVANCY_THRESHOLD` | `0.7` | Share of the answer's statements that bear on the question; lowered temporarily for quantisation |
| `EVAL_FAITHFULNESS_THRESHOLD` | `0.7` | Answer grounded in what the tool returned; lowered for quantisation, at the cost of one ungrounded claim in four |
| `EVAL_CORRECTNESS_THRESHOLD` | `0.6` | Threshold for the GEval criterion; lowered for judge noise |

The collection is **not** among them: the preflight goes through the retriever, which reads
`QDRANT_COLLECTION` from `app.settings`. There is no fixture for any of this —
`get_eval_settings()` is a cached singleton, so the code that needs a value just calls it.

## The gate

`tests/conftest.py` (the root conftest — `pytest_addoption` only works there):

- `pytest_addoption` registers `--run-eval`.
- `pytest_collection_modifyitems` attaches a skip marker to every `evaluation`-marked item when
  the flag is absent, with the reason `needs --run-eval (real OpenAI and Qdrant calls)`.
- `pytest_configure` registers the marker with `config.addinivalue_line('markers', ...)`, so
  `[tool.pytest.ini_options]` in `pyproject.toml` stays untouched.

Each module in `tests/integration/` declares `pytestmark = pytest.mark.evaluation` — the agent
suite as a list, alongside `pytest.mark.asyncio(loop_scope='session')`.

### Required fix in `pytest_configure`

`tests/conftest.py` currently runs `os.environ.setdefault('OPENAI_API_KEY', 'test-key')`. In
`pydantic-settings`, an environment variable outranks `.env`, so this stub would shadow the
real key and every OpenAI call in the eval would fail with 401.

Fix: set the stub only when `--run-eval` was not passed. `pytest_configure` receives `config`,
so `config.getoption('--run-eval')` is available at that point. Under `--run-eval` the key is
left alone and resolves from the environment or `.env` as usual.

## Error handling

| Condition | Behaviour |
|---|---|
| `--run-eval` absent | Skipped at collection with a reason. Default `pytest` stays free and offline. **Implemented** |
| A metric scores below its threshold | The test fails; `assert_metrics` logs every score against its threshold and the assertion message names each failing metric with the judge's reason. **Implemented** |
| OpenAI request fails mid-run | The test fails. A network or quota error is a real failure and must not be swallowed. **Implemented** — by doing nothing |
| Qdrant unreachable, or the collection does not exist | *Designed:* the preflight errors out, taking the suite with it. Not caught, because both errors already name the URL and the collection, and swallowing them into a skip would let a broken `make eval` exit 0. *Today:* no preflight, so the error surfaces from whichever test touches Qdrant first |
| Collection empty for `doc_type='general'` | *Designed:* skip with `Collection 'university_kb' has no chunks with doc_type='general' — ingest docs/handbook.pdf via POST /documents`. *Today:* not implemented — this burns judge calls on empty contexts and reports it as 21 metric failures |
| `OPENAI_API_KEY` missing or still the test stub | *Designed:* skip with an explanatory message. *Today:* not implemented — surfaces as a 401 from OpenAI |

The preflight was to use a filtered `count()` on the collection — cheap, and it distinguishes
"no Qdrant" from "Qdrant is up but nothing searchable was ingested", which are two different
user mistakes. It is the largest single piece of this design still unbuilt; the three rows
marked *Today* above are all downstream of its absence.

## Re-running after a chunking change

The intended workflow when the splitter is replaced:

1. Delete the existing points for the handbook, then re-ingest via `POST /documents`.
2. Re-run `make eval` and compare, in particular the three table cases (19–21).

**Why step 1 matters.** Point ids are `uuid5(f'{source}:{index}')`. A new splitter that
produces fewer chunks leaves the tail points from the previous run in the collection — stale
text, still searchable, silently mixed into results. A before/after comparison over a polluted
collection is not a valid comparison. Dropping the collection (or deleting by `source` filter)
before re-ingesting is required.

This step is a manual operation outside the eval suites; the suites only read.

## Cost and runtime

Per full run: ~21 embedding requests (retriever suite), 23 agent invocations, and — measured on
one agent golden with all three metrics — around ten judge calls per case, so on the order of
400 for the whole set rather than the 130 originally guessed. Each metric spends more than one
call: statements/truths extraction, then verdicts, then the reason.

On `gpt-4.1-mini` that is still a few cents. Runtime: one agent case with three metrics takes
~16s wall clock, most of it judge latency, and the metrics within a case already run
concurrently.

## Deliverables outside `tests/`

Written **after** the implementation lands and is verified:

1. **`Makefile`** — **done**, all four:

   | Target | Command | State |
   |---|---|---|
   | `eval` | `pytest tests/integration --run-eval --log-cli-level=INFO` | Done. `--log-cli-level=INFO` is what makes `assert_metrics` print the score table on passing runs, which is the point of having written it |
   | `eval-retriever` | same, on `test_retriever_eval.py`, `-v` | Done. Makes no agent call at all, so it is the cheap loop for iterating on chunking or search |
   | `eval-agent` | same, on `test_agent_eval.py`, `-v` | Done |
   | `eval-tables` | same, `-m tables -v` | Done. Selects goldens 19–21 in both suites — six cases |

2. **`README.md`** — **done.** An *Evaluating the RAG pipeline* section: what the two suites
   measure, that it costs money and a plain `pytest` does not, the prerequisites (Qdrant up,
   handbook ingested, the preflight that skips when it is not, `EVAL_MODEL_API_KEY`), where the
   thresholds live, and how to read a red run.

3. **`AGENTS.md`** (new file at the repo root) — **not written.** Operating instructions for
   coding agents: repository layout, how to run the normal offline test suite, the fact that
   `tests/integration/` is gated behind `--run-eval` and **must not** be run without being
   asked (it spends money), where the golden dataset lives, and how to add a case to it.

4. **`pyproject.toml`** — **done**, but only as of 2026-08-10. This document claimed it earlier
   while `deepeval` was still in the runtime `[project] dependencies`, so every deployed image
   carried the evaluation framework and its telemetry writer. It is in `[dependency-groups] dev`
   now, alongside the same "never imported by `app/`" comment `polyfactory` carries.

## Verification

The implementation is done when:

- ✅ After step 0 alone, `pytest` collects and the existing suites pass — checked before any
  eval code exists.
- ✅ `pytest` (no flag) collects `tests/integration/` and reports 43 skipped eval cases, makes
  no network calls, and the existing suites still pass.
- ✅ `pytest --run-eval tests/integration -v` runs all 43 cases against live Qdrant and OpenAI,
  each named by its golden id — `[19]`, not `[golden5]`.
- ⬜ With Qdrant stopped, the same command skips with the "not reachable" reason instead of
  erroring. The preflight fixture exists and its collection-empty branch is exercised; the
  **Qdrant-down** branch has not been run against a stopped Qdrant yet.
- ✅ `pytest --run-eval tests/integration -m tables -v` selects exactly the three table cases in
  each suite (six node ids). Verified by `--collect-only`, no judge calls spent. The design said
  `-k table`; a marker replaces it, for the substring reason in *State as built*.
- ✅ Every `make` target above exists. `eval-tables` and `eval-retriever` verified by collection;
  a full billed run of each is still outstanding.

Verified as working, at the time of this sync: `make eval` against a single un-sliced agent
golden (Architecture) — `Answer Relevancy 1.000`, `Correctness [GEval] 0.753`,
`Faithfulness 1.000`, all passing, in 16s. Answer relevancy was measured against the old 0.8
threshold there; the score is what matters, and it was a clean 1.00.

Full-suite runs on 2026-08-10, after the golden rewrites and the metric-set change:

| Run | Result | Failing |
|---|---|---|
| before the rewrites | 17/23 | 3, 4, 6, 9, 10, 23 |
| after, run A | 22/23 | 10 |
| after, run B (real `pytest`) | 18/23 | 1, 4, 8, 10, 12 |
| after, run C | 23/23 | — |
| after dropping case 10, run D (real `pytest`) | **21/22** | 6 |

Read that spread before trusting any single run. Cases 1, 4, 8 and 12 each failed exactly once
across four runs and passed the rest; case 1's failing reason literally read "the key fact and
figure are present and accurate" next to a score of 0.461. Only case 10 failed repeatedly, which
is why it, and only it, was dropped from the agent layer.

Measured on golden 5 (Northwood Merit Scholarship) while chasing the `Faithfulness` 0.75:
retrieval is stable — the `How to apply` line lands in the top 3 on four different phrasings of
the query — the answer's four claims are all supported, and the metric returned 1.00 on ten
consecutive runs (five on a fixed answer + context, five driving the agent end to end). The
0.75 was not reproduced; it remains a single flipped verdict out of four, which the quantisation
section explains and which the 0.8 threshold could not absorb.