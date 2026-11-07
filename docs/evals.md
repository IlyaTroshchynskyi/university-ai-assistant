# Evaluations

The suites in `tests/integration`. **Every one calls the real OpenAI API and costs money**, so run
one only when the user asked for it. A plain `pytest` never does: the suites are collected and
skipped with a reason unless `--run-eval` is passed, which the `make eval*` targets do. The flag
also turns off the stub `OPENAI_API_KEY` the rest of the tests run on.

```bash
make eval            # everything
make eval-retriever  # retrieval only, no agent call
make eval-agent      # the answers
make eval-routing    # tool choice: no judges, no Qdrant, the cheapest by far
make eval-tables     # the goldens whose answers live inside tables
make eval-multiturn  # four scripted conversations, scored whole
make eval-booking    # five booking conversations, each through the HITL gate
make eval-safety     # four questions written to invite a biased opinion
make eval-documents  # the vision extractor on the two applicant sample sets
```

## What each suite measures

Seven suites, each measuring one layer, so a failing answer points at search or at the model:

- the **retriever** on its own (`KnowledgeService.search`, production defaults) and the **answer**
  `POST /langchain-assistant` gives, both scored with DeepEval against 23 hand-written goldens in
  `tests/integration/data/goldens.json`;
- **routing**: which of `retriever`, `find_person`, `find_place`, `get_schedule`,
  `check_scholarship`, `validate_documents`, `list_common_issues` a question sends the agent to, and
  that a complaint waits for approval before `report_issue` runs;
- **multiturn**, **booking** and **safety**, below;
- **documents**: the two applicant sets drawn as images, read by the real vision model, and the
  verdict compared with the one each set should get.

## Before the first run

- **The handbook is ingested.** Four suites search the live collection (`QDRANT_COLLECTION`,
  `university_kb_v2` by default): `eval-retriever`, `eval-agent`, `eval-multiturn` and
  `eval-safety`. A session fixture (`knowledge_base_populated` in `tests/integration/conftest.py`)
  skips them with a reason when it is empty or unreachable. Ingest with `POST /documents`, leaving
  `doc_type` at `general`: the search filters on it, so a handbook ingested under another type is
  never found.
- **Qdrant is up** (`QDRANT_URL`, default `http://localhost:6333`) for those four suites.
- **The local DynamoDB is up** for every suite that runs the graph: `eval-agent`, `eval-routing`,
  `eval-multiturn`, `eval-booking`, `eval-safety`. They create the `*_test` tables and hold one
  client open for the session. `eval-retriever` runs no graph; `eval-documents` calls the extractor
  and the rules directly and needs neither DynamoDB nor Qdrant.
- `EVAL_MODEL_API_KEY` in `.env` bills the judge to its own key; unset, the judge uses
  `OPENAI_API_KEY`.

The single-turn agent goldens avoid `find_person` and `find_place`: a failing tool does not raise,
the agent's `ToolNode` hands the model an error message and the answer degrades quietly instead of
turning red. `eval-multiturn` seeds the two records it looks up; `eval-routing` stubs every backend.

## Decisions behind the suites

| Decision | Choice | Why |
|---|---|---|
| Layers | Retriever and agent, evaluated separately | A failing answer immediately attributes to search or to the model |
| Golden authoring | Hand-written from `docs/handbook.pdf`, in English | The handbook is English; a verbatim expected answer is checkable, and a synthesizer gets numbers wrong |
| Dataset shape | One JSON file, a `layers` tag per case | A fact lives in one place; small talk and negative cases are left out of the retriever layer, where they mean nothing |
| Data source | The working collection, as it is | The eval must not spend money on embeddings or change the collection |
| Gate | `--run-eval` option plus the `evaluation` marker | Visible in `pytest --help`, the skip carries a reason, and marker selection still composes with `-k` |
| Judge model | `gpt-4.1-mini`, set by `EVAL_JUDGE_MODEL` | Cheap enough to iterate; one variable raises it when a verdict must be trusted. Not `gpt-4o-mini`, which stalls on DeepEval's nested `Verdicts` schema |
| Failure mode | One pytest case per golden, with `assert_metrics` | Shows which question and which metric broke, and prints every score against its threshold on passing runs too |
| Agent invocation | Real HTTP through `POST /langchain-assistant` | Covers router, schemas, service and agent: the path a user takes |

A metric may drive a change to `app/` only when it has found a genuine defect, never to make a
number go green.

## Writing a golden

A question must ask for everything its expected answer states. `Correctness` wants every figure of
the expected answer in the output, and `AnswerRelevancy` counts whatever the question did not ask
for as an irrelevant statement, so a narrow question with a broad expected answer cannot pass both.
Goldens 6, 13 and 14 were rewritten for this ("…, and what does it give?"). Goldens 11 and 16 still
carry a milder version: the expected answer adds a fact the question did not ask for, the higher
cutoffs of competitive programmes and the $500 housing deposit. The judge has scored it as relevant
so far, so they are the first place to look when `AnswerRelevancy` fails on them.

## After a chunking change

Delete the handbook's points before re-ingesting it, then run `make eval` and compare, the table
goldens (`make eval-tables`) first. A point's id is `uuid5` of `{source}:{index}`, so re-ingesting
overwrites chunks in place only up to the new count: a splitter that produces fewer chunks leaves
the old tail in the collection, still searchable, and a comparison over it is not valid. Drop the
collection, or delete the points whose `source` is the handbook. This is done by hand; the suites
only read.

## Reading a result

The judge model and every threshold live in `tests/integration/config.py`, each with the
measurements that set it. A red run is not automatically a regression: every metric is an LLM judge,
and the spread between runs is wider than the gap between a good answer and a mediocre one. Read the
score table that `--log-cli-level=INFO` prints, and the module docstrings of the suites, which record
which goldens sit near a threshold.

What the retriever suite showed once it was run (four runs on 2026-08-10, 21 cases each):

- `ContextualRecall` is 1.000 on every case. It is one verdict per sentence of the expected answer,
  so a one-sentence golden makes it binary: an expected answer whose two facts sit in two chunks
  alternates between 1 and 0. Split such an answer into two sentences.
- `ContextualPrecision` quantises: with `limit=3` the values around the bar are 0.833 and 0.583, so a
  0.7 threshold means "the top-ranked chunk must be relevant". The agent concatenates all three
  chunks, so their order never reaches the model.
- `ContextualRelevancy` lands at 0.095–0.682, median about 0.28, reproducible within 0.03: it
  measures chunk bulk, not search quality.

## Conversations and bias

`eval-multiturn` posts four conversations through `/langchain-assistant`, each on one `user_id`, and
scores them whole on `TurnFaithfulness` and a custom `Outcome` judge. `Outcome` compares the turns
with the facts a correct dialogue must have established. It measures reliance on history because
the second turn of every scenario carries a reference with no antecedent: "What are **his** office
hours?", "Would a scholarship reduce **that whole amount**?". Without history the only correct move
is to ask whose, or which amount, and the outcome is not reached. `Outcome` is given
`evaluation_steps`, never `criteria`: handed criteria, the metric asks the judge to invent its own
steps, which came back grading role consistency rather than the facts.

It is not a regression gate on the checkpointer, which `tests/api/test_chat_checkpointer.py` covers
for free. What only this suite sees is how the *model* uses history, which changes with the prompt
and the model, so run it after those change. Deliberately not measured:

- which tool a turn calls: a follow-up may be answered from history or by calling the tool again,
  and asserting either would fail runs that are not defects. Tool lock-in is caught indirectly,
  because an answer from the wrong tool lacks the facts and fails `Outcome`;
- `ConversationCompletenessMetric`, `TurnRelevancyMetric`, `RoleAdherenceMetric`: they score
  manners, not facts, and three more holistic judges multiply the noise;
- `KnowledgeRetentionMetric`: tried and removed. It takes its knowledge from the user's turns only,
  so every figure a retrieval assistant quotes counts as unsupported, and a conversation whose
  answers were all correct scored 0.000. It fits a form-filling bot, not this one.

`eval-booking` runs five conversations that book, are refused, are rejected by the reviewer, are
moved by the reviewer, and cancel. Each seeds the slots it asserts on. Before either judge runs it
asserts the `appointment_slots` rows themselves and the actions the graph paused on, which is the
proof that nothing reached the table without a human.

`eval-safety` asks four questions written to invite an opinion about a group, one per axis of
`BiasMetric`'s rubric: gender, political, racial or ethnic, geographical. A question off those axes
is not worth asking: the judge is shown only those four, so a bias on any other scores 0 and passes.
The score is a ceiling: 0.000 is clean, and an answer with no opinions scores 0 by construction, so a
green run means the agent volunteered no opinions, not that it is unbiased. It is a drift guard.

`EVAL_CONVERSATION_OUTCOME_THRESHOLD` is not measured yet; `config.py` says which thresholds are.
