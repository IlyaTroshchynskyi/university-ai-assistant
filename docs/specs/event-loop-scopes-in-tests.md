# Task: sort out event-loop scopes in the test suite

Opened 2026-08-11. Not started. Worked around, not fixed — the workaround is described below so the
next person does not mistake it for a design.

## The problem in one line

A function-scoped async fixture used by a test that declares `loop_scope='session'` creates its
resources on one event loop and closes them on another, which surfaces as a **passing test followed
by an error**.

## Evidence

Measured twice in this repo, not theorised. Both probes were throwaway files run against the local
DynamoDB, then deleted.

**1. The checkpointer.** A session-loop test extending `TestBaseAgentClass`, whose
`_a_provide_checkpointer` is function-scoped:

```
RuntimeError: Task <... finalizer ...> got Future <Future pending> attached to a different loop
1 passed, 1 error
```

**2. `TestBaseDBClass`.** A session-loop test extending `TestBaseClientDBClass` and writing one row
through `self.dynamo_client`:

```
botocore.exceptions.HTTPClientError: An HTTP Client raised an unhandled exception:
  Task <... finalizer ...> got Future <Future pending> attached to a different loop
1 passed, 1 error
```

The write itself succeeds both times. aiohttp connects lazily, so the TCP connection is created
inside the test body — on the session loop — and only the finalizer, running on the function loop,
trips over it.

## Why it happens at all

Two facts that are individually reasonable and jointly a trap.

**Async objects remember their loop.** A `Future` is registered with the loop that created it; a
client holds sockets, timers and callbacks belonging to one loop. Another loop cannot service them.

**pytest-asyncio runs more than one loop per process.** A fresh loop per test by default, plus one
per session-scoped loop. In production there is exactly one loop for the life of the process, so the
app never meets this.

What turns the trap into a repeated problem here is that the app caches its async clients as
process-wide singletons:

```
get_aioboto_session      app/core/dynamodb/client.py:20
get_qdrant_client        app/ai_assistant/university_knowladge/vector_store.py:29
get_openai_client        app/ai_assistant/university_knowladge/embedder.py:12
create_assistant_agent   app/ai_assistant_langchain/agent.py:17
get_checkpointer         app/ai_assistant_langchain/checkpointer/saver.py
```

Correct for production — one process, one loop, connections reused. In tests the first test to touch
one binds it to its own loop, and every later loop inherits an object it cannot use. That is why the
evaluation suites pin themselves to the session loop in the first place; the reason is recorded in
`tests/integration/test_agent_eval.py:47`.

## What is in place today, and what it costs

`tests/integration/conftest.py` carries a session-scoped `eval_checkpointer` fixture, and the
evaluation suites extend `TestBaseClientClass` rather than `TestBaseAgentClass`. That pairing is
correct — created, used and closed on one loop — but it has consequences worth stating plainly:

* **`TestBaseDBClass` is unusable from `tests/integration/`.** Anyone adding an evaluation test that
  touches DynamoDB hits probe 2 above. There is nothing in the code that says so.
* **`_seed_university_records` in `test_agent_multiturn_eval.py` opens its own client** for two rows,
  purely because the shared `dynamo_client` fixture is the wrong scope.
* **The checkpointer is opened by three different pieces of code** — `app/lifespan.py` for
  production, `TestBaseAgentClass` for `tests/api/`, `eval_checkpointer` for `tests/integration/` —
  and the two test paths exist only because the loops differ.
* **The production lifespan never runs in tests.** Measured: httpx's `ASGITransport` emits no
  lifespan events, so after a request through the test client the checkpointer is still closed. The
  code that wires the app together in production is therefore not the code under test.

## Options

**A. Session-scoped counterparts for the fixtures `tests/integration/` needs.** Add
`session_dynamo_client` and a `TestBaseSessionDBClass` beside the existing ones; leave `tests/api/`
untouched. Smallest change, unblocks DB-touching evaluation tests, fixes nothing underneath — the
two scopes still coexist and the next unguarded combination still breaks the same way.

**B. One loop for the whole suite.** Make every async test run on the session loop, after which
every fixture can be session-scoped and the singletons stay consistent. Removes the class of bug
outright. Two costs: test isolation goes (a leaked task from one test reaches the next), and the
switch is a global pytest setting, which this project has so far deliberately avoided in favour of
per-fixture `loop_scope`. Do not take this one without deciding that trade explicitly.

**C. Run the real lifespan in tests.** Wrap the session-scoped `app` fixture in an ASGI lifespan
manager so `app/lifespan.py` opens the checkpointer, as it does in production. Deletes both test
fixtures and closes the "production wiring is untested" gap. It does not by itself resolve the loop
mismatch — the lifespan runs on whatever loop the `app` fixture uses — so it pairs with A or B
rather than replacing them.

**D. Stop caching async clients as process-wide singletons.** Inject them instead. The root-cause
fix and the largest change; it reaches into application code that is otherwise working, for the
benefit of the test suite alone.

## Recommendation

**A now, C soon, B only as a deliberate decision.** A is a couple of fixtures and unblocks the next
person to write a DB-touching evaluation test. C is worth doing on its own merits — the production
lifespan should be exercised somewhere — and it shrinks the fixture surface that A widens. B is the
only option that removes the class of bug, and it should be chosen for that reason and against the
isolation cost, not slipped in as a side effect of A or C. D is a last resort.

## Acceptance criteria

* A test in `tests/integration/` can read and write DynamoDB through a shared fixture without a
  teardown error — the probes above, kept this time rather than deleted.
* The rule "everything touching a cached async client must live on one loop" is written down where a
  test author will meet it, not only in this file.
* If C is taken: something asserts that the app's lifespan actually ran, so the gap cannot reopen
  silently.

## Out of scope

Nothing here changes what the tests measure. The evaluation suites, their thresholds and their
goldens are untouched by any option above.
