# Tests

- `tests/api/` drives the app over HTTP with `StubChatModel` (`tests/agent_stubs.py`): scripted
  replies, scripted routes, real graph, real checkpointer. The stub is installed by patching the
  model factory under **each importing module's own name** (`get_chat_model` in each agent's
  `agent.py`, `get_structured_output_model` in `graph/main_graph.py`) and clearing all three graph
  caches.
- `tests/dynamodb/` calls savers and services directly against the local DynamoDB.
- `tests/unit/` is for pure functions and needs neither DynamoDB nor a model.
- `tests/integration/` is the paid evaluations (`docs/evals.md`), skipped unless `--run-eval`.
- `tests/factory_models/` holds the row models that mirror what the seeder writes and the
  polyfactory factories; `tests/factory_helpers/` holds the creators, getters and deleters. Seed
  rows through these rather than through the agent. `factory_models/applicant_factory.py` builds
  the course's two applicant sets and draws any document as a PNG; `SAMPLE_DAY` is the date they
  are judged on, and a test that reaches the rules through the tool freezes the tool's date to it,
  so the sample passport does not expire under the suite.
- Test data is built in the test, not loaded from the repository: no sample images, JSON or PDFs
  are checked in (`*png` and `*pdf` are gitignored). A test that needs a file draws or writes it.
  `tests/integration/data/goldens.json`, the evaluation goldens, is the one exception.
- Each table has a fixture in `tests/conftest.py` that hands out a service and **empties the table
  afterwards**. Requesting one is how a test declares which tables it may leave rows in.
- The `app` fixture builds `create_app()`. `/ask`, `/documents`, `/slots` and `/health` are
  registered on the module-level `app` in `app/main.py`, not inside `create_app()`, so they do not
  exist in tests. httpx's `ASGITransport` sends no lifespan events either, so `TestBaseAgentClass`
  opens the checkpointer per test.

## Event loops

Ordinary tests run on a per-test loop; the eval suites run on the session loop
(`pytest.mark.asyncio(loop_scope='session')`).

An async object belongs to the loop that created it, and pytest-asyncio runs more than one loop per
process: a fresh one per test, plus the session loop. The app caches its async clients as
process-wide singletons (`get_aioboto_session`, the Qdrant and OpenAI clients, `get_checkpointer`,
the graph factories), which is right for production, where one loop lives as long as the process.
Under pytest the first test to touch one binds it to its loop.

So a function-scoped async fixture used from a session-loop test passes and then errors on
teardown: the test wrote through it on the session loop, and the finalizer, back on the function
loop, trips over a connection that belongs to neither (`1 passed, 1 error`). Use
`session_dynamo_client` there, and the `eval_checkpointer` fixture in
`tests/integration/conftest.py`. `TestBaseDBClass` does not work from `tests/integration/`.
