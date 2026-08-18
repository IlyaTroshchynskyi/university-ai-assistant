lint:
	ruff format
	ruff check --fix
	mypy --follow-imports=skip


run_app:
	uvicorn app.main:app --reload --port=8000


test-checkpointer:
	pytest tests/dynamodb/test_checkpointer.py -v


# Wipe and refill every table from db/seed/ (agent_checkpoints is left alone — it holds
# conversations nothing can rebuild). Also the fix for a drifted `dependants` counter, which is
# what a parent's DELETE refuses on: there is no production data here, so recounting in place would
# buy nothing a re-seed does not.
seed:
	uv run python -m db.load_dynamodb


# Every eval target makes real OpenAI calls and costs money. --log-cli-level is what makes
# assert_metrics print the score table on passing runs too, so a near-miss is visible before it
# turns into a failure. The two scoring suites also need Qdrant, and skip themselves with a reason
# if the collection is empty; eval-routing gates on neither.
eval:
	pytest tests/integration --run-eval --log-cli-level=INFO

# The retriever alone: no agent call at all, so this is the cheap loop to run while iterating on
# chunking or on the search itself.
eval-retriever:
	pytest tests/integration/test_retriever_eval.py --run-eval --log-cli-level=INFO -v

eval-agent:
	pytest tests/integration/test_agent_eval.py --run-eval --log-cli-level=INFO -v

# Which tool each question reaches for. No judges and no Qdrant — the tools are stubbed — so this
# is the one eval that stays cheap enough to run after every prompt or tool-description edit.
eval-routing:
	pytest tests/integration/test_tool_routing_eval.py --run-eval --log-cli-level=INFO -v

# Goldens 19-21, the ones whose answers live inside tables — what catches a chunking change that
# splits a table apart. A marker, not -k: substring matching would also select a section named
# something like "Timetable".
eval-tables:
	pytest tests/integration --run-eval --log-cli-level=INFO -m tables -v

eval-multiturn:
	pytest tests/integration/test_agent_multiturn_eval.py --run-eval --log-cli-level=INFO -v

# The booking subagent end to end: five conversations through /langchain-assistant, each one
# answering its own HITL pause. Needs DynamoDB (it seeds the slots it asserts on) but not Qdrant.
eval-booking:
	pytest tests/integration/test_booking_multiturn_eval.py --run-eval --log-cli-level=INFO -v

eval-safety:
	pytest tests/integration/test_safety_eval.py --run-eval --log-cli-level=INFO -v


typecheck:
	mypy --follow-imports=skip
