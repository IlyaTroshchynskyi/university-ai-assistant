lint:
	ruff format
	ruff check --fix


run_app:
	uvicorn app.main:app --reload --port=8000


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
