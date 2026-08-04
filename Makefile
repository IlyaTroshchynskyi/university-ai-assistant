lint:
	ruff format
	ruff check --fix


run_app:
	uvicorn app.main:app --reload --port=8000
