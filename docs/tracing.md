# Tracing with LangSmith

The LangSmith SDK configures itself from the process environment, so export the four names before
starting the app:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_pt_...          # https://smith.langchain.com → Settings → API keys
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_PROJECT=university-assistant
```

`.env` reaches the SDK too: `app/settings.py` imports `crewai`, and `crewai/llm.py` calls
`load_dotenv()` on import, so every name in `.env` is in `os.environ` before the app serves a
request. `LANGSMITH_TRACING=true` in `.env` is enough to trace, and `false` there is what keeps a
run untraced. An exported name still wins over `.env`.

All four are declared without defaults, so the app refuses to start until they are set. To run
untraced, set `LANGSMITH_TRACING=false` and leave the other three empty.

Nothing is instrumented by hand. LangChain and LangGraph emit a run tree, so one request arrives as
the whole thing: the router's classification, the subagent it chose, every tool call with its
arguments and result, and for `compare_programs` one retrieval branch per programme side by side
with `merge` waiting on all of them. That is the easiest way to *see* the fan-out: the branches
overlap on the trace's timeline.

Under pytest tracing is off for every run, the evaluations included: `.env` is in the environment
as soon as `app.settings` is imported, so `tests/conftest.py` sets `LANGSMITH_TRACING=false`. Only
the LangGraph assistant is traced.
