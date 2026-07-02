# University Assistant

A university assistant powered by [crewAI](https://crewai.com). An agent answers questions
about the university and decides which tool to use — a knowledge-base retriever, a professor
lookup, or a campus-place lookup — or replies directly for greetings and small talk. Exposed
both as a CLI flow and a FastAPI endpoint.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `app/ai_assistant/crews/university_crew/config/agents.yaml` to define the agent (role, goal, backstory / tool-routing instructions)
- Modify `app/ai_assistant/crews/university_crew/config/tasks.yaml` to define the task
- Modify `app/ai_assistant/crews/university_crew/university_crew.py` to add agents, tasks and attach tools
- Add or edit tools in `app/ai_assistant/tools/`
- Modify `app/ai_assistant/main.py` to change the flow and the `answer_question()` entrypoint
- Modify `app/main.py` to change the FastAPI app (endpoints)

## Running the Project

To kickstart your flow and begin execution, run this from the root folder of your project:

```bash
crewai run
```

This command initializes the University Assistant Flow. It runs the assistant against
a default question and prints the answer.

## Testing the tools (via the LLM)

The assistant has three tools (`app/ai_assistant/tools/`) and the agent decides which
one to call:

- **University Knowledge Retriever** — general questions (programs, admissions, policies…).
- **Find Professor** — look up a named professor/staff member.
- **Find Campus Place** — look up a named campus place (library, cafeteria, gym…).

Requires `OPENAI_API_KEY` in `.env`. Ask a question that should route to each tool and
watch the logs (`verbose=True`) to confirm which tool the agent picked:

```bash
# -> Find Professor
uv run run_with_trigger '{"question": "What is Professor Ivan email?"}'

# -> Find Campus Place
uv run run_with_trigger '{"question": "When does the Main Library open?"}'

# -> University Knowledge Retriever
uv run run_with_trigger '{"question": "What programs does the university offer?"}'

# -> no tool (greeting answered directly)
uv run run_with_trigger '{"question": "hi there"}'
```

You can also run it through the API:

```bash
uv run uvicorn app.main:app --reload
# then POST to http://127.0.0.1:8000/ask  with body {"question": "..."}
# or open http://127.0.0.1:8000/docs
```

## Understanding the project

- `app/main.py` — FastAPI app (`/ask`, `/health`).
- `app/ai_assistant/main.py` — the CrewAI flow and the async `answer_question()` entrypoint.
- `app/ai_assistant/crews/university_crew/` — the crew: the `university_assistant` agent, its
  task, and the `config/agents.yaml` / `config/tasks.yaml` definitions.
- `app/ai_assistant/tools/` — the three tools the agent can call.

The agent's tool-routing behavior lives in the `backstory` in `config/agents.yaml`; each tool's
`description` tells the agent when to use it.

## Support

For support, questions, or feedback regarding crewAI:

- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)