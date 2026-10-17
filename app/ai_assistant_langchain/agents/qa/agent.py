from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import ToolCall
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.ai_assistant_langchain.agents.qa.prompts import MAIN_CHAT_PROMPT
from app.ai_assistant_langchain.agents.qa.state import AssistantAgentState
from app.ai_assistant_langchain.agents.qa.tools import ASSISTANT_TOOLS, report_issue
from app.ai_assistant_langchain.graph.models import get_model_factory
from app.ai_assistant_langchain.graph.schemas import CustomContext


@lru_cache
def create_assistant_agent() -> CompiledStateGraph:
    model = get_model_factory()
    return create_agent(
        model=model,
        system_prompt=MAIN_CHAT_PROMPT,
        tools=ASSISTANT_TOOLS,
        state_schema=AssistantAgentState,
        context_schema=CustomContext,
        # NO checkpointer — `build_main_graph` owns it. A subgraph compiled with a saver of its own
        # runs on its own thread, and the parent's thread_id and interrupt plumbing break with it.
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=('tokens', 8000),
                keep=('messages', 20),
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    report_issue.name: {
                        'allowed_decisions': ['approve', 'edit', 'reject'],
                        'description': _describe_issue_report,
                    },
                },
            ),
        ],
    )


def _describe_issue_report(tool_call: ToolCall, state: AgentState, runtime: Runtime[CustomContext]) -> str:
    args = tool_call['args']
    return f'File a {args["category"]} issue: {args["text"]}'
