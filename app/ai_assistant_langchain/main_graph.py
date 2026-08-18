from functools import lru_cache
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel

from app.ai_assistant_langchain.agent import _get_model_factory, create_assistant_agent, create_booking_agent
from app.ai_assistant_langchain.agent_schemas import CustomContext
from app.ai_assistant_langchain.checkpointer.saver import get_checkpointer
from app.ai_assistant_langchain.enums import GraphNode
from app.ai_assistant_langchain.prompts import build_router_prompt

NO_HISTORY = '(no earlier messages — the conversation starts here)'


class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class Route(BaseModel):
    destination: GraphNode


def _recent_turns(messages: Sequence[BaseMessage], limit: int = 4, max_chars: int = 300) -> str:
    lines: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            speaker = 'applicant'
        elif isinstance(message, AIMessage) and not message.tool_calls:
            speaker = 'assistant'
        else:
            continue

        text = ' '.join(message.text.split())
        if not text:
            continue

        lines.append(f'{speaker}: {text[:max_chars]}')
        if len(lines) == limit:
            break

    return '\n'.join(reversed(lines)) if lines else NO_HISTORY


async def route(state: MainState) -> GraphNode:
    """Pick the handler for the newest message, with the turns before it as background.

    The message being routed stays a real ``HumanMessage`` while its background goes into the
    system prompt, so the model is never in doubt about which of the two it is classifying.
    """
    *earlier, current = state['messages']
    decision = (
        await _get_model_factory()
        .with_structured_output(Route)
        .ainvoke([SystemMessage(build_router_prompt(_recent_turns(earlier))), current])
    )
    return decision.destination


@lru_cache
def build_main_graph() -> CompiledStateGraph:
    builder = StateGraph(MainState, context_schema=CustomContext)

    builder.add_node(GraphNode.QA, create_assistant_agent())
    builder.add_node(GraphNode.BOOKING, create_booking_agent())

    builder.add_conditional_edges(
        START,
        route,
        {GraphNode.QA: GraphNode.QA, GraphNode.BOOKING: GraphNode.BOOKING},
    )

    builder.add_edge(GraphNode.QA, END)
    builder.add_edge(GraphNode.BOOKING, END)

    return builder.compile(checkpointer=get_checkpointer())
