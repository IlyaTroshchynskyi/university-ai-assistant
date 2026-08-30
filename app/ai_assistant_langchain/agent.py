from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    HumanInTheLoopMiddleware,
    ModelRequest,
    SummarizationMiddleware,
)
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import ToolCall
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.ai_assistant_langchain.agent_model import get_model_factory
from app.ai_assistant_langchain.agent_schemas import CustomContext
from app.ai_assistant_langchain.booking_tools import (
    book_appointment,
    cancel_appointment,
    find_earliest_open_slots,
    list_free_slots,
    list_my_bookings,
)
from app.ai_assistant_langchain.prompts import (
    build_booking_system_prompt,
    MAIN_CHAT_PROMPT,
)
from app.ai_assistant_langchain.tools import ASSISTANT_TOOLS


@lru_cache
def create_assistant_agent() -> CompiledStateGraph:
    model = get_model_factory()
    return create_agent(
        model=model,
        system_prompt=MAIN_CHAT_PROMPT,
        tools=ASSISTANT_TOOLS,
        context_schema=CustomContext,
        # NO checkpointer — `build_main_graph` owns it. A subgraph compiled with a saver of its own
        # runs on its own thread, and the parent's thread_id and interrupt plumbing break with it.
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=('tokens', 8000),
                keep=('messages', 20),
            ),
        ],
    )


@dynamic_prompt
def _booking_prompt(request: ModelRequest) -> str:
    """The booking prompt, rebuilt for every model call.

    Not passed as ``system_prompt=``: this factory is ``lru_cache``d, so the prompt string would be
    built once for the life of the process and "Today is …" would freeze on whatever day the first
    request happened to arrive. A process that stays up over midnight would then resolve every
    "tomorrow" and "this Friday" a day short — the very thing ``build_booking_system_prompt``'s
    docstring promises not to do. Middleware runs per call, so the graph stays cached and only the
    string is recomputed.
    """
    return build_booking_system_prompt()


def _describe_booking(tool_call: ToolCall, state: AgentState, runtime: Runtime[CustomContext]) -> str:
    """What the reviewer is being asked to approve, as a sentence rather than a log line.

    Without a ``description`` the middleware falls back to ``Tool: book_appointment`` followed by a
    Python dict repr of the arguments. Every argument is still in ``pending[].args`` verbatim, which
    is what a client renders and what an edit replaces; this is the part a person reads.
    """
    args = tool_call['args']
    return (
        f'Book a consultation on {args["date"]} at {args["start_time"]} for '
        f'{args["applicant_email"]}, about {args["topic"]}.'
    )


def _describe_cancellation(tool_call: ToolCall, state: AgentState, runtime: Runtime[CustomContext]) -> str:
    """The cancellation half of ``_describe_booking``. No topic — cancelling does not take one."""
    args = tool_call['args']
    return f'Cancel the consultation on {args["date"]} at {args["start_time"]} booked under {args["applicant_email"]}.'


@lru_cache
def create_booking_agent() -> CompiledStateGraph:
    return create_agent(
        model=get_model_factory(),
        tools=[list_free_slots, find_earliest_open_slots, list_my_bookings, book_appointment, cancel_appointment],
        context_schema=CustomContext,
        middleware=[
            _booking_prompt,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    book_appointment.name: {
                        'allowed_decisions': ['approve', 'edit', 'reject'],
                        'description': _describe_booking,
                    },
                    cancel_appointment.name: {
                        'allowed_decisions': ['approve', 'reject'],
                        'description': _describe_cancellation,
                    },
                },
            ),
        ],
        # NO checkpointer — the main graph owns it
    )
