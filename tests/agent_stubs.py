from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator
from unittest.mock import patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

import app.ai_assistant_langchain.agent as agent_module
from app.ai_assistant_langchain.enums import GraphNode
import app.ai_assistant_langchain.main_graph as main_graph_module
from app.ai_assistant_langchain.main_graph import build_main_graph

PATH_TO_MODEL_FACTORY = 'app.ai_assistant_langchain.agent._get_model_factory'
PATH_TO_ROUTER_MODEL_FACTORY = 'app.ai_assistant_langchain.main_graph._get_model_factory'
DEFAULT_STUB_REPLY = 'Stubbed assistant reply.'
DEFAULT_STUB_ROUTE = GraphNode.QA


class StubChatModel(BaseChatModel):
    """Replies from ``replies`` in order (falling back to a default) and records every message list
    it was called with, so a test can assert what the agent actually sent to the model — that is
    what proves the checkpointer replayed the history."""

    replies: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    routes: list[GraphNode] = Field(
        default_factory=list,
        description=(
            'Where the router sends each turn, in order. Empty means every turn goes to `qa`, '
            'which is what keeps the suites written before the router existed on the path they '
            'were written for.'
        ),
    )
    route_calls: list[LanguageModelInput] = Field(
        default_factory=list,
        description=(
            'Deliberately not `calls`: that list is what the *agent* sent, and a routing call '
            'landing in it would shift every index the checkpointer assertions are written against.'
        ),
    )

    @property
    def _llm_type(self) -> str:
        return 'stub-chat-model'

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        # create_agent binds the tool schemas; the scripted replies decide what gets called, so the
        # schemas are irrelevant here. The base class raises NotImplementedError.
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[LanguageModelInput, Any]:
        """The router's call. The base implementation goes through ``bind_tools``, which this stub
        no-ops, so it would come back with nothing to parse."""

        async def _decide(messages: LanguageModelInput) -> Any:
            self.route_calls.append(messages)
            index = len(self.route_calls) - 1
            destination = self.routes[index] if index < len(self.routes) else DEFAULT_STUB_ROUTE
            return schema(destination=destination)

        return RunnableLambda(_decide)

    def reset(self) -> None:
        self.replies.clear()
        self.calls.clear()
        self.routes.clear()
        self.route_calls.clear()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        reply = self.replies[index] if index < len(self.replies) else AIMessage(content=DEFAULT_STUB_REPLY)
        return ChatResult(generations=[ChatGeneration(message=reply)])


@lru_cache
def get_stub_model() -> StubChatModel:
    return StubChatModel()


@contextmanager
def stub_router_model() -> Generator[None, None, None]:
    """Keep the router on the stub for as long as requests are being made."""
    with patch(PATH_TO_ROUTER_MODEL_FACTORY, return_value=get_stub_model()):
        yield


@lru_cache
def get_test_agent() -> CompiledStateGraph:
    """Build the production graph with the stub model. Patching the factory rather than calling
    ``create_agent`` here keeps the checkpointer, middleware and tool wiring under test instead of
    re-declaring them; the lru_cache is cleared on both sides so the real agent is never cached."""
    with patch(PATH_TO_MODEL_FACTORY, return_value=get_stub_model()):
        _clear_graph_caches()
        graph = build_main_graph()
    _clear_graph_caches()
    return graph


def get_test_checkpointer() -> BaseCheckpointSaver:
    return get_test_agent().checkpointer


def _clear_graph_caches() -> None:
    """Every cached factory between the model and the compiled graph. All three, because the main
    graph holds the subagents it was built with — clearing only the outer one would hand the stub
    graph a pair of subagents wired to the real model."""
    agent_module.create_assistant_agent.cache_clear()
    agent_module.create_booking_agent.cache_clear()
    main_graph_module.build_main_graph.cache_clear()
