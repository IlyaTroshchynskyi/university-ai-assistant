from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator, Iterator
from unittest.mock import patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

import app.ai_assistant_langchain.agent as agent_module
from app.ai_assistant_langchain.enums import GraphNode
import app.ai_assistant_langchain.main_graph as main_graph_module
from app.ai_assistant_langchain.main_graph import build_main_graph

# Each importing module's own name, not ``agent_model.get_model_factory`` where the factory now
# lives: both modules bind it with ``from … import`` at import time, so patching it at the source
# would leave the copies they already hold — and the graph would build on the real model.
PATH_TO_MODEL_FACTORY = 'app.ai_assistant_langchain.agent.get_model_factory'
PATH_TO_ROUTER_MODEL_FACTORY = 'app.ai_assistant_langchain.main_graph.get_model_factory'
DEFAULT_STUB_REPLY = 'Stubbed assistant reply.'
DEFAULT_STUB_ROUTE = GraphNode.QA

# How much of a scripted reply goes into one chunk. Fixed width rather than per word: the
# concatenation assertions stay exact, whitespace included.
STREAM_PIECE_CHARS = 12


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
        no-ops, so it would come back with nothing to parse.

        The scripted destination is answered *through the model* rather than straight from
        ``routes``: the real router streams its JSON into the token stream at ``__start__``, and a
        lambda that never calls a model hides that leak — the test that the router does not leak
        would pass with the filter deleted.
        """

        async def _decide(messages: LanguageModelInput) -> Any:
            self.route_calls.append(messages)
            index = len(self.route_calls) - 1
            destination = self.routes[index] if index < len(self.routes) else DEFAULT_STUB_ROUTE
            reply = await self.ainvoke(messages, stub_reply=AIMessage(content=f'{{"destination": "{destination}"}}'))
            return schema.model_validate_json(reply.text)

        return RunnableLambda(_decide)

    def reset(self) -> None:
        self.replies.clear()
        self.calls.clear()
        self.routes.clear()
        self.route_calls.clear()

    def _next_reply(self, messages: list[BaseMessage], stub_reply: AIMessage | None = None) -> AIMessage:
        """The reply this call gets, and the bookkeeping that goes with it.

        ``stub_reply`` is the routing call answering itself (see ``with_structured_output``). It is
        deliberately not recorded in ``calls`` and does not consume a scripted reply: both are
        indexed by the *agent's* calls, and a routing call landing in them would shift every index
        the checkpointer assertions are written against.
        """
        if stub_reply is not None:
            return stub_reply

        self.calls.append(list(messages))
        index = len(self.calls) - 1
        return self.replies[index] if index < len(self.replies) else AIMessage(content=DEFAULT_STUB_REPLY)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self._next_reply(messages, kwargs.get('stub_reply'))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """The same scripted reply, in pieces.

        Without this the model cannot stream and LangChain falls back to ``_generate``, so a
        streaming turn carries one whole ``AIMessage`` and the token tests have nothing to assert.
        A synchronous ``_stream`` is enough — ``BaseChatModel._astream`` delegates to it.

        A reply that calls a tool goes out as a single chunk: its arguments are not text, and
        nothing here asserts on partial tool arguments.
        """
        reply = self._next_reply(messages, kwargs.get('stub_reply'))
        if reply.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls))
            return

        text = reply.text
        pieces = [text[start : start + STREAM_PIECE_CHARS] for start in range(0, len(text), STREAM_PIECE_CHARS)]
        for piece in pieces or ['']:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


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
