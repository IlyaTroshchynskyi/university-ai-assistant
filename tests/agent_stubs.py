"""The agent used by the API tests: the real graph produced by ``create_assistant_agent`` — same
tools, middleware and an ``InMemorySaver`` — with ``ChatOpenAI`` swapped for a scripted model, so
no test reaches OpenAI. Wired globally in ``override_app_test_dependencies``.
"""

from functools import lru_cache
from typing import Any
from unittest.mock import patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

import app.ai_assistant_langchain.agent as agent_module
from app.ai_assistant_langchain.agent import create_assistant_agent

PATH_TO_MODEL_FACTORY = 'app.ai_assistant_langchain.agent.get_model_factory'
DEFAULT_STUB_REPLY = 'Stubbed assistant reply.'


class StubChatModel(BaseChatModel):
    """Replies from ``replies`` in order (falling back to a default) and records every message list
    it was called with, so a test can assert what the agent actually sent to the model — that is
    what proves the checkpointer replayed the history."""

    replies: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'stub-chat-model'

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        # create_agent binds the tool schemas; the scripted replies decide what gets called, so the
        # schemas are irrelevant here. The base class raises NotImplementedError.
        return self

    def reset(self) -> None:
        self.replies.clear()
        self.calls.clear()

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


@lru_cache
def get_test_agent() -> CompiledStateGraph:
    """Build the production graph with the stub model. Patching the factory rather than calling
    ``create_agent`` here keeps the checkpointer, middleware and tool wiring under test instead of
    re-declaring them; the lru_cache is cleared on both sides so the real agent is never cached."""
    with patch(PATH_TO_MODEL_FACTORY, return_value=get_stub_model()):
        agent_module.create_assistant_agent.cache_clear()
        agent = create_assistant_agent()
    agent_module.create_assistant_agent.cache_clear()
    return agent


def get_test_checkpointer() -> BaseCheckpointSaver:
    return get_test_agent().checkpointer
