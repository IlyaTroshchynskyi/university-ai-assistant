from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from app.ai_assistant_langchain.agent_schemas import CustomContext
from app.ai_assistant_langchain.checkpointer.saver import get_checkpointer
from app.ai_assistant_langchain.prompts import MAIN_CHAT_PROMPT
from app.ai_assistant_langchain.tools import find_person, find_place, retriever
from app.settings import get_settings


@lru_cache
def create_assistant_agent() -> CompiledStateGraph:
    model = _get_model_factory()
    return create_agent(
        model=model,
        system_prompt=MAIN_CHAT_PROMPT,
        tools=[retriever, find_person, find_place],
        context_schema=CustomContext,
        checkpointer=get_checkpointer(),
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=('tokens', 8000),
                keep=('messages', 20),
            ),
        ],
    )


@lru_cache
def _get_model_factory() -> BaseChatModel | _ConfigurableModel:
    settings = get_settings()
    return init_chat_model(
        model=settings.MODEL_NAME,
        temperature=0,
        timeout=30,
        max_tokens=1000,
        max_retries=2,
        api_key=settings.OPENAI_API_KEY,
    )
