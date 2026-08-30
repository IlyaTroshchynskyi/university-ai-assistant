from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel

from app.settings import get_settings


@lru_cache
def get_model_factory() -> BaseChatModel | _ConfigurableModel:
    settings = get_settings()
    return init_chat_model(
        model=settings.MODEL_NAME,
        temperature=0,
        timeout=30,
        max_tokens=1000,
        max_retries=2,
        api_key=settings.OPENAI_API_KEY,
    )
