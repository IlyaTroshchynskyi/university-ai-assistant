"""Central application settings, loaded and validated from the environment / .env via
pydantic-settings. This is the single source of truth for the OpenAI credentials and model
name — nothing else calls load_dotenv or reads os.environ for these. Build every LLM (and
crew agent) through ``build_llm`` so they all share the same key and model."""

from functools import lru_cache

from crewai import LLM
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    OPENAI_API_KEY: str  # required — read from OPENAI_API_KEY
    MODEL_NAME: str = 'gpt-4o-mini'  # read from MODEL; used everywhere unless overridden


@lru_cache
def get_settings() -> Settings:
    """The settings singleton, built once and cached (raises if OPENAI_API_KEY is missing)."""
    return Settings()


@lru_cache
def build_llm() -> LLM:
    """A crewai LLM wired to the configured model and API key. Pass the key explicitly so no
    call depends on an ambient OPENAI_API_KEY in the environment."""
    settings = get_settings()
    return LLM(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY, temperature=0)
