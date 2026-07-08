from functools import lru_cache

from crewai import LLM
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    OPENAI_API_KEY: str  # required — read from OPENAI_API_KEY
    MODEL_NAME: str = 'gpt-4o-mini'  # read from MODEL; used everywhere unless overridden
    EMBEDDING_MODEL: str = 'text-embedding-3-small'  # OpenAI (dense) embedding model for the vector store
    SPARSE_MODEL: str = 'Qdrant/bm25'  # FastEmbed model for sparse (BM25 keyword) vectors
    QDRANT_URL: str = 'http://localhost:6333'  # the Qdrant service (see docker-compose.yml)
    QDRANT_COLLECTION: str = 'university_kb'  # default collection the vector service reads/writes
    # Minimum cosine similarity for a dense hit to count as relevant. Applied server-side (on the
    # dense query / dense prefetch) so weak matches never come back. See QdrantService.
    SEARCH_MIN_SCORE: float = 0.3


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
