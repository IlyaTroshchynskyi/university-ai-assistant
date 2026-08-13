from functools import lru_cache

from crewai import LLM
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str = 'gpt-4o-mini'
    EMBEDDING_MODEL: str = 'text-embedding-3-small'
    SPARSE_MODEL: str = 'Qdrant/bm25'
    QDRANT_URL: str = 'http://localhost:6333'
    QDRANT_COLLECTION: str = 'university_kb_v2'

    SEARCH_MIN_SCORE: float = 0.3

    # Relevance-leaning: at 0.5 the diversity term outbids relevance hard enough to spend a slot
    # on an unrelated passage (opening hours for a tuition question) and drop one that carries the
    # answer. Retrieval feeds an answer, so a near-duplicate costs less than a miss.
    SEARCH_MMR_LAMBDA: float = 0.8
    SEARCH_MMR_FETCH_MULT: int = 4

    VISION_MODEL: str = 'gpt-4o-mini'
    TABLE_SUMMARY_MODEL: str = 'gpt-4o-mini'
    STRUCTURED_CHUNK_SIZE: int = 1000
    STRUCTURED_CHUNK_OVERLAP: int = 200
    ENRICHMENT_ENABLED: bool = True

    DYNAMODB_ENDPOINT_URL: str | None = 'http://localhost:8001'
    # One table per entity (db/03-table-split.md). The exception is ``academic_groups``, which holds
    # groups *and* their schedule rows: the one pair in the model a single query has to return
    # together.
    DYNAMODB_FACULTIES_TABLE: str = 'faculties'
    DYNAMODB_PROGRAMS_TABLE: str = 'programs'
    DYNAMODB_PROFESSORS_TABLE: str = 'professors'
    DYNAMODB_COURSES_TABLE: str = 'courses'
    DYNAMODB_ROOMS_TABLE: str = 'rooms'
    DYNAMODB_PLACES_TABLE: str = 'places'
    DYNAMODB_GROUPS_TABLE: str = 'academic_groups'
    DYNAMODB_SLOTS_TABLE: str = 'appointment_slots'
    # The LangGraph checkpointer's table — conversation history, one partition per thread.
    DYNAMODB_CHECKPOINTS_TABLE: str = 'agent_checkpoints'

    AWS_REGION: str = 'us-east-1'
    AWS_ACCESS_KEY_ID: str = 'dummy'
    AWS_SECRET_ACCESS_KEY: str = 'dummy'

    # The judge model and the metric thresholds are not here: they configure the DeepEval suites,
    # which the running app never imports. See ``tests/integration/config.py``.

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


@lru_cache
def get_settings() -> Settings:
    """The settings singleton, built once and cached (raises if OPENAI_API_KEY is missing)."""
    return Settings()


# Todo delete from here
@lru_cache
def build_llm() -> LLM:
    """A crewai LLM wired to the configured model and API key. Pass the key explicitly so no
    call depends on an ambient OPENAI_API_KEY in the environment."""
    settings = get_settings()
    return LLM(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY, temperature=0)
