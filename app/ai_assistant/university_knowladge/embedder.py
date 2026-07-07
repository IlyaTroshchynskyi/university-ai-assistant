"""Asynchronous text embeddings via the OpenAI API. A cached ``AsyncOpenAI`` client singleton
(like the other factories) plus an ``OpenAIEmbedder`` service that takes the client as an
injected dependency, so it can be shared and swapped/mocked in tests."""

from functools import lru_cache

from openai import AsyncOpenAI

from app.settings import get_settings, Settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, settings: Settings):
        self._client = client
        self._model = settings.EMBEDDING_MODEL

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(get_openai_client(), get_settings())
