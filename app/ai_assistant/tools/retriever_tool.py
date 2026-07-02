from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class RetrieverToolInput(BaseModel):
    """Input schema for the RetrieverTool."""

    query: str = Field(
        ...,
        description="The user's question or search query about the university.",
    )


class RetrieverTool(BaseTool):
    name: str = 'University Knowledge Retriever'
    description: str = (
        'Retrieves relevant information about the university (programs, admissions, '
        'schedules, staff, policies, etc.). Pass a natural-language query describing '
        'what you need to know, and it returns the most relevant context passages.'
    )
    args_schema: Type[BaseModel] = RetrieverToolInput

    async def _arun(self, query: str) -> str:
        """Async retrieval — the path used when the crew runs via kickoff_async."""
        # TODO: connect to the vector DB (async client).
        # 1. Embed `query` with the same embedding model used for indexing.
        # 2. Run an async similarity search against the university knowledge base.
        # 3. Return the top-k matching passages (joined) as context.
        #
        # Example (to fill in once the async vector DB is wired up):
        #     results = await vector_store.asimilarity_search(query, k=4)
        #     return "\n\n".join(doc.page_content for doc in results)
        return f'[RetrieverTool placeholder] No vector DB connected yet. Received query: {query!r}'

    def _run(self, query: str) -> str:
        """Sync fallback — used if the tool is ever called outside an async run."""
        # TODO: connect to the vector DB (sync client), mirroring `_arun`.
        return f'[RetrieverTool placeholder] No vector DB connected yet. Received query: {query!r}'
