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
        'General-purpose knowledge base about the university: programs, courses, admissions, '
        'tuition, scholarships, deadlines, policies and other free-form information. Pass a '
        'natural-language query and it returns the most relevant context passages.'
    )
    args_schema: type[BaseModel] = RetrieverToolInput

    async def _run(self, query: str) -> str:
        """Retrieve relevant passages for the query.

        CrewAI detects that this is a coroutine and awaits it, so async I/O (an async vector-DB
        client, HTTP calls) can be used directly here.
        """
        # TODO: connect to the vector DB (async client).
        # 1. Embed `query` with the same embedding model used for indexing.
        # 2. Run an async similarity search against the university knowledge base.
        # 3. Return the top-k matching passages (joined) as context.
        #
        # Example:
        #     results = await vector_store.asimilarity_search(query, k=4)
        #     return "\n\n".join(doc.page_content for doc in results)
        return f'[RetrieverTool placeholder] No vector DB connected yet. Received query: {query!r}'
