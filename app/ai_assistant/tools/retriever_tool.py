import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Temporary MOCK knowledge base (stand-in for the vector DB). Remove once a real
# retriever is connected. Keyed by program name, matched loosely against the query so the
# compare / QA flows can be exercised with realistic data. Edit freely. ---
_MOCK_KB: dict[str, str] = {
    'computer science': (
        'Computer Science (BSc) — 3 years, full-time.\n'
        'Focus: algorithms, systems programming, software engineering and the theory of computation.\n'
        'Core courses: Programming, Data Structures & Algorithms, Operating Systems, Databases, '
        'Computer Networks, Software Engineering.\n'
        'Admission: strong maths background; entrance exam in maths and logic.\n'
        'Tuition: 3,500 EUR / year.\n'
        'Career outcomes: software engineer, backend/systems developer, DevOps engineer, research.'
    ),
    'data science': (
        'Data Science (BSc) — 3 years, full-time.\n'
        'Focus: statistics, machine learning and data engineering applied to real-world data.\n'
        'Core courses: Probability & Statistics, Machine Learning, Data Visualization, Big Data '
        'Tools, Databases, Deep Learning.\n'
        'Admission: maths and statistics background; entrance exam in maths.\n'
        'Tuition: 3,900 EUR / year.\n'
        'Career outcomes: data scientist, ML engineer, data analyst, BI specialist.'
    ),
}


def _mock_lookup(query: str) -> str | None:
    """Return mock passages for any known program named in the query, else None."""
    q = query.lower()
    hits = [text for name, text in _MOCK_KB.items() if name in q]
    return '\n\n'.join(hits) if hits else None


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
        'natural-language query and it returns the most relevant context passages. For a '
        'program, a single query with the program name returns all of its information at '
        'once — you do not need to query each topic (tuition, courses, …) separately.'
    )
    args_schema: type[BaseModel] = RetrieverToolInput

    async def _run(self, query: str) -> str:
        """Retrieve relevant passages for the query.

        CrewAI detects that this is a coroutine and awaits it, so async I/O (an async vector-DB
        client, HTTP calls) can be used directly here.
        """
        logger.info('Retriever query: %r', query)
        # MOCK: serve from the in-memory KB above so the flows return realistic data.
        # TODO: replace with the real vector DB (async client):
        # 1. Embed `query` with the same embedding model used for indexing.
        # 2. Run an async similarity search against the university knowledge base.
        # 3. Return the top-k matching passages (joined) as context.
        #
        # Example:
        #     results = await vector_store.asimilarity_search(query, k=4)
        #     return "\n\n".join(doc.page_content for doc in results)
        mock = _mock_lookup(query)
        if mock is not None:
            return mock
        return f'[RetrieverTool placeholder] No vector DB connected yet. Received query: {query!r}'
