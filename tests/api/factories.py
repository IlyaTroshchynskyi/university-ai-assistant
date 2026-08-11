"""Builders and stubs for the API tests.

Fixtures deliberately stay in the test modules: pytest resolves them by name off the module
namespace, so importing one here and there reads as an unused import (ruff F401, then F811 against
the test's own parameter).
"""

from langchain_core.messages import AIMessage
from qdrant_client.models import ScoredPoint

PATH_TO_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.tools.get_knowledge_service'


class StubKnowledgeService:
    """Replaces the Qdrant-backed service so the retriever tool never leaves the process."""

    def __init__(self, passages: str) -> None:
        self.passages = passages
        self.queries: list[str] = []

    async def search(self, query: str) -> list[ScoredPoint]:
        """Returns hits, not text: the tool reads ``payload['text']`` off each one, so a stub that
        answers with a bare string is iterated character by character."""
        self.queries.append(query)
        return [ScoredPoint(id=1, version=0, score=1.0, payload={'text': self.passages})]


def tool_call_reply(query: str, call_id: str = 'call_1') -> AIMessage:
    return AIMessage(content='', tool_calls=[{'id': call_id, 'name': 'retriever', 'args': {'query': query}}])
