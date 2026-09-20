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


def booking_tool_call_reply(call_id: str = 'call_book_1') -> AIMessage:
    """A `book_appointment` call — what the booking subagent pauses on for a human decision.

    Every argument ``_describe_booking`` reads is present: the middleware renders the sentence a
    reviewer sees from them, and a missing one is a ``KeyError`` inside the interrupt rather than a
    pause.
    """
    return AIMessage(
        content='',
        tool_calls=[
            {
                'id': call_id,
                'name': 'book_appointment',
                'args': {
                    'slot_id': 101,
                    'date': '2026-10-06',
                    'start_time': '09:00',
                    'applicant_email': 'applicant@example.com',
                    'topic': 'scholarship options',
                },
            }
        ],
    )


def compare_tool_call_reply(*programs: str, call_id: str = 'call_compare_1') -> AIMessage:
    """A `compare_programs` call — the `return_direct` tool, which answers the turn itself."""
    return AIMessage(
        content='',
        tool_calls=[{'id': call_id, 'name': 'compare_programs', 'args': {'programs': list(programs)}}],
    )
