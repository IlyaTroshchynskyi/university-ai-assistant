import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends

from app.ai_assistant_langchain.history.repository import ConversationHistoryRepository
from app.ai_assistant_langchain.history.schemas import NewMessage, Role, utc_now
from app.ai_assistant_langchain.schemas import ChatSchemaOut, HistoryMessage

logger = logging.getLogger(__name__)


class ConversationHistoryService:
    def __init__(
        self,
        repository: Annotated[ConversationHistoryRepository, Depends(ConversationHistoryRepository)],
    ) -> None:
        self._repository = repository

    async def record_turn(self, user_id: UUID, question: str, asked_at: str, result: ChatSchemaOut) -> None:
        messages = [NewMessage(Role.USER, question, asked_at)]
        if answer := self._answer_to_store(result):
            messages.append(NewMessage(Role.ASSISTANT, answer, utc_now()))

        await self._repository.append_turn(str(user_id), messages)
        logger.info('Stored %d message(s) for user=%s', len(messages), user_id)

    async def record_resumed_turn(self, user_id: UUID, result: ChatSchemaOut) -> None:
        answer = self._answer_to_store(result)
        if answer is None:
            return

        await self._repository.append_turn(str(user_id), [NewMessage(Role.ASSISTANT, answer, utc_now())])
        logger.info('Stored the resumed answer for user=%s', user_id)

    async def list_history(self, user_id: UUID) -> list[HistoryMessage]:
        """The stored conversation, oldest message first. An id nobody has written under is empty."""
        stored = await self._repository.list_message_by_user_id(str(user_id))
        logger.info('Loaded %d stored message(s) for user=%s', len(stored), user_id)
        return [HistoryMessage(role=message.role, content=message.content) for message in stored]

    @staticmethod
    def _answer_to_store(result: ChatSchemaOut) -> str | None:
        """The answer this turn is to store, or ``None`` when it has none to store.

        Two cases have none. A turn that stopped on an approval carries the sentence a reviewer
        reads, not an answer — ``status`` is ``pending_approval``. And an answer that is blank is
        not a message: it would replay as an empty line in the transcript.
        """
        if result.status != 'answer' or not result.message.strip():
            return None
        return result.message
