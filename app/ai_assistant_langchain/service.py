import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends
from langchain_core.messages import HumanMessage

from app.ai_assistant_langchain.agent_service import AgentService
from app.ai_assistant_langchain.schemas import ChatSchemaOut

logger = logging.getLogger(__name__)


class BaseChatService:
    def __init__(
        self,
        agent_service: Annotated[AgentService, Depends(AgentService)],
    ) -> None:
        self._agent_service = agent_service

    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        logger.info('Processing query for user=%s', user_id)

        messages = [HumanMessage(content=message)]
        logger.info('Built context with %d messages for user=%s', len(messages), user_id)

        response = await self._agent_service.run_agent(messages, user_id=str(user_id))
        assistant_reply: str = response['messages'][-1].content
        logger.info('Agent replied for user=%s', user_id)

        return ChatSchemaOut(message=assistant_reply)


class ChatService(BaseChatService):
    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        result = await super().process_user_query(user_id, message)
        return result
