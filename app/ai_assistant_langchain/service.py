import logging
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends
from langchain_core.messages import HumanMessage

from app.ai_assistant_langchain.agent_schemas import AgentResponse
from app.ai_assistant_langchain.agent_service import AgentService
from app.ai_assistant_langchain.schemas import ChatSchemaOut, Decision, PendingApproval, UserQuery

logger = logging.getLogger(__name__)


class BaseChatService:
    def __init__(
        self,
        agent_service: Annotated[AgentService, Depends(AgentService)],
    ) -> None:
        self._agent_service = agent_service

    async def process_turn(self, query: UserQuery) -> ChatSchemaOut:
        if query.decisions is not None:
            return await self.process_reviewer_decisions(query.user_id, query.decisions)

        return await self.process_user_query(query.user_id, cast(str, query.query))

    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        logger.info('Processing query for user=%s', user_id)

        messages = [HumanMessage(content=message)]
        logger.info('Built context with %d messages for user=%s', len(messages), user_id)

        response = await self._agent_service.run_agent(messages, user_id=str(user_id))
        logger.info('Agent replied for user=%s', user_id)

        return self._to_chat_out(response)

    async def process_reviewer_decisions(self, user_id: UUID, decisions: list[Decision]) -> ChatSchemaOut:
        """Answer the approval this thread is waiting on.

        Alongside ``process_user_query`` rather than inside it: nothing about this turn comes from
        the applicant — no message is added to the conversation, the reviewer's verdict is.
        """
        logger.info('Processing %d decision(s) for user=%s', len(decisions), user_id)

        response = await self._agent_service.resume_agent(decisions, user_id=str(user_id))
        logger.info('Agent resumed for user=%s', user_id)

        return self._to_chat_out(response)

    @staticmethod
    def _to_chat_out(response: AgentResponse) -> ChatSchemaOut:
        """The graph's result as the endpoint returns it — an answer, or an approval to make."""
        interrupts = response.get('__interrupt__')
        if not interrupts:
            assistant_reply: str = response['messages'][-1].content
            return ChatSchemaOut(message=assistant_reply)

        request = interrupts[0].value
        pending = [
            PendingApproval(
                action=action['name'],
                args=action['args'],
                allowed_decisions=config['allowed_decisions'],
            )
            for action, config in zip(request['action_requests'], request['review_configs'], strict=True)
        ]

        logger.info('Graph paused on %s, awaiting decisions', [approval.action for approval in pending])
        return ChatSchemaOut(
            status='pending_approval',
            message='\n\n'.join(
                action.get('description') or f'Please confirm {action["name"]}' for action in request['action_requests']
            ),
            pending=pending,
        )


class ChatService(BaseChatService):
    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        result = await super().process_user_query(user_id, message)
        return result
