import asyncio
import logging
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends
from langchain.agents.middleware.human_in_the_loop import HITLRequest
from langchain_core.messages import HumanMessage, ToolMessage

from app.ai_assistant_langchain.agent_schemas import AgentResponse
from app.ai_assistant_langchain.agent_service import AgentService
from app.ai_assistant_langchain.history.schemas import Role, utc_now
from app.ai_assistant_langchain.history.service import ConversationHistoryService
from app.ai_assistant_langchain.schemas import (
    ChatSchemaOut,
    Decision,
    HistoryMessage,
    PendingApproval,
    UserQuery,
)

logger = logging.getLogger(__name__)

# What a failed ``return_direct`` tool is answered with. Deliberately says nothing about the tool or
# the error: the alternative is what the tool itself produces, which is pydantic's
# "programs: Field required" — an internal argument name and a validator's phrasing, shown to an
# applicant as though it were the answer to their question.
TOOL_FAILURE_MESSAGE = 'Sorry — something went wrong while I was putting that answer together. Could you ask me again?'


class BaseChatService:
    def __init__(
        self,
        agent_service: Annotated[AgentService, Depends(AgentService)],
        history_service: Annotated[ConversationHistoryService, Depends(ConversationHistoryService)],
    ) -> None:
        self._agent_service = agent_service
        self._history_service = history_service

    async def process_turn(self, query: UserQuery) -> ChatSchemaOut:
        if query.decisions is not None:
            return await self.process_reviewer_decisions(query.user_id, query.decisions)

        return await self.process_user_query(query.user_id, cast(str, query.query))

    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        logger.info('Processing query for user=%s', user_id)

        messages = [HumanMessage(content=message)]
        logger.info('Built context with %d messages for user=%s', len(messages), user_id)

        # Taken now, not at the write below
        asked_at = utc_now()
        response = await self._agent_service.run_agent(messages, user_id=str(user_id))
        logger.info('Agent replied for user=%s', user_id)

        result = self._to_chat_out(response)
        if self._get_failed_tool(response) is None:
            await self._history_service.record_turn(user_id, message, asked_at, result)
        return result

    async def process_reviewer_decisions(self, user_id: UUID, decisions: list[Decision]) -> ChatSchemaOut:
        """Answer the approval this thread is waiting on.

        Alongside ``process_user_query`` rather than inside it: nothing about this turn comes from
        the applicant — no message is added to the conversation, the reviewer's verdict is.
        """
        logger.info('Processing %d decision(s) for user=%s', len(decisions), user_id)

        response = await self._agent_service.resume_agent(decisions, user_id=str(user_id))
        logger.info('Agent resumed for user=%s', user_id)

        result = self._to_chat_out(response)
        if self._get_failed_tool(response) is None:
            await self._history_service.record_resumed_turn(user_id, result)
        return result

    async def get_history(self, user_id: UUID) -> list[HistoryMessage]:
        logger.info('Loading history for user=%s', user_id)

        async with asyncio.TaskGroup() as tg:
            stored = tg.create_task(self._history_service.list_history(user_id))
            pause = tg.create_task(self._agent_service.find_paused_request(str(user_id)))

        history, paused = stored.result(), pause.result()
        if paused is not None:
            message, pending = self._describe_pause(paused)
            history.append(HistoryMessage(role=Role.ASSISTANT, content=message, pending=pending))

        logger.info(
            'Replaying %d message(s)%s for user=%s',
            len(history),
            ' plus a pending approval' if paused is not None else '',
            user_id,
        )
        return history

    @staticmethod
    def _describe_pause(request: HITLRequest) -> tuple[str, list[PendingApproval]]:
        """A pause as a caller sees it: the sentence to show, and the actions to decide on.

        Shared by the live turn and by a replay so the two cannot drift — a reviewer who reloaded
        must be answering the same thing, described the same way, as one who did not.
        """
        pending = [
            PendingApproval(
                action=action['name'],
                args=action['args'],
                allowed_decisions=config['allowed_decisions'],
            )
            for action, config in zip(request['action_requests'], request['review_configs'], strict=True)
        ]
        message = '\n\n'.join(
            action.get('description') or f'Please confirm {action["name"]}' for action in request['action_requests']
        )
        return message, pending

    @staticmethod
    def _get_failed_tool(response: AgentResponse) -> ToolMessage | None:
        last = response['messages'][-1]
        return last if isinstance(last, ToolMessage) and last.status == 'error' else None

    def _to_chat_out(self, response: AgentResponse) -> ChatSchemaOut:
        """The graph's result as the endpoint returns it — an answer, or an approval to make."""
        interrupts = response.get('__interrupt__')
        if not interrupts:
            if (failure := self._get_failed_tool(response)) is not None:
                logger.warning('Tool %s ended the turn on an error: %s', failure.name, failure.text)
                return ChatSchemaOut(message=TOOL_FAILURE_MESSAGE)

            assistant_reply: str = response['messages'][-1].content
            return ChatSchemaOut(message=assistant_reply)

        message, pending = self._describe_pause(cast(HITLRequest, interrupts[0].value))

        logger.info('Graph paused on %s, awaiting decisions', [approval.action for approval in pending])
        return ChatSchemaOut(status='pending_approval', message=message, pending=pending)


class ChatService(BaseChatService):
    async def process_user_query(self, user_id: UUID, message: str) -> ChatSchemaOut:
        result = await super().process_user_query(user_id, message)
        return result
