import asyncio
import logging
from typing import Annotated, AsyncIterator, cast
from uuid import UUID

from fastapi import Depends
from langchain.agents.middleware.human_in_the_loop import HITLRequest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import MessagesStreamPart, StreamPart, ValuesStreamPart
from pydantic import BaseModel
from sse_starlette import ServerSentEvent

from app.ai_assistant_langchain.agent_schemas import AgentResponse
from app.ai_assistant_langchain.agent_service import AgentService
from app.ai_assistant_langchain.graphs.compare_programs.graph import MERGE
from app.ai_assistant_langchain.history.schemas import Role, utc_now
from app.ai_assistant_langchain.history.service import ConversationHistoryService
from app.ai_assistant_langchain.runs import assert_free, claimed, register
from app.ai_assistant_langchain.schemas import (
    ChatSchemaOut,
    Decision,
    HistoryMessage,
    PendingApproval,
    UserQuery,
)
from app.ai_assistant_langchain.stream_schemas import ErrorEvent, TokenEvent
from app.ai_assistant_langchain.tools import DIRECT_ANSWER_TOOLS
from app.core.exceptions import IncompleteStreamError

logger = logging.getLogger(__name__)

# What a failed ``return_direct`` tool is answered with. Deliberately says nothing about the tool or
# the error: the alternative is what the tool itself produces, which is pydantic's
# "programs: Field required" — an internal argument name and a validator's phrasing, shown to an
# applicant as though it were the answer to their question.
TOOL_FAILURE_MESSAGE = 'Sorry — something went wrong while I was putting that answer together. Could you ask me again?'


SSE_MEDIA_TYPE = 'text/event-stream'
MODEL_NODE = 'model'
SUBAGENT_DEPTH = 1

# A graph a tool runs is one namespace below the subagent, and its tokens are not the answer — with
# one exception, ``compare_programs``, whose graph writes the answer itself (§4.4). The node is
# imported rather than spelled again here, so renaming it there cannot leave this pointing at
# nothing.
TOOL_GRAPH_DEPTH = 2
DIRECT_ANSWER_NODE = MERGE
STREAM_FAILURE_MESSAGE = 'Sorry — something went wrong while I was answering. Could you ask me again?'


class BaseChatService:
    def __init__(
        self,
        agent_service: Annotated[AgentService, Depends(AgentService)],
        history_service: Annotated[ConversationHistoryService, Depends(ConversationHistoryService)],
    ) -> None:
        self._agent_service = agent_service
        self._history_service = history_service

    async def process_turn(self, query: UserQuery) -> ChatSchemaOut:
        async with claimed(str(query.user_id)):
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

        return await self._settle_query(user_id, message, asked_at, response)

    async def process_reviewer_decisions(self, user_id: UUID, decisions: list[Decision]) -> ChatSchemaOut:
        """Answer the approval this thread is waiting on.

        Alongside ``process_user_query`` rather than inside it: nothing about this turn comes from
        the applicant — no message is added to the conversation, the reviewer's verdict is.
        """
        logger.info('Processing %d decision(s) for user=%s', len(decisions), user_id)

        response = await self._agent_service.resume_agent(decisions, user_id=str(user_id))
        logger.info('Agent resumed for user=%s', user_id)

        return await self._settle_decisions(user_id, response)

    async def start_streaming_turn(self, query: UserQuery) -> asyncio.Queue[ServerSentEvent | None]:
        user_id = str(query.user_id)
        if query.decisions is None:
            await self._agent_service.refuse_while_paused(user_id)
        else:
            await self._agent_service.build_hitl_decisions(query.decisions, user_id)

        queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()
        assert_free(user_id)
        register(user_id, asyncio.create_task(self._run_streaming_turn(queue, query)))
        return queue

    async def stream_events(self, queue: asyncio.Queue[ServerSentEvent | None]) -> AsyncIterator[ServerSentEvent]:
        while (event := await queue.get()) is not None:
            yield event

    async def _settle_query(self, user_id: UUID, message: str, asked_at: str, response: AgentResponse) -> ChatSchemaOut:
        """What the turn was, and its transcript row when there is one to write."""
        result = self._to_chat_out(response)
        if self._get_failed_tool(response) is None:
            await self._history_service.record_turn(user_id, message, asked_at, result)
        return result

    async def _settle_decisions(self, user_id: UUID, response: AgentResponse) -> ChatSchemaOut:
        result = self._to_chat_out(response)
        if self._get_failed_tool(response) is None:
            await self._history_service.record_resumed_turn(user_id, result)
        return result

    async def _run_streaming_turn(self, queue: asyncio.Queue[ServerSentEvent | None], query: UserQuery) -> None:
        """The detached half of a streaming turn."""
        try:
            if query.decisions is not None:
                await self._stream_reviewer_decisions(queue, query.user_id, query.decisions)
            else:
                await self._stream_user_query(queue, query.user_id, cast(str, query.query))
        except Exception:
            logger.exception('Streaming turn failed for user=%s', query.user_id)
            queue.put_nowait(self._event(ErrorEvent(detail=STREAM_FAILURE_MESSAGE), 'error'))
        finally:
            queue.put_nowait(None)

    async def _stream_user_query(
        self, queue: asyncio.Queue[ServerSentEvent | None], user_id: UUID, message: str
    ) -> None:
        logger.info('Streaming query for user=%s', user_id)

        # Taken now, not at the write below — the same reason `process_user_query` documents.
        asked_at = utc_now()
        response = await self._forward_tokens(
            queue,
            self._agent_service.stream_agent([HumanMessage(content=message)], user_id=str(user_id)),
        )

        self._finish(queue, await self._settle_query(user_id, message, asked_at, response))

    async def _stream_reviewer_decisions(
        self, queue: asyncio.Queue[ServerSentEvent | None], user_id: UUID, decisions: list[Decision]
    ) -> None:
        logger.info('Streaming %d decision(s) for user=%s', len(decisions), user_id)

        response = await self._forward_tokens(queue, self._agent_service.stream_resume(decisions, user_id=str(user_id)))

        self._finish(queue, await self._settle_decisions(user_id, response))

    async def _forward_tokens(
        self, queue: asyncio.Queue[ServerSentEvent | None], parts: AsyncIterator[StreamPart]
    ) -> AgentResponse:
        last_values: ValuesStreamPart | None = None

        streamed_direct = False

        async for part in parts:
            if part['type'] == 'values' and part['ns'] == ():
                last_values = part
                continue
            if part['type'] != 'messages':
                continue

            if self._is_direct_answer_token(part):
                streamed_direct = True
            elif self._is_direct_answer(part) and streamed_direct:
                # The same text the tool's graph has just written, now arriving whole. Forwarded
                # again it would print the comparison twice. The flag resets, so a second
                # ``return_direct`` call in the same turn is judged on its own.
                streamed_direct = False
                continue
            elif not (self._is_answer_token(part) or self._is_direct_answer(part)):
                continue

            queue.put_nowait(self._event(TokenEvent(text=part['data'][0].text), 'token'))

        if last_values is None:
            raise IncompleteStreamError('The graph ended without a root `values` part.')

        interrupts = list(last_values['interrupts'])
        return cast(AgentResponse, {**last_values['data'], **({'__interrupt__': interrupts} if interrupts else {})})

    @staticmethod
    def _is_answer_token(part: MessagesStreamPart) -> bool:
        chunk, metadata = part['data']
        return (
            len(part['ns']) == SUBAGENT_DEPTH
            and metadata.get('langgraph_node') == MODEL_NODE
            and isinstance(chunk, AIMessageChunk)
            and bool(chunk.text)
        )

    @staticmethod
    def _is_direct_answer_token(part: MessagesStreamPart) -> bool:
        chunk, metadata = part['data']
        return (
            len(part['ns']) == TOOL_GRAPH_DEPTH
            and metadata.get('langgraph_node') == DIRECT_ANSWER_NODE
            and isinstance(chunk, AIMessageChunk)
            and bool(chunk.text)
        )

    @staticmethod
    def _is_direct_answer(part: MessagesStreamPart) -> bool:
        chunk, _ = part['data']
        return (
            len(part['ns']) == SUBAGENT_DEPTH
            and isinstance(chunk, ToolMessage)
            and chunk.name in DIRECT_ANSWER_TOOLS
            and chunk.status != 'error'
        )

    @classmethod
    def _finish(cls, queue: asyncio.Queue[ServerSentEvent | None], result: ChatSchemaOut) -> None:
        queue.put_nowait(cls._event(result, 'pending' if result.status == 'pending_approval' else 'done'))

    @staticmethod
    def _event(payload: BaseModel, name: str) -> ServerSentEvent:
        return ServerSentEvent(data=payload.model_dump_json(), event=name)

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
    """What the route depends on. Nothing of its own, deliberately.

    It used to override ``process_user_query`` with a call to ``super()`` and a return — a no-op,
    but a loaded one: the streaming twin of that method is ``_stream_user_query`` on the base class
    and does not go through the override, so the first real line added here would have taken effect
    for JSON only. Anything this class grows has to cover both representations, or it belongs on
    ``BaseChatService`` where both already meet.
    """
