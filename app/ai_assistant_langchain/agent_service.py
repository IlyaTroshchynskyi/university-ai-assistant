import logging
from typing import Annotated, cast

from fastapi import Depends
from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    ApproveDecision,
    EditDecision,
    HITLRequest,
    RejectDecision,
)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.ai_assistant_langchain.agent_schemas import AgentResponse, CustomContext
from app.ai_assistant_langchain.main_graph import build_main_graph
from app.ai_assistant_langchain.schemas import Decision
from app.core.exceptions import ConflictingStatusError

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(
        self,
        graph: Annotated[CompiledStateGraph, Depends(build_main_graph)],
    ):
        self._graph = graph

    async def run_agent(
        self,
        messages: list[BaseMessage],
        user_id: str,
    ) -> AgentResponse:
        await self._refuse_while_paused(user_id)

        logger.info('Invoking agent for user_id=%s with %d messages', user_id, len(messages))
        response = cast(
            AgentResponse,
            await self._graph.ainvoke(
                {'messages': messages},
                config=self._config(user_id),
                context=CustomContext(user_id=user_id),
            ),
        )
        logger.info('Agent response received for user_id=%s', user_id)
        return response

    async def resume_agent(self, decisions: list[Decision], user_id: str) -> AgentResponse:
        """Answer every action this thread is paused on and let it run to the end.
        A list, because one model turn can request several gated actions at once — a reschedule
        books the new time and releases the old one — and the middleware answers all of them or
        raises. Sending fewer used to wedge the thread on the interrupt with no way out: the resume
        failed, the state did not move, and every later attempt failed the same way.
        """
        request = await self._get_paused_request(user_id)
        actions, configs = request['action_requests'], request['review_configs']

        if len(decisions) != len(actions):
            raise ConflictingStatusError(
                f'This thread is paused on {len(actions)} action(s) '
                f'({", ".join(action["name"] for action in actions)}), but {len(decisions)} '
                f'decision(s) were sent — send one per action, in the order they were listed.'
            )

        hitl_decisions: list[ApproveDecision | EditDecision | RejectDecision] = []
        for decision, action, config in zip(decisions, actions, configs, strict=True):
            if decision.type not in config['allowed_decisions']:
                raise ConflictingStatusError(
                    f"'{decision.type}' is not a decision {config['action_name']} accepts — "
                    f'this pause takes {" or ".join(config["allowed_decisions"])}.'
                )
            hitl_decisions.append(self._get_hitl_decision(decision, action))

        logger.info('Resuming user_id=%s with %s', user_id, [decision.type for decision in decisions])
        response = cast(
            AgentResponse,
            await self._graph.ainvoke(
                Command(resume={'decisions': hitl_decisions}),
                config=self._config(user_id),
                context=CustomContext(user_id=user_id),
            ),
        )
        logger.info('Resumed response received for user_id=%s', user_id)
        return response

    @staticmethod
    def _get_hitl_decision(
        decision: Decision, action: ActionRequest
    ) -> ApproveDecision | EditDecision | RejectDecision:
        if decision.type == 'approve':
            return {'type': 'approve'}

        if decision.type == 'reject':
            return {'type': 'reject', 'message': decision.message} if decision.message else {'type': 'reject'}

        return {'type': 'edit', 'edited_action': {'name': action['name'], 'args': decision.args or {}}}

    async def _refuse_while_paused(self, user_id: str) -> None:
        """Refuse a new message while this thread is waiting on an approval.

        ``ainvoke`` with a fresh input does not resume an interrupted thread — it starts a new run
        from START, and LangGraph drops the interrupted task when it does. The booking waiting for
        review would vanish silently: nothing written, no word to the applicant who was last told to
        expect it, and no card left for the reviewer, whose ``decisions`` would then fail with "this
        thread is not paused on anything". Refusing the turn keeps all three in step. The cost is
        that the applicant cannot say anything else until the review lands, which is the trade this
        makes deliberately: a stalled conversation is recoverable, a booking nobody knows was
        dropped is not.
        """
        request = await self._find_paused_request(user_id)
        if request is None:
            return

        actions = ', '.join(action['name'] for action in request['action_requests'])
        raise ConflictingStatusError(
            f'This thread is paused on {actions} and is waiting for a decision, so it cannot take a '
            f'new message — send `decisions` for the pending action(s) first.'
        )

    async def _find_paused_request(self, user_id: str) -> HITLRequest | None:
        """The approval this thread is stopped on, or ``None`` when it is not stopped at all."""
        state = await self._graph.aget_state(self._config(user_id))
        for task in state.tasks:
            for interrupt in task.interrupts:
                return cast(HITLRequest, interrupt.value)

        return None

    async def _get_paused_request(self, user_id: str) -> HITLRequest:
        request = await self._find_paused_request(user_id)
        if request is None:
            raise ConflictingStatusError('This thread is not paused on anything — there is nothing to decide.')

        return request

    @staticmethod
    def _config(user_id: str) -> RunnableConfig:
        return {'configurable': {'thread_id': user_id}}
