import logging
from typing import Annotated, cast

from fastapi import Depends
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from app.ai_assistant_langchain.agent import create_assistant_agent
from app.ai_assistant_langchain.agent_schemas import AgentResponse, CustomContext

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(
        self,
        agent: Annotated[CompiledStateGraph, Depends(create_assistant_agent)],
    ):
        self._agent = agent

    async def run_agent(
        self,
        messages: list[BaseMessage],
        user_id: str,
    ) -> AgentResponse:
        logger.info('Invoking agent for user_id=%s with %d messages', user_id, len(messages))
        response = cast(
            AgentResponse,
            await self._agent.ainvoke(
                {'messages': messages},
                config={'configurable': {'thread_id': user_id}},
                context=CustomContext(user_id=user_id),
            ),
        )
        logger.info('Agent response received for user_id=%s', user_id)
        return response
