from typing import TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class CustomContext(BaseModel):
    user_id: str


class RetrieverToolInput(BaseModel):
    """Input schema for the RetrieverTool."""

    query: str = Field(
        description="The user's question or search query about the university.",
    )


class AgentResponse(TypedDict):
    messages: list[BaseMessage]
