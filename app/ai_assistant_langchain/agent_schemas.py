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


class FindPersonToolInput(BaseModel):
    """Input schema for the FindPersonTool."""

    name: str = Field(
        description='The full name or first name of the professor or staff member to look up.',
    )


class FindPlaceToolInput(BaseModel):
    """Input schema for the FindPlaceTool."""

    name: str = Field(
        description="The name of the campus place to look up, e.g. 'Main Library', 'Cafeteria', 'Gym'.",
    )


class AgentResponse(TypedDict):
    messages: list[BaseMessage]
