from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt
from pydantic import BaseModel, Field

from app.ai_assistant_langchain.applicant_documents.schemas import DocumentImage


class AgentRunContext(BaseModel):
    user_id: str
    documents: list[DocumentImage] = Field(
        default_factory=list,
        description=(
            'The files attached to the message being answered, in the order they were attached. '
            'Empty on a message that carries none.'
        ),
    )


class AgentResponse(TypedDict):
    messages: list[BaseMessage]

    __interrupt__: NotRequired[list[Interrupt]]
    """Present only when the graph paused. LangGraph puts it into the returned state itself, which
    is why it is spelled with the dunders: this is the key it actually uses, not one we chose."""
