from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, computed_field, ConfigDict, Field

from app.core.dynamodb.indexes import thread_key


class Role(StrEnum):
    USER = 'user'
    ASSISTANT = 'assistant'


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


@dataclass(frozen=True, slots=True)
class NewMessage:
    role: Role
    content: str
    created_at: str


class MessageRow(BaseModel):
    """One line of a conversation, as the table stores it."""

    # ``populate_by_name`` so the field can still be set by its own name: the alias below is how the
    # row is *read*, not how it is built.
    model_config = ConfigDict(populate_by_name=True)

    user_id: str
    role: Role
    content: str
    # Kept out of the item and read back off the key, because the key is where it already is: ``sk``
    # is this instant, and writing it a second time under its own name would put the same string on
    # every row twice.
    created_at: str = Field(exclude=True, validation_alias='sk')

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pk(self) -> str:
        return thread_key(self.user_id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.created_at
