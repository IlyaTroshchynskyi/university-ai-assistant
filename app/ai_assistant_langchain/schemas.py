from uuid import UUID

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    query: str = Field(min_length=1, max_length=128)
    user_id: UUID


class ChatSchemaOut(BaseModel):
    message: str
