from uuid import uuid4

from pydantic import BaseModel, Field


class GroupRow(BaseModel):
    pk: str = Field(default_factory=lambda: f'GROUP#{uuid4()}')
    sk: str = '#META'
    gsi1pk: str
    gsi1sk: str = 'GROUP#Test Group'
    entity_type: str = 'group'
