from uuid import uuid4

from pydantic import BaseModel, Field


class ProgramRow(BaseModel):
    """A programme as the table stores it, cut down to what a faculty's dependants check can see."""

    pk: str = Field(default_factory=lambda: f'PROGRAM#{uuid4()}')
    sk: str = '#META'
    gsi1pk: str
    gsi1sk: str = 'PROGRAM#Test Programme'
    entity_type: str = 'program'
