from pydantic import BaseModel

from app.core.dynamodb.schemas import Slot


class SlotRow(BaseModel):
    pk: str
    sk: str
    gsi1pk: str
    gsi1sk: str
    gsi2pk: str | None = None
    gsi2sk: str | None = None
    id: int
    date: str
    start_time: str
    end_time: str
    status: str
    topic: str | None = None
    booked_by: str | None = None

    @classmethod
    def from_entity(cls, slot: Slot) -> 'SlotRow':
        row = cls(
            pk=slot.date,
            sk=f'{slot.start_time}#{slot.id}',
            gsi1pk=f'STATUS#{slot.status}',
            gsi1sk=f'{slot.date}#{slot.start_time}',
            **slot.model_dump(exclude={'expires_at'}),
        )
        if slot.booked_by:
            row.gsi2pk = f'STUDENT#{slot.booked_by}'
            row.gsi2sk = f'{slot.date}#{slot.start_time}'
        return row
