from enum import StrEnum

from pydantic import BaseModel


class SlotStatus(StrEnum):
    OPEN = 'open'
    BOOKED = 'booked'


class Professor(BaseModel):
    id: int
    full_name: str
    title: str
    faculty_id: int
    email: str
    room_id: int
    office_hours: str


class Place(BaseModel):
    id: int
    name: str
    building: str
    floor: str  # free-form: '1', '1-3', 'lobby'
    opening_hours: str


class Slot(BaseModel):
    id: int
    date: str  # 'YYYY-MM-DD'
    start_time: str
    end_time: str
    status: SlotStatus
    topic: str | None = None
    booked_by: str | None = None  # student email; set once booked
    expires_at: int | None = None  # epoch seconds — TTL attribute
