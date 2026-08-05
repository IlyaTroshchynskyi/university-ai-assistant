from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from boto3.dynamodb.conditions import ConditionBase
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


@dataclass(frozen=True, slots=True)
class TransactPut:
    """A 'write this item' step of a transaction, with the condition guarding it."""

    item: BaseModel
    condition: ConditionBase | None = None


@dataclass(frozen=True, slots=True)
class TransactDelete:
    """A 'remove the item at this key' step of a transaction, with the condition guarding it."""

    key: dict[str, Any]
    condition: ConditionBase | None = None


TransactAction = TransactPut | TransactDelete
