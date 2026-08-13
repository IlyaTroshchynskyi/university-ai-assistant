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


# Every action below carries an optional ``table``. ``TransactWriteItems`` is not scoped to one
# table — it spans any table in the region — but ``DynamoDBService`` is, so an action naming its own
# table is how a repository reaches outside its own. Left unset it means 'the service's table',
# which is what all but the counter-maintaining actions want.


@dataclass(frozen=True, slots=True)
class TransactPut:
    """A 'write this item' step of a transaction, with the condition guarding it."""

    item: BaseModel
    condition: ConditionBase | None = None
    table: str | None = None


@dataclass(frozen=True, slots=True)
class TransactDelete:
    """A 'remove the item at this key' step of a transaction, with the condition guarding it."""

    key: dict[str, Any]
    condition: ConditionBase | None = None
    table: str | None = None


@dataclass(frozen=True, slots=True)
class TransactConditionCheck:
    """An 'assert this about an item we are *not* writing' step of a transaction.

    How a foreign key is enforced: a program may only be created while its faculty exists, and
    reading the faculty first would be a race — another request could delete it between the read and
    the write. Checking it inside the same transaction closes that window, since DynamoDB evaluates
    every action against one consistent view and cancels the whole thing if any condition fails.

    ``condition`` is mandatory here, unlike on the other actions: a check that asserts nothing has
    no reason to be in the transaction at all."""

    key: dict[str, Any]
    condition: ConditionBase
    table: str | None = None


@dataclass(frozen=True, slots=True)
class TransactUpdate:
    """An 'apply this update expression' step of a transaction — how a counter on another item is
    maintained atomically with the write that changes it."""

    key: dict[str, Any]
    update_expression: str
    expression_values: dict[str, Any]
    expression_names: dict[str, str] | None = None
    condition: ConditionBase | None = None
    table: str | None = None


TransactAction = TransactPut | TransactDelete | TransactConditionCheck | TransactUpdate
