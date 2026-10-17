from datetime import date, time
from typing import ClassVar

from pydantic import BaseModel, computed_field, EmailStr, Field

from app.api.v1.rooms.enums import (
    IssueCategory,
    IssueStatus,
    ProfessorTitle,
    RoomType,
    SlotStatus,
    SlotTopic,
    Weekday,
)
from app.core.dynamodb.base_items import TableItem
from app.core.dynamodb.indexes import build_issue_status_key, build_issue_votes_key

# Todo refactor to diff modules


class GroupCreate(BaseModel):
    name: str = Field(min_length=2, max_length=30)  # e.g. 'CS-1'
    program_id: str


class Group(GroupCreate):
    id: str


class GroupItem(TableItem):
    entity: ClassVar[str] = 'GROUP'


class ProfessorCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    title: ProfessorTitle
    faculty_id: str
    email: EmailStr
    room_id: str  # their office, a room of type OFFICE
    office_hours: str = Field(min_length=3, max_length=120)  # free-form, e.g. 'Mon 14:00–16:00'


class Professor(ProfessorCreate):
    id: str


class ProfessorItem(TableItem):
    entity: ClassVar[str] = 'PROF'


class CourseCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    faculty: str = Field(min_length=2, max_length=100)  # faculty *name*, not id
    professor_id: str


class Course(CourseCreate):
    id: str


class CourseItem(TableItem):
    entity: ClassVar[str] = 'COURSE'


class CreateRoom(BaseModel):
    building: str = Field(min_length=2, max_length=80)
    floor: int = Field(ge=-2, le=50)
    number: int = Field(ge=1, le=9999)
    type: RoomType


class Room(CreateRoom):
    id: str


class RoomItem(TableItem, CreateRoom):
    """A room as the table stores it: the API fields plus the primary key."""

    entity: ClassVar[str] = 'ROOM'


class PlaceCreate(BaseModel):
    """``places.json`` — a campus location students ask about (cafeteria, library, gym, …)."""

    name: str = Field(min_length=2, max_length=100)
    building: str = Field(min_length=2, max_length=80)
    floor: str = Field(min_length=1, max_length=20)  # free-form: '1', '1–3', 'lobby'
    opening_hours: str = Field(min_length=3, max_length=200)  # 'Mon–Fri 07:30–20:00; Sat 09:00–16:00; Sun closed'


class Place(PlaceCreate):
    id: str


class ScheduleCreate(BaseModel):
    """``schedule.json`` — one weekly class: a course taught to a group, in a room, at a time."""

    course_id: str
    group_id: str
    professor_id: str
    room_id: str
    weekday: Weekday
    start_time: time
    end_time: time


class Schedule(ScheduleCreate):
    id: str


# --- admissions consultations -------------------------------------------------------------------


class SlotCreate(BaseModel):
    """``appointment_slots.json`` — a consultation slot the admissions office opens. It starts open
    and unbooked, so ``status``/``booked_by`` are not part of creating one."""

    date: date
    start_time: time
    end_time: time
    topic: SlotTopic | None = None


class Slot(SlotCreate):
    id: str
    status: SlotStatus
    booked_by: EmailStr | None = None  # set once the slot is booked


# --- student reports ----------------------------------------------------------------------------


class IssueCreate(BaseModel):
    """``reported_issues.json`` — something a student reports. It opens with no votes, so
    ``status``/``votes`` are not part of creating one."""

    category: IssueCategory = Field(
        description="The kind of problem: 'admin', 'facilities', 'food', 'housing', 'schedule' or 'other'."
    )
    text: str = Field(
        min_length=5,
        max_length=500,
        description='The problem in a sentence or two, as the student described it: what is wrong and where.',
    )


class Issue(IssueCreate):
    id: str
    status: IssueStatus = Field(description="Where the issue stands: 'open', 'in_progress' or 'resolved'.")
    votes: int = Field(ge=0, le=1_000_000, description='How many students have backed the issue.')


class IssueItem(Issue):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def pk(self) -> str:
        return self.category

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return build_issue_votes_key(self.votes, self.id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsi1pk(self) -> str:
        return build_issue_status_key(self.status)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsi1sk(self) -> str:
        return self.sk
