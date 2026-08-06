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
from app.core.dynamodb.base_items import ListedItem

# Todo refactor to diff modules


class GroupCreate(BaseModel):
    """``groups.json`` — a student cohort enrolled on one programme."""

    name: str = Field(min_length=2, max_length=30)  # e.g. 'CS-1'
    program_id: str


class Group(GroupCreate):
    id: str


class ProfessorCreate(BaseModel):
    """``professors.json`` — teaching staff, each with an office and consultation hours."""

    full_name: str = Field(min_length=2, max_length=120)
    title: ProfessorTitle
    faculty_id: str
    email: EmailStr
    room_id: str  # their office, a room of type OFFICE
    office_hours: str = Field(min_length=3, max_length=120)  # free-form, e.g. 'Mon 14:00–16:00'


class Professor(ProfessorCreate):
    id: str


class CourseCreate(BaseModel):
    """``courses.json`` — a taught course, owned by a faculty and run by one professor."""

    name: str = Field(min_length=2, max_length=120)
    faculty: str = Field(min_length=2, max_length=100)  # faculty *name*, not id
    professor_id: str


class Course(CourseCreate):
    id: str


class CreateRoom(BaseModel):
    building: str = Field(min_length=2, max_length=80)
    floor: int = Field(ge=-2, le=50)
    number: int = Field(ge=1, le=9999)
    type: RoomType


class Room(CreateRoom):
    id: str


class RoomItem(ListedItem, CreateRoom):
    """A room as the table stores it: the API fields plus the single-table and GSI1 keys."""

    entity: ClassVar[str] = 'ROOM'

    @computed_field
    @property
    def gsi1sk(self) -> str:
        """Orders the room listing by building, then door number — zero-padded, because a sort key
        is compared as a string ('0105' < '0201', where '105' > '201' would be)."""
        return f'{self.building}#{self.number:04d}'


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

    category: IssueCategory
    text: str = Field(min_length=5, max_length=500)


class Issue(IssueCreate):
    id: str
    status: IssueStatus
    votes: int = Field(ge=0, le=1_000_000)
