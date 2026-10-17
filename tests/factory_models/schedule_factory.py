from uuid import uuid4

from pydantic import BaseModel

from db.load_dynamodb import WEEKDAY_NUM

from app.api.v1.rooms.enums import Weekday


class NamedGroupRow(BaseModel):
    pk: str
    sk: str = '#META'
    gsi1pk: str
    gsi1sk: str
    entity_type: str = 'group'
    id: str
    name: str
    program_id: str

    @classmethod
    def build(cls, name: str) -> 'NamedGroupRow':
        group_id, program_id = str(uuid4()), str(uuid4())
        return cls(
            pk=f'GROUP#{group_id}',
            gsi1pk=f'PROGRAM#{program_id}',
            gsi1sk=f'GROUP#{group_id}',
            id=group_id,
            name=name,
            program_id=program_id,
        )


class CourseRow(BaseModel):
    pk: str
    sk: str = '#META'
    gsi1pk: str
    gsi1sk: str
    gsi2pk: str
    gsi2sk: str
    entity_type: str = 'course'
    id: str
    name: str
    faculty_id: str
    professor_id: str

    @classmethod
    def build(cls, name: str, professor_id: str) -> 'CourseRow':
        course_id, faculty_id = str(uuid4()), str(uuid4())
        return cls(
            pk=f'COURSE#{course_id}',
            gsi1pk=f'FACULTY#{faculty_id}',
            gsi1sk=f'COURSE#{name}',
            gsi2pk=f'PROF#{professor_id}',
            gsi2sk=f'COURSE#{course_id}',
            id=course_id,
            name=name,
            faculty_id=faculty_id,
            professor_id=professor_id,
        )


class ClassRow(BaseModel):
    pk: str
    sk: str
    gsi1pk: str
    gsi1sk: str
    gsi2pk: str
    gsi2sk: str
    entity_type: str = 'schedule'
    id: str
    course_id: str
    group_id: str
    professor_id: str
    room_id: str
    weekday: str
    start_time: str
    end_time: str

    @classmethod
    def build(
        cls,
        course_id: str,
        group_id: str,
        professor_id: str,
        room_id: str,
        weekday: Weekday,
        start_time: str,
        end_time: str,
    ) -> 'ClassRow':
        sort_key = f'SCHED#{WEEKDAY_NUM[weekday]}#{start_time}'
        return cls(
            pk=f'GROUP#{group_id}',
            sk=sort_key,
            gsi1pk=f'PROF#{professor_id}',
            gsi1sk=sort_key,
            gsi2pk=f'ROOM#{room_id}',
            gsi2sk=sort_key,
            id=str(uuid4()),
            course_id=course_id,
            group_id=group_id,
            professor_id=professor_id,
            room_id=room_id,
            weekday=weekday,
            start_time=start_time,
            end_time=end_time,
        )
