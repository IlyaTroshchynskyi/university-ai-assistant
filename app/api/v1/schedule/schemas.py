from pydantic import BaseModel, Field

from app.api.v1.rooms.enums import Weekday


class ScheduleEntry(BaseModel):
    course: str | None = Field(
        description="The course taught, e.g. 'Intro to Programming'. None when the course record no longer exists.",
    )
    group: str | None = Field(
        description="The student group's code, e.g. 'CS-1'. None when the group record no longer exists.",
    )
    professor: str | None = Field(
        description=(
            "Who teaches the class, by full name, e.g. 'Dr. Alan Whitfield'. None when the professor "
            'record no longer exists.'
        ),
    )
    building: str | None = Field(
        description="The building the class is held in, e.g. 'Turing Hall'. None when the room no longer exists.",
    )
    room_number: int | None = Field(
        description='The door number of the room in that building. None when the room no longer exists.',
    )
    weekday: Weekday = Field(description="The day of the week the class is held on: 'Mon' to 'Sun'.")
    start_time: str = Field(description="When the class starts, 24-hour 'HH:MM'.")
    end_time: str = Field(description="When the class ends, 24-hour 'HH:MM'.")
