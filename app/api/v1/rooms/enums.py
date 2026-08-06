from enum import StrEnum

# Todo move later to other entities


class ProfessorTitle(StrEnum):
    PROFESSOR = 'Professor'
    ASSOCIATE_PROFESSOR = 'Associate Professor'


class RoomType(StrEnum):
    LECTURE_HALL = 'lecture_hall'
    LAB = 'lab'
    OFFICE = 'office'


class Weekday(StrEnum):
    MON = 'Mon'
    TUE = 'Tue'
    WED = 'Wed'
    THU = 'Thu'
    FRI = 'Fri'
    SAT = 'Sat'
    SUN = 'Sun'


class SlotStatus(StrEnum):
    OPEN = 'open'
    BOOKED = 'booked'


class SlotTopic(StrEnum):
    GENERAL = 'general'
    PROGRAMS = 'programs'
    SCHOLARSHIPS = 'scholarships'
    HOUSING = 'housing'
    INTERNATIONAL = 'international'


class IssueCategory(StrEnum):
    FACILITIES = 'facilities'
    FOOD = 'food'
    SCHEDULE = 'schedule'
    HOUSING = 'housing'
    ADMIN = 'admin'
    OTHER = 'other'


class IssueStatus(StrEnum):
    OPEN = 'open'
    IN_PROGRESS = 'in_progress'
    RESOLVED = 'resolved'
