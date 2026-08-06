from enum import StrEnum


class Degree(StrEnum):
    BSC = 'BSc'
    BA = 'BA'
    LLB = 'LLB'


class ProgramCreateOutcome(StrEnum):
    CREATED = 'created'
    FACULTY_NOT_FOUND = 'faculty_not_found'
    NAME_TAKEN = 'name_taken'
