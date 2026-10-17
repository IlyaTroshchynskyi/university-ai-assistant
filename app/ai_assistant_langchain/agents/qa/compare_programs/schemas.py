from operator import add
from typing import Annotated, TypedDict


class ProgramFacts(TypedDict):
    program: str
    summary: str


class CompareState(TypedDict):
    programs: list[str]
    results: Annotated[list[ProgramFacts], add]
    comparison: str


class GatherInput(TypedDict):
    program: str


class GatherUpdate(TypedDict):
    results: list[ProgramFacts]


class MergeUpdate(TypedDict):
    comparison: str
