from datetime import date
import json
import logging
from typing import cast

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, join_passages
from app.ai_assistant_langchain.agents.qa.compare_programs.graph import get_compare_programs_graph
from app.ai_assistant_langchain.agents.qa.state import AssistantAgentState
from app.ai_assistant_langchain.agents.qa.tool_inputs import (
    CheckScholarshipToolInput,
    CompareProgramsToolInput,
    DEFAULT_COMMON_ISSUE_COUNT,
    FindPersonToolInput,
    FindPlaceToolInput,
    GetScheduleToolInput,
    ListCommonIssuesToolInput,
    RetrieverToolInput,
)
from app.ai_assistant_langchain.applicant_documents.document_rules import check_applicant_documents
from app.ai_assistant_langchain.applicant_documents.enums import DocumentField
from app.ai_assistant_langchain.applicant_documents.extractors import extract_document_fields, get_extractor
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentFields
from app.ai_assistant_langchain.graph.schemas import CustomContext
from app.api.v1.issues.issues_repository import open_issues_repository
from app.api.v1.places.places_repository import open_places_repository
from app.api.v1.professors.professors_repository import open_professors_repository
from app.api.v1.rooms.enums import IssueCategory, Weekday
from app.api.v1.rooms.schemas import IssueCreate
from app.api.v1.schedule.schedule_service import open_schedule_service
from app.api.v1.scholarships.enums import Faculty
from app.api.v1.scholarships.scholarship_rules import evaluate_scholarships
from app.core.exceptions import NotFoundError

logger = logging.getLogger(__name__)

# How many programmes one comparison covers. Not a schema constraint: ``compare_programs`` is
# ``return_direct``, so a list rejected by pydantic would reach the applicant as its error text.
MAX_PROGRAMS = 5

UNREAD_DOCUMENTS_NOTICE = 'The attached documents could not be read this time, so they were not checked.'
IDENTIFIER_FIELDS = (DocumentField.PASSPORT_NUMBER, DocumentField.TAX_ID)
VISIBLE_IDENTIFIER_CHARACTERS = 2

NO_REPORTED_ISSUES_NOTICE = 'No student issues have been reported.'

NO_TIMETABLE_FILTER = (
    'The timetable is looked up by a group, a professor, a course or a weekday, and none of them was given.'
)


@tool(args_schema=RetrieverToolInput)
async def retriever(query: str) -> str:
    """Search the university knowledge base — the handbook — for the passages most relevant to a query.

    It holds the free-form information about the university: programs, courses, admissions,
    tuition, fees, scholarships, deadlines and policies. Takes a natural-language query and returns
    the matching passages as text.
    """
    logger.info('Retriever query: %r', query)
    return join_passages(await get_knowledge_service().search(query))


@tool(args_schema=FindPersonToolInput)
async def find_person(name: str) -> list[dict] | str:
    """Look up a professor or staff member by their name.

    Returns each match with their title, faculty, office/room, email and office hours, or a sentence
    saying that nobody by that name was found.
    """
    async with open_professors_repository() as repo:
        professors = await repo.find_professors_by_name(name)

    logger.info('FindPerson %r -> %s', name, f'{len(professors)} match(es)' if professors else 'none')
    if not professors:
        return f'No professor found with the name {name!r}.'

    return [professor.model_dump() for professor in professors]


@tool(args_schema=FindPlaceToolInput)
async def find_place(name: str) -> dict | str:
    """Look up a campus facility by its name: library, cafeteria, gym, admissions office, dormitory and the like.

    Returns its building, floor and opening hours, or a sentence saying that no place by that name
    was found.
    """
    async with open_places_repository() as repo:
        place = await repo.find_place_by_name(name)

    logger.info('FindPlace %r -> %s', name, place.name if place else 'none')
    if place is None:
        return f'No campus place found with the name {name!r}.'

    return place.model_dump()


@tool(args_schema=GetScheduleToolInput)
async def get_schedule(
    group: str | None = None,
    professor: str | None = None,
    course: str | None = None,
    weekday: Weekday | None = None,
) -> list[dict] | str:
    """Look up the weekly class timetable: which classes are held, when, where and by whom.

    Filters by any of group, professor, course and weekday. Returns each matching class with its
    course, group, professor, building, room number, weekday and start and end time, in timetable
    order, or a sentence saying what was not found.
    """
    filters = {'group': group, 'professor': professor, 'course': course, 'weekday': weekday}
    given = {name: value for name, value in filters.items() if value}
    logger.info('GetSchedule %s', given or '(no filter)')

    if not given:
        return NO_TIMETABLE_FILTER

    try:
        async with open_schedule_service() as service:
            entries = await service.find_schedule(group=group, professor=professor, course=course, weekday=weekday)
    except NotFoundError as unknown:
        return f'{unknown} This does not mean there are no classes.'

    if not entries:
        described = ', '.join(f'{name}={str(value)!r}' for name, value in given.items())
        course_matching = ' A course is matched as part of its stored name.' if course else ''
        return f'No classes found for {described}.{course_matching}'

    return [entry.model_dump() for entry in entries]


@tool(args_schema=CheckScholarshipToolInput)
async def check_scholarship(
    gpa: float,
    family_income: float | None = None,
    entrance_exam_score: float | None = None,
    is_female: bool | None = None,
    faculty: Faculty | None = None,
    annual_tuition: float | None = None,
) -> dict:
    """Apply the university's scholarship rules to one applicant's own facts, thresholds included.

    Returns every scholarship under exactly one of `qualified`, `missed` and `undecided`, each with
    its award and the rule that decided it; `undecided` means a fact the rule needs was not given.
    `awarded` is the one scholarship the applicant would hold, since a student holds one at a time,
    `awarded_reason` says why that one or what is missing to tell, and `tuition_after_award` is the
    annual tuition left to pay with it.
    """
    check = evaluate_scholarships(
        gpa=gpa,
        family_income=family_income,
        entrance_exam_score=entrance_exam_score,
        is_female=is_female,
        faculty=faculty,
        annual_tuition=annual_tuition,
    )
    logger.info(
        'CheckScholarship -> qualified: %s; awarded: %s',
        ', '.join(match.name for match in check.qualified) or 'none',
        check.awarded or 'none',
    )
    return check.model_dump()


@tool
async def validate_documents(runtime: ToolRuntime[CustomContext, AssistantAgentState]) -> Command | str:
    """Read the files attached to the applicant's message and check their admission documents against the rules."""
    attached_files = runtime.context.documents
    try:
        extracted_fields = await extract_document_fields(attached_files, get_extractor())
    except ExceptionGroup as failures:
        logger.warning('ValidateDocuments could not read %d file(s): %r', len(attached_files), failures.exceptions)
        return UNREAD_DOCUMENTS_NOTICE

    verdict = check_applicant_documents(
        [_mask_document_identifiers(fields) for fields in extracted_fields],
        today=date.today(),
        documents_on_file=[
            DocumentFields.model_validate(document) for document in runtime.state.get('documents_on_file', [])
        ],
    )
    logger.info(
        'ValidateDocuments %d file(s), %d document(s) on file -> is_valid: %s; issues: %s',
        len(attached_files),
        len(verdict.documents_on_file),
        verdict.is_valid,
        ', '.join(verdict.issues) or 'none',
    )

    verdict_json = verdict.model_dump(mode='json')
    return Command(
        update={
            'documents_on_file': verdict_json['documents_on_file'],
            'messages': [
                ToolMessage(
                    json.dumps(verdict_json, ensure_ascii=False),
                    tool_call_id=cast(str, runtime.tool_call_id),
                    name=validate_documents.name,
                )
            ],
        }
    )


@tool(args_schema=ListCommonIssuesToolInput)
async def list_common_issues(limit: int = DEFAULT_COMMON_ISSUE_COUNT) -> list[dict] | str:
    """Read the problems students have reported on campus, the most-voted first.

    Returns each issue with its category, its text, its status — 'open', 'in_progress' or 'resolved' —
    and its votes, the number of students who backed it, or a sentence saying that none has been
    reported.
    """
    async with open_issues_repository() as repository:
        issues = await repository.list_most_voted_issues(limit)

    logger.info('ListCommonIssues limit=%d -> %d issue(s)', limit, len(issues))
    if not issues:
        return NO_REPORTED_ISSUES_NOTICE

    return [issue.model_dump(exclude={'id'}) for issue in issues]


@tool(args_schema=IssueCreate)
async def report_issue(category: IssueCategory, text: str) -> dict:
    """File a problem a student reports on campus as a new issue: open, with no votes yet.

    Returns the filed issue with its id, category, text, status and votes.
    """
    async with open_issues_repository() as repository:
        created_issue = await repository.create_issue(IssueCreate(category=category, text=text))

    logger.info('ReportIssue %s filed as %s', category, created_issue.id)
    return created_issue.model_dump()


@tool(args_schema=CompareProgramsToolInput, return_direct=True)
async def compare_programs(programs: list[str]) -> str:
    """Compare two to five named study programmes side by side — focus, courses, duration, admission
    requirements, tuition and career outcomes.

    Looks every programme up itself and returns the finished comparison as text, which goes to the
    applicant word for word and ends the turn.
    """
    # Case-insensitively unique, in the order given. Two names for one programme is not a comparison:
    # each duplicate would search the same thing again and take a second block in a prompt whose
    # every rule is about keeping the programmes apart.
    unique: dict[str, str] = {}
    for program in programs:
        unique.setdefault(program.strip().casefold(), program.strip())

    named = [program for program in unique.values() if program]
    logger.info('CompareProgrammes %s', ' vs '.join(repr(program) for program in named) or '(nothing named)')

    # Answered here rather than by a ``min_length`` on the schema: this tool is ``return_direct``, so
    # a rejected argument list would reach the applicant as pydantic's error text.
    if len(named) < 2:
        if named:
            return f'{named[0]} is one programme, not two. Which other one should I compare it with?'
        return 'Which programmes would you like me to compare?'

    # Every extra programme is another embedding, another search and another block in the merge
    # prompt, and a side-by-side answer stops being readable long before the cost stops growing.
    compared, dropped = named[:MAX_PROGRAMS], named[MAX_PROGRAMS:]

    graph = get_compare_programs_graph()
    state = await graph.ainvoke({'programs': compared})
    comparison: str = state['comparison']

    if not dropped:
        return comparison

    return (
        f'{comparison}\n\nI compared the first {MAX_PROGRAMS} — {", ".join(dropped)} '
        f'{"is" if len(dropped) == 1 else "are"} not in here. Ask again for '
        f'{"it" if len(dropped) == 1 else "them"} and I will cover '
        f'{"it" if len(dropped) == 1 else "them"}.'
    )


ASSISTANT_TOOLS = [
    retriever,
    find_person,
    find_place,
    get_schedule,
    check_scholarship,
    validate_documents,
    list_common_issues,
    report_issue,
    compare_programs,
]


DIRECT_ANSWER_TOOLS = frozenset(item.name for item in ASSISTANT_TOOLS if item.return_direct)


def _mask_document_identifiers(fields: DocumentFields) -> DocumentFields:
    masked_identifiers = {
        field.value: _mask_identifier(identifier)
        for field in IDENTIFIER_FIELDS
        if (identifier := getattr(fields, field)) and identifier.strip()
    }
    return fields.model_copy(update=masked_identifiers)


def _mask_identifier(identifier: str) -> str:
    visible_length = VISIBLE_IDENTIFIER_CHARACTERS if len(identifier) > VISIBLE_IDENTIFIER_CHARACTERS else 0
    hidden_length = len(identifier) - visible_length
    return '*' * hidden_length + identifier[hidden_length:]
