import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, TypeAlias

from app.api.v1.professors.professors_repository import ProfessorsRepository
from app.api.v1.rooms.enums import Weekday
from app.api.v1.rooms.schemas import CourseItem, GroupItem, ProfessorItem, RoomItem, Schedule
from app.api.v1.schedule.schedule_repository import ScheduleRepository
from app.api.v1.schedule.schemas import ScheduleEntry
from app.core.dynamodb.base_items import TableItem
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.indexes import normalize_name
from app.core.exceptions import NotFoundError
from app.settings import get_settings

logger = logging.getLogger(__name__)


Names: TypeAlias = dict[str, str]


_WEEKDAY_ORDER = {weekday: position for position, weekday in enumerate(Weekday)}
_TIME_FORMAT = '%H:%M'


def _strip_value(value: str | None) -> str | None:
    return (value.strip() or None) if value else None


class ScheduleService:
    def __init__(
        self,
        schedule: ScheduleRepository,
        professors: ProfessorsRepository,
        courses: DynamoDBService,
        rooms: DynamoDBService,
    ) -> None:
        self._schedule = schedule
        self._professors = professors
        self._courses = courses
        self._rooms = rooms

    async def find_schedule(
        self,
        group: str | None = None,
        professor: str | None = None,
        course: str | None = None,
        weekday: Weekday | None = None,
    ) -> list[ScheduleEntry]:
        groups, professors = await self._resolve(_strip_value(group), _strip_value(professor))

        classes = [
            row
            for row in await self._read_classes(groups, professors)
            if (weekday is None or row.weekday == weekday) and (professors is None or row.professor_id in professors)
        ]

        courses: Names = {}
        wanted = normalize_name(course or '')
        if wanted:
            courses = await self._find_names(self._courses, CourseItem, {row.course_id for row in classes}, 'name')
            classes = [row for row in classes if wanted in normalize_name(courses.get(row.course_id, ''))]

        entries = await self._hydrate(classes, courses=courses, groups=groups or {}, professors=professors or {})
        return sorted(entries, key=lambda entry: (_WEEKDAY_ORDER[entry.weekday], entry.start_time))

    async def _resolve(self, group: str | None, professor: str | None) -> tuple[Names | None, Names | None]:
        async with asyncio.TaskGroup() as lookups:
            groups_task = lookups.create_task(self._schedule.find_groups_by_name(group)) if group else None
            professors_task = (
                lookups.create_task(self._professors.find_professors_by_name(professor)) if professor else None
            )

        groups: Names | None = None
        if groups_task is not None:
            groups = groups_task.result()
            if not groups:
                raise NotFoundError(f'No group is called {group!r}.')

        professors: Names | None = None
        if professors_task is not None:
            professors = {str(found.id): found.full_name for found in professors_task.result() or []}
            if not professors:
                raise NotFoundError(
                    f'No professor was found by the name {professor!r}. The lookup takes a full name '
                    'or a first name, not a surname alone.'
                )

        return groups, professors

    async def _read_classes(self, groups: Names | None, professors: Names | None) -> list[Schedule]:
        if groups is not None:
            reads = [self._schedule.list_group_classes(group_id) for group_id in groups]
        elif professors is not None:
            reads = [self._schedule.list_professor_classes(professor_id) for professor_id in professors]
        else:
            return await self._schedule.list_classes()

        async with asyncio.TaskGroup() as queries:
            tasks = [queries.create_task(read) for read in reads]

        return [row for task in tasks for row in task.result()]

    async def _hydrate(
        self, classes: list[Schedule], courses: Names, groups: Names, professors: Names
    ) -> list[ScheduleEntry]:
        if not classes:
            return []

        async with asyncio.TaskGroup() as lookups:
            courses_task = lookups.create_task(
                self._find_names(self._courses, CourseItem, {row.course_id for row in classes} - courses.keys(), 'name')
            )
            groups_task = lookups.create_task(
                self._find_names(self._schedule, GroupItem, {row.group_id for row in classes} - groups.keys(), 'name')
            )
            professors_task = lookups.create_task(
                self._find_names(
                    self._professors,
                    ProfessorItem,
                    {row.professor_id for row in classes} - professors.keys(),
                    'full_name',
                )
            )
            rooms_task = lookups.create_task(self._find_rows(self._rooms, RoomItem, {row.room_id for row in classes}))

        courses = courses | courses_task.result()
        groups = groups | groups_task.result()
        professors = professors | professors_task.result()
        rooms = rooms_task.result()

        entries: list[ScheduleEntry] = []
        for row in classes:
            room = rooms.get(row.room_id, {})
            entry = ScheduleEntry(
                course=courses.get(row.course_id),
                group=groups.get(row.group_id),
                professor=professors.get(row.professor_id),
                building=room.get('building'),
                room_number=room.get('number'),
                weekday=row.weekday,
                start_time=row.start_time.strftime(_TIME_FORMAT),
                end_time=row.end_time.strftime(_TIME_FORMAT),
            )
            if None in (entry.course, entry.group, entry.professor, entry.building):
                logger.warning('Class %s references a record that no longer exists: %s', row.id, entry)
            entries.append(entry)

        return entries

    @staticmethod
    async def _find_rows(table: DynamoDBService, item: type[TableItem], ids: set[str]) -> dict[str, dict]:
        if not ids:
            return {}

        rows = await table.batch_get([item.key(item_id) for item_id in ids])
        return {str(row['id']): row for row in rows if 'id' in row}

    @classmethod
    async def _find_names(cls, table: DynamoDBService, item: type[TableItem], ids: set[str], field: str) -> Names:
        rows = await cls._find_rows(table, item, ids)
        return {item_id: str(row[field]) for item_id, row in rows.items() if field in row}


@asynccontextmanager
async def open_schedule_service() -> AsyncGenerator[ScheduleService, None]:
    """The timetable lookup on one client: the four tables it reads share a connection rather than
    each opening its own."""
    settings = get_settings()
    async with open_dynamo_client(get_aioboto_session(), settings) as client:
        yield ScheduleService(
            schedule=ScheduleRepository(client, settings),
            professors=ProfessorsRepository(client, settings),
            courses=DynamoDBService(client, settings.DYNAMODB_COURSES_TABLE),
            rooms=DynamoDBService(client, settings.DYNAMODB_ROOMS_TABLE),
        )
