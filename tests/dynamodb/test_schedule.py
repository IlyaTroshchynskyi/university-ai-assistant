from collections import Counter
from contextlib import contextmanager
import json
from typing import Generator
from unittest.mock import patch

import pytest
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.tools import get_schedule, NARROW_THE_TIMETABLE
from app.api.v1.rooms.enums import Weekday
from app.api.v1.schedule.schedule_service import open_schedule_service
from app.api.v1.schedule.schemas import ScheduleEntry
from app.core.dynamodb.base_service import DynamoDBService
from app.core.exceptions import NotFoundError
from app.settings import get_settings
from tests.factories.factory_creators import (
    create_test_class_row,
    create_test_course_row,
    create_test_named_group_row,
    create_test_professor_row,
    create_test_room,
)
from tests.factories.factory_deleters import delete_test_room
from tests.factories.rooms_factory import RoomCreationFactory

ALAN = 'Dr. Alan Whitfield'
SOPHIA = 'Dr. Sophia Martens'
INTRO = 'Intro to Programming'
STRUCTURES = 'Data Structures'
TURING = 'Turing Hall'
LOVELACE = 'Lovelace Building'

PATH_TO_SCHEDULE_SERVICE = 'app.ai_assistant_langchain.tools.open_schedule_service'


@contextmanager
def recorded_batch_gets() -> Generator[Counter[str], None, None]:
    fetched: Counter[str] = Counter()
    batch_get = DynamoDBService.batch_get

    async def counting(self: DynamoDBService, keys: list[dict], consistent_read: bool = False) -> list[dict]:
        fetched[self._table_name] += len(keys)
        return await batch_get(self, keys, consistent_read)

    with patch.object(DynamoDBService, 'batch_get', counting):
        yield fetched


class TestBaseScheduleClass:
    @pytest.fixture(autouse=True)
    async def _a_provide_timetable(
        self,
        dynamo_client: DynamoDBClient,
        groups_table: DynamoDBService,
        professors_table: DynamoDBService,
        courses_table: DynamoDBService,
        rooms_table: DynamoDBService,
    ) -> None:
        self.dynamo_client = dynamo_client

        turing = await create_test_room(RoomCreationFactory.build(building=TURING, number=201), dynamo_client)
        lovelace = await create_test_room(RoomCreationFactory.build(building=LOVELACE, number=210), dynamo_client)
        self.lovelace_room_id = lovelace.id

        alan = await create_test_professor_row(dynamo_client, id=1, full_name=ALAN)
        sophia = await create_test_professor_row(dynamo_client, id=2, full_name=SOPHIA)

        intro = await create_test_course_row(dynamo_client, INTRO, alan.id)
        structures = await create_test_course_row(dynamo_client, STRUCTURES, sophia.id)

        cs1 = await create_test_named_group_row(dynamo_client, 'CS-1')
        cs2 = await create_test_named_group_row(dynamo_client, 'CS-2')

        for course, group, professor, room, weekday, start_time, end_time in (
            (intro, cs1, alan, turing, Weekday.WED, '09:00', '10:30'),
            (structures, cs1, sophia, lovelace, Weekday.MON, '11:00', '12:30'),
            (intro, cs1, alan, turing, Weekday.MON, '09:00', '10:30'),
            (intro, cs2, alan, turing, Weekday.TUE, '09:00', '10:30'),
        ):
            await create_test_class_row(
                dynamo_client, course.id, group.id, professor.id, room.id, weekday, start_time, end_time
            )

        self.cs1_mon_intro = ScheduleEntry(
            course=INTRO,
            group='CS-1',
            professor=ALAN,
            building=TURING,
            room_number=201,
            weekday=Weekday.MON,
            start_time='09:00',
            end_time='10:30',
        )
        self.cs1_mon_structures = ScheduleEntry(
            course=STRUCTURES,
            group='CS-1',
            professor=SOPHIA,
            building=LOVELACE,
            room_number=210,
            weekday=Weekday.MON,
            start_time='11:00',
            end_time='12:30',
        )
        self.cs1_wed_intro = self.cs1_mon_intro.model_copy(update={'weekday': Weekday.WED})
        self.cs2_tue_intro = self.cs1_mon_intro.model_copy(update={'group': 'CS-2', 'weekday': Weekday.TUE})

    @staticmethod
    async def find_schedule(
        group: str | None = None,
        professor: str | None = None,
        course: str | None = None,
        weekday: Weekday | None = None,
    ) -> list[ScheduleEntry]:
        async with open_schedule_service() as service:
            return await service.find_schedule(group=group, professor=professor, course=course, weekday=weekday)


class TestFindSchedule(TestBaseScheduleClass):
    async def test_group_returns_its_classes_with_names_in_timetable_order(self) -> None:
        found = await self.find_schedule(group='CS-1')

        assert found == [self.cs1_mon_intro, self.cs1_mon_structures, self.cs1_wed_intro]

    async def test_group_is_found_whatever_case_it_is_written_in(self) -> None:
        found = await self.find_schedule(group='cs-1')

        assert found == [self.cs1_mon_intro, self.cs1_mon_structures, self.cs1_wed_intro]

    async def test_professor_returns_their_classes_across_groups(self) -> None:
        found = await self.find_schedule(professor='Alan')

        assert found == [self.cs1_mon_intro, self.cs2_tue_intro, self.cs1_wed_intro]

    async def test_professor_and_weekday(self) -> None:
        found = await self.find_schedule(professor='Alan', weekday=Weekday.MON)

        assert found == [self.cs1_mon_intro]

    async def test_group_and_course_matches_part_of_the_course_name(self) -> None:
        found = await self.find_schedule(group='CS-1', course='programming')

        assert found == [self.cs1_mon_intro, self.cs1_wed_intro]

    async def test_group_and_professor_leaves_out_another_professors_class(self) -> None:
        found = await self.find_schedule(group='CS-1', professor='Sophia')

        assert found == [self.cs1_mon_structures]

    async def test_course_alone(self) -> None:
        found = await self.find_schedule(course='data structures')

        assert found == [self.cs1_mon_structures]

    async def test_classes_read_from_several_groups_are_put_in_timetable_order(self) -> None:
        found = await self.find_schedule(course='programming')

        assert found == [self.cs1_mon_intro, self.cs2_tue_intro, self.cs1_wed_intro]

    async def test_weekday_alone(self) -> None:
        found = await self.find_schedule(weekday=Weekday.TUE)

        assert found == [self.cs2_tue_intro]

    async def test_unknown_group_is_an_error_not_an_empty_timetable(self) -> None:
        with pytest.raises(NotFoundError, match='CS-9'):
            await self.find_schedule(group='CS-9')

    async def test_unknown_professor_is_an_error_not_an_empty_timetable(self) -> None:
        with pytest.raises(NotFoundError, match='Nobody'):
            await self.find_schedule(professor='Nobody')

    async def test_course_is_matched_whatever_its_spacing(self) -> None:
        found = await self.find_schedule(course='  data   structures ')

        assert found == [self.cs1_mon_structures]

    async def test_course_filter_fetches_the_other_names_only_for_the_classes_it_keeps(self) -> None:
        settings = get_settings()

        with recorded_batch_gets() as fetched:
            found = await self.find_schedule(course='data structures')

        assert found == [self.cs1_mon_structures]
        assert +fetched == Counter(
            {
                settings.DYNAMODB_COURSES_TABLE: 2,
                settings.DYNAMODB_GROUPS_TABLE: 1,
                settings.DYNAMODB_PROFESSORS_TABLE: 1,
                settings.DYNAMODB_ROOMS_TABLE: 1,
            }
        )

    async def test_names_the_lookup_resolved_are_not_fetched_again(self) -> None:
        settings = get_settings()

        with recorded_batch_gets() as fetched:
            found = await self.find_schedule(group='CS-1', professor='Sophia')

        assert found == [self.cs1_mon_structures]
        assert +fetched == Counter({settings.DYNAMODB_COURSES_TABLE: 1, settings.DYNAMODB_ROOMS_TABLE: 1})

    async def test_class_outlives_its_deleted_room(self) -> None:
        await delete_test_room(self.lovelace_room_id, self.dynamo_client)

        found = await self.find_schedule(group='CS-1', weekday=Weekday.MON)

        roomless = self.cs1_mon_structures.model_copy(update={'building': None, 'room_number': None})
        assert found == [self.cs1_mon_intro, roomless]


class TestGetScheduleTool(TestBaseScheduleClass):
    async def test_model_reads_the_matches_as_json(self) -> None:
        message = await get_schedule.ainvoke(
            {
                'type': 'tool_call',
                'name': get_schedule.name,
                'args': {'group': 'CS-1', 'weekday': 'Mon'},
                'id': 'call-1',
            }
        )

        assert json.loads(message.content) == [
            {
                'course': INTRO,
                'group': 'CS-1',
                'professor': ALAN,
                'building': TURING,
                'room_number': 201,
                'weekday': 'Mon',
                'start_time': '09:00',
                'end_time': '10:30',
            },
            {
                'course': STRUCTURES,
                'group': 'CS-1',
                'professor': SOPHIA,
                'building': LOVELACE,
                'room_number': 210,
                'weekday': 'Mon',
                'start_time': '11:00',
                'end_time': '12:30',
            },
        ]

    async def test_no_match_names_the_filters_that_were_sent(self) -> None:
        found = await get_schedule.ainvoke({'group': 'CS-2', 'weekday': 'Mon'})

        assert found == "No classes found for group='CS-2', weekday='Mon'."

    async def test_unknown_name_is_told_apart_from_an_empty_timetable(self) -> None:
        found = await get_schedule.ainvoke({'group': 'CS-9', 'weekday': 'Mon'})

        assert found == (
            "No group is called 'CS-9'. Check the name with the applicant; this does not mean there are no classes."
        )

    async def test_no_match_on_a_course_says_how_courses_are_matched(self) -> None:
        found = await get_schedule.ainvoke({'group': 'CS-1', 'course': 'Introduction to Programming'})

        assert found == (
            "No classes found for group='CS-1', course='Introduction to Programming'. A course is "
            'matched as part of its stored name, so try a shorter, distinctive part of it before '
            'telling the applicant there are none.'
        )

    async def test_no_filter_asks_to_narrow_without_reading_anything(self) -> None:
        with patch(PATH_TO_SCHEDULE_SERVICE) as opened:
            found = await get_schedule.ainvoke({})

        assert found == NARROW_THE_TIMETABLE
        opened.assert_not_called()

    async def test_empty_string_is_a_filter_that_was_not_given(self) -> None:
        with patch(PATH_TO_SCHEDULE_SERVICE) as opened:
            found = await get_schedule.ainvoke({'group': '', 'course': ''})

        assert found == NARROW_THE_TIMETABLE
        opened.assert_not_called()

    async def test_whitespace_is_a_filter_that_was_not_given(self) -> None:
        with patch(PATH_TO_SCHEDULE_SERVICE) as opened:
            found = await get_schedule.ainvoke({'professor': '  ', 'course': ' '})

        assert found == NARROW_THE_TIMETABLE
        opened.assert_not_called()
