"""Seed loader for the local DynamoDB (the model from 02-dynamodb-model.md, split as 03-table-split.md).

One table per entity, with one deliberate exception:
  · faculties, programs, professors, courses, rooms, places — an entity each, a name
                        reservation filed beside the entity that owns it;
  · academic_groups   — groups *and* their schedule rows, the one pair a single query
                        has to return together;
  · appointment_slots — independent, dated, with a TTL on expires_at;
  · reported_issues   — independent (student reports).

What the script does:
  1. (re)creates every table with its own GSI set (+ enables TTL on appointment_slots);
  2. reads db/seed/*.json and turns the records into items following the key maps;
  3. loads everything in batches, table by table;
  4. runs demo queries, to show the model in action.

Running it (the local DynamoDB has to be up — docker compose up -d dynamodb):

    uv run python db/load_dynamodb.py

The endpoint and credentials come from environment variables (the defaults match app/settings.py
and docker-compose.yml), so OPENAI_API_KEY and the rest aren't needed to run it.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key

# The model, not the API: FacultyNameItem is the single definition of the key reserving a faculty
# name, and the seed has to write it the same way POST /faculties does — a hand-rolled copy here
# would drift the moment either side changed.
from app.api.v1.faculty.schemas import FacultyNameItem
from app.api.v1.programs.schemas import ProgramNameItem
from app.core.dynamodb.indexes import normalize_name, normalize_name_key

logger = logging.getLogger(__name__)

# --- configuration (defaults match app/settings.py) ---------------------------------------------
ENDPOINT_URL = os.getenv('DYNAMODB_ENDPOINT_URL', 'http://localhost:8001')
REGION = os.getenv('AWS_REGION', 'us-east-1')
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'dummy')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy')

T_FACULTIES = os.getenv('DYNAMODB_FACULTIES_TABLE', 'faculties')
T_PROGRAMS = os.getenv('DYNAMODB_PROGRAMS_TABLE', 'programs')
T_PROFESSORS = os.getenv('DYNAMODB_PROFESSORS_TABLE', 'professors')
T_COURSES = os.getenv('DYNAMODB_COURSES_TABLE', 'courses')
T_ROOMS = os.getenv('DYNAMODB_ROOMS_TABLE', 'rooms')
T_PLACES = os.getenv('DYNAMODB_PLACES_TABLE', 'places')
T_GROUPS = os.getenv('DYNAMODB_GROUPS_TABLE', 'academic_groups')
T_SLOTS = 'appointment_slots'
T_ISSUES = 'reported_issues'
T_CHECKPOINTS = os.getenv('DYNAMODB_CHECKPOINTS_TABLE', 'agent_checkpoints')

# Which GSIs each table is created with. The seeder is the only place that knows the whole map, so
# it is written once here rather than spelled out again at every ``recreate_table`` call.
GSI_NAME_INDEX = ('GSI_NAME', 'gsi_name_pk', 'gsi_name_sk')
TABLE_INDEXES: dict[str, tuple[list[int], list[tuple[str, str, str]]]] = {
    T_FACULTIES: ([], []),  # listed by Scan; its name reservations are keyed, not indexed
    T_PROGRAMS: ([1], []),  # P4 — a faculty's programmes
    T_PROFESSORS: ([1], [GSI_NAME_INDEX]),  # P5 — a faculty's professors; P15 — search by name
    T_COURSES: ([1, 2], []),  # P6 — a faculty's courses; P8 — a professor's courses
    T_ROOMS: ([], []),  # listed by Scan
    T_PLACES: ([], []),  # searched by Scan + contains (P14)
    T_GROUPS: ([1, 2], []),  # P7/P2 share GSI1 here; P3 — a room's occupancy
    T_SLOTS: ([1, 2], []),
    T_ISSUES: ([1], []),
}

SEED_DIR = Path(__file__).parent / 'seed'

# Monday=1 … Sunday=7 — so that weekdays sort correctly in the SK (otherwise 'Fri' < 'Mon').
WEEKDAY_NUM = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((SEED_DIR / f'{name}.json').read_text(encoding='utf-8'))


def _clean(item: dict[str, Any]) -> dict[str, Any]:
    """DynamoDB doesn't store None — drop the empty attributes (e.g. topic/booked_by on an open slot)."""
    return {k: v for k, v in item.items() if v is not None}


# The identifier fields: an entity's own id and every reference to another one. Kept as a superset:
# not every listed field is currently reached (groups are built as a literal dict, so their
# program_id is stringified inline), but an entity later routed through _str_ids gets it for free.
_ID_FIELDS = frozenset({'id', 'faculty_id', 'program_id', 'professor_id', 'room_id', 'group_id', 'course_id'})


def _str_ids(record: dict[str, Any]) -> dict[str, Any]:
    """The seed's numeric ids, as strings — because the API schemas (``app/api/v1/rooms/schemas.py``)
    declare both ``id`` and every reference as a string: otherwise ``Room(**row)`` fails on a seeded
    record (``id`` comes back from DynamoDB as a ``Decimal``), while one created through the API
    passes, its id being a UUID. Leaves the keys (pk/sk/GSI) alone — an id reaches those through an
    f-string anyway."""
    return {k: str(v) if k in _ID_FIELDS else v for k, v in record.items()}


def _end_of_day_epoch(date_str: str) -> int:
    """Epoch seconds at the end of the day (UTC) — the TTL value: a past slot deletes itself."""
    day = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int((day + timedelta(days=1)).timestamp())


def _reject_duplicate_faculty_names(faculties: list[dict[str, Any]]) -> None:
    """Stop the load if two seeded faculties share a name.

    Programmes and courses name their faculty rather than referencing its id, so the lookup below is
    keyed by name — and a repeated name would quietly resolve to whichever faculty came last, taking
    every programme and course pointing at the other one with it. The API refuses a duplicate name
    outright (``FacultyNameItem``); the seed has no business creating what the API forbids, so it
    fails loudly here instead."""
    seen = Counter(normalize_name(f['name']) for f in faculties)
    repeated = [name for name, count in seen.items() if count > 1]
    if repeated:
        raise ValueError(f'faculties.json: duplicate faculty names: {", ".join(sorted(repeated))}')


def _reject_duplicate_program_names(programs: list[dict[str, Any]]) -> None:
    """Stop the load if two seeded programmes share a name within one faculty.

    ``POST /programs`` reserves that pair in a row of its own. Seeding a collision would write one
    reservation row for two programmes, leaving whichever lost the race holding a name that nothing
    reserves — so, as with faculties, the seed refuses to create what the API forbids."""
    # Both halves normalized, as ``_reject_duplicate_faculty_names`` does — grouping by the raw
    # faculty name would read 'Computer  Science' and 'Computer Science' as two different faculties
    # and let a duplicate programme name through.
    seen = Counter((normalize_name(p['faculty']), normalize_name(p['name'])) for p in programs)
    repeated = [f'{faculty}/{name}' for (faculty, name), count in seen.items() if count > 1]
    if repeated:
        raise ValueError(f'programs.json: duplicate names within a faculty: {", ".join(sorted(repeated))}')


# --- seed -> items transformation (following the key maps in 02-dynamodb-model.md) --------------
def build_items() -> dict[str, list[dict[str, Any]]]:
    """Every seeded row, keyed by the table it belongs in.

    A mapping rather than one flat list, because there is no longer one destination — and building
    it in a single pass is what lets the dependant counters be computed from the same records that
    produce the rows they count."""
    faculties = _load('faculties')
    programs = _load('programs')
    groups = _load('groups')
    professors = _load('professors')
    courses = _load('courses')
    rooms = _load('rooms')
    places = _load('places')
    schedule = _load('schedule')

    # The seed stores a faculty by NAME, so it is resolved to a faculty_id here. A programme is keyed
    # by its **id**, matching what ``POST /programs`` writes, and everything referencing a programme
    # references that id — a primary key cannot be renamed, so nothing human-editable belongs in one.
    _reject_duplicate_faculty_names(faculties)
    _reject_duplicate_program_names(programs)
    faculty_id_by_name = {f['name']: f['id'] for f in faculties}

    # The counters the delete guards read (`FacultyRepository.delete_faculty`). The API maintains
    # them inside the transaction that creates or removes a child; the seeder writes with
    # BatchWriteItem and bypasses the repositories entirely, so it has to count for itself. Getting
    # this wrong is invisible until someone deletes a seeded faculty and orphans six programmes.
    faculty_dependants = _count_faculty_dependants(programs, professors, courses, faculty_id_by_name)
    program_dependants = Counter(str(g['program_id']) for g in groups)

    faculty_items: list[dict[str, Any]] = []
    for f in faculties:
        faculty_items.append(
            {
                'pk': f'FACULTY#{f["id"]}',
                'sk': '#META',
                'entity_type': 'faculty',
                'dependants': faculty_dependants[str(f['id'])],
                **_str_ids(f),
            }
        )
        # The row reserving the name, written exactly as POST /faculties writes it. A seeded faculty
        # without one leaves its name free for the API to hand out a second time.
        faculty_items.append(FacultyNameItem(name=f['name'], faculty_id=str(f['id'])).model_dump())

    program_items: list[dict[str, Any]] = []
    for p in programs:
        fid = faculty_id_by_name[p['faculty']]
        program_items.append(
            {
                'pk': f'PROGRAM#{p["id"]}',
                'sk': '#META',
                'gsi1pk': f'FACULTY#{fid}',
                'gsi1sk': f'PROGRAM#{p["name"]}',
                'entity_type': 'program',
                'dependants': program_dependants[str(p['id'])],
                'id': str(p['id']),
                'name': p['name'],
                'faculty_id': str(fid),
                'degree': p['degree'],
                'duration_years': p['duration_years'],
                'tuition_usd': p['tuition_usd'],
            }
        )
        # The row reserving the programme's name, written exactly as POST /programs writes it. A
        # seeded programme without one leaves its name free for the API to hand out a second time.
        program_items.append(ProgramNameItem(faculty_id=str(fid), name=p['name'], program_id=str(p['id'])).model_dump())

    # Groups and schedule share a table *and* a partition: a class is filed under its group's pk, so
    # 'the group and its timetable' is one query (P1). The only colocation the split kept.
    group_items: list[dict[str, Any]] = []
    for g in groups:
        group_items.append(
            {
                'pk': f'GROUP#{g["id"]}',
                'sk': '#META',
                # By programme **id**, matching the programme's own pk.
                'gsi1pk': f'PROGRAM#{g["program_id"]}',
                'gsi1sk': f'GROUP#{g["id"]}',
                'entity_type': 'group',
                'id': str(g['id']),
                'name': g['name'],
                'program_id': str(g['program_id']),
            }
        )

    for s in schedule:  # one item serves P1 (group) / P2 (professor) / P3 (room)
        wd = WEEKDAY_NUM[s['weekday']]
        sched_sk = f'SCHED#{wd}#{s["start_time"]}'
        group_items.append(
            {
                'pk': f'GROUP#{s["group_id"]}',
                'sk': sched_sk,
                'gsi1pk': f'PROF#{s["professor_id"]}',
                'gsi1sk': sched_sk,
                'gsi2pk': f'ROOM#{s["room_id"]}',
                'gsi2sk': sched_sk,
                'entity_type': 'schedule',
                **_str_ids(s),
            }
        )

    professor_items: list[dict[str, Any]] = []
    for pr in professors:
        professor_items.append(
            {
                'pk': f'PROF#{pr["id"]}',
                'sk': '#META',
                'gsi1pk': f'FACULTY#{pr["faculty_id"]}',
                'gsi1sk': f'PROF#{pr["full_name"]}',
                # GSI_NAME (sparse, professors only): search by name without knowing the faculty — P15.
                # The key is normalized by the same function the query side uses, so the two match.
                'gsi_name_pk': 'PROF',
                'gsi_name_sk': normalize_name_key(pr['full_name']),
                'entity_type': 'professor',
                **_str_ids(pr),
            }
        )

    course_items: list[dict[str, Any]] = []
    for c in courses:
        fid = faculty_id_by_name[c['faculty']]
        course_items.append(
            {
                'pk': f'COURSE#{c["id"]}',
                'sk': '#META',
                'gsi1pk': f'FACULTY#{fid}',
                'gsi1sk': f'COURSE#{c["name"]}',
                'gsi2pk': f'PROF#{c["professor_id"]}',
                'gsi2sk': f'COURSE#{c["id"]}',
                'entity_type': 'course',
                'id': str(c['id']),
                'name': c['name'],
                'faculty_id': str(fid),
                'professor_id': str(c['professor_id']),
            }
        )

    # Rooms and places carry no GSI keys any more. Both used to need a constant partition
    # (`TYPE#ROOM`, `TYPE#PLACE`) to be readable inside the shared table; a table of their own is
    # that partition, so the keys would index nothing that the scan does not already reach.
    room_items = [  # 'type' here is the room type, not to be confused with entity_type
        {'pk': f'ROOM#{r["id"]}', 'sk': '#META', 'entity_type': 'room', **_str_ids(r)} for r in rooms
    ]

    place_items = [
        {
            'pk': f'PLACE#{pl["id"]}',
            'sk': '#META',
            # a denormalized lowercase copy of the name — for a case-insensitive
            # FilterExpression contains() (DynamoDB expressions have no lower()).
            'name_lower': pl['name'].lower(),
            'entity_type': 'place',
            **_str_ids(pl),
        }
        for pl in places
    ]

    return {
        T_FACULTIES: faculty_items,
        T_PROGRAMS: program_items,
        T_GROUPS: group_items,
        T_PROFESSORS: professor_items,
        T_COURSES: course_items,
        T_ROOMS: room_items,
        T_PLACES: place_items,
    }


def _count_faculty_dependants(
    programs: list[dict[str, Any]],
    professors: list[dict[str, Any]],
    courses: list[dict[str, Any]],
    faculty_id_by_name: dict[str, Any],
) -> Counter[str]:
    """How many programmes, professors and courses each seeded faculty owns.

    All three count, not just programmes: the counter stands in for 'anything is still filed under
    this faculty', and a faculty seeded with a professor but no programme must be as undeletable as
    one with both. Programmes and courses name their faculty, professors reference its id — hence
    the two ways of reaching the same key."""
    counts: Counter[str] = Counter()
    counts.update(str(faculty_id_by_name[p['faculty']]) for p in programs)
    counts.update(str(pr['faculty_id']) for pr in professors)
    counts.update(str(faculty_id_by_name[c['faculty']]) for c in courses)
    return counts


def build_slot_items() -> list[dict[str, Any]]:
    """The appointment_slots table. Keys carry no SLOT# prefix (the table holds a single item type).
    gsi2 (STUDENT#email) appears only on a booked slot. expires_at is for the TTL."""
    items = []
    for sl in _load('appointment_slots'):
        item = {
            'pk': sl['date'],
            'sk': f'{sl["start_time"]}#{sl["id"]}',
            'gsi1pk': f'STATUS#{sl["status"]}',
            'gsi1sk': f'{sl["date"]}#{sl["start_time"]}',
            'expires_at': _end_of_day_epoch(sl['date']),
            **_str_ids(sl),
        }
        if sl.get('booked_by'):
            item['gsi2pk'] = f'STUDENT#{sl["booked_by"]}'
            item['gsi2sk'] = f'{sl["date"]}#{sl["start_time"]}'
        items.append(_clean(item))
    return items


def build_issue_items() -> list[dict[str, Any]]:
    """The reported_issues table. pk=category, sk=votes(zero-padded)#id, GSI1 by status."""
    items = []
    for iss in _load('reported_issues'):
        votes_sk = f'{iss["votes"]:04d}#{iss["id"]}'
        items.append(
            {
                'pk': iss['category'],
                'sk': votes_sk,
                'gsi1pk': f'STATUS#{iss["status"]}',
                'gsi1sk': votes_sk,
                **_str_ids(iss),
            }
        )
    return items


# --- working with the tables --------------------------------------------------------------------
def _client():
    return boto3.client(
        'dynamodb',
        endpoint_url=ENDPOINT_URL,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def _resource():
    return boto3.resource(
        'dynamodb',
        endpoint_url=ENDPOINT_URL,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def table_definition(
    name: str,
    gsi_numbers: list[int],
    extra_gsis: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """The ``create_table`` arguments for a table keyed on pk/sk, with its GSIs (projection ALL).

    ``gsi_numbers`` — the overloaded indexes, with generic `gsi{n}pk`/`gsi{n}sk` keys (GSI1, GSI2).
    ``extra_gsis`` — the named special-purpose indexes: a list of `(index_name, pk_attr, sk_attr)`
    tuples (e.g. GSI_NAME, for professor search by name).

    Separate from ``recreate_table`` so the tests (``tests/dynamodb/conftest.py``) use this same
    definition — with their own client and their own table names, and without a copy of the schema."""
    attrs = [{'AttributeName': 'pk', 'AttributeType': 'S'}, {'AttributeName': 'sk', 'AttributeType': 'S'}]
    gsis = []
    for n in gsi_numbers:
        attrs += [
            {'AttributeName': f'gsi{n}pk', 'AttributeType': 'S'},
            {'AttributeName': f'gsi{n}sk', 'AttributeType': 'S'},
        ]
        gsis.append(
            {
                'IndexName': f'GSI{n}',
                'KeySchema': [
                    {'AttributeName': f'gsi{n}pk', 'KeyType': 'HASH'},
                    {'AttributeName': f'gsi{n}sk', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        )
    for index_name, pk_attr, sk_attr in extra_gsis or []:
        attrs += [{'AttributeName': pk_attr, 'AttributeType': 'S'}, {'AttributeName': sk_attr, 'AttributeType': 'S'}]
        gsis.append(
            {
                'IndexName': index_name,
                'KeySchema': [
                    {'AttributeName': pk_attr, 'KeyType': 'HASH'},
                    {'AttributeName': sk_attr, 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        )

    definition: dict[str, Any] = {
        'TableName': name,
        'BillingMode': 'PAY_PER_REQUEST',
        'AttributeDefinitions': attrs,
        'KeySchema': [{'AttributeName': 'pk', 'KeyType': 'HASH'}, {'AttributeName': 'sk', 'KeyType': 'RANGE'}],
    }
    if gsis:  # DynamoDB rejects an empty GlobalSecondaryIndexes
        definition['GlobalSecondaryIndexes'] = gsis
    return definition


def recreate_table(
    name: str,
    gsi_numbers: list[int],
    extra_gsis: list[tuple[str, str, str]] | None = None,
) -> None:
    """(Re)creates the table from ``table_definition``: drops the old one, if any, and waits.

    Only safe for a table this script refills on the same run. For one holding data it cannot
    regenerate, use ``create_table_if_absent``.
    """
    client = _client()
    if name in client.list_tables().get('TableNames', []):
        client.delete_table(TableName=name)
        client.get_waiter('table_not_exists').wait(TableName=name)

    _create_table(client, name, gsi_numbers, extra_gsis)


def create_table_if_absent(
    name: str,
    gsi_numbers: list[int],
    extra_gsis: list[tuple[str, str, str]] | None = None,
) -> None:
    """Create the table only when it is missing, leaving an existing one and its rows alone.

    The counterpart to ``recreate_table``, and the right one for any table this script does not
    seed. ``agent_checkpoints`` is the case that forced it: its rows are people's conversations,
    which nothing can regenerate — re-seeding faculties after the assistant has been in use must
    not cost them their history.

    Note what this does *not* do: an existing table is accepted as-is, without checking that its
    key schema still matches ``table_definition``. Changing the layout of a table holding live data
    is a migration, and a seed loader is the wrong place to attempt one.
    """
    client = _client()
    if name in client.list_tables().get('TableNames', []):
        logger.info('· table %s already exists — left untouched (it holds data this script cannot rebuild)', name)
        return

    _create_table(client, name, gsi_numbers, extra_gsis)


def _create_table(
    client: Any,
    name: str,
    gsi_numbers: list[int],
    extra_gsis: list[tuple[str, str, str]] | None = None,
) -> None:
    """Create the table and wait for it to come up. Shared by both callers above, which differ only
    in what they do about a table that is already there."""
    client.create_table(**table_definition(name, gsi_numbers, extra_gsis))
    client.get_waiter('table_exists').wait(TableName=name)
    named = [g[0] for g in (extra_gsis or [])]
    logger.info('· table %s created (GSI: %s)', name, (gsi_numbers or []) + named or 'none')


def enable_ttl(name: str, attribute: str) -> None:
    _client().update_time_to_live(
        TableName=name,
        TimeToLiveSpecification={'Enabled': True, 'AttributeName': attribute},
    )
    logger.info('· TTL enabled on %s.%s', name, attribute)


def load(name: str, items: list[dict[str, Any]]) -> None:
    table = _resource().Table(name)
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
    logger.info('· %s: %d items loaded', name, len(items))


# --- access-pattern demo ------------------------------------------------------------------------
def _print_rows(title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    body = '\n'.join('   ' + '  '.join(f'{f}={r.get(f)}' for f in fields) for r in rows)
    logger.info('\n### %s  (%d rows)\n%s', title, len(rows), body)


def demo() -> None:
    groups = _resource().Table(T_GROUPS)
    programs = _resource().Table(T_PROGRAMS)
    professors = _resource().Table(T_PROFESSORS)
    slots = _resource().Table(T_SLOTS)
    issues = _resource().Table(T_ISSUES)

    rule = '=' * 70
    logger.info('\n%s\nDEMO: same data, different questions — each answered by a SINGLE query\n%s', rule, rule)

    # P1 — schedule of group 1  [academic_groups, base: pk=GROUP#1]. The colocation the split kept:
    # the group's own row and its classes share a partition, so both come back in one query.
    r = groups.query(KeyConditionExpression=Key('pk').eq('GROUP#1') & Key('sk').begins_with('SCHED#'))
    _print_rows(
        'P1 · Schedule of group 1  [academic_groups base]',
        r['Items'],
        ['weekday', 'start_time', 'end_time', 'course_id', 'room_id'],
    )

    # P2 — schedule of professor 1  [academic_groups, GSI1: gsi1pk=PROF#1]. The one GSI still
    # overloaded on purpose: PROGRAM# for a group row, PROF# for a class row.
    r = groups.query(
        IndexName='GSI1', KeyConditionExpression=Key('gsi1pk').eq('PROF#1') & Key('gsi1sk').begins_with('SCHED#')
    )
    _print_rows(
        'P2 · Schedule of professor 1  [academic_groups GSI1]',
        r['Items'],
        ['weekday', 'start_time', 'group_id', 'room_id'],
    )

    # P4 — programs of faculty 1  [programs, GSI1: gsi1pk=FACULTY#1]
    r = programs.query(
        IndexName='GSI1', KeyConditionExpression=Key('gsi1pk').eq('FACULTY#1') & Key('gsi1sk').begins_with('PROGRAM#')
    )
    _print_rows('P4 · Programs of faculty 1  [programs GSI1]', r['Items'], ['name', 'degree', 'tuition_usd'])

    # P5 — professors of faculty 1  [professors, GSI1: gsi1pk=FACULTY#1]. Worth showing now that it
    # is a query against a table of its own: before the split it shared GSI1's FACULTY# partition
    # with P4 and P6, and the begins_with was what kept the three apart.
    r = professors.query(
        IndexName='GSI1', KeyConditionExpression=Key('gsi1pk').eq('FACULTY#1') & Key('gsi1sk').begins_with('PROF#')
    )
    _print_rows('P5 · Professors of faculty 1  [professors GSI1]', r['Items'], ['full_name', 'title', 'email'])

    # P15 — professor search by name  [professors, GSI_NAME: gsi_name_pk=PROF, begins_with]
    r = professors.query(
        IndexName='GSI_NAME',
        KeyConditionExpression=Key('gsi_name_pk').eq('PROF') & Key('gsi_name_sk').begins_with('alan'),
    )
    _print_rows(
        "P15 · Professor search by name 'alan'  [professors GSI_NAME]",
        r['Items'],
        ['full_name', 'title', 'email', 'office_hours'],
    )

    # P9 — open slots on a date  [appointment_slots, base: pk=date, filter status=open]
    r = slots.query(KeyConditionExpression=Key('pk').eq('2026-10-06'), FilterExpression=Attr('status').eq('open'))
    _print_rows(
        'P9 · Open slots on 2026-10-06  [appointment_slots base]', r['Items'], ['start_time', 'topic', 'status']
    )

    # P11 — issues in the facilities category, top by votes  [reported_issues, base + reverse]
    r = issues.query(KeyConditionExpression=Key('pk').eq('facilities'), ScanIndexForward=False)
    _print_rows(
        'P11 · Issues in «facilities», top by votes  [reported_issues base]', r['Items'], ['votes', 'status', 'text']
    )


def main() -> None:
    # The script is run by hand, and all of its output is logger.info: with no handler, logging
    # shows WARNING and above only, so we configure it here. The format is bare ('%(message)s'), so
    # the console reads like ordinary command output rather than an application log.
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info('DynamoDB endpoint: %s', ENDPOINT_URL)

    for name, (gsi_numbers, extra_gsis) in TABLE_INDEXES.items():
        recreate_table(name, gsi_numbers=gsi_numbers, extra_gsis=extra_gsis)
    # The agent's conversation memory. No GSI: every read is scoped to a thread, and the thread is
    # the partition. Created rather than recreated — the tables above are wiped and refilled below,
    # but this one is never seeded, and its rows are conversations a re-seed must not cost.
    create_table_if_absent(T_CHECKPOINTS, gsi_numbers=[])
    enable_ttl(T_SLOTS, 'expires_at')

    for name, items in build_items().items():
        load(name, items)
    load(T_SLOTS, build_slot_items())
    load(T_ISSUES, build_issue_items())

    demo()

    hints = '\n'.join(
        f'  aws dynamodb scan --table-name {t} --endpoint-url {ENDPOINT_URL}' for t in (*TABLE_INDEXES, T_CHECKPOINTS)
    )
    logger.info('\nDone. To inspect the tables in full (AWS CLI):\n%s', hints)


if __name__ == '__main__':
    main()
