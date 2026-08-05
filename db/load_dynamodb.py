"""Seed loader for the local DynamoDB (the model from 02-dynamodb-model.md).

Three tables — not 'one for everything', but 'a few':
  · university        — the academic core (single-table): faculty, program, group,
                        professor, course, room, place, schedule;
  · appointment_slots — independent, dated, with a TTL on expires_at;
  · reported_issues   — independent (student reports).

What the script does:
  1. (re)creates all three tables with their GSIs (+ enables TTL on appointment_slots);
  2. reads db/seed/*.json and turns the records into items following the key maps;
  3. loads everything in batches;
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
from app.api.v1.faculty.schemas import FacultyNameItem, normalize_faculty_name
from app.core.dynamodb.indexes import normalize_name_key

logger = logging.getLogger(__name__)

# --- configuration (defaults match app/settings.py) ---------------------------------------------
ENDPOINT_URL = os.getenv('DYNAMODB_ENDPOINT_URL', 'http://localhost:8001')
REGION = os.getenv('AWS_REGION', 'us-east-1')
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'dummy')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy')

T_UNIVERSITY = os.getenv('DYNAMODB_TABLE', 'university')
T_SLOTS = 'appointment_slots'
T_ISSUES = 'reported_issues'

SEED_DIR = Path(__file__).parent / 'seed'

# Monday=1 … Sunday=7 — so that weekdays sort correctly in the SK (otherwise 'Fri' < 'Mon').
WEEKDAY_NUM = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((SEED_DIR / f'{name}.json').read_text(encoding='utf-8'))


def _clean(item: dict[str, Any]) -> dict[str, Any]:
    """DynamoDB doesn't store None — drop the empty attributes (e.g. topic/booked_by on an open slot)."""
    return {k: v for k, v in item.items() if v is not None}


# The identifier fields: an entity's own id and every reference to another one. programs.program_id
# is a slug, already a string, so it passes through; groups.program_id is a number and becomes one.
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
    seen = Counter(normalize_faculty_name(f['name']) for f in faculties)
    repeated = [name for name, count in seen.items() if count > 1]
    if repeated:
        raise ValueError(f'faculties.json: duplicate faculty names: {", ".join(sorted(repeated))}')


# --- seed -> items transformation (following the key maps in 02-dynamodb-model.md) --------------
def build_university_items() -> list[dict[str, Any]]:
    faculties = _load('faculties')
    programs = _load('programs')
    groups = _load('groups')
    professors = _load('professors')
    courses = _load('courses')
    rooms = _load('rooms')
    places = _load('places')
    schedule = _load('schedule')

    # The seed stores a faculty by NAME, and groups.program_id as a number; we resolve them to a
    # faculty_id and a program slug, as settled in the document.
    _reject_duplicate_faculty_names(faculties)
    faculty_id_by_name = {f['name']: f['id'] for f in faculties}
    program_slug_by_id = {p['id']: p['program_id'] for p in programs}

    items: list[dict[str, Any]] = []

    for f in faculties:
        items.append({'pk': f'FACULTY#{f["id"]}', 'sk': '#META', 'entity_type': 'faculty', **_str_ids(f)})
        # The row reserving the name, written exactly as POST /faculties writes it. A seeded faculty
        # without one leaves its name free for the API to hand out a second time.
        items.append(FacultyNameItem(name=f['name'], faculty_id=str(f['id'])).model_dump())

    for p in programs:
        fid = faculty_id_by_name[p['faculty']]
        items.append(
            {
                'pk': f'PROGRAM#{p["program_id"]}',
                'sk': '#META',
                'gsi1pk': f'FACULTY#{fid}',
                'gsi1sk': f'PROGRAM#{p["name"]}',
                'entity_type': 'program',
                'id': str(p['id']),
                'program_id': p['program_id'],
                'name': p['name'],
                'faculty_id': str(fid),
                'degree': p['degree'],
                'duration_years': p['duration_years'],
                'tuition_usd': p['tuition_usd'],
            }
        )

    for g in groups:
        slug = program_slug_by_id[g['program_id']]
        items.append(
            {
                'pk': f'GROUP#{g["id"]}',
                'sk': '#META',
                'gsi1pk': f'PROGRAM#{slug}',
                'gsi1sk': f'GROUP#{g["id"]}',
                'entity_type': 'group',
                'id': str(g['id']),
                'name': g['name'],
                'program_id': str(g['program_id']),
                'program_slug': slug,
            }
        )

    for pr in professors:
        items.append(
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

    for c in courses:
        fid = faculty_id_by_name[c['faculty']]
        items.append(
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

    for r in rooms:  # 'type' here is the room type, not to be confused with entity_type
        items.append(
            {
                'pk': f'ROOM#{r["id"]}',
                'sk': '#META',
                # as with places: a constant GSI1 partition, so the whole room list takes a single
                # query. Ordered by building + zero-padded door number (a sort key is a string).
                'gsi1pk': 'TYPE#ROOM',
                'gsi1sk': f'{r["building"]}#{r["number"]:04d}',
                'entity_type': 'room',
                **_str_ids(r),
            }
        )

    for pl in places:
        items.append(
            {
                'pk': f'PLACE#{pl["id"]}',
                'sk': '#META',
                'gsi1pk': 'TYPE#PLACE',
                'gsi1sk': f'PLACE#{pl["name"]}',
                # a denormalized lowercase copy of the name — for a case-insensitive
                # FilterExpression contains() (DynamoDB expressions have no lower()).
                'name_lower': pl['name'].lower(),
                'entity_type': 'place',
                **_str_ids(pl),
            }
        )

    for s in schedule:  # one item serves P1 (group) / P2 (professor) / P3 (room)
        wd = WEEKDAY_NUM[s['weekday']]
        sched_sk = f'SCHED#{wd}#{s["start_time"]}'
        items.append(
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

    return items


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
    """(Re)creates the table from ``table_definition``: drops the old one, if any, and waits."""
    client = _client()
    if name in client.list_tables().get('TableNames', []):
        client.delete_table(TableName=name)
        client.get_waiter('table_not_exists').wait(TableName=name)

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
    uni = _resource().Table(T_UNIVERSITY)
    slots = _resource().Table(T_SLOTS)
    issues = _resource().Table(T_ISSUES)

    rule = '=' * 70
    logger.info('\n%s\nDEMO: same data, different questions — each answered by a SINGLE query\n%s', rule, rule)

    # P1 — schedule of group 1  [university, base: pk=GROUP#1]
    r = uni.query(KeyConditionExpression=Key('pk').eq('GROUP#1') & Key('sk').begins_with('SCHED#'))
    _print_rows(
        'P1 · Schedule of group 1  [university base]',
        r['Items'],
        ['weekday', 'start_time', 'end_time', 'course_id', 'room_id'],
    )

    # P2 — schedule of professor 1  [university, GSI1: gsi1pk=PROF#1]
    r = uni.query(
        IndexName='GSI1', KeyConditionExpression=Key('gsi1pk').eq('PROF#1') & Key('gsi1sk').begins_with('SCHED#')
    )
    _print_rows(
        'P2 · Schedule of professor 1  [university GSI1]',
        r['Items'],
        ['weekday', 'start_time', 'group_id', 'room_id'],
    )

    # P4 — programs of faculty 1  [university, GSI1: gsi1pk=FACULTY#1]
    r = uni.query(
        IndexName='GSI1', KeyConditionExpression=Key('gsi1pk').eq('FACULTY#1') & Key('gsi1sk').begins_with('PROGRAM#')
    )
    _print_rows('P4 · Programs of faculty 1  [university GSI1]', r['Items'], ['name', 'degree', 'tuition_usd'])

    # P15 — professor search by name  [university, GSI_NAME: gsi_name_pk=PROF, begins_with]
    r = uni.query(
        IndexName='GSI_NAME',
        KeyConditionExpression=Key('gsi_name_pk').eq('PROF') & Key('gsi_name_sk').begins_with('alan'),
    )
    _print_rows(
        "P15 · Professor search by name 'alan'  [university GSI_NAME]",
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

    recreate_table(T_UNIVERSITY, gsi_numbers=[1, 2], extra_gsis=[('GSI_NAME', 'gsi_name_pk', 'gsi_name_sk')])
    recreate_table(T_SLOTS, gsi_numbers=[1, 2])
    recreate_table(T_ISSUES, gsi_numbers=[1])
    enable_ttl(T_SLOTS, 'expires_at')

    load(T_UNIVERSITY, build_university_items())
    load(T_SLOTS, build_slot_items())
    load(T_ISSUES, build_issue_items())

    demo()

    hints = '\n'.join(
        f'  aws dynamodb scan --table-name {t} --endpoint-url {ENDPOINT_URL}' for t in (T_UNIVERSITY, T_SLOTS, T_ISSUES)
    )
    logger.info('\nDone. To inspect the tables in full (AWS CLI):\n%s', hints)


if __name__ == '__main__':
    main()
