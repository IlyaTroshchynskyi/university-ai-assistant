"""Global Secondary Index names, in one place, so callers reference an enum instead of loose
string literals (see the key design in ``db/02-dynamodb-model.md``).

``GSI1``/``GSI2`` are the overloaded indexes reused across all three tables (their *meaning* depends
on the table); ``GSI_NAME`` is the ``university``-only sparse index for professor name search.

``normalize_name_key`` lives here too, next to the ``GSI_NAME_SK`` it produces: a sort key only
matches a query when both sides derive it the same way, so the writer (``db/load_dynamodb.py``) and
the reader (``UniversityRepository.find_professors_by_name``) have to share one implementation."""

from enum import StrEnum


class Index(StrEnum):
    """A GSI name. ``str`` subclass, so a member can be passed straight where a string is expected
    (e.g. ``DynamoDBService.query(index_name=...)``)."""

    # university: Program/Professor/Course (by FACULTY#), Group (by PROGRAM#), Place (by TYPE#PLACE),
    # Schedule (by PROF#) · appointment_slots: Slot (by STATUS#) · reported_issues: Issue (by STATUS#)
    GSI1 = 'GSI1'
    # university: Course (by PROF#), Schedule (by ROOM#) · appointment_slots: Slot (by STUDENT#, sparse)
    GSI2 = 'GSI2'
    # university only: Professor — name search (sparse, PROF partition)
    GSI_NAME = 'GSI_NAME'


class KeyAttr(StrEnum):
    """Key/index **attribute** names shared across the tables — the generic ``pk``/``sk`` plus each
    GSI's partition/sort keys. Pass a member straight into ``Key(...)`` instead of a literal."""

    # base partition: Faculty/Program/Group/Professor/Course/Room/Place=<ENTITY>#{id},
    # Schedule=GROUP#{group_id}, Slot={date}, Issue={category}
    PK = 'pk'
    # base sort: entities=#META, Schedule=SCHED#{wd}#{start}, Slot={start}#{id}, Issue={votes}#{id}
    SK = 'sk'
    # GSI1 partition — Program/Professor/Course=FACULTY#{faculty_id}, Group=PROGRAM#{program_id}, Place=TYPE#PLACE,
    # Schedule=PROF#{professor_id}, Slot/Issue=STATUS#{status}
    GSI1_PK = 'gsi1pk'
    # GSI1 sort — Program=PROGRAM#{name}, Group=GROUP#{id}, Professor=PROF#{full_name}, Course=COURSE#{name},
    # Place=PLACE#{name}, Schedule=SCHED#{wd}#{start}, Slot={date}#{start}, Issue={votes}#{id}
    GSI1_SK = 'gsi1sk'
    # GSI2 partition — Course=PROF#{professor_id}, Schedule=ROOM#{room_id}, Slot=STUDENT#{booked_by}
    GSI2_PK = 'gsi2pk'
    # GSI2 sort — Course=COURSE#{id}, Schedule=SCHED#{wd}#{start}, Slot={date}#{start}
    GSI2_SK = 'gsi2sk'
    # GSI_NAME partition — Professor=PROF (constant)
    GSI_NAME_PK = 'gsi_name_pk'
    # GSI_NAME sort — Professor=normalized full_name (lowercased, title-stripped)
    GSI_NAME_SK = 'gsi_name_sk'


def normalize_name(name: str) -> str:
    """A name as a uniqueness key stores it: lowercased, with runs of whitespace collapsed to one
    space ('Computer  SCIENCE ' -> 'computer science').

    Lives here for the same reason as ``normalize_name_key`` below: a key only matches when both
    sides derive it the same way, so the writer and the reader have to share one implementation —
    and so do the entities that reserve a name (faculties, programmes), which is why this is not
    per-entity."""
    return ' '.join(name.lower().split())


# Leading words dropped from a name before it becomes a key. A sort key supports ``begins_with`` but
# not ``contains``, so a stored 'Dr. Alan Whitfield' would never be found by 'Alan' — the honorific
# has to go from the key itself. Query-side titles ('Professor Alan') are stripped for the same
# reason, and repeatedly, so that 'Associate Professor Alan' normalizes down to 'alan' as well.
_HONORIFICS = frozenset({'dr', 'prof', 'professor', 'associate', 'mr', 'mrs', 'ms', 'miss'})


def normalize_name_key(full_name: str) -> str:
    """A person's name as ``GSI_NAME_SK`` stores it: lowercased, with any leading honorific or title
    removed ('Dr. Alan Whitfield' -> 'alan whitfield').

    Used on both sides of the index — by the loader when writing the key, and by the repository when
    building the ``begins_with`` prefix — because a query normalized differently from the stored
    value simply matches nothing. Returns '' for a blank or title-only name, which callers treat as
    'nothing to search for' rather than passing an empty prefix to DynamoDB (it rejects one)."""
    parts = full_name.strip().lower().split()
    while parts and parts[0].rstrip('.') in _HONORIFICS:
        parts = parts[1:]
    return ' '.join(parts)
