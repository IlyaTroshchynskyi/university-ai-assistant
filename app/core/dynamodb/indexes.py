"""Global Secondary Index names, in one place, so callers reference an enum instead of loose
string literals (see the key design in ``db/02-dynamodb-model.md``, and the table split in
``db/03-table-split.md``).

``GSI1``/``GSI2`` are generic key *names* reused by several tables rather than one overloaded index:
each table's GSI1 now means a single thing, because each table holds a single entity. The one
exception is ``academic_groups``, whose GSI1 is genuinely overloaded (``PROGRAM#`` for a group,
``PROF#`` for one of its classes) — the two item types share a partition there on purpose.
``GSI_NAME`` is the ``professors``-only sparse index for name search.

``normalize_name_key`` lives here too, next to the ``GSI_NAME_SK`` it produces: a sort key only
matches a query when both sides derive it the same way, so the writer (``db/load_dynamodb.py``) and
the reader (``ProfessorsRepository.find_professors_by_name``) have to share one implementation."""

from enum import StrEnum


class Index(StrEnum):
    """A GSI name. ``str`` subclass, so a member can be passed straight where a string is expected
    (e.g. ``DynamoDBService.query(index_name=...)``)."""

    # programs/professors/courses: by FACULTY# · academic_groups: group by PROGRAM#, class by PROF#
    # appointment_slots: Slot by STATUS# · reported_issues: Issue by STATUS#
    GSI1 = 'GSI1'
    # courses: by PROF# · academic_groups: class by ROOM# · appointment_slots: Slot by STUDENT# (sparse)
    GSI2 = 'GSI2'
    # professors only: name search (sparse, constant PROF partition)
    GSI_NAME = 'GSI_NAME'


class KeyAttr(StrEnum):
    """Key/index **attribute** names shared across the tables — the generic ``pk``/``sk`` plus each
    GSI's partition/sort keys. Pass a member straight into ``Key(...)`` instead of a literal."""

    # base partition — entity tables: <ENTITY>#{id} (a class shares its group's GROUP#{group_id});
    # appointment_slots: {date}; reported_issues: {category}
    PK = 'pk'
    # base sort — an entity's own row: #META; a name reservation: #UNIQUE; a class: SCHED#{wd}#{start};
    # appointment_slots: {start}#{id}; reported_issues: {votes}#{id}
    SK = 'sk'
    # GSI1 partition — programs/professors/courses: FACULTY#{faculty_id}; academic_groups:
    # PROGRAM#{program_id} (group) / PROF#{professor_id} (class); slots/issues: STATUS#{status}
    GSI1_PK = 'gsi1pk'
    # GSI1 sort — programs: PROGRAM#{name}; professors: PROF#{full_name}; courses: COURSE#{name};
    # academic_groups: GROUP#{id} (group) / SCHED#{wd}#{start} (class); slots: {date}#{start};
    # issues: {votes}#{id}
    GSI1_SK = 'gsi1sk'
    # GSI2 partition — courses: PROF#{professor_id}; academic_groups: ROOM#{room_id} (class);
    # slots: STUDENT#{booked_by}
    GSI2_PK = 'gsi2pk'
    # GSI2 sort — courses: COURSE#{id}; academic_groups: SCHED#{wd}#{start}; slots: {date}#{start}
    GSI2_SK = 'gsi2sk'
    # GSI_NAME partition — professors: PROF (constant)
    GSI_NAME_PK = 'gsi_name_pk'
    # GSI_NAME sort — professors: normalized full_name (lowercased, title-stripped)
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
