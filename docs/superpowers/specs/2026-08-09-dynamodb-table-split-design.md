# Splitting the `university` table, and keeping locality where it is paid for

**Date:** 2026-08-09
**Status:** Implemented 2026-08-12. The result is documented in `db/03-table-split.md`.

Two things the implementation decided that this design left open:

- **`ProgramsRepository.has_dependants` was not covered here.** It guards `DELETE /programs/{id}` by
  querying GSI1 for `PROGRAM#{id}` — a partition that ceases to exist once groups move to
  `academic_groups`, so the guard would have died silently. Programmes carry a `dependants` counter
  of their own, maintained the same way, rather than reading across tables.
- **`dependants` is not a field on `FacultyItem`/`ProgramItem`.** It is maintained by `ADD` and read
  only by a condition, so keeping it off the schema is what holds this refactor to its promise of
  changing no API contract — the attribute would otherwise appear in every response body.

## Overview

`university` is a single table holding seven entity types (faculty, program, group, professor,
course, room, place) plus the schedule rows and the name-reservation rows. This splits it into
seven tables — one per entity — with a single deliberate exception: **group and schedule stay
together**, because that is the one place in the model where a query has to return two item types
at once.

The refactor changes no API contract. The only externally visible change is *how* `DELETE
/faculties/{id}` decides to answer 409, which moves from a read to a write condition and stops
being a race.

`appointment_slots` and `reported_issues` are already separate and are not touched.

## Motivation

Single-table is a means, not an end — `db/02-dynamodb-model.md:19` says so, and the criterion it
states ("together if one query reads them; apart if the entity has its own lifecycle") is not
currently met by six of the seven entities in the table.

What the shared table costs today, in code that exists:

| Cost | Where |
|------|-------|
| `list_faculties` scans the whole table (89 seeded items) to return ~6 rows | `app/api/v1/faculty/repository.py:49` |
| `list_programs` does the same | `app/api/v1/programs/repository.py:50` |
| Listing rooms needs an invented constant GSI1 partition (`TYPE#ROOM`) because every row has a distinct `pk` | `app/core/dynamodb/base_items.py:60` |
| Every row carries `entity_type` purely so listings can filter the other entities out | `db/load_dynamodb.py`, both scans above |
| `FacultyNameItem` must deliberately omit `entity_type` so the faculty scan steps over it | `app/api/v1/faculty/schemas.py:26` |
| GSI1's meaning depends on the item type, so `indexes.py` needs a legend to be readable | `app/core/dynamodb/indexes.py:36-45` |

What it buys today: nothing. The one co-read pattern that justifies a shared table — P1, "a group
and its schedule in one query" — has no code. There is no repository touching `GROUP#`, `SCHED#` or
`COURSE#`; the `university` table's GSI2 has no consumer at all (only `SlotsRepository` uses GSI2,
in a different table).

Splitting is also the cheaper direction to be wrong in. Moving rows out of a shared table preserves
their keys verbatim (`FACULTY#{id}` / `#META` is a valid key in a dedicated `faculties` table).
Merging separate tables later would mean rewriting keys, adding a discriminator, dual-writing and
backfilling indexes. Starting split forfeits only the co-read optimisation — which is needed
exactly when the access pattern is known, i.e. when the uncertainty that motivates splitting has
already resolved.

## The criterion

An entity is colocated with its parent only when all three hold:

1. **No entry point of its own.** It is never read by an id that identifies it alone.
2. **Bounded volume.** The number of children per parent has a known ceiling.
3. **Shared lifecycle.** It is deleted or archived with the parent.

Applied to this model:

| Pair | Entry point | Bounded | Lifecycle | Verdict |
|------|-------------|---------|-----------|---------|
| group → schedule rows | no — a class is only ever read via its group, professor or room | yes — a weekly timetable | yes — dies with the group | **together** |
| faculty → programs | no — `PROGRAM#{id}` | no | no | apart |
| faculty → professors | no — `PROF#{id}` | no | no | apart |
| faculty → courses | no — `COURSE#{id}` | no | no | apart |
| program → groups | no — `GROUP#{id}` | no | no | apart |
| entity → its name reservation | yes, none of its own | yes — exactly one | yes — deleted with the entity | **together** |

The name-reservation rows passing the same three tests is not a coincidence to gloss over: it is
the criterion validating itself on a row type that was never designed against it. `FACULTY_NAME#`
lives in `faculties`, `PROGRAM_NAME#` lives in `programs`, and `create_faculty` stays a
single-table transaction as a result.

## Table map

```
faculties          FACULTY#{id}/#META           entity row (carries `dependants`)
                   FACULTY_NAME#{name}/#UNIQUE  uniqueness reservation
                   no GSI                       listing: Scan, filtered sk = #META

programs           PROGRAM#{id}/#META
                   PROGRAM_NAME#{fid}#{name}/#UNIQUE
                   GSI1  FACULTY#{faculty_id} / PROGRAM#{name}              P4
                   listing: Scan, filtered sk = #META

professors         PROF#{id}/#META
                   GSI1      FACULTY#{faculty_id} / PROF#{full_name}        P5
                   GSI_NAME  PROF / {normalize_name_key(full_name)}         P15

courses            COURSE#{id}/#META
                   GSI1  FACULTY#{faculty_id} / COURSE#{name}               P6
                   GSI2  PROF#{professor_id} / COURSE#{id}                  P8

rooms              ROOM#{id}/#META              no GSI, listing: Scan
places             PLACE#{id}/#META             no GSI, search: Scan + contains   P14

academic_groups    GROUP#{id}/#META                    the group
                   GROUP#{id}/SCHED#{wd}#{start}       its classes
                   GSI1  PROGRAM#{program_id} / GROUP#{id}        (group)    P7
                         PROF#{professor_id} / SCHED#{wd}#{start} (class)    P2
                   GSI2  ROOM#{room_id} / SCHED#{wd}#{start}      (class)    P3

appointment_slots  unchanged
reported_issues    unchanged
```

Note that `academic_groups` **keeps an overloaded GSI1** — `PROGRAM#` for a group row, `PROF#` for
a class row. This is the single-table technique being removed everywhere else, retained here
deliberately: in this table the two item types genuinely share a partition and are read together,
so the overloading is paid for rather than inherited.

Key values are unchanged everywhere. No item's `pk`, `sk` or GSI keys are rewritten by this
refactor; rows only move to a different table.

## Cross-table transactions

`DynamoDBService.transact_write` hardcodes one table (`app/core/dynamodb/base_service.py:120`):

```python
entry: dict[str, Any] = {'TableName': self._table_name}
```

The dependant counter needs `ProgramsRepository` to write the faculty row and the program row in
one transaction, i.e. into two tables. `TransactWriteItems` supports this natively (same region,
up to 100 actions); only our wrapper does not.

**Chosen approach: the action names its table.** Each action dataclass in
`app/core/dynamodb/schemas.py` gains `table: str | None = None`, and `transact_write` resolves
`action.table or self._table_name`. Existing call sites are untouched, because omitting the field
keeps today's behaviour.

Rejected alternatives:

- *A standalone `TransactionRunner`* — conceptually cleaner (a transaction is not an operation
  *of* a table), but introduces a new object and rewrites every existing call site to serve two
  places that need it.
- *Repository composition* (`ProgramsRepository` holding a `FacultyRepository` and asking it for a
  ready-made action) — hides table names but couples repositories, and still needs the chosen
  mechanism underneath.

A fourth action type is added at the same time:

```python
@dataclass(frozen=True, slots=True)
class TransactUpdate:
    """An 'apply this update expression' step of a transaction — how a counter on another item is
    maintained atomically with the write that changes it."""

    key: dict[str, Any]
    update_expression: str
    expression_values: dict[str, Any]
    expression_names: dict[str, str] | None = None
    condition: ConditionBase | None = None
    table: str | None = None
```

`_TRANSACT_MEMBER` gains `TransactUpdate: 'Update'`. `transact_write` grows one branch for the
update-specific members (`UpdateExpression`, and merging `expression_values` into the same
placeholder namespace the condition already uses).

## Referential integrity: the `dependants` counter

### The problem being solved

`FacultyRepository.has_dependants` (`app/api/v1/faculty/repository.py:58`) is one GSI1 query on the
`FACULTY#{id}` partition, which today holds programs, professors and courses at once. After the
split that partition no longer exists — the three types are in three tables.

Replacing it with three parallel queries would work, but would keep the flaw
`db/02-dynamodb-model.md:107` already records: the check is a **read**, so a program created between
the check and the delete is orphaned. Making the read three times wider does not narrow that window.

The fix is to move the invariant into a write condition, which DynamoDB evaluates atomically as
part of the operation itself.

### The invariant

`faculties` rows carry `dependants: int` = the number of programs, professors and courses filed
under that faculty. It is maintained by whatever creates or deletes one of those children, inside
the same transaction as the child's own write.

Today only `ProgramsRepository` can create a child, so it is the only repository this spec teaches
to maintain the counter. Professors and courses reach the counter through the seeder (which writes
the initial value) and will maintain it themselves when they get CRUD — that obligation belongs to
their `crud-roadmap.md` rungs and is recorded here so it is not rediscovered later.

### Create

`create_program` keeps exactly three actions. The `ConditionCheck` that asserts the faculty exists
is **replaced** by an `Update` that both increments and asserts:

```python
TransactUpdate(
    FacultyItem.key(creation.faculty_id),
    'ADD dependants :one',
    {':one': 1},
    condition=Attr(KeyAttr.PK).exists(),
    table=self._faculties_table,
)
TransactPut(item, condition=Attr(KeyAttr.PK).not_exists())
TransactPut(ProgramNameItem(...), condition=Attr(KeyAttr.PK).not_exists())
```

`ADD` treats a missing attribute as zero, so no initialisation step is needed. Action positions are
unchanged, so `_FACULTY_EXISTS = 0` / `_NAME_RESERVATION = 2` and the `ProgramCreateOutcome`
mapping keep working as they do today.

### Delete child

`delete_program` gains a third action:

```python
TransactDelete(ProgramItem.key(program_id), condition=Attr(KeyAttr.PK).exists())
TransactDelete(ProgramNameItem.key(program.faculty_id, program.name))
TransactUpdate(FacultyItem.key(program.faculty_id), 'ADD dependants :minus', {':minus': -1},
               table=self._faculties_table)
```

### Delete parent

`has_dependants` is deleted. The guard becomes a condition on the delete itself:

```python
TransactDelete(
    FacultyItem.key(faculty_id),
    condition=Attr(KeyAttr.PK).exists()
              & (Attr('dependants').not_exists() | Attr('dependants').eq(0)),
)
TransactDelete(FacultyNameItem.key(faculty.name))
```

`Attr('dependants').not_exists()` is required, not defensive noise: rows written before this change
(and any written by a seed that predates it) have no counter, and must stay deletable.

### Telling 404 from 409

Position 0 of `CancellationReasons` now fails for two different reasons — the faculty does not
exist, or it still has dependants. They are distinguished by a `get_item` issued **after** the
cancellation. This read is not a race: the transaction has already been rejected, so nothing was
deleted, and the answer only decides which status code to return.

`delete_faculty` currently reads the faculty first (to learn its name for the reservation key) and
returns `False` when absent. That read stays — it is needed for the name — so the flow is:

1. `get_item` → `None` → **404** (as today).
2. Transaction → cancelled at position 0 → **409** (dependants).
3. Transaction → cancelled elsewhere → re-raise.

The service layer's existing 404/409 mapping is unchanged.

### Known weaknesses

- **Counter drift.** A lost decrement (a bug, a row deleted through the AWS console, a partial
  seed) leaves a faculty permanently undeletable. Mitigated by a repair path, not prevented.
- **Contention.** Every child create writes the same faculty row. Irrelevant at this scale; a
  documented ceiling rather than a problem.
- **The seeder bypasses repositories.** `db/load_dynamodb.py` writes children with
  `BatchWriteItem`, so it must compute and write `dependants` itself. Getting this wrong is the
  most likely defect in the whole refactor: a seeded faculty would appear to have no dependants and
  delete cleanly, orphaning six programmes.

### Repair path

`make repair-counters` runs `db/repair_counters.py`, which for each faculty counts rows in
`programs`, `professors` and `courses` under `gsi1pk = FACULTY#{id}` and writes
`SET dependants = :n`. It is the reconciliation the drift weakness requires, and doubles as the
fix-up for any manual surgery.

## What is removed

| Removed | Reason |
|---------|--------|
| `ListedItem` (`app/core/dynamodb/base_items.py:60`) | Only `RoomItem` uses it. A dedicated table needs no constant partition to be listable — a Scan is the listing. |
| `GSI1` on rooms | Existed solely to make `TYPE#ROOM` queryable inside the shared table. |
| `FacultyRepository.has_dependants` | Superseded by the delete condition. |
| `Attr('entity_type').eq(...)` filters | Listings filter `Attr('sk').eq('#META')` instead — one rule across every entity table: an entity row is `#META`, a reservation row is `#UNIQUE`. |
| `DYNAMODB_UNIVERSITY_TABLE` | Replaced by `DYNAMODB_GROUPS_TABLE`. Removed outright, not left as a fallback. |
| `UniversityRepository` | Split into `ProfessorsRepository` and `PlacesRepository` — two tables need two repositories. |

The `entity_type` attribute stays on rows. Nothing queries it any more, but it costs nothing and
remains useful in dumps, logs and the seeder's own bookkeeping.

## Settings

Seven new settings in `app/settings.py`, each read directly with no fallback to the old name:

```python
DYNAMODB_FACULTIES_TABLE: str = 'faculties'
DYNAMODB_PROGRAMS_TABLE: str = 'programs'
DYNAMODB_PROFESSORS_TABLE: str = 'professors'
DYNAMODB_COURSES_TABLE: str = 'courses'
DYNAMODB_ROOMS_TABLE: str = 'rooms'
DYNAMODB_PLACES_TABLE: str = 'places'
DYNAMODB_GROUPS_TABLE: str = 'academic_groups'
```

`DYNAMODB_SLOTS_TABLE` is unchanged. `reported_issues` keeps its current arrangement (a constant in
`tests/db_utils.py`, no setting) — out of scope here.

## Repositories

| Repository | Table | Change |
|------------|-------|--------|
| `FacultyRepository` | `faculties` | table name; `has_dependants` deleted; `delete_faculty` guarded by condition; 409 detection via post-cancellation read |
| `ProgramsRepository` | `programs` | table name; `ConditionCheck` → `TransactUpdate` increment; `delete_program` decrements; `list_programs` filters on `sk` |
| `RoomsRepository` | `rooms` | table name; `list_rooms` becomes a Scan |
| `ProfessorsRepository` (new) | `professors` | `find_professors_by_name`, moved from `UniversityRepository` |
| `PlacesRepository` (new) | `places` | `find_place_by_name`, moved; GSI1 query becomes a Scan with the same `contains` filter |
| `SlotsRepository` | `appointment_slots` | unchanged |

Repositories needing a second table (currently only `ProgramsRepository`, for the faculty counter)
take that table's name from `Settings` in `__init__` and pass it as `table=` on the action. They do
not hold a second `DynamoDBService`.

Splitting `UniversityRepository` reaches one consumer outside the API layer:
`app/ai_assistant/tools/finder_tools.py` opens it at lines 31 and 59 for the professor and place
lookups. `open_university_repository` is replaced by `open_professors_repository` and
`open_places_repository`, and the two tools each open the one they need. This is the only point
where the refactor touches the assistant, and the tools' behaviour and signatures do not change.

`Index` enum docstrings in `app/core/dynamodb/indexes.py` are rewritten: the per-item-type legend
becomes a per-table one, which is shorter, because most tables now hold one item type.

## Seeder

`db/load_dynamodb.py`:

- `main()` calls `recreate_table` eight times with per-table GSI sets, instead of three.
- `build_items` returns a mapping of table name → items rather than one flat list; `load` is called
  per table.
- Faculty items gain `dependants`, computed from the programmes, professors and courses being
  seeded in the same run.
- `demo()` targets the new tables. P5 ("a faculty's professors") is added alongside the existing
  P1/P2/P4/P15, since it is now a query against a dedicated table and is currently unshown.
- `table_definition` needs no change — it is already parameterised by GSI set.

## Tests

- `tests/db_utils.py`: `get_table_specs()` returns eight specs; `university_service` is replaced by
  a helper taking a table name.
- `tests/conftest.py`: one fixture per table; `TestBaseDBClass` exposes the tables its subclasses
  actually use rather than a single `university_table`.
- `tests/factories/`: `RoomRow` loses `gsi1pk`/`gsi1sk`; `GroupRow` keeps them; a faculty factory
  gains `dependants`.
- `tests/api/test_faculties.py`: the "409 when dependants exist" test now creates a real programme
  through the API (which is what increments the counter) rather than writing a child row directly.
- New coverage: deleting a programme decrements the counter and re-permits deleting its faculty.

Every test table is created and dropped by the existing session fixture, so no migration of test
data is involved.

The two agent tools in `finder_tools.py` change which repository they open. They are exercised
through the agent, not by calling `_run` directly, so their coverage stays where it is — the split
must not become an excuse to add direct tool tests.

## Documentation

- **`db/03-table-split.md`** (new) — the criterion, the table map, what single-table cost in this
  codebase, and why `academic_groups` is the one exception.
- **`db/02-dynamodb-model.md`** — corrected only where it became false: the "how many tables"
  section, the `university` row of its table inventory, and the P16 Scan discussion. A pointer to
  `03` is added. Its analysis of single-table trade-offs is deliberately preserved: it is the
  reasoning `03` builds on, and rewriting it would erase why the exception exists.
- **`docs/specs/crud-roadmap.md`** — the Course / Professor / Group / Schedule rungs reference keys
  and tables that change; their descriptions are updated. The ladder itself is unaffected: every
  lesson it teaches (a parent enforced in the write, two GSIs from one write, a sparse search
  index, rows inside a parent partition) survives the split intact.
- **`README.md`** and **`Makefile`** — the `repair-counters` target, and the table list wherever
  the current one table is named.

## Rollback

The refactor is a pure move: no key values change, and the seeder recreates every table from
`db/seed/` on demand. Reverting is `git revert` plus one `python -m db.load_dynamodb` run. There is
no production data and no dual-write window to unwind.

## Out of scope

- CRUD APIs for courses, professors, groups and schedule — the `crud-roadmap.md` rungs, each its
  own spec.
- Implementing the read patterns that have no code (P1, P2, P3, P6, P8). The tables and keys that
  serve them are created and seeded; the repositories are not written here. This means
  `academic_groups` is built for a co-read that still has no caller — accepted deliberately, since
  the alternative is to design the split around the assumption that the schedule will never be
  implemented.
- `ScanIndexForward` on `DynamoDBService.query` (migration step 6 of `02`), whose only consumers
  are unimplemented patterns.
- A `DYNAMODB_ISSUES_TABLE` setting.
- Archiving faculties instead of deleting them — a plausible replacement for the whole 409 path,
  but an API change rather than a modelling one.