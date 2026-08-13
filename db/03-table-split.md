# Splitting the `university` table (stage 3)

`02-dynamodb-model.md` designed one table for the academic core. This splits it into seven — one
per entity — with a single deliberate exception: **groups and their schedule rows stay together**.

Nothing about the keys changed. Every `pk`, `sk` and GSI value is exactly what `02` specified;
rows only moved to a different table. That is what made the refactor a pure move, and it is why the
reasoning in `02` is preserved rather than rewritten: this document is the amendment, not the
replacement.

## Why

`02` states the criterion itself: single-table is **a means, not an end**, justified where items
are *related and read together*. Six of the seven entities in `university` did not meet it, and the
one that did had no code.

What the shared table cost, measured in this codebase rather than in general:

| Cost | Where |
|------|-------|
| `list_faculties` scanned all 89 seeded rows to return 6 | `app/api/v1/faculty/repository.py` |
| `list_programs` did the same | `app/api/v1/programs/repository.py` |
| Listing rooms needed an invented constant partition `TYPE#ROOM`, because every row had a `pk` of its own | `ListedItem`, since removed |
| Every row carried `entity_type` purely so a listing could filter the other entities out | the two scans above |
| `FacultyNameItem` had to *omit* `entity_type` so the faculty scan would step over it | `app/api/v1/faculty/schemas.py` |
| GSI1's meaning depended on the item type, so `indexes.py` needed a legend to be readable | `app/core/dynamodb/indexes.py` |

What it bought: nothing yet. The one co-read that justifies a shared table — P1, a group and its
schedule in one query — had no repository, and `university`'s GSI2 had no consumer at all.

Splitting was also the cheaper direction to be wrong in. Moving rows out preserves their keys
verbatim (`FACULTY#{id}` / `#META` is a valid key in a dedicated `faculties` table). Merging
separate tables later would mean rewriting keys, adding a discriminator, dual-writing and
backfilling indexes. Starting split forfeits only the co-read optimisation — which is needed
exactly when the access pattern is known, i.e. when the uncertainty that motivates splitting has
already resolved.

## The criterion

An entity is colocated with its parent only when all three hold:

1. **No entry point of its own.** It is never read by an id that identifies it alone.
2. **Bounded volume.** The number of children per parent has a known ceiling.
3. **Shared lifecycle.** It is deleted or archived with the parent.

| Pair | Entry point | Bounded | Lifecycle | Verdict |
|------|-------------|---------|-----------|---------|
| group → schedule rows | no — a class is only read via its group, professor or room | yes — a weekly timetable | yes — dies with the group | **together** |
| faculty → programs | no — `PROGRAM#{id}` | no | no | apart |
| faculty → professors | no — `PROF#{id}` | no | no | apart |
| faculty → courses | no — `COURSE#{id}` | no | no | apart |
| program → groups | no — `GROUP#{id}` | no | no | apart |
| entity → its name reservation | none of its own | yes — exactly one | yes — deleted with the entity | **together** |

The name reservations passing the same three tests is not a coincidence to gloss over: it is the
criterion validating itself on a row type that was never designed against it. `FACULTY_NAME#` lives
in `faculties`, `PROGRAM_NAME#` in `programs`, and `create_faculty` stays a single-table transaction
as a result.

## Table map

```
faculties          FACULTY#{id}/#META           entity row (carries `dependants`)
                   FACULTY_NAME#{name}/#UNIQUE  uniqueness reservation
                   no GSI                       listing: Scan, filtered sk = #META

programs           PROGRAM#{id}/#META           entity row (carries `dependants`)
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

`academic_groups` **keeps an overloaded GSI1** — `PROGRAM#` for a group row, `PROF#` for a class
row. This is the single-table technique being removed everywhere else, retained here deliberately:
in this table the two item types genuinely share a partition and are read together, so the
overloading is paid for rather than inherited.

`sk` still earns its keep in a one-entity table: `#META` is the entity's own row, `#UNIQUE` a
reservation beside it. Every listing is the same scan filtered on `sk = #META`, whichever entity it
lists — which is what replaced the `entity_type` filters. The attribute itself stays on rows: nothing
queries it, but it costs nothing and names the item in a dump or a log.

## Cross-table transactions

`TransactWriteItems` spans any table in the region; only our wrapper was scoped to one. Each action
dataclass in `app/core/dynamodb/schemas.py` now carries `table: str | None = None`, and
`transact_write` resolves `action.table or self._table_name` — so existing call sites are untouched,
and a repository reaching another table says so on the action rather than holding a second service.

A fourth action type, `TransactUpdate`, was added at the same time: `Update` is what a counter needs,
and it was the one `TransactWriteItems` member the wrapper had never emitted.

## Referential integrity: the `dependants` counter

### The problem

`has_dependants` was one GSI1 query on the `FACULTY#{id}` partition, which held programmes,
professors and courses at once. After the split that partition does not exist — the three types are
in three tables.

Three parallel queries would work, but would keep the flaw `02` already records: the check is a
**read**, so a programme created between the check and the delete is orphaned. Making the read three
times wider does not narrow that window.

### The invariant

A `faculties` row carries `dependants: int` — the number of programmes, professors and courses filed
under it. A `programs` row carries the same count of its groups. Each is maintained by whatever
creates or deletes a child, inside the same transaction as the child's own write, and the delete
guard reads it as a **condition** rather than as a prior read:

```python
TransactDelete(
    FacultyItem.key(faculty_id),
    condition=Attr(KeyAttr.PK).exists()
              & (Attr('dependants').not_exists() | Attr('dependants').eq(0)),
)
```

`not_exists()` is required, and not merely for rows written before the counter existed. It is the
**ordinary** case: `dependants` is not a field of `FacultyItem`, so `create_faculty` writes a row
without it, and the attribute only comes into being on the first `ADD`. Every faculty that has never
had a child is `not_exists()`, and dropping that half of the condition would make a freshly created
faculty undeletable.

`create_program` keeps exactly three actions — the `ConditionCheck` asserting the faculty exists was
*replaced* by an `Update` that both increments and asserts, since a conditional `ADD` fails exactly
when the row it would update is absent. `ADD` reads a missing attribute as zero, so no faculty needs
initialising, and the action positions are unchanged.

The programme counter is the same mechanism one rung down. It is what `DELETE /programs/{id}` refuses
on, now that a programme's groups live in `academic_groups` and a query for them would be a read of
another table — reopening the race the counter exists to close. Only the seeder and the test factory
create groups today; group CRUD inherits the obligation.

### Telling 404 from 409

Position 0 of `CancellationReasons` now fails for two reasons — the row is gone, or it still has
dependants. They are told apart by a `get_item` issued **after** the cancellation. That read is not a
race: the transaction has already been rejected, nothing was deleted, and its answer only picks a
status code. The repositories return `DeleteOutcome` (`app/core/enums.py`) instead of a `bool`, and
the service's 404/409 mapping is otherwise unchanged.

`app/core/enums.py` holds one `CreateOutcome` and one `DeleteOutcome` for every entity. Both describe
the *shape* of a transactional write — a create that reserves a name under a parent, a delete that
refuses while children remain — rather than the entity it touched, so a per-entity copy would differ
only in its name. Hence `PARENT_NOT_FOUND`, not `FACULTY_NOT_FOUND`: which parent it is changes per
entity, the position in the transaction that reports it does not. What genuinely differs is the error
message, and that stays at the call site, where the service knows which parent it asked for.

### Known weaknesses

- **Counter drift.** A lost decrement (a bug, a row deleted through the AWS console, a partial seed)
  leaves a parent permanently undeletable, and nothing detects it — the guard believes the number,
  not the tables. The fix is `make seed`, which rebuilds every counter along with every row. That is
  adequate only because there is no data here that a re-seed would destroy; the day there is, this
  needs a reconciliation that recounts in place instead.
- **Contention.** Every child create writes the same parent row. Irrelevant at this scale; a
  documented ceiling rather than a problem.
- **The seeder bypasses the repositories.** `db/load_dynamodb.py` writes children with
  `BatchWriteItem`, so it computes and writes `dependants` itself, in `_count_faculty_dependants`.
  Getting this wrong is the most likely defect in the whole refactor: a seeded faculty would appear
  to have no dependants and delete cleanly, orphaning six programmes. **Nothing checks it.** The API
  tests exercise the counters only through create and delete, and never look at a seeded row, so a
  broken `_count_faculty_dependants` fails silently. It was verified once by hand, against the split
  it was written for; a test over `build_items()` would be the cheap way to keep it verified, and is
  not written.

## What was removed

| Removed | Reason |
|---------|--------|
| `ListedItem` | Only `RoomItem` used it. A dedicated table needs no constant partition to be listable — the Scan is the listing. |
| `GSI1` on rooms and places | Existed solely to make `TYPE#ROOM` / `TYPE#PLACE` queryable inside the shared table. |
| `FacultyRepository.has_dependants`, `ProgramsRepository.has_dependants` | Superseded by the delete conditions. |
| `Attr('entity_type').eq(...)` filters | Listings filter `Attr('sk').eq('#META')` instead. |
| `DYNAMODB_UNIVERSITY_TABLE` | Replaced by seven settings, each read directly. Removed outright, not left as a fallback. |
| `UniversityRepository` | Split into `ProfessorsRepository` and `PlacesRepository` — two tables need two repositories. |

Ordering that a GSI used to provide moved into Python where it was cheap to keep: `list_rooms` sorts
by `(building, number)` after the scan, which needs no zero-padding, because those are integers here
rather than a lexicographic sort key.

## What this does not do

- CRUD for courses, professors, groups and schedule — the `docs/specs/crud-roadmap.md` rungs.
- The read patterns that still have no code (P1, P2, P3, P6, P8). The tables and keys serving them are
  created and seeded; the repositories are not written here. `academic_groups` is therefore built for
  a co-read that has no caller yet — accepted deliberately, since the alternative is to design the
  split around the assumption that the schedule will never be implemented.
- `ScanIndexForward` on `DynamoDBService.query`, whose only consumers are those unimplemented patterns.
- Archiving faculties instead of deleting them — a plausible replacement for the whole 409 path, but
  an API change rather than a modelling one.

## Rollback

A pure move: no key values changed, and the seeder recreates every table from `db/seed/` on demand.
Reverting is `git revert` plus one `make seed`. There is no production data and no dual-write window
to unwind.

## One leftover, on an already-seeded database

`make seed` no longer names `university`, so it no longer drops it either: a local DynamoDB seeded
before this change keeps the old table, full and unread. Nothing breaks — the repositories point
elsewhere — but the rows are dead weight and the table is confusing to find in a listing. Drop it by
hand once:

```bash
aws dynamodb delete-table --table-name university --endpoint-url http://localhost:8001
```

A sum worth checking before you do: the seven new tables hold 104 rows between them, which is exactly
what `university` held. The split neither lost nor duplicated a row.
