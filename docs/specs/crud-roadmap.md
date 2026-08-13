# CRUD roadmap: which entity to build next

Which entity to give a CRUD API next, ordered so that each one introduces **one** new idea rather
than several at once. Access-pattern numbers (P1, P4, …) refer to the table in
`db/02-dynamodb-model.md`; key layouts refer to `app/core/dynamodb/indexes.py`. Which table each
entity lives in is `db/03-table-split.md` — one per entity, so a rung below means a new table, not a
new partition prefix in a shared one.

## Where we are

| Done                                  | What it established                                                                                                                                   |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Room** (`app/api/v1/rooms/`)        | Its own partition, listing a whole table by Scan, conditional put and delete                                                                            |
| **Faculty** (`app/api/v1/faculty/`)   | Uniqueness of a business field via a reservation row written in a `TransactWriteItems`; a dependency guard as a **write condition** on `dependants`; listing by Scan (P16) |
| **Program** (`app/api/v1/programs/`)  | A parent enforced in the write, composite uniqueness (`PROGRAM_NAME#{faculty_id}#{name}`), and a counter maintained across two tables in one transaction |

One gap worth naming before picking the next entity:

- **No entity exposes a `U` yet.** `DynamoDBService.update_item` exists, and `transact_write` can now
  emit an `Update` — but no repository exposes a rename: all three offer create / find / delete /
  list and nothing else. What is shipped so far is CRD, not CRUD. The rename-as-transaction described
  for Program below (release the old reservation, take the new one, update the row) is still unbuilt
  and is still the version of `U` worth learning.

**Every rung below inherits one obligation.** A faculty's `dependants` counter is what
`DELETE /faculties/{id}` refuses on, and it is only correct while *every* writer maintains it.
`ProgramsRepository` does; the seeder does; Professor and Course must increment it on create and
decrement it on delete, in the same transaction as the row itself, or a faculty will delete cleanly
while children still point at it. Group owes the same to its programme's counter. Nothing verifies a
counter automatically, and a wrong one fails silently in the safe-looking direction — the parent
deletes when it should have refused — so each rung should test its own increment and decrement.

---

## The ladder after that

### Course

In the `courses` table. Two indexes at once: `GSI1 = FACULTY#{faculty_id}` (P6) and
`GSI2 = PROF#{professor_id}` (P8). Two parents to validate on create — one of which is in a third
table — and reassigning a course to another professor rewrites the GSI2 keys. The first entity where
a single write maintains two index projections.

### Professor

In the `professors` table: `GSI1` by faculty (P5) plus the sparse `GSI_NAME` index built through
`normalize_name_key` (P15). Renaming rewrites a search key that **the assistant already reads** — see
`ProfessorsRepository.find_professors_by_name`. The first CRUD that closes the loop with the agent,
so its tests should cover the search path, not just the row.

### Group

In `academic_groups`, child of Program (`GSI1 = PROGRAM#{id}`, P7) — a second level of nesting, but
structurally a repeat of Program. It does own one thing nothing else does: `academic_groups` holds
two item types, so a group's delete has to take its schedule rows with it (they share its partition,
and nothing else will ever collect them). Low new learning otherwise; worth doing quickly, or
skipping until something needs it.

### Schedule

The hardest, and qualitatively different from everything above. A schedule entry is not a `#META`
row with a partition of its own: it is `pk = GROUP#{group_id}`, `sk = SCHED#{wd}#{start}` — a related
row living inside the parent's partition, in the one table the split deliberately kept shared —
indexed by professor (`GSI1`, P2) and by room (`GSI2`, P3), and read as a range (P1). It is also the
rung that finally gives `academic_groups`' co-read a caller.

It is also the first entity with real business invariants: one room cannot host two classes in the
same weekday-and-time slot, and neither can one professor. In DynamoDB that is the reservation-row
pattern generalised to composite keys — several lock rows per lesson, written in one transaction.
Those lock rows are the one place the colocation criterion has to be applied afresh: a lock keyed by
room belongs wherever it can be written in the same transaction as the class it guards.

---

## Summary

| Order | Entity        | The one new idea                                                                               |
|-------|---------------|------------------------------------------------------------------------------------------------|
| ✅     | **Program**   | A parent enforced in the write, composite uniqueness, a counter kept across two tables         |
| 1     | **Course**    | Two GSIs maintained by one write                                                               |
| 2     | **Professor** | A sparse search index the assistant reads                                                      |
| 3     | **Group**     | Second nesting level; deleting a parent that owns rows in its own partition                    |
| 4     | **Schedule**  | Rows inside a parent partition + business invariants as lock rows                              |

Course and Professor also owe the faculty counter their increments and decrements; Group owes them to
its programme. See 'Where we are' above.

Two things every rung reuses rather than reinvents: `CreateOutcome` and `DeleteOutcome`
(`app/core/enums.py`) are the return vocabulary of a guarded write for *all* entities — a new
repository maps its cancelled positions onto them and writes its own error message, and should not
add a `CourseCreateOutcome`. The counter attribute has one spelling, `DEPENDANTS`
(`app/core/dynamodb/base_items.py`), for the same reason: a guard reading a name nothing writes
refuses nothing, silently.