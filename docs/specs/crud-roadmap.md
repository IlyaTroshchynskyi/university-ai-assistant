# CRUD roadmap: which entity to build next

Which entity to give a CRUD API next, ordered so that each one introduces **one** new idea rather
than several at once. Access-pattern numbers (P1, P4, …) refer to the table in
`db/02-dynamodb-model.md`; key layouts refer to `app/core/dynamodb/indexes.py`.

## Where we are

| Done                                | What it established                                                                                                                                                 |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Room** (`app/api/v1/rooms/`)      | Its own partition, listing over a constant GSI1 partition (`TYPE#ROOM`, P13/P14 shape), conditional put and delete                                                  |
| **Faculty** (`app/api/v1/faculty/`) | Uniqueness of a business field via a reservation row written in a `TransactWriteItems`, plus a dependency guard on delete (`has_dependants`), listing by Scan (P16) |

Two gaps worth naming before picking the next entity:

- **No entity exposes a `U` yet.** `DynamoDBService.update_item` exists
  (`app/core/dynamodb/base_service.py:63`), but no repository calls it: both expose create / find /
  delete / list and nothing else. What is shipped so far is CRD, not CRUD.
- **`TransactWriteItems` supports only two of the three actions we need.**
  `app/core/dynamodb/schemas.py` defines `TransactPut` and `TransactDelete`; there is no
  `ConditionCheck`, which is how a write asserts that *another* item exists without modifying it.

---

## Next: Program

The smallest genuine step up from Faculty, because it is the first entity with a **parent**.

### What it introduces

**A foreign key, enforced in the write.** Program's `gsi1pk` is `FACULTY#{faculty_id}`, so creating
one is only valid if that faculty exists. Checking with a read first is a race; the correct form is
a `ConditionCheck` on the faculty row inside the same transaction as the program's `Put`. That means
adding a third action type to `TransactAction` and handling it in `DynamoDBService.transact_write` —
a small, self-contained addition to machinery that already works.

**It activates a guard that is currently dormant.** `FacultyRepository.has_dependants` queries GSI1
on `FACULTY#{id}`, but nothing created through the API is filed under that partition today — only
the seed writes such rows. Until Program (or Professor, or Course) exists, "a faculty with
dependants cannot be deleted" is untested by anything the API can produce. Program makes that path
real.

**Parent-scoped listing (P4).** `GET /faculties/{id}/programs` is a GSI1 query on `FACULTY#{id}`
with `begins_with('PROGRAM#')` on the sort key. This is a third listing shape: Room lists from a
constant partition, Faculty lists by Scan, and this one is bounded by a parent — the most common
shape in practice.

**Composite uniqueness.** A program's name is unique *within its faculty*, so the reservation row's
key generalises from `FACULTY_NAME#{name}` to `PROGRAM_NAME#{faculty_id}#{name}`. Same pattern as
`FacultyNameItem`, one dimension harder.

**The first real update.** Renaming a program cannot be a plain `UpdateItem`: the old reservation
has to be released and the new one taken atomically with the change itself — a three-action
transaction (delete the old reservation, put the new one, update the program). This is the version
of `U` worth learning, rather than a single `SET` on an unconstrained attribute.

### Starting point

Draft `ProgramCreate` / `Program` models already exist in `app/api/v1/rooms/schemas.py` (under the
`# Todo refactor to diff modules` note) and can move into `app/api/v1/programs/`.

One correction when they move: the draft carries `faculty: str` holding the faculty **name**, which
is the seed-file shape. The API should take `faculty_id`, since that is what the GSI1 key is built
from.

---

## The ladder after that

### Course

Two indexes at once: `GSI1 = FACULTY#{faculty_id}` (P6) and `GSI2 = PROF#{professor_id}` (P8). Two
parents to validate on create, and reassigning a course to another professor rewrites the GSI2 keys.
The first entity where a single write maintains two index projections.

### Professor

`GSI1` by faculty (P5) plus the sparse `GSI_NAME` index built through `normalize_name_key` (P15).
Renaming rewrites a search key that **the assistant already reads** — see
`UniversityRepository.find_professors_by_name`. The first CRUD that closes the loop with the agent,
so its tests should cover the search path, not just the row.

### Group

Child of Program (`GSI1 = PROGRAM#{id}`, P7) — a second level of nesting, but structurally a
repeat of Program. Low new learning; worth doing quickly, or skipping until something needs it.

### Schedule

The hardest, and qualitatively different from everything above. A schedule entry is not a `#META`
row with a partition of its own: it is `pk = GROUP#{group_id}`, `sk = SCHED#{wd}#{start}` — a related
row living inside the parent's partition — indexed by professor (`GSI1`, P2) and by room (`GSI2`,
P3), and read as a range (P1).

It is also the first entity with real business invariants: one room cannot host two classes in the
same weekday-and-time slot, and neither can one professor. In DynamoDB that is the reservation-row
pattern generalised to composite keys — several lock rows per lesson, written in one transaction.

---

## Summary

| Order | Entity        | The one new idea                                                                               |
|-------|---------------|------------------------------------------------------------------------------------------------|
| 1     | **Program**   | A parent: `ConditionCheck`, parent-scoped listing, composite uniqueness, rename-as-transaction |
| 2     | **Course**    | Two GSIs maintained by one write                                                               |
| 3     | **Professor** | A sparse search index the assistant reads                                                      |
| 4     | **Group**     | Second nesting level (mostly a repeat)                                                         |
| 5     | **Schedule**  | Rows inside a parent partition + business invariants as lock rows                              |