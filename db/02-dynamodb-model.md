# Moving to DynamoDB (stage 2: the model)

Designing the DynamoDB model on top of the relational schema from `01-relational-schema.md`.

> **Amended by [`03-table-split.md`](03-table-split.md).** The `university` table described below was
> later split into one table per entity, keeping groups and their schedule rows together. **No key
> values changed** — every `pk`, `sk` and GSI value in this document is still exactly what the code
> writes; the rows simply live in different tables. The sections whose *tables* became wrong are
> marked inline. The analysis itself is deliberately preserved: it is the reasoning `03` builds on,
> and it is what states the criterion the split was measured against.

## Approach

Designing for DynamoDB starts **from the queries, not from the entities** — that is the key difference from relational design ([AWS: NoSQL design](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-general-nosql-design.html)). The guiding principles from the AWS documentation:

1. **Access patterns first, schema second.** You cannot design a table until you know the questions it has to answer. The schema is shaped around the most frequent and most important queries, so that each one is served by a single request.
2. **As few tables as possible (ideally one).** Fewer tables → better scalability, simpler permissions and backups. The exception is time-series data, or fundamentally different access patterns.
3. **Denormalization instead of JOINs.** DynamoDB does not do JOINs — related data is either duplicated or stored side by side, so an answer comes back in one query ([AWS: modeling relational data](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-relational-modeling.html)).
4. **Locality of reference.** Related items live in one partition (a shared PK) and are ordered the way they need to be read, via the SK.
5. **Composite keys (PK + SK)** group 'child' items under a 'parent' and make range queries possible (`begins_with`, `between`).
6. **GSIs for alternative patterns.** The same item is indexed under a different key, opening up one more way to read it.
7. **Even distribution.** Keys are designed so that traffic spreads across partitions without hot keys.

### How many tables: a single-table core, with the independent entities kept apart

Single-table is **a means, not an end**. Its one real payoff is fetching several **related** item types in one query (locality). It is justified only where the items are related and read together. The AWS rule of 'as few tables as possible' has an explicit exception in that same documentation: datasets with **fundamentally different access patterns** are kept separate.

So here it is **not 'one table for everything', but 'a few tables'**:

| Table | Entities | Why |
|-------|----------|-----|
| **`university`** (single-table) | faculty, program, group, professor, course, room, place | Interrelated through `schedule` → they benefit from locality (a group plus its schedule in one query). `place` is a small static reference list, cheaper to keep here. |
| **`appointment_slots`** (separate) | consultation slots | Self-contained (no cross-queries with the academic core), **dated → a TTL** auto-deletes past slots, write spikes during the booking period → independent scaling. |
| **`reported_issues`** (separate) | student reports | Fully self-contained, with a life of its own (votes/statuses), potentially a stream — it doesn't get in the reference data's way. |

The criterion for splitting: **together** — if they are read by one query or share access patterns; **apart** — if the entity is independent, with its own lifecycle (TTL, archival) or traffic profile.

> **Superseded — see [`03-table-split.md`](03-table-split.md).** Applying the criterion above to the
> code that actually got written, six of the seven entities in `university` failed it: each has an
> entry point of its own, unbounded volume and a lifecycle independent of its parent. The single
> co-read that justified the shared table — a group plus its schedule — is the one pair `03` keeps
> together, in `academic_groups`. `university` is now `faculties`, `programs`, `professors`,
> `courses`, `rooms`, `places` and `academic_groups`. The criterion did not change; only the answer
> it gives once the access patterns are known did.

> A caveat: if a 'student' entity turns up later, along with a 'student plus their bookings in one request' query, co-locating bookings with the student will start to make sense. Until such a pattern exists, we don't merge them pre-emptively.

**The access layer.** The existing `app/ai_assistant/dynamodb/base_service.py` (`DynamoDBService`) holds **the name of one table** and works with the generic `pk`/`sk` plus GSIs via `index_name`. For three tables we simply build three `DynamoDBService` instances (one per table) — no code changes needed.

> One code change did turn out to be needed, and `03` makes it: a `TransactWriteItems` spanning two
> tables (a programme and its faculty's dependant counter). Each transaction action gained an optional
> `table`, since only our wrapper — not DynamoDB — assumed a single one.

> Note: `DynamoDBService.query` does not currently forward `ScanIndexForward`. For 'top N by votes' (P11), or a schedule in reverse order, that parameter will have to be added.

---

## The full list of access patterns

The **Table** column is as of `03-table-split.md`; the key columns are unchanged from the original design.

| # | Query | Table | How it is served |
|---|-------|-------|------------------|
| P1 | A group's schedule | `academic_groups` | base: `pk = GROUP#{id}`, `sk begins_with SCHED#` |
| P2 | A professor's schedule | `academic_groups` | GSI1: `gsi1pk = PROF#{id}`, `gsi1sk begins_with SCHED#` |
| P3 | A room's occupancy | `academic_groups` | GSI2: `gsi2pk = ROOM#{id}`, `gsi2sk begins_with SCHED#` |
| P4 | A faculty's programmes | `programs` | GSI1: `gsi1pk = FACULTY#{id}`, `gsi1sk begins_with PROGRAM#` |
| P5 | A faculty's professors | `professors` | GSI1: `gsi1pk = FACULTY#{id}`, `gsi1sk begins_with PROF#` |
| P6 | A faculty's courses | `courses` | GSI1: `gsi1pk = FACULTY#{id}`, `gsi1sk begins_with COURSE#` |
| P7 | A programme's groups | `academic_groups` | GSI1: `gsi1pk = PROGRAM#{id}`, `gsi1sk begins_with GROUP#` |
| P8 | A professor's courses | `courses` | GSI2: `gsi2pk = PROF#{id}`, `gsi2sk begins_with COURSE#` |
| P13 | A point read (faculty/programme/professor/course/room/place by id) | its own | base: `pk = <ENTITY>#{id}`, `sk = #META` |
| P14 | A campus place by name | `places` | **Scan** + filter `contains(name_lower, …)` — the filter was always post-read, so the index it ran over bought nothing once the table held places alone |
| P16 | Every faculty | `faculties` | **Scan** + filter `sk = #META` — cheap now that the table holds ~6 faculties, see 'P16: the one Scan' below |
| P15 | Professor search **by name** (without knowing the faculty) | `professors` | GSI_NAME: `gsi_name_pk = PROF`, `gsi_name_sk begins_with {name}` |
| P9 | Open slots on a date (topic optional) | `appointment_slots` | base: `pk = {date}`, filter `status = open` (+ `topic`) |
| P9b | Open slots by status (all dates) | `appointment_slots` | GSI1: `gsi1pk = STATUS#{status}` |
| P10 | One student's bookings | `appointment_slots` | GSI2: `gsi2pk = STUDENT#{email}` |
| P11 | Reports by category (top by votes) | `reported_issues` | base: `pk = {category}`, `ScanIndexForward=False` |
| P12 | Reports by status | `reported_issues` | GSI1: `gsi1pk = STATUS#{status}` |

---

## Key design

Every table uses the same key names (`pk`/`sk` plus `gsi1pk`/`gsi1sk`, `gsi2pk`/`gsi2sk`) — so that one and the same `DynamoDBService` can work with them. The key *values* are specific to each table.

Conventions for SK values (so that ordering comes out right):
- **The weekday** is encoded as a number `1..7` (`Mon=1`), otherwise the strings `Mon/Fri/...` sort incorrectly: `SCHED#1#09:00`.
- **Times** in `HH:MM` format sort correctly lexicographically.
- **Votes** are zero-padded (`0052`), so that numeric ordering works as string ordering; the top is read with `ScanIndexForward=False`.

### Table 1 — `university` (single-table, the academic core)

> **The key map below is still current; the table it sits in is not.** These rows now live in seven
> tables ([`03-table-split.md`](03-table-split.md)) with their keys byte-for-byte unchanged — that is
> what made the split a pure move. Two entries did lose keys, both constants that only ever existed
> to make an entity reachable inside the shared table: Room's `TYPE#ROOM` (added after this table was
> written) and Place's `TYPE#PLACE`. Programme's own name reservation, added later, is
> `PROGRAM_NAME#{faculty_id}#{name}` / `#UNIQUE`.

Base keys `pk`/`sk`, two GSIs with overloaded keys, plus one named special-purpose index:
- **GSI1** — 'by owner/parent': faculty→programmes/professors/courses, programme→groups, professor→schedule, type→places.
- **GSI2** — 'the secondary relation': room→schedule, professor→courses.
- **GSI_NAME** — professor search by name (P15). Unlike GSI1/GSI2 it is **not overloaded**: one item type (professor), its own key pair `gsi_name_pk`/`gsi_name_sk`.

| Entity | pk | sk | gsi1pk | gsi1sk | gsi2pk | gsi2sk |
|--------|----|----|--------|--------|--------|--------|
| Faculty | `FACULTY#{id}` | `#META` | — ¹ | — ¹ | — | — |
| Program | `PROGRAM#{id}` | `#META` | `FACULTY#{faculty_id}` | `PROGRAM#{name}` | — | — |
| Group | `GROUP#{id}` | `#META` | `PROGRAM#{program_id}` | `GROUP#{id}` | — | — |
| Professor | `PROF#{id}` | `#META` | `FACULTY#{faculty_id}` | `PROF#{full_name}` | — | — |
| Course | `COURSE#{id}` | `#META` | `FACULTY#{faculty_id}` | `COURSE#{name}` | `PROF#{professor_id}` | `COURSE#{id}` |
| Room | `ROOM#{id}` | `#META` | — | — | — | — |
| Place | `PLACE#{id}` | `#META` | `TYPE#PLACE` ³ | `PLACE#{name}` ³ | — | — |
| Schedule | `GROUP#{group_id}` | `SCHED#{wd}#{start}` | `PROF#{professor_id}` | `SCHED#{wd}#{start}` | `ROOM#{room_id}` | `SCHED#{wd}#{start}` |
| Faculty name ² | `FACULTY_NAME#{name}` | `#UNIQUE` | — | — | — | — |

¹ Faculty is the one entity carrying no GSI keys, which is why P16 is served by a Scan — see below.

² Not an entity — a constraint. See 'Reserving a faculty name' below.

³ Dropped by `03-table-split.md`: a constant partition gathering every place into one 'directory' is
what a table of places already is.

#### Reserving a faculty name

DynamoDB enforces uniqueness on the primary key and nowhere else. A faculty's own key is `FACULTY#{uuid4}`, minted fresh on every create, so `attribute_not_exists(pk)` on it can never fail — two `POST /faculties {"name": "Computer Science"}` would both succeed, which matters because programmes and courses in the seed name their faculty rather than referencing its id (`db/load_dynamodb.py`, the `faculty` field of `programs.json` / `courses.json`; the API takes a `faculty_id`).

So the name gets a primary key of its own: `pk = FACULTY_NAME#{name}`, `sk = #UNIQUE`, lowercased with whitespace collapsed (`FacultyNameItem`, `normalize_name`) so that 'Computer Science' and 'computer  science' land on the same key. `FacultyRepository.create_faculty` writes it and the faculty in one `TransactWriteItems`, conditioned on `attribute_not_exists(pk)` — the *second* action is the one a duplicate name fails on, which is how the cancelled transaction is told apart from any other. `delete_faculty` removes both rows, again transactionally: a reservation outliving its faculty would keep a name unusable for good.

The row deliberately carries no `entity_type` and no GSI keys — nothing indexes it, and P16's `entity_type = faculty` filter steps over it rather than trying to read it as a faculty. The seed writes one per faculty (six rows) and refuses to load a `faculties.json` with a repeated name, which would otherwise collapse into whichever faculty came last and take every programme pointing at the other one with it.

#### Deleting a faculty

Programmes, professors and courses reference a faculty by id and are **not** deleted with it, so `DELETE /faculties/{id}` first asks whether any exist and answers **409** if they do, rather than cascading. All three sit in the one GSI1 partition `FACULTY#{id}` (P4/P5/P6), so the check is a single query capped at one row (`FacultyRepository.has_dependants`).

It is a read, though: a programme created immediately after the check passes is still orphaned. Closing that needs a dependant counter on the faculty row, updated by whatever creates a programme — which is the point at which to add it, since nothing creates one yet.

> **Done, in [`03-table-split.md`](03-table-split.md).** The split forced the question: the three
> child types no longer share a partition, so the query above has nothing to read. `has_dependants`
> is gone, the faculty row carries `dependants`, and the guard is a **condition on the delete**
> itself — which is what closes the window this paragraph describes. Programmes gained the same
> counter for their groups. A drifted counter is repaired by re-seeding, there being no data here
> that a re-seed would destroy.

#### P16: the one Scan

'Every faculty' (`GET /faculties`, `FacultyRepository.list_faculties`) is served by a **Scan of the whole `university` table with a filter on `entity_type = faculty`** — the one place in the model that breaks the 'an index, not a Scan' rule stated for GSI_NAME below. Recorded here rather than left implicit, because the code and this document have to agree on it.

What that costs: a DynamoDB `FilterExpression` is applied **after** the read, so every call reads (and is billed for) the entire table — 89 items from the seed, unbounded in production — to return ~6 faculties. `DynamoDBService.scan` is called without a `limit`, so `_paginate` follows `LastEvaluatedKey` to the end. Faculty also has no sort key of any kind, so the listing comes back in **undefined order**.

Why it stands for now: faculties are a tiny, near-static reference list, and the endpoint is administrative rather than on a student-facing path. The `entity_type` filter is what keeps every other entity in the shared table out of the result.

The fix, when it is worth doing, is the same shape as P14 (`TYPE#PLACE`) — and it is three changes, not one, because a GSI is sparse and rows written without its keys never enter it:

1. `FacultyItem` extends `ListedItem` instead of `TableItem` (giving `gsi1pk = TYPE#FACULTY`), declaring `gsi1sk = FACULTY#{name}` so the listing is ordered.
2. `list_faculties` becomes a Query on GSI1 — the `entity_type` filter then falls away, since the index holds faculties only.
3. `db/load_dynamodb.py` writes `gsi1pk`/`gsi1sk` for faculties. **Without this the seeded faculties are invisible to the new query** and the endpoint quietly returns `[]`.

> **Resolved differently by [`03-table-split.md`](03-table-split.md).** The split fixed the cost
> without building the index: the Scan now reads `faculties`, a table holding ~6 faculties and their
> name reservations, so 'reads the entire table' and 'returns the whole answer' became the same
> thing. The filter is `sk = #META` rather than `entity_type = faculty` — one rule for every entity
> table. `ListedItem` was removed, and the three-step fix above is moot: it existed to make one entity
> type reachable inside a table shared with six others. Ordering, where an endpoint promised it
> (`list_rooms`), is applied in Python after the scan. The listing is still unordered for faculties,
> which nothing has asked for.

**The special-purpose `GSI_NAME` index (Professor only):**

| Entity | gsi_name_pk | gsi_name_sk |
|--------|-------------|-------------|
| Professor | `PROF` (constant) | `{full_name}` — normalized: lowercased, title stripped (`Dr. Alan Whitfield` → `alan whitfield`) |

Why a separate index: we already had a 'the professors of a **faculty**' pattern (P5, through GSI1 on `FACULTY#{id}`), but not 'find a professor **by name**, without knowing the faculty'. That is a new access pattern — we serve it with an index rather than a Scan:
- **The constant `PROF` partition** gathers every professor into a single 'directory' ordered by name → a Query with `begins_with` instead of reading the whole table. The partition is small and almost never written to (reference data), so it won't turn hot.
- **Normalizing the key** (stripping the title, lowercasing) is needed because a sort key in a Query supports only `begins_with` (not `contains`): without stripping `Dr.`, a search for `alan` would not find `Dr. Alan Whitfield`. The tool applies the same normalization to the query.
- **The limitation:** `begins_with` is a prefix. Searching by **first** name or full name works; by surname ('Whitfield') it does not. If searching by surname is needed too — either a second key on the surname, or, for a small reference list, a deliberate Scan + `contains`.
- The index is **sparse**: only professor items carry `gsi_name_*`, so it holds exactly 10 records rather than all 83.

### Table 2 — `appointment_slots` (separate, with a TTL)

One GSI on the status, a second (sparse) one on the student. The `expires_at` attribute (epoch seconds at the end of the slot's day) → we enable a **TTL** on it, and DynamoDB deletes past slots by itself.

| Entity | pk | sk | gsi1pk | gsi1sk | gsi2pk | gsi2sk |
|--------|----|----|--------|--------|--------|--------|
| Slot | `{date}` | `{start}#{id}` | `STATUS#{status}` | `{date}#{start}` | `STUDENT#{booked_by}` | `{date}#{start}` |

### Table 3 — `reported_issues` (separate)

The partition is the category, the ordering is by votes (zero-padded). GSI1 is on the status.

| Entity | pk | sk | gsi1pk | gsi1sk |
|--------|----|----|--------|--------|
| Issue | `{category}` | `{votes}#{id}` | `STATUS#{status}` | `{votes}#{id}` |

The model's key techniques:
- **A group and its schedule live in one partition** (`pk = GROUP#{id}`): the group's meta item (`sk = #META`) and its classes (`sk = SCHED#...`) are read by a single query (P1 plus the group's own details).
- **One schedule item serves 3 patterns** (P1/P2/P3): the base key by group, GSI1 by professor, GSI2 by room. The data isn't duplicated — only the projected keys in the GSIs are.
- **The slot's GSI2** (`STUDENT#{email}`) is a sparse index: an item enters it only once the slot is booked (once `booked_by` appears).
- **`Course` carries `faculty_id`/`professor_id` both as attributes AND as GSI keys** — denormalization for the sake of P6/P8, without touching the faculty/professor tables.
- **In the separate tables the keys are cleaner**: a slot has `pk = {date}` (no `SLOT#` prefix), a report has `pk = {category}` — a type prefix isn't needed, since the table holds a single item type.

### SK vs GSI: how one item ends up in different indexes

- **SK** is part of the primary key of the **base** table. It orders and groups items **inside** one partition (one pk).
- **GSI** is a **separate** index with *its own* key pair (`gsi1pk`/`gsi1sk`). It gives another way to search — by a different field than pk. The base table and each GSI have their own 'partition + ordering' pair.

#### Part 1. One item — three indexes

A schedule record is stored **once**. Its attributes:

```
course_id=1  group_id=1  professor_id=1  room_id=1  Mon 09:00–10:30
```

But it has the keys of three indexes filled in at once, so DynamoDB shows it in three 'views':

| Index | Partition | Ordering |
|-------|-----------|----------|
| Base table | `pk = GROUP#1` | `sk = SCHED#1#09:00` |
| GSI1 | `gsi1pk = PROF#1` | `gsi1sk = SCHED#1#09:00` |
| GSI2 | `gsi2pk = ROOM#1` | `gsi2sk = SCHED#1#09:00` |

The same item → three different queries against it:

| Query | On which index | What it returns |
|-------|----------------|-----------------|
| `pk = GROUP#1` | base table | the schedule of group 1 (P1) |
| `gsi1pk = PROF#1` | GSI1 | the schedule of professor 1 (P2) |
| `gsi2pk = ROOM#1` | GSI2 | the occupancy of room 1 (P3) |

#### Part 2. What the SK does inside one partition

The SK is what lets several related items sit under one pk, ordered for reading. Here is what the `GROUP#1` partition looks like in the base table — the group itself plus its classes:

| pk | sk | what the item is |
|----|----|------------------|
| `GROUP#1` | `#META` | the 'group CS-1' object (name, program_id) |
| `GROUP#1` | `SCHED#1#09:00` | a class: Mon 09:00 |
| `GROUP#1` | `SCHED#1#11:00` | a class: Mon 11:00 |
| `GROUP#1` | `SCHED#3#09:00` | a class: Wed 09:00 |

A single `pk = GROUP#1` query returns **all of those rows at once**, ordered by sk (`#META` first, then the classes by day and time). This is what 'keeping related data together' means — instead of four queries in a relational database (group + schedule + join…) we get one.

> In short: `sk` is the ordering **inside** the base table; `gsi1pk`/`gsi1sk`, `gsi2pk`/`gsi2sk` are the keys of **other** indexes. They all play the same 'sort key' role (ordering within their own partition), but they belong to different indexes.

---

## Item examples

```jsonc
// A programme (P4, P7, P13)
{
  "pk": "PROGRAM#cs-software-engineering", "sk": "#META",
  "gsi1pk": "FACULTY#1", "gsi1sk": "PROGRAM#Software Engineering",
  "type": "program", "program_id": "cs-software-engineering",
  "name": "Software Engineering", "faculty_id": 1,
  "degree": "BSc", "duration_years": 4, "tuition_usd": 24500
}

// A scheduled class — serves P1/P2/P3 with one item
{
  "pk": "GROUP#1", "sk": "SCHED#1#09:00",           // the group's schedule
  "gsi1pk": "PROF#1", "gsi1sk": "SCHED#1#09:00",    // the professor's schedule
  "gsi2pk": "ROOM#1", "gsi2sk": "SCHED#1#09:00",    // the room's occupancy
  "type": "schedule", "course_id": 1, "group_id": 1,
  "professor_id": 1, "room_id": 1,
  "weekday": "Mon", "start_time": "09:00", "end_time": "10:30"
}

// A consultation slot — the appointment_slots table (P9/P9b/P10).
// booked_by and gsi2 appear only after a booking; expires_at is for the TTL.
{
  "pk": "2026-10-06", "sk": "09:00#1",
  "gsi1pk": "STATUS#open", "gsi1sk": "2026-10-06#09:00",
  "date": "2026-10-06", "start_time": "09:00", "end_time": "09:30",
  "topic": "general", "status": "open", "booked_by": null,
  "expires_at": 1759795200
}

// A report — the reported_issues table (P11/P12)
{
  "pk": "facilities", "sk": "0052#1",
  "gsi1pk": "STATUS#open", "gsi1sk": "0052#1",
  "category": "facilities", "status": "open", "votes": 52,
  "text": "Wi-Fi keeps dropping in the Turing Hall lecture halls during class."
}
```

---

## Migration steps

1. **Pin down the access patterns** (the table above) — when new ones appear, update it first, then the keys.
2. **Model it in NoSQL Workbench** — check the partitions/GSIs visually, against the real data in `db/seed/`.
3. **Create the tables** with their GSIs, billing mode `PAY_PER_REQUEST` for the dev environment; enable a **TTL** on `appointment_slots` over the `expires_at` attribute. *(Three when this was written — `university`, `appointment_slots`, `reported_issues`; nine since `03-table-split.md`, plus `agent_checkpoints`, which is created only when missing.)*
4. **Write a seed → items loader** — transforming the JSON in `db/seed/` into items following the key maps (BatchWriteItem). Implemented in `db/load_dynamodb.py`.
5. **Implement repositories on top of `DynamoDBService`** — one service instance per table, one method per access pattern (P1…P16), with `Key(...)`/`begins_with`.
6. **Add a `ScanIndexForward` parameter to `DynamoDBService.query`** — needed for P11 (top reports) and for the schedule in reverse order.
7. **Decide on the 'student' entity** (note #3 in `01-relational-schema.md`): for now `booked_by` is just an email attribute; a full student profile could be introduced as a separate table or item type (`pk = STUDENT#{email}`) — and then co-locating bookings would make sense again.