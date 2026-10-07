# M06 — the `get_schedule` tool

**Date:** 2026-10-04
**Status:** Implemented on `add_schedule_tool`. The function tests pass against the local DynamoDB;
the five routing cases are written but have not been run — `make eval-routing` calls OpenAI.
Revised 2026-10-06 after the review in `review-add_schedule_tool-2026-10-04.md` (findings 1, 3, 6,
8, 9, 11, 12, 13, 14); this document describes the code as it stands after those fixes.
Revised 2026-10-07: the rules of behaviour moved out of the tool's docstring, argument descriptions
and result sentences into the `get_schedule` section of `MAIN_CHAT_PROMPT`.
**Source:** course milestone 06 (`06-schedule-tool.md`), tasks T06.1–T06.3. The bonus T06.4
(`is_room_free`) is out of scope.

## Overview

The LangChain assistant gets a fifth tool, `get_schedule`, that answers timetable questions: "when
is Intro to Programming for CS-1?", "what does Dr. Alan Whitfield teach on Monday?", "what does CS-1
have this week?". It takes four optional filters — `group`, `professor`, `course`, `weekday` — and
returns the matching classes with names instead of ids, ordered by weekday and start time.

The timetable is not in the handbook (`docs/handbook.pdf` has no group codes, course names or class
times), so this is one of the cases the DynamoDB repositories exist for rather than a job for the
retriever.

**Nothing stored changes.** No new attribute, no new index, no change to `db/load_dynamodb.py` or
to the seed. Every read below goes through a key or index that already exists; where none exists,
the read is a Scan and is recorded as one.

## What the table gives us

A class is one row in `academic_groups`, filed under its group's partition:

```
pk = GROUP#{group_id}     sk = SCHED#{wd}#{start}
gsi1pk = PROF#{professor_id}    gsi1sk = SCHED#{wd}#{start}
gsi2pk = ROOM#{room_id}         gsi2sk = SCHED#{wd}#{start}
course_id, group_id, professor_id, room_id, weekday ('Mon'), start_time ('09:00'), end_time
```

Two things follow from that shape.

**The row holds ids, the answer needs names.** Course, group, professor and room are four other
rows in four tables. DynamoDB has no join, so the names are fetched in a second step — a
`BatchGetItem` per table. Copying the names onto the class row was the alternative and was turned
down: it changes the stored model, and every later rename would owe the class rows a rewrite.

**The applicant gives names, the keys take ids.** P1 needs a group id and P2 a professor id. A
professor's id is reachable by name through `GSI_NAME` (P15). A group's is not: nothing indexes a
group by name. A course has no index by name either, and a weekday alone selects no partition.

## Reading the timetable

Five steps, always in this order: resolve names to ids, read classes by the narrowest key
available, drop what the id-level filters exclude, apply the course filter, then fetch the
remaining names and sort.

### 1. Resolve

| Filter | How it becomes ids |
|---|---|
| `group` | Scan `academic_groups`, filter `sk = #META` (P17, new); the name is compared in the application, through `normalize_name` on both sides, so `cs-1` finds `CS-1`. Exact match, not a prefix. |
| `professor` | `ProfessorsRepository.find_professors_by_name` — the lookup `find_person` already uses. |
| `course` | Not resolved. Applied to the fetched course name in step 4. |
| `weekday` | Nothing to resolve: it is the `Weekday` enum. |

Both lookups return `id -> name`, not just ids: the names are what step 5 would otherwise go back
to the database for. They share nothing, so when both filters are given they run in one
`asyncio.TaskGroup` rather than one after the other.

A `group` or a `professor` that resolves to nothing raises `NotFoundError`. It is not an empty
result: "nobody is called that" and "that group has no classes" are different answers, and a
caller handed an empty list for both tells the applicant that a professor teaches nothing when it
was only the spelling of the name that did not match.

An empty or whitespace-only string is a filter that was not given, in the service and in the tool
alike.

### 2. Read

| Given | Read | Pattern |
|---|---|---|
| `group` | base table, `pk = GROUP#{id}`, `sk begins_with SCHED#` — once per matched group | P1 |
| `professor`, no `group` | GSI1, `gsi1pk = PROF#{id}`, `gsi1sk begins_with SCHED#` — once per matched professor | P2 |
| neither | Scan, filter `begins_with(sk, 'SCHED#')` | P18 (new) |

The weekday is not put into the key condition. `SCHED#{wd}#` would need the reader to derive the
weekday's number the way `db/load_dynamodb.py` does, and a group's or a professor's whole week is a
handful of rows — filtering them in the application costs nothing and keeps the reader from
depending on an encoding only the loader owns.

### 3. Narrow by id

Applied to the rows just read, before any name is fetched:

- `weekday` — the row's `weekday` attribute equals it;
- `professor`, when `group` chose the read — the row's `professor_id` is one of the resolved ids.

### 4. Course filter

Only when `course` is given. The course names of the remaining rows are fetched — one
`batch_get` on `courses` — and a row is kept when the filter is a substring of its course's name,
both sides through `normalize_name` (`programming` matches `Intro to Programming`, whatever the
case or spacing). A class whose course row is missing has no name to match and is left out.

This runs before the other names are fetched, so the group, professor and room tables are read
only for the classes the filter keeps.

### 5. Hydrate and order

The `course_id`, `group_id`, `professor_id` and `room_id` of the remaining rows are fetched with
`batch_get` — one call per table, in one `asyncio.TaskGroup`. An id whose name is already held is
not fetched again: the group and professor names from step 1, the course names from step 4. Keys
come from `TableItem.key` (`GroupItem`, `ProfessorItem`, `CourseItem`, `RoomItem`).

Parent rows are read one way for all four tables — as stored, without validating them against the
entity's schema. Those schemas carry bounds meant for creating an entity, and a stored row that
predates them is still a row to print a name from.

A parent that no longer exists is not an error and does not drop the class. `DELETE /rooms/{id}`
checks nothing about classes, so a class can already outlive its room. Such a class stays in the
result with `None` in the fields that parent would have filled, and the service logs a warning.

The result is ordered by weekday, in `Weekday`'s declaration order (Mon → Sun), then by
`start_time`.

### What one question costs

"What does Dr. Alan Whitfield teach on Monday?" — a `GSI_NAME` query, a GSI1 query, then three
`BatchGetItem` in parallel (courses, groups, rooms; his own name is already held): three round
trips of latency.

"When is Intro to Programming for CS-1?" — one Scan to find the group, one Query for its classes,
one batch get for the course names, then two in parallel (professors, rooms): four round trips. A
course filter costs one round trip more than it used to, and in exchange the other tables are read
for the matching classes only.

## Components

### `app/api/v1/schedule/schemas.py`

`ScheduleEntry` — one class as the model reads it. Every field carries `Field(description=...)`.

| Field | Type | Notes |
|---|---|---|
| `course` | `str \| None` | course name |
| `group` | `str \| None` | group code, e.g. `CS-1` |
| `professor` | `str \| None` | full name as stored, e.g. `Dr. Alan Whitfield` |
| `building` | `str \| None` | the room's building |
| `room_number` | `int \| None` | the room's door number |
| `weekday` | `Weekday` | `Mon` … `Sun` |
| `start_time` | `str` | `HH:MM` |
| `end_time` | `str` | `HH:MM` |

The five optional fields are `None` only when the record they come from has been deleted; each
description says so.

Times are strings on purpose. LangChain turns a tool result into text with `json.dumps` and falls
back to `str()` when that raises; a `datetime.time` anywhere in the result makes it raise, and the
model then reads a Python repr of the whole list.

### `app/api/v1/schedule/schedule_repository.py`

`ScheduleRepository(DynamoDBService)` over `DYNAMODB_GROUPS_TABLE`, one method per read:

- `find_groups_by_name(name: str) -> dict[str, str]` — `id -> name`, P17
- `list_group_classes(group_id: str) -> list[Schedule]` — P1
- `list_professor_classes(professor_id: str) -> list[Schedule]` — P2
- `list_classes() -> list[Schedule]` — P18

`Schedule` is the existing model in `app/api/v1/rooms/schemas.py`. It types the times as
`datetime.time`, which is what validates a stored value; `ScheduleEntry` renders them back as
`HH:MM`.

The partition keys in the two Queries are asked of `GroupItem` and `ProfessorItem` rather than
spelled as `GROUP#…` / `PROF#…`. Those two and `CourseItem` are new in
`app/api/v1/rooms/schemas.py`, beside `RoomItem`: `TableItem` subclasses that carry an `entity`
prefix and no fields yet, there so that `TableItem.key` is the one place a key is built.

### `app/api/v1/schedule/schedule_service.py`

```python
class ScheduleService:
    def __init__(
        self,
        schedule: ScheduleRepository,
        professors: ProfessorsRepository,
        courses: DynamoDBService,
        rooms: DynamoDBService,
    ) -> None: ...

    async def find_schedule(
        self,
        group: str | None = None,
        professor: str | None = None,
        course: str | None = None,
        weekday: Weekday | None = None,
    ) -> list[ScheduleEntry]: ...


@asynccontextmanager
async def open_schedule_service() -> AsyncGenerator[ScheduleService, None]: ...
```

`find_schedule` is the five steps above. It raises `NotFoundError` (`app/core/exceptions.py`) when
`group` or `professor` names nobody. With no filter at all it returns the whole timetable — that is
a legitimate question to ask a function; whether the *model* should be handed the answer is the
tool's decision, below.

`open_schedule_service` opens one DynamoDB client and builds all four collaborators on it, the way
`open_professors_repository` builds one. Groups are hydrated through `ScheduleRepository` itself
and professors through `ProfessorsRepository` — both are `DynamoDBService`s and inherit
`batch_get`.

A service rather than a fifth repository method because the lookup spans four tables, and a
repository here is one table.

### `app/ai_assistant_langchain/agent_schemas.py`

`GetScheduleToolInput` — `group`, `professor`, `course`: `OptionalFilter = None`; `weekday: Weekday
| None = None`. Each description says what to pass and that a filter the question did not name is
left out. `professor` says "full name or first name", as `FindPersonToolInput` does, because it is
the same lookup.

`OptionalFilter` is `Annotated[str | None, BeforeValidator(...)]`: a string of nothing but
whitespace validates to `None`. It is done in the type so that the tool's "was anything given?"
test sees the same answer the lookup does — `' '` is truthy, and let through as a course it matched
every class there is.

### `app/ai_assistant_langchain/tools.py`

```python
@tool(args_schema=GetScheduleToolInput)
async def get_schedule(
    group: str | None = None,
    professor: str | None = None,
    course: str | None = None,
    weekday: Weekday | None = None,
) -> list[dict] | str:
```

Added to `ASSISTANT_TOOLS`. Not `return_direct`.

| Call | Result |
|---|---|
| no filter at all | `NO_TIMETABLE_FILTER`: the timetable is looked up by a group, a professor, a course or a weekday, and none of them was given. No read is made. |
| a group or professor nobody is called | the service's `NotFoundError`, as text: `No group is called 'CS-9'. This does not mean there are no classes.` |
| nothing matches | `No classes found for group='CS-2', weekday='Mon'.` — the filters that were sent, so the model can relay which of them it was. When `course` was one of them, a sentence follows saying a course is matched as part of its stored name. |
| matches | `[entry.model_dump() for entry in entries]` |

No filter is refused rather than answered with everything: the whole timetable is the one result
here that grows with the university, and it would be spent on a question too vague to answer well.
The refusal names `retriever` because a call with no filter is as likely a misrouted question that
was never about classes; told only to ask which group, the model put that question to someone who
had asked when the term starts.

A course cannot be reported unknown the way a group can — it is matched as part of a name, not
looked up — which is why its miss carries a hint instead of an error.

Dicts rather than the models themselves, as `find_person` and `find_place` do — a list of pydantic
models reaches the model as `[ScheduleEntry(course='…', …)]`, a list of dicts as JSON.

The docstring states the contract only: what the tool looks up, which filters it takes and what
it returns. A result says what happened and nothing about what to do next.

### `app/ai_assistant_langchain/prompts.py`

- `MAIN_CHAT_PROMPT` — the `get_schedule` section is the routing signal and holds every rule of
  behaviour. It names the questions the tool is for (when and where a course is taught, what a
  group has on a day, what a professor teaches), says to pass only the filters the message names,
  and draws the one line that is easy to blur: a professor's *office hours* are not classes and
  stay with `find_person`. It also says what to do with each result that is not a timetable: ask
  which group, professor, course or day when no filter was given (or go to `retriever` for term
  dates and programme contents), check an unknown name with the applicant, and try a shorter part
  of a course name before saying there are no classes.
- `ROUTER_PROMPT` — "class timetables" joins the list of what `qa` handles. "When is the CS-1
  lecture on Monday?" asks about a time without asking to meet anyone, which is the shape the
  router already treats as not-booking; naming it removes the doubt.

## Tests

### Routing — `tests/integration/test_tool_routing_eval.py`

Five cases, run by the existing `make eval-routing`:

| Question | Expected |
|---|---|
| When is Intro to Programming for group CS-1? | `get_schedule` |
| What does Dr. Alan Whitfield teach on Monday? | `get_schedule` |
| What are Dr. Alan Whitfield's office hours on Monday? | `find_person` |
| When does the Fall term start and end? | `retriever` |
| Which courses does the Computer Science programme include? | `retriever` |

The third is the boundary T06.3 asks for: a question about a professor that must *not* move to the
new tool. It names the same professor and the same weekday as the case above it — exactly the
filters `get_schedule` takes — and differs only in asking about office hours rather than classes.

T06.3 words it as "which floor is the professor's office?". That question is not used: the routing
metric is an exact match with a threshold of 1.0, and `find_person` returns a `room_id` with no
floor, so a model that cannot answer from the record reaches for a second tool and fails the case
for a reason that has nothing to do with routing.

The last two guard the other boundary: term dates and a programme's curriculum sound like a
timetable and live in the handbook.

`open_schedule_service` is stubbed like the two repositories beside it, with a service that answers
every lookup with one class and echoes the filters it was given into it — for the reason
`StubProfessorsRepository` echoes the name.

### Functions — `tests/dynamodb/test_schedule.py`

No model. Rows are seeded into the `*_test` tables and the functions are called directly.

`ScheduleService.find_schedule`:

- by group — that group's classes only, names filled in, Mon before Wed, 09:00 before 11:00;
- by group, name in another case — still found;
- by professor — P2, across two groups;
- by professor and weekday;
- by group and course — substring, case-insensitive;
- by group and professor — a class of that group taught by someone else is left out;
- by course alone, and by weekday alone — the Scan path;
- by course alone across two groups — the one ordering DynamoDB does not do for us;
- a course written with stray and doubled spaces — still matched;
- unknown group, and unknown professor — `NotFoundError`;
- a class whose room was deleted — returned, `building` and `room_number` are `None`;
- which tables `BatchGetItem` is asked for, and for how many keys: a course filter fetches the
  other names for the surviving classes only, and names the lookup already resolved are not
  fetched again. Counted with a spy around `DynamoDBService.batch_get` that still calls it.

`get_schedule` (the tool, through `ainvoke`):

- matches — a list of dicts with exactly `ScheduleEntry`'s fields, times as `HH:MM` strings;
- nothing matches — the "No classes found" string naming the filters;
- nothing matches and a course was named — the same, with the hint about how courses are matched;
- an unknown group — the "No group is called" string, not "No classes found";
- no filter, an empty string, a whitespace-only string — the narrowing sentence, with
  `open_schedule_service` never entered (asserted with a `patch` on it).

The suite requests `groups_table`, `professors_table`, `courses_table` and `rooms_table`: requesting
a table fixture is how a test here says which tables it may leave rows in.

`tests/factories/` gains row models for a class and a course, and a way to seed a group with an id
and a name. They are added next to the existing factories; `GroupRow` and `create_test_group_row`
keep writing the rows they write today.

## Documentation

- `db/02-dynamodb-model.md`, the access-pattern table — two rows:
  - **P17** — a group by name: `academic_groups`, Scan + filter `sk = #META`, name compared in the
    application. Costlier than P14, whose table holds places alone: this one also holds every
    class row, so the Scan grows with the timetable rather than with the number of groups.
  - **P18** — every class: `academic_groups`, Scan + filter `begins_with(sk, SCHED#)`. Serves the
    timetable questions that name only a course or only a weekday.
- `README.md` — `get_schedule` joins the tools the routing suite is said to cover.

No new make target: the routing cases run under `make eval-routing`, the function tests under a
plain `pytest`.

## Out of scope

- T06.4 `is_room_free` (bonus).
- Any index or attribute that would turn P17 or P18 into a Query.
- Schedule CRUD — still the last rung of `docs/specs/crud-roadmap.md`.

## Known limits

- **A surname alone finds no professor.** `GSI_NAME` matches a prefix of the normalized full name,
  so `Whitfield` misses `Dr. Alan Whitfield`. Inherited from `find_person`, and the fix belongs to
  both at once. The tool now says the name was not found rather than that there are no classes,
  so the model can ask for the first name — but it still cannot find him by surname.
- **At most five professors per name.** `find_professors_by_name` caps its result there.
- **Relative days are not resolved.** "What does CS-1 have tomorrow?" needs today's date, and the
  QA prompt does not carry one (the booking prompt does). The model has to ask which day.
- **Two reads are Scans.** Finding a group by name reads all of `academic_groups` — class rows
  included, so its cost follows the timetable, not the number of groups — and so does a question
  naming only a course or a weekday. A course-only question then fetches the course names of every
  class to decide which to keep; the other three tables are read for the survivors alone. Fine for
  a seed of ten groups and ten classes; the fix, when it is worth making, is a name index on group
  rows and a course index on class rows, both of which are changes to the stored model.
- **A lookup that fails inside a `TaskGroup` arrives as an `ExceptionGroup`.** Resolving the two
  names, reading several partitions and fetching the names all run that way, so a DynamoDB error or
  a malformed class row surfaces wrapped rather than as itself. Review finding 4; not addressed.
