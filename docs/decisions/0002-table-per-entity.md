# 0002. A table per entity, not one `university` table

Status: accepted, supersedes the single `university` table in `db/02-dynamodb-model.md`
Date: 2026-08-09

## Context

`db/02-dynamodb-model.md` put the academic core (faculty, programme, group, professor, course,
room, place) into one single-table `university`, with `appointment_slots` and `reported_issues`
apart. Its own criterion was that single-table is a means, justified where items are related and
read together. Once the code was written, the shared table cost: `list_faculties` scanned all 89
seeded rows to return 6, listing rooms needed an invented partition `TYPE#ROOM`, every row carried
`entity_type` only so a listing could filter the others out, and GSI1 meant something different per
item type. It bought nothing yet: the one co-read that justifies it, a group with its schedule
(P1), had no repository.

## Options

- Keep the single `university` table.
- One table per entity, colocating a child with its parent only where the criterion holds.

## Decision

One table per entity. A child stays with its parent only when it has no entry point of its own, a
bounded count per parent, and the parent's lifecycle. Groups and their schedule rows pass, so
`academic_groups` holds both; so do the name reservations (`FACULTY_NAME#`, `PROGRAM_NAME#`), which
stay in their entity's table. Every other pair fails.

Splitting was also the cheaper way to be wrong: rows moved out keep their keys verbatim, while
merging separate tables later means rewriting keys, a discriminator, dual writes and backfilled
indexes.

## Consequences

- No key changed; every `pk`, `sk` and GSI value in `db/02-dynamodb-model.md` is what the code
  writes, in a different table. The table map is in `db/03-table-split.md`.
- A write that spans two tables (a programme and its faculty's `dependants`) is one
  `TransactWriteItems`; each action in `app/core/dynamodb/schemas.py` carries an optional `table`.
- Referential integrity across tables is the `dependants` counter, since one `FACULTY#{id}`
  partition no longer holds a faculty's programmes, professors and courses.
- `academic_groups` keeps an overloaded GSI1 (`PROGRAM#` for a group, `PROF#` for a class), the one
  place the single-table technique is kept on purpose.
