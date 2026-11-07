# 0001. DynamoDB, not Postgres

Status: accepted
Date: 2026-08-02

## Context

The seed data (`db/seed/*.json`: faculties, programmes, professors, courses, rooms, places, groups
and their schedule) is relational: `db/01-relational-schema.md` designs it as tables with foreign
keys, and that design is complete. The project is a learning project, and learning DynamoDB is one
of its goals.

## Options

- Postgres, with the schema in `db/01-relational-schema.md`.
- DynamoDB, modelled from the access patterns (`db/02-dynamodb-model.md`).

## Decision

DynamoDB, chosen to learn it. The relational schema stays as the analysis the DynamoDB model was
built from.

## Consequences

- Every read is a numbered access pattern in `db/02-dynamodb-model.md`. A new read is added there
  first; a read with no index is a Scan and is recorded as a deliberate one.
- Related data is denormalised or kept side by side instead of joined. Names behind ids are fetched
  at read time with `BatchGetItem`.
- Referential integrity is code: a parent's `dependants` counter, maintained in the same
  transaction as the child's write, is what a `DELETE` refuses on (`db/03-table-split.md`). The
  seeder bypasses the repositories and keeps the counters itself.
- The agent's checkpoints and the transcript live in DynamoDB too
  (`docs/decisions/0003-own-dynamodb-checkpointer.md`).
- Every test run needs DynamoDB Local (`docker compose up -d dynamodb`, port 8001).
- Table layout: `docs/decisions/0002-table-per-entity.md`.
