# Glossary

What a word means in this repository, and what to call the thing in new code. Values live in the
code; this file says where.

| Term | Here it means | In code | Do not call it |
| --- | --- | --- | --- |
| document issue | A problem found in an applicant's documents on file | `ISSUE_CODES` in `applicant_documents/document_rules.py`, `Verdict.issues` | "error", bare "issue" |
| reported issue | A student's complaint, filed through the QA agent and voted on | `reported_issues` table, `report_issue` and `list_common_issues` tools, `app/api/v1/issues/` | bare "issue" |
| file outcome | What became of one file of a message: added, replaced, unchanged, duplicate, unrecognized. Never affects `is_valid` | `FileOutcome` in `applicant_documents/enums.py`, `Verdict.file_outcomes` | "issue" |
| documents on file | The latest file of each applicant document read earlier in the conversation, as fields, masked | `documents_on_file` in `AssistantAgentState` and `MainState` | "uploads": the images themselves are never stored |
| applicant sets | The course's two sample sets of applicant documents, one valid and one broken | `build_documents`, `build_broken_documents` in `tests/factory_models/applicant_factory.py` | "fixtures" |
| thread | One conversation of the LangGraph assistant; its key is the `user_id` the request posts | `thread_id` in the graph config, `build_thread_key` | "session", which is the CrewAI assistant's `session_id` |
| checkpoint | The graph's own state of a thread, only meaningful to LangGraph | `agent_checkpoints`, `checkpointer/` | "history" |
| conversation history, transcript | What was said in a thread, a row per message; what the history endpoint replays | `conversation_history`, `history/` | "checkpoint" |
| pending approval | A turn paused before a gated write, waiting for a human | `status: "pending_approval"`, `pending`, `HumanInTheLoopMiddleware` | "confirmation" |
| decisions | The answer to a pending approval: approve, edit or reject, one per paused action | `Decision`, the `decisions` form field | "resume payload" |
| dependants | A parent row's count of the rows filed under it; a `DELETE` of the parent refuses while it is above zero | `dependants` on faculty and programme rows | "references" |
| reference tables | The seeded tables of university data, wiped and refilled by `make seed`. Not `agent_checkpoints` or `conversation_history` | `db/load_dynamodb.py` | "fixtures", "test data" |
| access pattern | A numbered read from DynamoDB, `P1` to `P22`; a new read gets a number first | the table in `db/02-dynamodb-model.md` | "query" alone |
| golden | A hand-written question with its expected answer, for the evaluations | `tests/integration/data/goldens.json` | "test case" alone |
| milestone, task | A unit of the course this project follows, such as `M06` or `T06.3`; the course files are not in the repository | commit messages, docs | |
| `university_knowladge` | The knowledge base package; the misspelling is the real name | `app/ai_assistant/university_knowladge/` | do not "fix" the spelling in imports |
| subagent | One of the two agents the main graph routes to, `qa` or `booking` | `GraphNode`, `agents/` | "crew", which is CrewAI's |
