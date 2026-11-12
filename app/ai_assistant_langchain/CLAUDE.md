# The LangGraph assistant

Rules for changing this package. The shape of the graph is in the root `CLAUDE.md`; how a feature
works, in the `docs/` file each section names.

## QA agent (`agents/qa/`)

- Tools are in `tools.py` (`ASSISTANT_TOOLS`). `compare_programs` is `return_direct` and runs its
  own graph (`agents/qa/compare_programs/`), which fans out one retrieval per programme with `Send`
  and merges.
- `check_scholarship` reads nothing: the handbook's scholarship rules are code in
  `app/api/v1/scholarships/scholarship_rules.py`, so a rule that changes in the handbook has to
  change there too; `TestRulesMatchTheHandbook` fails when the two drift. A faculty added through
  the API has to be added to the `Faculty` enum in `app/api/v1/scholarships/enums.py` as well, or
  the tool rejects it.
- `list_common_issues` and `report_issue` read and write `reported_issues` through
  `app/api/v1/issues/`. `report_issue` is the QA agent's one write, and it sits behind
  `HumanInTheLoopMiddleware` the way a booking does: the student approves, edits or rejects the
  category and text before anything is filed, and the answer arrives as `decisions`, which skip the
  router. An issue's sort key is its zero-padded votes (`build_issue_votes_key`), shared by the
  seeder and `IssueItem`, because the "most voted" order is read straight off it.

## Tools and prompts

- A tool returns `model_dump()` dicts or a string, never pydantic models: LangChain renders a result
  with `json.dumps` and falls back to `str()`, so a model — or a `datetime.time` — reaches the LLM as
  a Python repr. A tool that also has to write graph state returns a `Command` whose `ToolMessage`
  holds that same JSON — `validate_documents` is the one that does.
- What an agent should do lives in its system prompt; what a tool is lives in the tool. Every rule
  of behaviour — when to call a tool and when not, what to look up or ask first, what to do with
  each kind of result — is in `MAIN_CHAT_PROMPT`, which has one section per QA tool, or in the
  booking prompt, which is laid out by task (finding a time, booking, cancelling, moving). A tool's
  docstring says only what it does and what its result means, a `Field(description=...)` only what
  the argument is and its format, and a result only what happened — none of them names another tool
  or tells the model what to do next. `tests/unit/test_tool_texts.py` holds the part of that a test
  can check, over `ASSISTANT_TOOLS` and `BOOKING_TOOLS`.
- Adding a QA tool means, in `agents/qa/`: input schema in `tool_inputs.py`, the tool and
  `ASSISTANT_TOOLS` in `tools.py`, a section in `MAIN_CHAT_PROMPT` (`prompts.py`); and a stub plus
  cases in `tests/integration/test_tool_routing_eval.py`.

## Applicant documents (`validate_documents`, `applicant_documents/`)

How it works: `docs/applicant-documents.md`. What a change has to keep:

- `validate_documents` takes no arguments from the model: it reads the files off
  `AgentRunContext.documents`, which a message sent with `files` fills. The run context is not
  checkpointed, so the images exist for that one request and are stored nowhere.
- `documents_on_file` is a key of `AssistantAgentState` and of `MainState`; it has to be in both,
  or it does not survive the subagent node. The tool reads it from `runtime.state` and writes it
  back, which is why it returns a `Command` carrying its own `ToolMessage` rather than a dict.
- A turn that carries files skips the router (`route_message` sends it to `qa`). The model's copy
  of the message ends with `ATTACHMENT_LINE_TEMPLATE`, which counts the files and never names them;
  a message that is files alone is that line and nothing else, never an English sentence of ours,
  because the model answers in the language of what it reads. The transcript keeps the text as
  written.
- A failed read is answered with a sentence, never raised.
- Reading and judging stay apart: an `Extractor` (`OpenAIVisionExtractor`, on `VISION_MODEL`)
  reads each image into `DocumentFields`, and `check_applicant_documents` in `document_rules.py`
  judges. The rules must not depend on what read the fields.
- The `issues` codes are a fixed vocabulary, the course's, which its two applicant sets are tested
  against (`build_documents`, `build_broken_documents`), and they are about the documents on file
  only. What became of each file of the message is `file_outcomes`, which never changes `is_valid`.
- `ISSUE_CODES` and the `FileOutcome` enum are the two vocabularies the model reads. A value added
  to either has to be explained in three places: the matching `Verdict` field description, for
  whoever reads the code (the model never sees an output schema); the `validate_documents`
  docstring, where the model learns what it means; and `MAIN_CHAT_PROMPT`, where it learns what to
  tell the applicant. `tests/unit/test_tool_texts.py` checks the first and the last.

## Request path

`router.py` → `ChatService` (`chat_service.py`: one turn, JSON or SSE, and what gets recorded) →
`AgentService` (`agent_service.py`: invoke, resume, stream, validate decisions) plus
`ConversationHistoryService` (`history/`).

- The request is a form, `UserQuery` read through `Form()`: `query`, or `decisions` as a JSON array
  in one field, or `query` with up to `MAX_FILES_PER_MESSAGE` `files`. A JSON body is a 422. The
  same endpoint streams when the request sends `Accept: text/event-stream` (`docs/streaming.md`).
- For a message with files the router refuses a busy or paused thread, then reads the uploads
  (`applicant_documents/document_reader.py`: PNG and JPEG as they are, a PDF rendered to an image,
  anything else a 415) and hands them to the same `run_turn` or `start_streaming_turn`. There is
  one path for a message, with files or without: every step takes `documents`, empty when the
  message carries none. The uploads are read before a stream starts, so a refused file still gets
  its status code.
- PDF pages are drawn in a worker thread behind `PDF_RENDER_LOCK`, because PyMuPDF must not run on
  two threads at once.
- Anything that refuses a turn (paused, busy, bad `decisions`) runs before the stream starts: after
  the first byte the status is already 200.

## Models and caches

- The graph, both agents and the chat model are `lru_cache`d factories. The booking prompt is
  rebuilt per call (it carries today's date); the QA prompt is a constant and knows no date.
- There are two chat models (`graph/models.py`). `get_chat_model` is the one whose answers are
  shown, and it streams. `get_structured_output_model` is for a call whose answer is parsed and
  never shown, the router's, and is built with `disable_streaming=True`, as is the extractor's
  vision model. Under a streamed turn LangGraph makes every model call in the graph stream, and
  `langchain-openai` dumps the events of a streamed structured-output call with pydantic's
  serializer warnings on, which prints one `PydanticSerializationUnexpectedValue` per call. A new
  `with_structured_output` call goes on the second model.
- Subagents are compiled without a checkpointer; the main graph owns the only one. A subagent with
  a saver of its own runs on its own thread, and the pause plumbing breaks.
