# Checking an applicant's documents

How the LangGraph assistant checks the passport, the tax code and the school certificate an
applicant sends. The code is `app/ai_assistant_langchain/applicant_documents/`; the tool is
`validate_documents` in the QA agent.

## Sending files

A message can carry files: the same `POST /langchain-assistant`, with up to `MAX_FILES_PER_MESSAGE`
parts named `files` next to `user_id` and `query`, each a PNG, a JPEG or a PDF of at most
`MAX_DOCUMENT_MB`.

```bash
curl -X POST http://127.0.0.1:8000/langchain-assistant \
  -F user_id=11111111-1111-1111-1111-111111111111 \
  -F files=@passport.png -F files=@tax_code.png -F files=@certificate.pdf
```

Nothing says which file is which document: the assistant tells them apart by reading them. They can
come as a pack or one message at a time: what is read off a file is kept for the rest of the
conversation, and the verdict is always about everything received so far. A PDF is read by its
first page, drawn at `PDF_RENDER_DPI` and never more than `MAX_RENDERED_SIDE_PX` on its longer
side, so a page of any declared size costs a bounded amount of memory.

Without a `query` the transcript records the turn as `DEFAULT_DOCUMENTS_QUERY`. The model is not
shown that sentence: it is ours, in English, and the model answers in the language of the last
message it reads. The answer is the usual `ChatSchemaOut`, and it streams like any other turn.
Files cannot travel with `decisions`.

A message with files skips the router and goes to the QA agent, the only one that can read them.
The model's copy of the message ends with `ATTACHMENT_LINE_TEMPLATE`, a line counting the files,
because nothing else tells it that an "is it fine now?" arrived with a new certificate. A message
of files alone is that line and nothing else, and the prompt tells the model to keep answering in
the language the applicant last wrote in. The line carries a count, not the file names, which are
the applicant's own text. The transcript keeps the message as written.

## What is kept

**The files are not stored; what was read off them is, for the conversation.** The images travel
in the graph's run context (`AgentRunContext.documents`) for that one request, which is where
`validate_documents` reads them. The fields it read are kept in the graph state as
`documents_on_file`, one entry per document, the latest file of each, so they are checkpointed with
the rest of the thread. That lets an applicant send the passport now and the certificate two
messages later, or ask "what is still missing?" with no file at all. A file sent later replaces the
earlier file of the same document, so a sharper scan corrects a blurred one. A new conversation
starts with nothing on file.

A passport number and a tax id are masked but for their last two characters (`*******67`) before
they are kept or shown to the model: the rules only need to know they were there, and two
characters are enough for an applicant to recognise the document.

## Two stages

Checking is two stages that do not know about each other:

1. **Extraction** turns each image into `DocumentFields`, the first of which is `document_type`:
   a passport, a tax code, a certificate, or none of the three. `Extractor` in `extractors.py` is
   the interface; `OpenAIVisionExtractor` is the one implementation, a multimodal call to
   `VISION_MODEL` with a structured answer. All files of a message are read at once, one call each.
2. **Validation** is code, `check_applicant_documents` in `document_rules.py`: the same rules
   whatever read the fields. It files each document of the message under the `document_type` it was
   read as, in place of an earlier one, and judges the documents on file.

Keep the two apart: the rules must not depend on what read the fields.

## The verdict

`is_valid` plus `issues`, one code per problem with the documents on file. The codes are
`ISSUE_CODES` in `document_rules.py`, in the vocabulary of the course's two sample sets: a missing
document, a missing or unreadable field, an expired passport, and a name or a date of birth on which
the documents disagree.

A passport has to show a full name, a date of birth, a number and an expiry date; a tax code a full
name, a date of birth and a tax id; a certificate a full name and a GPA.

Two names are the same when they are made of the same words: case, punctuation and word order do not
count, so `KOVALENKO, MARIA` matches `Maria Kovalenko`. A word one document has and another lacks,
such as a middle name or a patronymic, is still a mismatch. A date printed in digits alone, with
nothing to show which number is the day, is read day first.

What became of each file of the message is a second list, `file_outcomes`, in the order attached:
the values of the `FileOutcome` enum (`enums.py`): added, replaced, unchanged, a duplicate of
another file in the same message, or not one of the three documents. An outcome is not an issue:
`is_valid` is about the documents on file alone, so a stray screenshot after three good documents is
mentioned, not held against them. The outcomes are what let the assistant say "I already had this
passport" instead of repeating the verdict. A replacement counts even when it is worse: a
certificate with another name on it, sent after a good one, makes the set invalid.

The verdict also carries `documents_on_file`, what it judged, and `extracted_fields`, what was read
off each file of this message, in the order of the outcomes.

`ISSUE_CODES` and `FileOutcome` are the two vocabularies the model reads. A value added to either is
explained in the matching `Verdict` field description and in `MAIN_CHAT_PROMPT`;
`tests/unit/test_tool_texts.py` fails until it is.

## Refusals and failures

- A file that is not a PNG, a JPEG or a PDF is a `415` naming the file, and so is a PDF that cannot
  be opened, is password-protected or has no pages. A file over the size limit is a `413`, one file
  too many a `422`. A busy or paused thread is a `409` before any file is read. All of it happens
  before anything reaches the model, and before a stream starts.
- The size limit is per file and checked after the request has arrived, so the total size of a
  request is whatever the server in front of the app allows.
- PDF pages are drawn one at a time, in a worker thread behind `PDF_RENDER_LOCK`, because PyMuPDF
  must not run on two threads at once: a second PDF waits for the first.
- When a file cannot be read (the vision model timed out or refused the image), the tool answers
  with a sentence saying nothing was checked, instead of failing the turn, and the assistant asks
  for the files again. One unread file leaves all files of that message unread, since the reads run
  together, and what was on file stays as it was.

## Test data

No applicant's document is in the repository. The two sets the course describes, a valid one and a
broken one with a misspelt name, no tax code and no GPA, are built by
`tests/factory_models/applicant_factory.py`, which also draws each document as an image when a test
needs one. Every `pytest` checks the rules against both sets; `make eval-documents` has the real
vision model read the five images, which costs money.
