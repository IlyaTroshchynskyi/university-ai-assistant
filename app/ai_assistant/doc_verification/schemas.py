"""Data models for document verification. The vision LLM reads all uploaded documents in
a single call and returns this verdict directly — there is no separate deterministic rules
step or per-field extraction surface anymore."""

from typing import Literal

from pydantic import BaseModel

# The document types an applicant is expected to submit. Fed into the vision prompt so the
# model knows what to look for and which types are required.
DocType = Literal['passport', 'tax_code', 'certificate']


class Verdict(BaseModel):
    """The LLM's judgement over all submitted documents at once: whether they pass, plus a
    ready, applicant-facing message explaining what is right and what is wrong."""

    is_valid: bool
    message: str
