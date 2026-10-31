from typing import Annotated, Literal, Self
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel, Field, Json, model_validator

from app.ai_assistant_langchain.history.schemas import Role

DecisionType = Literal['approve', 'edit', 'reject']

ToolArgs = dict[str, str | int]

MAX_QUERY_LENGTH = 128
MAX_FILES_PER_MESSAGE = 5
DEFAULT_DOCUMENTS_QUERY = 'Please check my documents.'
ATTACHMENT_LINE_TEMPLATE = '[Files attached to this message: {file_count}]'


class Decision(BaseModel):
    """What the reviewer does with an action the graph paused on."""

    type: DecisionType
    args: ToolArgs | None = Field(
        default=None,
        description=(
            'Required for `edit`: the **complete** replacement arguments, not a patch — they replace '
            "the model's set outright."
        ),
    )
    message: str | None = Field(
        default=None,
        description='Optional for `reject`: why, in words the model relays to the applicant.',
    )

    @model_validator(mode='after')
    def _check_edit_carries_args(self) -> Self:
        if self.type == 'edit' and not self.args:
            raise ValueError("an 'edit' decision must carry the replacement `args`")
        return self


class UserQuery(BaseModel):
    user_id: UUID
    query: str | None = Field(default=None, min_length=1, max_length=MAX_QUERY_LENGTH)
    decisions: Json[Annotated[list[Decision], Field(min_length=1)]] | None = Field(
        default=None,
        description=(
            'Set instead of `query` to answer a pending approval on this thread. A JSON array in one '
            'form field, and one entry per action in `pending`, in the same order: a single model '
            'turn can request more than one gated action, and they are approved or refused together '
            'or not at all.'
        ),
    )
    files: list[UploadFile] = Field(
        default_factory=list,
        max_length=MAX_FILES_PER_MESSAGE,
        description=(
            'The documents sent with `query`, each a PNG, a JPEG or a PDF. Sent without `query`, '
            'they are a request to check them.'
        ),
    )

    @model_validator(mode='after')
    def _check_exactly_one_input(self) -> Self:
        has_message = self.query is not None or bool(self.files)
        has_decisions = self.decisions is not None
        if has_message == has_decisions:
            raise ValueError('send either `query`, with or without `files`, or `decisions` — not both, and not neither')
        return self


class PendingApproval(BaseModel):
    """One action waiting for a human, as the caller needs to see it to decide."""

    action: str
    args: ToolArgs
    allowed_decisions: list[DecisionType]


class HistoryMessage(BaseModel):
    """One line of a stored conversation, as a client replays it."""

    role: Role
    content: str
    pending: list[PendingApproval] | None = Field(
        default=None,
        description=(
            'Set on the last row when the thread is paused: the actions it is waiting on, in the '
            'same shape `POST /langchain-assistant` returns them. A client that reloaded can then '
            'redraw the approval and answer it. Absent on every ordinary row, so a backend with no '
            'gated actions never sends it and a client that ignores it still reads a correct '
            'transcript.'
        ),
    )


class ChatSchemaOut(BaseModel):
    message: str
    status: Literal['answer', 'pending_approval'] = Field(
        default='answer',
        description=(
            '`pending_approval` means nothing was written: the graph is paused mid-action and waits '
            'for `decisions` on this thread. `message` then describes what is being asked, not an answer.'
        ),
    )
    pending: list[PendingApproval] | None = Field(
        default=None,
        description=(
            'Every action the pause covers, not just the first. One model turn can ask for two — a '
            'reschedule books the new time and releases the old one — and answering only the one you '
            'were shown is how a reviewer approves a write they never saw.'
        ),
    )
