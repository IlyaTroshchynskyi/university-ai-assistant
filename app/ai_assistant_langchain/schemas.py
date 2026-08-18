from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

DecisionType = Literal['approve', 'edit', 'reject']

ToolArgs = dict[str, str | int]


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
    def _edit_carries_args(self) -> Self:
        if self.type == 'edit' and not self.args:
            raise ValueError("an 'edit' decision must carry the replacement `args`")
        return self


class UserQuery(BaseModel):
    user_id: UUID
    query: str | None = Field(default=None, min_length=1, max_length=128)
    decisions: list[Decision] | None = Field(
        default=None,
        min_length=1,
        description=(
            'Set instead of `query` to answer a pending approval on this thread. A list, and one '
            'entry per action in `pending`, in the same order: a single model turn can request more '
            'than one gated action, and they are approved or refused together or not at all.'
        ),
    )

    @model_validator(mode='after')
    def _exactly_one(self) -> Self:
        both_missing = self.query is None and self.decisions is None
        both_given = self.query is not None and self.decisions is not None
        if both_missing or both_given:
            raise ValueError('send either `query` or `decisions` — not both, and not neither')
        return self


class PendingApproval(BaseModel):
    """One action waiting for a human, as the caller needs to see it to decide."""

    action: str
    args: ToolArgs
    allowed_decisions: list[DecisionType]


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
