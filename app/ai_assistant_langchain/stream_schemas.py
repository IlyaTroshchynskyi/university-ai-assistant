from pydantic import BaseModel, Field


class TokenEvent(BaseModel):
    """`event: token` — a piece of the answer, as it is written."""

    text: str = Field(
        description=(
            'The text to append to the assistant bubble. Not a word, and not necessarily small: a '
            '`return_direct` tool sends its whole answer as one of these.'
        ),
    )


class ErrorEvent(BaseModel):
    """`event: error` — the turn failed after the response had already started (§6.3)."""

    detail: str = Field(
        description='What to show the applicant. Never a traceback: this is read by the person who asked.',
    )
