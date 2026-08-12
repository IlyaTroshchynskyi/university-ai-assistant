"""Document verification in a single vision call. The LLM sees every uploaded document at
once, self-identifies each one, checks it against the admission criteria (including
cross-document checks), and returns a Verdict with a ready applicant-facing message. No
per-document extraction, no separate deterministic rules step, no separate explain step."""

from datetime import date

from app.ai_assistant.doc_verification.schemas import Verdict
from app.settings import build_llm


def _image_data_uri(image_b64: str) -> str:
    """Base64 data URI, accepted as an image_url by crewai/litellm."""
    return f'data:image/jpeg;base64,{image_b64}'


def _criteria_prompt() -> str:
    """The full instruction paired with the document images: what to look for, which rules
    make the submission valid, and how to answer. Today's date is injected so the model can
    judge expiry."""
    return (
        "You are a university admissions assistant validating an applicant's uploaded "
        'documents. You are shown one or more document images. Identify each document '
        'yourself and judge the whole submission against these rules:\n'
        '\n'
        'Required documents (ALL must be present among the images):\n'
        '- passport: must show full name, date of birth, and passport number.\n'
        '- tax code / taxpayer document: must show the tax id number.\n'
        '- certificate (school certificate): must show full name, certificate type, and GPA.\n'
        '\n'
        'Cross-document rules:\n'
        '- The full name on the passport and on the certificate must match.\n'
        f'- The passport must not be expired (today is {date.today().isoformat()}).\n'
        '\n'
        'Judging:\n'
        '- A required document that is missing, unreadable, or of the wrong type makes the '
        'submission invalid.\n'
        '- A required field you cannot actually read makes the submission invalid. Never '
        'guess or invent values.\n'
        '\n'
        'Set is_valid to true only if every rule above is satisfied. In message, write a '
        'short, friendly note to the applicant explaining what is right and what is wrong '
        '(name each concrete problem in plain language). If everything passes, congratulate '
        'them briefly. Do not invent problems beyond the rules above.'
    )


async def verify(images_b64: list[str]) -> Verdict:
    """Validate all uploaded documents in one vision call and return the verdict.

    ``images_b64`` are base64-encoded image bytes (kept as strings so they survive the
    flow's JSON state persistence)."""
    if not images_b64:
        return Verdict(is_valid=False, message='No documents were uploaded, so there is nothing to verify.')

    content: list[dict[str, object]] = [{'type': 'text', 'text': _criteria_prompt()}]
    content += [{'type': 'image_url', 'image_url': {'url': _image_data_uri(img)}} for img in images_b64]
    llm = build_llm()
    return await llm.acall(messages=[{'role': 'user', 'content': content}], response_model=Verdict)
