from app.ai_assistant_langchain.applicant_documents.document_rules import REQUIRED_FIELDS
from app.ai_assistant_langchain.applicant_documents.enums import DocumentType

DOCUMENT_NAMES: dict[DocumentType, str] = {
    DocumentType.PASSPORT: 'passport',
    DocumentType.TAX_CODE: 'tax identification card',
    DocumentType.CERTIFICATE: 'school-leaving certificate',
}

EXTRACTION_PROMPT_TEMPLATE = """
You are reading one image that an applicant sent to a university admissions office with
their application. It should be one of three documents, and nothing says which.

First decide what the image shows and set `document_type` to that:
- "passport" for a passport;
- "tax_code" for a tax identification card or certificate, the document that carries a
  person's individual tax number;
- "certificate" for a school-leaving certificate or high-school diploma;
- null for anything else: another kind of certificate, another document, or a picture
  that is not a document at all.

Then copy what is printed on it. The fields that are needed depend on the document:
{required_fields}
- anything else: none.

How to copy them:
- Copy each text value exactly as it is printed. Do not correct a spelling, do not
  reorder a name, and do not work a value out from other values: a GPA is the overall
  figure the certificate states, never one computed from the grades.
- A date is the one value you rewrite: give it as year, month, day however the document
  prints it. When it is printed in digits alone and nothing on the document shows which
  number is the day, the day comes first.
- A field that is not printed on the document stays null.
- A field that is printed but that you cannot read with confidence, because it is faint,
  blurred, cropped or covered, stays null and its name goes into `unreadable_fields`.
  Never guess a value.
"""


def build_extraction_prompt() -> str:
    required_fields = '\n'.join(
        f'- a {DOCUMENT_NAMES[document_type]}: {", ".join(f"`{field}`" for field in fields)};'
        for document_type, fields in REQUIRED_FIELDS.items()
    )
    return EXTRACTION_PROMPT_TEMPLATE.format(required_fields=required_fields)
