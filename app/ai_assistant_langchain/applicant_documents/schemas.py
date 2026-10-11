from datetime import date

from pydantic import BaseModel, Field

from app.ai_assistant_langchain.applicant_documents.enums import DocumentField, DocumentType, FileOutcome, ImageFormat


class DocumentImage(BaseModel):
    content: bytes = Field(repr=False, description='The image itself.')
    mime_type: ImageFormat = Field(description='The format the image is encoded in.')


class DocumentFields(BaseModel):
    document_type: DocumentType | None = Field(
        description=(
            'The kind of document the image actually shows: a passport, a tax identification card, or a '
            'school-leaving certificate (high-school diploma). Null when it is none of the three.'
        ),
    )
    full_name: str | None = Field(
        default=None,
        description="The holder's full name, exactly as printed.",
    )
    date_of_birth: date | None = Field(
        default=None,
        description="The holder's date of birth.",
    )
    passport_number: str | None = Field(
        default=None,
        description='The passport number, exactly as printed.',
    )
    expiry_date: date | None = Field(
        default=None,
        description='The date the passport expires.',
    )
    tax_id: str | None = Field(
        default=None,
        description='The tax identification number, exactly as printed.',
    )
    gpa: float | None = Field(
        default=None,
        description='The overall GPA the certificate states: 3.8 for "3.8 / 4.0".',
    )
    unreadable_fields: list[DocumentField] = Field(
        default_factory=list,
        description=(
            'The fields above that are printed on the document but cannot be read with confidence: '
            'faint, blurred, cropped or covered.'
        ),
    )


class Verdict(BaseModel):
    is_valid: bool = Field(
        description='True when all three required documents are on file, readable and consistent with each other.',
    )
    issues: list[str] = Field(
        description=(
            'One code per problem with the documents on file: `missing_document:<document>`, '
            '`missing_field:<document>.<field>`, `unreadable:<document>.<field>`, `expired:<document>`, '
            '`name_mismatch`, `date_of_birth_mismatch`.'
        ),
    )
    extracted_fields: list[DocumentFields] = Field(
        description='What was read off each file attached to this message, in the order the files were attached.',
    )
    file_outcomes: list[FileOutcome] = Field(
        description=(
            'What became of each file attached to this message, in the same order as `extracted_fields`: '
            '`added`, the document was not on file and now is; `replaced`, the file took the place of an earlier '
            'file of the same document; `unchanged`, the file reads exactly like the one already on file; '
            '`duplicate`, another file of this message is the same document, so neither was taken; '
            '`unrecognized`, the file is none of the three documents and was not taken.'
        ),
    )
    documents_on_file: list[DocumentFields] = Field(
        description=(
            'The documents the verdict judges: the latest file of each document received in this conversation, '
            'the files of this message included.'
        ),
    )
