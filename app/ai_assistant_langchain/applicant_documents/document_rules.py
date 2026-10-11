from collections.abc import Mapping, Sequence
from datetime import date
import re
from typing import TypeAlias

from app.ai_assistant_langchain.applicant_documents.enums import DocumentField, DocumentType, FileOutcome
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentFields, Verdict

FieldValue: TypeAlias = str | date | float

REQUIRED_FIELDS: dict[DocumentType, tuple[DocumentField, ...]] = {
    DocumentType.PASSPORT: (
        DocumentField.FULL_NAME,
        DocumentField.DATE_OF_BIRTH,
        DocumentField.PASSPORT_NUMBER,
        DocumentField.EXPIRY_DATE,
    ),
    DocumentType.TAX_CODE: (
        DocumentField.FULL_NAME,
        DocumentField.DATE_OF_BIRTH,
        DocumentField.TAX_ID,
    ),
    DocumentType.CERTIFICATE: (
        DocumentField.FULL_NAME,
        DocumentField.GPA,
    ),
}

MISSING_DOCUMENT_ISSUE = 'missing_document'
MISSING_FIELD_ISSUE = 'missing_field'
UNREADABLE_ISSUE = 'unreadable'
EXPIRED_ISSUE = 'expired'
MISMATCH_ISSUES: dict[DocumentField, str] = {
    DocumentField.FULL_NAME: 'name_mismatch',
    DocumentField.DATE_OF_BIRTH: 'date_of_birth_mismatch',
}
ISSUE_CODES = (
    MISSING_DOCUMENT_ISSUE,
    MISSING_FIELD_ISSUE,
    UNREADABLE_ISSUE,
    EXPIRED_ISSUE,
    *MISMATCH_ISSUES.values(),
)
TAKEN_FILE_OUTCOMES = (FileOutcome.ADDED, FileOutcome.REPLACED)


def check_applicant_documents(
    attached_documents: Sequence[DocumentFields],
    today: date,
    documents_on_file: Sequence[DocumentFields] = (),
) -> Verdict:
    earlier_documents = {fields.document_type: fields for fields in documents_on_file}
    file_outcomes = [_pick_file_outcome(fields, attached_documents, earlier_documents) for fields in attached_documents]
    taken_documents = {
        fields.document_type: fields
        for fields, outcome in zip(attached_documents, file_outcomes, strict=True)
        if outcome in TAKEN_FILE_OUTCOMES
    }
    merged_documents = {**earlier_documents, **taken_documents}
    updated_documents = {
        document_type: merged_documents[document_type]
        for document_type in DocumentType
        if document_type in merged_documents
    }

    issues = _check_cross_document_match(updated_documents)
    for document_type in DocumentType:
        issues.extend(_check_document(document_type, updated_documents.get(document_type), today))

    return Verdict(
        is_valid=not issues,
        issues=issues,
        extracted_fields=list(attached_documents),
        file_outcomes=file_outcomes,
        documents_on_file=list(updated_documents.values()),
    )


def _pick_file_outcome(
    fields: DocumentFields,
    attached_documents: Sequence[DocumentFields],
    earlier_documents: Mapping[DocumentType | None, DocumentFields],
) -> FileOutcome:
    if fields.document_type is None:
        return FileOutcome.UNRECOGNIZED

    copies = [other for other in attached_documents if other.document_type is fields.document_type]
    if len(copies) > 1:
        return FileOutcome.DUPLICATE

    earlier_document = earlier_documents.get(fields.document_type)
    if earlier_document is None:
        return FileOutcome.ADDED
    return FileOutcome.UNCHANGED if earlier_document == fields else FileOutcome.REPLACED


def _check_cross_document_match(documents: Mapping[DocumentType, DocumentFields]) -> list[str]:
    issues: list[str] = []
    for field, issue in MISMATCH_ISSUES.items():
        read_values = {
            _normalize_field_value(field_value)
            for fields in documents.values()
            if (field_value := _find_read_field_value(fields, field)) is not None
        }
        if len(read_values) > 1:
            issues.append(issue)
    return issues


def _check_document(document_type: DocumentType, fields: DocumentFields | None, today: date) -> list[str]:
    if fields is None:
        return [f'{MISSING_DOCUMENT_ISSUE}:{document_type}']

    issues: list[str] = []
    for field in REQUIRED_FIELDS[document_type]:
        if field in fields.unreadable_fields:
            issues.append(f'{UNREADABLE_ISSUE}:{document_type}.{field}')
        elif _find_read_field_value(fields, field) is None:
            issues.append(f'{MISSING_FIELD_ISSUE}:{document_type}.{field}')

    if DocumentField.EXPIRY_DATE in REQUIRED_FIELDS[document_type]:
        expiry_date = _find_read_field_value(fields, DocumentField.EXPIRY_DATE)
        if isinstance(expiry_date, date) and expiry_date < today:
            issues.append(f'{EXPIRED_ISSUE}:{document_type}')
    return issues


def _find_read_field_value(fields: DocumentFields, field: DocumentField) -> FieldValue | None:
    if field in fields.unreadable_fields:
        return None
    field_value: FieldValue | None = getattr(fields, field)
    if isinstance(field_value, str) and not field_value.strip():
        return None
    return field_value


def _normalize_field_value(field_value: FieldValue) -> FieldValue:
    if isinstance(field_value, str):
        return ' '.join(sorted(re.findall(r'\w+', field_value.casefold())))
    return field_value
