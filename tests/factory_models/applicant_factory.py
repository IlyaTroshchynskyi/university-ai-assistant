from collections.abc import Sequence
from datetime import date

from polyfactory import Use
from polyfactory.factories.pydantic_factory import ModelFactory
import pymupdf

from app.ai_assistant_langchain.applicant_documents.enums import DocumentField, DocumentType, ImageFormat
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentFields, DocumentImage

SAMPLE_DAY = date(2026, 10, 7)
APPLICANT_NAME = 'Maria Kovalenko'
APPLICANT_DATE_OF_BIRTH = date(2007, 3, 14)
BROKEN_SET_ISSUES = ['name_mismatch', 'missing_document:tax_code', 'missing_field:certificate.gpa']

MultipartFiles = list[tuple[str, tuple[str, bytes, str]]]

FILES_FORM_FIELD = 'files'
DOCUMENT_TITLES: dict[DocumentType | None, tuple[str, str]] = {
    DocumentType.PASSPORT: ('REPUBLIC PASSPORT', 'Identity Document'),
    DocumentType.TAX_CODE: ('TAX IDENTIFICATION', 'Individual Tax Number'),
    DocumentType.CERTIFICATE: ('SCHOOL CERTIFICATE', 'High-School Diploma'),
    None: ('LIBRARY CARD', 'City Public Library'),
}
FIELD_LABELS: dict[DocumentField, str] = {
    DocumentField.FULL_NAME: 'FULL NAME',
    DocumentField.DATE_OF_BIRTH: 'DATE OF BIRTH',
    DocumentField.PASSPORT_NUMBER: 'PASSPORT NUMBER',
    DocumentField.EXPIRY_DATE: 'EXPIRY DATE',
    DocumentField.TAX_ID: 'TAX ID',
    DocumentField.GPA: 'OVERALL GPA',
}

PAGE_WIDTH = 600
LEFT_MARGIN = 40
TITLE_HEIGHT = 110
FIELD_HEIGHT = 55
IMAGE_DPI = 150


class DocumentFieldsFactory(ModelFactory[DocumentFields]):
    __model__ = DocumentFields

    document_type: DocumentType | None = None
    full_name: str | None = None
    date_of_birth: date | None = None
    passport_number: str | None = None
    expiry_date: date | None = None
    tax_id: str | None = None
    gpa: float | None = None
    unreadable_fields = Use(list)


class PassportFieldsFactory(DocumentFieldsFactory):
    document_type = DocumentType.PASSPORT
    full_name = APPLICANT_NAME
    date_of_birth = APPLICANT_DATE_OF_BIRTH
    passport_number = 'FA1234567'
    expiry_date = date(2031, 6, 10)


class TaxCodeFieldsFactory(DocumentFieldsFactory):
    document_type = DocumentType.TAX_CODE
    full_name = APPLICANT_NAME
    date_of_birth = APPLICANT_DATE_OF_BIRTH
    tax_id = '3456789012'


class CertificateFieldsFactory(DocumentFieldsFactory):
    document_type = DocumentType.CERTIFICATE
    full_name = APPLICANT_NAME
    gpa = 3.8


DOCUMENT_FACTORIES: dict[DocumentType, type[DocumentFieldsFactory]] = {
    DocumentType.PASSPORT: PassportFieldsFactory,
    DocumentType.TAX_CODE: TaxCodeFieldsFactory,
    DocumentType.CERTIFICATE: CertificateFieldsFactory,
}


def build_documents(**overrides: DocumentFields | None) -> list[DocumentFields]:
    documents: dict[DocumentType, DocumentFields | None] = {
        document_type: factory.build() for document_type, factory in DOCUMENT_FACTORIES.items()
    }
    for document_type, fields in overrides.items():
        documents[DocumentType(document_type)] = fields
    return [fields for fields in documents.values() if fields is not None]


def build_broken_documents() -> list[DocumentFields]:
    return build_documents(
        tax_code=None,
        certificate=CertificateFieldsFactory.build(full_name='Marie Kovalenko', gpa=None),
    )


def build_document_image(fields: DocumentFields) -> DocumentImage:
    printed_fields = [
        (FIELD_LABELS[field], _format_printed_value(field, field_value))
        for field in DocumentField
        if (field_value := getattr(fields, field)) is not None
    ]

    pdf = pymupdf.open()
    page = pdf.new_page(width=PAGE_WIDTH, height=TITLE_HEIGHT + FIELD_HEIGHT * len(printed_fields) + LEFT_MARGIN)
    title, subtitle = DOCUMENT_TITLES[fields.document_type]
    page.insert_text((LEFT_MARGIN, 55), title, fontsize=24)
    page.insert_text((LEFT_MARGIN, 80), subtitle, fontsize=12)
    for row, (label, printed_value) in enumerate(printed_fields):
        row_top = TITLE_HEIGHT + FIELD_HEIGHT * row
        page.insert_text((LEFT_MARGIN, row_top + 15), label, fontsize=9)
        page.insert_text((LEFT_MARGIN, row_top + 38), printed_value, fontsize=16)

    return DocumentImage(content=bytes(page.get_pixmap(dpi=IMAGE_DPI).tobytes('png')), mime_type=ImageFormat.PNG)


def build_document_images(documents: Sequence[DocumentFields]) -> list[DocumentImage]:
    return [build_document_image(fields) for fields in documents]


def build_document_files(documents: Sequence[DocumentFields]) -> MultipartFiles:
    return [
        (FILES_FORM_FIELD, (f'scan_{position}.png', image.content, image.mime_type.value))
        for position, image in enumerate(build_document_images(documents), start=1)
    ]


def _format_printed_value(field: DocumentField, field_value: str | date | float) -> str:
    if field is DocumentField.GPA:
        return f'{field_value} / 4.0'
    if isinstance(field_value, date):
        return field_value.isoformat()
    return str(field_value)
