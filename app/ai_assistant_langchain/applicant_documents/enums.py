from enum import StrEnum


class DocumentType(StrEnum):
    PASSPORT = 'passport'
    TAX_CODE = 'tax_code'
    CERTIFICATE = 'certificate'


class DocumentField(StrEnum):
    FULL_NAME = 'full_name'
    DATE_OF_BIRTH = 'date_of_birth'
    PASSPORT_NUMBER = 'passport_number'
    EXPIRY_DATE = 'expiry_date'
    TAX_ID = 'tax_id'
    GPA = 'gpa'


class FileOutcome(StrEnum):
    ADDED = 'added'
    REPLACED = 'replaced'
    UNCHANGED = 'unchanged'
    DUPLICATE = 'duplicate'
    UNRECOGNIZED = 'unrecognized'


class ImageFormat(StrEnum):
    PNG = 'image/png'
    JPEG = 'image/jpeg'
