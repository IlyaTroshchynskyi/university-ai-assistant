import pytest

from app.ai_assistant_langchain.agents.qa.tools import _mask_document_identifiers
from app.ai_assistant_langchain.applicant_documents.enums import DocumentField
from tests.factory_models.applicant_factory import (
    CertificateFieldsFactory,
    DocumentFieldsFactory,
    PassportFieldsFactory,
    TaxCodeFieldsFactory,
)


class TestMaskDocumentIdentifiers:
    def test_passport_number_keeps_only_its_last_two_characters(self) -> None:
        masked_fields = _mask_document_identifiers(PassportFieldsFactory.build(passport_number='FA1234567'))

        assert masked_fields.passport_number == '*******67'

    def test_tax_id_keeps_only_its_last_two_characters(self) -> None:
        masked_fields = _mask_document_identifiers(TaxCodeFieldsFactory.build(tax_id='3456789012'))

        assert masked_fields.tax_id == '********12'

    def test_both_identifiers_of_one_document_are_masked(self) -> None:
        fields = DocumentFieldsFactory.build(passport_number='FA1234567', tax_id='3456789012')

        masked_fields = _mask_document_identifiers(fields)

        assert masked_fields.passport_number == '*******67'
        assert masked_fields.tax_id == '********12'

    @pytest.mark.parametrize(
        ('identifier', 'masked_identifier'),
        [('A', '*'), ('A1', '**'), ('A12', '*12'), ('A123', '**23')],
        ids=['one character', 'two characters', 'three characters', 'four characters'],
    )
    def test_identifier_of_two_characters_or_fewer_is_masked_whole_and_a_longer_one_shows_two(
        self, identifier: str, masked_identifier: str
    ) -> None:
        masked_fields = _mask_document_identifiers(PassportFieldsFactory.build(passport_number=identifier))

        assert masked_fields.passport_number == masked_identifier

    def test_identifier_the_document_does_not_carry_stays_absent(self) -> None:
        masked_fields = _mask_document_identifiers(CertificateFieldsFactory.build())

        assert masked_fields.passport_number is None
        assert masked_fields.tax_id is None

    @pytest.mark.parametrize('blank_identifier', ['', '   '], ids=['empty', 'spaces only'])
    def test_blank_identifier_stays_blank(self, blank_identifier: str) -> None:
        masked_fields = _mask_document_identifiers(PassportFieldsFactory.build(passport_number=blank_identifier))

        assert masked_fields.passport_number == blank_identifier

    def test_every_other_field_is_left_as_it_was_read(self) -> None:
        fields = PassportFieldsFactory.build(unreadable_fields=[DocumentField.EXPIRY_DATE])

        masked_fields = _mask_document_identifiers(fields)

        assert masked_fields.model_dump(exclude={'passport_number'}) == fields.model_dump(exclude={'passport_number'})

    def test_fields_passed_in_are_not_changed(self) -> None:
        fields = PassportFieldsFactory.build(passport_number='FA1234567')

        _mask_document_identifiers(fields)

        assert fields.passport_number == 'FA1234567'
