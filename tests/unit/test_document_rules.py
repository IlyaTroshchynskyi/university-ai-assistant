from datetime import date, timedelta

import pytest

from app.ai_assistant_langchain.applicant_documents.document_rules import check_applicant_documents
from app.ai_assistant_langchain.applicant_documents.enums import DocumentField, DocumentType, FileOutcome
from tests.factories.applicant_factory import (
    BROKEN_SET_ISSUES,
    build_broken_documents,
    build_documents,
    CertificateFieldsFactory,
    DOCUMENT_FACTORIES,
    DocumentFieldsFactory,
    PassportFieldsFactory,
    SAMPLE_DAY,
    TaxCodeFieldsFactory,
)

TODAY = SAMPLE_DAY


class TestCompleteApplication:
    def test_three_consistent_documents_are_valid(self) -> None:
        verdict = check_applicant_documents(build_documents(), today=TODAY)

        assert verdict.is_valid is True
        assert verdict.issues == []

    def test_verdict_carries_what_was_read_off_each_file_in_the_order_attached(self) -> None:
        documents = build_documents()

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.extracted_fields == documents


class TestMissingDocument:
    @pytest.mark.parametrize('document_type', list(DocumentType))
    def test_document_that_was_not_attached_is_missing(self, document_type: DocumentType) -> None:
        verdict = check_applicant_documents(build_documents(**{document_type: None}), today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == [f'missing_document:{document_type}']

    def test_nothing_attached_misses_all_three_in_order(self) -> None:
        verdict = check_applicant_documents([], today=TODAY)

        assert verdict.issues == [
            'missing_document:passport',
            'missing_document:tax_code',
            'missing_document:certificate',
        ]

    def test_documents_are_told_apart_by_what_they_are_not_by_the_order_they_came_in(self) -> None:
        documents = list(reversed(build_documents()))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is True
        assert verdict.extracted_fields == documents
        assert verdict.documents_on_file == build_documents()


class TestDuplicateDocument:
    def test_two_files_of_one_document_in_a_message_are_duplicates_and_neither_is_taken(self) -> None:
        documents = [*build_documents(), PassportFieldsFactory.build()]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.file_outcomes == [
            FileOutcome.DUPLICATE,
            FileOutcome.ADDED,
            FileOutcome.ADDED,
            FileOutcome.DUPLICATE,
        ]
        assert verdict.documents_on_file == build_documents(passport=None)

    def test_document_sent_only_as_duplicates_is_still_missing(self) -> None:
        documents = [*build_documents(), PassportFieldsFactory.build()]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['missing_document:passport']

    def test_every_copy_of_a_document_is_a_duplicate(self) -> None:
        documents = [TaxCodeFieldsFactory.build(), TaxCodeFieldsFactory.build(), TaxCodeFieldsFactory.build()]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.file_outcomes == [FileOutcome.DUPLICATE] * 3

    def test_duplicated_document_is_not_checked_field_by_field(self) -> None:
        documents = [
            *build_documents(),
            PassportFieldsFactory.build(passport_number=None, expiry_date=TODAY - timedelta(days=1)),
        ]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_document:passport']

    def test_duplicated_document_takes_no_part_in_the_name_match(self) -> None:
        documents = [*build_documents(), CertificateFieldsFactory.build(full_name='Somebody Else')]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_document:certificate']

    def test_file_sent_in_place_of_another_document_leaves_both_documents_missing(self) -> None:
        documents = build_documents(certificate=PassportFieldsFactory.build())

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_document:passport', 'missing_document:certificate']

    def test_every_copy_is_still_reported_as_extracted(self) -> None:
        second_passport = PassportFieldsFactory.build(full_name='Somebody Else')
        documents = [*build_documents(), second_passport]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.extracted_fields == documents

    def test_duplicates_leave_the_document_already_on_file_in_its_place(self) -> None:
        documents_on_file = build_documents()
        attached_documents = [
            PassportFieldsFactory.build(full_name='Somebody Else'),
            PassportFieldsFactory.build(full_name='Somebody Else'),
        ]

        verdict = check_applicant_documents(attached_documents, today=TODAY, documents_on_file=documents_on_file)

        assert verdict.is_valid is True
        assert verdict.file_outcomes == [FileOutcome.DUPLICATE, FileOutcome.DUPLICATE]
        assert verdict.documents_on_file == documents_on_file


class TestUnrecognizedDocument:
    def test_file_that_is_none_of_the_three_documents_is_unrecognized(self) -> None:
        documents = [*build_documents(), DocumentFieldsFactory.build()]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.file_outcomes[-1] is FileOutcome.UNRECOGNIZED

    def test_unrecognized_file_does_not_spoil_three_documents_that_are_in_order(self) -> None:
        documents = [*build_documents(), DocumentFieldsFactory.build()]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is True
        assert verdict.issues == []

    def test_unrecognized_file_sent_in_place_of_a_document_leaves_that_document_missing(self) -> None:
        documents = build_documents(tax_code=DocumentFieldsFactory.build())

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['missing_document:tax_code']
        assert verdict.file_outcomes == [FileOutcome.ADDED, FileOutcome.UNRECOGNIZED, FileOutcome.ADDED]

    def test_unrecognized_file_takes_no_part_in_the_name_match(self) -> None:
        documents = [*build_documents(), DocumentFieldsFactory.build(full_name='Somebody Else')]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_unrecognized_file_is_reported_as_extracted_but_not_put_on_file(self) -> None:
        library_card = DocumentFieldsFactory.build(full_name='Somebody Else')
        documents = [*build_documents(), library_card]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.extracted_fields == documents
        assert verdict.documents_on_file == build_documents()


class TestFileOutcomes:
    def test_document_that_was_not_on_file_is_added(self) -> None:
        verdict = check_applicant_documents([PassportFieldsFactory.build()], today=TODAY)

        assert verdict.file_outcomes == [FileOutcome.ADDED]

    def test_file_that_reads_differently_from_the_one_on_file_replaces_it(self) -> None:
        documents_on_file = build_documents()
        renewed_passport = PassportFieldsFactory.build(expiry_date=date(2036, 6, 10))

        verdict = check_applicant_documents([renewed_passport], today=TODAY, documents_on_file=documents_on_file)

        assert verdict.file_outcomes == [FileOutcome.REPLACED]
        assert verdict.documents_on_file == build_documents(passport=renewed_passport)

    def test_file_that_reads_like_the_one_on_file_changes_nothing(self) -> None:
        documents_on_file = build_documents()

        verdict = check_applicant_documents(
            [PassportFieldsFactory.build()], today=TODAY, documents_on_file=documents_on_file
        )

        assert verdict.file_outcomes == [FileOutcome.UNCHANGED]
        assert verdict.is_valid is True
        assert verdict.documents_on_file == documents_on_file

    def test_replacement_that_is_worse_than_the_one_on_file_still_replaces_it(self) -> None:
        documents_on_file = build_documents()
        certificate_without_gpa = CertificateFieldsFactory.build(full_name='Marie Kovalenko', gpa=None)

        verdict = check_applicant_documents([certificate_without_gpa], today=TODAY, documents_on_file=documents_on_file)

        assert verdict.file_outcomes == [FileOutcome.REPLACED]
        assert verdict.is_valid is False
        assert verdict.issues == ['name_mismatch', 'missing_field:certificate.gpa']

    def test_outcomes_follow_the_order_the_files_were_attached_in(self) -> None:
        documents_on_file = [PassportFieldsFactory.build(), TaxCodeFieldsFactory.build()]
        attached_documents = [
            DocumentFieldsFactory.build(),
            CertificateFieldsFactory.build(),
            TaxCodeFieldsFactory.build(tax_id='9999999999'),
            PassportFieldsFactory.build(),
        ]

        verdict = check_applicant_documents(attached_documents, today=TODAY, documents_on_file=documents_on_file)

        assert verdict.file_outcomes == [
            FileOutcome.UNRECOGNIZED,
            FileOutcome.ADDED,
            FileOutcome.REPLACED,
            FileOutcome.UNCHANGED,
        ]

    def test_message_without_files_has_no_outcomes(self) -> None:
        verdict = check_applicant_documents([], today=TODAY, documents_on_file=build_documents())

        assert verdict.file_outcomes == []


class TestDocumentsOnFile:
    def test_document_sent_alone_is_put_on_file_and_the_other_two_are_missing(self) -> None:
        passport = PassportFieldsFactory.build()

        verdict = check_applicant_documents([passport], today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['missing_document:tax_code', 'missing_document:certificate']
        assert verdict.documents_on_file == [passport]

    def test_document_sent_later_joins_the_ones_on_file(self) -> None:
        passport, tax_code = PassportFieldsFactory.build(), TaxCodeFieldsFactory.build()

        verdict = check_applicant_documents([tax_code], today=TODAY, documents_on_file=[passport])

        assert verdict.issues == ['missing_document:certificate']
        assert verdict.documents_on_file == [passport, tax_code]

    def test_last_document_completes_a_valid_set(self) -> None:
        documents_on_file = build_documents(certificate=None)

        verdict = check_applicant_documents(
            [CertificateFieldsFactory.build()], today=TODAY, documents_on_file=documents_on_file
        )

        assert verdict.is_valid is True
        assert verdict.issues == []
        assert verdict.documents_on_file == build_documents()

    def test_documents_on_file_are_judged_when_nothing_is_attached(self) -> None:
        verdict = check_applicant_documents([], today=TODAY, documents_on_file=build_documents())

        assert verdict.is_valid is True
        assert verdict.extracted_fields == []
        assert verdict.documents_on_file == build_documents()

    def test_later_file_of_a_document_replaces_the_one_on_file(self) -> None:
        documents_on_file = build_documents(certificate=CertificateFieldsFactory.build(gpa=None))
        readable_certificate = CertificateFieldsFactory.build()

        verdict = check_applicant_documents([readable_certificate], today=TODAY, documents_on_file=documents_on_file)

        assert verdict.is_valid is True
        assert verdict.documents_on_file == build_documents()

    def test_document_attached_is_matched_against_the_ones_on_file(self) -> None:
        certificate = CertificateFieldsFactory.build(full_name='Marie Kovalenko')

        verdict = check_applicant_documents(
            [certificate], today=TODAY, documents_on_file=[PassportFieldsFactory.build()]
        )

        assert verdict.issues == ['name_mismatch', 'missing_document:tax_code']

    def test_documents_on_file_are_listed_in_document_order_whatever_order_they_arrived_in(self) -> None:
        passport, certificate = PassportFieldsFactory.build(), CertificateFieldsFactory.build()

        verdict = check_applicant_documents([passport], today=TODAY, documents_on_file=[certificate])

        assert verdict.documents_on_file == [passport, certificate]

    def test_only_the_files_of_this_message_are_reported_as_extracted(self) -> None:
        tax_code = TaxCodeFieldsFactory.build()

        verdict = check_applicant_documents([tax_code], today=TODAY, documents_on_file=[PassportFieldsFactory.build()])

        assert verdict.extracted_fields == [tax_code]

    def test_document_on_file_is_judged_on_the_day_of_the_check_not_of_its_arrival(self) -> None:
        documents_on_file = build_documents(passport=PassportFieldsFactory.build(expiry_date=TODAY))

        verdict = check_applicant_documents([], today=TODAY + timedelta(days=1), documents_on_file=documents_on_file)

        assert verdict.issues == ['expired:passport']


class TestRequiredFields:
    @pytest.mark.parametrize(
        ('document_type', 'field'),
        [
            (DocumentType.PASSPORT, DocumentField.FULL_NAME),
            (DocumentType.PASSPORT, DocumentField.DATE_OF_BIRTH),
            (DocumentType.PASSPORT, DocumentField.PASSPORT_NUMBER),
            (DocumentType.PASSPORT, DocumentField.EXPIRY_DATE),
            (DocumentType.TAX_CODE, DocumentField.FULL_NAME),
            (DocumentType.TAX_CODE, DocumentField.DATE_OF_BIRTH),
            (DocumentType.TAX_CODE, DocumentField.TAX_ID),
            (DocumentType.CERTIFICATE, DocumentField.FULL_NAME),
            (DocumentType.CERTIFICATE, DocumentField.GPA),
        ],
    )
    def test_required_field_that_is_absent_is_a_missing_field(
        self, document_type: DocumentType, field: DocumentField
    ) -> None:
        documents = build_documents(**{document_type: DOCUMENT_FACTORIES[document_type].build(**{field: None})})

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == [f'missing_field:{document_type}.{field}']

    def test_blank_text_is_a_missing_field(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(passport_number='   '))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_field:passport.passport_number']

    def test_field_printed_but_illegible_is_unreadable(self) -> None:
        documents = build_documents(
            certificate=CertificateFieldsFactory.build(gpa=None, unreadable_fields=[DocumentField.GPA]),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['unreadable:certificate.gpa']

    def test_value_given_for_a_field_flagged_unreadable_is_not_trusted(self) -> None:
        documents = build_documents(
            tax_code=TaxCodeFieldsFactory.build(tax_id='3456789012', unreadable_fields=[DocumentField.TAX_ID]),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['unreadable:tax_code.tax_id']

    def test_field_the_document_does_not_require_may_be_absent(self) -> None:
        documents = build_documents(certificate=CertificateFieldsFactory.build(date_of_birth=None, expiry_date=None))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_unreadable_field_the_document_does_not_require_is_ignored(self) -> None:
        documents = build_documents(
            certificate=CertificateFieldsFactory.build(unreadable_fields=[DocumentField.DATE_OF_BIRTH]),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_several_fields_are_reported_in_the_order_the_document_lists_them(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(passport_number=None, full_name=None))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_field:passport.full_name', 'missing_field:passport.passport_number']


class TestNameMatch:
    def test_name_spelled_differently_on_one_document_is_a_mismatch(self) -> None:
        documents = build_documents(certificate=CertificateFieldsFactory.build(full_name='Marie Kovalenko'))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['name_mismatch']

    def test_three_different_names_are_one_mismatch(self) -> None:
        documents = build_documents(
            tax_code=TaxCodeFieldsFactory.build(full_name='Mariya Kovalenko'),
            certificate=CertificateFieldsFactory.build(full_name='Marie Kovalenko'),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['name_mismatch']

    def test_letter_case_and_spacing_do_not_make_a_mismatch(self) -> None:
        documents = build_documents(certificate=CertificateFieldsFactory.build(full_name='  MARIA   kovalenko '))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    @pytest.mark.parametrize(
        'passport_name',
        ['KOVALENKO MARIA', 'Kovalenko, Maria', 'Kovalenko Maria.'],
        ids=['surname first', 'comma between', 'trailing dot'],
    )
    def test_word_order_and_punctuation_do_not_make_a_mismatch(self, passport_name: str) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(full_name=passport_name))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_name_with_a_word_the_others_lack_is_a_mismatch(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(full_name='Maria Ivanivna Kovalenko'))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['name_mismatch']

    def test_same_letters_split_into_other_words_are_a_mismatch(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(full_name='Kovalenkomaria'))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['name_mismatch']

    def test_document_without_a_name_is_a_missing_field_not_a_mismatch(self) -> None:
        documents = build_documents(tax_code=TaxCodeFieldsFactory.build(full_name=None))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['missing_field:tax_code.full_name']

    def test_name_flagged_unreadable_takes_no_part_in_the_match(self) -> None:
        documents = build_documents(
            tax_code=TaxCodeFieldsFactory.build(
                full_name='Mar1a K0valenko', unreadable_fields=[DocumentField.FULL_NAME]
            ),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['unreadable:tax_code.full_name']


class TestDateOfBirthMatch:
    def test_different_dates_of_birth_are_a_mismatch(self) -> None:
        documents = build_documents(tax_code=TaxCodeFieldsFactory.build(date_of_birth=date(2007, 3, 15)))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['date_of_birth_mismatch']

    def test_document_without_a_date_of_birth_takes_no_part_in_the_match(self) -> None:
        documents = build_documents(certificate=CertificateFieldsFactory.build(date_of_birth=None))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_name_mismatch_is_reported_before_the_date_of_birth_one(self) -> None:
        documents = build_documents(
            tax_code=TaxCodeFieldsFactory.build(full_name='Marie Kovalenko', date_of_birth=date(2007, 3, 15)),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == ['name_mismatch', 'date_of_birth_mismatch']


class TestPassportExpiry:
    def test_passport_past_its_expiry_date_is_expired(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(expiry_date=TODAY - timedelta(days=1)))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == ['expired:passport']

    def test_passport_is_still_good_on_its_expiry_date(self) -> None:
        documents = build_documents(passport=PassportFieldsFactory.build(expiry_date=TODAY))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []

    def test_expiry_date_on_a_document_that_does_not_expire_is_ignored(self) -> None:
        documents = build_documents(certificate=CertificateFieldsFactory.build(expiry_date=TODAY - timedelta(days=1)))

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == []


class TestIssueOrder:
    def test_mismatches_come_first_then_each_document_in_turn(self) -> None:
        documents = [
            DocumentFieldsFactory.build(),
            CertificateFieldsFactory.build(),
            CertificateFieldsFactory.build(),
            TaxCodeFieldsFactory.build(full_name='Marie Kovalenko', tax_id=None),
            PassportFieldsFactory.build(expiry_date=date(2020, 1, 1)),
        ]

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == [
            'name_mismatch',
            'expired:passport',
            'missing_field:tax_code.tax_id',
            'missing_document:certificate',
        ]

    def test_missing_document_keeps_its_place_among_the_documents(self) -> None:
        documents = build_documents(
            passport=PassportFieldsFactory.build(expiry_date=date(2020, 1, 1)),
            tax_code=None,
            certificate=CertificateFieldsFactory.build(full_name='Marie Kovalenko', gpa=None),
        )

        verdict = check_applicant_documents(documents, today=TODAY)

        assert verdict.issues == [
            'name_mismatch',
            'expired:passport',
            'missing_document:tax_code',
            'missing_field:certificate.gpa',
        ]


class TestBrokenApplicantSet:
    def test_broken_set_gets_exactly_the_issues_the_course_names(self) -> None:
        verdict = check_applicant_documents(build_broken_documents(), today=TODAY)

        assert verdict.is_valid is False
        assert verdict.issues == BROKEN_SET_ISSUES
