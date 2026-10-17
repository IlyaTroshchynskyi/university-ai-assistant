from collections.abc import Callable

import pytest

from app.ai_assistant_langchain.applicant_documents.document_rules import check_applicant_documents
from app.ai_assistant_langchain.applicant_documents.extractors import extract_document_fields, get_extractor
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentFields
from tests.factory_models.applicant_factory import (
    BROKEN_SET_ISSUES,
    build_broken_documents,
    build_document_images,
    build_documents,
    SAMPLE_DAY,
)

pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
]

DocumentSetBuilder = Callable[[], list[DocumentFields]]


class TestDocumentValidation:
    @pytest.mark.parametrize(
        ('build_document_set', 'expected_issues'),
        [(build_documents, []), (build_broken_documents, BROKEN_SET_ISSUES)],
        ids=['valid', 'broken'],
    )
    async def test_drawn_set_is_read_and_judged_as_the_course_expects(
        self, build_document_set: DocumentSetBuilder, expected_issues: list[str]
    ) -> None:
        images = build_document_images(build_document_set())

        extracted_fields = await extract_document_fields(images, get_extractor())
        verdict = check_applicant_documents(extracted_fields, today=SAMPLE_DAY)

        assert verdict.is_valid is (not expected_issues)
        assert sorted(verdict.issues) == sorted(expected_issues)
