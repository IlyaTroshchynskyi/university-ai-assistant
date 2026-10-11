import asyncio
from contextlib import contextmanager
import io
import threading
import time
from typing import Iterator
from unittest.mock import patch

import pymupdf
import pytest
from starlette.datastructures import UploadFile

from app.ai_assistant_langchain.applicant_documents.document_reader import (
    MAX_DOCUMENT_BYTES,
    MAX_RENDERED_SIDE_PX,
    read_applicant_documents,
    read_document_image,
)
from app.ai_assistant_langchain.applicant_documents.enums import ImageFormat
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentImage
from app.core.exceptions import DocumentTooLargeError, UnsupportedDocumentError
from tests.factories.applicant_factory import build_document_image, PassportFieldsFactory

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
JPEG_SIGNATURE = b'\xff\xd8\xff'
PASSPORT_PNG = build_document_image(PassportFieldsFactory.build()).content
JPEG_BYTES = JPEG_SIGNATURE + b'\xe0\x00\x10JFIF-scan'
PDF_WITHOUT_PAGES = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R /Size 2 >>\n%%EOF\n'
PATH_TO_PDF_RENDERER = 'app.ai_assistant_langchain.applicant_documents.document_reader._render_first_pdf_page'
PATH_TO_PDF_OPENER = 'app.ai_assistant_langchain.applicant_documents.document_reader.pymupdf.open'
RENDER_OVERLAP_SECONDS = 0.05


def build_upload(content: bytes, filename: str = 'scan') -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename, size=len(content))


def build_pdf(width: float = 595, height: float = 842) -> bytes:
    pdf = pymupdf.open()
    pdf.new_page(width=width, height=height).insert_text((72, 72), 'REPUBLIC PASSPORT')
    return bytes(pdf.tobytes())


def build_encrypted_pdf() -> bytes:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), 'REPUBLIC PASSPORT')
    return bytes(pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw='user', owner_pw='owner'))


class RenderTracker:
    def __init__(self) -> None:
        self.peak_concurrency = 0
        self._concurrent_renders = 0
        self._counter_lock = threading.Lock()
        self._open_document = pymupdf.open

    @contextmanager
    def open_document(self, *args: object, **kwargs: object) -> Iterator[pymupdf.Document]:
        with self._counter_lock:
            self._concurrent_renders += 1
            self.peak_concurrency = max(self.peak_concurrency, self._concurrent_renders)
        try:
            time.sleep(RENDER_OVERLAP_SECONDS)
            with self._open_document(*args, **kwargs) as pdf:
                yield pdf
        finally:
            with self._counter_lock:
                self._concurrent_renders -= 1


def build_mupdf_error() -> Exception:
    try:
        pymupdf.Pixmap(PNG_SIGNATURE + b'\x00' * 16)
    except pymupdf.mupdf.FzErrorBase as error:
        return error
    raise AssertionError('MuPDF opened an image that has no data')


def measure_image(image: DocumentImage) -> tuple[int, int]:
    pixmap = pymupdf.Pixmap(image.content)
    return pixmap.width, pixmap.height


class TestReadDocumentImage:
    async def test_png_is_kept_as_it_was_sent(self) -> None:
        image = await read_document_image(build_upload(PASSPORT_PNG))

        assert image == DocumentImage(content=PASSPORT_PNG, mime_type=ImageFormat.PNG)

    async def test_jpeg_is_kept_as_it_was_sent(self) -> None:
        image = await read_document_image(build_upload(JPEG_BYTES))

        assert image == DocumentImage(content=JPEG_BYTES, mime_type=ImageFormat.JPEG)

    async def test_format_comes_from_the_bytes_not_from_the_file_name(self) -> None:
        image = await read_document_image(build_upload(PASSPORT_PNG, 'passport.jpg'))

        assert image.mime_type is ImageFormat.PNG

    async def test_pdf_is_rendered_to_a_png(self) -> None:
        image = await read_document_image(build_upload(build_pdf()))

        assert image.mime_type is ImageFormat.PNG
        assert image.content.startswith(PNG_SIGNATURE)

    @pytest.mark.parametrize(
        'content',
        [b'just some text', b'', b'%PDF-1.7 not really a pdf', build_encrypted_pdf(), PDF_WITHOUT_PAGES],
        ids=['text', 'empty', 'corrupt pdf', 'password-protected pdf', 'pdf without pages'],
    )
    async def test_anything_else_is_refused_by_its_file_name(self, content: bytes) -> None:
        with pytest.raises(UnsupportedDocumentError, match='"tax_card"'):
            await read_document_image(build_upload(content, 'tax_card'))

    async def test_pdf_the_renderer_gives_up_on_is_refused_by_its_file_name(self) -> None:
        with (
            patch(PATH_TO_PDF_RENDERER, side_effect=build_mupdf_error()),
            pytest.raises(UnsupportedDocumentError, match='"passport_scan"'),
        ):
            await read_document_image(build_upload(build_pdf(), 'passport_scan'))

    async def test_file_at_the_size_limit_is_accepted(self) -> None:
        content = PNG_SIGNATURE.ljust(MAX_DOCUMENT_BYTES, b'\x00')

        image = await read_document_image(build_upload(content))

        assert len(image.content) == MAX_DOCUMENT_BYTES

    async def test_file_over_the_size_limit_is_refused_by_its_file_name(self) -> None:
        content = PNG_SIGNATURE.ljust(MAX_DOCUMENT_BYTES + 1, b'\x00')

        with pytest.raises(DocumentTooLargeError, match='"certificate_photo"'):
            await read_document_image(build_upload(content, 'certificate_photo'))


class TestPdfPageSize:
    async def test_ordinary_page_is_rendered_at_full_resolution(self) -> None:
        image = await read_document_image(build_upload(build_pdf()))

        assert measure_image(image) == (1240, 1755)

    async def test_oversized_page_is_rendered_within_the_side_limit(self) -> None:
        oversized_pdf = build_pdf(width=2880, height=2880)

        image = await read_document_image(build_upload(oversized_pdf))

        assert measure_image(image) == (MAX_RENDERED_SIDE_PX, MAX_RENDERED_SIDE_PX)

    async def test_longest_side_of_a_wide_page_is_the_one_limited(self) -> None:
        wide_pdf = build_pdf(width=2880, height=1440)

        image = await read_document_image(build_upload(wide_pdf))

        assert measure_image(image) == (MAX_RENDERED_SIDE_PX, MAX_RENDERED_SIDE_PX // 2)


class TestPdfRenderingThreads:
    async def test_two_pdfs_are_never_inside_the_renderer_at_the_same_time(self) -> None:
        uploads = [build_upload(build_pdf()), build_upload(build_pdf())]
        tracker = RenderTracker()

        with patch(PATH_TO_PDF_OPENER, tracker.open_document):
            async with asyncio.TaskGroup() as task_group:
                for upload in uploads:
                    task_group.create_task(read_document_image(upload))

        assert tracker.peak_concurrency == 1


class TestReadApplicantDocuments:
    async def test_reads_every_attached_file_in_the_order_attached(self) -> None:
        documents = await read_applicant_documents([build_upload(JPEG_BYTES), build_upload(PASSPORT_PNG)])

        assert documents == [
            DocumentImage(content=JPEG_BYTES, mime_type=ImageFormat.JPEG),
            DocumentImage(content=PASSPORT_PNG, mime_type=ImageFormat.PNG),
        ]

    async def test_nothing_attached_reads_nothing(self) -> None:
        assert await read_applicant_documents([]) == []
