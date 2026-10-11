import asyncio
from collections.abc import Sequence
import threading

import pymupdf
from starlette.datastructures import UploadFile

from app.ai_assistant_langchain.applicant_documents.enums import ImageFormat
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentImage
from app.core.exceptions import DocumentTooLargeError, UnsupportedDocumentError

MAX_DOCUMENT_MB = 5
MAX_DOCUMENT_BYTES = MAX_DOCUMENT_MB * 1024 * 1024
PDF_RENDER_DPI = 150
PDF_POINTS_PER_INCH = 72
MAX_RENDERED_SIDE_PX = 2000

IMAGE_SIGNATURES: dict[bytes, ImageFormat] = {
    b'\x89PNG\r\n\x1a\n': ImageFormat.PNG,
    b'\xff\xd8\xff': ImageFormat.JPEG,
}
PDF_SIGNATURE = b'%PDF-'
PDF_READ_ERRORS = (RuntimeError, ValueError, IndexError, pymupdf.mupdf.FzErrorBase)

# PyMuPDF must not run on two threads at once. Its documentation says so in these words: "PyMuPDF
# does not support running on multiple threads - doing so may cause incorrect behaviour or even
# crash Python itself."
#
# A page is drawn in a worker thread (`asyncio.to_thread` below) so that the event loop keeps
# answering other requests meanwhile. Each upload gets a thread of its own, so two applicants
# sending a PDF in the same moment would put two threads inside the library together, and a crash
# there takes the whole process down, with every conversation in it.
#
# The lock lets one thread in at a time. A second upload waits inside its own worker thread, not on
# the event loop, so nothing else is held up. The lock is a `threading.Lock`, not an `asyncio.Lock`:
# what has to be kept apart is threads, and the function that takes it is not a coroutine.
#
# It does not cover the handbook ingestion (`structured/parser.py`), which calls PyMuPDF on the event
# loop's own thread without taking this lock.
PDF_RENDER_LOCK = threading.Lock()


async def read_applicant_documents(uploads: Sequence[UploadFile]) -> list[DocumentImage]:
    return [await read_document_image(upload) for upload in uploads]


async def read_document_image(upload: UploadFile) -> DocumentImage:
    content = await upload.read(MAX_DOCUMENT_BYTES + 1)
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(
            f'The file "{upload.filename}" is larger than the {MAX_DOCUMENT_MB} MB a document may be.'
        )

    for signature, image_format in IMAGE_SIGNATURES.items():
        if content.startswith(signature):
            return DocumentImage(content=content, mime_type=image_format)

    if content.startswith(PDF_SIGNATURE):
        try:
            rendered_page = await asyncio.to_thread(_render_first_pdf_page, content)
        except PDF_READ_ERRORS as read_error:
            raise UnsupportedDocumentError(
                f'The file "{upload.filename}" is a PDF that cannot be opened.'
            ) from read_error
        return DocumentImage(content=rendered_page, mime_type=ImageFormat.PNG)

    raise UnsupportedDocumentError(f'The file "{upload.filename}" is not a PNG, a JPEG or a PDF.')


def _render_first_pdf_page(content: bytes) -> bytes:
    with PDF_RENDER_LOCK, pymupdf.open(stream=content, filetype='pdf') as pdf:
        page = pdf[0]
        longest_side = max(page.rect.width, page.rect.height)
        zoom = min(PDF_RENDER_DPI / PDF_POINTS_PER_INCH, MAX_RENDERED_SIDE_PX / longest_side)
        return bytes(page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes('png'))
