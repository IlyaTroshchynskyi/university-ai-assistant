"""Extract plain text from PDF bytes via pdfplumber (the same library CrewAI's own
PDFKnowledgeSource uses). A thin service so it can be injected and later swapped (e.g. OCR)."""

import io

import pdfplumber


class PdfLoader:
    """Reads the concatenated text of all pages from an in-memory PDF."""

    def load(self, data: bytes) -> str:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return '\n'.join(page.extract_text() or '' for page in pdf.pages).strip()
