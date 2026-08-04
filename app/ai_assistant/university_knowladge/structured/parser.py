"""Structure-aware PDF parser built on PyMuPDF4LLM."""

from abc import ABC, abstractmethod
from collections import Counter
import re
import tempfile

import pymupdf
import pymupdf4llm

from app.ai_assistant.university_knowladge.structured.schemas import DocElement

_IMAGE_REF = re.compile(r'^!\[.*?\]\((.+?)\)$')
_PAGE_NUMBER = re.compile(r'(Page)\s+\d+', re.IGNORECASE)
_HEADING = re.compile(r'^#{1,6}\s+(.*)$')


def _heading_text(line: str) -> str | None:
    """Return the cleaned heading text if ``line`` is a Markdown heading, else None."""
    match = _HEADING.match(line.strip())
    if not match:
        return None
    return match.group(1).strip().strip('*').strip()


def _split_blocks(text: str) -> list[list[str]]:
    """Split page markdown into blocks of consecutive non-blank lines."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _is_table_block(lines: list[str]) -> bool:
    """A Markdown table: every line is a pipe row."""
    return len(lines) >= 2 and all(line.lstrip().startswith('|') for line in lines)


def _running_lines(pages: list[dict]) -> set[str]:
    """Lines that repeat verbatim on at least half the pages — the running
    header/footer that should be treated as noise rather than content."""
    if len(pages) < 2:
        return set()
    seen = Counter()

    for page in pages:
        for line in {ln.strip() for ln in page.get('text', '').splitlines() if ln.strip()}:
            seen[line] += 1
    threshold = len(pages) / 2
    return {line for line, count in seen.items() if count >= threshold}


def _is_noise(line: str, running: set[str]) -> bool:
    """Running headers/footers and page numbers are furniture, not content."""
    stripped = line.strip()
    return stripped in running or bool(_PAGE_NUMBER.search(stripped))


def _clean_blocks(text: str, running: set[str]) -> list[list[str]]:
    """The page's blocks with noise lines removed, dropping blocks that removal empties."""
    blocks = ([line for line in block if not _is_noise(line, running)] for block in _split_blocks(text))
    return [block for block in blocks if block]


def _classify_block(block: list[str], page: int, section: str | None) -> DocElement:
    """Turn a cleaned block into the element its shape implies."""
    image_match = _IMAGE_REF.match(block[0]) if len(block) == 1 else None
    if image_match:
        return DocElement(kind='image', page=page, section=section, raw=image_match.group(1))

    if _is_table_block(block):
        return DocElement(kind='table', page=page, section=section, raw='\n'.join(block))
    return DocElement(kind='text', page=page, section=section, raw='\n'.join(block))


def parse_pages(pages: list[dict]) -> list[DocElement]:
    """Classify PyMuPDF4LLM page dicts into ordered :class:`DocElement`s."""
    running = _running_lines(pages)
    elements: list[DocElement] = []
    section: str | None = None

    for page in pages:
        page_no = page.get('metadata', {}).get('page', 0)
        for block in _clean_blocks(page.get('text', ''), running):
            heading = _heading_text(block[0]) if len(block) == 1 else None
            if heading is not None:
                # A heading is not an element of its own; it labels what follows.
                section = heading
                continue
            elements.append(_classify_block(block, page_no, section))
    return elements


class Parser(ABC):
    """Parses a PDF into an ordered list of typed :class:`DocElement`s. Behind this
    interface so the parsing engine is a swappable dependency (mirrors ``TextSplitter``)."""

    @abstractmethod
    def parse(self, data: bytes, source: str) -> list[DocElement]: ...


class StructuredPdfParser(Parser):
    """Renders a PDF to per-page Markdown via PyMuPDF4LLM (extracting images to
    ``image_dir``) and classifies the result into :class:`DocElement`s.

    Image elements carry the on-disk path of the extracted PNG so enrichment can
    read it; the caller owns cleanup of ``image_dir`` once enrichment is done."""

    def __init__(self, image_dir: str | None = None):
        self._image_dir = image_dir

    def parse(self, data: bytes, source: str) -> list[DocElement]:
        image_dir = self._image_dir or tempfile.mkdtemp(prefix='pdf-images-')
        doc = pymupdf.open(stream=data, filetype='pdf')
        try:
            pages = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                write_images=True,
                image_path=image_dir,
                filename=source,  # base name for extracted images (doc has no path when opened from bytes)
                show_progress=False,
            )
        finally:
            doc.close()

        return parse_pages(pages)
