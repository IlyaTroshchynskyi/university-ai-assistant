from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
CURRENT_DOCUMENTS = [
    'CLAUDE.md',
    'README.md',
    'app/ai_assistant_langchain/CLAUDE.md',
    'tests/CLAUDE.md',
    *sorted(str(document.relative_to(ROOT)) for document in ROOT.glob('docs/*.md')),
    *sorted(str(document.relative_to(ROOT)) for document in ROOT.glob('docs/decisions/*.md')),
    *sorted(str(document.relative_to(ROOT)) for document in ROOT.glob('db/*.md')),
]
GITIGNORED_PATHS = {'db/seed/', 'docs/handbook.pdf'}
PATH_PATTERN = re.compile(r'`((?:app|tests|db|docs)/[\w./-]+)`')
MAKE_TARGET_PATTERN = re.compile(r'`make ([\w-]+)`')


@pytest.mark.parametrize('document', CURRENT_DOCUMENTS)
def test_document_paths_exist(document: str) -> None:
    text = (ROOT / document).read_text()
    missing_paths = [
        path for path in PATH_PATTERN.findall(text) if path not in GITIGNORED_PATHS and not (ROOT / path).exists()
    ]
    assert missing_paths == []


@pytest.mark.parametrize('document', CURRENT_DOCUMENTS)
def test_document_make_targets_exist(document: str) -> None:
    text = (ROOT / document).read_text()
    makefile = (ROOT / 'Makefile').read_text()
    missing_targets = [
        target
        for target in MAKE_TARGET_PATTERN.findall(text)
        if not re.search(rf'^{re.escape(target)}:', makefile, re.MULTILINE)
    ]
    assert missing_targets == []
