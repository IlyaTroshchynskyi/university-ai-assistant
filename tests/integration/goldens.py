from enum import StrEnum
import json
from pathlib import Path

from pydantic import BaseModel
import pytest

GOLDENS_PATH = Path(__file__).parent / 'data' / 'goldens.json'

# Goldens whose answer lives inside a table are tagged with this section suffix and get the
# ``tables`` marker, which is what ``make eval-tables`` selects on.
_TABLE_SECTION_SUFFIX = '(table)'


class Layer(StrEnum):
    RETRIEVER = 'retriever'
    AGENT = 'agent'


class GoldenCase(BaseModel):
    """One question with its ground truth.

    No ``context`` field: both suites take their ``retrieval_context`` from the live retriever, so
    the ``context`` blocks in ``goldens.json`` were parsed and never read. Pydantic ignores them.
    They are the raw material for a ``HallucinationMetric`` — the one metric that scores an answer
    against curated ground truth instead of whatever was retrieved — if that is ever wanted; note
    """

    id: int
    question: str
    expected_answer: str
    layers: list[Layer]
    section: str | None = None

    @property
    def is_table(self) -> bool:
        """Whether this golden's answer lives inside a table, and so is what a chunking change
        splitting a table apart would break first. Goldens 19-21 today."""
        return bool(self.section) and self.section.strip().endswith(_TABLE_SECTION_SUFFIX)


def load_goldens(layer: Layer) -> list[GoldenCase]:
    raw_cases = json.loads(GOLDENS_PATH.read_text(encoding='utf-8'))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]
    return [case for case in cases if layer in case.layers]


def golden_params(layer: Layer) -> list[pytest.param]:
    """``load_goldens``, wrapped so each case carries its golden id as the pytest node id.

    Parametrizing on ``load_goldens`` directly is what you must *not* do: pytest then names the
    cases by position in the filtered list, and the two layers filter differently — golden 10 is
    retriever-only, 22 and 23 are agent-only. ``[golden19]`` meant golden 21 in the agent suite and
    golden 20 in the retriever one, so a failure could not be traced back to ``goldens.json``
    without counting. Now it reads ``[21]``.
    """
    return [
        pytest.param(case, id=str(case.id), marks=pytest.mark.tables if case.is_table else ())
        for case in load_goldens(layer)
    ]
