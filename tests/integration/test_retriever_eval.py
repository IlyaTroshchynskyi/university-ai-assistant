"""Retriever evaluation: how good are the passages ``KnowledgeService.search`` returns for each
golden, judged against the golden's expected answer.

Search is called with its production defaults (``limit=3``, ``doc_type='general'``) — the same
method and arguments the ``retriever`` tool uses, so this measures the configuration the agent
really runs.

What the first full run (2026-08-10, 21 cases) established:

* **Recall 1.000 on every case.** Whatever each expected answer needs, the search brings back.
  This is the metric that would catch a real retrieval regression, and it is clean.
* **Precision 1.000 on 19 of 21.** Ranking is fine almost everywhere.
* **Relevancy 0.095–0.682, median ~0.28.** Structural, not a defect — it is the share of
  retrieved *sentences* on topic, so it measures chunk bulk. It reproduces to within 0.03 on a
  re-run, which is why ``EVAL_CONTEXTUAL_RELEVANCY_THRESHOLD`` could be set just under its
  measured floor rather than guessed.

Three things to know before treating a red run as a regression:

1. **Recall gets one verdict per sentence of the expected answer, so a one-sentence golden makes
   it strictly binary — 1.000 or 0.000, nothing between.** Golden 2 kept failing on it: its
   expected answer was the fragment *"5 years, at $22,000 per year."*, whose two facts live in
   two different chunks, and the judge flipped between "both are there, in nodes 1 and 3" and
   "not combined in one sentence". Splitting it into two sentences fixed it — 1.000 on three
   consecutive measurements, where the fragment gave 1.000 · 0.000 · 1.000 · 0.000.

   Goldens 1, 15 and 21 still carry single-fragment expected answers (*"$26,000 per year."*,
   *"$8,200 per year."*, *"A BSc, 4 years, at $22,500 per year."*). They pass today because each
   fact sits in one chunk, but 21 packs three facts into one sentence and is the next candidate
   to flip.
2. **Precision can collapse outright.** Golden 14 went from 0.583 to 0.000 and back to 1.000 on
   unchanged passages. A 0.000 next to a 1.000 from the previous run is a judge that fell over.
   Re-run that one case before believing it.
3. **Golden 19 fails precision for a real reason** — 0.500 in three runs out of three. Asking
   when the Fall and Spring terms run puts a payment-plans chunk above the academic-year table.
   The dates are still retrieved (recall 1.000), they are just ranked second. This one is a
   genuine ranking finding and is left failing on purpose.

Thresholds live in ``tests/integration/config.py``; each carries the measurements behind it.
"""

from deepeval.test_case import LLMTestCase
import pytest

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service
from tests.integration.conftest import assert_metrics, retriever_metrics
from tests.integration.goldens import golden_params, GoldenCase, Layer

# Session loop, for the same reason as the agent suite: ``get_qdrant_client`` and the OpenAI
# client behind ``get_embedder`` are cached singletons, so the first case binds them to whatever
# loop it ran on. On the default function loop every later case then talks to a loop nobody is
# running any more.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('knowledge_base_populated'),
]

GOLDENS = golden_params(Layer.RETRIEVER)


class TestRetrieverEvaluation:
    @pytest.mark.parametrize('golden', GOLDENS)
    async def test_retriever_evaluation(self, golden: GoldenCase) -> None:
        docs = await get_knowledge_service().search(golden.question)

        test_case = LLMTestCase(
            input=golden.question,
            expected_output=golden.expected_answer,
            retrieval_context=[doc.payload['text'] for doc in docs if doc.payload],
        )

        await assert_metrics(test_case, retriever_metrics())
