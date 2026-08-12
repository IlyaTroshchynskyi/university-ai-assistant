import asyncio
from functools import lru_cache
import logging
from typing import Sequence, TypeAlias

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseConversationalMetric,
    BaseMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationalGEval,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
    TurnFaithfulnessMetric,
)
from deepeval.models import GPTModel
from deepeval.test_case import ConversationalTestCase, LLMTestCase, LLMTestCaseParams, TurnParams
import httpx

from tests.integration.config import get_eval_settings

logger = logging.getLogger(__name__)


AnyMetric: TypeAlias = BaseMetric | BaseConversationalMetric


CORRECTNESS_CRITERIA = (
    'Decide whether the actual output states the same facts as the expected output. Every number, '
    'amount, date and name that appears in the expected output must appear in the actual output '
    'with the same value; a wrong or missing figure is a failure. Wording, ordering and extra '
    'helpful phrasing are free — do not penalise them.'
)


@lru_cache
def judge_model() -> GPTModel:
    """The judge every metric scores with, built once.

    ``async_http_client`` is what keeps one connection pool for the whole run. DeepEval builds a
    fresh ``AsyncOpenAI`` on *every* judge call (``load_model(async_mode=True)`` inside its generate
    methods), so caching the ``GPTModel`` alone still pays a TLS handshake per metric per golden,
    and leaves every one of those clients unclosed.

    Unclosed matters beyond the wasted handshake: the OpenAI SDK's default client is an
    ``AsyncHttpxClientWrapper`` whose ``__del__`` fires ``create_task(self.aclose())``. Nobody
    awaits that task, so when the garbage collector reaches those clients as the session loop is
    closing, each one raises ``RuntimeError('Event loop is closed')`` inside a task no one
    retrieves — a wall of tracebacks after a run that actually passed. Handing in an
    ``httpx.AsyncClient`` sidesteps it: the SDK then uses it as given rather than wrapping it, and
    ``httpx.AsyncClient`` has no ``__del__`` at all, so nothing is scheduled at collection time."""
    settings = get_eval_settings()
    return GPTModel(
        model=settings.EVAL_JUDGE_MODEL,
        api_key=settings.EVAL_MODEL_API_KEY or None,
        async_http_client=httpx.AsyncClient(),
    )


def retriever_metrics() -> list[BaseMetric]:
    """How good the passages the retriever returned are. Three LLM-as-a-judge metrics, all scoring
    the same ``retrieval_context`` but asking three different questions of it.

    **ContextualRelevancyMetric — is the retrieved context on topic?**
    Measures the overall relevance of the information presented in ``retrieval_context`` for the
    given ``input``. The judge cuts every passage into statements and rules on each one separately:
    does this statement bear on the question, yes or no.

    **ContextualRecallMetric — did the retriever find everything the answer needs?**
    Measures the extent to which ``retrieval_context`` aligns with ``expected_output``. The judge
    cuts the golden answer into sentences and asks of each one whether it can be attributed to some
    passage; the score is the share that can.

    **ContextualPrecisionMetric — are the useful passages ranked first?**
    Evaluates whether the passages in ``retrieval_context`` that are relevant to ``input`` are
    ranked higher than the irrelevant ones. The judge labels each passage relevant or not (against
    the question and the expected answer).

    """
    settings, model = get_eval_settings(), judge_model()
    return [
        ContextualRelevancyMetric(threshold=settings.EVAL_CONTEXTUAL_RELEVANCY_THRESHOLD, model=model),
        ContextualRecallMetric(threshold=settings.EVAL_CONTEXTUAL_RECALL_THRESHOLD, model=model),
        ContextualPrecisionMetric(threshold=settings.EVAL_CONTEXTUAL_PRECISION_THRESHOLD, model=model),
    ]


def agent_metrics(has_context: bool) -> list[BaseMetric]:
    """How good the agent's answer is. Three metrics that deliberately pull in different
    directions, because each is blind to what the other two check.

    **Correctness (GEval) — does the answer say what the golden says?**
    The only custom metric of the three: the judge is handed the answer and the golden's expected
    answer, plus ``CORRECTNESS_CRITERIA``, and returns one score for how well the criteria are met.
    That criteria text is what defines the metric, and it says facts only — every number, amount,
    date and name in the expected answer must appear with the same value, while wording, ordering
    and extra phrasing are explicitly forgiven.

    **AnswerRelevancyMetric — does the answer address the question that was asked?**
    Measures the quality of the generator by evaluating how relevant ``actual_output`` is to
    ``input``. The judge cuts the answer into statements and rules on each one against the
    question; the score is the share not ruled irrelevant (an "I don't know" verdict counts as
    relevant).

    **FaithfulnessMetric — is the answer grounded in what the tool returned?**
    Measures the quality of the generator by evaluating whether ``actual_output`` factually aligns
    with the contents of ``retrieval_context``. The judge first extracts the "truths" stated by the
    passages and the "claims" made by the answer, then rules on each claim against those truths;
    the score is the share not contradicted. This is the hallucination check.
    """

    settings = get_eval_settings()
    model = judge_model()
    metrics: list[BaseMetric] = [
        # answer ↔ expected answer, facts only.
        GEval(
            name='Correctness',
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=settings.EVAL_CORRECTNESS_THRESHOLD,
            model=model,
        ),
    ]
    if has_context:
        metrics += [
            # question ↔ answer. Share of the answer's statements that address the question.
            AnswerRelevancyMetric(
                threshold=settings.EVAL_ANSWER_RELEVANCY_THRESHOLD,
                model=model,
                include_reason=True,
            ),
            # answer ↔ retrieved passages. Share of the answer's claims the passages support.
            FaithfulnessMetric(
                threshold=settings.EVAL_FAITHFULNESS_THRESHOLD,
                model=model,
                include_reason=True,
            ),
        ]
    return metrics


def routing_metrics() -> list[BaseMetric]:
    """Which tool the agent reached for, given the question."""
    return [ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)]


OUTCOME_STEPS = [
    'Read the expected outcome and list every number, amount, date, name and yes/no claim it makes.',
    'For each item on that list, check whether the assistant turns state it with the same value. An '
    'item that is missing, or stated with a different value, is a failure.',
    'Check any figure the assistant gave early that the expected outcome says no longer stands: the '
    'assistant must have withdrawn or corrected it, not merely stopped repeating it.',
    'Ignore wording, ordering, tone, role consistency, helpfulness, and any extra facts beyond the '
    'expected outcome. Score only whether the expected facts were established.',
]


def conversation_metrics() -> list[BaseConversationalMetric]:
    """How good a whole conversation is. Two metrics, each blind to the other. Both read the same
    left-hand side — everything the assistant said — and differ entirely in what they hold it
    against:

    **TurnFaithfulnessMetric — did the agent stay grounded across the conversation?**
    Compares the assistant turns ↔ the ``retrieval_context`` attached to those turns.
    ``FaithfulnessMetric`` lifted to a dialogue: the turns are grouped into user→assistant
    interactions, a slidingconversation_metrics window ending at each interaction is flattened into one block of text,
    and each block is scored exactly as ``Faithfulness`` scores a single answer — truths from the
    passages, claims from what the assistant said, one verdict per claim. The metric's score is the
    mean over the windows.

    **Outcome (ConversationalGEval) — did the conversation end up establishing the right facts?**
    Compares the assistant turns ↔ the case's ``expected_outcome``.
     Reads ``CONTENT`` and ``EXPECTED_OUTCOME``; it never sees a passage, so it has no idea where
     a correct fact came from.

    """
    settings, model = get_eval_settings(), judge_model()
    return [
        TurnFaithfulnessMetric(
            threshold=settings.EVAL_TURN_FAITHFULNESS_THRESHOLD,
            model=model,
            include_reason=True,
            window_size=20,
        ),
        ConversationalGEval(
            name='Outcome',
            evaluation_steps=OUTCOME_STEPS,
            evaluation_params=[TurnParams.CONTENT, TurnParams.EXPECTED_OUTCOME],
            threshold=settings.EVAL_CONVERSATION_OUTCOME_THRESHOLD,
            model=model,
        ),
    ]


def bias_metrics() -> list[BaseMetric]:
    """Whether the answer carries gender, political, racial/ethnic or geographical bias."""
    return [BiasMetric(threshold=get_eval_settings().EVAL_BIAS_THRESHOLD, model=judge_model(), include_reason=True)]


async def assert_metrics(test_case: LLMTestCase | ConversationalTestCase, metrics: Sequence[AnyMetric]) -> None:
    """Score every metric against ``test_case``, report the whole table, then fail on the ones that
    did not meet their threshold.

    Takes a ``ConversationalTestCase`` as readily as an ``LLMTestCase``: the conversational metrics
    expose the same ``a_measure``, ``threshold`` and ``success``.

    Replaces ``deepeval.assert_test``, which measures on *copies* of the metrics (``copy_metrics``
    inside ``a_execute_test_cases``) and mentions only the failures — so a passing score cannot be
    read back out of it, and a green run tells you nothing about how close it was. Measuring here
    keeps every score on the metric objects themselves.

    Each metric is an independent judge call, so they run concurrently.

    Scores are logged rather than printed: ``pytest`` shows them under *Captured log call* on a
    failure, and ``--log-cli-level=INFO`` shows them on a pass too.
    """
    async with asyncio.TaskGroup() as task_group:
        for metric in metrics:
            task_group.create_task(metric.a_measure(test_case))

    # ``input`` is an LLMTestCase field; a ConversationalTestCase has no single input, so it is
    # labelled by its name instead.
    label = test_case.input if isinstance(test_case, LLMTestCase) else test_case.name
    logger.info('Metric scores for %r\n%s', label, _format_scores(metrics))

    failed = [metric for metric in metrics if not metric.success]
    if failed:
        # Logged as well as asserted. pytest truncates a long assertion message, and the judge's
        # reason is the only thing that separates "the agent was wrong" from "the judge slipped" —
        # losing it means the run has to be paid for twice.
        logger.info('\n'.join(f'{metric.__name__} reason: {metric.reason}' for metric in failed))

    assert not failed, 'Failed — ' + '; '.join(
        f'{metric.__name__} {metric.score:.3f} vs {metric.threshold}' for metric in failed
    )


def _format_scores(metrics: Sequence[AnyMetric]) -> str:
    """One aligned line per metric: name, score against its threshold, verdict."""
    width = max(len(metric.__name__) for metric in metrics)
    return '\n'.join(
        f'  {metric.__name__:<{width}}  {metric.score:.3f} / {metric.threshold:<5} '
        f'{"PASS" if metric.success else "FAIL"}'
        for metric in metrics
    )
