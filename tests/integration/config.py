"""Settings for the DeepEval suites only."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    # Not gpt-4o-mini: it stalls on DeepEval's ``Verdicts`` schema — the nested one with a
    # ``Literal`` enum and an optional ``reason``, which FaithfulnessMetric uses for its verdict
    # step. Same prompt answers in 1.7s with no schema and times out past 70s with it, while
    # gpt-4.1-mini and gpt-4o both return in ~2s. Flat schemas (``Truths``) are fine on all three.
    EVAL_JUDGE_MODEL: str = 'gpt-4.1-mini'

    # Key for the judge. Leave it unset to bill the same key as the app: ``judge_model()`` passes
    # ``None`` in that case, which is the only value that makes deepeval fall back to its own
    # resolution — it treats an empty string as a real (invalid) key and raises.
    EVAL_MODEL_API_KEY: str = ''

    # One threshold per metric: they measure different things and are not comparable.
    #
    # Relevancy scores the *share* of retrieved sentences that bear on the question, so chunk size
    # caps it. Measured across all 21 goldens the spread is 0.095–0.682, median ~0.28, while recall
    # was 1.000 on every case — nothing was missing, the chunks were just bulky. 0.08 sits under
    # the measured floor, so it detects a regression rather than grading quality. Raise it once
    # ``structured/chunker.py`` is wired into ``KnowledgeService.ingest_pdf``.
    EVAL_CONTEXTUAL_RELEVANCY_THRESHOLD: float = 0.08

    # Coverage and ranking are not capped by chunk size and sit at 1.000 nearly everywhere. When
    # they fail they collapse to 0.000 rather than drift — recall scores one verdict per *sentence*
    # of the expected answer, so a one-sentence golden makes it strictly binary. Lowering these
    # would not help; there is nothing between 0.000 and 1.000 to aim at.
    EVAL_CONTEXTUAL_RECALL_THRESHOLD: float = 0.7
    EVAL_CONTEXTUAL_PRECISION_THRESHOLD: float = 0.7

    # 0.7, temporarily; it was 0.8, then 0.75. A ratio of statements, so the score is quantised by
    # how many the answer has — four give 1.000, 0.750, 0.500; seven give 0.857, 0.714. 0.8 failed
    # a correct four-statement answer over a closing "let me know if you need anything else", and
    # 0.75 sat in the unreachable gap between 0.857 and 0.714. Raise it only against measured
    # scores, never on principle.
    EVAL_ANSWER_RELEVANCY_THRESHOLD: float = 0.7

    # 0.7, down from 0.8, for the same quantisation reason. Not a free change: this is the
    # hallucination check, and at 0.7 one ungrounded claim out of four passes. Of the lowered
    # thresholds this is the one to raise first.
    EVAL_FAITHFULNESS_THRESHOLD: float = 0.7

    # 0.6, down from 0.7: GEval reads the judge's score logprobs rather than asking for a verdict,
    # so an undecided judge lands near 0.7 on its own. Five runs on one unchanged, fully correct
    # answer scored 0.675, 0.705, 0.744, 0.839 and a pass. The cure is a steadier judge, not a
    # lower bar — pass ``evaluation_steps`` instead of free-form ``criteria``, and/or run this one
    # metric on gpt-4.1. Raise back to 0.7 once either is in.
    EVAL_CORRECTNESS_THRESHOLD: float = 0.6

    # --- multi-turn and bias suites --------------------------------------------------------------
    #
    # Neither of the two below is measured: neither suite has been run against a live agent. They are
    # borrowed from the closest single-turn metric, and both suites are instruments rather than gates
    # until these carry observed numbers the way the thresholds above do. EVAL_BIAS_THRESHOLD is the
    # exception and does not need a measurement — see its own comment.

    # Same family as EVAL_FAITHFULNESS_THRESHOLD and quantised the same way. First of these to raise:
    # it is the hallucination check, and ``false_premise`` exists to make it bite.
    EVAL_TURN_FAITHFULNESS_THRESHOLD: float = 0.7

    # A GEval, so it inherits the logprob wobble documented on EVAL_CORRECTNESS_THRESHOLD.
    EVAL_CONVERSATION_OUTCOME_THRESHOLD: float = 0.6

    # Inverted — a ceiling, not a floor: BiasMetric sets ``success = score <= threshold``. The score
    # is ``biased_opinions / extracted_opinions``, and a short answer yields one to four opinions, so
    # everything reachable below 0.5 is 0, 0.25 or 0.333 — there is no room between 0 and 0.25 to
    # leave slack in, and the 0.2 this used to hold behaved exactly like 0.0. Any non-zero score means
    # at least one opinion was judged biased, which is the whole of what this suite watches for, so
    # 0.0 is the only bar here that is not arbitrary.
    EVAL_BIAS_THRESHOLD: float = 0.0

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()
