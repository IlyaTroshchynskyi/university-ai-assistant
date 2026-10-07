import json
from pathlib import Path

from pydantic import ValidationError
import pymupdf
import pytest

from app.ai_assistant_langchain.tools import ASSISTANT_TOOLS, check_scholarship
from app.api.v1.scholarships import scholarship_rules as rules
from app.api.v1.scholarships.enums import Faculty
from app.api.v1.scholarships.schemas import ScholarshipCheck
from app.api.v1.scholarships.scholarship_rules import evaluate_scholarships

MERIT = 'Northwood Merit Scholarship'
DEANS = "Dean's Excellence Award"
NEED_BASED = 'Need-Based Access Grant'
WOMEN_IN_STEM = 'Women in STEM Scholarship'

QUALIFIED = 'qualified'
MISSED = 'missed'
UNDECIDED = 'undecided'

HANDBOOK = Path(__file__).parents[2] / 'docs' / 'handbook.pdf'


def build_verdict_map(check: ScholarshipCheck) -> dict[str, str]:
    matches_by_verdict = {QUALIFIED: check.qualified, MISSED: check.missed, UNDECIDED: check.undecided}
    return {match.name: verdict for verdict, matches in matches_by_verdict.items() for match in matches}


def find_scholarship_reason(check: ScholarshipCheck, scholarship: str) -> str:
    matches = [*check.qualified, *check.missed, *check.undecided]
    return next(match.reason for match in matches if match.name == scholarship)


class TestNorthwoodMeritScholarship:
    @pytest.mark.parametrize(('gpa', 'verdict'), [(3.69, MISSED), (3.7, QUALIFIED), (3.71, QUALIFIED)])
    def test_gpa_threshold(self, gpa: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=gpa)

        assert build_verdict_map(check)[MERIT] == verdict

    def test_award_is_half_of_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.7)

        assert check.qualified[0].award == '50% tuition discount'

    def test_reason_of_a_qualified_scholarship_is_the_rule_it_met(self) -> None:
        check = evaluate_scholarships(gpa=3.7)

        assert find_scholarship_reason(check, MERIT) == 'GPA 3.7 meets the 3.7 minimum.'

    def test_reason_of_a_missed_scholarship_is_the_rule_it_failed(self) -> None:
        check = evaluate_scholarships(gpa=3.69)

        assert find_scholarship_reason(check, MERIT) == 'GPA 3.69 is below the 3.7 minimum.'


class TestDeansExcellenceAward:
    @pytest.mark.parametrize(('gpa', 'verdict'), [(3.89, MISSED), (3.9, QUALIFIED), (3.91, QUALIFIED)])
    def test_gpa_threshold(self, gpa: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=gpa, entrance_exam_score=90)

        assert build_verdict_map(check)[DEANS] == verdict

    @pytest.mark.parametrize(('score', 'verdict'), [(89, MISSED), (89.9, MISSED), (90, QUALIFIED), (91, QUALIFIED)])
    def test_entrance_exam_threshold(self, score: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=score)

        assert build_verdict_map(check)[DEANS] == verdict

    def test_award_is_a_flat_amount(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90)

        assert [match.award for match in check.qualified if match.name == DEANS] == ['$8,000 per year']

    def test_reason_of_a_qualified_scholarship_lists_every_condition(self) -> None:
        check = evaluate_scholarships(gpa=3.95, entrance_exam_score=92)

        assert (
            find_scholarship_reason(check, DEANS)
            == 'GPA 3.95 meets the 3.9 minimum; entrance exam score 92 meets the 90 minimum.'
        )

    def test_reason_of_a_missed_scholarship_leaves_out_the_conditions_that_were_met(self) -> None:
        check = evaluate_scholarships(gpa=3.95, entrance_exam_score=89.5)

        assert find_scholarship_reason(check, DEANS) == 'Entrance exam score 89.5 is below the 90 minimum.'

    def test_unknown_exam_score_leaves_it_undecided(self) -> None:
        check = evaluate_scholarships(gpa=3.9)

        assert build_verdict_map(check)[DEANS] == UNDECIDED
        assert find_scholarship_reason(check, DEANS) == 'The entrance exam score was not given.'

    def test_failed_gpa_decides_it_whatever_the_exam_score_would_be(self) -> None:
        check = evaluate_scholarships(gpa=3.89)

        assert build_verdict_map(check)[DEANS] == MISSED
        assert find_scholarship_reason(check, DEANS) == 'GPA 3.89 is below the 3.9 minimum.'


class TestNeedBasedAccessGrant:
    @pytest.mark.parametrize(('income', 'verdict'), [(29_999.99, QUALIFIED), (30_000, MISSED), (30_000.01, MISSED)])
    def test_family_income_limit(self, income: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=3.0, family_income=income)

        assert build_verdict_map(check)[NEED_BASED] == verdict

    @pytest.mark.parametrize(('gpa', 'verdict'), [(2.99, MISSED), (3.0, QUALIFIED), (3.01, QUALIFIED)])
    def test_gpa_threshold(self, gpa: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=gpa, family_income=25_000)

        assert build_verdict_map(check)[NEED_BASED] == verdict

    def test_award_is_thirty_percent_of_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.0, family_income=25_000)

        assert [match.award for match in check.qualified] == ['30% tuition discount']

    def test_reason_of_a_qualified_scholarship_lists_every_condition(self) -> None:
        check = evaluate_scholarships(gpa=3.2, family_income=25_000)

        assert find_scholarship_reason(check, NEED_BASED) == (
            'Family income $25,000 is below the $30,000 limit; GPA 3.2 meets the 3.0 minimum.'
        )

    def test_income_exactly_at_the_limit_is_named_as_the_failed_rule(self) -> None:
        check = evaluate_scholarships(gpa=3.2, family_income=30_000)

        assert find_scholarship_reason(check, NEED_BASED) == 'Family income $30,000 is not below the $30,000 limit.'

    def test_income_a_cent_under_the_limit_is_not_rounded_up_to_it(self) -> None:
        check = evaluate_scholarships(gpa=3.2, family_income=29_999.99)

        assert find_scholarship_reason(check, NEED_BASED) == (
            'Family income $29,999.99 is below the $30,000 limit; GPA 3.2 meets the 3.0 minimum.'
        )

    def test_income_is_judged_in_whole_cents(self) -> None:
        check = evaluate_scholarships(gpa=3.2, family_income=29_999.999)

        assert build_verdict_map(check)[NEED_BASED] == MISSED
        assert find_scholarship_reason(check, NEED_BASED) == 'Family income $30,000 is not below the $30,000 limit.'

    def test_unknown_income_leaves_it_undecided(self) -> None:
        check = evaluate_scholarships(gpa=3.2)

        assert build_verdict_map(check)[NEED_BASED] == UNDECIDED
        assert find_scholarship_reason(check, NEED_BASED) == 'The family income was not given.'

    def test_failed_gpa_decides_it_whatever_the_income_would_be(self) -> None:
        check = evaluate_scholarships(gpa=2.99)

        assert build_verdict_map(check)[NEED_BASED] == MISSED


class TestWomenInStemScholarship:
    @pytest.mark.parametrize(('gpa', 'verdict'), [(3.49, MISSED), (3.5, QUALIFIED), (3.51, QUALIFIED)])
    def test_gpa_threshold(self, gpa: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=gpa, is_female=True, faculty=Faculty.COMPUTER_SCIENCE)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == verdict

    @pytest.mark.parametrize(
        ('faculty', 'verdict'),
        [
            (Faculty.COMPUTER_SCIENCE, QUALIFIED),
            (Faculty.ENGINEERING, QUALIFIED),
            (Faculty.NATURAL_SCIENCES, QUALIFIED),
            (Faculty.ECONOMICS_AND_BUSINESS, MISSED),
            (Faculty.LAW, MISSED),
            (Faculty.ARTS_AND_DESIGN, MISSED),
        ],
    )
    def test_qualifying_faculties(self, faculty: Faculty, verdict: str) -> None:
        check = evaluate_scholarships(gpa=3.5, is_female=True, faculty=faculty)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == verdict

    def test_applicant_who_is_not_female_misses_it(self) -> None:
        check = evaluate_scholarships(gpa=3.5, is_female=False, faculty=Faculty.ENGINEERING)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == MISSED
        assert find_scholarship_reason(check, WOMEN_IN_STEM) == 'The applicant is not female.'

    def test_award_is_a_flat_amount(self) -> None:
        check = evaluate_scholarships(gpa=3.5, is_female=True, faculty=Faculty.ENGINEERING)

        assert [match.award for match in check.qualified] == ['$5,000 per year']

    def test_reason_of_a_qualified_scholarship_lists_every_condition(self) -> None:
        check = evaluate_scholarships(gpa=3.6, is_female=True, faculty=Faculty.COMPUTER_SCIENCE)

        assert find_scholarship_reason(check, WOMEN_IN_STEM) == (
            'The applicant is female; Computer Science is a qualifying faculty; GPA 3.6 meets the 3.5 minimum.'
        )

    def test_reason_of_a_wrong_faculty_names_the_ones_that_qualify(self) -> None:
        check = evaluate_scholarships(gpa=3.6, is_female=True, faculty=Faculty.LAW)

        assert find_scholarship_reason(check, WOMEN_IN_STEM) == (
            'Law is not a qualifying faculty (Computer Science, Engineering, Natural Sciences).'
        )

    def test_unstated_gender_leaves_it_undecided(self) -> None:
        check = evaluate_scholarships(gpa=3.6, faculty=Faculty.COMPUTER_SCIENCE)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == UNDECIDED
        assert find_scholarship_reason(check, WOMEN_IN_STEM) == 'The applicant has not said whether they are female.'

    def test_unknown_faculty_leaves_it_undecided(self) -> None:
        check = evaluate_scholarships(gpa=3.6, is_female=True)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == UNDECIDED
        assert find_scholarship_reason(check, WOMEN_IN_STEM) == 'The faculty of the programme was not given.'

    def test_reason_of_an_undecided_scholarship_lists_every_missing_fact(self) -> None:
        check = evaluate_scholarships(gpa=3.6)

        assert find_scholarship_reason(check, WOMEN_IN_STEM) == (
            'The applicant has not said whether they are female; the faculty of the programme was not given.'
        )

    def test_failed_condition_decides_it_whatever_is_still_unknown(self) -> None:
        check = evaluate_scholarships(gpa=3.6, faculty=Faculty.LAW)

        assert build_verdict_map(check)[WOMEN_IN_STEM] == MISSED


class TestAdmissionMinimum:
    @pytest.mark.parametrize(('score', 'verdict'), [(59, MISSED), (60, QUALIFIED), (61, QUALIFIED)])
    def test_entrance_exam_threshold(self, score: float, verdict: str) -> None:
        check = evaluate_scholarships(gpa=3.8, entrance_exam_score=score)

        assert build_verdict_map(check)[MERIT] == verdict

    def test_score_below_it_fails_every_scholarship(self) -> None:
        check = evaluate_scholarships(
            gpa=3.95,
            family_income=25_000,
            entrance_exam_score=59,
            is_female=True,
            faculty=Faculty.COMPUTER_SCIENCE,
        )

        assert build_verdict_map(check) == {MERIT: MISSED, DEANS: MISSED, NEED_BASED: MISSED, WOMEN_IN_STEM: MISSED}
        assert check.awarded is None

    def test_reason_says_the_applicant_cannot_be_admitted(self) -> None:
        check = evaluate_scholarships(gpa=3.8, entrance_exam_score=40)

        assert find_scholarship_reason(check, MERIT) == 'Entrance exam score 40 is below the 60 needed to be admitted.'

    def test_unknown_score_blocks_nothing(self) -> None:
        check = evaluate_scholarships(gpa=3.8)

        assert build_verdict_map(check)[MERIT] == QUALIFIED


class TestEvaluateScholarships:
    def test_every_scholarship_is_given_exactly_one_verdict(self) -> None:
        check = evaluate_scholarships(gpa=3.8, family_income=25_000)

        assert build_verdict_map(check) == {
            MERIT: QUALIFIED,
            DEANS: MISSED,
            NEED_BASED: QUALIFIED,
            WOMEN_IN_STEM: UNDECIDED,
        }
        assert len([*check.qualified, *check.missed, *check.undecided]) == 4

    def test_qualifying_for_nothing_is_an_empty_list_not_an_error(self) -> None:
        check = evaluate_scholarships(
            gpa=2.5,
            family_income=50_000,
            entrance_exam_score=50,
            is_female=False,
            faculty=Faculty.LAW,
        )

        assert check.qualified == []
        assert build_verdict_map(check) == {MERIT: MISSED, DEANS: MISSED, NEED_BASED: MISSED, WOMEN_IN_STEM: MISSED}


class TestAwardedScholarship:
    def test_nothing_qualified_awards_nothing(self) -> None:
        check = evaluate_scholarships(gpa=3.4, family_income=40_000)

        assert check.awarded is None
        assert check.awarded_reason == 'No scholarship qualifies on the facts given.'

    def test_nothing_qualified_yet_names_what_is_still_undecided(self) -> None:
        check = evaluate_scholarships(gpa=3.6)

        assert check.awarded is None
        assert check.awarded_reason == (
            'Nothing qualifies yet; still undecided for lack of a fact: Need-Based Access Grant, Women in STEM '
            'Scholarship.'
        )

    def test_undecided_scholarship_worth_more_makes_the_award_not_final(self) -> None:
        check = evaluate_scholarships(gpa=3.6, is_female=True, faculty=Faculty.COMPUTER_SCIENCE, annual_tuition=26_000)

        assert check.awarded == WOMEN_IN_STEM
        assert check.awarded_reason == (
            'It is the only scholarship that qualifies. With annual tuition of $26,000 it is worth $5,000 a year. '
            'Not final: Need-Based Access Grant would be worth more if the missing facts are met.'
        )

    def test_without_the_tuition_an_undecided_award_of_the_other_kind_makes_it_not_final(self) -> None:
        check = evaluate_scholarships(gpa=3.9)

        assert check.awarded == MERIT
        assert check.awarded_reason == (
            "It is the only scholarship that qualifies. Not final: Dean's Excellence Award, Women in STEM "
            'Scholarship could be worth more once the missing facts and the annual tuition are known.'
        )

    def test_only_qualified_scholarship_is_the_one_awarded(self) -> None:
        check = evaluate_scholarships(gpa=3.7, family_income=40_000, is_female=False)

        assert check.awarded == MERIT
        assert check.awarded_reason == 'It is the only scholarship that qualifies.'

    def test_only_qualified_scholarship_is_priced_when_the_tuition_is_known(self) -> None:
        check = evaluate_scholarships(gpa=3.8, annual_tuition=24_500)

        assert check.awarded == MERIT
        assert check.awarded_reason == (
            'It is the only scholarship that qualifies. With annual tuition of $24,500 it is worth $12,250 a year.'
        )

    def test_larger_of_two_discounts_is_awarded_without_knowing_the_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.8, family_income=25_000, is_female=False)

        assert check.awarded == MERIT
        assert check.awarded_reason == (
            'A student holds one scholarship at a time, the one worth the most. Northwood Merit '
            'Scholarship (50% tuition discount) is worth more than Need-Based Access Grant (30% tuition '
            'discount).'
        )

    def test_discount_against_a_flat_amount_is_not_decided_without_the_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90)

        assert [match.name for match in check.qualified] == [MERIT, DEANS]
        assert check.awarded is None
        assert check.awarded_reason == (
            'A student holds one scholarship at a time, the one worth the most. Whether Northwood Merit '
            "Scholarship (50% tuition discount) or Dean's Excellence Award ($8,000 per year) is worth "
            'more depends on the annual tuition of the programme, which was not given.'
        )

    def test_smaller_discount_is_left_out_of_the_question_about_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90, family_income=25_000)

        assert [match.name for match in check.qualified] == [MERIT, DEANS, NEED_BASED]
        assert check.awarded is None
        assert NEED_BASED not in check.awarded_reason

    @pytest.mark.parametrize(('tuition', 'awarded_name'), [(15_999.99, DEANS), (16_000.01, MERIT)])
    def test_tuition_decides_between_merit_and_deans_excellence(self, tuition: float, awarded_name: str) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90, annual_tuition=tuition)

        assert check.awarded == awarded_name

    @pytest.mark.parametrize(('tuition', 'awarded_name'), [(16_000, WOMEN_IN_STEM), (17_000, NEED_BASED)])
    def test_tuition_decides_between_the_access_grant_and_women_in_stem(
        self, tuition: float, awarded_name: str
    ) -> None:
        check = evaluate_scholarships(
            gpa=3.6,
            family_income=25_000,
            is_female=True,
            faculty=Faculty.ENGINEERING,
            annual_tuition=tuition,
        )

        assert check.awarded == awarded_name

    def test_equal_worth_goes_to_the_scholarship_the_handbook_lists_first(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90, annual_tuition=16_000)

        assert check.awarded == MERIT

    def test_reason_shows_what_each_qualified_scholarship_is_worth(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=90, annual_tuition=24_500)

        assert check.awarded_reason == (
            'A student holds one scholarship at a time, the one worth the most. With annual tuition of '
            "$24,500: Northwood Merit Scholarship is worth $12,250 a year, Dean's Excellence Award is "
            'worth $8,000 a year.'
        )


class TestTuitionAfterAward:
    def test_it_is_unknown_without_the_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.7)

        assert check.tuition_after_award is None

    def test_it_is_unknown_when_nothing_is_awarded(self) -> None:
        check = evaluate_scholarships(gpa=3.4, family_income=40_000, annual_tuition=20_000)

        assert check.tuition_after_award is None

    def test_flat_amount_is_worth_no_more_than_the_tuition(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=95, annual_tuition=6_000)

        assert check.awarded == DEANS
        assert check.tuition_after_award == 0
        assert "Dean's Excellence Award is worth $6,000 a year" in check.awarded_reason

    def test_tuition_is_judged_in_whole_cents(self) -> None:
        check = evaluate_scholarships(gpa=3.9, entrance_exam_score=95, annual_tuition=15_999.996)

        assert check.awarded == MERIT
        assert check.tuition_after_award == 8_000


class TestHandbookExamples:
    def test_software_engineering_student_with_merit(self) -> None:
        check = evaluate_scholarships(gpa=3.8, faculty=Faculty.COMPUTER_SCIENCE, annual_tuition=24_500)

        assert check.awarded == MERIT
        assert check.tuition_after_award == 12_250

    def test_female_data_science_student_gets_women_in_stem(self) -> None:
        check = evaluate_scholarships(
            gpa=3.6,
            is_female=True,
            faculty=Faculty.COMPUTER_SCIENCE,
            annual_tuition=26_000,
        )

        assert check.awarded == WOMEN_IN_STEM
        assert check.tuition_after_award == 21_000

    def test_higher_gpa_replaces_women_in_stem_with_merit(self) -> None:
        check = evaluate_scholarships(
            gpa=3.7,
            is_female=True,
            faculty=Faculty.COMPUTER_SCIENCE,
            annual_tuition=26_000,
        )

        assert [match.name for match in check.qualified] == [MERIT, WOMEN_IN_STEM]
        assert check.awarded == MERIT
        assert check.tuition_after_award == 13_000

    def test_law_student_on_a_low_income_gets_the_access_grant(self) -> None:
        check = evaluate_scholarships(gpa=3.2, family_income=25_000, faculty=Faculty.LAW, annual_tuition=18_500)

        assert check.awarded == NEED_BASED
        assert check.tuition_after_award == 12_950


class TestCheckScholarshipTool:
    async def test_model_reads_the_check_as_json(self) -> None:
        message = await check_scholarship.ainvoke(
            {
                'type': 'tool_call',
                'name': check_scholarship.name,
                'args': {'gpa': 3.8, 'family_income': 25_000},
                'id': 'call-1',
            }
        )

        assert json.loads(message.content) == {
            'qualified': [
                {'name': MERIT, 'award': '50% tuition discount', 'reason': 'GPA 3.8 meets the 3.7 minimum.'},
                {
                    'name': NEED_BASED,
                    'award': '30% tuition discount',
                    'reason': 'Family income $25,000 is below the $30,000 limit; GPA 3.8 meets the 3.0 minimum.',
                },
            ],
            'missed': [
                {'name': DEANS, 'award': '$8,000 per year', 'reason': 'GPA 3.8 is below the 3.9 minimum.'},
            ],
            'undecided': [
                {
                    'name': WOMEN_IN_STEM,
                    'award': '$5,000 per year',
                    'reason': (
                        'The applicant has not said whether they are female; the faculty of the programme '
                        'was not given.'
                    ),
                },
            ],
            'awarded': MERIT,
            'awarded_reason': (
                'A student holds one scholarship at a time, the one worth the most. Northwood Merit '
                'Scholarship (50% tuition discount) is worth more than Need-Based Access Grant (30% tuition '
                'discount). Not final: Women in STEM Scholarship could be worth more once the missing facts '
                'and the annual tuition are known.'
            ),
            'tuition_after_award': None,
        }

    async def test_every_fact_the_applicant_gave_reaches_the_rules(self) -> None:
        result = await check_scholarship.ainvoke(
            {
                'gpa': 3.95,
                'family_income': 25_000,
                'entrance_exam_score': 95,
                'is_female': True,
                'faculty': 'Natural Sciences',
                'annual_tuition': 15_000,
            }
        )

        assert [match['name'] for match in result['qualified']] == [MERIT, DEANS, NEED_BASED, WOMEN_IN_STEM]
        assert result['awarded'] == DEANS

    async def test_gpa_is_required(self) -> None:
        with pytest.raises(ValidationError):
            await check_scholarship.ainvoke({'family_income': 25_000})

    @pytest.mark.parametrize('gpa', [-0.1, 4.1])
    async def test_gpa_off_the_four_point_scale_is_rejected(self, gpa: float) -> None:
        with pytest.raises(ValidationError):
            await check_scholarship.ainvoke({'gpa': gpa})

    @pytest.mark.parametrize('score', [-1, 101])
    async def test_exam_score_off_the_hundred_point_scale_is_rejected(self, score: float) -> None:
        with pytest.raises(ValidationError):
            await check_scholarship.ainvoke({'gpa': 3.9, 'entrance_exam_score': score})

    @pytest.mark.parametrize('amounts', [{'family_income': -1}, {'annual_tuition': 0}])
    async def test_amount_that_cannot_be_real_is_rejected(self, amounts: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            await check_scholarship.ainvoke({'gpa': 3.9, **amounts})

    async def test_faculty_the_handbook_does_not_have_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            await check_scholarship.ainvoke({'gpa': 3.6, 'is_female': True, 'faculty': 'Medicine'})

    def test_qa_agent_is_given_the_tool(self) -> None:
        assert check_scholarship in ASSISTANT_TOOLS


@pytest.fixture(scope='module')
def handbook_text() -> str:
    with pymupdf.open(HANDBOOK) as document:
        return ' '.join(' '.join(page.get_text() for page in document).split())


@pytest.mark.skipif(not HANDBOOK.exists(), reason='docs/handbook.pdf is gitignored and absent from a fresh clone')
class TestRulesMatchTheHandbook:
    @pytest.mark.parametrize(
        'sentence',
        [
            f'minimum score of {rules.ADMISSION_MIN_EXAM_SCORE:g} is required to be considered',
            f'any admitted student with a GPA of {rules.MERIT_MIN_GPA} or higher',
            (
                f'GPA of {rules.DEANS_MIN_GPA} or higher AND an entrance exam score of '
                f'{rules.DEANS_MIN_EXAM_SCORE:g} or more'
            ),
            (
                f'family income below ${rules.NEED_BASED_INCOME_LIMIT:,.0f} per year AND a GPA of '
                f'{rules.NEED_BASED_MIN_GPA} or higher'
            ),
            f'Sciences program, with a GPA of {rules.WOMEN_IN_STEM_MIN_GPA} or higher',
            'female students enrolled in a {}, or {} program'.format(
                ', '.join(rules.WOMEN_IN_STEM_FACULTIES[:-1]),
                rules.WOMEN_IN_STEM_FACULTIES[-1],
            ),
            f'Award: {rules.MERIT.award}',
            f'Award: {rules.DEANS_EXCELLENCE.award}',
            f'Award: {rules.NEED_BASED.award}',
            f'Award: {rules.WOMEN_IN_STEM.award}',
        ],
    )
    def test_handbook_states_the_rule_the_code_applies(self, handbook_text: str, sentence: str) -> None:
        assert sentence in handbook_text
