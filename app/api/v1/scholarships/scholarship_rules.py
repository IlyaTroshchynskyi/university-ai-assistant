from __future__ import annotations

from dataclasses import dataclass

from app.api.v1.scholarships.enums import AwardKind, Faculty
from app.api.v1.scholarships.schemas import ScholarshipCheck, ScholarshipMatch

ADMISSION_MIN_EXAM_SCORE = 60.0
MERIT_MIN_GPA = 3.7
DEANS_MIN_GPA = 3.9
DEANS_MIN_EXAM_SCORE = 90.0
NEED_BASED_INCOME_LIMIT = 30_000.0
NEED_BASED_MIN_GPA = 3.0
WOMEN_IN_STEM_MIN_GPA = 3.5
WOMEN_IN_STEM_FACULTIES = (Faculty.COMPUTER_SCIENCE, Faculty.ENGINEERING, Faculty.NATURAL_SCIENCES)

ONE_SCHOLARSHIP_RULE = 'A student holds one scholarship at a time, the one worth the most.'


@dataclass(frozen=True, slots=True)
class Scholarship:
    name: str
    kind: AwardKind
    value: float

    @property
    def award(self) -> str:
        if self.kind is AwardKind.TUITION_DISCOUNT:
            return f'{self.value:.0%} tuition discount'
        return f'{_format_money(self.value)} per year'

    @property
    def label(self) -> str:
        return f'{self.name} ({self.award})'

    def compute_worth(self, annual_tuition: float) -> float:
        if self.kind is AwardKind.TUITION_DISCOUNT:
            return self.value * annual_tuition
        return min(self.value, annual_tuition)


@dataclass(frozen=True, slots=True)
class Condition:
    is_met: bool | None
    statement: str


MERIT = Scholarship('Northwood Merit Scholarship', AwardKind.TUITION_DISCOUNT, 0.5)
DEANS_EXCELLENCE = Scholarship("Dean's Excellence Award", AwardKind.YEARLY_AMOUNT, 8_000)
NEED_BASED = Scholarship('Need-Based Access Grant', AwardKind.TUITION_DISCOUNT, 0.3)
WOMEN_IN_STEM = Scholarship('Women in STEM Scholarship', AwardKind.YEARLY_AMOUNT, 5_000)


def evaluate_scholarships(
    gpa: float,
    family_income: float | None = None,
    entrance_exam_score: float | None = None,
    is_female: bool | None = None,
    faculty: Faculty | None = None,
    annual_tuition: float | None = None,
) -> ScholarshipCheck:
    if annual_tuition is not None:
        annual_tuition = round(annual_tuition, 2)

    admission_conditions = _check_admission(entrance_exam_score)
    rules = {
        MERIT: [_check_gpa_minimum(gpa, MERIT_MIN_GPA)],
        DEANS_EXCELLENCE: [
            _check_gpa_minimum(gpa, DEANS_MIN_GPA),
            _check_exam_score_minimum(entrance_exam_score, DEANS_MIN_EXAM_SCORE),
        ],
        NEED_BASED: [
            _check_income_limit(family_income, NEED_BASED_INCOME_LIMIT),
            _check_gpa_minimum(gpa, NEED_BASED_MIN_GPA),
        ],
        WOMEN_IN_STEM: [
            _check_gender(is_female),
            _check_faculty(faculty, WOMEN_IN_STEM_FACULTIES),
            _check_gpa_minimum(gpa, WOMEN_IN_STEM_MIN_GPA),
        ],
    }

    qualified_matches: dict[Scholarship, ScholarshipMatch] = {}
    missed_matches: list[ScholarshipMatch] = []
    undecided_matches: dict[Scholarship, ScholarshipMatch] = {}

    for scholarship, scholarship_conditions in rules.items():
        conditions = [*admission_conditions, *scholarship_conditions]
        failed_conditions = [condition for condition in conditions if condition.is_met is False]
        unknown_conditions = [condition for condition in conditions if condition.is_met is None]
        if failed_conditions:
            missed_matches.append(_build_scholarship_match(scholarship, failed_conditions))
        elif unknown_conditions:
            undecided_matches[scholarship] = _build_scholarship_match(scholarship, unknown_conditions)
        else:
            qualified_matches[scholarship] = _build_scholarship_match(scholarship, conditions)

    awarded_scholarship, awarded_reason = _pick_awarded_scholarship(
        list(qualified_matches),
        list(undecided_matches),
        annual_tuition,
    )
    return ScholarshipCheck(
        qualified=list(qualified_matches.values()),
        missed=missed_matches,
        undecided=list(undecided_matches.values()),
        awarded=awarded_scholarship.name if awarded_scholarship else None,
        awarded_reason=awarded_reason,
        tuition_after_award=_compute_tuition_after_award(awarded_scholarship, annual_tuition),
    )


def _check_admission(exam_score: float | None) -> list[Condition]:
    if exam_score is None or exam_score >= ADMISSION_MIN_EXAM_SCORE:
        return []

    return [
        Condition(
            False,
            f'entrance exam score {_format_number(exam_score)} is below the '
            f'{_format_number(ADMISSION_MIN_EXAM_SCORE)} needed to be admitted',
        )
    ]


def _check_gpa_minimum(gpa: float, minimum: float) -> Condition:
    if gpa >= minimum:
        return Condition(True, f'GPA {gpa} meets the {minimum} minimum')
    return Condition(False, f'GPA {gpa} is below the {minimum} minimum')


def _check_exam_score_minimum(score: float | None, minimum: float) -> Condition:
    if score is None:
        return Condition(None, 'the entrance exam score was not given')
    if score >= minimum:
        return Condition(
            True, f'entrance exam score {_format_number(score)} meets the {_format_number(minimum)} minimum'
        )
    return Condition(
        False, f'entrance exam score {_format_number(score)} is below the {_format_number(minimum)} minimum'
    )


def _check_income_limit(income: float | None, limit: float) -> Condition:
    if income is None:
        return Condition(None, 'the family income was not given')
    income = round(income, 2)
    if income < limit:
        return Condition(True, f'family income {_format_money(income)} is below the {_format_money(limit)} limit')
    return Condition(False, f'family income {_format_money(income)} is not below the {_format_money(limit)} limit')


def _check_gender(is_female: bool | None) -> Condition:
    if is_female is None:
        return Condition(None, 'the applicant has not said whether they are female')
    if is_female:
        return Condition(True, 'the applicant is female')
    return Condition(False, 'the applicant is not female')


def _check_faculty(faculty: Faculty | None, qualifying_faculties: tuple[Faculty, ...]) -> Condition:
    if faculty is None:
        return Condition(None, 'the faculty of the programme was not given')
    if faculty in qualifying_faculties:
        return Condition(True, f'{faculty} is a qualifying faculty')
    return Condition(False, f'{faculty} is not a qualifying faculty ({", ".join(qualifying_faculties)})')


def _build_scholarship_match(scholarship: Scholarship, deciding_conditions: list[Condition]) -> ScholarshipMatch:
    reason = '; '.join(condition.statement for condition in deciding_conditions)
    return ScholarshipMatch(name=scholarship.name, award=scholarship.award, reason=f'{reason[0].upper()}{reason[1:]}.')


def _pick_awarded_scholarship(
    qualified_scholarships: list[Scholarship],
    undecided_scholarships: list[Scholarship],
    annual_tuition: float | None,
) -> tuple[Scholarship | None, str]:
    if not qualified_scholarships and undecided_scholarships:
        undecided_names = ', '.join(scholarship.name for scholarship in undecided_scholarships)
        return None, f'Nothing qualifies yet; still undecided for lack of a fact: {undecided_names}.'
    if not qualified_scholarships:
        return None, 'No scholarship qualifies on the facts given.'

    most_valuable_scholarship, reason = _pick_most_valuable_scholarship(qualified_scholarships, annual_tuition)
    if most_valuable_scholarship is None:
        return None, reason

    rival_names = ', '.join(
        rival.name for rival in undecided_scholarships if _can_outrank(rival, most_valuable_scholarship, annual_tuition)
    )
    if not rival_names:
        return most_valuable_scholarship, reason
    if annual_tuition is None:
        return most_valuable_scholarship, (
            f'{reason} Not final: {rival_names} could be worth more once the missing facts and the annual '
            'tuition are known.'
        )
    return most_valuable_scholarship, (
        f'{reason} Not final: {rival_names} would be worth more if the missing facts are met.'
    )


def _pick_most_valuable_scholarship(
    qualified_scholarships: list[Scholarship],
    annual_tuition: float | None,
) -> tuple[Scholarship | None, str]:
    if len(qualified_scholarships) == 1:
        only_scholarship = qualified_scholarships[0]
        if annual_tuition is None:
            return only_scholarship, 'It is the only scholarship that qualifies.'
        return only_scholarship, (
            f'It is the only scholarship that qualifies. With annual tuition of {_format_money(annual_tuition)} '
            f'it is worth {_format_money(only_scholarship.compute_worth(annual_tuition))} a year.'
        )

    if annual_tuition is not None:
        best_scholarship = max(
            qualified_scholarships,
            key=lambda scholarship: scholarship.compute_worth(annual_tuition),
        )
        worth_summary = ', '.join(
            f'{scholarship.name} is worth {_format_money(scholarship.compute_worth(annual_tuition))} a year'
            for scholarship in qualified_scholarships
        )
        return best_scholarship, (
            f'{ONE_SCHOLARSHIP_RULE} With annual tuition of {_format_money(annual_tuition)}: {worth_summary}.'
        )

    leaders: list[Scholarship] = []
    for kind in AwardKind:
        same_kind_scholarships = [scholarship for scholarship in qualified_scholarships if scholarship.kind is kind]
        if same_kind_scholarships:
            leaders.append(max(same_kind_scholarships, key=lambda scholarship: scholarship.value))
    if len(leaders) == 1:
        best_scholarship = leaders[0]
        other_labels = ', '.join(
            scholarship.label for scholarship in qualified_scholarships if scholarship is not best_scholarship
        )
        return best_scholarship, f'{ONE_SCHOLARSHIP_RULE} {best_scholarship.label} is worth more than {other_labels}.'

    discount_leader, amount_leader = leaders
    return None, (
        f'{ONE_SCHOLARSHIP_RULE} Whether {discount_leader.label} or {amount_leader.label} is worth more '
        'depends on the annual tuition of the programme, which was not given.'
    )


def _can_outrank(rival: Scholarship, awarded_scholarship: Scholarship, annual_tuition: float | None) -> bool:
    if annual_tuition is not None:
        return rival.compute_worth(annual_tuition) > awarded_scholarship.compute_worth(annual_tuition)
    if rival.kind is not awarded_scholarship.kind:
        return True
    return rival.value > awarded_scholarship.value


def _compute_tuition_after_award(awarded_scholarship: Scholarship | None, annual_tuition: float | None) -> float | None:
    if awarded_scholarship is None or annual_tuition is None:
        return None
    return round(annual_tuition - awarded_scholarship.compute_worth(annual_tuition), 2)


def _format_money(amount: float) -> str:
    return f'${amount:,.2f}'.removesuffix('.00')


def _format_number(number: float) -> str:
    return str(number).removesuffix('.0')
