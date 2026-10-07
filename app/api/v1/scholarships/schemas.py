from pydantic import BaseModel, Field


class ScholarshipMatch(BaseModel):
    name: str = Field(description='The scholarship, named as the handbook names it.')
    award: str = Field(description='What it is worth: a share of annual tuition or a flat amount per year.')
    reason: str = Field(
        description=(
            "The rule that decided it, with the applicant's own figures: the conditions met for a "
            'qualified scholarship, the ones failed for a missed one, the facts missing for an '
            'undecided one.'
        ),
    )


class ScholarshipCheck(BaseModel):
    qualified: list[ScholarshipMatch] = Field(
        description='Scholarships whose every condition the applicant meets. Empty when there is none.',
    )
    missed: list[ScholarshipMatch] = Field(
        description='Scholarships with at least one condition the applicant fails.',
    )
    undecided: list[ScholarshipMatch] = Field(
        description='Scholarships that fail nothing so far but depend on a fact that was not given.',
    )
    awarded: str | None = Field(
        description=(
            'The one scholarship out of `qualified` the applicant would hold: a student holds one at a '
            'time, the one worth the most. None when nothing qualifies, or when a tuition discount '
            'competes with a flat amount and the annual tuition was not given.'
        ),
    )
    awarded_reason: str = Field(description='Why that one is awarded, or what is missing to tell which one would be.')
    tuition_after_award: float | None = Field(
        description=(
            'Annual tuition left to pay once the awarded scholarship is applied. Fees are never discounted and '
            'are not in this figure. None when the tuition was not given or nothing is awarded.'
        ),
    )
