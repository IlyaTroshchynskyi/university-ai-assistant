from enum import StrEnum


class Faculty(StrEnum):
    COMPUTER_SCIENCE = 'Computer Science'
    ECONOMICS_AND_BUSINESS = 'Economics & Business'
    ENGINEERING = 'Engineering'
    LAW = 'Law'
    ARTS_AND_DESIGN = 'Arts & Design'
    NATURAL_SCIENCES = 'Natural Sciences'


class AwardKind(StrEnum):
    TUITION_DISCOUNT = 'tuition_discount'
    YEARLY_AMOUNT = 'yearly_amount'
