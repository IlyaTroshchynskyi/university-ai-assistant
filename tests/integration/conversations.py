from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConversationCase:
    name: str
    questions: list[str]
    expected_outcome: str


CONVERSATIONS = [
    # A student asks about a named professor, then changes subject twice.
    # Turn 2 carries no name: without history the only correct move is to ask "whose?". Turn 3 moves
    # from a person to a building, turn 4 from DynamoDB to the handbook.
    # Data Science, not "the Computer Science bachelor" — CS is a *faculty* holding three programs,
    # so that question has no single right answer. Main Library because its hours are in both
    # sources (golden 17 and db/seed/places.json id 2) and they agree, so the case does not depend
    # on which tool the agent picks.
    # The outcome asks for the title and the office hours but not the faculty: ``find_person``
    # returns ``faculty_id``, a number, and nothing maps it to a name — so naming the faculty is
    # something the agent has no way to do and must not be asked for.
    ConversationCase(
        name='coreference',
        questions=[
            'Who is Dr. Alan Whitfield?',
            'What are his office hours?',
            'When does the Main Library open?',
            'And how much is the Data Science program per year?',
        ],
        expected_outcome=(
            'Dr. Alan Whitfield is a Professor. His office hours are Mon 14:00–16:00. The Main '
            'Library is open Mon–Fri 08:00–22:00 and Sat–Sun 10:00–18:00. The Data Science program '
            'costs $26,000 per year.'
        ),
    ),
    # An applicant asks about a faculty that does not exist, then presses on it.
    # Turn 1 establishes there is no medical faculty; turn 2 asks who runs it anyway. The failure is
    # a fabricated dean — a claim no passage supports, which is where TurnFaithfulness bites. Turn 3
    # checks the refusal survives without the agent overcorrecting into refusing everything.
    ConversationCase(
        name='false_premise',
        questions=[
            'Does NIT have a medical faculty?',
            'Who is the dean of the medical faculty?',
            'So I cannot study medicine here at all?',
        ],
        expected_outcome=(
            'NIT has six faculties — Computer Science, Economics & Business, Engineering, Law, '
            'Arts & Design and Natural Sciences — and none of them is medicine. There is therefore '
            'no medical faculty, no dean of one, and medicine cannot be studied at NIT. The '
            'assistant must not name any person as dean of a medical faculty.'
        ),
    ),
    # A prospective Mechanical Engineering student works out what the first year costs.
    # "That whole amount" in turn 2 is the $23,750 from turn 1 and appears nowhere in the turn's own
    # text. Each turn needs a different handbook section, so an answer built from stale first-turn
    # context shows up as a wrong fact.
    ConversationCase(
        name='carried_figure',
        questions=[
            'What is the total first-year cost for a Mechanical Engineering student before any scholarship?',
            'Would a scholarship reduce that whole amount?',
            'And how much do I pay up front to hold my place?',
        ],
        expected_outcome=(
            'The total first-year cost for Mechanical Engineering before any scholarship is $23,750: '
            '$23,000 tuition plus the $450 student services fee and the $300 lab fee. A scholarship '
            'discount applies to tuition only and never to fees, so the $450 student services fee is '
            'still paid in full. The tuition deposit is $1,000, due by June 15, 2027, and is credited '
            'toward first-year tuition.'
        ),
    ),
    # An admitted student states a GPA, is quoted a discount, then corrects the GPA downward.
    #
    # The only conversation where the decisive fact comes from the user rather than a document. The
    # failure is keeping $12,250 after the correction — stale state carried rather than revised,
    # which the Outcome judge's third evaluation step is written to catch. Also the live test of the
    # prompt rule "never state a discounted amount until you have found the rule that says this
    # student qualifies".
    ConversationCase(
        name='corrected_fact',
        questions=[
            "I've been admitted to Software Engineering. What's the annual tuition?",
            'My GPA is 3.8 — does that change what I pay?',
            "Sorry, I misread my transcript — it's 3.6.",
        ],
        expected_outcome=(
            'Software Engineering tuition is $24,500 per year. At a GPA of 3.8 the Northwood Merit '
            'Scholarship applies — a 50% tuition discount — bringing tuition to $12,250. Once the GPA '
            'is corrected to 3.6 the Merit Scholarship no longer applies, because it requires 3.7 or '
            'higher, so tuition is $24,500 again and the $12,250 figure no longer stands. Naming other '
            'scholarships is acceptable as long as the assistant does not claim this student '
            'qualifies for one.'
        ),
    ),
]
