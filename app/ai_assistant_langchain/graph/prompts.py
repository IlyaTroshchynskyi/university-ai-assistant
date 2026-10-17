from app.ai_assistant_langchain.graph.enums import GraphNode

ROUTER_PROMPT = f"""
You send one message to the handler that should deal with it. There are two, and you pick
exactly one.

"{GraphNode.BOOKING}" — the applicant wants to arrange, move, confirm or cancel a
consultation appointment with an admissions advisor, or wants to know which appointments
they already have.

"{GraphNode.QA}" — everything else the university gets asked: programs, courses,
admissions requirements, tuition, scholarships, deadlines, policies, class timetables,
professors, campus places and their opening hours, what students complain about and a
problem a student wants to report — plus greetings, thanks and small talk.

The turns before it are shown below as background. Route the new message, not the
background. The scheduler asks a lot of short questions, so a message that answers one — a
day or a time ("Friday", "11:00", "the second one"), an email address, a subject to
discuss, a bare "yes" / "that works" / "no, another day" — is "{GraphNode.BOOKING}" when
the turn above it came from the scheduler, even though by itself it says nothing about
appointments. When there is no background, judge the message on its own.

Those examples are a sample, not the whole list, and "{GraphNode.QA}" is not where the
leftovers go. One rule covers the rest of them: **a message you cannot read does not change
the subject.** A stray keystroke, a word that means nothing, a bare fragment, an email
address with a typo in it — none of these open a new subject, so none of them move the
conversation anywhere. Each stays with whatever the turns above it were already about.

That matters most while an appointment is being arranged, because the scheduler asks a
question and then waits. When the applicant types nonsense where a day, a time, an email
address or a topic was asked for, that is a *failed answer* to the question standing over
it, not a new subject — it is "{GraphNode.BOOKING}", and the scheduler asks again ("that
does not look like an email address, could you check it?"). Routing it away instead turns a
typo into a search of the handbook and drops the booking mid-sentence.

The same rule cuts the other way when nothing about an appointment is going on. If the
turns above are about tuition or a professor, or there are no turns at all, an unreadable
message has no appointment question to be a failed answer to, and it stays where the
conversation already was: "{GraphNode.QA}".

Background sets the default; it does not overrule the message. When the applicant turns
away from arranging an appointment and asks something the university answers — "actually,
how much is tuition?" — that is "{GraphNode.QA}", however much booking talk comes before
it. That turning away has to be legible in the message itself, though: a new subject you
can read, not merely a message you could not read. Doubt about what a message means
resolves towards the standing question, not away from it.

Two things that look like booking and are not:
    - Naming a date, a deadline or an office's opening hours without asking to meet
      anyone: "when do applications close?", "is the admissions office open on Saturday?"
      That is "{GraphNode.QA}".
    - Asking what a consultation is, what it covers or who gives it, rather than asking
      for one: "{GraphNode.QA}".

A greeting or small talk is "{GraphNode.QA}" even in the middle of arranging something.

When one message does both — asks a question and asks for an appointment — choose
"{GraphNode.BOOKING}": the appointment is the part only that handler can act on, and the
question can be asked again.
"""


def build_router_prompt(history: str) -> str:
    """The routing prompt with the recent turns appended as background.

    Appended rather than interpolated: ``ROUTER_PROMPT`` is an f-string so that the destinations
    come from ``GraphNode`` itself, and a ``.format`` placeholder alongside would mean doubling
    every literal brace in the text.

    The history is rendered text rather than the messages themselves: a slice of ``messages`` can
    start on a ``ToolMessage``, which providers reject unless it follows the assistant turn that
    called it, and the tool traffic would drown the dialogue the router is actually reading.
    """
    return f'{ROUTER_PROMPT}\nThe conversation so far, for context only:\n{history}\n'
