# 0004. The router sees the recent turns; no sticky routing

Status: accepted
Date: 2026-08-18

## Context

The main graph routes every message to `qa` or `booking` with one structured-output call. It used
to classify the message alone, so a reply such as "Monday works" to the booking agent's "Which day
suits you?" carried no evidence of a booking and could land in `qa`, which knows nothing of a
booking half done. A HITL resume is not affected: it re-enters the paused node and never reaches the
router.

## Options

- **Sticky routing:** remember the last node in `MainState` and stay there until the booking ends.
- **Context for the router:** show it the last few turns as text in its system prompt, and keep the
  message being routed as the actual `HumanMessage`.

## Decision

Context for the router (`_format_recent_turns` in `app/ai_assistant_langchain/graph/main_graph.py`,
`build_router_prompt` in `app/ai_assistant_langchain/graph/prompts.py`).

Sticky routing has no exit condition. The booking agent ends its turn after every question it asks,
so "the turn ended" and "the booking ended" look the same in the state. Detecting the end needs
another classifier call, a "done" signal out of the booking agent, or "stay until
`book_appointment` succeeds", which breaks on a cancellation, an abandoned booking or an off-topic
question. The subagents are compiled agents added as nodes and return only `messages`, so writing
an `active` flag also needs wrapper nodes. And its failure is worse: a misrouted message today
costs a repeat, while a user held in `booking`, whose prompt refuses everything but appointments,
has no way out.

The turns are rendered as text, not passed as a message list: a slice of `messages` can start on a
`ToolMessage`, which providers reject without the matching `tool_calls` before it, and text lets the
tool traffic be dropped.

## Consequences

- Every message goes through the router again; there is no "booking in progress" flag anywhere.
- "Actually, how much is tuition?" after five booking turns still routes to `qa`: the context sets
  the default, it does not lock the destination.
- The router's prompt grows by a few short turns (`limit`, `max_characters` in
  `_format_recent_turns`).
